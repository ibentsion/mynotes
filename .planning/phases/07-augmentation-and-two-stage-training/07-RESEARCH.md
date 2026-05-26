# Phase 7: Augmentation & Two-Stage Training - Research

**Researched:** 2026-05-26
**Domain:** PyTorch augmentation pipeline extension (albumentations) + training loop refactor (two-stage CTC training)
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Add `albumentations` as a dependency. Elastic deformation via `albumentations.ElasticTransform` + `albumentations.GridDistortion` applied inside `AugmentTransform.__call__` after existing rotation/brightness/noise. `(1, H, W) float32 tensor` → `(H, W) float32 numpy` → apply albumentations → back to `(1, H, W) tensor`. Conversion only when `elastic_alpha > 0`.
- **D-02:** Conservative defaults — `--elastic_alpha 0` (off by default), `--elastic_sigma` ~4-6 (Claude's discretion). When `elastic_alpha == 0`, no albumentations import in hot path.
- **D-03:** Elastic applies to **real fine-tuning train split only** — not synthetic pre-training, not val.
- **D-04:** Extract `_run_loop(model, train_ds, val_ds, optimizer, scheduler, ctc_loss, device, args, logger, series_prefix)` private helper containing the epoch loop. `run_training()` calls `_run_loop()` once (fine-tuning) or delegates to `_run_pretrain()` when `--pretrain_manifest` is set.
- **D-05:** Pre-train mode (`--pretrain_manifest` set): build synthetic dataset, random train/val split via `--val_frac`, run loop with `series_prefix="pretrain/"`, upload `checkpoint_pretrain.pt` as ClearML artifact, then return — do NOT proceed to fine-tuning.
- **D-06:** Fine-tune mode (`--pretrain_checkpoint_path` set): load model weights from checkpoint, create fresh `AdamW` with `--lr`, fresh `ReduceLROnPlateau` scheduler, log with existing series names.
- **D-07:** Optimizer and scheduler fully reset between stages.
- **D-08:** Pre-training val split is random (no `build_half_page_units()`) using `--val_frac`.
- **D-09:** Single ClearML task per invocation. Pre-training scalars under `pretrain/` series prefix.
- **D-10:** `checkpoint_pretrain.pt` uploaded as ClearML artifact. Fine-tuned `checkpoint.pt` is the production artifact.

### Claude's Discretion

- Exact `ElasticTransform` + `GridDistortion` parameter defaults for `--elastic_sigma` and grid distort strength
- Whether `_run_loop()` returns `best_val_cer: float` or a richer struct
- Exact series name casing/separators for pretrain scalars (`pretrain/val_cer` vs `pretrain_val_cer`)
- ClearML artifact name for pre-trained checkpoint

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUG-01 | `AugmentTransform` gains elastic deformation (`ElasticTransform` + `GridDistortion` via `albumentations`) applied after existing affine transforms | Albumentations 2.0.8 verified on PyPI; float32 HWC API confirmed; conversion pattern verified |
| AUG-02 | Elastic deformation strength configurable via `--elastic_alpha` (default `0` = disabled) and `--elastic_sigma` flags in `train_ctc.py` | Existing argparse pattern is `p.add_argument(...)` + `task.connect(vars(args))` |
| TRAIN-01 | `train_ctc.py` adds `--pretrain_manifest` + `--pretrain_epochs` (default `0` = skip); when set, pre-trains on synthetic data | Two-call approach supersedes the literal ROADMAP success criterion; refactor extracts `_run_loop` |
| TRAIN-02 | Pre-training validates on held-out fraction of synthetic set; fine-tuning uses real val set; `--pretrain_lr` controls pre-train LR | D-08 random split; D-06 fresh optimizer; ClearML scalar separation via series_prefix |
</phase_requirements>

---

## Summary

Phase 7 has two independent, well-scoped additions to two files: `ctc_utils.py` (AugmentTransform) and `train_ctc.py` (two-stage training). The design is already locked in full detail via CONTEXT.md — this research validates implementation correctness and surfaces edge cases.

**Albumentations:** Version 2.0.8 is current (released 2025-05-27) and well-established (GitHub: albumentations-team/albumentations, 87+ releases, Python >=3.9). The critical API detail for this phase is that albumentations 2.x requires HWC format `(H, W, C)` for direct transform calls; the project's tensor contract is `(1, H, W)` float32, so the conversion must add and then strip the channel dim. Using `A.Compose([...])` handles `(H, W)` automatically, but direct transform calls need `img[:, :, np.newaxis]`. Two breaking changes from albumentations 1.x are relevant: `always_apply` is removed (use `p=1`), and `value=` is renamed to `fill=`.

**Two-stage training:** `run_training()` currently spans ~265 lines. After extracting `_run_loop()`, the three functions (`run_training`, `_run_pretrain`, `_run_loop`) must each stay under 100 lines (CLAUDE.md constraint). The `on_epoch_end` callback, which tune.py depends on for Optuna pruning, must be threaded into `_run_loop`'s signature — it is NOT called during pre-training (pre-training has no external pruner). The `tune.py::_objective` function constructs `argparse.Namespace` directly — it must be updated to include the new flags with safe defaults (`elastic_alpha=0.0`, `pretrain_manifest=None`, `pretrain_checkpoint_path=None`).

**Primary recommendation:** Add albumentations via `uv add albumentations==2.0.8`; extend `AugmentTransform` with deferred import + `(H, W, 1)` conversion; extract `_run_loop` with `series_prefix` + `on_epoch_end` params; add `_run_pretrain` for the synthetic path; update `tune.py::_objective` Namespace with new keys at safe defaults.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Elastic augmentation | Training data pipeline (ctc_utils.py) | — | Transform applied in CropDataset.__getitem__ via AugmentTransform; GPU/CPU agnostic |
| Pre-training loop | Training script (train_ctc.py) | ClearML (reporting) | Epoch loop logic is training-tier; ClearML is passive observer |
| Checkpoint save/load | Training script (train_ctc.py) | Filesystem | torch.save / torch.load; checkpoint path is a CLI arg |
| Pre-train val split | Training script (train_ctc.py) | — | Random index split on synthetic manifest; no page structure |
| ClearML series prefix | Training script (train_ctc.py) | — | logger.report_scalar title/series strings controlled in _run_loop |
| Optimizer reset | Training script (train_ctc.py) | — | Each stage creates its own AdamW + ReduceLROnPlateau instance |
| tune.py compatibility | tune.py (_objective) | train_ctc.py | _objective Namespace must include new keys with safe defaults |

---

## Standard Stack

### Core (Phase 7 addition)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| albumentations | 2.0.8 [VERIFIED: PyPI] | ElasticTransform + GridDistortion | Industry-standard augmentation library; float32 support confirmed; well-maintained (87 releases, github.com/albumentations-team/albumentations) |

### Existing (no changes needed)

| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| torch | 2.11.0 (pinned) | Model, optimizer, checkpoint | Already in uv.lock |
| pandas | 3.0.2 (pinned) | Manifest CSV loading for synthetic | Already in uv.lock |
| clearml | 2.1.5 (pinned) | Task logging, artifact upload | Already in uv.lock |

### Transitive deps added by albumentations 2.0.8

| Dep | Status | Note |
|-----|--------|------|
| scipy >=1.10.0 | Already in uv.lock (transitive) | No version conflict expected |
| pydantic >=2.9.2 | Already in uv.lock (pydantic==2.13.3) | Compatible |
| albucore ==0.0.24 | New transitive dep, not yet in uv.lock | Will be resolved by `uv add` |
| opencv-python-headless >=4.9.0 | Already pinned at 4.13.0.92 | Compatible — satisfies albu requirement |

**Installation:**
```bash
uv add albumentations==2.0.8
```

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| albumentations | PyPI | 7+ years (v0.0.0 first seen 2018) | Multi-million/month (widely cited in ML literature) | github.com/albumentations-team/albumentations | Not run (unavailable) | Approved — major well-known library |

**Packages removed due to slopcheck [SLOP] verdict:** none

**Packages flagged as suspicious [SUS]:** none

*slopcheck was unavailable at research time. albumentations is tagged `[VERIFIED: PyPI]` because it is confirmed by official documentation at albumentations.ai, has 87+ releases since 2018, and is the canonical augmentation library cited in ML papers and PyTorch ecosystem guides. The `[ASSUMED]` tag is not appropriate for a library of this standing, but the planner should note that slopcheck was not run.*

---

## Architecture Patterns

### System Architecture Diagram

```
train_ctc.py::main()
  │
  ├─ args.pretrain_manifest set?
  │     YES → _run_pretrain(model, args, logger)
  │              │
  │              ├─ Build CropDataset from synthetic manifest (no augment)
  │              ├─ Random val split (val_frac, no page structure)
  │              ├─ Fresh AdamW(lr=pretrain_lr) + ReduceLROnPlateau
  │              ├─ _run_loop(..., series_prefix="pretrain/", on_epoch_end=None)
  │              │     └─ epoch loop: train+val → logger.report_scalar("pretrain/...")
  │              ├─ torch.save → checkpoint_pretrain.pt
  │              ├─ upload_file_artifact(task, "checkpoint_pretrain", ...)
  │              └─ return best_pretrain_cer
  │
  └─ Fine-tune path (always, after optional pre-train)
        │
        ├─ args.pretrain_checkpoint_path set? → torch.load(weights_only=True)
        ├─ Build real CropDataset(augment=AugmentTransform(elastic_alpha=...) if aug_copies>0)
        │     └─ AugmentTransform.__call__ → elastic if elastic_alpha > 0
        │          └─ tensor (1,H,W) → numpy (H,W,1) → albu → numpy (H,W) → tensor (1,H,W)
        ├─ build_half_page_units → page-safe train/val split
        ├─ Fresh AdamW(lr=lr) + ReduceLROnPlateau
        └─ _run_loop(..., series_prefix="", on_epoch_end=on_epoch_end)
              └─ epoch loop: train+val → logger.report_scalar("loss"/"cer"/...)
                             → on_epoch_end(epoch, val_cer)  ← tune.py pruner hook
```

### Recommended Project Structure

No new files or directories. Phase 7 modifies:
```
src/
├── ctc_utils.py     # AugmentTransform: add elastic_alpha, elastic_sigma params + elastic path
└── train_ctc.py     # _build_parser: new flags; run_training: extract _run_loop + _run_pretrain
```

### Pattern 1: Albumentations Tensor Conversion

**What:** Convert between PyTorch `(1, H, W) float32` and albumentations `(H, W, C)` requirement.

**When to use:** Inside `AugmentTransform.__call__`, guarded by `elastic_alpha > 0`.

**Example:**
```python
# Source: albumentations.ai/docs (official API docs) + verified numpy/torch behavior
def __call__(self, tensor: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    # ... existing rotation, brightness, noise ...

    if self.elastic_alpha > 0:
        import albumentations as A  # noqa: PLC0415  # deferred — not imported when alpha=0

        transform = A.Compose([
            A.ElasticTransform(
                alpha=self.elastic_alpha,
                sigma=self.elastic_sigma,
                border_mode=0,   # BORDER_CONSTANT — consistent with padding_mode="border" intent
                fill=0.0,        # float32 fill value
                p=1.0,
            ),
            A.GridDistortion(
                num_steps=5,
                distort_range=(-0.15, 0.15),   # conservative: small distortion for Hebrew diacritics
                border_mode=0,
                fill=0.0,
                p=1.0,
            ),
        ])
        img_np = tensor.squeeze(0).numpy()          # (H, W) float32
        img_hwc = img_np[:, :, np.newaxis]          # (H, W, 1) — required by albu direct calls
        result = transform(image=img_hwc)["image"]  # (H, W, 1) float32
        tensor = torch.from_numpy(result[:, :, 0]).unsqueeze(0)  # back to (1, H, W)

    return tensor
```

**Key notes:**
- Use `A.Compose([...])` to wrap both transforms — handles `(H, W)` auto-expand internally, but explicit `(H, W, 1)` is safer for direct calls
- `border_mode=0` is `cv2.BORDER_CONSTANT` — nearest in spirit to `padding_mode="border"` used by existing affine grid (though "border" = replicate; for elastic, constant fill=0 is fine since distortions are small)
- `fill=0.0` not `value=0.0` — albumentations 2.x breaking change
- `p=1.0` — probability is always 1 since the outer `if elastic_alpha > 0` already gates it
- Import must be deferred (`# noqa: PLC0415`) to match existing pattern — hot path avoids import when `elastic_alpha=0`
- `np` must be imported: `import numpy as np` inside the deferred block or added to module imports

### Pattern 2: _run_loop Extraction

**What:** Extract the epoch loop from `run_training()` into a private helper.

**When to use:** Called by both `run_training()` (fine-tune path) and `_run_pretrain()` (synthetic path).

**Signature:**
```python
def _run_loop(
    model: CRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_df: pd.DataFrame,         # for debug sample selection
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ctc_loss: torch.nn.CTCLoss,
    ctc_loss_per_sample: torch.nn.CTCLoss,
    device: torch.device,
    args: argparse.Namespace,
    logger: Any,
    charset: list[str],
    checkpoint_path: Path,
    series_prefix: str = "",       # "pretrain/" for pre-training, "" for fine-tuning
    on_epoch_end: Callable[[int, float], None] | None = None,
) -> float:                        # returns best_val_cer
```

**Notes:**
- `debug_samples` selection requires `val_df` — pass it explicitly rather than reconstructing inside
- `series_prefix` propagates to all `logger.report_scalar(title=..., series=...)` calls
- `on_epoch_end` is `None` during pre-training (no Optuna pruner)
- `patience_left` state lives inside `_run_loop`; early stopping uses `args.patience`
- Checkpoint save uses `checkpoint_path` arg so pre-training saves to `checkpoint_pretrain.pt` and fine-tuning to `checkpoint.pt`

### Pattern 3: Random Split for Synthetic Data

**What:** Simple index-based random split for synthetic pre-training — no page structure.

**When to use:** In `_run_pretrain()` when building train/val from the synthetic manifest.

**Example:**
```python
# Source: CONTEXT.md D-08 — random val_frac split, no build_half_page_units
import math

n = len(synth_df)
n_val = max(1, math.ceil(n * args.val_frac))
indices = list(range(n))
val_idx = indices[:n_val]
train_idx = indices[n_val:]
# Then: CropDataset(synth_df.iloc[train_idx]), CropDataset(synth_df.iloc[val_idx])
```

**Notes:**
- Deterministic (no shuffle before split) — reproducible across runs
- Synthetic rows have no `page_path` or `page_num` — `build_half_page_units()` would crash on them

### Pattern 4: Weight Loading for Fine-tune from Pre-trained

**What:** Load pre-trained weights into a freshly constructed model before fine-tuning.

**Example:**
```python
# Source: PyTorch docs (torch.load with weights_only=True)
if args.pretrain_checkpoint_path is not None:
    state = torch.load(
        args.pretrain_checkpoint_path,
        weights_only=True,
        map_location=device,
    )
    model.load_state_dict(state)
```

**Notes:**
- `weights_only=True` is the established pattern in this codebase (see `run_training` line ~599)
- `map_location=device` ensures CPU checkpoints load correctly on GPU agents

### Anti-Patterns to Avoid

- **Passing numpy arrays without channel dim to direct albu transforms:** `(H, W)` arrays fail with direct `ElasticTransform()(image=arr)` calls in albu 2.x. Always use `(H, W, 1)` or `A.Compose([...])`.
- **Using `value=` instead of `fill=` in albumentations 2.x:** Hard failure. All fill parameters are now named `fill=`.
- **Using `always_apply=True` in albumentations 2.x:** Removed. Use `p=1.0`.
- **Importing albumentations at module level:** Breaks the `elastic_alpha=0` default — no import means no overhead and no import error for users who haven't installed albumentations (not applicable here since it'll be a project dep, but deferred import matches existing `# noqa: PLC0415` pattern).
- **Calling `on_epoch_end` during pre-training:** Pre-training has no Optuna pruner. Pass `on_epoch_end=None` to `_run_loop()` when called from `_run_pretrain()`.
- **Forgetting to update `tune.py::_objective` Namespace:** `_objective` constructs `argparse.Namespace` directly without calling argparse. New flags added to `_build_parser` are NOT automatically available in `_objective`. Must manually add `elastic_alpha=0.0`, `elastic_sigma=args.elastic_sigma`, `pretrain_manifest=None`, `pretrain_checkpoint_path=sweep_args.pretrain_checkpoint_path` (or `None`).
- **Running `build_half_page_units` on synthetic manifest:** Synthetic rows have no `page_path`/`y`/`h` columns matching real manifest schema — this call will crash.
- **Applying elastic augmentation during pre-training:** D-03 explicitly prohibits elastic on synthetic crops.
- **Not clamping result to `[0, 1]` after elastic:** Albumentations auto-clips float32 to `[0, 1]`, but adding an explicit `torch.clamp(tensor, 0.0, 1.0)` after conversion is defensive and matches existing pattern in AugmentTransform.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Elastic deformation | Custom displacement field with OpenCV remap | `albumentations.ElasticTransform` | Handles seeded randomness, float32 precision, border modes, and produces biologically-plausible distortions |
| Grid distortion | Custom grid warp | `albumentations.GridDistortion` | Correct interpolation, normalized mode keeps pixels in-bounds |
| Checkpoint loading | Custom state dict manipulation | `torch.load(..., weights_only=True)` | Already established pattern; handles device mapping correctly |

**Key insight:** Elastic deformation from scratch requires implementing Gaussian-smoothed random displacement fields and bilinear interpolation — all handled by albumentations with tested edge cases.

---

## Common Pitfalls

### Pitfall 1: Albumentations HW vs HWC Shape

**What goes wrong:** `ElasticTransform()(image=np_hw_array)` raises an error in albumentations 2.x when called directly (not via `A.Compose`).

**Why it happens:** Albumentations 2.x requires the channel dimension for direct transform calls. `A.Compose` adds it transparently but individual transform instances do not.

**How to avoid:** Either use `A.Compose([transform])` (which handles `(H, W)` automatically), or add channel dim: `img_np[:, :, np.newaxis]` before passing and `result[:, :, 0]` after.

**Warning signs:** `TypeError` or `AssertionError` about image dimensions in the augmentation step.

### Pitfall 2: `_run_loop` Line Count

**What goes wrong:** `_run_loop` grows past 100 lines (CLAUDE.md hard limit).

**Why it happens:** The existing epoch loop in `run_training` is already ~115 lines. Passing it verbatim to `_run_loop` violates the constraint.

**How to avoid:** Refactor the debug sample reporting block into a separate private function `_report_epoch_debug(...)` or `_report_val_debug(...)`. This was already done for `_report_annotated_crop`, `_report_prob_heatmap`, `_report_saliency_panel`, and `_report_char_distribution` — the same pattern applies.

**Warning signs:** Counting lines during implementation; `ruff` does not enforce this, so it must be checked manually.

### Pitfall 3: `tune.py::_objective` Namespace Out of Sync

**What goes wrong:** `run_training()` accesses `args.elastic_alpha` (or other new attrs) and `_objective`'s hardcoded Namespace doesn't include it → `AttributeError`.

**Why it happens:** `_objective` bypasses argparse entirely (documented: "ClearML monkey-patches both parse_args and parse_known_args") — new flags added to `_build_parser` don't automatically reach `_objective`.

**How to avoid:** Every new flag added to `_build_parser` must also be added to the `argparse.Namespace(...)` call in `_objective` with a safe default. For this phase:
- `elastic_alpha=0.0` (off by default — fine-tuning trials don't need to change this)
- `elastic_sigma=4.0` (matches `_build_parser` default)
- `pretrain_manifest=None` (fine-tuning trials don't pre-train)
- `pretrain_epochs=0`
- `pretrain_lr=1e-3` (unused when pretrain_manifest=None)
- `pretrain_checkpoint_path=sweep_args.pretrain_checkpoint_path` (forwarded from sweep args — HPO trials start from pre-trained weights when user passes `--pretrain_checkpoint_path` to tune.py)

**Warning signs:** `AttributeError: Namespace object has no attribute 'elastic_alpha'` during HPO trials.

### Pitfall 4: `tune.py` Parser Missing `--pretrain_checkpoint_path`

**What goes wrong:** tune.py's `_build_parser()` doesn't accept `--pretrain_checkpoint_path`, so passing it to `uv run tune-hpo` fails.

**Why it happens:** tune.py has its own `_build_parser()` separate from train_ctc.py. The HPO workflow (pre-train once, tune with shared checkpoint) requires tune.py to accept and forward `--pretrain_checkpoint_path`.

**How to avoid:** Add `--pretrain_checkpoint_path` to tune.py's `_build_parser()`. Forward `sweep_args.pretrain_checkpoint_path` into the `argparse.Namespace(...)` in `_objective`.

**Warning signs:** Argparse error when passing `--pretrain_checkpoint_path` to `tune-hpo`.

### Pitfall 5: Pre-training and Fine-tuning in Wrong Order in `run_training`

**What goes wrong:** `run_training()` runs the fine-tuning loop and THEN the pre-training loop, or runs both unconditionally.

**Why it happens:** Easy to invert the `if --pretrain_manifest` branch; the two-call approach means pre-training and fine-tuning are SEPARATE invocations — `run_training` should do either one or the other, never both in sequence.

**How to avoid:** `run_training()` logic:
1. If `args.pretrain_manifest` set → call `_run_pretrain(...)` → return immediately (do NOT continue to fine-tune)
2. Else → proceed with standard fine-tune (optionally loading checkpoint if `args.pretrain_checkpoint_path` set)

**Warning signs:** Pre-trained weights being overwritten by fine-tuning in the same invocation.

### Pitfall 6: ClearML Series Name Format

**What goes wrong:** Pre-training scalars use `pretrain/train_loss` as `series` but ClearML `report_scalar` signature is `(title, series, iteration, value)` — mixing up `title` and `series` breaks dashboard grouping.

**Why it happens:** The existing fine-tuning pattern uses `title="loss", series="train"`. The pretrain prefix pattern `pretrain/train_loss` could be interpreted as either the title or series.

**How to avoid:** Match the existing convention: `title="loss", series="pretrain/train"`. This keeps the same `title` ("loss", "cer") and adds the prefix to the `series` name. Example:
- Pre-train: `logger.report_scalar(title="loss", series="pretrain/train", iteration=epoch, value=train_loss)`
- Pre-train: `logger.report_scalar(title="cer", series="pretrain/val", iteration=epoch, value=val_cer)`
- Fine-tune: unchanged — `title="loss", series="train"` etc.

This is the interpretation consistent with "pretrain/ series prefix" in D-09.

### Pitfall 7: Pre-existing Failing Tests

**What goes wrong:** Two tests in `test_train_ctc.py` (`test_status_filter_keeps_only_labeled`, `test_charset_build_receives_labeled_labels`) fail before Phase 7 touches any code. They use `fake_build_charset()` without the `extra_words` kwarg added in a quick task.

**Why it happens:** Test stubs were written before `--words_file` was added in a quick task.

**How to avoid:** Fix these two tests in Wave 0 (add `extra_words=None` to the mock's signature) to establish a clean baseline before adding Phase 7 tests. Don't blame Phase 7 for pre-existing failures.

**Warning signs:** 2 failures in baseline test run (`pytest tests/test_ctc_utils.py tests/test_train_ctc.py -q`).

---

## Code Examples

### Albumentations Elastic Transform (Claude's Discretion Defaults)

The locked `elastic_sigma` range is ~4-6. For Hebrew handwriting at 64px height with diacritics:

```python
# Source: albumentations.ai official docs + parameter reasoning
# alpha=30-80 at sigma=4-5 produces moderate warping visible but not text-destroying
# GridDistortion distort_range=(-0.15, 0.15) is ~half the default — conservative for small text
import albumentations as A  # noqa: PLC0415

transform = A.Compose([
    A.ElasticTransform(
        alpha=self.elastic_alpha,     # CLI: --elastic_alpha, default=0 (off)
        sigma=self.elastic_sigma,     # CLI: --elastic_sigma, default=5.0
        border_mode=0,                # cv2.BORDER_CONSTANT
        fill=0.0,
        p=1.0,
    ),
    A.GridDistortion(
        num_steps=5,
        distort_range=(-0.15, 0.15),  # conservative; user can increase via future CLI flag
        border_mode=0,
        fill=0.0,
        p=1.0,
    ),
])
```

Recommended default: `elastic_sigma=5.0` (midpoint of 4-6 range). `elastic_alpha=30` is the default-off starting value to document in help text.

### Full AugmentTransform Extension

```python
def __init__(
    self,
    rotation_max: float = 7.0,
    brightness_delta: float = 0.10,
    noise_sigma: float = 0.02,
    elastic_alpha: float = 0.0,   # 0 = disabled (D-02)
    elastic_sigma: float = 5.0,   # only used when elastic_alpha > 0
) -> None:
    self.rotation_max = rotation_max
    self.brightness_delta = brightness_delta
    self.noise_sigma = noise_sigma
    self.elastic_alpha = elastic_alpha
    self.elastic_sigma = elastic_sigma
```

### Pre-training Checkpoint Upload

```python
# Source: src/clearml_utils.py::upload_file_artifact (established pattern)
pretrain_ckpt_path = args.output_dir / "checkpoint_pretrain.pt"
torch.save(model.state_dict(), pretrain_ckpt_path)
upload_file_artifact(task, "checkpoint_pretrain", pretrain_ckpt_path)
```

### ClearML Series Naming Convention

```python
# Source: existing train_ctc.py pattern (lines 540-551) + D-09
# Pre-training — series has "pretrain/" prefix, title is unchanged
logger.report_scalar(title="loss", series=f"{series_prefix}train", iteration=epoch, value=train_loss)
logger.report_scalar(title="loss", series=f"{series_prefix}val", iteration=epoch, value=val_loss)
logger.report_scalar(title="cer", series=f"{series_prefix}val", iteration=epoch, value=val_cer)
# series_prefix="pretrain/" → "pretrain/train", "pretrain/val"
# series_prefix=""          → "train", "val" (backward compatible)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `always_apply=True` | `p=1.0` | albumentations 2.0 | Hard break — `always_apply` arg removed |
| `value=0` (fill) | `fill=0` | albumentations 2.0 | Hard break — old name raises TypeError |
| `distort_limit=0.3` (GridDistortion) | `distort_range=(-0.3, 0.3)` | albumentations 2.x | Parameter renamed and made tuple |
| (H, W) numpy direct to transform | (H, W, 1) for direct calls; (H, W) via Compose | albumentations 2.x | Direct calls require channel dim |

**Deprecated/outdated:**
- albumentations `distort_limit` kwarg: replaced by `distort_range` tuple in 2.x
- albumentations `mask_fill_value` → `fill_mask`
- albumentations `value` → `fill` for all fill parameters

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Recommended `elastic_alpha` default-off value is `0` and suggested starting `alpha=30` in help text | Code Examples | Low — alpha=0 disables feature regardless |
| A2 | `distort_range=(-0.15, 0.15)` is the correct 2.x kwarg name for GridDistortion magnitude | Code Examples | Albumentations would raise TypeError at runtime — verified from official API docs [CITED: albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/distortion/] |
| A3 | `series_prefix=""` (empty string) correctly produces `"train"` and `"val"` series names (not `"/train"`) | Code Examples | ClearML series names would have leading slash — visual only, not functional |
| A4 | `tune.py` needs `--pretrain_checkpoint_path` added to its own `_build_parser` | Common Pitfalls | HPO workflow with pre-training would be CLI-unusable |

**If this table is empty:** All other claims were verified or cited.

---

## Open Questions

1. **`_run_loop` debug sample handling**
   - What we know: `_report_annotated_crop`, `_report_prob_heatmap`, `_report_saliency_panel` are called inside the epoch loop and require `val_df`, `debug_samples` (list of `(crop_path, label)`), `model`, `charset`, `device`
   - What's unclear: Pre-training val_df is a synthetic DataFrame — `crop_path` and `label` columns exist (per SYN-02 manifest schema compatibility), so debug samples CAN work during pre-training. But it adds ~5 debug renders per epoch which may be unwanted for synthetic pre-training.
   - Recommendation: Pass `debug_samples=[]` to `_run_loop` when called from `_run_pretrain` to skip debug rendering during synthetic pre-training. Or pass `n_debug=0`.

2. **`_run_loop` signature width**
   - What we know: The proposed signature has ~14 parameters, which approaches the CLAUDE.md ≤5 positional params limit
   - What's unclear: The limit is "≤5 positional params" — all params after the first 5 should be keyword-only
   - Recommendation: Use `*` to enforce keyword-only after position 5, or group related args into a dataclass/namedtuple. Since the function is private (`_run_loop`) and not a public API, no docstring is required.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| albumentations | AugmentTransform elastic path | Not yet installed | 2.0.8 (target) | None — must be added via `uv add` |
| torch | Model, checkpoint | Available | 2.11.0+cpu | — |
| numpy | Tensor↔numpy conversion in AugmentTransform | Available | 2.4.4 | — |
| opencv-python-headless | Albumentations dep | Available | 4.13.0.92 | — |
| scipy | Albumentations dep (transitive) | In uv.lock | (locked) | — |
| pydantic | Albumentations dep (transitive) | In uv.lock (2.13.3) | 2.13.3 | — |
| albucore | Albumentations dep (transitive) | Not yet in uv.lock | 0.0.24 (target) | Pulled in by `uv add albumentations` |

**Missing dependencies with no fallback:**
- `albumentations==2.0.8` — must be added to pyproject.toml via `uv add albumentations==2.0.8`

**Missing dependencies with fallback:**
- None

---

## Security Domain

> security_enforcement not explicitly disabled in config.json — but this phase adds no authentication, sessions, user input, cryptography, or external service calls. The only new external input is `--pretrain_manifest` (a file path) and `--pretrain_checkpoint_path` (also a file path). Both follow the existing pattern of CLI path arguments already present in the codebase.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | — |
| V3 Session Management | No | — |
| V4 Access Control | No | — |
| V5 Input Validation | Partial | Path existence checked before use (same as existing `--manifest` guard: exit code 2 on missing file) |
| V6 Cryptography | No | — |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Loading untrusted checkpoint | Tampering | `torch.load(..., weights_only=True)` — already the established pattern; prevents arbitrary code exec via pickle |

---

## Project Constraints (from CLAUDE.md)

- ≤100 lines/function, cyclomatic complexity ≤8 — `_run_loop` must not grow past this; extract debug reporting into helper if needed
- ≤5 positional params — `_run_loop` has many params; use keyword-only after position 5
- Absolute imports only — `import albumentations as A` is fine; deferred inside function with `# noqa: PLC0415`
- `uv` for deps, `ruff` for lint/format, `ty` for types
- No docstrings on private functions
- Zero warnings policy — ruff, ty must pass clean after changes
- `noqa: F401` on unused imports kept for test patchability (established pattern)
- Replace, don't deprecate — old `_apply_params_file` + related patterns remain unchanged; new flags extend, not replace
- No speculative features — only flags listed in locked decisions

---

## Sources

### Primary (HIGH confidence)
- [albumentations official API docs](https://albumentations.ai/docs/api-reference/albumentations/augmentations/geometric/distortion/) — ElasticTransform and GridDistortion parameter signatures, float32 support
- [PyPI albumentations](https://pypi.org/project/albumentations/) — version 2.0.8, release date 2025-05-27
- `src/ctc_utils.py` lines 156–208 — AugmentTransform current contract (verified by reading)
- `src/train_ctc.py` lines 329–606 — run_training current implementation (verified by reading)
- `src/tune.py` lines 127–171 — _objective Namespace construction (verified by reading)
- albumentations 2.0.8 `requires_dist` from PyPI JSON API — dependency list verified

### Secondary (MEDIUM confidence)
- [albumentations 2.0 breaking changes](https://github.com/albumentations-team/albumentations/releases/tag/2.0.0) — `value` → `fill`, `always_apply` removal, `distort_limit` → `distort_range`
- WebSearch confirmation that albumentations 2.x requires `(H, W, 1)` for direct transform calls

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — albumentations 2.0.8 verified on PyPI with full requires_dist checked
- Architecture: HIGH — all code read directly; decisions locked in CONTEXT.md
- Pitfalls: HIGH — derived from reading actual code (tune.py Namespace pattern, line counts, test failures)

**Research date:** 2026-05-26
**Valid until:** 2026-06-26 (albumentations 2.x API is stable; uv.lock will track exact versions)
