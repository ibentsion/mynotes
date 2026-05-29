# Phase 7: Augmentation & Two-Stage Training - Context

**Gathered:** 2026-05-26
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase delivers two independent capabilities wired into `train_ctc.py` and `ctc_utils.py`:

1. **Elastic deformation** — `AugmentTransform` gains `ElasticTransform` + `GridDistortion` via
   `albumentations`, applied after existing rotation/brightness/noise. Configurable via
   `--elastic_alpha` (default 0 = off) and `--elastic_sigma`. Elastic applies to the real
   fine-tuning train split only — not to synthetic pre-training, not to val.

2. **Two-stage training (two-call approach)** — `train_ctc.py` gains two new modes:
   - **Pre-train mode** (`--pretrain_manifest` + `--pretrain_epochs`): trains on synthetic crops,
     logs to ClearML with `pretrain/` series prefix, saves `checkpoint_pretrain.pt` as a
     ClearML artifact. Does NOT proceed to fine-tuning.
   - **Fine-tune mode** (`--pretrain_checkpoint_path`): loads pre-trained weights, resets
     optimizer to `--lr`, runs fine-tuning on real labeled data with existing behavior.

   HPO trials (tune.py) use `--pretrain_checkpoint_path` so all trials start from the same
   pre-trained weights — pre-training runs once, outside of HPO.

**ROADMAP update required:** Phase 7 success criterion currently says "one call does both stages."
Update to reflect the two-call interface decided here.

Out of scope: changes to tune.py internals, evaluate.py, or anything outside `ctc_utils.py` and
`train_ctc.py`.

</domain>

<decisions>
## Implementation Decisions

### Elastic Deformation

- **D-01:** Add `albumentations` as a dependency. Elastic deformation is implemented via
  `albumentations.ElasticTransform` + `albumentations.GridDistortion` applied inside
  `AugmentTransform.__call__` after the existing rotation/brightness/noise transforms.
  The `(1, H, W) float32 tensor` is converted to `(H, W) float32 numpy` → apply albumentations
  → convert back to `(1, H, W) tensor`. Conversion happens inside `__call__` only when
  `elastic_alpha > 0`.
- **D-02:** Conservative defaults to protect Hebrew diacritics — `--elastic_alpha 0` (off by
  default), `--elastic_sigma` at a small value (Claude's discretion, ~4-6). When `elastic_alpha
  == 0`, no albumentations import occurs in the hot path.
- **D-03:** Elastic applies to the **real fine-tuning train split only**. Synthetic pre-training
  crops are already distorted by TRDG rendering — piling elastic on top is redundant and may
  degrade pre-training signal. Val split always receives clean crops (Phase 4 D-04 unchanged).

### Two-Stage Training Loop Architecture

- **D-04:** Extract `_run_loop(model, train_ds, val_ds, optimizer, scheduler, ctc_loss, device,
  args, logger, series_prefix)` as a private helper containing the epoch loop. `run_training()`
  calls `_run_loop()` once (fine-tuning only) or delegates to a `_run_pretrain()` function when
  `--pretrain_manifest` is set. Keeps each function under 100 lines (CLAUDE.md constraint).
- **D-05:** **Pre-train mode** (`--pretrain_manifest` set): build synthetic dataset, random
  train/val split using `--val_frac` (no page structure in synthetic data), run pre-training
  loop with `series_prefix="pretrain/"`, upload `checkpoint_pretrain.pt` as ClearML artifact,
  then return — do NOT proceed to fine-tuning. This is a standalone invocation.
- **D-06:** **Fine-tune mode** (`--pretrain_checkpoint_path` set): load model weights from the
  specified checkpoint path before starting training. Create a fresh `AdamW` optimizer with
  `--lr` (not `--pretrain_lr`) and a fresh `ReduceLROnPlateau` scheduler. Log with existing
  series names (`train_loss`, `val_loss`, `val_cer`) for backward compatibility.
- **D-07:** Optimizer is fully reset between stages — pre-training and fine-tuning each get their
  own `AdamW` instance with the appropriate learning rate (`--pretrain_lr` vs `--lr`). LR
  scheduler resets too (fresh `ReduceLROnPlateau`).

### Pre-Training Validation

- **D-08:** Pre-training val split is a **random split** using `--val_frac` (same default as
  fine-tuning, 0.15). No `build_half_page_units()` — synthetic rows have no `page_path` structure.
  Random indices split by `val_frac` applied to the synthetic manifest rows.

### ClearML Reporting and Checkpoints

- **D-09:** Single ClearML task per invocation. Pre-training scalars logged under `pretrain/`
  series prefix (`pretrain/train_loss`, `pretrain/val_loss`, `pretrain/val_cer`). Fine-tuning
  uses existing series names for backward compatibility.
- **D-10:** `checkpoint_pretrain.pt` uploaded as a ClearML artifact on the pre-training task.
  Fine-tuning run receives `--pretrain_checkpoint_path` pointing to the local path (downloaded
  from ClearML cache or specified directly). Only the final fine-tuned `checkpoint.pt` is the
  production artifact.

### Claude's Discretion

- Exact albumentations `ElasticTransform` + `GridDistortion` parameter defaults for `--elastic_sigma`
  and grid distort strength (prioritize legibility at small alpha values)
- Whether `_run_loop()` returns `best_val_cer: float` or a richer result struct
- Exact series name casing/separators for pretrain scalars (`pretrain/val_cer` vs `pretrain_val_cer`)
- ClearML artifact name for the pre-trained checkpoint

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Code to Extend
- `src/ctc_utils.py` lines ~156–208 — `AugmentTransform.__init__` and `__call__`: the class to
  extend with elastic; note the `(1, H, W) float32 tensor` contract and `padding_mode="border"` pattern
- `src/train_ctc.py::run_training()` — the function to refactor; note the `on_epoch_end` callback
  used by tune.py (must be preserved in `_run_loop` or equivalent)
- `src/tune.py` — how it calls `run_training(args, on_epoch_end=...)` per trial; fine-tuning from
  `--pretrain_checkpoint_path` must remain compatible with this call site

### Reusable Utilities
- `src/clearml_utils.py` — `init_task()`, `upload_file_artifact()` for uploading `checkpoint_pretrain.pt`

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v1.1 Requirements — AUG-01, AUG-02, TRAIN-01, TRAIN-02: exact
  acceptance criteria (note: two-call interface supersedes TRAIN-01/TRAIN-02 literal phrasing)
- `.planning/ROADMAP.md` §Phase 7 — success criteria to update (two-call interface)

### Prior Phase Decisions
- `.planning/phases/04-data-augmentation-and-gpu-training-via-clearml-agent/04-CONTEXT.md` —
  D-01 through D-05: augmentation patterns, no-flip rule, conservative params, online-only, val always clean
- `.planning/phases/06-synthetic-generation/06-CONTEXT.md` — D-07/D-08: synthetic ClearML task
  name, manifest schema for pre-training input

### Project Constraints
- `.planning/PROJECT.md` §Constraints — no additional heavy dependencies (albumentations is
  explicitly approved here as required for elastic; no other new deps)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `AugmentTransform` in `src/ctc_utils.py` — extend `__init__` to accept `elastic_alpha=0.0` and
  `elastic_sigma=4.0`; extend `__call__` to apply albumentations transforms when `elastic_alpha > 0`
- `run_training()` in `src/train_ctc.py` — refactor by extracting the epoch loop into `_run_loop()`;
  the `on_epoch_end` callback must survive the refactor (tune.py depends on it)
- `upload_file_artifact()` in `src/clearml_utils.py` — reuse to upload `checkpoint_pretrain.pt`
- `remap_dataset_paths()` in `src/clearml_utils.py` — reuse to remap synthetic manifest paths when
  loading via `--pretrain_dataset_id` (if needed)

### Established Patterns
- `argparse` + `task.connect(vars(args))` (TRAN-07): add new flags (`--elastic_alpha`,
  `--elastic_sigma`, `--pretrain_manifest`, `--pretrain_epochs`, `--pretrain_lr`,
  `--pretrain_checkpoint_path`) following this pattern
- Module-level ClearML imports for test patchability (established across all phases)
- `outputs/` directory for checkpoints; `checkpoint.pt` is the canonical fine-tuned output name
- `# noqa: PLC0415` for deferred imports inside `run_training()` (existing pattern)

### Integration Points
- `tune.py::_objective()` calls `run_training(args, on_epoch_end=...)` — the refactored
  `run_training()` must still accept `on_epoch_end` and behave identically when
  `--pretrain_manifest` is not set
- `CropDataset` in `ctc_utils.py` passes crops through `AugmentTransform.__call__` — the new
  elastic path must not break the `(1, H, W)` tensor in/out contract
- `config.yaml` (added in quick task 260526-o8k) is read by training to get dataset IDs —
  no changes expected here, but planner should be aware it exists

</code_context>

<specifics>
## Specific Ideas

- **Albumentations conversion pattern:** inside `AugmentTransform.__call__`, when `elastic_alpha > 0`:
  `img_np = tensor.squeeze(0).numpy()` → apply albumentations → `tensor = torch.from_numpy(result).unsqueeze(0)`
- **Two-call workflow:** `uv run generate-synthetic --count 5000 --output_dir outputs/synthetic/` →
  `uv run train-ctc --pretrain_manifest outputs/synthetic/manifest.csv --pretrain_epochs 30` →
  `uv run train-ctc --manifest outputs/manifest.csv --pretrain_checkpoint_path outputs/checkpoint_pretrain.pt`
- **HPO from pre-trained weights:** `uv run tune-hpo --pretrain_checkpoint_path outputs/checkpoint_pretrain.pt`
  (tune.py passes `--pretrain_checkpoint_path` to each trial via args)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 07-augmentation-and-two-stage-training*
*Context gathered: 2026-05-26*
