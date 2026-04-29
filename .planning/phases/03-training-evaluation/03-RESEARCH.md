# Phase 3: Training & Evaluation - Research

**Researched:** 2026-04-28
**Domain:** PyTorch CRNN+CTC training, ClearML task logging, Hebrew OCR evaluation
**Confidence:** HIGH

## Summary

Phase 3 builds two independent CLI scripts: `train_ctc.py` trains a CRNN+CTC model on
labeled crops and saves `checkpoint.pt` + `charset.json`; `evaluate.py` loads those artifacts,
runs greedy CTC inference, and exports `eval_report.csv`. Both log to ClearML.

The primary research gaps were: (1) correct `pyproject.toml` config for CPU-only torch via `uv`,
(2) exact `torch.nn.CTCLoss` input shape and normalization, and (3) how to compute CER without
adding a new heavy dependency. All three are resolved below with HIGH confidence from official
docs.

**Primary recommendation:** Use PyTorch 2.11.0 from the PyTorch CPU index via `uv`, implement
CER as a hand-rolled edit-distance (no new dep), follow the `prepare_data.py` argparse +
`Task.connect()` pattern exactly for both new scripts.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01** Input sizing: 64px fixed height, proportional width; batch-padded to longest width via
  custom `collate_fn`.
- **D-02** Output: `--output_dir` CLI arg, default `outputs/model/`; artifacts: `checkpoint.pt`,
  `charset.json`, `eval_report.csv`.
- **D-03** Split unit: half-page. Midpoint = pixel height of page image read from `page_path`.
  Crops with center-y above midpoint → top half (`{page_num}.0`); below → bottom half
  (`{page_num}.1`).
- **D-04** 20% of half-page units to val, rounded up, minimum 1. Sorted deterministically; no
  random seed needed.
- **D-05** Greedy CTC decode: argmax each timestep, collapse consecutive repeats, remove blank.
  No extra dependencies.
- Inherited patterns: `init_task()` before `argparse.parse_args()`, module-level ClearML imports,
  `clearml_utils.py` helpers reused, `task.connect(vars(args), name="hyperparams")`, pandas for
  manifest I/O.
- PyTorch must be added to `pyproject.toml` as a CPU-only wheel.

### Claude's Discretion

- Specific CNN topology (conv layers, filter counts, pooling) — standard small CRNN for document
  OCR is fine.
- Unicode normalization form for Hebrew charset (NFC is standard).
- Minimum labeled-crops guard threshold.
- Batch collate padding implementation detail.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRAN-01 | train_ctc.py loads only crops with status=labeled from manifest.csv | manifest_schema.py `status` column; pandas filter `df[df.status == "labeled"]` |
| TRAN-02 | Charset built dynamically from labeled Hebrew text with Unicode normalization | `unicodedata.normalize('NFC', label)` on each label before set union; sort for determinism |
| TRAN-03 | Train/val split by page (not random crop) to prevent leakage | Half-page unit split algorithm documented below |
| TRAN-04 | Model is CRNN: CNN feature extractor → BiLSTM → CTC loss | Standard pattern; `torch.nn.CTCLoss` built-in; architecture in Code Examples |
| TRAN-05 | Training runs on CPU (device auto-detected via torch.device) | `torch.device("cuda" if torch.cuda.is_available() else "cpu")` — CPU forced at MVP scale |
| TRAN-06 | Best model checkpoint and charset.json saved to disk | `torch.save(model.state_dict(), ...)` + `json.dump(charset, ...)` on val-CER improvement |
| TRAN-07 | Training hyperparameters via CLI, connected to ClearML via Task.connect() | argparse pattern from `prepare_data.py`; `task.connect(vars(args), name="hyperparams")` |
| TRAN-08 | ClearML task `train_baseline_ctc` logs train loss, val loss, val CER per epoch; uploads checkpoint, charset.json, config | `task.get_logger().report_scalar()`; `upload_file_artifact()` from `clearml_utils.py` |
| EVAL-01 | evaluate.py runs inference on validation set using saved checkpoint | Load `checkpoint.pt` with `torch.load()`; load `charset.json`; rebuild model |
| EVAL-02 | CER computed on validation set | Hand-rolled edit distance (no new dep) or `jiwer.cer()` — see Don't Hand-Roll |
| EVAL-03 | eval_report.csv exported: image_path, target, prediction, is_exact | pandas DataFrame; `is_exact = prediction == target` |
| EVAL-04 | ClearML task `evaluate_model` logs final CER, exact match rate, uploads eval_report.csv | `report_scalar()` at iteration=0; `upload_file_artifact()` |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.11.0+cpu | CRNN model, CTCLoss, inference | Project decision; PyTorch is the project ML stack |
| torchvision | not needed | — | Only torch needed; no pretrained vision models |
| pandas | 3.0.2 (already in deps) | manifest I/O | Already in project; all prior scripts use it |
| opencv-python-headless | 4.13.0.92 (already in deps) | Image resize to 64px height | Already installed; avoid adding Pillow |
| unicodedata | stdlib | NFC normalize Hebrew labels | No new dep; stdlib module |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jiwer | 3.x | CER computation (alternative) | Only if hand-rolled edit distance is unacceptable |
| json | stdlib | charset.json read/write | Always; no extra dep |

**Decision on CER:** CER = (edit distance at character level) / len(reference). A hand-rolled
Levenshtein over character lists is ~15 lines, needs zero new deps, and is clearly correct for
this scale (50-120 crops). Use `jiwer` only if the planner explicitly decides the extra dep is
acceptable. `jiwer` is MEDIUM confidence (WebSearch); hand-rolled is HIGH (pure stdlib).

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled CER | `jiwer.cer()` | jiwer adds a dependency (RapidFuzz) but is battle-tested; fine for MVP if you want to avoid writing the algorithm |
| Hand-rolled CER | `torchmetrics.text.CharErrorRate` | torchmetrics is a large dep; not justified for a single metric |

**Installation (new dep only — add to pyproject.toml):**
```bash
uv add torch
```
With pyproject.toml config below — `uv sync` then installs the CPU wheel.

**Version verification (confirmed 2026-04-28):**
```
torch: 2.11.0 (latest stable, released 2026-03-23)
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── train_ctc.py        # CLI: load labeled crops, train CRNN+CTC, save checkpoint
├── evaluate.py         # CLI: load checkpoint, run greedy decode, write eval_report.csv
outputs/
└── model/              # --output_dir default: checkpoint.pt, charset.json, eval_report.csv
tests/
├── test_train_ctc.py   # unit tests: dataset, collate, split, charset
└── test_evaluate.py    # unit tests: decode, CER, report writing
```

### Pattern 1: PyTorch CPU-only via uv (pyproject.toml)

Add to `pyproject.toml` — `torch` goes to `[project] dependencies`, plus two new sections:

```toml
# in [project] dependencies:
"torch==2.11.0",

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

Source: [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) — HIGH confidence.

**Why `explicit = true`:** Prevents uv from resolving unrelated packages (e.g., numpy, pandas)
from the PyTorch wheel server. Only `torch` is routed there.

**Note:** PyPI's `torch` (without `+cpu` suffix) on Linux installs CUDA 13.0 wheels by default
as of 2.11.0. The `[tool.uv.sources]` redirect is required on Linux for true CPU-only builds.

### Pattern 2: CTCLoss Input Format

```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
# log_probs: (T, N, C) — T=time steps, N=batch, C=num_classes (including blank)
# Apply log_softmax along class dim (dim=2)
log_probs = model(images)          # shape: (T, N, C)
log_probs = log_probs.log_softmax(2)
# Blank index defaults to 0; target indices must NOT use 0
ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

`zero_infinity=True` prevents gradient explosions when input is shorter than target — safe
default for variable-width images.

### Pattern 3: CRNN Architecture (small, document OCR)

```python
# Standard minimal CRNN for fixed-height (64px) variable-width crops
class CRNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),   # H:32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),  # H:16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)),  # H:8
        )
        # After CNN: height = 8, channels = 128 → feature dim = 1024
        self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, num_classes)  # 512 = 256 * 2 (bidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, 64, W)
        x = self.cnn(x)                        # (N, 128, 8, W')
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(w, b, c * h)  # (W', N, 1024)
        x, _ = self.rnn(x)                     # (W', N, 512)
        return self.fc(x)                       # (W', N, num_classes) — no softmax here
```

`W'` (time steps T for CTC) = `W // 4` after two `MaxPool2d(2,2)` and one `MaxPool2d((2,1))`.
The `(2,1)` pool halves height without reducing width again, giving more temporal resolution.

**num_classes** = `len(charset) + 1` (the +1 is the blank token at index 0).
Charset indices: blank=0, charset chars=1..N.

### Pattern 4: Custom collate_fn for variable-width batches

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def crnn_collate(batch: list[tuple[torch.Tensor, list[int]]]):
    """Pads images to max width in batch; returns lengths for CTCLoss."""
    images, labels = zip(*batch)
    # images: list of (1, 64, W_i) tensors — variable W
    max_w = max(img.size(2) for img in images)
    padded = torch.zeros(len(images), 1, 64, max_w)
    for i, img in enumerate(images):
        padded[i, :, :, : img.size(2)] = img
    # labels: list of int lists (variable length)
    label_tensor = torch.cat([torch.tensor(l, dtype=torch.long) for l in labels])
    input_lengths = torch.tensor(
        [img.size(2) // 4 for img in images], dtype=torch.long  # divide by CNN width reduction
    )
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return padded, label_tensor, input_lengths, target_lengths
```

Source: [PyTorch docs: collate_fn](https://docs.pytorch.org/docs/stable/data.html) — HIGH confidence pattern.

**Critical:** `input_lengths` must equal the T dimension of `log_probs` fed to CTCLoss. After
two `MaxPool2d(2,2)` and one `MaxPool2d((2,1))`, width is divided by 4. This must match exactly.

### Pattern 5: Half-page train/val split

```python
import cv2
import math

def build_half_page_units(df: pd.DataFrame) -> dict[str, list[int]]:
    """Returns mapping half_page_id -> list of row indices."""
    units: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        img = cv2.imread(row["page_path"], cv2.IMREAD_GRAYSCALE)
        midpoint = img.shape[0] / 2
        center_y = row["y"] + row["h"] / 2
        suffix = ".0" if center_y < midpoint else ".1"
        key = f"{row['page_num']}{suffix}"
        units.setdefault(key, []).append(idx)
    return units

def split_units(units: dict, val_frac: float = 0.2) -> tuple[list, list]:
    keys = sorted(units.keys())  # deterministic
    n_val = max(1, math.ceil(len(keys) * val_frac))
    val_keys = keys[:n_val]      # first n_val units go to val
    train_keys = keys[n_val:]
    return train_keys, val_keys
```

**Pitfall:** Page images for different pages have different heights. Must read height from
`page_path` per-row, not assume a fixed height. Cache by `page_path` to avoid re-reading.

### Pattern 6: ClearML task pattern (matches prepare_data.py)

```python
# Module-level import — required for test patchability
from clearml import Task
from src.clearml_utils import init_task, upload_file_artifact

def main() -> int:
    task = init_task("handwriting-hebrew-ocr", "train_baseline_ctc", tags=["phase-3"])
    parser = _build_parser()
    args = parser.parse_args()
    task.connect(vars(args), name="hyperparams")
    # ... training ...
    logger = task.get_logger()
    logger.report_scalar(title="loss", series="train", iteration=epoch, value=train_loss)
    logger.report_scalar(title="loss", series="val", iteration=epoch, value=val_loss)
    logger.report_scalar(title="cer", series="val", iteration=epoch, value=val_cer)
    upload_file_artifact(task, "checkpoint", checkpoint_path)
    upload_file_artifact(task, "charset", charset_path)
```

### Pattern 7: Greedy CTC decode

```python
def greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """Greedy CTC: argmax → collapse repeats → remove blank. Input: (T, C)."""
    indices = log_probs.argmax(dim=1).tolist()
    result = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != blank:
                result.append(idx)
            prev = idx
    return result
```

No external dep. Matches D-05 exactly.

### Pattern 8: CER computation (hand-rolled)

```python
def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via dynamic-programming edit distance."""
    r, h = list(reference), list(hypothesis)
    if not r:
        return float(len(h))
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(r)][len(h)] / len(r)
```

### Pattern 9: Image preprocessing for Dataset

```python
import cv2
import torch

def load_crop(path: str, target_h: int = 64) -> torch.Tensor:
    """Load grayscale crop, resize to target_h with proportional width, normalize to [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_w = max(1, int(w * target_h / h))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).float() / 255.0  # (H, W)
    return tensor.unsqueeze(0)  # (1, H, W)
```

### Anti-Patterns to Avoid

- **Using `torch.load()` without `weights_only=True`:** In PyTorch 2.x, `torch.load()` without
  `weights_only=True` emits a FutureWarning and will become an error. Use
  `torch.load(path, weights_only=True)` for checkpoint loading.
- **Putting blank at last index:** By convention, blank=0 and Hebrew chars start at index 1.
  If blank is placed at the end, targets must be re-indexed. Stick to blank=0.
- **Random split on crops:** Would allow crops from the same page in both train and val,
  inflating val accuracy. Locked to half-page unit split.
- **Calling argparse.parse_args() before init_task():** ClearML SDK captures argparse at init
  time. Swapping order causes args to not appear in HYPERPARAMETERS.
- **Not caching page heights:** Reading the same page image twice per crop is O(N*pages).
  Cache `{page_path: height}` before the split loop.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CTC loss | Custom CTC gradient | `torch.nn.CTCLoss` | CTC forward/backward is numerically subtle; blank alignment has many edge cases |
| BiLSTM | Custom RNN cell | `torch.nn.LSTM(bidirectional=True)` | Handles gradient flow, hidden state, packing |
| Image padding to batch | Manual numpy stack | Custom `collate_fn` with `torch.zeros` | Collate gives correct `input_lengths` too |
| Checkpoint serialization | Custom pickle/JSON | `torch.save(model.state_dict(), ...)` | Standard; `weights_only=True` is safe to load |

**Key insight:** The hardest part of this phase is wiring CTCLoss correctly (blank index, log_softmax,
input_lengths matching CNN output width). Get that plumbing right first; model topology matters
less at MVP scale.

## Common Pitfalls

### Pitfall 1: input_lengths Mismatch
**What goes wrong:** CTCLoss raises `RuntimeError: Expected input_lengths to be ...` or
silently produces infinite loss.
**Why it happens:** The width passed as `input_lengths` must equal the actual T dimension of
`log_probs` after the CNN reduces width. Each `MaxPool2d(2,2)` halves width; `MaxPool2d((2,1))`
does not. Two `(2,2)` pools → divide by 4. If you pad the image to width W then collate feeds
`W//4` but the CNN output has a different width (due to odd-width edge cases), loss explodes.
**How to avoid:** Compute `input_lengths` from the actual output of the CNN, not from the padded
image width. Or ensure padded widths are always divisible by 4 (pad to nearest multiple of 4).
**Warning signs:** `loss = nan` or `loss = inf` in first epoch.

### Pitfall 2: Blank Index Collision
**What goes wrong:** Model learns to predict blank for everything; training loss oscillates high.
**Why it happens:** If charset index 0 is a real Hebrew character AND blank is also 0, target
sequences contain 0, which CTCLoss forbids (targets cannot equal blank).
**How to avoid:** Reserve index 0 for blank. Charset chars get indices 1..N.
`num_classes = len(charset) + 1`.

### Pitfall 3: RTL Label Encoding
**What goes wrong:** CER is inflated; model never converges on Hebrew text.
**Why it happens:** CRNN reads images left-to-right spatially, but Hebrew text is right-to-left.
The spatial sequence of glyphs in a Hebrew crop runs RTL. The label must encode chars in the
same order they appear visually (RTL), not LTR.
**How to avoid:** When encoding label strings to integer sequences, reverse the string first
(or confirm the label is stored RTL in manifest — if the annotator typed RTL, it may already be
in visual order). Validate with a single crop: print predicted char sequence and compare visually.
**Warning signs:** Predictions are reversed Hebrew text.

### Pitfall 4: log_softmax Missing
**What goes wrong:** CTCLoss NaN or incorrect gradients.
**Why it happens:** CTCLoss expects log-probabilities. Raw model logits fed directly to CTCLoss
will not produce correct gradients.
**How to avoid:** Always apply `.log_softmax(2)` to model output before CTCLoss. Do NOT apply
softmax in the model's forward() — apply it outside, in the loss computation step.
**Warning signs:** Loss values in double digits that never decrease, or NaN immediately.

### Pitfall 5: Page Height Varies Across Pages
**What goes wrong:** Half-page split puts all crops of a page in train because midpoint is wrong.
**Why it happens:** Assuming a fixed page height (e.g., A4 at 300dpi = 3508px) rather than
reading the actual pixel height from `page_path`.
**How to avoid:** Read height from `cv2.imread(page_path).shape[0]`. Cache by path.
**Warning signs:** One set has 0 validation crops from certain pages.

### Pitfall 6: Empty Charset After Filtering
**What goes wrong:** `KeyError` or empty charset.json; training crashes.
**Why it happens:** All crops filtered as `status == "labeled"` have empty `label` string.
**How to avoid:** Guard: fail with a clear error if labeled crops < N or if any labeled crop has
empty label. Context says target is 50-120 crops; fail if fewer than 5 labeled.

### Pitfall 7: torch.load FutureWarning → Error
**What goes wrong:** `evaluate.py` prints warnings or fails in future PyTorch versions.
**Why it happens:** `torch.load(path)` without `weights_only=True` is deprecated since ~2.0.
**How to avoid:** `torch.load(path, weights_only=True)` for state dicts.

## Code Examples

Verified patterns from official sources:

### CTCLoss complete usage
```python
# Source: https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html
import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

# log_probs: (T, N, C) — output of model.forward() after log_softmax
# targets: 1D concatenated target sequences (int64), no blank tokens (no 0)
# input_lengths: (N,) — T dimension per sample
# target_lengths: (N,) — length of each target sequence
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
```

### Charset build
```python
import unicodedata

def build_charset(labels: list[str]) -> list[str]:
    """Build sorted charset from labeled strings (NFC normalized). Index 0 reserved for blank."""
    chars: set[str] = set()
    for label in labels:
        label = unicodedata.normalize("NFC", label)
        chars.update(label)
    return sorted(chars)  # deterministic; index in list = char index (+ 1 for blank offset)
```

### Checkpoint save/load
```python
# Save (train_ctc.py)
torch.save(model.state_dict(), output_dir / "checkpoint.pt")

# Load (evaluate.py)
state = torch.load(output_dir / "checkpoint.pt", weights_only=True)
model = CRNN(num_classes=len(charset) + 1)
model.load_state_dict(state)
model.eval()
```

### eval_report.csv structure
```python
# EVAL-03: required columns
report = pd.DataFrame({
    "image_path": [row["crop_path"] for row in val_rows],
    "target": targets,
    "prediction": predictions,
    "is_exact": [t == p for t, p in zip(targets, predictions)],
})
report.to_csv(output_dir / "eval_report.csv", index=False)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.load(f)` | `torch.load(f, weights_only=True)` | PyTorch 2.0+ | FutureWarning without it |
| pip + requirements.txt for torch | `uv` + `[tool.uv.sources]` index routing | uv 0.5.3+ | CPU-only wheels on Linux without CUDA |
| PyPI torch on Linux | PyTorch CPU wheel index (`download.pytorch.org/whl/cpu`) | PyTorch 2.11 | PyPI default on Linux now installs CUDA 13.0 |

**Deprecated/outdated:**
- `torch.load(path)` without `weights_only`: deprecated, will raise error in future version
- Installing torch from PyPI directly on Linux for CPU-only: installs CUDA 13.0 wheels as of 2.11.0

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All scripts | ✓ | 3.13.13 (project .venv) | — |
| uv | Dependency management | ✓ | 0.11.7 | — |
| torch | Training/evaluation | ✗ (not in project .venv) | — | Must install via `uv add` |
| opencv-python-headless | Image load/resize | ✓ | 4.13.0.92 (in .venv) | — |
| pandas | Manifest I/O | ✓ | 3.0.2 (in .venv) | — |
| ruff | Linting | ✓ | in dev deps | — |
| ty | Type checking | ✓ | in dev deps | — |

**Missing dependencies with no fallback:**
- `torch` is not installed in the project's `.venv` (Python 3.13). Must be added to
  `pyproject.toml` and installed before any Phase 3 scripts can run.

**Missing dependencies with fallback:**
- None beyond torch.

**Note:** A system-level `torch 2.11.0+cu130` exists in `~/git/ccc/venv` (Python 3.12),
but the project uses its own `.venv` at Python 3.13 where torch is absent.

## Open Questions

1. **RTL label storage convention in manifest**
   - What we know: Annotators type Hebrew in the Streamlit review app. Hebrew input methods
     produce RTL strings stored in Unicode logical order (RTL).
   - What's unclear: When a CRNN reads a Hebrew crop left-to-right spatially, does visual order
     match logical order for typical Hebrew handwriting? (Spacing between words can make this
     ambiguous.)
   - Recommendation: In the plan, add a validation step: print first crop's predicted char
     sequence alongside its target to confirm order convention before training full dataset.

2. **Minimum labeled-crop guard threshold**
   - What we know: Target MVP dataset is 50-120 labeled crops (from PROJECT.md context).
   - What's unclear: What threshold should cause `train_ctc.py` to fail fast vs. proceed?
   - Recommendation: Fail with a clear error if fewer than 10 labeled crops found. This is
     Claude's discretion per CONTEXT.md.

3. **CNN width reduction factor**
   - What we know: Architecture uses two `MaxPool2d(2,2)` → width ÷ 4. `input_lengths` in
     CTCLoss must match actual CNN output T dimension exactly.
   - What's unclear: If padded image width W is not divisible by 4, floor division may produce
     off-by-one vs. actual CNN output.
   - Recommendation: In `collate_fn`, pad width up to the nearest multiple of 4 before stacking.
     This eliminates the ambiguity.

## Sources

### Primary (HIGH confidence)
- [PyTorch CTCLoss docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html) — input shape, blank index, log_softmax requirement
- [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/) — CPU-only wheel config
- [Python unicodedata stdlib](https://docs.python.org/3/library/unicodedata.html) — NFC normalization
- [PyTorch collate_fn docs](https://docs.pytorch.org/docs/stable/data.html) — collate_fn interface
- Codebase: `src/clearml_utils.py`, `src/prepare_data.py`, `src/manifest_schema.py` — project patterns

### Secondary (MEDIUM confidence)
- [ClearML hyperparameter docs](https://clear.ml/docs/latest/docs/guides/reporting/hyper_parameters/) — Task.connect pattern
- WebSearch: CRNN+CTC PyTorch minimal architecture (cross-verified with PyTorch docs for CTCLoss input format)
- WebSearch: PyTorch 2.11 release date (cross-verified with GitHub releases)

### Tertiary (LOW confidence)
- WebSearch: RTL label reversal advice — not verified against Hebrew-specific CRNN implementations.
  Flag for validation during implementation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — torch 2.11.0 version confirmed from PyPI; uv config verified from official docs
- Architecture: HIGH — CTCLoss API from official docs; CRNN pattern is well-established
- Pitfalls: MEDIUM — input_lengths/blank/RTL issues verified from docs; RTL convention is LOW (needs empirical validation)
- CER implementation: HIGH — hand-rolled edit distance is trivially correct; jiwer is MEDIUM

**Research date:** 2026-04-28
**Valid until:** 2026-05-28 (stable stack; PyTorch releases quarterly)
