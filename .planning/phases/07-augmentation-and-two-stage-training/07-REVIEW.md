---
phase: 07-augmentation-and-two-stage-training
reviewed: 2026-05-29T11:30:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - src/ctc_utils.py
  - src/train_ctc.py
  - src/tune.py
  - tests/test_ctc_utils.py
  - tests/test_train_ctc.py
  - pyproject.toml
  - src/clearml_utils.py
  - src/run_config.py
findings:
  critical: 3
  warning: 4
  info: 2
  total: 9
status: fixed
fixes_applied: 7
---

# Phase 07: Code Review Report

**Reviewed:** 2026-05-29T11:30:00Z
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 7 adds elastic augmentation, `aug_copies`-based dataset expansion, and a two-stage
pre-train/fine-tune workflow. The core logic is sound, but three correctness defects were found:
(1) `tune.py`'s per-trial `Namespace` is missing `words_file`, which causes an `AttributeError`
whenever a caller has set a `--words_file` in `config.yaml` and the HPO path is used;
(2) `blank_bias_init` is overwritten by `load_state_dict` when a pre-trained checkpoint is loaded,
defeating the intent of the flag; and (3) `best_val_cer` in tune's summary print can be `None`,
causing a `TypeError` crash at the format string. Four warnings round out logic gaps in the
augmentation seeding contract and a dead parameter.

---

## Critical Issues

### CR-01: `tune.py` `_objective` Namespace missing `words_file` — `AttributeError` at runtime

**File:** `src/tune.py:141-171`
**Issue:** `_objective` constructs `train_args` with an explicit `argparse.Namespace` that omits
`words_file`. `run_training` accesses `args.words_file` unconditionally at line 711 of
`train_ctc.py` (`if args.words_file is not None:`). When the caller has `words_file` set via
`config.yaml` hyperparams (injected by `load_config` / `parser.set_defaults`), the fine-tuning
path of `run_training` raises `AttributeError: 'Namespace' object has no attribute 'words_file'`.
Unlike other `getattr(args, "x", None)` guards used elsewhere in the file, this attribute is
accessed directly.

**Fix:**
```python
train_args = argparse.Namespace(
    ...
    words_file=None,  # add this line
    pretrain_manifest=None,
    ...
)
```

---

### CR-02: `blank_bias_init` silently overwritten when loading a pre-trained checkpoint

**File:** `src/train_ctc.py:731-734`
**Issue:** The fine-tune path applies `blank_bias_init` to `model.fc.bias.data[0]` on line 731,
then immediately overwrites the entire model state (including that bias) by loading the pre-trained
checkpoint on line 734:

```python
model.fc.bias.data[0] = args.blank_bias_init   # line 731 — set
...
model.load_state_dict(state)                    # line 734 — overwrites it
```

The initialization has zero effect when a pretrain checkpoint is provided, which contradicts the
documented intent of `--blank_bias_init` and the help text which says to "try -3.0 or -4.0 if
blank_frac stays > 0.9 after training stabilizes". The pre-trained checkpoint already encodes a
blank bias learned on synthetic data; fine-tuning silently resets it to whatever was in the
checkpoint, making `--blank_bias_init` a no-op for the two-stage path.

**Fix:** Apply the blank bias override *after* loading the checkpoint, so the user's explicit
setting wins:
```python
if getattr(args, "pretrain_checkpoint_path", None) is not None:
    state = torch.load(args.pretrain_checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded pre-trained weights from {args.pretrain_checkpoint_path}")
model.fc.bias.data[0] = args.blank_bias_init   # always apply after any checkpoint load
```

---

### CR-03: `tune.py` main crashes with `TypeError` when `best_val_cer` is `None`

**File:** `src/tune.py:259`
**Issue:** `_write_best_params` stores `best_val_cer = None` when `best.value is None` (line 208).
After the `study.best_trial is None` guard passes, `best_params["best_val_cer"]` can still be
`None` if Optuna records a completed trial with `value=None` (edge case in some Optuna versions).
Line 259 then does:

```python
print(f"Best trial {best_params['trial_number']}: CER={best_params['best_val_cer']:.4f}")
```

`None:.4f` raises `TypeError: unsupported format character`. In normal Optuna operation
`study.best_trial` implies `value is not None`, but the defensive `None` guard in
`_write_best_params` signals the author anticipated the case. The format string does not.

**Fix:**
```python
cer_str = f"{best_params['best_val_cer']:.4f}" if best_params['best_val_cer'] is not None else "N/A"
print(f"Best trial {best_params['trial_number']}: CER={cer_str}")
```

---

## Warnings

### WR-01: Elastic deformation in `AugmentTransform` ignores the `seed` parameter

**File:** `src/ctc_utils.py:213-238`
**Issue:** When `elastic_alpha > 0`, the albumentations `A.Compose` pipeline is instantiated
without seeding. The `seed` parameter sets a PyTorch `rng` (line 187) used for rotation and
brightness steps, but albumentations uses its own numpy-based RNG that is never seeded from it.
This means:
- `AugmentTransform(elastic_alpha=30.0)(tensor, seed=42)` is non-deterministic even with an
  explicit seed — the rotation/brightness steps are deterministic but the elastic step is not.
- The docstring promises "deterministic output" for `seed=int`, which is violated.
- Tests `test_augment_transform_elastic_preserves_shape` and
  `test_augment_transform_elastic_values_clamped` rely on a fixed seed expecting reproducibility
  that doesn't hold for the elastic step.

**Fix:** Seed numpy before calling albumentations when a seed is provided:
```python
if self.elastic_alpha > 0:
    import albumentations as A
    import numpy as np
    if seed is not None:
        np.random.seed(seed)   # seed albumentations' numpy RNG
    transform = A.Compose([...])
    ...
```

---

### WR-02: Unused parameter `ctc_loss_per_sample` in `_eval_val_epoch`

**File:** `src/train_ctc.py:375`
**Issue:** `_eval_val_epoch` declares `ctc_loss_per_sample` as a keyword-only argument but never
calls it inside the function body. The parameter is only used inside `_run_loop` (line 486) for
counting inf-loss batches during training. Accepting it in the val function is dead interface
surface that misleads readers into thinking per-sample val losses are computed there.

**Fix:** Remove the parameter from `_eval_val_epoch`'s signature and all call sites that pass it:
```python
# in _eval_val_epoch signature:
def _eval_val_epoch(
    model: Any,
    val_loader: Any,
    val_df: Any,
    *,
    ctc_loss: Any,
    # remove ctc_loss_per_sample
    charset: list[str],
    device: Any,
) -> ...:
```

---

### WR-03: `effective_n` in `_setup_finetune_loaders` counts only real crops, excludes synthetic

**File:** `src/train_ctc.py:649-652`
**Issue:** When `synthetic_df` is appended to `train_base_df` (line 656-658), the logged
`effective_train_size` and printed "effective dataset size" count only `len(train_idx)` real
crops (line 648), not `len(train_base_df) * (1 + aug_copies)`. The actual `CropDataset` is built
from `train_base_df` which includes synthetic rows, making the reported number undercount the true
training set size by however many synthetic crops were added. This is a diagnostic/logging
discrepancy that can cause confusion during training.

```python
# Current (wrong when synthetic_df is non-None):
effective_n = len(train_idx) * (1 + args.aug_copies)

# Correct (computed after concat):
train_base_len = len(train_idx) + (len(synthetic_df) if synthetic_df is not None else 0)
effective_n = train_base_len * (1 + args.aug_copies)
```
Note: the `effective_n` computation and print happen before `synthetic_df` is known (augment
block runs before the synthetic concat). Move the print and `task.connect` call to after the
concat.

---

### WR-04: `val_df` row indexing in `_eval_val_epoch` is fragile and will silently corrupt when val split has an augmented `CropDataset`

**File:** `src/train_ctc.py:388-414`
**Issue:** `_eval_val_epoch` manually tracks a `dataset_idx` counter to map val loader batches
back to `val_df.iloc[dataset_idx + n]` for per-sample CER logging. This only works correctly if
the val DataLoader iterates through rows in the same order as `val_df`, with no augmentation,
shuffle=False, and drop_last=False. Currently val is constructed with `augment=None` and
`shuffle=False`, which satisfies those constraints. However, there is no enforcement — a future
caller who passes a shuffled or augmented val loader would produce silently wrong per-sample CER
attribution (incorrect crop paths and ground-truth labels mapped to the wrong decoded output).
The approach of reconstructing ground truth from `labels` tensor (offset arithmetic) is already
present and correct; the `val_df` indexing adds fragility for no benefit beyond logging the crop
path.

**Fix:** Derive the crop path from `tgt_text` attribution already available in the loop, or at
minimum add a `drop_last=False, shuffle=False` assertion comment near the val loader construction
that makes the assumption explicit.

---

## Info

### IN-01: `from torch.utils.data import Dataset` placed mid-file with E402 suppression

**File:** `src/ctc_utils.py:248`
**Issue:** The `Dataset` import is placed after the class definitions it logically supports, with
an `# noqa: E402` suppression. This is a code organisation smell — nothing prevents moving it to
the top of the file with the other torch imports at lines 9-11.

**Fix:** Move to the import block at the top:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
```

---

### IN-02: `run_config.py` uses a relative `Path("config.yaml")` as a module-level default

**File:** `src/run_config.py:6`
**Issue:** `CONFIG_PATH = Path("config.yaml")` resolves relative to the current working directory
at call time, not relative to the module file. Callers running from a directory other than the
repo root (e.g., `pytest` invoked from a subdirectory, or a notebook in a different cwd) will
silently get an empty config dict rather than an error. The `CONFIG_PATH` env-var override
(line 11) is a workaround but not discoverable. This is consistent with other path defaults in
the project but worth flagging as it has caused silent mis-configuration before.

**Fix:** No immediate code change needed if the project convention is always to run from repo
root. Document the assumption explicitly, or use `Path(__file__).parent.parent / "config.yaml"`
for a repo-relative default.

---

_Reviewed: 2026-05-29T11:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
