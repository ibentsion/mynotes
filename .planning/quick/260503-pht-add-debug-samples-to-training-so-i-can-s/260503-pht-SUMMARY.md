---
phase: quick-260503-pht
plan: 01
subsystem: training
tags: [training, clearml, debug, ctc]
dependency_graph:
  requires: [src.ctc_utils.predict_single]
  provides: [per-epoch debug-sample text logs in ClearML]
  affects: [src/train_ctc.py]
tech_stack:
  added: []
  patterns: [ClearML report_text with msg= kwarg]
key_files:
  modified: [src/train_ctc.py]
decisions:
  - Used msg= keyword for logger.report_text — installed ClearML Logger.report_text signature is (msg, level, print_console, ...), not body=
  - DEBUG_SAMPLES kept as module-level constant, not CLI arg, per quick-task scope
metrics:
  duration: ~5 min
  completed: 2026-05-01
---

# Quick Task 260503-pht: Add Debug Samples to Training Summary

**One-liner:** Per-epoch ClearML text panel showing predictions for 5 fixed val crops so training progress is visible at a glance.

## What Was Done

Added debug-sample logging to `src/train_ctc.py`:

- `predict_single` imported from `src.ctc_utils` (alphabetical position in the import block)
- `DEBUG_SAMPLES = 5` constant placed after imports, before `_build_parser`
- `val_df` extracted once from `labeled.iloc[val_idx].reset_index(drop=True)` and reused for both `CropDataset` and the `debug_samples` list
- `debug_samples`: list of `(crop_path, label)` tuples for the first `min(5, len(val_df))` val rows — captured once, before the epoch loop
- After the three `report_scalar` calls each epoch, a `torch.no_grad()` block runs `predict_single` on each debug crop and builds a text block in the documented format
- Logged via `logger.report_text(title="debug_samples", series="val", iteration=epoch, print_console=False, msg=text_block)`
- Existing console `print(...)`, scalar reporting, checkpoint save, and return codes are unchanged

## Decisions Made

- **`msg=` not `body=`**: The installed ClearML SDK's `Logger.report_text` takes `msg` as the first positional/keyword argument. The plan's context already flagged this as a possible deviation — confirmed via `inspect.signature` and used `msg=` accordingly.
- **No CLI flag**: `DEBUG_SAMPLES` is a constant per the plan's explicit instruction.

## Deviations from Plan

None — plan executed exactly as written. The `msg=` vs `body=` question was pre-answered in the plan context; no structural deviation.

## Verification

- `ruff check src/train_ctc.py`: 0 warnings
- `ruff format --check src/train_ctc.py`: already formatted
- `ty check src/train_ctc.py`: 0 errors
- `uv run pytest tests/test_train_ctc.py -q`: 8 passed

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1    | 1a9638f | feat(quick-260503-pht-01): add per-epoch debug-sample logging to CTC training loop |

## Self-Check: PASSED

- [x] `src/train_ctc.py` modified and committed at 1a9638f
- [x] `predict_single` imported from `src.ctc_utils`
- [x] `DEBUG_SAMPLES = 5` constant present
- [x] `debug_samples` list materialized once before epoch loop
- [x] `logger.report_text(title="debug_samples", ...)` call present in epoch loop
- [x] All linters and type checker pass with zero warnings
- [x] All 8 existing tests pass
