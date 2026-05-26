---
phase: quick
plan: 260526-nk3
subsystem: clearml-integration
tags: [synthetic-data, clearml, training]
dependency_graph:
  requires: [outputs/synthetic/crops/*.png, outputs/synthetic/manifest.csv]
  provides: [scripts/register_synthetic_dataset.py, remap_synthetic_paths, train_ctc --synthetic_dataset_id, tune --synthetic_dataset_id]
  affects: [src/train_ctc.py, src/tune.py, src/clearml_utils.py]
tech_stack:
  added: []
  patterns: [deferred-clearml-import-inside-run_training, getattr-guard-for-namespace-backward-compat]
key_files:
  created: [scripts/register_synthetic_dataset.py]
  modified: [src/clearml_utils.py, src/train_ctc.py, src/tune.py]
decisions:
  - getattr guard on synthetic_dataset_id in run_training keeps it backward-compatible when tune.py passes a Namespace without the field
  - Deferred clearml.Dataset import inside run_training matches existing pattern (after execute_remotely boundary)
metrics:
  duration_minutes: 15
  completed_date: "2026-05-26"
  tasks_completed: 2
  files_modified: 4
---

# Quick Task 260526-nk3: Register Synthetic Dataset and Wire into Training Summary

**One-liner:** ClearML dataset registration script for outputs/synthetic/ plus --synthetic_dataset_id flag wired into train_ctc and tune to concat synthetic crops into train split only.

## What Was Done

Implemented three coordinated changes to make synthetic Hebrew crops available to remote GPU training runs:

1. **`src/clearml_utils.py`** — Added `remap_synthetic_paths()` function that remaps only `crop_path` (no `page_path` — synthetic rows have none). Follows the same Dataset.get + get_local_copy pattern as `remap_dataset_paths`.

2. **`scripts/register_synthetic_dataset.py`** — New standalone CLI that validates `outputs/synthetic/crops/` (PNGs) and `outputs/synthetic/manifest.csv` exist, then calls `maybe_create_dataset` with the crops folder mapped to `"crops"` and the manifest as a file. Prints the returned dataset ID. No Task.init (pure dataset registration script).

3. **`src/train_ctc.py`** — Added `--synthetic_dataset_id` argument to `_build_parser()`. In `run_training()`, when the arg is set: fetches the synthetic manifest from ClearML, reads it, filters for `status == "labeled"`, remaps crop paths via `remap_synthetic_paths`, and `pd.concat`s onto the real train split after `split_units()`. Val split is never touched. Uses `getattr` guard for backward-compat with callers that omit the field.

4. **`src/tune.py`** — Added `--synthetic_dataset_id` to `_build_parser()` and forwarded it as `synthetic_dataset_id=sweep_args.synthetic_dataset_id` in `_objective()`'s `argparse.Namespace` construction.

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1: remap_synthetic_paths + register script | a92d47a | src/clearml_utils.py, scripts/register_synthetic_dataset.py |
| 2: wire --synthetic_dataset_id | f66399d | src/train_ctc.py, src/tune.py |

## Verification

- `ruff check` and `ty check`: all pass, zero warnings
- `pytest tests/`: 185 passed

## Deviations from Plan

**1. [Rule 1 - Bug] Fixed import line length violation**
- **Found during:** Task 2 ruff check
- **Issue:** `from src.clearml_utils import init_task, remap_dataset_paths, remap_synthetic_paths, upload_file_artifact` was 105 chars (limit 100)
- **Fix:** Reformatted to multi-line parenthesized import
- **Files modified:** src/train_ctc.py
- **Commit:** f66399d (included in task commit)

## Known Stubs

None — no data flows to UI, all wiring is in training pipeline.

## Self-Check: PASSED

- `scripts/register_synthetic_dataset.py`: FOUND
- `src/clearml_utils.py` contains `remap_synthetic_paths`: FOUND
- Commit a92d47a: FOUND
- Commit f66399d: FOUND
