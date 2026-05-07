---
phase: quick/260507-oa8-generate-a-test-dataset-with-the-desired
plan: 01
subsystem: clearml-dataset
tags: [clearml, dataset, integration-test, bug-fix]
dependency_graph:
  requires: []
  provides: [fixed-clearml-upload, dataset-roundtrip-test]
  affects: [src/clearml_utils.py, src/prepare_data.py]
tech_stack:
  added: []
  patterns: [clearml-dataset-path-kwarg, use_current_task-conditional]
key_files:
  created:
    - tests/test_clearml_dataset_roundtrip.py
  modified:
    - src/clearml_utils.py
    - src/prepare_data.py
    - tests/test_clearml_utils.py
decisions:
  - "ClearML kwarg is dataset_path (not target_folder) in clearml==2.1.5"
  - "Chose tuple list signature [Path | tuple[Path, str]] over dict; minimizes diff, preserves backward compat for bare Path callers"
  - "use_current_task conditioned on Task.current_task() is not None: fixes is_final() returning False when no task running"
metrics:
  duration: ~18 minutes
  completed: 2026-05-07T14:40:14Z
  tasks: 2
  files_modified: 4
---

# Quick 260507-oa8: Fix ClearML Dataset Upload Bug and Add Roundtrip Integration Test

**One-liner:** Fixed `dataset_path` kwarg threading and `use_current_task` conditional in
`maybe_create_dataset` so remote agents no longer see "Extracted file missing" and an
integration test proves the fix round-trips through real ClearML before full re-upload.

## What Was Done

### Task 1: Fix `maybe_create_dataset` signature and `prepare_data.py` call site

**Root cause (confirmed):** `prepare_data.py` called `ds.add_files(str(crops_dir))` without
`dataset_path`. ClearML's upload zip stored files under `crops/UUID.png` but the registration
expected them at flat `UUID.png`. `remap_dataset_paths` already expected `<root>/crops/<name>`
so the fix is to make registration match extraction.

**ClearML kwarg confirmed:** `Dataset.add_files(..., dataset_path=...)` — verified via
`uv run python -c "from clearml import Dataset; help(Dataset.add_files)"` against clearml==2.1.5.

**Signature choice:** Extended `folders` to `list[Path | tuple[Path, str]]`. Bare `Path` entries
preserve the old behavior (no `dataset_path`); tuple entries use the second element as
`dataset_path`. This is a back-compat extension with minimal diff — no existing callers break.

**prepare_data.py:** Updated to `folders=[(pages_dir, "pages"), (crops_dir, "crops")]`.

**Unit tests added (2):**
- `test_maybe_create_dataset_uses_target_folder_for_tuple_entries` — asserts tuple entries
  produce `add_files(path, dataset_path="pages"/"crops")` calls
- `test_maybe_create_dataset_bare_path_uses_no_target_folder` — regression guard for bare Path

### Task 2: Integration test + second bug fix

**Integration test created:** `tests/test_clearml_dataset_roundtrip.py` with 3 tests:
1. `test_dataset_roundtrip_extracts_under_pages_and_crops` — verifies `<root>/pages/p1.png`,
   `<root>/crops/p1_top_0.png`, `<root>/manifest.csv` all exist after real ClearML download
2. `test_dataset_roundtrip_remap_paths_resolve` — verifies `remap_dataset_paths` produces
   paths that exist on disk
3. `test_dataset_roundtrip_one_epoch_training` — full end-to-end: upload synthetic dataset,
   download, train 1 epoch, assert `checkpoint.pt` + `charset.json` written

**Second bug found and fixed (Rule 1 — Bug):** `maybe_create_dataset` used
`use_current_task=True` unconditionally. ClearML's `finalize()` with `use_current_task=True`
only calls `task.flush()`, not `task.close()` + `task.mark_completed()`. This leaves the
dataset task in "in_progress" status. `is_final()` checks task status, so it returned False,
and `get_local_copy()` raised `ValueError: Cannot get a local copy of a dataset that was not
finalized/closed`.

**Fix:** Conditioned on `Task.current_task() is not None`. When a task IS running (the normal
`prepare_data.py` flow), `use_current_task=True` attaches to it (D-01 constraint preserved).
When no task is running (standalone test calls), `use_current_task=False` so `finalize()`
properly calls `mark_completed()`.

**Dataset layout verified in ClearML web UI:** synthetic datasets created under project
`handwriting-hebrew-ocr-test` with files visible under `pages/` and `crops/` paths.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `use_current_task=True` prevents `get_local_copy()` without active task**
- **Found during:** Task 2, first integration test run
- **Issue:** `finalize()` with `use_current_task=True` only flushes, never marks task
  completed. `is_final()` returns False. `get_local_copy()` raises ValueError.
- **Fix:** `use_current_task = Task.current_task() is not None` before `Dataset.create()`
- **Files modified:** `src/clearml_utils.py`, `tests/test_clearml_utils.py` (updated 3 tests
  to mock `Task.current_task` for the `use_current_task` assertion)
- **Commits:** 8aaedbb

**2. [Rule 2 - Missing functionality] `test_maybe_create_dataset_full_lifecycle` assertion stale**
- Updated test to mock `Task.current_task` returning a non-None value so assertion matches
  the new conditional behavior when a task IS running.

## Performance

- All 10 `tests/test_clearml_utils.py` tests pass (< 1s)
- All 3 `tests/test_clearml_dataset_roundtrip.py` tests pass (~134s — real ClearML uploads)
- `uv run ruff check src tests` clean
- `uv run ty check src tests` clean

## Commits

| Hash | Description |
|------|-------------|
| 7fb35bc | feat(quick-260507-oa8): fix maybe_create_dataset to register folders under dataset_path prefix |
| 8aaedbb | feat(quick-260507-oa8): add integration test for ClearML dataset roundtrip + fix finalization |

## Follow-up

Re-upload the full production dataset using the fixed `prepare-data` CLI. The correct call is
now:
```bash
uv run python -m src.prepare_data \
  --pdf_dir <pdf_dir> \
  --output_dir <output_dir> \
  --dataset_name data-pipeline-v1.0.3
```
The `folders=[(pages_dir, "pages"), (crops_dir, "crops")]` fix is already in `prepare_data.py`.
After re-upload, resubmit GPU training with `--dataset_id <new_id>`.

## Known Stubs

None.

## Self-Check: PASSED

- tests/test_clearml_dataset_roundtrip.py: FOUND
- src/clearml_utils.py: FOUND
- src/prepare_data.py: FOUND
- tests/test_clearml_utils.py: FOUND
- Commit 7fb35bc: FOUND
- Commit 8aaedbb: FOUND
