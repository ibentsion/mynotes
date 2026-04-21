---
phase: 01-data-pipeline
plan: "04"
subsystem: pipeline
tags: [pdf2image, opencv, pandas, clearml, argparse, pytest]

requires:
  - phase: 01-02
    provides: clearml_utils helpers (init_task, upload_file_artifact, report_manifest_stats, maybe_create_dataset)
  - phase: 01-03
    provides: region_detector (preprocess_page, detect_regions) and flagging (flag_region, FLAG_NAMES)

provides:
  - src/prepare_data.py CLI that converts PDF folders to grayscale crops + manifest.csv + review_queue.csv + ClearML task
  - src/manifest_schema.py MANIFEST_COLUMNS constant (13 columns, single source of truth)
  - tests/fixtures/make_synthetic_pdf.py Pillow-based synthetic PDF generator for CI
  - tests/test_prepare_data.py end-to-end smoke test + schema unit test

affects: [02-review-app, 03-training]

tech-stack:
  added: []
  patterns:
    - "Subprocess-based smoke test for CLI modules (captures returncode + stdout/stderr)"
    - "ClearML Dataset.create with use_current_task=True to avoid Task.init conflict inside running task"
    - "paths_only=True in convert_from_path to avoid loading all pages into RAM (Pitfall 1)"
    - "Task.init before argparse.parse_args() for ClearML arg auto-logging (Pitfall 2)"

key-files:
  created:
    - src/prepare_data.py
    - src/manifest_schema.py
    - tests/fixtures/__init__.py
    - tests/fixtures/make_synthetic_pdf.py
    - tests/test_prepare_data.py
  modified:
    - src/clearml_utils.py
    - tests/test_clearml_utils.py

key-decisions:
  - "Dataset.create(use_current_task=True) required — calling without it triggers Task.init conflict when a task is already running"
  - "Subprocess timeout set to 120s in smoke test — ClearML init adds ~5-15s; pipeline itself is fast"
  - "ty: ignore comment used on convert_from_path return type — pdf2image lacks overloaded stubs for paths_only=True"

patterns-established:
  - "Manifest schema: MANIFEST_COLUMNS constant in src/manifest_schema.py is the single source of truth for column names and order"
  - "ClearML offline mode (CLEARML_OFFLINE_MODE=1) used in all tests to prevent server calls"

requirements-completed: [DATA-01, DATA-04, DATA-05, DATA-06, FLAG-06, CLML-01, CLML-02, CLML-04, CLML-05]

duration: 35min
completed: 2026-04-21
---

# Phase 1 Plan 4: Prepare Data Pipeline Summary

**prepare_data.py CLI wiring pdf2image + OpenCV + flagging heuristics into manifest.csv + review_queue.csv + ClearML data_prep task, verified by subprocess smoke test on synthetic PDF**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-21T17:10:00Z
- **Completed:** 2026-04-21T17:45:00Z
- **Tasks:** 2 of 3 completed (Task 3 is human-verify checkpoint — awaiting user)
- **Files modified:** 7

## Accomplishments

- Created `src/manifest_schema.py` with `MANIFEST_COLUMNS` (13 columns) as single source of truth for pipeline and downstream phases
- Implemented `src/prepare_data.py`: full CLI pipeline from PDF directory to grayscale crops + manifest.csv + review_queue.csv + ClearML task with all args auto-logged
- Created `tests/fixtures/make_synthetic_pdf.py` Pillow-based PDF generator and `tests/test_prepare_data.py` with 2 tests; full suite of 26 tests passes

## Task Commits

1. **Task 1: manifest schema + synthetic fixture + failing smoke test (RED)** - `07db6b9` (test)
2. **Task 2: implement prepare_data.py pipeline (GREEN)** - `3a91691` (feat)
3. **Task 3: human-verify on real PDF** - PENDING checkpoint

## Files Created/Modified

- `src/prepare_data.py` - Main pipeline CLI: PDF dir -> manifest.csv + review_queue.csv + ClearML task
- `src/manifest_schema.py` - MANIFEST_COLUMNS constant (13 columns)
- `src/clearml_utils.py` - Fixed maybe_create_dataset to use use_current_task=True
- `tests/fixtures/__init__.py` - Empty init for fixtures package
- `tests/fixtures/make_synthetic_pdf.py` - Pillow-based synthetic PDF generator
- `tests/test_prepare_data.py` - Smoke test + schema unit test
- `tests/test_clearml_utils.py` - Updated assertion for use_current_task=True kwarg

## Decisions Made

- `Dataset.create(use_current_task=True)` — ClearML raises `UsageError` if `Dataset.create` is called inside a running task without this flag; it internally calls `Task.init()` which conflicts.
- `ty: ignore[invalid-argument-type]` on `Path(page_path_str)` — pdf2image lacks typed overloads for `paths_only=True`, so the type checker sees the return as `Image` objects rather than strings.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ClearML Dataset.create conflict inside running task**
- **Found during:** Task 2 (implement prepare_data.py)
- **Issue:** `maybe_create_dataset` called `Dataset.create(...)` without `use_current_task=True`. When called inside an already-running ClearML task, this triggered an internal `Task.init()` which raised `UsageError: Current task already created and requested project name does not match current project name`.
- **Fix:** Added `use_current_task=True` to `Dataset.create(...)` in `src/clearml_utils.py`. Updated `tests/test_clearml_utils.py` assertion to match.
- **Files modified:** src/clearml_utils.py, tests/test_clearml_utils.py
- **Verification:** Pipeline exits 0 in CLEARML_OFFLINE_MODE; all 26 tests pass.
- **Committed in:** 3a91691 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 bug)
**Impact on plan:** Essential fix — without it the pipeline always crashes when ClearML is available.

## Issues Encountered

- `ty` type checker flagged `Path(page_path_str)` as invalid because pdf2image stubs don't model the `paths_only=True` overload. Suppressed with `ty: ignore[invalid-argument-type]` and inline comment explaining rationale.

## Known Stubs

None — all columns and data flow are wired end-to-end.

## User Setup Required

None — offline mode works for CI. For production use with ClearML server, run `clearml-init` once.

## Next Phase Readiness

- Phase 2 (review app) can import `MANIFEST_COLUMNS` from `src.manifest_schema` and read `outputs/manifest.csv`
- Phase 3 (training) can consume the crops directory and manifest for dataset loading
- Awaiting human verification checkpoint (Task 3) — user must run pipeline on real PDF and confirm outputs

---
*Phase: 01-data-pipeline*
*Completed: 2026-04-21*
