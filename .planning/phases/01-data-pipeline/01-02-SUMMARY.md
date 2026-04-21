---
phase: 01-data-pipeline
plan: 02
subsystem: infra
tags: [clearml, python, testing, mocking, pytest, ruff, ty]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    plan: 01
    provides: "Project scaffolding, pyproject.toml, src/ and tests/ layout with uv venv"
provides:
  - "src/clearml_utils.py with four typed ClearML helpers: init_task, upload_file_artifact, report_manifest_stats, maybe_create_dataset"
  - "tests/test_clearml_utils.py with 5 pytest tests, all passing with ClearML mocked"
affects:
  - 01-04
  - all pipeline scripts in phase 01 that import clearml_utils

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ClearML helper module pattern: module-level Task/Dataset imports for patchability in tests"
    - "TDD with unittest.mock.patch on src.clearml_utils.Task/Dataset symbols"
    - "init_task must be called before argparse.parse_args() (Pitfall 2 from research)"
    - "Dataset lifecycle: create -> add_files (per folder) -> upload -> finalize (Pitfall 5)"

key-files:
  created:
    - src/clearml_utils.py
    - tests/test_clearml_utils.py
  modified: []

key-decisions:
  - "Module-level `from clearml import Dataset, Task` (not inline imports) enables test patching via `src.clearml_utils.Task`"
  - "tags defaults to empty list [] not None — avoids ClearML API call with None"
  - "Path stringified via str(path) in upload_file_artifact and add_files — ClearML SDK expects str not Path"
  - "int() cast on df['is_flagged'].sum() ensures scalar int not numpy int64 for ClearML scalar logging"

patterns-established:
  - "Pattern: All ClearML SDK imports at module level (not inside functions) for testability"
  - "Pattern: ruff format run before committing test files (formatter adjusts call arg alignment)"

requirements-completed:
  - CLML-03

# Metrics
duration: 2min
completed: 2026-04-21
---

# Phase 1 Plan 02: ClearML Shared Helpers Summary

**Four typed ClearML helpers (init_task, upload_file_artifact, report_manifest_stats, maybe_create_dataset) with full pytest coverage using mocked SDK — no real ClearML server calls**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-21T14:10:25Z
- **Completed:** 2026-04-21T14:12:30Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 2

## Accomplishments

- `src/clearml_utils.py` exports four stable typed functions covering the full ClearML Task/Dataset lifecycle
- `tests/test_clearml_utils.py` has 5 pytest tests verifying contracts with `unittest.mock.patch` on module-level symbols
- ruff (lint + format) and ty type checker both exit clean

## Task Commits

1. **Task 1: Write clearml_utils tests (RED)** - `2c74a69` (test)
2. **Task 2: Implement clearml_utils.py (GREEN)** - `26855a2` (feat)

## Files Created/Modified

- `src/clearml_utils.py` - Four ClearML helpers: init_task, upload_file_artifact, report_manifest_stats, maybe_create_dataset
- `tests/test_clearml_utils.py` - Five pytest tests verifying each helper with mocked ClearML SDK

## Decisions Made

- Module-level `from clearml import Dataset, Task` (not deferred imports) allows `@patch("src.clearml_utils.Task")` to work in tests
- `tags` parameter defaults to `[]` not `None` — passing None to ClearML's `tags=` arg is not tested/specified in the SDK
- Paths stringified with `str(folder)` in `add_files` loop to satisfy ClearML string expectation (verified via test assertions)
- `int(df["is_flagged"].sum())` cast prevents numpy int64 type from being passed to `report_scalar value=`

## Deviations from Plan

None - plan executed exactly as written. The ruff format deviation (test file reformatted) is expected workflow, not a plan deviation.

## Issues Encountered

- ruff format reformatted `tests/test_clearml_utils.py` after initial write (call argument alignment). Fixed before commit — no impact on tests or behavior.

## User Setup Required

None - no external service configuration required. ClearML server calls are mocked in all tests.

## Next Phase Readiness

- `src/clearml_utils` is fully importable and tested; ready for plan 01-04 (prepare_data.py) to use `from src.clearml_utils import ...`
- No blockers

---
*Phase: 01-data-pipeline*
*Completed: 2026-04-21*
