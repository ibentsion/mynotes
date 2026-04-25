---
phase: 02-review-annotation
plan: "02"
subsystem: clearml
tags: [clearml, pandas, manifest, tdd, sync]

# Dependency graph
requires:
  - phase: 01-data-pipeline
    provides: clearml_utils.init_task + upload_file_artifact, manifest_schema.MANIFEST_COLUMNS
provides:
  - sync_review_to_clearml(manifest_path) callable in-process for Plan 03 Streamlit sync button
  - summarize_status_counts(df) returns zero-filled dict for all 5 known statuses
  - CLI: uv run python -m src.review_to_clearml --manifest <path>
  - SYNC-01: per-status scalars logged to ClearML task manual_review_summary
  - SYNC-02: manifest.csv uploaded as artifact named "manifest"
affects: [02-03-review-app]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TDD RED/GREEN cycle: write failing tests first, then implement"
    - "parse_known_args pre-flight check before Task.init to avoid spurious ClearML task on typo"
    - "Module-level imports of init_task/upload_file_artifact required for patch() test patchability"

key-files:
  created:
    - src/review_to_clearml.py
    - tests/test_review_to_clearml.py
  modified: []

key-decisions:
  - "Pre-flight parse_known_args before Task.init: catches missing manifest path cheaply without creating a spurious ClearML task"
  - "KNOWN_STATUSES tuple ensures zero-filled output dict is consistent run-to-run for dashboard axis stability"
  - "sync_review_to_clearml returns dict[str,int] so Streamlit caller can render count confirmation in-app"

patterns-established:
  - "summarize_status_counts: always return all 5 known statuses, zero-filled for absent ones"
  - "sync_review_to_clearml callable in-process — no subprocess needed from Streamlit"

requirements-completed: [SYNC-01, SYNC-02]

# Metrics
duration: 4min
completed: 2026-04-25
---

# Phase 02 Plan 02: review_to_clearml Summary

**Standalone CLI + reusable function that uploads manifest.csv + per-status scalars to ClearML task manual_review_summary (SYNC-01 + SYNC-02)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-25T19:10:57Z
- **Completed:** 2026-04-25T19:14:16Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 2

## Accomplishments

- `summarize_status_counts(df)` returns zero-filled dict for all 5 known statuses (unlabeled/labeled/skip/bad_seg/merge_needed)
- `sync_review_to_clearml(manifest_path)` validates schema, inits ClearML task, uploads artifact, reports scalars — callable in-process by Plan 03
- CLI `uv run python -m src.review_to_clearml --manifest <path>` with pre-flight path validation and non-zero exit on errors
- All 6 tests pass; full 32-test suite still green; ruff + ty both clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests (RED)** - `7323cbb` (test)
2. **Task 2: Implement review_to_clearml.py (GREEN)** - `83284be` (feat)

**Plan metadata:** (final commit hash below)

_Note: TDD tasks — test commit then feat commit_

## Files Created/Modified

- `src/review_to_clearml.py` - CLI + sync_review_to_clearml + summarize_status_counts
- `tests/test_review_to_clearml.py` - 6 unit/integration tests with mocked ClearML

## Decisions Made

- Pre-flight `parse_known_args` before `Task.init` catches typos in `--manifest` without spawning an empty ClearML task. ClearML still auto-logs args because the final `parse_args()` is called after `init_task`.
- `sync_review_to_clearml` returns `dict[str, int]` so the Streamlit sync button (Plan 03) can display count summary in-app without re-reading the CSV.
- `KNOWN_STATUSES` is a module-level constant so the dashboard always sees the same 5 axis labels even when some statuses have zero rows.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed import sort order in test file**
- **Found during:** Task 2 (GREEN verification)
- **Issue:** ruff I001 — `import sys` before `import subprocess` violated isort ordering
- **Fix:** `uv run ruff check tests/test_review_to_clearml.py --fix` reordered to `import subprocess` first
- **Files modified:** tests/test_review_to_clearml.py
- **Verification:** `uv run ruff check src/ tests/` exits 0
- **Committed in:** 83284be (included in Task 2 commit)

---

**Total deviations:** 1 auto-fixed (import sort, Rule 1)
**Impact on plan:** Trivial style fix; no scope creep.

## Issues Encountered

None — implementation matched plan spec exactly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `sync_review_to_clearml(manifest_path)` is ready to import in Plan 03 Streamlit review app for the sidebar sync button
- `summarize_status_counts` exposed for any future dashboard needs
- SYNC-01 and SYNC-02 requirements closed
