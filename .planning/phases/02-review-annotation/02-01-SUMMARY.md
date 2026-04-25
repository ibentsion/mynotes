---
phase: 02-review-annotation
plan: "01"
subsystem: review-app
tags: [streamlit, review, navigation, session-state, tdd]
dependency_graph:
  requires: [01-04-SUMMARY]
  provides: [src/review_app.py, src/review_state.py]
  affects: [02-02, 02-03]
tech_stack:
  added: [streamlit==1.56.0]
  patterns: [Streamlit session_state for navigation, filesystem-backed JSON state, TDD red-green cycle]
key_files:
  created:
    - src/review_app.py
    - src/review_state.py
    - tests/test_review_state.py
  modified:
    - pyproject.toml
    - uv.lock
decisions:
  - "streamlit pinned at 1.56.0 (latest stable; plan suggested 1.40.0 which is outdated)"
  - "D-02: review_queue.csv row order honored via left-merge; falls back to manifest order"
  - "D-03: .review_state.json persisted on every Prev/Next and filter change; resets index on filter switch"
  - "D-06: Crop N of M position indicator rendered via st.caption above the crop image"
metrics:
  duration_minutes: 5
  completed_date: "2026-04-25"
  tasks_completed: 2
  files_changed: 5
---

# Phase 2 Plan 1: Streamlit Review App Skeleton Summary

Streamlit review app skeleton with manifest load, schema validation, sidebar filter selector (unlabeled/flagged/labeled/all), Prev/Next queue navigation following review_queue.csv order, crop image display with metadata expander, and filesystem-backed session position persistence via .review_state.json.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add streamlit dep + review_state helper + unit tests | 3b958e3 | pyproject.toml, uv.lock, src/review_state.py, tests/test_review_state.py |
| 2 | Implement review_app.py skeleton | 25149c3 | src/review_app.py |

## Decisions Made

1. **streamlit==1.56.0** — Plan suggested 1.40.0 but that is three minor versions behind the current stable at execution time. Bumped to latest stable.

2. **review_queue.csv merge strategy** — `_resolve_queue` does a left-merge of queue crop_path order onto the manifest DataFrame, then drops rows with no matching manifest entry. This ensures queue ordering is honored and gracefully handles out-of-sync queue/manifest rows.

3. **with_filter helper** — Added as a pure function on ReviewState to make filter changes atomic (filter + index reset) and testable without Streamlit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused `Path` import from tests/test_review_state.py**
- **Found during:** Task 1 - ruff check
- **Issue:** Plan template included `from pathlib import Path` in test file, but none of the test functions use `Path` directly (they use `tmp_path` from pytest fixtures)
- **Fix:** Removed the unused import to satisfy ruff F401
- **Files modified:** tests/test_review_state.py
- **Commit:** 3b958e3

**2. [Rule 1 - Deviation] Bumped streamlit version from 1.40.0 to 1.56.0**
- **Found during:** Task 1 - Step A (pre-install version check)
- **Issue:** Plan specified streamlit==1.40.0 but PyPI shows 1.56.0 as the latest stable; pinning an outdated version is undesirable
- **Fix:** Pinned 1.56.0 per plan instruction ("bump to the latest stable and record the chosen version")
- **Files modified:** pyproject.toml, uv.lock
- **Commit:** 3b958e3

### Pre-existing Issue (out of scope)

`tests/test_prepare_data.py::test_prepare_data_end_to_end_on_synthetic_pdf` times out at 120s due to ClearML dataset finalization hanging in offline mode. This failure exists on the branch before this plan and is not caused by any changes here. Logged to deferred-items per deviation scope rules.

## Verification Results

- `uv run pytest tests/test_review_state.py -v` — 8 passed
- `uv run pytest --ignore=tests/test_prepare_data.py` — 32 passed (26 pre-existing + 8 new)
- `uv run ruff check src/ tests/` — exit 0
- `uv run ty check src/review_state.py src/review_app.py tests/test_review_state.py` — exit 0
- `uv run python -c "import src.review_app; import src.review_state"` — no errors
- streamlit 1.56.0 importable

## Known Stubs

None. The skeleton is complete for navigation. Edit fields (transcription, status, notes, autosave) and ClearML sync are intentionally deferred to Plan 02-03 as documented in the plan objective.

## Self-Check: PASSED

- src/review_app.py: FOUND
- src/review_state.py: FOUND
- tests/test_review_state.py: FOUND
- 02-01-SUMMARY.md: FOUND
- Commit 3b958e3: FOUND
- Commit 25149c3: FOUND
- Commit 9fd499f: FOUND
