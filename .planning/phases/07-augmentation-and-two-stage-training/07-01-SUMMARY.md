---
phase: 07-augmentation-and-two-stage-training
plan: "01"
subsystem: testing
tags: [pytest, mocking, ctc_utils, build_charset]

requires:
  - phase: 04-data-augmentation-and-gpu-training-via-clearml-agent
    provides: ctc_utils.py with build_charset(labels, extra_words=None) signature

provides:
  - Green test baseline for Phase 7 feature work (all test_train_ctc.py tests passing)

affects:
  - 07-augmentation-and-two-stage-training plans 02+

tech-stack:
  added: []
  patterns:
    - "Mock signatures must match actual function signature including optional kwargs"

key-files:
  created: []
  modified:
    - tests/test_train_ctc.py

key-decisions:
  - "No code changes needed — only test mock signatures updated to match build_charset(labels, extra_words=None)"

patterns-established:
  - "Pattern: when patching with side_effect, mock function must accept all kwargs the real function accepts"

requirements-completed: []

duration: 5min
completed: 2026-05-29
---

# Phase 7 Plan 01: Fix test mock signatures for green baseline

**Two fake_build_charset mocks updated to accept extra_words=None, matching build_charset signature and unblocking all 31 test_train_ctc tests**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-29T00:00:00Z
- **Completed:** 2026-05-29T00:05:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Fixed `fake_build_charset(labels)` in `test_status_filter_keeps_only_labeled` to accept `extra_words=None`
- Fixed `capture_build_charset(labels)` in `test_charset_build_receives_labeled_labels` to accept `extra_words=None`
- All 31 tests in `tests/test_train_ctc.py` pass; 48 in `tests/test_ctc_utils.py` pass

## Task Commits

1. **Task 1: Fix fake_build_charset mock signatures** - `b92ffbb` (fix)

## Files Created/Modified
- `tests/test_train_ctc.py` - Updated two local mock function signatures to include `extra_words=None`

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
The plan's `verify` command uses `cd /home/ido/git/mynotes && uv run pytest ...` which runs against
the main repo, not the worktree. Verification was run from within the worktree directory instead
and confirmed both tests pass. This is expected behavior in worktree execution mode.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Green baseline established: all test_train_ctc.py and test_ctc_utils.py tests pass
- Phase 7 Plan 02 (elastic augmentation) can now begin without pre-existing test failures masking new failures

---
*Phase: 07-augmentation-and-two-stage-training*
*Completed: 2026-05-29*
