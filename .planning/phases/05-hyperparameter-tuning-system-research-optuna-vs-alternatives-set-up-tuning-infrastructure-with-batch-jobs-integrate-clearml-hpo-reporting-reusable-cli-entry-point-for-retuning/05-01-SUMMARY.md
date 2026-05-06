---
phase: 05-hyperparameter-tuning-system
plan: 01
subsystem: model
tags: [optuna, crnn, pytorch, hpo, hyperparameter-tuning]

# Dependency graph
requires:
  - phase: 04-data-augmentation-and-gpu-training-via-clearml-agent
    provides: CRNN model in ctc_utils.py, train_ctc.py with BiLSTM(256, 2) hardcoded

provides:
  - CRNN.__init__ with rnn_hidden=256 and num_layers=2 kwargs (default-arg backward-compatible)
  - optuna==4.8.0 installed and locked in pyproject.toml + uv.lock
  - 5 new tests covering architecture sweep space {128,256,512} x {1,2}

affects: [05-02, 05-03, tune.py, train_ctc.py]

# Tech tracking
tech-stack:
  added: [optuna==4.8.0]
  patterns:
    - CRNN constructor kwargs with default-preserving refactor pattern
    - TDD RED/GREEN for model architecture parameterization

key-files:
  created: []
  modified:
    - src/ctc_utils.py
    - tests/test_ctc_utils.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "optuna==4.8.0 pinned with == (exact) per project convention; no dual-backend flag"
  - "CRNN default args rnn_hidden=256, num_layers=2 preserve byte-identical architecture — existing train_ctc.py call sites need zero changes"
  - "fc layer uses rnn_hidden * 2 (BiLSTM doubles hidden size) — critical for shape correctness"

patterns-established:
  - "Pattern: CRNN parameterization with default-arg backward compatibility — add kwargs with defaults equal to previous hardcoded values"

requirements-completed: [HPO-01, HPO-02]

# Metrics
duration: 3min
completed: 2026-05-06
---

# Phase 5 Plan 01: Parameterize CRNN and Add Optuna Summary

**CRNN.__init__ parameterized with rnn_hidden/num_layers kwargs and optuna==4.8.0 pinned — unblocks HPO tuner plans 02/03**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-06T17:36:27Z
- **Completed:** 2026-05-06T17:39:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- optuna 4.8.0 installed and importable from project venv; locked in pyproject.toml and uv.lock
- CRNN.__init__ now accepts `rnn_hidden: int = 256` and `num_layers: int = 2`; default args preserve identical architecture to pre-plan hardcoded values
- fc layer correctly uses `rnn_hidden * 2` (BiLSTM bidirectional doubling — avoids Pitfall 1 from RESEARCH.md)
- 5 new tests cover {128,256,512} hidden sizes x {1,2} layers and forward-pass correctness

## Task Commits

Each task was committed atomically:

1. **Task 1: Add optuna 4.8.0 to project dependencies** - `e21b55a` (chore)
2. **Task 2 RED: Failing tests for CRNN params** - `5d986d0` (test)
3. **Task 2 GREEN: Implement CRNN parameterization** - `98daf0e` (feat)

## Files Created/Modified

- `src/ctc_utils.py` - CRNN.__init__ updated with rnn_hidden/num_layers kwargs; fc uses rnn_hidden * 2
- `tests/test_ctc_utils.py` - 5 new parameterization tests added; fixed pre-existing unused variable in _make_aug_df
- `pyproject.toml` - optuna==4.8.0 added to [project] dependencies
- `uv.lock` - optuna 4.8.0 locked with full dependency tree

## Decisions Made

- optuna pinned with `==` (exact version) per project CLAUDE.md convention
- Default values `rnn_hidden=256, num_layers=2` chosen to match pre-plan hardcoded values; no existing call sites need updating
- fc layer formula `rnn_hidden * 2` is the only place that depends on rnn_hidden (CNN output 128*8=1024 is fixed)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed pre-existing unused variable in test helper**
- **Found during:** Task 2 (ruff check)
- **Issue:** `page_path` in `_make_aug_df` was assigned but never used (ruff F841); the variable was meant to be passed to `_make_df_row` as page_path
- **Fix:** Renamed `page_path` to `page_img` and passed it to `_make_df_row` so rows have a real page path
- **Files modified:** tests/test_ctc_utils.py
- **Verification:** ruff check exits 0, all 45 tests pass
- **Committed in:** 98daf0e (Task 2 feat commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix)
**Impact on plan:** Pre-existing ruff warning in test helper; fix also makes the test helper more correct (rows now have a real page_path instead of "fake.png"). No scope creep.

## Issues Encountered

- Pre-existing ty error in `AugmentTransform.__call__` (affine_grid expects `list[int]`, gets `Size`). This pre-dates Plan 01 — confirmed via `git stash` check. Zero new type errors introduced by this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02 can now vary `rnn_hidden` and `num_layers` via CRNN constructor kwargs
- Plan 03 can import optuna 4.8.0 for the HPO sweep
- train_ctc.py needs `--rnn_hidden` and `--num_layers` CLI args added (Plan 02 task) before tune.py can dispatch training with different architectures
- All 45 ctc_utils tests pass; no regressions

---
*Phase: 05-hyperparameter-tuning-system*
*Completed: 2026-05-06*
