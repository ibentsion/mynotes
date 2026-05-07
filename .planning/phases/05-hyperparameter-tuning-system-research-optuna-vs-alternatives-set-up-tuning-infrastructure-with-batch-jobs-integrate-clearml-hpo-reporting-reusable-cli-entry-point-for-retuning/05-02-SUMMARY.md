---
phase: 05-hyperparameter-tuning-system
plan: "02"
subsystem: training
tags: [pytorch, clearml, argparse, crnn, ctc, hpo, optuna]

# Dependency graph
requires:
  - phase: 05-01
    provides: CRNN(num_classes, rnn_hidden=256, num_layers=2) parameterized constructor
provides:
  - train_ctc.py CLI flags --rnn_hidden, --num_layers, --params
  - _apply_params_file() JSON round-trip loader (D-10)
  - run_training(args, on_epoch_end=None) -> float helper for in-process Optuna pruning
  - exit code 6 for missing --params file
affects:
  - 05-03  # tune.py consumes run_training() signature; plan 03 imports it in-process

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "run_training() as public helper: caller (main or tune.py) owns ClearML task lifecycle; helper uses Task.current_task()"
    - "on_epoch_end callback propagates exceptions — Optuna pruning relies on TrialPruned escaping the loop"
    - "_apply_params_file mutates args BEFORE task.connect() so JSON-loaded values are tracked in ClearML"
    - "Deferred imports inside run_training (torch/ctc_utils) preserved for execute_remotely agent safety"
    - "return 0 after execute_remotely in main() to handle mocked-task test isolation"

key-files:
  created: []
  modified:
    - src/train_ctc.py
    - tests/test_train_ctc.py

key-decisions:
  - "run_training uses Task.current_task() rather than accepting task as arg — keeps tune.py caller simple and matches ClearML idiom"
  - "on_epoch_end placed AFTER checkpoint save so pruned trials still save their best checkpoint"
  - "return 0 added after execute_remotely to prevent run_training being called with a mocked task object in tests"
  - "ValueError raised (not sys.exit) from run_training for empty split — enables main() and tune.py to handle differently"

patterns-established:
  - "Phase-5 HPO entry point pattern: _apply_params_file -> task.connect -> run_training(on_epoch_end=cb)"

requirements-completed:
  - HPO-03
  - HPO-04
  - HPO-05

# Metrics
duration: 45min
completed: 2026-05-07
---

# Phase 5 Plan 02: CLI Flags + run_training() Helper Summary

**train_ctc.py extended with --rnn_hidden, --num_layers, --params flags plus a public run_training(args, on_epoch_end) helper that Optuna tune.py calls in-process for pruning**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-07T05:30:00Z
- **Completed:** 2026-05-07T06:07:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `--rnn_hidden {128,256,512}`, `--num_layers {1,2}`, `--params PATH` CLI flags to train_ctc.py
- Added `_apply_params_file()` that loads best_params.json and casts values to existing arg types before task.connect() (RESEARCH.md Pitfall 4 + D-10)
- Extracted `run_training(args, on_epoch_end=None) -> float` public helper containing the training loop; main() reduced to argparse + ClearML setup + early-exit guards
- Added exit code 6 for missing --params file; preserved exit codes 2-5
- Added "phase-5" tag to ClearML task when --params is used
- 13 new tests: 8 covering parser flags and JSON round-trip (Task 1), 5 covering run_training contract (Task 2)

## Task Commits

Each task was committed atomically (TDD pattern: test RED → feat GREEN):

1. **Task 1 RED: failing tests for --rnn_hidden, --num_layers, --params** - `afdfff8` (test)
2. **Task 1 GREEN: add flags + _apply_params_file + CRNN wiring** - `48357a1` (feat)
3. **Task 2 RED: failing tests for run_training() helper** - `7d69b07` (test)
4. **Task 2 GREEN: extract run_training() with on_epoch_end callback** - `b0b6f67` (feat)

(05-01 prerequisite commits cherry-picked: `2b6a1ef`, `afb7166`, `64c5e6d`)

## Files Created/Modified
- `src/train_ctc.py` - New flags in _build_parser(), _apply_params_file() helper, run_training() function, slimmed main()
- `tests/test_train_ctc.py` - 13 new tests for Task 1 parser flags/JSON and Task 2 callback contract

## Decisions Made
- `run_training` uses `Task.current_task()` rather than accepting `task` as arg — keeps tune.py caller simple and matches ClearML idiom
- `on_epoch_end` placed AFTER checkpoint save so pruned trials still save their best checkpoint from previous epochs
- `return 0` added after `execute_remotely` in `main()` to prevent `run_training` being called when a mocked task is in play (test isolation)
- `ValueError` raised (not `sys.exit`) from `run_training` for empty split — allows main() to map to exit code 5, and tune.py to handle differently if needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added `return 0` after `execute_remotely` in main()**
- **Found during:** Task 2 (run_training helper extraction)
- **Issue:** When `init_task` is mocked in tests with `--enqueue`, `execute_remotely()` doesn't terminate the process; `run_training()` was then called and `Task.current_task()` returned `None`, causing AttributeError
- **Fix:** Added `return 0` immediately after `task.execute_remotely(...)` call in main()
- **Files modified:** src/train_ctc.py
- **Verification:** `test_enqueue_calls_execute_remotely_after_connect` and `test_enqueue_uses_gpu_tag` pass
- **Committed in:** b0b6f67 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test isolation)
**Impact on plan:** Minimal change, necessary for test correctness and to prevent calling run_training with no active ClearML task.

## Known Pre-existing Test Failures

The following 6 tests were failing BEFORE this plan (established by running the test suite against the cherry-picked baseline before Task 2 implementation):

- `test_build_parser_aug_defaults` — expects `aug_copies==0` but default was changed to `4` in commit `3c24b03`
- `test_status_filter_keeps_only_labeled` — patches `src.train_ctc.build_charset` which is a deferred import (inside main/run_training), not module-level
- `test_charset_build_receives_labeled_labels` — same deferred-import patch issue
- `test_no_page_leakage_between_train_and_val` — patches `src.train_ctc.split_units`, same issue
- `test_val_dataset_has_no_augment` — patches `src.train_ctc.CropDataset`, same issue
- `test_dataset_id_calls_remap` — patches `src.train_ctc.split_units` inside the `with patch(...)` block, same issue

Root cause: Quick fix `d9f5d7d` moved ctc_utils imports inside `main()` for agent safety (execute_remotely workaround), but the tests were written when imports were at module level. These tests need updating in a future plan to patch `src.ctc_utils.X` instead of `src.train_ctc.X`.

## Issues Encountered
- The 05-01 dependency (CRNN parameterization + optuna install) was completed by a parallel agent in branch `worktree-agent-aa56cbf3e847deee0`. Cherry-picked the relevant commits to bring this worktree up to date before implementing Task 2.

## Next Phase Readiness
- `run_training(args, on_epoch_end=cb)` is the contract that plan 05-03's `tune.py::objective` will consume in-process for Optuna pruning
- All 3 requirements for this plan are complete: HPO-03 (CLI flags), HPO-04 (--params round-trip), HPO-05 (run_training helper)
- plan 05-03 can now implement the Optuna tuner using the `run_training` import

---
*Phase: 05-hyperparameter-tuning-system*
*Completed: 2026-05-07*
