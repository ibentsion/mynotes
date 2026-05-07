---
phase: 05-hyperparameter-tuning-system
plan: 03
subsystem: training
tags: [optuna, clearml, hpo, hyperparameter-tuning, crnn, ctc, argparse]

# Dependency graph
requires:
  - phase: 05-01
    provides: CRNN with rnn_hidden/num_layers params exposed, optuna==4.8.0 installed
  - phase: 05-02
    provides: run_training(args, on_epoch_end) helper + _build_parser() exposed from train_ctc
provides:
  - src/tune.py — Optuna HPO sweep CLI with MedianPruner, per-trial ClearML tasks, best_params.json output
  - tune-hpo console script entry point in pyproject.toml
  - tests/test_tune.py — 12 tests covering search space, pruning, tag separation, enqueue flow
  - outputs/best_params.json gitignore explicit comment
affects:
  - any future phase that retunes or consumes best_params.json
  - train_ctc.py --params round-trip (already established in plan 02)

# Tech tracking
tech-stack:
  added: []  # optuna 4.8.0 was added in plan 01; no new dependencies in this plan
  patterns:
    - Optuna study with direction="minimize" (CER lower is better) + MedianPruner for pruning
    - Per-trial ClearML task created inside objective(), closed in finally (Pitfall 3 pattern)
    - Orchestrator ClearML task "hpo_sweep" tagged ["phase-5"] only — NOT "hpo-trial"
    - execute_remotely() on orchestrator task dispatches entire sweep to GPU agent (Open Question 1 resolution)
    - _make_pruning_callback() extracts callback closure to keep _objective under 100-line limit
    - call_order tracker list for cross-mock ordering assertions in tests

key-files:
  created:
    - src/tune.py
    - tests/test_tune.py
  modified:
    - .gitignore
    - pyproject.toml

key-decisions:
  - "tune.py splits _objective into _init_trial_task + _make_pruning_callback helpers to keep all functions under 100-line limit"
  - "_make_pruning_callback returns tuple[list[float], Callable[[int,float],None]] — typed correctly for ty compliance"
  - "enqueue ordering test uses a shared call_order list instead of cross-mock index comparison (cross-mock ordering is undefined)"
  - "test type annotations use typing.cast instead of float()/int() for dict[str, object] value comparisons"

patterns-established:
  - "Pattern: call_order list for ordering verification across multiple mocks"
  - "Pattern: typing.cast for params dict[str,object] values in tests instead of type: ignore"

requirements-completed: [HPO-06, HPO-07, HPO-08, HPO-09, HPO-10, HPO-11, HPO-12]

# Metrics
duration: 24min
completed: 2026-05-07
---

# Phase 05 Plan 03: tune.py HPO Sweep CLI Summary

**Optuna 4.8.0 sweep CLI (src/tune.py) with MedianPruner, per-trial ClearML tasks tagged "hpo-trial", orchestrator "hpo_sweep" task, and outputs/best_params.json round-trip**

## Performance

- **Duration:** 24 min
- **Started:** 2026-05-07T09:49:49Z
- **Completed:** 2026-05-07T10:14:00Z
- **Tasks:** 2
- **Files modified:** 4 (created src/tune.py, tests/test_tune.py; modified .gitignore, pyproject.toml)

## Accomplishments

- Implemented `src/tune.py`: full Optuna HPO sweep CLI with MedianPruner (n_startup_trials=5, n_warmup_steps=5), per-trial ClearML task creation, pruning callback via try/finally (Pitfall 3), and outputs/best_params.json serialization
- `--enqueue` dispatches the entire orchestrator task to the GPU agent before `study.optimize()` runs, so Optuna pruning works normally on the GPU machine (Open Question 1 resolution; avoids Pitfall 7)
- Orchestrator ClearML task "hpo_sweep" logs CER-per-trial scalar and "HPO Results" table with all trial params
- 12 tests in `tests/test_tune.py` covering all behavioral contracts: parser defaults, PARAM_KEYS order, search space bounds, best_params.json output, pruning callback, task-close-on-failure, tag separation, enqueue ordering, missing manifest guard, gitignore membership, smoke imports
- `tune-hpo` console script registered in pyproject.toml; `.gitignore` updated with explicit D-11 comment

## Task Commits

1. **Task 1: Implement src/tune.py CLI** - `4db0a27` (feat)
2. **Task 2: Tests for tune.py** - `1c809c0` (test)

## Files Created/Modified

- `src/tune.py` — Optuna HPO sweep CLI: _build_parser, _suggest_params, _objective, _init_trial_task, _make_pruning_callback, _report_hpo_results, _write_best_params, main; 206 lines
- `tests/test_tune.py` — 12 tests covering all behavioral contracts; 301 lines
- `.gitignore` — Added explicit comment for outputs/best_params.json per D-11
- `pyproject.toml` — Added `tune-hpo = "src.tune:main"` console script entry

## Decisions Made

- Extracted `_init_trial_task` and `_make_pruning_callback` as standalone helpers (in addition to `_objective`) to keep all functions under the 100-line CLAUDE.md limit
- Return type of `_make_pruning_callback` typed as `tuple[list[float], Callable[[int, float], None]]` (importing `Callable` from `collections.abc`) to satisfy ty type checker
- Test for enqueue ordering uses a `call_order: list[str]` accumulator with side_effect callbacks — cross-mock index comparison doesn't work because each mock has independent call numbering
- Test for params bounds uses `typing.cast` instead of `float(object)` to satisfy ty (which correctly rejects `float(object)`)
- The `_build_trial_args` standalone helper shown in the plan snippet had a scoping bug (referenced `trial.number` outside closure) — correctly avoided per plan's explicit instruction; inlined inside `_objective` instead

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed _build_trial_args standalone helper with scoping bug**
- **Found during:** Task 1 (tune.py implementation)
- **Issue:** The plan's code skeleton included `_build_trial_args` which referenced `trial.number` outside the trial closure — plan itself flagged this as an intentional bug to avoid
- **Fix:** Inlined trial args construction inside `_objective` as the plan instructed; did not create `_build_trial_args`
- **Files modified:** src/tune.py
- **Committed in:** 4db0a27

**2. [Rule 1 - Bug] Fixed test ordering assertion using cross-mock indices**
- **Found during:** Task 2 (tests implementation)
- **Issue:** Initial `test_enqueue_calls_execute_remotely_before_optimize` used `orch_task.mock_calls` index vs `study.mock_calls` index — these are independent sequences so comparison was meaningless (exec_idx=1 < optimize_idx=0 was False)
- **Fix:** Replaced with `call_order: list[str]` accumulator using `side_effect` callbacks on both `execute_remotely` and `study.optimize`; asserted ordering via list indices
- **Files modified:** tests/test_tune.py
- **Committed in:** 1c809c0

---

**Total deviations:** 2 auto-fixed (both Rule 1 bugs)
**Impact on plan:** Both fixes were directly caused by issues in the plan's code sketches (one flagged explicitly, one a test design flaw). No scope creep.

## Issues Encountered

- `ty check` rejected `float(params["lr"])` because `params` is `dict[str, object]` and `float.__new__` doesn't accept `object` — resolved with `typing.cast(float, params["lr"])` pattern
- `ruff` split `from src.train_ctc import _build_parser as _build_train_parser, run_training` into two import lines (isort rule) — applied fix automatically
- Full test suite runs slowly due to actual CRNN training in test_train_ctc.py; 5 tests showed intermittent ClearML singleton collision failures when run concurrently — these are pre-existing flakiness issues, not caused by this plan

## User Setup Required

None - no external service configuration required. `tune-hpo --help` works after `uv sync`.

## Next Phase Readiness

- Phase 5 is now complete: tune.py → outputs/best_params.json → train_ctc.py --params round-trip is fully operational
- `tune-hpo --manifest data/manifest.csv --n_trials 20` can be run whenever dataset grows to retune
- For GPU execution: `tune-hpo --manifest data/manifest.csv --n_trials 20 --enqueue` dispatches entire sweep to ClearML GPU agent

## Known Stubs

None — all outputs are wired. best_params.json is intentionally gitignored (D-11); it's a real artifact produced by running tune.py, not a placeholder.

---
*Phase: 05-hyperparameter-tuning-system*
*Completed: 2026-05-07*
