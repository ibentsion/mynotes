---
phase: 07-augmentation-and-two-stage-training
plan: "04"
subsystem: training
tags: [pytorch, crnn, ctc, two-stage-training, pretrain, optuna]

# Dependency graph
requires:
  - phase: 07-03
    provides: elastic augmentation wired into AugmentTransform; clearml_utils.get_dataset_root
  - phase: 07-02
    provides: elastic_alpha/elastic_sigma CLI flags in train_ctc
  - phase: 07-01
    provides: get_dataset_root helper with flat/prefixed symlink fix
provides:
  - "_run_pretrain(): standalone synthetic pre-training invocation that saves checkpoint_pretrain.pt"
  - "_run_loop(): extracted epoch loop with series_prefix and on_epoch_end params"
  - "Two-call workflow: pretrain on synthetic CSV → fine-tune with --pretrain_checkpoint_path"
  - "tune.py passes pretrain_checkpoint_path to every HPO trial"
affects: [phase-08, train_ctc, tune, hpo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_report_epoch_debug helper pattern: extract ClearML debug logging out of epoch loop"
    - "series_prefix kwarg on _run_loop distinguishes pretrain/ vs default metric series names"
    - "Two-call CLI interface: separate pretrain invocation writes checkpoint; fine-tune reads it"

key-files:
  created: []
  modified:
    - src/train_ctc.py
    - src/tune.py
    - tests/test_train_ctc.py

key-decisions:
  - "Pre-training uses random val split (no page structure in synthetic data); fine-tuning uses build_half_page_units"
  - "on_epoch_end (Optuna pruning callback) only called during fine-tuning; pretrain passes None"
  - "torch.load with weights_only=True enforced for checkpoint loading (security T-07-04-01)"
  - "_report_epoch_debug extracted from _run_loop to stay within 100-line CLAUDE.md limit"
  - "HPO does not tune pretrain hyperparams; tune.py forwards pretrain_checkpoint_path as warm-start only"

patterns-established:
  - "series_prefix pattern: prefix='pretrain/' for pre-training metrics, '' for fine-tune"
  - "Deferred imports (noqa: PLC0415) for torch/ctc_utils inside function bodies"

requirements-completed: [TRAIN-01, TRAIN-02]

# Metrics
duration: 30min
completed: 2026-05-29
---

# Phase 7 Plan 04: Two-Stage Training Interface Summary

**Extracted _run_loop/_run_pretrain/_report_epoch_debug from run_training; wired pretrain_checkpoint_path into tune.py for warm-start HPO**

## Performance

- **Duration:** ~30 min (continuation from prior session)
- **Started:** 2026-05-29T13:15:53Z
- **Completed:** 2026-05-29T13:34:06Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `run_training()` refactored from 275-line monolith into 5 helpers, each under 100 lines
- Two-call CLI interface: `--pretrain_manifest` triggers pretrain-only mode, saves `checkpoint_pretrain.pt`; `--pretrain_checkpoint_path` loads weights before fine-tuning
- Pre-training metrics logged with `series_prefix="pretrain/"` so ClearML shows distinct pretrain/finetune series
- `tune.py` accepts `--pretrain_checkpoint_path` and forwards it to every HPO trial Namespace
- 5 new tests covering pretrain mode, no-finetune guard, checkpoint loading, and on_epoch_end backward compat

## Task Commits

Each task was committed atomically:

1. **Task 1: Add pretrain CLI flags and extract _run_loop/_run_pretrain** - `e393deb` (feat)
2. **Task 1 (continued): Extract _report_epoch_debug, fix ruff E501/I001** - `d9a3eca` (refactor)
3. **Task 2 RED: Add 5 pretrain/finetune tests** - `8f5702b` (test)
4. **Task 2 GREEN: Forward pretrain_checkpoint_path in tune.py** - `64bdcb0` (feat)

_Note: master merge (2dc2a07) brought in Plans 07-01 through 07-03 (elastic augmentation, get_dataset_root)_

## Files Created/Modified
- `src/train_ctc.py` - Extracted `_eval_val_epoch`, `_report_epoch_debug`, `_run_loop`, `_run_pretrain`, `_setup_finetune_loaders`; added 4 pretrain CLI flags
- `src/tune.py` - Added `--pretrain_checkpoint_path` flag; populated 4 pretrain fields in `_objective` Namespace
- `tests/test_train_ctc.py` - 5 new tests for two-stage training interface

## Decisions Made
- Pre-training uses random val split rather than `build_half_page_units` because synthetic data has no page structure
- `on_epoch_end` (Optuna MedianPruner callback) is only passed during fine-tuning; pretrain uses `None`
- `torch.load(..., weights_only=True)` enforced per security constraint T-07-04-01 (prevents pickle RCE)
- `_report_epoch_debug` extracted as a separate helper to keep `_run_loop` under 100 lines (CLAUDE.md hard limit)
- HPO sweep does not tune pretrain hyperparameters; `--pretrain_checkpoint_path` is passed as a warm-start only

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Extracted _report_epoch_debug to enforce 100-line function limit**
- **Found during:** Task 1 post-commit verification
- **Issue:** `_run_loop` was 106 lines, exceeding CLAUDE.md ≤100 lines/function hard limit
- **Fix:** Extracted debug sample logging and saliency panel into `_report_epoch_debug()`
- **Files modified:** src/train_ctc.py
- **Verification:** AST line count check — all functions 87 lines or fewer
- **Committed in:** d9a3eca (refactor commit)

**2. [Rule 1 - Bug] Fixed ruff E501 (27 line-length violations) and I001 (import ordering)**
- **Found during:** Task 1 ruff check
- **Issue:** Long `logger.report_scalar` calls exceeded 100-char limit; `src.ctc_utils` imported before `torch.utils.data`
- **Fix:** Added `sp = series_prefix` shorthand; wrapped multi-arg calls; fixed import ordering with ruff --fix
- **Files modified:** src/train_ctc.py
- **Verification:** `uv run ruff check src/train_ctc.py src/tune.py` — All checks passed
- **Committed in:** d9a3eca (refactor commit)

---

**Total deviations:** 2 auto-fixed (Rule 1 bugs)
**Impact on plan:** Both necessary for CLAUDE.md compliance (line limit, zero warnings policy). No scope creep.

## Issues Encountered
- TDD ordering: Task 1 (implementation) was completed before Task 2 (tests) per plan structure. The RED phase tests passed immediately since the implementation already existed. Documented as expected — the plan's task order drives implementation-first; RED/GREEN is nominal.
- Merge required: This worktree predated Plans 07-01 through 07-03. A master merge was needed to get `elastic_alpha/sigma` params and `get_dataset_root` before tests would be correct.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Two-stage training interface complete and tested
- Pre-train on synthetic data: `python -m src.train_ctc --pretrain_manifest data/synthetic_manifest.csv --pretrain_epochs 30 --output_dir outputs/pretrain`
- Fine-tune with warm start: `python -m src.train_ctc --manifest data/manifest.csv --pretrain_checkpoint_path outputs/pretrain/checkpoint_pretrain.pt`
- HPO with warm start: `python -m src.tune --pretrain_checkpoint_path outputs/pretrain/checkpoint_pretrain.pt`

## Self-Check: PASSED

All 4 files found. All 4 task commits verified in git log.

---
*Phase: 07-augmentation-and-two-stage-training*
*Completed: 2026-05-29*
