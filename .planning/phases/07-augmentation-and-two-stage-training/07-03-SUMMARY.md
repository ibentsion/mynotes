---
phase: 07-augmentation-and-two-stage-training
plan: "03"
subsystem: training-cli
tags: [elastic-deformation, train_ctc, tune, cli-flags, tdd]

requires:
  - phase: 07-augmentation-and-two-stage-training
    plan: "02"
    provides: AugmentTransform with elastic_alpha/elastic_sigma params in ctc_utils.py

provides:
  - --elastic_alpha and --elastic_sigma CLI flags in train_ctc._build_parser()
  - AugmentTransform construction in run_training() passes elastic params from args
  - tune.py _objective Namespace includes elastic_alpha=0.0 and elastic_sigma=5.0
  - 3 new elastic CLI flag tests in tests/test_train_ctc.py

affects:
  - 07-augmentation-and-two-stage-training plan 04 (train_ctc now has elastic params)

tech-stack:
  added: []
  patterns:
    - "Deferred imports in run_training() mean patch target is src.ctc_utils.AugmentTransform, not src.train_ctc.AugmentTransform"

key-files:
  created: []
  modified:
    - src/train_ctc.py
    - src/tune.py
    - tests/test_train_ctc.py

key-decisions:
  - "Patch target for AugmentTransform mock is src.ctc_utils.AugmentTransform — AugmentTransform is imported via deferred import inside run_training(), not at module level in train_ctc.py; plan suggested src.train_ctc.AugmentTransform which does not exist as a module attribute"
  - "Merged master into worktree before executing — worktree was branched before Plan 07-02 commits landed; fast-forward merge brought in albumentations dep and elastic AugmentTransform changes needed for this plan"

patterns-established:
  - "Pattern: when a function uses deferred imports (noqa: PLC0415), mock the symbol at its origin module (src.ctc_utils.X), not the caller module (src.train_ctc.X)"

requirements-completed:
  - AUG-02

duration: ~10min
completed: 2026-05-29
---

# Phase 7 Plan 03: Wire --elastic_alpha/--elastic_sigma CLI flags

**--elastic_alpha and --elastic_sigma wired into train_ctc._build_parser() and AugmentTransform construction; tune.py _objective Namespace updated with elastic safe defaults; 3 new tests all pass**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-29T09:55:00Z
- **Completed:** 2026-05-29T10:03:27Z
- **Tasks:** 2 (TDD: 1 RED commit + 1 GREEN commit covering both tasks)
- **Files modified:** 3

## Accomplishments

- Added `--elastic_alpha` (type=float, default=0.0) and `--elastic_sigma` (type=float, default=5.0) to `_build_parser()` with descriptive help text
- Extended `AugmentTransform(...)` constructor call in `run_training()` to pass `elastic_alpha=args.elastic_alpha` and `elastic_sigma=args.elastic_sigma`; remains inside the `aug_copies > 0` guard (D-03)
- Added `elastic_alpha=0.0` and `elastic_sigma=5.0` to `tune.py _objective` Namespace — prevents AttributeError in HPO trials (RESEARCH.md Pitfall 3)
- All 48 tests in `test_train_ctc.py` + `test_tune.py` pass; ruff check exits 0

## Task Commits

1. **RED - Failing elastic tests** - `94cc035` (test)
2. **GREEN - Elastic flags implementation** - `1070bf8` (feat)

## Files Created/Modified

- `src/train_ctc.py` — added `--elastic_alpha` and `--elastic_sigma` flags to `_build_parser()`; added `elastic_alpha=args.elastic_alpha, elastic_sigma=args.elastic_sigma` to `AugmentTransform` constructor
- `src/tune.py` — added `elastic_alpha=0.0, elastic_sigma=5.0` to `_objective` Namespace
- `tests/test_train_ctc.py` — added 3 tests: `test_build_parser_elastic_defaults`, `test_elastic_alpha_nonzero_wires_into_augment_transform`, `test_tune_objective_namespace_has_elastic_attrs`

## Decisions Made

- **Patch target src.ctc_utils.AugmentTransform**: Plan spec said "Patches src.ctc_utils.AugmentTransform to spy on constructor kwargs" but also showed `@patch("src.train_ctc.AugmentTransform")`. Because `train_ctc.run_training()` uses a deferred import (`from src.ctc_utils import AugmentTransform  # noqa: PLC0415` inside the function body), `AugmentTransform` is never a module-level attribute of `src.train_ctc`. Patching the origin `src.ctc_utils.AugmentTransform` works correctly.
- **Merged master before execution**: This worktree branched off before the Plan 07-02 commits. Fast-forward merge from master brought in albumentations, elastic AugmentTransform, and planning artifacts without conflicts.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected mock patch target for AugmentTransform**
- **Found during:** GREEN phase verification (elastic wiring test)
- **Issue:** The test used `@patch("src.train_ctc.AugmentTransform")` per plan spec, but train_ctc uses a deferred import — `AugmentTransform` doesn't exist as a module-level attribute in `src.train_ctc`; patch raised `AttributeError`.
- **Fix:** Changed to `@patch("src.ctc_utils.AugmentTransform")` — the correct origin module.
- **Files modified:** `tests/test_train_ctc.py`
- **Commit:** `1070bf8`

## Known Stubs

None. All flags are wired through to `AugmentTransform`. `elastic_alpha=0.0` default is intentional (disabled by default, D-02).

## Threat Flags

No new security-relevant surface. T-07-03-01 mitigated: `test_tune_objective_namespace_has_elastic_attrs` regression guard confirms `elastic_alpha=0.0` and `elastic_sigma=5.0` remain in the `_objective` Namespace.

## Self-Check: PASSED

All key files found; both commits (94cc035 RED, 1070bf8 GREEN) verified in git log.

---
*Phase: 07-augmentation-and-two-stage-training*
*Completed: 2026-05-29*
