---
phase: 07-augmentation-and-two-stage-training
plan: "02"
subsystem: augmentation
tags: [albumentations, elastic-deformation, AugmentTransform, ctc_utils, pytest]

requires:
  - phase: 07-augmentation-and-two-stage-training
    plan: "01"
    provides: Green test baseline for Phase 7 feature work

provides:
  - albumentations==2.0.8 dependency in pyproject.toml
  - AugmentTransform with optional elastic deformation (elastic_alpha, elastic_sigma params)
  - 5 elastic augmentation tests in test_ctc_utils.py

affects:
  - 07-augmentation-and-two-stage-training plans 03+ (train_ctc.py can now use elastic augmentation)

tech-stack:
  added:
    - albumentations==2.0.8 (elastic deformation; albucore, simsimd, stringzilla pulled as deps)
  patterns:
    - "Deferred import with # noqa: PLC0415 for optional heavy deps inside conditional blocks"
    - "albumentations 2.x API: fill= not value=, p=1.0 not always_apply=True, distort_limit= not distort_range="
    - "Numpy (H, W, 1) channel dim required for albumentations; squeeze/unsqueeze to match (1, H, W) PyTorch contract"

key-files:
  created: []
  modified:
    - pyproject.toml
    - src/ctc_utils.py
    - tests/test_ctc_utils.py
    - tests/test_train_ctc.py

key-decisions:
  - "Use distort_limit= not distort_range= — RESEARCH.md had incorrect API name for GridDistortion in albumentations 2.0.8"
  - "Deferred albumentations import inside if elastic_alpha > 0 block to avoid import overhead when feature is disabled"
  - "Applied worktree-inherited mock fix (extra_words=None) for pre-existing test_train_ctc.py failures that were fixed in 07-01 but not present in this worktree"

patterns-established:
  - "Pattern: albumentations 2.x requires (H, W, C) numpy input via A.Compose; use tensor.squeeze(0).numpy()[:, :, np.newaxis]"
  - "Pattern: verify parameter names by inspecting actual installed library API, not just RESEARCH.md"

requirements-completed:
  - AUG-01

duration: ~15min
completed: 2026-05-29
---

# Phase 7 Plan 02: Add albumentations dep and extend AugmentTransform with elastic deformation

**albumentations==2.0.8 added as dependency; AugmentTransform gains elastic_alpha/elastic_sigma params with deferred import; 5 new elastic tests all pass**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-29T09:40:00Z
- **Completed:** 2026-05-29T09:54:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `albumentations==2.0.8` to `pyproject.toml` dependencies; `uv sync` installs it cleanly
- Extended `AugmentTransform.__init__` with `elastic_alpha: float = 0.0` and `elastic_sigma: float = 5.0`
- Added elastic deformation block in `AugmentTransform.__call__` guarded by `if self.elastic_alpha > 0`
- Deferred `import albumentations as A  # noqa: PLC0415` inside the block (zero import overhead when disabled)
- Output tensor stays `(1, H, W) float32` with values clamped to `[0, 1]` via `torch.clamp`
- Added 5 tests: defaults, zero-alpha skip, shape preservation, value clamping, content modification
- All 84 tests in `test_ctc_utils.py` + `test_train_ctc.py` pass

## Task Commits

1. **Task 1: Add albumentations dep and extend AugmentTransform** - `3846703` (feat)
2. **Task 2: Add elastic augmentation tests** - `58500b0` (test)

## Files Created/Modified

- `pyproject.toml` — added `albumentations==2.0.8` dependency
- `uv.lock` — updated with albumentations + transitive deps (albucore, simsimd, stringzilla)
- `src/ctc_utils.py` — extended `AugmentTransform.__init__` and `__call__` with elastic path
- `tests/test_ctc_utils.py` — added 5 `test_augment_transform_elastic_*` tests
- `tests/test_train_ctc.py` — applied worktree-inherited mock signature fix (pre-existing issue)

## Decisions Made

- **distort_limit= not distort_range=**: The plan's `<interfaces>` section stated `distort_range=(-0.15, 0.15)` based on RESEARCH.md, but actual albumentations 2.0.8 API uses `distort_limit=`. Verified by inspecting `inspect.signature(A.GridDistortion.__init__)`. Used the correct parameter name.
- **Numpy import deferred**: `numpy` is not imported at module level in `ctc_utils.py` (only via `# noqa: PLC0415` inside `compute_char_saliency`), so the elastic block also uses a deferred import.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Used distort_limit= instead of distort_range= for GridDistortion**
- **Found during:** Task 1 implementation
- **Issue:** Plan's `<interfaces>` section specified `distort_range=(-0.15, 0.15)` but albumentations 2.0.8 GridDistortion uses `distort_limit=`. Initial implementation emitted a `UserWarning: Argument(s) 'distort_range' are not valid for transform GridDistortion`.
- **Fix:** Changed to `distort_limit=(-0.15, 0.15)` after verifying with `inspect.signature(A.GridDistortion.__init__)`
- **Files modified:** `src/ctc_utils.py`
- **Commit:** `3846703`

**2. [Rule 3 - Blocking] Applied pre-existing mock signature fix from worktree divergence**
- **Found during:** Task 2 done-criteria check (`uv run pytest tests/test_ctc_utils.py tests/test_train_ctc.py -q`)
- **Issue:** This worktree was created before the 07-01 fix was merged into master. Two tests in `test_train_ctc.py` failed with `TypeError: fake_build_charset() got an unexpected keyword argument 'extra_words'` — the same issue fixed by Plan 07-01 commit `b92ffbb`.
- **Fix:** Applied same fix: `fake_build_charset(labels, extra_words=None)` and `capture_build_charset(labels, extra_words=None)`
- **Files modified:** `tests/test_train_ctc.py`
- **Commit:** `58500b0`

## Known Stubs

None — all implemented functionality is wired and tested. The `elastic_alpha=0.0` default is intentional (disabled by default per D-02), not a stub.

## Threat Flags

No new security-relevant surface introduced. `albumentations` is a widely-known library; exact version pin `==2.0.8` per T-07-02-01 mitigation. `torch.clamp(tensor, 0.0, 1.0)` applied after elastic conversion per T-07-02-02 mitigation.

---
*Phase: 07-augmentation-and-two-stage-training*
*Completed: 2026-05-29*
