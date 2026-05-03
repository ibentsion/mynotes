---
phase: 04-data-augmentation-and-gpu-training-via-clearml-agent
plan: 01
subsystem: training
tags: [pytorch, augmentation, crnn, ctc, clearml, tdd]

requires:
  - phase: 03-training-evaluation
    provides: CropDataset in train_ctc.py, CRNN+CTC training pipeline, ctc_utils.py utilities

provides:
  - AugmentTransform class in ctc_utils.py (rotation, brightness jitter, Gaussian noise)
  - Augmentation-aware CropDataset in ctc_utils.py with augment/aug_copies params
  - --aug_copies, --rotation_max, --brightness_delta, --noise_sigma CLI flags in train_ctc.py
  - Effective dataset size printed and logged to ClearML when aug_copies > 0

affects:
  - 04-02-PLAN  # GPU training plan uses CropDataset from ctc_utils; same training CLI

tech-stack:
  added: []
  patterns:
    - "AugmentTransform as seeded callable: takes (tensor, seed) returns same-shape tensor — deterministic per logical dataset index"
    - "CropDataset aug_copies: __len__ = real_len * (1 + copies); copy_idx=0 always returns clean crop"
    - "torch.rand(1, generator=rng) for seeded uniform sampling (torch.empty does not accept generator kwarg)"
    - "F.affine_grid + F.grid_sample with padding_mode=border for rotation without blank contamination"

key-files:
  created: []
  modified:
    - src/ctc_utils.py
    - src/train_ctc.py
    - tests/test_ctc_utils.py
    - tests/test_train_ctc.py

key-decisions:
  - "AugmentTransform uses torch.rand(1, generator=rng) not torch.empty(..., generator=rng) — torch.empty does not accept generator kwarg in torch 2.11"
  - "CropDataset moved from train_ctc.py to ctc_utils.py — augmentation logic belongs in the dataset utility module"
  - "encode_label and load_crop removed from train_ctc.py imports — they were only used by CropDataset which is now in ctc_utils.py"
  - "val_ds always created as CropDataset(val_df, charset) with no augment= kwarg — defaults to None per D-04"

patterns-established:
  - "Pattern: seeded augmentation callable takes (tensor, seed: int) for deterministic per-index transforms"
  - "Pattern: CropDataset backward-compatible with augment=None, aug_copies=0 defaults — Phase 3 tests unchanged"

requirements-completed: []

duration: 17min
completed: 2026-05-03
---

# Phase 04 Plan 01: Online Data Augmentation Summary

**AugmentTransform class (rotation ±7°, brightness ±10%, Gaussian noise sigma=0.02) wired into CropDataset via aug_copies multiplier; train_ctc.py gets 4 new CLI flags; val split always clean**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-05-03T19:53:34Z
- **Completed:** 2026-05-03T20:10:54Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- AugmentTransform applies rotation, brightness jitter, and Gaussian noise via pure PyTorch (no torchvision); seeded per logical dataset index for reproducibility
- CropDataset moved from train_ctc.py to ctc_utils.py and extended with augment/aug_copies params; __len__ multiplied by (1 + copies); copy_idx=0 always returns clean unaugmented crop
- train_ctc.py gains --aug_copies (default 0), --rotation_max, --brightness_delta, --noise_sigma flags; prints and logs effective dataset size when aug_copies > 0
- 14 new tests (5 AugmentTransform unit tests + 5 CropDataset aug tests + 4 train_ctc augmentation integration tests); all 106 tests pass including backward-compat with Phase 3

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Add failing tests for AugmentTransform and CropDataset** - `b3998f2` (test)
2. **Task 1 GREEN: Add AugmentTransform class and augmentation-aware CropDataset to ctc_utils** - `e77aa90` (feat)
3. **Task 2 RED: Add failing tests for augmentation CLI flags in train_ctc** - `54cfabe` (test)
4. **Task 2 GREEN: Wire augmentation CLI flags into train_ctc and remove CropDataset** - `bd55b2e` (feat)

**Plan metadata:** (docs commit — see state updates)

_Note: TDD tasks have multiple commits (test → feat)_

## Files Created/Modified

- `/home/ido/git/mynotes/src/ctc_utils.py` - Added AugmentTransform class and Dataset import; CropDataset extended with augment/aug_copies params; import torch.nn.functional as F added
- `/home/ido/git/mynotes/src/train_ctc.py` - CropDataset class removed; AugmentTransform and CropDataset imported from ctc_utils; 4 new CLI flags added; augmentation setup block wired in main()
- `/home/ido/git/mynotes/tests/test_ctc_utils.py` - 10 new tests for AugmentTransform and augmented CropDataset
- `/home/ido/git/mynotes/tests/test_train_ctc.py` - 4 new tests for augmentation CLI flags, backward compat, effective size print, and D-04 val-clean guarantee

## Decisions Made

- `torch.rand(1, generator=rng)` used instead of `torch.empty(..., generator=rng)` — torch.empty does not accept the `generator` keyword argument in torch 2.11; torch.rand does
- CropDataset moved to ctc_utils.py as specified in PLAN.md action step; `encode_label` and `load_crop` imports removed from train_ctc.py since they were only used by the now-removed CropDataset class
- val_ds constructed as `CropDataset(val_df, charset)` without `augment=` kwarg (defaults to None) — clean and explicit per D-04

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] torch.empty API mismatch — switched to torch.rand**
- **Found during:** Task 1 GREEN phase
- **Issue:** Plan code used `torch.empty(1, generator=rng).uniform_(...)` but torch.empty does not accept `generator` keyword arg in torch 2.11.0. Also tried `torch.empty((1,), generator=rng)` — same error.
- **Fix:** Replaced with `torch.rand(1, generator=rng) * 2 * max - max` for uniform sampling in range; same for brightness delta
- **Files modified:** src/ctc_utils.py
- **Verification:** All 5 AugmentTransform tests pass including determinism (same seed = same output) and seed diversity (different seeds = different output)
- **Committed in:** e77aa90 (Task 1 feat commit)

**2. [Rule 1 - Bug] Removed unused imports after CropDataset migration**
- **Found during:** Task 2 GREEN phase
- **Issue:** After removing CropDataset from train_ctc.py, `encode_label` and `load_crop` became unused imports; ruff check reported F401 errors
- **Fix:** Removed `encode_label` and `load_crop` from the ctc_utils import block in train_ctc.py
- **Files modified:** src/train_ctc.py
- **Verification:** `uv run ruff check src/train_ctc.py` passes clean
- **Committed in:** bd55b2e (Task 2 feat commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the deviations documented above.

## Known Stubs

None — all augmentation wiring is fully implemented and tested.

## Next Phase Readiness

- Phase 04 Plan 02 (GPU training via ClearML agent) can proceed: CropDataset and AugmentTransform are stable, all tests pass
- The augmentation flags are backward-compatible (aug_copies=0 default); existing training workflows unchanged

---
*Phase: 04-data-augmentation-and-gpu-training-via-clearml-agent*
*Completed: 2026-05-03*

## Self-Check: PASSED

- FOUND: src/ctc_utils.py
- FOUND: src/train_ctc.py
- FOUND: tests/test_ctc_utils.py
- FOUND: tests/test_train_ctc.py
- FOUND: .planning/phases/04-data-augmentation-and-gpu-training-via-clearml-agent/04-01-SUMMARY.md
- FOUND commit: b3998f2 (test RED task 1)
- FOUND commit: e77aa90 (feat GREEN task 1)
- FOUND commit: 54cfabe (test RED task 2)
- FOUND commit: bd55b2e (feat GREEN task 2)
