---
phase: 03-training-evaluation
plan: "02"
subsystem: training
tags: [pytorch, crnn, ctc, clearml, hebrew-ocr, argparse]

# Dependency graph
requires:
  - phase: 03-01
    provides: ctc_utils.py with CRNN, build_charset, crnn_collate, split_units, etc.
  - phase: 01-data-pipeline
    provides: clearml_utils.py helpers, manifest_schema.py, prepare_data.py patterns
provides:
  - src/train_ctc.py — CLI that trains CRNN+CTC on labeled Hebrew crops, logs to ClearML
  - outputs/model/checkpoint.pt — best-val-CER model state dict
  - outputs/model/charset.json — NFC-normalized sorted Hebrew charset
  - tests/test_train_ctc.py — 8 tests covering all TRAN requirements
affects:
  - 03-03-evaluate — loads checkpoint.pt + charset.json produced here

# Tech tracking
tech-stack:
  added: []
  patterns:
    - CropDataset: torch.utils.data.Dataset wrapping manifest DataFrame + charset
    - TDD RED/GREEN: test file written and confirmed failing before implementation
    - in-process test patching for spy tests (avoids subprocess patch-scope mismatch)

key-files:
  created:
    - src/train_ctc.py
    - tests/test_train_ctc.py
  modified: []

key-decisions:
  - "In-process spy for leakage test: subprocess can't see in-process patches; test converted to sys.argv injection + @patch('src.train_ctc.Task') to spy on split_units"
  - "noqa: F401 on 'from clearml import Task': unused but required for @patch('src.train_ctc.Task') test patchability — documented pattern from prepare_data.py"
  - "__getitem__ parameter named 'index' not 'idx': ty enforces LSP compliance with parent Dataset.__getitem__(self, index)"

patterns-established:
  - "Pattern: CropDataset.__getitem__ parameter must be named 'index' (LSP compliance with torch Dataset)"
  - "Pattern: subprocess-based tests cannot spy on in-process patches — use sys.argv injection for patching tests"

requirements-completed: [TRAN-01, TRAN-02, TRAN-03, TRAN-04, TRAN-05, TRAN-06, TRAN-07, TRAN-08]

# Metrics
duration: 7min
completed: 2026-04-29
---

# Phase 3 Plan 02: train_ctc.py Summary

**CRNN+CTC training CLI that filters labeled crops by half-page split, trains on CPU with ClearML logging, and saves best-val-CER checkpoint + charset.json (TRAN-01..08)**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-29T17:32:05Z
- **Completed:** 2026-04-29T17:39:09Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 2 created

## Accomplishments

- `src/train_ctc.py` — runnable CLI implementing full CRNN+CTC training loop on CPU, wiring all ctc_utils functions + ClearML task lifecycle
- `tests/test_train_ctc.py` — 8 tests covering CLI defaults, 4 guard exit codes, status filter, charset delegation, page-leakage protection, and one-epoch end-to-end smoke test
- All TRAN-01..08 requirements observable in code and verified by tests; full 92-test suite passes

## Task Commits

Each task was committed atomically:

1. **Task 1: RED gate — failing tests for train_ctc** - `9a2bccb` (test)
2. **Task 2: GREEN gate — implement train_ctc.py** - `723a843` (feat)

**Plan metadata:** (docs commit below)

_Note: TDD tasks have two commits (test RED → feat GREEN)_

## Files Created/Modified

- `~/git/mynotes/src/train_ctc.py` — CLI: manifest filter, charset build, half-page split, CRNN+CTC training loop, ClearML logging, checkpoint save
- `~/git/mynotes/tests/test_train_ctc.py` — 8 unit/integration tests with CLEARML_OFFLINE_MODE=1

## Decisions Made

- **In-process spy for leakage test:** The plan sketched `test_no_page_leakage_between_train_and_val` using subprocess + in-process patching of `split_units`, which can't work (subprocess runs a separate Python interpreter). Converted the test to use `sys.argv` injection + `@patch("src.train_ctc.Task")` to run `main()` in-process, enabling the spy to capture `(train_keys, val_keys)`.

- **noqa: F401 on Task import:** `from clearml import Task` is flagged unused by ruff F401, but it is required at module level so `@patch("src.train_ctc.Task")` works in tests. Added `# noqa: F401` with an explanatory comment; this is the project-established pattern from `prepare_data.py`.

- **`__getitem__` parameter named `index`:** ty enforces LSP compliance — the parent `Dataset.__getitem__` signature uses `index`, so `CropDataset.__getitem__` must too. Renamed from `idx` to `index`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_no_page_leakage_between_train_and_val test design**
- **Found during:** Task 2 (GREEN gate — running tests)
- **Issue:** Test used `_run_cli()` (subprocess) + `patch("src.train_ctc.split_units")` to spy on the spy. Subprocess runs a separate Python process that can't see in-process patches, so `captured_split` was always empty, causing the test to fail even with correct implementation.
- **Fix:** Converted test to call `main()` in-process using `sys.argv` injection (same pattern as `test_status_filter_keeps_only_labeled`). Kept `@patch("src.train_ctc.Task")` to suppress ClearML. The spy on `split_units` now successfully captures `(train_keys, val_keys)`.
- **Files modified:** `tests/test_train_ctc.py`
- **Verification:** All 8 tests pass (GREEN).
- **Committed in:** `723a843` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug in test design)
**Impact on plan:** Bug in the test plan that would have caused a green implementation to appear failing. Fixed in-place during Task 2; no scope creep.

## Issues Encountered

- `ruff SIM117`: nested `with` statements in tests (auto-fixed by combining into single `with (patch(...), patch(...)):`)
- `ty invalid-method-override`: `CropDataset.__getitem__` parameter name mismatch with parent class (auto-fixed by renaming `idx` → `index`)

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `outputs/model/checkpoint.pt` and `outputs/model/charset.json` format defined and tested; Plan 03-03 (`evaluate.py`) can load them
- Training runs successfully on CPU with CLEARML_OFFLINE_MODE=1 for offline development
- Real training requires labeled crops in manifest.csv (`status == "labeled"`) and min 10 labeled rows

## Self-Check: PASSED

- src/train_ctc.py: FOUND
- tests/test_train_ctc.py: FOUND
- 03-02-SUMMARY.md: FOUND
- commit 9a2bccb: FOUND
- commit 723a843: FOUND

---
*Phase: 03-training-evaluation*
*Completed: 2026-04-29*
