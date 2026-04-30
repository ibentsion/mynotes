---
plan: 03-03
phase: 03-training-evaluation
status: complete
completed: 2026-04-30
---

## Summary

Implemented `src/evaluate.py` — CLI that loads a trained CRNN+CTC checkpoint, runs greedy decode on the held-out validation split (same half-page split logic as train_ctc.py), writes `eval_report.csv`, and logs CER + exact match rate to ClearML task `evaluate_model`.

## What Was Built

- `src/evaluate.py` — standalone CLI satisfying EVAL-01..04: loads `checkpoint.pt` + `charset.json`, reproduces val split via `build_half_page_units` + `split_units(val_frac=args.val_frac)`, pads crops to width%4, runs greedy decode, writes 4-column `eval_report.csv`, logs scalars to ClearML and uploads artifact
- `tests/test_evaluate.py` — 7 tests covering all EVAL requirements; test harness uses `setsid` + process group kill to handle ClearML's background monitor subprocess

## Key Files

### Created
- `src/evaluate.py`
- `tests/test_evaluate.py`

## Commits

- `test(03-03): add failing tests for evaluate.py — RED gate` (worktree)
- `feat(03-03): implement evaluate.py CLI — greedy decode, eval_report.csv, ClearML logging (EVAL-01..04)`

## Requirements Closed

EVAL-01, EVAL-02, EVAL-03, EVAL-04

## Deviations

- Plan 03-03 agent hit a rate limit before committing. `evaluate.py` was complete in the worktree and rescued to main branch by the orchestrator.
- Test harness upgraded to use `setsid`/`os.killpg` to kill ClearML's background monitor process group — prevents test hangs.

## Self-Check: PASSED
