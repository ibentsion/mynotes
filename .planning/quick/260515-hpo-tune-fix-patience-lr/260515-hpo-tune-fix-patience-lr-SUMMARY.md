---
phase: quick/260515-hpo-tune-fix-patience-lr
plan: 01
status: complete
subsystem: training-stability
tags: [hpo, tune, optuna, early-stopping, lr-schedule, adamw]
dependency_graph:
  requires: [src/tune.py, src/train_ctc.py]
  provides: [stable-hpo-sweep-config]
  affects: [ClearML HPO tune runs]
key_files:
  modified:
    - src/tune.py
    - src/train_ctc.py
commits:
  - f73ee20  # fix(tune): patience=0 + max LR 3e-3→1e-3
  - ba8dd2c  # fix(train): ReduceLROnPlateau patience 3→6
metrics:
  duration: ~10 minutes
  completed: 2026-05-15
  tasks: 3
  files_modified: 2
  tests_run: 79
---

# Quick Task 260515-hpo-tune-fix-patience-lr: HPO Tune Config Fixes

**One-liner:** Fixed the interaction bug where early stopping (patience=5) raced Optuna's
MedianPruner (n_warmup_steps=5), killing trials before the pruner gathered signal;
also lowered max LR and increased LR plateau patience.

## Root Cause Analysis

The previous tune run (`f8eae65d5a5b4d8f8071b7cd657f7221`) showed a square-wave
oscillation pattern: blank_frac and empty_frac alternating 0↔1 per trial, CER
between 0.65–0.75 (worse than prior 0.5966 baseline). Each trial ran only 5–7 epochs.

**Primary bug:** `patience=5` in `_objective`'s Namespace == `n_warmup_steps=5` in
the MedianPruner. Early stopping fired at epoch 5 for any trial whose CER didn't
improve (which included all blank-collapsed trials), bypassing Optuna's pruner
entirely. Optuna got no useful comparative signal.

**Secondary:** Max LR 3e-3 caused ~half of trials to blank-collapse immediately with
AdamW on ~80 training crops. Log-uniform sampling meant ~40% of trials landed in the
problematic upper LR range.

**Tertiary:** ReduceLROnPlateau patience=3 halved LR on noise for a tiny dataset,
compounding recovery difficulty in high-LR trials.

## Changes Made

| Fix | File | Change |
|-----|------|--------|
| FIX-01 | tune.py _objective Namespace | patience=5 → patience=0 (Optuna pruner is sole termination) |
| FIX-02 | tune.py _suggest_params | max LR 3e-3 → 1e-3 |
| FIX-03 | train_ctc.py ReduceLROnPlateau | patience=3 → patience=6 |

## New Tune Run Enqueued

Task ID: `3e29d2ed0f6047f0857ad068b4647314`  
Queue: `ofek`  
Dataset: `6d4d23aba6de4e9dbb47590ab4d5ba29` (data_prep v1.0.4)  
Params: 20 trials, n_startup_trials=5, n_warmup_steps=5

## Self-Check: PASSED

79 tests pass. ruff clean. ty clean.
