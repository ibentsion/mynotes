---
phase: quick/260515-hpo-tune-fix-patience-lr
plan: 01
status: in_progress
subsystem: training-stability
tags: [hpo, tune, optuna, early-stopping, lr-schedule]
dependency_graph:
  requires: [src/tune.py, src/train_ctc.py]
  provides: [stable-hpo-sweep]
  affects: [ClearML HPO tune runs]
key_files:
  to_modify:
    - src/tune.py
    - src/train_ctc.py
---

# Quick Task 260515-hpo-tune-fix-patience-lr

**One-liner:** Fix HPO tune config degradation: disable early stopping during HPO
(patience=0), lower max LR from 3e-3 to 1e-3, and increase ReduceLROnPlateau
patience 3→6. Root cause: early stopping races Optuna pruner (both fire at epoch 5).

## Root Cause

Three issues causing worse CER (~0.65) vs prior baseline (0.5966):

1. **patience=5 in tune.py Namespace == n_warmup_steps=5** — early stopping fires before
   Optuna pruner sees any data, killing trials after only 5-7 epochs. Optuna gets no
   useful signal; trials terminated via wrong mechanism.

2. **Max LR 3e-3 too high with AdamW** on ~80 training crops — upper half of log-sampled
   LR range causes immediate blank collapse, producing alternating collapsed/ok pattern.

3. **ReduceLROnPlateau patience=3 too short** — halves LR after 3 non-improving epochs on
   a tiny dataset where 3 epochs is noise. Compounds blank collapse recovery difficulty.

## Tasks

- [ ] FIX-01: tune.py: set patience=0 in _objective Namespace (disable early stopping during HPO)
- [ ] FIX-02: tune.py: lower max LR from 3e-3 to 1e-3 in _suggest_params
- [ ] FIX-03: train_ctc.py: increase ReduceLROnPlateau patience 3→6
- [ ] TEST: run pytest -q, ruff check, ty check
- [ ] PUSH: git push
- [ ] ENQUEUE: launch new tune run on ofek queue with latest dataset
- [ ] STATE: update STATE.md Quick Tasks Completed table
