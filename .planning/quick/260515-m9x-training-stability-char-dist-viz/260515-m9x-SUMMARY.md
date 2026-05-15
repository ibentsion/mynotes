---
phase: quick/260515-m9x
plan: 01
status: complete
subsystem: training-stability
tags: [training, clearml, optuna, hpo, early-stopping, lr-schedule, adamw, char-distribution]
dependency_graph:
  requires: [src/train_ctc.py, src/ctc_utils.py, src/tune.py]
  provides: [stable-training-pipeline, char-distribution-viz]
  affects: [ClearML training runs, HPO sweep quality]
key_files:
  modified:
    - src/train_ctc.py
    - src/ctc_utils.py
    - src/tune.py
    - tests/test_train_ctc.py
    - tests/test_ctc_utils.py
commits:
  - 8e6c82f  # M9X-01: narrow HPO search space
  - 4803153  # M9X-02: AdamW + weight_decay
  - bea9442  # M9X-03: ReduceLROnPlateau
  - ddafab2  # M9X-04: early stopping + restore-best
  - 9d95323  # M9X-05: crnn_collate CTC padding fix
  - 97830be  # M9X-06: configurable blank bias
  - 060bb06  # M9X-07: char distribution visualization
metrics:
  duration: ~30 minutes
  completed: 2026-05-15
  tasks: 7
  files_modified: 5
  tests_added: 2
  tests_total: 93
---

# Quick Task 260515-m9x: Training Stability Improvements + Char Distribution Viz

**One-liner:** Applied 7 training stability improvements from tune run analysis, each
committed atomically: AdamW, LR schedule, early stopping, CTC padding fix, configurable
blank bias, narrowed HPO search space, and character distribution visualization.

## Requirements Completed

- **M9X-01:** Narrow tune.py LR range [1e-4,1e-2] → [5e-5,3e-3]; drop rnn_hidden=512.
- **M9X-02:** Adam → AdamW with weight_decay=1e-4. `--weight_decay` arg added.
- **M9X-03:** ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6) on val_cer. LR logged as scalar.
- **M9X-04:** Early stopping with patience=5 (configurable) + restore-best checkpoint after loop.
- **M9X-05:** crnn_collate ensures padded_w//4 >= max_label_len+2, eliminating inf_loss permanently.
- **M9X-06:** `--blank_bias_init` (default=-2.0). Space/niqqud are real chars, not blank-like.
- **M9X-07:** `_report_char_distribution` logs horizontal bar chart + sortable table to ClearML.

## Blank-Like Character Analysis

Only the synthetic CTC blank token (index 0) warrants negative bias. Real Hebrew characters
including space, niqqud (vowel points), and narrow letters like yod (י) or vav (ו) must NOT
be suppressed — they are valid predictions that the model should learn freely. If blank_frac
stays >0.9 after training stabilizes with AdamW + LR schedule, try `--blank_bias_init -3.0`
or `-4.0` (but expect diminishing returns — blank collapse is a training-data-size problem,
not primarily an initialization problem).

## Self-Check: PASSED

93 tests pass. ruff clean. ty clean on all modified files.
