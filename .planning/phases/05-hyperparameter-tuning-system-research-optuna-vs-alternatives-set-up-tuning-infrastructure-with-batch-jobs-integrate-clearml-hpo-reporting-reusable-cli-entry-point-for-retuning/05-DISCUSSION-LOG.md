# Phase 5: Hyperparameter Tuning System - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-05
**Phase:** 05-hyperparameter-tuning-system
**Areas discussed:** Search space, Tuner & execution, Trial budget & pruning, Results & best-config UX

---

## Search Space

| Option | Description | Selected |
|--------|-------------|----------|
| Training params | lr, batch_size, epochs | ✓ |
| Architecture params | rnn_hidden ∈ {128,256,512}, num_layers ∈ {1,2} | ✓ |
| Augmentation params | aug_copies, rotation_max, brightness_delta, noise_sigma | ✓ |

**User's choice:** All three groups

**Architecture ranges follow-up:**

| Option | Description | Selected |
|--------|-------------|----------|
| 128 / 256 / 512, layers 1–2 | Full practical range | ✓ |
| 256 / 512 only, layers 2 fixed | Width only | |
| Claude decides ranges | Delegate to Claude | |

**Notes:** Architecture params currently hardcoded — must be exposed as CLI args first.

---

## Tuner & Execution

### Tuner selection

| Option | Description | Selected |
|--------|-------------|----------|
| Optuna standalone | Local Optuna, separate ClearML task per trial | |
| ClearML HyperParameterOptimizer | Native ClearML HPO | |
| Simple random/grid search | Plain tune.py, no extra deps | |

**User's choice:** "1 and 2 - I want to compare them and learn about the difference"

**Follow-up — how to compare:**

| Option | Description | Selected |
|--------|-------------|----------|
| Research decides, implement one | Researcher recommends; implement winner only | ✓ |
| Implement both, flag-selectable | --backend optuna\|clearml | |
| Implement Optuna now, ClearML HPO later | Ship Optuna, defer ClearML HPO | |

### Execution mode

| Option | Description | Selected |
|--------|-------------|----------|
| Local CPU | Trials run locally | |
| GPU queue via --enqueue | Enqueue each trial to ClearML GPU queue | ✓ |
| Flag-selectable (both) | --remote flag | |

---

## Trial Budget & Pruning

### Budget

| Option | Description | Selected |
|--------|-------------|----------|
| 20–30 trials, CLI flag | --n_trials N, default 20 | ✓ |
| Fixed 50 trials | More thorough | |
| Time-based budget | --timeout N_minutes | |

### Pruning

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — prune by CER after first few epochs | MedianPruner or equivalent | ✓ |
| No pruning | Each trial runs full epochs | |
| Claude decides | Add if backend supports it cleanly | |

---

## Results & Best-Config UX

### Output format

| Option | Description | Selected |
|--------|-------------|----------|
| Print best + ClearML report | Stdout + ClearML scalar plots + best-params table | |
| Auto-write best_params.json | Writes outputs/best_params.json | |
| Both | Print + ClearML report AND write JSON file | ✓ |

**Notes:** train_ctc.py gets --params flag to load best_params.json.

### Entry point

| Option | Description | Selected |
|--------|-------------|----------|
| Standalone tune.py | python -m src.tune --manifest ... | ✓ |
| Extend train_ctc.py with --tune mode | Keeps everything in one script | |

---

## Claude's Discretion

- LR sweep range and distribution
- Exact batch_size values
- Epochs range for trials
- Augmentation sweep ranges
- Pruning warm-up epoch count
- ClearML task name for orchestrator

## Deferred Ideas

None
