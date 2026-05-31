---
quick_id: 20260531-obs
slug: hpo-observability-resume-importances
date: 2026-05-31
description: HPO observability, resume support, param importances, and optuna-dashboard
---

# Quick Task: HPO Observability, Resume, and Param Importances

## Tasks

### T1: Progress callback
- Add `_make_progress_callback(n_trials_target)` to `src/tune.py`
- Prints `Trial N: CER=X (K/total done, best=Y)` after each COMPLETE trial
- Wired into `study.optimize(callbacks=[...])`

### T2: SQLite-backed resume
- Add `--storage` (SQLite path) and `--study_name` args to `_build_parser()`
- `create_study(storage=storage_url, study_name=study_name, load_if_exists=True)`
- Compute `to_run = max(0, n_trials - len(prev_complete))` — resume runs only remaining trials
- Print resume summary when prior trials detected

### T3: Param importances
- After `study.optimize()`, compute `optuna.importance.get_param_importances(study)`
- Print sorted fANOVA scores to stdout
- Log as table to ClearML orchestrator task
- Guard: skip if fewer than 2 completed trials

### T4: Artifact upload + dashboard hint
- Upload SQLite DB as ClearML artifact when `--storage` is set
- Print `optuna-dashboard sqlite:///path` command at end
- Add `optuna-dashboard==0.18.0` to dev dependencies in pyproject.toml
