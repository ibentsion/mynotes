---
quick_id: 20260601-hpo
status: complete
date: 2026-06-01
commit: 67b7da0
---

# Summary

## Answers to follow-up questions

**SQLite on abort**: Written after every `trial.report(val_cer, epoch)` call (each epoch) AND
after trial completion. If killed mid-trial, that trial is `RUNNING` state in DB (not counted
as complete) — all prior completed trials are intact. No data loss for completed work.

**Agent dependency**: `optuna.importance` uses sklearn (already in prod deps). `optuna-dashboard`
is dev-only, never imported by `tune.py`. SQLite via Python stdlib `sqlite3`. No new prod deps.

**Resume tested**: `test_resume_sqlite_study` — runs 3 trials with SQLite, then resumes to 5
total (runs exactly 2 more), verifies best CER is from the resumed run.

**Inspection CLI**: `tune-hpo-inspect` — shows trial counts (complete/pruned/stale-running),
best params, fANOVA importances with bar chart. `--task_id` downloads ClearML artifact.
`--dashboard` launches optuna-dashboard. `--mode pretrain|finetune` selects param key list.

**"Every ~10 iterations"**: Not needed — Optuna already writes to SQLite after every epoch
(via `trial.report()`). Sub-trial granularity is already there.

**Iterations vs trials**: 1 trial = 1 training run = N epochs (20–50 for finetune, 10–50 for pretrain).
Each epoch = 1 dataset pass. SQLite updated per epoch and per trial.

**Both modes work**: `--mode pretrain` uses `PRETRAIN_PARAM_KEYS`, `--mode finetune` uses `PARAM_KEYS`.
`study_name` defaults to `hpo_pretrain`/`hpo_finetune`. Separate `--storage` paths recommended.

## Files changed

- `tests/test_tune.py`: `test_resume_sqlite_study`, `test_smoke_imports` updated
- `src/hpo_inspect.py`: new inspection CLI (`tune-hpo-inspect` entry point)
- `pyproject.toml`: `tune-hpo-inspect` entry point added
- `CLAUDE.md`: HPO workflow section (run/resume/inspect for both modes)
