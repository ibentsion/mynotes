---
phase: quick-260508-hqp
plan: 01
subsystem: tune
tags: [hpo, clearml, argparse, optuna, fix]
dependency_graph:
  requires: [src/train_ctc.py::run_training]
  provides: [src/tune.py::_objective (argparse-free)]
  affects: [ClearML GPU agent sweep execution]
tech_stack:
  added: []
  patterns: [direct argparse.Namespace construction to bypass ClearML monkey-patching]
key_files:
  modified:
    - src/tune.py
    - tests/test_tune.py
decisions:
  - "_objective now constructs argparse.Namespace directly; no parser.parse_* calls so ClearML cannot intercept and inject tune-only flags into train_ctc's parser"
metrics:
  duration: ~5 minutes
  completed: 2026-05-08
  tasks_completed: 1
  files_changed: 2
---

# Quick Task 260508-hqp: Fix HPO Sweep API Error Summary

One-liner: Bypass argparse in `_objective` via direct `Namespace` construction so ClearML agent sys.argv injection cannot cause "unrecognized arguments" failures.

## What Changed in src/tune.py

### Removed import

```python
# REMOVED:
from src.train_ctc import _build_parser as _build_train_parser
```

`run_training` is still imported; `_build_parser` is not needed in `_objective` anymore.

### _objective: before

```python
cli = ["--manifest", str(sweep_args.manifest), "--min_labeled", str(sweep_args.min_labeled)]
if sweep_args.dataset_id is not None:
    cli += ["--dataset_id", sweep_args.dataset_id]
train_args, _ = _build_train_parser().parse_known_args(cli)
for k, v in params.items():
    setattr(train_args, k, v)
train_args.output_dir = sweep_args.output_dir / f"trial_{trial.number}"
```

### _objective: after

```python
train_args = argparse.Namespace(
    manifest=sweep_args.manifest,
    output_dir=sweep_args.output_dir / f"trial_{trial.number}",
    val_frac=0.2,
    min_labeled=sweep_args.min_labeled,
    num_workers=0,
    params=None,
    enqueue=False,
    queue_name="gpu",
    dataset_id=sweep_args.dataset_id,
    lr=params["lr"],
    batch_size=params["batch_size"],
    epochs=params["epochs"],
    rnn_hidden=params["rnn_hidden"],
    num_layers=params["num_layers"],
    aug_copies=params["aug_copies"],
    rotation_max=params["rotation_max"],
    brightness_delta=params["brightness_delta"],
    noise_sigma=params["noise_sigma"],
)
```

The `try/except/finally` block (TrialPruned re-raise, `trial_task.close()`) is unchanged.

## The Two New Tests and What They Prove

### test_objective_ignores_extra_sys_argv_args

Puts `["tune.py", "--n_trials", "20", "--n_startup_trials", "5", "--n_warmup_steps", "5"]` in `sys.argv` — exactly what the ClearML GPU agent injects — then calls `_objective` with `run_training` and `init_task` mocked.

With the old `parse_known_args` code this would have triggered argparse's systematic argument injection (ClearML monkey-patches `parse_known_args` to pull from stored task hyperparams, injecting tune-only flags that train_ctc's parser rejects). With the new direct-Namespace code the test passes cleanly.

### test_e2e_one_trial_no_enqueue

Writes a tiny one-row `manifest.csv` (with `status=labeled`), sets `sys.argv` to `["tune.py", "--manifest", ..., "--n_trials", "1", "--min_labeled", "1", "--output_dir", ...]`, mocks `run_training` and `init_task`, and calls `main()` without mocking Optuna. Verifies:
- `main()` returns 0
- `best_params.json` is written to `output_dir`
- All `PARAM_KEYS` are present in the JSON

## ClearML Monkey-Patch Interaction: Now Bypassed

ClearML's agent overrides `ArgumentParser.parse_args` and `parse_known_args` to inject all stored task hyperparams from the server into `sys.argv`. When `_objective` called `_build_train_parser().parse_known_args(cli)`, ClearML intercepted it and inserted `--n_trials`, `--n_startup_trials`, `--n_warmup_steps` — which train_ctc's parser does not recognize, causing the crash.

`argparse.Namespace(...)` is plain object construction — no parser instance, no `parse_*` method, nothing for ClearML to hook.

## Next Manual Step

Trigger a real GPU sweep to confirm the agent reaches its first trial without error:

```bash
git push
uv run python -m src.tune --enqueue --queue_name gpu --n_trials 2 --dataset_id <id>
```

Watch the agent log for "Trial 0" starting cleanly with no "unrecognized arguments" message.

## Deviations from Plan

None — plan executed exactly as written (ruff format auto-corrected one inline trailing whitespace in tune.py `_build_parser`).

## Self-Check: PASSED

- src/tune.py exists: FOUND
- tests/test_tune.py exists: FOUND
- commit 714f9bf: FOUND
- `argparse.Namespace(` in src/tune.py: FOUND
- `_build_parser as _build_train_parser` absent from src/tune.py: CONFIRMED
- `test_objective_ignores_extra_sys_argv_args` in tests/test_tune.py: FOUND
- `test_e2e_one_trial_no_enqueue` in tests/test_tune.py: FOUND
- 14/14 tests passing: CONFIRMED
