---
phase: quick
plan: 260526-o8k
subsystem: config
tags: [config, dataset-ids, hyperparams, run_config]
completed: "2026-05-26T17:32:29Z"
duration_minutes: 45
tasks_completed: 3
files_created: 2
files_modified: 7

dependency_graph:
  provides:
    - config.yaml: single source of truth for dataset IDs and hyperparam defaults
    - src/run_config.py: load_config() and update_config() shared by all callers
  affects:
    - src/train_ctc.py: reads dataset IDs and hyperparams from config on startup
    - src/tune.py: reads dataset IDs from config; writes best PARAM_KEYS after sweep
    - scripts/register_synthetic_dataset.py: writes synthetic_id to config after upload
    - scripts/create_standalone_dataset.py: writes real_id to config after finalize

tech_stack:
  added:
    - pyyaml==6.0.3 (pinned in pyproject.toml)
  patterns:
    - parser.set_defaults(**config) for zero-friction CLI invocation with config fallback
    - autouse pytest fixture to isolate tests from on-disk config.yaml

key_files:
  created:
    - config.yaml
    - src/run_config.py
  modified:
    - pyproject.toml
    - scripts/register_synthetic_dataset.py
    - scripts/create_standalone_dataset.py
    - src/train_ctc.py
    - src/tune.py
    - tests/conftest.py
    - tests/test_evaluate.py
    - tests/test_tune.py

key_decisions:
  - pyyaml==6.0.3 used (6.0.3 was installed in venv, plan said 6.0.2; used installed version)
  - ty: ignore[invalid-assignment] for nested dict traversal in update_config — ty can't infer dict narrowing through loop
  - ty: ignore[invalid-argument-type] on **{"datasets.real_id": ...} calls — ty incorrectly infers str as Path for path kwarg
  - ty: ignore[unresolved-attribute] on datasets.get() calls — ty loses type after dict lookup returns object
  - tune.py uses truthiness check (if args.dataset_id) not is-not-None for manifest resolution so empty-string override works in tests
  - autouse _no_config_yaml fixture in conftest.py patches load_config to {} for all in-process tests; subprocess test uses --dataset_id "" to override config
  - test_evaluate.py explicit load_config patch kept alongside autouse for clarity (redundant but explicit)
---

# Quick Task 260526-o8k: config.yaml for dataset IDs and hyperparam defaults

config.yaml + src/run_config.py give all scripts a single source of truth for dataset IDs and tuned hyperparams — eliminating manual ID copy-pasting across CLI invocations.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create config.yaml and src/run_config.py; add pyyaml | b37d21a | config.yaml, src/run_config.py, pyproject.toml |
| 2 | Wire update_config into register and create_standalone scripts | b7450f1 | scripts/register_synthetic_dataset.py, scripts/create_standalone_dataset.py |
| 3 | Wire load_config into train_ctc.py and tune.py | ab9b7af | src/train_ctc.py, src/tune.py, tests/conftest.py, tests/test_evaluate.py, tests/test_tune.py |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test suite breakage from config.yaml pickup in tests**
- **Found during:** Task 3 verification
- **Issue:** train_ctc.main() and tune.main() now read config.yaml, which sets `dataset_id` to the real ClearML dataset ID. Tests that call main() in-process or via subprocess without `--dataset_id` picked up this value and tried to download the ClearML dataset, failing with FileNotFoundError or uncaught exception.
- **Fix:** Added autouse `_no_config_yaml` fixture in conftest.py that patches `src.train_ctc.load_config` and `src.tune.load_config` to return `{}` for all in-process tests. For the subprocess test in test_tune.py, added `--dataset_id ""` to the CLI args and changed manifest resolution condition from `is not None` to truthiness check (empty string is falsy, so it won't trigger dataset download).
- **Files modified:** tests/conftest.py, tests/test_evaluate.py, tests/test_tune.py, src/tune.py
- **Commit:** ab9b7af

**2. [Rule 3 - Deviation] pyyaml version 6.0.3 instead of 6.0.2**
- **Found during:** Task 1
- **Issue:** Plan said 6.0.2 but 6.0.3 was already installed in the venv. Using the installed version avoids a downgrade.
- **Fix:** Pinned pyyaml==6.0.3
- **Commit:** b37d21a

**3. [Rule 3 - Deviation] ty: ignore directives instead of type: ignore**
- **Found during:** Task 1, Task 2, Task 3
- **Issue:** Plan specified `type: ignore[assignment]` but ty uses its own `# ty: ignore[...]` suppression syntax (established project pattern).
- **Fix:** Used `# ty: ignore[invalid-assignment]`, `# ty: ignore[unresolved-attribute]`, `# ty: ignore[invalid-argument-type]` per project conventions seen in src/ctc_utils.py, src/prepare_data.py.
- **Commit:** b37d21a, b7450f1, ab9b7af

## Verification Results

- `uv run ruff check` on all modified files: PASSED
- `uv run ty check` on all modified files: PASSED
- `uv run python -c "from src.run_config import load_config; c = load_config(); print(c['datasets']['real_id'])"` prints `02210b47a0534666b0462a175bf2af9d`
- `uv run pytest tests/ -q --ignore=tests/test_clearml_dataset_roundtrip.py`: 182 passed

## Known Stubs

None. All config.yaml values are real (real_id is the active ClearML dataset ID, hyperparams are the pre-tune defaults that work as fallback).

## Self-Check: PASSED
