---
phase: 04-data-augmentation-and-gpu-training-via-clearml-agent
plan: 02
subsystem: training
tags: [clearml, pytorch, python, ocr, training, gpu, agent, wsl2]

# Dependency graph
requires:
  - phase: 04-01
    provides: AugmentTransform and CropDataset with augmentation in ctc_utils.py, aug CLI flags in train_ctc.py
  - phase: 03-training-evaluation
    provides: CRNN+CTC training loop in train_ctc.py, ClearML logging patterns
provides:
  - remap_dataset_paths() in clearml_utils.py for in-memory manifest path remapping from ClearML dataset cache
  - --enqueue, --queue_name, --dataset_id CLI flags in train_ctc.py
  - task.execute_remotely() call (after task.connect()) for GPU queue dispatch
  - docs/clearml-agent-setup.md with WSL2 agent setup, CUDA 12.8 requirements, smoke test
affects:
  - future phases that dispatch training to ClearML queues
  - WSL2 GPU agent setup for training at scale

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "remap_dataset_paths returns df copy (D-10): never mutate original manifest DataFrame"
    - "init_task tags computed from args before call: tags = ['phase-4', 'gpu'] if args.enqueue else ['phase-4']"
    - "execute_remotely always placed after task.connect(): connects hyperparams first, then dispatches to agent"
    - "patch src.train_ctc.init_task for ordering tests (not src.train_ctc.Task): Task.init goes through clearml_utils"

key-files:
  created:
    - docs/clearml-agent-setup.md
  modified:
    - src/clearml_utils.py
    - src/train_ctc.py
    - tests/test_clearml_utils.py
    - tests/test_train_ctc.py

key-decisions:
  - "patch src.train_ctc.init_task instead of src.train_ctc.Task for execute_remotely ordering test: Task.init is called inside clearml_utils.init_task, so patching src.train_ctc.Task does not control the returned task object"
  - "remap happens before task.connect() and before output_dir mkdir: paths must be correct before any computation"
  - "dataset_id remapping is in-memory only: manifest.csv is never modified (D-10)"

patterns-established:
  - "Pattern: remap_dataset_paths called before connect: ensures remapped paths are reflected in ClearML hyperparams"

requirements-completed: []

# Metrics
duration: 35min
completed: 2026-05-04
---

# Phase 4 Plan 02: ClearML Remote Execution Summary

**ClearML GPU dispatch via --enqueue/--dataset_id: task.connect() before execute_remotely(), in-memory path remapping via remap_dataset_paths(), and WSL2 agent setup guide for RTX 5060 (CUDA 12.8)**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-05-04T00:00:00Z
- **Completed:** 2026-05-04
- **Tasks:** 3
- **Files modified:** 5 (2 src, 2 tests, 1 docs created)

## Accomplishments

- Added `remap_dataset_paths()` to `clearml_utils.py`: downloads ClearML dataset by ID and remaps crop_path/page_path columns in-memory from original machine paths to cache root paths
- Added `--enqueue`, `--queue_name`, `--dataset_id` flags to `train_ctc.py`; `execute_remotely()` is called after `task.connect()` per RESEARCH.md Pattern 3 ordering constraint; tags updated from "phase-3" to "phase-4" (+ "gpu" when enqueuing)
- Created `docs/clearml-agent-setup.md`: WSL2 setup guide covering prerequisites (NVIDIA driver >= 576.x), agent install (`clearml-agent==3.0.0`), `clearml.conf` with `extra_index_url` for CUDA 12.8 PyTorch wheel, Pitfall 2 warning against `apt install nvidia-cuda-toolkit`, start commands, enqueue workflow, smoke test, and version table

## Task Commits

Each task was committed atomically:

1. **Task 1: Add remap_dataset_paths to clearml_utils.py and test it** - `e5f54c6` (feat)
2. **Task 2: Add --enqueue and --dataset_id flags to train_ctc.py and test them** - `a7f4ab0` (feat)
3. **Task 3: Write docs/clearml-agent-setup.md for WSL2 GPU agent** - `e55ba77` (docs)

_Note: TDD tasks (1 and 2) had combined test+implementation commits_

## Files Created/Modified

- `src/clearml_utils.py` - Added `remap_dataset_paths(df, dataset_id)` function after `maybe_create_dataset`
- `src/train_ctc.py` - Added `--enqueue`, `--queue_name`, `--dataset_id` flags; remap_dataset_paths import; execute_remotely call after connect; tags updated to phase-4
- `tests/test_clearml_utils.py` - Added 3 tests: path replacement, immutability, Dataset.get call
- `tests/test_train_ctc.py` - Added 5 tests: parser defaults, connect-before-execute_remotely ordering, gpu tag, remap call, backward compat
- `docs/clearml-agent-setup.md` - New file: WSL2 agent setup guide with CUDA 12.8 requirements

## Decisions Made

- **patch `src.train_ctc.init_task` for ordering test, not `src.train_ctc.Task`**: Patching `src.train_ctc.Task` does not control the task object returned by `init_task()` because `Task.init()` is called inside `clearml_utils.init_task`, not in `train_ctc` directly. Patching `init_task` gives direct control over the mock task object.
- **remap placed before output_dir.mkdir and task.connect**: Ensures remapped paths are present when hyperparams are connected to ClearML, and before any filesystem operations on the output dir. Per RESEARCH.md Pattern 3, connect must come before execute_remotely.
- **No structural changes required**: All additions are additive to existing functions/files; backward compatibility maintained throughout.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_enqueue_calls_execute_remotely_after_connect patching target**
- **Found during:** Task 2 (test implementation)
- **Issue:** Plan specified `@patch("src.train_ctc.Task")` with `mock_task_cls.init.return_value` to get task object. This approach doesn't work: `Task.init()` is called inside `clearml_utils.init_task()` (not in `train_ctc`), so patching `src.train_ctc.Task` doesn't intercept the task object.
- **Fix:** Changed to `@patch("src.train_ctc.init_task")` with `mock_init_task.return_value` as the mock task. This directly controls the task object returned to `main()`.
- **Files modified:** tests/test_train_ctc.py
- **Verification:** Test passes, call order assertion (connect < execute_remotely) verified
- **Committed in:** a7f4ab0 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug in plan's test implementation guidance)
**Impact on plan:** The fix is equivalent to what the plan intended — verifying call ordering. No scope creep.

## Issues Encountered

- The worktree lacked Plan 01's changes (branch was based on commit before Plan 01). Merged master (`08d8b41`) into the worktree branch via fast-forward before starting. This is a normal parallel agent setup concern.
- `test_prepare_data_end_to_end_on_synthetic_pdf` fails intermittently when run in the full test suite alongside other tests (isolation issue, not caused by this plan's changes — passes when run alone). Documented as pre-existing flaky test.

## User Setup Required

None — all changes are code-only. The WSL2 agent setup requires manual steps documented in `docs/clearml-agent-setup.md` but no automated configuration is needed for local dev.

## Next Phase Readiness

- `train_ctc.py` is ready to dispatch training jobs to ClearML GPU queue
- WSL2 agent can be set up following `docs/clearml-agent-setup.md`
- After WSL2 agent setup: run `uv run python -m src.train_ctc --manifest data/manifest.csv --output_dir outputs/model --enqueue --queue_name gpu --dataset_id <id>` to start a GPU training job
- Phase 4 complete: both augmentation (Plan 01) and remote GPU training (Plan 02) are implemented

---
*Phase: 04-data-augmentation-and-gpu-training-via-clearml-agent*
*Completed: 2026-05-04*
