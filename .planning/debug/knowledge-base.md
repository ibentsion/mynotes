# GSD Debug Knowledge Base

Resolved debug sessions. Used by `gsd-debugger` to surface known-pattern hypotheses at the start of new investigations.

---

## failing-tests-and-gpu-training — CLEARML_PROC_MASTER_ID env var pollutes subprocess tests
- **Date:** 2026-05-07
- **Error patterns:** AttributeError NoneType get_logger, StubObject subscriptable, Task.current_task None, CLEARML_PROC_MASTER_ID, test_prepare_data, subprocess, pytest slow, ClearML offline, execute_remotely, GPU training, min_labeled
- **Root cause:** ClearML writes CLEARML_PROC_MASTER_ID to os.environ when task.close() is called. In-process tests that invoke any ClearML task path (even via @patch of Task but not init_task) leak this var; subprocesses spawned by _run_cli() inherit it, causing Task.init() to return StubObject and Task.current_task() to return None. Separately: real ClearML Task.init()/close() take 10-15s, real CRNN training takes 15-47s/epoch — both too slow for unit tests.
- **Fix:** (1) Add autouse pytest fixture in conftest.py: `os.environ.pop("CLEARML_PROC_MASTER_ID", None)` after yield. (2) Patch both `init_task` and `Task` in all in-process tests; convert subprocess _run_cli() tests to in-process main() calls. (3) Use `--rnn_hidden 128 --num_layers 1` CLI args for training-heavy tests to keep epoch time ~0.2s. (4) The test for train+evaluate must use default rnn_hidden to avoid checkpoint architecture mismatch.
- **Files changed:** tests/conftest.py, tests/test_evaluate.py, tests/test_prepare_data.py, tests/test_train_ctc.py
---

