---
status: resolved
trigger: "Fix failing tests, enforce <5s per test, fix ClearML GPU training"
created: 2026-05-07T00:00:00Z
updated: 2026-05-07T01:30:00Z
---

## Current Focus

hypothesis: CONFIRMED — all three root causes identified
test: Applying fixes
expecting: All tests pass in <5s
next_action: Fix autouse fixture to clean CLEARML_PROC_MASTER_ID; fix slow tests; GPU training already fixed in prior commit

## Symptoms

expected: All pytest unit tests pass in <5s each; ClearML remote agent trains on GPU without crashing
actual: Some tests fail (unknown which); some tests take >5s; ClearML GPU training fails
errors: "AttributeError: 'NoneType' object has no attribute 'get_logger'"; "TypeError: 'StubObject' object is not subscriptable"
reproduction: pytest -q
started: GPU training was broken and fixes attempted but did not resolve

## Eliminated

- hypothesis: GPU training broken by missing execute_remotely fallthrough
  evidence: Already fixed in commit 8c72aff; confirmed by agent reaching run_training() and hitting min_labeled guard
  timestamp: 2026-05-07

- hypothesis: Task.current_task() broken in offline mode
  evidence: Works fine in isolation; only fails when CLEARML_PROC_MASTER_ID is in env
  timestamp: 2026-05-07

## Evidence

- timestamp: 2026-05-07
  checked: pytest full run output
  found: 6 failures: test_prepare_data_end_to_end_on_synthetic_pdf (StubObject subscript), 5x train_ctc subprocess tests (Task.current_task()=None)
  implication: Subprocess tests are contaminated by parent process state

- timestamp: 2026-05-07
  checked: CLEARML_PROC_MASTER_ID env var behavior
  found: task.close() writes CLEARML_PROC_MASTER_ID to os.environ; when inherited by subprocess, Task.init() returns StubObject and current_task()=None
  implication: autouse _close_clearml_task_after_test fixture and explicit task.close() calls leave CLEARML_PROC_MASTER_ID in os.environ; subsequent _run_cli subprocesses inherit it and ClearML degrades to stub mode

- timestamp: 2026-05-07
  checked: Slow test timings
  found: run_training tests (30-47s each), subprocess tests (15-20s each), in-process tests with real Task.init (15-16s each)
  implication: ClearML Task.init() offline mode takes several seconds; actual training takes 15-45s; need to either mock init_task or use smaller models + fewer epochs

## Resolution

root_cause: |
  Issue 1 (failures): ClearML sets CLEARML_PROC_MASTER_ID in os.environ when task.close() is called.
  In-process tests that run main() with @patch("src.train_ctc.Task") still call the real Task.init()
  via clearml_utils.init_task(). When those tests close the task, CLEARML_PROC_MASTER_ID pollutes
  os.environ. Subsequent subprocess tests (via _run_cli) inherit this env var, causing Task.init()
  to return a StubObject and Task.current_task() to return None.
  
  Issue 2 (slow tests): Tests use real CRNN training (even with 12 crops, 1 epoch = 15-47s).
  Also the real ClearML Task.init() adds startup overhead even when training is mocked/skipped.
  
  Issue 3 (GPU training): Already fixed in commit 8c72aff. Root cause was (a) return 0 after
  execute_remotely killing the agent and (b) manifest not available on agent. But training then
  failed with "only 0 labeled crops" — the dataset manifest has 0 labeled rows (they are
  "unlabeled" status). This is a data issue, not a code issue.

fix:
  1. Fix autouse fixture: after task.close(), pop CLEARML_PROC_MASTER_ID from os.environ
  2. Fix run_training tests that call task.close() explicitly: pop CLEARML_PROC_MASTER_ID
  3. Fix slow tests: patch init_task for tests that don't need actual ClearML, use smaller model
  4. GPU training: needs labeled data with status="labeled" in the manifest

verification: |
  151 tests pass in ~22s total; slowest individual test 2.5s.
  Ruff reports no lint errors on modified files.
  Committed as f44f27a.
files_changed:
  - tests/conftest.py
  - tests/test_evaluate.py
  - tests/test_prepare_data.py
  - tests/test_train_ctc.py
