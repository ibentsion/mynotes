---
phase: quick-260507-plf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/create_standalone_dataset.py
autonomous: true
requirements:
  - QUICK-PLF-01
must_haves:
  truths:
    - "A new standalone ClearML dataset v3 exists with no parent_datasets"
    - "Dataset contains manifest.csv at root with 979 labeled rows"
    - "Dataset contains all crop PNG files under crops/ prefix"
    - "Dataset.get_local_copy() returns a path containing both manifest.csv and crops/*.png"
    - "Training task is enqueued to queue 'ofek' against the new dataset ID"
    - "Polling loop runs until task reaches completed/failed/aborted state"
    - "If task fails, last 50 lines of console log are printed"
  artifacts:
    - path: "scripts/create_standalone_dataset.py"
      provides: "Standalone CLI script that creates the v3 dataset, enqueues training, and polls status"
  key_links:
    - from: "scripts/create_standalone_dataset.py"
      to: "~/.clearml/cache/storage_manager/datasets/ds_02210b47a0534666b0462a175bf2af9d/manifest.csv"
      via: "shutil.copy into tempdir"
      pattern: "ds_02210b47a0534666b0462a175bf2af9d/manifest.csv"
    - from: "scripts/create_standalone_dataset.py"
      to: "data/crops/"
      via: "Dataset.add_files(..., dataset_path='crops')"
      pattern: "add_files.*dataset_path=.crops."
    - from: "scripts/create_standalone_dataset.py"
      to: "src.train_ctc CLI"
      via: "subprocess.run with --enqueue --queue_name ofek --dataset_id <new_id>"
      pattern: "src\\.train_ctc.*--enqueue"
---

<objective>
Resolve the persistent parent-child merge bug in ClearML datasets by creating a brand new standalone dataset v3 (no parent) under project 'handwriting-hebrew-ocr', name 'data_prep'. The new dataset embeds the labeled manifest.csv (979 rows, sourced from child v1.0.2 cache) and all local data/crops/*.png files directly. After upload+finalize, verify get_local_copy() roundtrips, then enqueue the training task to queue 'ofek' with the new dataset ID and poll until completed or failed.

Purpose: Unblock GPU training that has been failing on dataset extraction errors due to the v1.0.0 parent's flat+prefixed file duplication bug. A fresh standalone dataset bypasses ClearML's parent-merge code path entirely.

Output: A new ClearML dataset ID printed to stdout, a remote training task ID printed and polled to terminal state.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@src/train_ctc.py
@src/clearml_utils.py

<interfaces>
<!-- Key interfaces the executor needs. Extracted from project memory + planning context. -->

ClearML Dataset SDK (clearml==1.x):
```python
from clearml import Dataset, Task

# Create standalone dataset (NO parent_datasets kwarg)
ds = Dataset.create(
    dataset_name='data_prep',
    dataset_project='handwriting-hebrew-ocr',
)

# Add files: dataset_path controls the in-dataset storage prefix
ds.add_files(path='/abs/path/to/manifest.csv')                  # stored as 'manifest.csv'
ds.add_files(path='data/crops', dataset_path='crops')           # stored as 'crops/<filename>.png'

ds.upload()
ds.finalize()
print(ds.id)  # 32-char hex

# Verify roundtrip
local = Dataset.get(dataset_id=ds.id).get_local_copy()
# local/manifest.csv and local/crops/*.png must exist
```

ClearML Task polling:
```python
t = Task.get_task(task_id=...)
t.get_status()           # returns: 'created', 'queued', 'in_progress', 'completed', 'failed', 'stopped', ...
t.get_reported_console_output(number_of_reports=50)  # list[str] last N console lines
```

Training CLI invocation (from src/train_ctc.py):
```bash
uv run python -m src.train_ctc --enqueue --queue_name ofek --dataset_id <new_id>
```
This entry prints the spawned remote task ID to stdout — capture it via stdout parsing or by reading the most recent task in the project after invocation.

Known good child manifest path (from key_context):
~/.clearml/cache/storage_manager/datasets/ds_02210b47a0534666b0462a175bf2af9d/manifest.csv
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create scripts/create_standalone_dataset.py — build dataset, verify, enqueue, poll</name>
  <files>scripts/create_standalone_dataset.py</files>
  <action>
Create a single self-contained CLI script at scripts/create_standalone_dataset.py that performs the full sequence end-to-end. Use `set -euo pipefail`-equivalent strict failure behavior (raise on any error; do not swallow exceptions).

Script structure (single main() function, may use small helpers to stay under the 100-line/function limit):

1. **Build standalone dataset:**
   - `from clearml import Dataset, Task`
   - `ds = Dataset.create(dataset_name='data_prep', dataset_project='handwriting-hebrew-ocr')` — explicitly NO parent_datasets argument.
   - Resolve child manifest: `Path('~/.clearml/cache/storage_manager/datasets/ds_02210b47a0534666b0462a175bf2af9d/manifest.csv').expanduser()`. Assert it exists; fail fast with a clear message naming the missing path if not.
   - Copy manifest into a `tempfile.mkdtemp()` directory (avoids ClearML storing the absolute cache path as the in-dataset key) and call `ds.add_files(str(tmpdir / 'manifest.csv'))` so it lands at `manifest.csv` in the dataset root.
   - Resolve local crops dir: `Path('data/crops')`. Assert it exists and contains at least one `*.png`; fail fast otherwise.
   - `ds.add_files(path='data/crops', dataset_path='crops')` so files land at `crops/<name>.png`.
   - `ds.upload()` then `ds.finalize()`.
   - Capture and print: `Dataset ID: {ds.id}`.

2. **Roundtrip verification (before enqueueing training):**
   - `local = Path(Dataset.get(dataset_id=ds.id).get_local_copy())`
   - Assert `(local / 'manifest.csv').is_file()`.
   - Assert at least one PNG exists under `local / 'crops'` via `next((local / 'crops').glob('*.png'))` (allow StopIteration to surface as a clear AssertionError).
   - Print counts: number of pngs found and manifest line count.

3. **Git push gate:**
   - Run `subprocess.run(['git', 'push'], check=True)` — required so the ClearML agent can check out the exact commit (per project memory feedback_push_before_clearml.md). If there is nothing to push, `git push` will exit 0 with "Everything up-to-date"; that's fine.

4. **Enqueue training task:**
   - `subprocess.run(['uv', 'run', 'python', '-m', 'src.train_ctc', '--enqueue', '--queue_name', 'ofek', '--dataset_id', ds.id], check=True, capture_output=True, text=True)`.
   - Parse the spawned task ID from stdout. The train_ctc CLI prints lines such as `ClearML Task: created new task id=<32hex>` (search via regex `r'task id[=:\s]+([0-9a-f]{32})'`, case-insensitive). If no match, fall back to querying `Task.get_tasks(project_name='handwriting-hebrew-ocr')` sorted by creation, pick the most recent — but prefer stdout parsing.
   - Print: `Training task ID: {task_id}`.

5. **Poll loop:**
   - Use `Task.get_task(task_id=task_id)`.
   - Loop with `time.sleep(30)` between status checks.
   - Terminal states: `{'completed', 'failed', 'stopped', 'closed'}` — break when reached.
   - Print status transitions (only when status changes, to keep output clean): `[{elapsed_s}s] status={status}`.
   - Hard timeout: 7200 seconds (2 hours). On timeout, print warning and exit non-zero.

6. **Failure diagnostics:**
   - When loop exits with status in `{'failed', 'stopped'}`, fetch last 50 console lines via `task.get_reported_console_output(number_of_reports=50)` and print each line. Exit with code 1.
   - On `completed`, print success and exit 0.

Constraints:
- Use `uv run python` only when invoking external commands; the script itself runs as a normal Python module.
- No docstrings except a brief module-level header (script is operational, not a library).
- Absolute imports only.
- Each function ≤100 lines; split build/verify/enqueue/poll into helpers if needed.
- Use `tracing`-equivalent: plain `print()` is acceptable here (this is an ad-hoc operational script, not library code).
- Do NOT swallow exceptions. Let them propagate with full tracebacks.

After writing the script, run a syntax/import smoke check: `uv run python -c "import ast; ast.parse(open('scripts/create_standalone_dataset.py').read())"` to catch syntax errors before execution.
  </action>
  <verify>
    <automated>uv run python -c "import ast; ast.parse(open('scripts/create_standalone_dataset.py').read())" &amp;&amp; uv run ruff check scripts/create_standalone_dataset.py</automated>
  </verify>
  <done>
scripts/create_standalone_dataset.py exists, parses cleanly, passes ruff, and contains all five sections (build / verify / git push / enqueue / poll+diagnostics). Each function under 100 lines.
  </done>
</task>

<task type="auto">
  <name>Task 2: Execute the script — create v3 dataset, enqueue training, poll to terminal state</name>
  <files>(no source changes; runs scripts/create_standalone_dataset.py)</files>
  <action>
Run the script created in Task 1 end-to-end:

```bash
uv run python scripts/create_standalone_dataset.py
```

Expected runtime: dataset upload likely ~minutes (depends on crop count + bandwidth), polling up to 2 hours.

Capture and report from the script's stdout:
1. The new dataset ID (32-hex).
2. Roundtrip verification counts (manifest rows + png count).
3. The training task ID.
4. Final task status.
5. If failed: the last 50 console lines.

If the script exits 0 (completed): success — record the dataset ID and task ID for the SUMMARY.
If the script exits non-zero (failed/timeout): record the dataset ID, task ID, and failure console output for the SUMMARY. Do NOT retry inside this task — surface the failure for user decision.

Do not modify src/train_ctc.py or any production code in this task. The script is operational tooling; train_ctc remains untouched.
  </action>
  <verify>
    <automated>test -f scripts/create_standalone_dataset.py &amp;&amp; uv run python scripts/create_standalone_dataset.py</automated>
  </verify>
  <done>
The script ran to a terminal task state. Dataset ID, task ID, and final status are captured. On failure, last 50 console lines are captured for diagnosis. The new standalone dataset exists in ClearML and was successfully roundtripped via get_local_copy().
  </done>
</task>

</tasks>

<verification>
Phase-level checks after both tasks:
- `scripts/create_standalone_dataset.py` exists, lints clean.
- A new ClearML dataset under project 'handwriting-hebrew-ocr', name 'data_prep' exists with no parent (verify via ClearML web UI or `Dataset.get(dataset_id=<new_id>)._task.parent` is None/empty).
- The new dataset's `get_local_copy()` produces a directory containing `manifest.csv` and `crops/*.png`.
- A training task is visible in queue 'ofek' or has progressed to in_progress/completed/failed.
- Final task status is one of {completed, failed, stopped, closed} (not stuck in queued/in_progress).
</verification>

<success_criteria>
- New standalone ClearML dataset ID printed and recorded.
- Dataset roundtrip succeeds: manifest.csv at root + crops/*.png present after get_local_copy().
- Training task enqueued to queue 'ofek' with the new dataset ID.
- Polling reached a terminal state (completed/failed/stopped) or hit the 2h timeout with a clear warning.
- On task failure: last 50 console lines were retrieved and printed for downstream diagnosis.
- No edits to src/train_ctc.py or other production code; only scripts/create_standalone_dataset.py was added.
</success_criteria>

<output>
After completion, create `.planning/quick/260507-plf-create-a-standalone-clearml-dataset-v3-d/260507-plf-SUMMARY.md` recording:
- New dataset ID
- Training task ID
- Final task status
- If failed: pasted last 50 console lines
- Any deviations from the plan
</output>
