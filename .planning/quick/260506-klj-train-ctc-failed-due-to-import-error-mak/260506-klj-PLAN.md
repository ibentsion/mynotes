---
phase: quick
plan: 260506-klj
type: execute
wave: 1
depends_on: []
files_modified:
  - src/train_ctc.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "train_ctc.py starts without importing torch, pandas, or ctc_utils at module level"
    - "When --enqueue is set, task.execute_remotely() is called before any heavy import executes on the agent"
    - "All existing tests pass after the refactor"
  artifacts:
    - path: "src/train_ctc.py"
      provides: "ClearML-agent-safe training script"
      contains: "import torch"
  key_links:
    - from: "module level"
      to: "main() post-execute_remotely block"
      via: "deferred import of torch, pandas, ctc_utils"
      pattern: "import torch"
---

<objective>
Move heavy imports (torch, pandas, src.ctc_utils) inside main() so they execute only after
task.execute_remotely() exits the local process. The ClearML agent re-runs the script from the
top; if those imports are at module level the agent fails before main() runs when the packages
are missing from its venv.

Purpose: Fix the remote import error that crashes ClearML agent execution.
Output: src/train_ctc.py with lazy heavy imports; all tests pass.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/train_ctc.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Defer heavy imports to after execute_remotely</name>
  <files>src/train_ctc.py</files>
  <action>
Restructure imports so only stdlib and clearml remain at module level.

Module-level imports to KEEP (unchanged):
  - argparse, sys, pathlib.Path  (stdlib)
  - from clearml import Task  # noqa: F401  (test patchability — established pattern from STATE.md)
  - from src.clearml_utils import init_task, remap_dataset_paths, upload_file_artifact
    (clearml_utils is lightweight; init_task is called BEFORE execute_remotely so it must be importable early)

Module-level imports to REMOVE:
  - import pandas as pd
  - import torch
  - from torch.utils.data import DataLoader
  - from src.ctc_utils import (CRNN, AugmentTransform, CropDataset, build_charset,
      build_half_page_units, cer, crnn_collate, greedy_decode, predict_single,
      resolve_device, save_charset, split_units)

Inside main(), add a deferred import block IMMEDIATELY AFTER the execute_remotely block
(i.e., after `if args.enqueue: task.execute_remotely(...)`). The block runs on both the
local path (enqueue=False) and the agent path (after execute_remotely returns on the agent):

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.ctc_utils import (
        CRNN,
        AugmentTransform,
        CropDataset,
        build_charset,
        build_half_page_units,
        cer,
        crnn_collate,
        greedy_decode,
        predict_single,
        resolve_device,
        save_charset,
        split_units,
    )

No other logic changes — keep all existing comments, line numbers for non-import code
unchanged as much as possible.

Note: the `AugmentTransform | None` type annotation on line 116 is inside main(), so
it resolves correctly once AugmentTransform is imported above.
  </action>
  <verify>
    <automated>cd /home/ido/git/mynotes && python -c "import src.train_ctc" && echo "module-level import OK"</automated>
  </verify>
  <done>
    - python -c "import src.train_ctc" succeeds with no torch/pandas import at module load time
    - grep confirms no top-level "import torch" or "import pandas" outside the function body
    - The deferred import block appears inside main() after the execute_remotely block
  </done>
</task>

<task type="auto">
  <name>Task 2: Verify tests still pass</name>
  <files>src/train_ctc.py</files>
  <action>
Run the existing train_ctc test suite to confirm the refactor did not break any test.
No code changes in this task — verification only.
  </action>
  <verify>
    <automated>cd /home/ido/git/mynotes && python -m pytest tests/ -q -x 2>&1 | tail -20</automated>
  </verify>
  <done>All tests pass (0 failures, 0 errors). The patch mock on src.train_ctc.Task still works
because Task remains at module level.</done>
</task>

</tasks>

<verification>
After both tasks:
1. `python -c "import src.train_ctc"` prints nothing and exits 0
2. `rg "^import torch|^import pandas|^from torch|^from src.ctc_utils" src/train_ctc.py` returns no matches
3. `python -m pytest tests/ -q` passes with 0 failures
</verification>

<success_criteria>
- No heavy imports at module level in train_ctc.py
- ClearML agent can import the script before torch/pandas are available
- All existing tests pass unchanged
</success_criteria>

<output>
After completion, create `.planning/quick/260506-klj-train-ctc-failed-due-to-import-error-mak/260506-klj-SUMMARY.md`
</output>
