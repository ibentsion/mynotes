---
quick_id: 260506-klj
status: complete
date: 2026-05-06
commit: d9f5d7d
---

# Quick Task 260506-klj: Defer heavy imports past execute_remotely

## What was done

Moved `import torch`, `from torch.utils.data import DataLoader`, and
`from src.ctc_utils import (...)` from module level to inside `main()`,
immediately after the `execute_remotely` block.

`import pandas as pd` stays at module level — it is used before
`execute_remotely` to read and validate the manifest.

## Why

The ClearML agent re-runs the script from the top on the agent machine.
If torch/ctc_utils are at module level, Python fails with
`ModuleNotFoundError: No module named 'torch'` before `main()` is even
called — even if those packages are listed in `requirements.txt` and will
be installed. Deferring them past `execute_remotely` means the agent
only needs clearml + pandas to get through the early setup, and loads
the heavy deps on its second pass when its venv is fully configured.

## Files changed

- `src/train_ctc.py` — deferred imports; ruff I001 fixed with `--fix`
