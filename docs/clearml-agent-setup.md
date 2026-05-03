# ClearML Agent Setup — WSL2 GPU (RTX 5060 / Blackwell)

Guide for setting up a ClearML agent in WSL2 to run GPU training jobs dispatched
by `train_ctc.py --enqueue`. Targets the Windows RTX 5060 host (sm_120 / Blackwell,
requires CUDA 12.8 wheel).

---

## Section 1: Prerequisites

- **Windows NVIDIA GeForce driver >= 576.x** (Game Ready driver; supports CUDA 12.8 / sm_120
  for RTX 5060). Download from nvidia.com/drivers.
- **WSL2 Ubuntu installed and working**: run `wsl --install Ubuntu` in PowerShell if not present.
- **git installed in WSL2**: `sudo apt-get install git`
- **Python 3.x available in WSL2** (any version >= 3.9 is fine; the task venv is created fresh
  by the agent — it does not need to match the dev machine Python version).

---

## Section 2: Install ClearML Agent in WSL2

Install as a system-level package in WSL2 — **not** inside the project venv:

```bash
pip install clearml-agent==3.0.0
```

The agent and the project share the same ClearML server but use separate Python environments.

---

## Section 3: Configure the Agent

Run the setup wizard to store API credentials:

```bash
clearml-agent init
```

The wizard prompts for your ClearML server credentials (same server and credentials used for
local dev). After the wizard completes, edit `~/clearml.conf` to add the agent configuration
block. The critical addition is the `extra_index_url` for the CUDA 12.8 PyTorch wheel — this
is required for RTX 5060 (Blackwell sm_120) support.

Add or update the `agent` section in `~/clearml.conf`:

```hocon
agent {
    git_user: ""          # leave blank to use SSH key
    git_pass: ""
    worker_id: "wsl2-gpu-worker"

    package_manager {
        type: pip
        # GPU torch wheel index for CUDA 12.8 (RTX 5060 / Blackwell sm_120)
        extra_index_url: ["https://download.pytorch.org/whl/cu128"]
        system_site_packages: false
    }
}
```

The `extra_index_url` ensures the agent installs `torch==2.11.0+cu128` (CUDA wheel) in the
task venv rather than the CPU wheel from PyPI. The project's `pyproject.toml` is unchanged
and continues to use the CPU wheel for local dev.

---

## Section 4: CUDA Note — Do NOT Run `apt install nvidia-cuda-toolkit`

**Do NOT run:**
```bash
apt install nvidia-cuda-toolkit   # WRONG — breaks WSL2 CUDA
```

**Why:** The Windows NVIDIA driver injects `libcuda.so` into WSL2 automatically. Installing
the CUDA toolkit from Ubuntu's apt repos can overwrite this driver stub and break CUDA access.

Instead, verify CUDA is available via `nvidia-smi` (should display the RTX 5060). If
`nvidia-smi` works but `torch.cuda.is_available()` returns False after setup, the toolkit
was likely installed from apt — remove it and the driver stub will be restored on WSL2 restart.

```bash
nvidia-smi   # should show RTX 5060 with driver version
```

---

## Section 5: Start the Agent

**Foreground mode** (recommended for first run — shows live logs):
```bash
clearml-agent daemon --queue gpu --gpus 0 --foreground
```

**Background mode** (normal operation):
```bash
clearml-agent daemon --detached --queue gpu --gpus 0
```

The agent polls the `gpu` queue and executes tasks in isolated Python venvs. The queue
is implicitly created on first daemon startup if it does not exist. Alternatively,
create it via the ClearML web UI: Settings > Workers & Queues > New Queue.

---

## Section 6: Enqueue a Training Job

**Important:** Commit Phase 4 changes before enqueueing. The ClearML agent clones the
git repo at the commit captured by `Task.init()`. Uncommitted changes are stored as a diff
and re-applied — committing first ensures clean reproducibility (RESEARCH.md Pitfall 6).

From the **local dev machine** (not WSL2), run:

```bash
uv run python -m src.train_ctc \
  --manifest data/manifest.csv \
  --output_dir outputs/model \
  --enqueue \
  --queue_name gpu \
  --dataset_id <your-dataset-id>
```

The `--dataset_id` value is shown in the ClearML web UI under the Datasets tab. The
script creates the ClearML task, connects hyperparameters, enqueues to the `gpu` queue,
and exits. The WSL2 agent picks up the task, clones the repo, installs deps (including the
CUDA torch wheel), and runs training on the RTX 5060.

---

## Section 7: Smoke Test

Before running real training, verify agent and SDK compatibility with a minimal task:

```python
from clearml import Task
t = Task.init(project_name="test", task_name="smoke")
t.execute_remotely(queue_name="gpu")
```

If the task appears in the ClearML web UI with status `queued` and the agent picks it up
and marks it `completed`, the setup is correct. If the agent fails to pick it up, check:
- Agent is running (`clearml-agent daemon --queue gpu --foreground`)
- Credentials match between local dev and WSL2 (`~/.clearml.conf` on both machines)
- `clearml-agent --version` is compatible with the SDK version (see version table below)

---

## Section 8: Version Summary

| Component          | Version       | Notes                                             |
|--------------------|---------------|---------------------------------------------------|
| clearml-agent      | 3.0.0         | Install in WSL2 (not project venv)                |
| clearml SDK        | 2.1.5         | Pinned in pyproject.toml                          |
| PyTorch (dev)      | 2.11.0        | CPU wheel — pyproject.toml unchanged              |
| PyTorch (agent)    | 2.11.0+cu128  | CUDA wheel via extra_index_url in clearml.conf    |
| CUDA               | 12.8          | Required for RTX 5060 (Blackwell sm_120)          |
| Python (agent)     | 3.9+          | Agent-side; task venv created fresh per run       |
