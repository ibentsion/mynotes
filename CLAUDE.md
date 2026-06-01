<!-- GSD:project-start source:PROJECT.md -->
## Project

**Hebrew Handwriting OCR Pipeline**

A local Python MVP pipeline for personal Hebrew handwritten OCR, built around a human-in-the-loop workflow. It converts scanned PDF pages into region crops, flags suspicious segmentations for manual review, trains a CRNN+CTC baseline model on minimal labeled data, and logs all data versions, metrics, and artifacts to ClearML.

**Core Value:** A reviewable, labeled dataset of personal Hebrew handwriting that a baseline OCR model can train on — getting the data pipeline and annotation workflow right matters more than model accuracy at MVP.

### Constraints

- **Runtime**: Python 3.13; GPU training via ClearML agent (queue: ofek, RTX 5060, WSL2) — CPU fallback works but is not a design constraint
- **Stack**: pdf2image + Poppler, OpenCV, PyTorch, Streamlit, ClearML — no additional heavy dependencies
- **Data**: Personal Hebrew notes only; privacy-sensitive — stays local
- **Reproducibility**: Git commit, package versions, and all configs stored in ClearML per run
- **Modularity**: Scripts are independent CLI tools, not a monolithic app — easy to extend or replace individual steps
- **Poppler**: Required system dependency for pdf2image on Linux
<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

## Training Workflow

### Pipeline overview

```
synthetic data → pretrain (train-ctc --mode pretrain)
                     └─ checkpoint_pretrain.pt (ClearML artifact)
                           └─ finetune (train-ctc --mode finetune --pretrain_checkpoint_path <task_id>)
                                  └─ checkpoint.pt → evaluate
```

### Config files

- `config/pretrain.yaml` — pretrain hyperparams, synthetic dataset ID, queue
- `config/finetune.yaml` — finetune hyperparams, real dataset ID, pretrain checkpoint ref, queue

All `train-ctc` flags have config equivalents. Config values become parser defaults;
CLI flags override them.

### Pretrain (synthetic data)

```bash
# Local CPU run
uv run train-ctc --mode pretrain --manifest outputs/synthetic/manifest.csv

# Enqueue to GPU agent (reads dataset_id and queue_name from config/pretrain.yaml)
uv run train-ctc --mode pretrain --enqueue
```

Produces `outputs/model/checkpoint_pretrain.pt` and uploads it as ClearML artifact
`checkpoint_pretrain` on the task. Copy the ClearML **task ID** from the results URL —
you'll need it for finetune.

### Finetune

```bash
# From scratch (no pretrain checkpoint)
uv run train-ctc --mode finetune --enqueue

# From a pretrain checkpoint — pass ClearML task ID or local .pt path
uv run train-ctc --mode finetune --enqueue \
  --pretrain_checkpoint_path <clearml_pretrain_task_id>
```

**Preferred: set in config** so you don't need to pass it every time. Uncomment and fill
`config/finetune.yaml`:
```yaml
pretrain_checkpoint_path: <clearml_pretrain_task_id>   # or outputs/model/checkpoint_pretrain.pt
```

When a task ID is given, the agent fetches the `checkpoint_pretrain` artifact directly
from ClearML — no manual download needed. A local `.pt` path works for local runs only.

### Evaluate

```bash
# Runs greedy decode on val split, writes outputs/model/eval_report.csv
uv run evaluate --manifest data/manifest.csv
```

Reads `outputs/model/checkpoint.pt` and `outputs/model/charset.json` by default.

### HPO → training flow

After HPO completes, best params are written back to `config/pretrain.yaml` or
`config/finetune.yaml` automatically (the `update_config` call in `tune.py`). Re-run
training immediately after to use them — no manual config editing needed.

```bash
# After pretrain HPO finishes → train with best pretrain params
uv run train-ctc --mode pretrain --enqueue

# After finetune HPO finishes → train with best finetune params
uv run train-ctc --mode finetune --enqueue
```

## HPO Workflow

### Terminology
- **Trial**: one training run with one hyperparameter set (epochs = 20–50 per trial)
- **n_trials**: total completed trials the study should reach (resume continues toward this target)
- SQLite is written after every `trial.report(val_cer, epoch)` call AND on trial completion — safe to abort mid-trial; in-flight trial stays `RUNNING` in DB, all prior completed trials are preserved

### Param keys by mode
- **finetune**: `lr`, `batch_size`, `epochs`, `rnn_hidden`, `num_layers`, `aug_copies`, `rotation_max`, `brightness_delta`, `noise_sigma`
- **pretrain**: `pretrain_lr`, `pretrain_epochs`, `batch_size`, `rnn_hidden`, `num_layers`

### Running HPO

```bash
# Finetune — local run with persistent SQLite for resume + visualization
uv run tune-hpo --manifest data/manifest.csv --n_trials 20 --mode finetune \
  --storage outputs/hpo_finetune.db

# Pretrain — local run
uv run tune-hpo --manifest data/manifest.csv --n_trials 20 --mode pretrain \
  --storage outputs/hpo_pretrain.db

# Enqueue to GPU agent (queue: ofek) — entire sweep runs remotely
uv run tune-hpo --dataset_id <clearml_dataset_id> --n_trials 20 --mode finetune \
  --storage outputs/hpo_finetune.db --enqueue --queue_name ofek
```

On remote runs: the SQLite is created on the agent machine and uploaded as ClearML artifact
`optuna_study_db` when the sweep completes. Epoch-level intermediate values are stored
in SQLite throughout the run (via `trial.report`) — visible in optuna-dashboard even for
aborted trials.

### Resuming after abort

Re-run with the same `--storage` and `--n_trials`. The sweep counts existing completed
trials and runs only the remaining ones:

```bash
# Resumes: e.g. 7 done → runs 13 more to reach 20
uv run tune-hpo --manifest data/manifest.csv --n_trials 20 --mode finetune \
  --storage outputs/hpo_finetune.db
# Output: "Resuming study 'hpo_finetune': 7 previous trials completed, running 13 more"
```

For remote runs that aborted: download the SQLite artifact first, then resume locally
or download and inspect:
```bash
uv run tune-hpo-inspect --task_id <clearml_task_id> --mode finetune
```

### Inspecting results

```bash
# Inspect local DB — trial counts, best params, fANOVA importances
uv run tune-hpo-inspect --storage outputs/hpo_finetune.db --mode finetune
uv run tune-hpo-inspect --storage outputs/hpo_pretrain.db --mode pretrain

# Download from ClearML task and inspect
uv run tune-hpo-inspect --task_id <clearml_task_id> --mode finetune

# Launch interactive optuna-dashboard (requires optuna-dashboard in dev deps)
uv run tune-hpo-inspect --storage outputs/hpo_finetune.db --mode finetune --dashboard
# Or directly:
optuna-dashboard sqlite:///outputs/hpo_finetune.db
```

fANOVA importances require ≥2 completed trials. Both pretrain and finetune modes are
fully supported — specify `--mode` to select the correct param key list.

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
