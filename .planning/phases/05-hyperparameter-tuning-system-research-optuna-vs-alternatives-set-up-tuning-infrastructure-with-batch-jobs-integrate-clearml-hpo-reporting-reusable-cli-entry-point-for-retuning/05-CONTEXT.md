# Phase 5: Hyperparameter Tuning System - Context

**Gathered:** 2026-05-05
**Status:** Ready for planning

<domain>
## Phase Boundary

A standalone hyperparameter tuning pipeline that:
1. Sweeps training, architecture, and augmentation parameters across multiple trials
2. Runs each trial as a ClearML task (GPU queue via `--enqueue` from Phase 4)
3. Surfaces the best config via printed summary, ClearML report, and `outputs/best_params.json`
4. Lives in a standalone `src/tune.py` CLI — re-runnable on demand as the dataset grows

Out of scope: new model architectures, post-correction, active learning, changes to
`evaluate.py`, `review_app.py`, or the data pipeline.

</domain>

<decisions>
## Implementation Decisions

### Search Space

- **D-01:** Tune all three parameter groups:
  - **Training:** `lr`, `batch_size`, `epochs`
  - **Architecture:** `rnn_hidden` ∈ {128, 256, 512}, `num_layers` ∈ {1, 2}
  - **Augmentation:** `aug_copies`, `rotation_max`, `brightness_delta`, `noise_sigma`
- **D-02:** Architecture params (`rnn_hidden`, `num_layers`) are currently hardcoded in
  `CRNN.__init__` in `ctc_utils.py`. They must be exposed as CLI args to `train_ctc.py`
  and passed through to `CRNN()` before the tuner can vary them. This is a prerequisite
  task within Phase 5.

### Tuner Selection

- **D-03:** The researcher agent must evaluate **Optuna (standalone)** vs **ClearML
  HyperParameterOptimizer** and recommend one. Key comparison axes:
  - Setup complexity given the existing `train_ctc.py` + ClearML task pattern
  - GPU queue trial dispatching (both need to work with `--enqueue`)
  - Pruning support (kill bad trials early by CER after a few epochs)
  - ClearML UI integration for comparing trial results
  - Maintenance burden (ClearML HPO is version-sensitive; Optuna is stable)
- **D-04:** Implement **one backend only** — the researcher's recommendation. No flag-selectable
  dual backends. Clean, single approach.

### Execution Mode

- **D-05:** Trials run via **GPU queue** (`--enqueue` flag from Phase 4). The ClearML agent
  on WSL2/RTX 5060 handles execution. Local CPU fallback is manual (omit `--enqueue`).
- **D-06:** Each trial is a separate ClearML task, tagged with `"phase-5"` and `"hpo-trial"`.
  This gives per-trial metric visibility in the ClearML UI.

### Trial Budget & Pruning

- **D-07:** `--n_trials N` CLI flag, default 20. Configurable per run — no hardcoded budget.
- **D-08:** Early stopping / pruning enabled: kill trials where CER is not improving after
  a configurable number of epochs (e.g. 5 epochs). Reduces wasted GPU time on clearly bad
  configurations. Use the chosen backend's pruning mechanism (Optuna `MedianPruner` or
  ClearML HPO early-stop policy).

### Results & Best-Config UX

- **D-09:** On completion, `tune.py` does **both**:
  1. Prints best params + best val CER to stdout
  2. Writes `outputs/best_params.json` with the winning config
  3. Logs a ClearML summary report: scalar plots (CER per trial), table of all trial params + CER
- **D-10:** `train_ctc.py` gets a `--params best_params.json` flag that loads and applies a
  saved params file. This makes re-using the tuner's output zero-friction.
- **D-11:** `outputs/best_params.json` is gitignored (machine/dataset-specific artifact).

### Re-tuning Entry Point

- **D-12:** Standalone `src/tune.py` — same CLI style as `train_ctc.py`:
  ```
  python -m src.tune --manifest data/manifest.csv --n_trials 20 [--enqueue] [--queue_name gpu]
  ```
  User re-runs this whenever they want to retune (e.g. after dataset grows significantly).

### Inherited Patterns (unchanged)

- `init_task()` called early; `task.connect()` for hyperparameter logging
- Module-level ClearML imports for test patchability
- `argparse` + `task.connect(vars(args))` pattern (TRAN-07)
- `--enqueue` / `--queue_name` flags (Phase 4 D-07, D-08)

### Claude's Discretion

- Exact LR range and distribution (log-uniform recommended: 1e-4 to 1e-2)
- Exact batch_size values to sweep (suggested: 4, 8, 16)
- Exact epochs range (suggested: 20–50, or fixed at 30 with early stopping doing the work)
- Augmentation sweep ranges (within Phase 4's conservative parameter guidelines)
- Number of pruning warm-up epochs before pruning kicks in (suggested: 5)
- ClearML task name for the tuner orchestrator task (suggested: `hpo_sweep`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Training Code
- `src/train_ctc.py` — script to extend with `--params` flag and arch CLI args
- `src/ctc_utils.py` — `CRNN.__init__` (needs `rnn_hidden`, `num_layers` params), `CropDataset`
- `src/clearml_utils.py` — `init_task()`, `remap_dataset_paths()` patterns to reuse

### Phase 4 Context (prereqs)
- `.planning/phases/04-data-augmentation-and-gpu-training-via-clearml-agent/04-CONTEXT.md` —
  D-07 (`--enqueue`), D-08 (task naming/tags), D-09/D-10 (`--dataset_id` + path remapping)

### Project Constraints
- `.planning/PROJECT.md` §Constraints — no new heavy deps; stack: PyTorch, ClearML, argparse
- `.planning/PROJECT.md` §Context — GPU target is RTX 5060 via ClearML agent on WSL2

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/train_ctc.py::build_args()` — argparse setup to extend with `--params`, `--rnn_hidden`, `--num_layers`
- `src/ctc_utils.py::CRNN` — `__init__` needs `rnn_hidden` and `num_layers` params exposed
- `src/clearml_utils.py::init_task()` — reuse for tuner orchestrator task
- Phase 4 `--enqueue` / `--queue_name` flags — tune.py dispatches each trial the same way

### Established Patterns
- `argparse` + `task.connect(vars(args))` for ClearML hyperparameter tracking (TRAN-07)
- Separate ClearML task per training run — tune.py should spawn N separate tasks
- `outputs/` directory for model artifacts — `best_params.json` goes here

### Integration Points
- `CRNN.__init__` is where `rnn_hidden` and `num_layers` must be plumbed through
- `train_ctc.py::main()` is where `--params` file loading applies (before argparse defaults)
- `tune.py` orchestrates by calling into `train_ctc` logic or subprocess — researcher decides

</code_context>

<specifics>
## Specific Ideas

- User explicitly wants to **understand the difference** between Optuna and ClearML HPO before
  deciding — researcher must provide a clear, concrete comparison (not just a summary)
- Re-tuning should feel as natural as re-training: same invocation style, same ClearML visibility
- `best_params.json` + `--params` flag creates a round-trip: tune → save → train with winner

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 05-hyperparameter-tuning-system*
*Context gathered: 2026-05-05*
