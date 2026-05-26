# Phase 5: Hyperparameter Tuning System - Research

**Researched:** 2026-05-05
**Domain:** Hyperparameter optimization with Optuna + ClearML integration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Tune three parameter groups: training (lr, batch_size, epochs), architecture
  (rnn_hidden ∈ {128,256,512}, num_layers ∈ {1,2}), augmentation (aug_copies, rotation_max,
  brightness_delta, noise_sigma)
- **D-02:** Architecture params (rnn_hidden, num_layers) currently hardcoded in CRNN.__init__ —
  must be exposed as CLI args to train_ctc.py and passed to CRNN() before tuner can vary them
- **D-03:** Researcher must evaluate Optuna (standalone) vs ClearML HyperParameterOptimizer and
  recommend ONE. Key axes: setup complexity, GPU queue trial dispatching, pruning support,
  ClearML UI integration, maintenance burden
- **D-04:** Implement ONE backend only — no flag-selectable dual backends
- **D-05:** Trials run via GPU queue (--enqueue from Phase 4). Local CPU fallback = omit --enqueue
- **D-06:** Each trial is a separate ClearML task, tagged "phase-5" and "hpo-trial"
- **D-07:** --n_trials N CLI flag, default 20
- **D-08:** Early stopping/pruning: kill trials where CER not improving after configurable
  warm-up epochs. Use chosen backend's pruning mechanism
- **D-09:** On completion: print best params + val CER to stdout, write outputs/best_params.json,
  log ClearML summary report (scalar plots CER per trial, table of all trial params + CER)
- **D-10:** train_ctc.py gets --params best_params.json flag to load and apply saved params
- **D-11:** outputs/best_params.json is gitignored (machine/dataset-specific artifact)
- **D-12:** Standalone src/tune.py with CLI:
  `python -m src.tune --manifest data/manifest.csv --n_trials 20 [--enqueue] [--queue_name gpu]`
- Inherited: init_task() called early, task.connect() for hyperparameter logging, module-level
  ClearML imports for test patchability, argparse + task.connect(vars(args)) pattern (TRAN-07),
  --enqueue / --queue_name flags (Phase 4 D-07, D-08)

### Claude's Discretion

- Exact LR range and distribution (log-uniform recommended: 1e-4 to 1e-2)
- Exact batch_size values to sweep (suggested: 4, 8, 16)
- Exact epochs range (suggested: 20-50, or fixed at 30 with early stopping)
- Augmentation sweep ranges (within Phase 4's conservative parameter guidelines)
- Number of pruning warm-up epochs before pruning kicks in (suggested: 5)
- ClearML task name for the tuner orchestrator task (suggested: hpo_sweep)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

## Summary

Phase 5 adds a hyperparameter tuning system on top of the existing `train_ctc.py` / ClearML
training pipeline established in Phases 3–4. The core question is: which backend to use.

**Recommendation: Standalone Optuna (not ClearML HyperParameterOptimizer).**

ClearML's `HyperParameterOptimizer` is a compelling option — it has native Optuna support via
`OptimizerOptuna`, per-trial ClearML task cloning, and queue dispatch built-in. However, it
carries a critical prerequisite: the base task (train_ctc) must already exist as a registered
ClearML task (from a prior run) before HPO can begin. This creates a bootstrap problem for a
fresh environment. More importantly, `HyperParameterOptimizer` wraps Optuna rather than exposing
it directly: pruning requires the base task to report intermediate metrics via `trial.report()`
during training, which means `train_ctc.py` would need to be modified to accept and propagate an
Optuna `trial` object across process boundaries — architecturally awkward for a subprocess-per-
trial model. The HPO controller also runs a persistent process that polls for completed trials,
adding operational complexity with limited benefit for N=20 sequential trials.

Standalone Optuna keeps the architecture simple. The tune.py orchestrator creates a `study`,
calls `study.optimize()` with an `objective()` function that directly invokes the training logic
in-process (no subprocess), reports intermediate CER values via `trial.report()`, and prunes
with `trial.should_prune()`. After completion, `tune.py` creates its own ClearML orchestrator
task to log a comparison table and CER-per-trial scalar. Each trial's training still creates its
own ClearML task via `init_task()` — matching D-06. The --enqueue path dispatches that task to
the GPU queue with `task.execute_remotely()` and exits; the trial result is read back from the
ClearML server after the agent completes it. For the local (no --enqueue) path, training runs
in-process with Optuna pruning active.

**Primary recommendation:** Use standalone Optuna 4.8.0 with MedianPruner. tune.py creates its
own ClearML orchestrator task; each trial calls train_ctc logic directly or via execute_remotely.

---

## Backend Decision: Optuna vs ClearML HyperParameterOptimizer

| Axis | Optuna standalone | ClearML HyperParameterOptimizer |
|------|-------------------|----------------------------------|
| Setup complexity | Low — pip install optuna, write tune.py | High — base task must be pre-registered in ClearML server |
| Per-trial ClearML task | Manual: call init_task() inside objective() | Automatic: clones base task, injects params |
| GPU queue dispatch | Manual: task.execute_remotely() in objective() | Automatic: execution_queue= parameter |
| Optuna pruning | First-class: trial.report(), trial.should_prune() | Requires train_ctc.py to report intermediate metrics — cross-process friction |
| API stability | HIGH — Optuna 4.8.0 released 2026-03-16, mature | MEDIUM — ClearML automation module evolves frequently |
| ClearML UI result visibility | Per-trial tasks show metrics natively; orchestrator logs table | Native HPO dashboard, parallel-coordinate plots |
| N=20 trials sequential | study.optimize() — simple, blocking | Async poll loop — designed for parallel/concurrent workloads |
| Maintenance burden | Low — single dependency, stable API | Medium — tied to ClearML SDK release cadence |
| Cold-start requirement | None | train_ctc must have been run once to produce base_task_id |

**Verdict: Optuna standalone.**

The ClearML HPO approach excels for parallel multi-GPU workloads where the orchestrator manages
concurrency. For N=20 sequential trials on a single GPU queue, the overhead is not justified.
The cold-start requirement (base task must exist) is a concrete friction point for a
"re-runnable on demand" use case. Optuna's pruning integration is first-class with direct
trial.report() calls; ClearML HPO pruning requires metric surfacing across process/clone
boundaries. Optuna 4.8.0 is actively maintained (latest release: 2026-03-16).

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| optuna | 4.8.0 | Hyperparameter search + pruning | Actively maintained, best-in-class TPE sampler, first-class pruning API |
| clearml | 2.1.5 | Per-trial task logging, orchestrator report | Already installed; project standard |
| torch | 2.11.0 | Training (no change) | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | — | Write outputs/best_params.json | Final best-params serialization |
| argparse (stdlib) | — | CLI definition for tune.py | Matches project CLI convention |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Optuna | ClearML HyperParameterOptimizer + OptimizerOptuna | Better UI for parallel HPO; adds cold-start requirement and cross-process pruning friction |
| Optuna | Ray Tune | More powerful distributed HPO; heavy dependency, not in project stack |
| MedianPruner | HyperbandPruner | Hyperband is faster; MedianPruner is simpler and sufficient for N=20 |

**Installation:**
```bash
uv add optuna==4.8.0
```

**Version verification (confirmed):**
- optuna 4.8.0 — PyPI upload date 2026-03-16 (current stable)
- clearml 2.1.5 — already installed (latest is 2.1.6; upgrade optional, not required)

---

## Architecture Patterns

### Recommended Project Structure
```
src/
├── tune.py             # NEW: HPO orchestrator CLI
├── train_ctc.py        # MODIFIED: add --params, --rnn_hidden, --num_layers
├── ctc_utils.py        # MODIFIED: CRNN.__init__ accepts rnn_hidden, num_layers
└── clearml_utils.py    # UNCHANGED: init_task() reused by tune.py
outputs/
└── best_params.json    # Generated: gitignored
```

### Pattern 1: Optuna Objective with In-Process Training + ClearML Task per Trial

**What:** The `objective(trial)` function samples parameters, creates a ClearML task, runs
training, reports CER per epoch with `trial.report()`, and raises `optuna.TrialPruned` when
`trial.should_prune()` fires. When --enqueue is set, `task.execute_remotely()` dispatches to
the GPU queue; the objective then polls the task for completion and reads the final CER scalar.

**When to use:** Default path for all tune.py runs.

**Local (no --enqueue) pattern:**
```python
# Source: Optuna docs + ClearML patterns established in project
def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    epochs = trial.suggest_int("epochs", 20, 50)
    rnn_hidden = trial.suggest_categorical("rnn_hidden", [128, 256, 512])
    num_layers = trial.suggest_categorical("num_layers", [1, 2])
    aug_copies = trial.suggest_int("aug_copies", 0, 6)
    rotation_max = trial.suggest_float("rotation_max", 0.0, 15.0)
    brightness_delta = trial.suggest_float("brightness_delta", 0.0, 0.20)
    noise_sigma = trial.suggest_float("noise_sigma", 0.0, 0.05)

    trial_task = init_task(
        "handwriting-hebrew-ocr",
        f"hpo_trial_{trial.number}",
        tags=["phase-5", "hpo-trial"],
    )
    trial_task.connect({
        "lr": lr, "batch_size": batch_size, "epochs": epochs,
        "rnn_hidden": rnn_hidden, "num_layers": num_layers,
        "aug_copies": aug_copies, "rotation_max": rotation_max,
        "brightness_delta": brightness_delta, "noise_sigma": noise_sigma,
    }, name="hyperparams")

    best_cer = _run_trial_locally(
        trial=trial,
        trial_task=trial_task,
        manifest=manifest_path,
        # ... all params
    )
    return best_cer
```

**GPU queue (--enqueue) path:**

When --enqueue is active, the objective must enqueue the trial task and then poll for its
completion to retrieve the CER. This means:
1. Create a ClearML task for the trial (or build args and call task.execute_remotely())
2. Poll `task.get_last_scalar_metrics()` or use `Task.get_task(task_id).get_last_scalar_metrics()`
3. Return the final val CER to Optuna

For the enqueue path, Optuna pruning (per-epoch) cannot function because training happens
remotely — the trial.report() / should_prune() calls require in-process access to per-epoch
metrics. The planner must decide whether pruning is local-only or if the GPU path skips pruning.
**Research conclusion:** Pruning is only meaningful on the local path. The GPU enqueue path runs
each trial to completion; early stopping within train_ctc.py (if the training loss plateaus)
still applies. This is a reasonable tradeoff at N=20.

### Pattern 2: CRNN Architecture Parameterization

**What:** Expose `rnn_hidden` and `num_layers` as parameters to `CRNN.__init__`. Currently
hardcoded at `rnn_hidden=256, num_layers=2`.

**Current CRNN.__init__ (line 309):**
```python
self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=False)
self.fc = nn.Linear(512, num_classes)  # 512 = 256 * 2 (bidirectional)
```

**Required change:**
```python
def __init__(self, num_classes: int, rnn_hidden: int = 256, num_layers: int = 2) -> None:
    super().__init__()
    self.cnn = nn.Sequential(...)  # unchanged
    self.rnn = nn.LSTM(128 * 8, rnn_hidden, num_layers=num_layers, bidirectional=True, batch_first=False)
    self.fc = nn.Linear(rnn_hidden * 2, num_classes)  # *2 for bidirectional
```

The `fc` layer output size depends on `rnn_hidden`: `rnn_hidden * 2` because of bidirectional.
This is the only dependency — the CNN output (128*8=1024) is fixed regardless of rnn_hidden.

**train_ctc.py change:**
```python
p.add_argument("--rnn_hidden", type=int, default=256,
               help="CRNN BiLSTM hidden size (D-02 Phase 5)")
p.add_argument("--num_layers", type=int, default=2,
               help="CRNN BiLSTM number of layers (D-02 Phase 5)")
# ...
model = CRNN(num_classes=len(charset) + 1, rnn_hidden=args.rnn_hidden, num_layers=args.num_layers)
```

### Pattern 3: best_params.json Round-Trip

**What:** tune.py writes the winner config; train_ctc.py reads it via --params.

**Output format (outputs/best_params.json):**
```json
{
  "lr": 0.00123,
  "batch_size": 8,
  "epochs": 35,
  "rnn_hidden": 256,
  "num_layers": 2,
  "aug_copies": 4,
  "rotation_max": 7.5,
  "brightness_delta": 0.08,
  "noise_sigma": 0.025,
  "best_val_cer": 0.42,
  "trial_number": 7,
  "n_trials_run": 20
}
```

**train_ctc.py --params loading pattern:**
```python
p.add_argument("--params", type=Path, default=None,
               help="Load hyperparams from JSON file (outputs from tune.py D-10)")
# In main(), before task.connect():
if args.params:
    import json
    loaded = json.loads(args.params.read_text())
    for k, v in loaded.items():
        if hasattr(args, k):
            setattr(args, k, type(getattr(args, k))(v))
```

Apply before `task.connect(vars(args))` so ClearML logs the loaded values.

### Pattern 4: Optuna Pruning in Training Loop

**What:** Inside the local-path training loop, report val CER after each epoch and check prune.

```python
# Source: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
for epoch in range(epochs):
    # ... train epoch ...
    val_cer = _compute_val_cer(model, val_loader, charset, device, ctc_loss)
    trial.report(val_cer, epoch)
    if trial.should_prune():
        raise optuna.TrialPruned()
```

**MedianPruner configuration:**
```python
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,   # don't prune until 5 trials complete
    n_warmup_steps=5,     # don't prune within first 5 epochs of any trial
    interval_steps=1,     # check every epoch
)
study = optuna.create_study(direction="minimize", pruner=pruner)
```

### Pattern 5: ClearML Orchestrator Task Reporting

**What:** tune.py creates its own ClearML task (hpo_sweep) that logs comparison results.

```python
# After study.optimize() completes:
orch_task = init_task("handwriting-hebrew-ocr", "hpo_sweep", tags=["phase-5"])
logger = orch_task.get_logger()

# Report CER-per-trial scalar
for t in study.trials:
    if t.value is not None:
        logger.report_scalar(title="cer", series="val", iteration=t.number, value=t.value)

# Report all-trials table
rows = [[t.number, t.value or float("nan")] + [t.params.get(k) for k in PARAM_KEYS]
        for t in study.trials]
logger.report_table(title="HPO Results", series="trials", iteration=0, table_plot=pd.DataFrame(...))
```

### Anti-Patterns to Avoid

- **Don't use ClearML HyperParameterOptimizer for this phase:** requires base_task_id pre-existing;
  pruning is awkward cross-process; designed for parallel workloads, not sequential N=20.
- **Don't put trial.report() inside train_ctc.py:** train_ctc is a standalone CLI that runs
  without Optuna. Keep the Optuna dependency in tune.py only. The training loop in the local
  path is duplicated/extracted, or train_ctc is called with a callback argument.
- **Don't hardcode best_params.json path:** accept it as a CLI argument to train_ctc.py.
- **Don't forget rnn_hidden * 2 for fc layer:** fc input must match biLSTM hidden * 2.
- **Don't tag orchestrator task as "hpo-trial":** use "phase-5" only; trials get "phase-5" +
  "hpo-trial".

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Bayesian hyperparameter search | Custom grid/random loop | optuna.create_study + TPESampler (default) | TPE converges 5-10x faster than random search; median pruner adds further savings |
| Early trial stopping | Epoch counter + manual threshold | optuna.pruners.MedianPruner | Handles n_startup_trials, warmup steps, min_iterations automatically |
| Search space distributions | Linspace / manual sampling | trial.suggest_float(log=True) | Log-uniform for LR is the correct distribution; manual sampling requires separate random state |
| Trial result tracking | CSV file | optuna.study.trials + ClearML scalars | Study stores full trial history, params, pruned/failed states |

**Key insight:** Optuna's value is the sampler (TPE), not just organization. A manual grid or
random loop loses the adaptive search that makes HPO worthwhile on a small trial budget.

---

## Common Pitfalls

### Pitfall 1: CRNN fc Layer Size After rnn_hidden Change
**What goes wrong:** Change `rnn_hidden` but forget `fc = nn.Linear(rnn_hidden * 2, num_classes)`.
Training crashes with shape mismatch at the fc layer.
**Why it happens:** BiLSTM doubles the hidden size (forward + backward). The fc input must be
`rnn_hidden * 2`, not `rnn_hidden`.
**How to avoid:** Parameterize both in `__init__`: `self.fc = nn.Linear(rnn_hidden * 2, num_classes)`.
**Warning signs:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` at first forward pass.

### Pitfall 2: task.connect() Before execute_remotely() (Phase 4 established pattern)
**What goes wrong:** If `task.connect()` is called after `execute_remotely()`, hyperparameters
aren't visible in ClearML UI and won't be overridden by the agent.
**Why it happens:** `execute_remotely()` is the fork point — everything before it runs locally
AND remotely; everything after runs only remotely.
**How to avoid:** In train_ctc.py, `task.connect(vars(args))` already precedes `execute_remotely()`
(established in Phase 4). Keep this order when adding --rnn_hidden, --num_layers.

### Pitfall 3: Optuna Trial Tags Lost on Pruned/Failed Trials
**What goes wrong:** A pruned trial raises `TrialPruned` — if the ClearML task was already
created, it stays in "running" state and never gets closed.
**Why it happens:** `task.close()` is not called on exception path.
**How to avoid:** Wrap the trial body in try/finally:
```python
try:
    best_cer = _run_training(...)
    return best_cer
except optuna.TrialPruned:
    trial_task.close()
    raise
finally:
    trial_task.close()
```

### Pitfall 4: --params Flag Overrides Wrong Type
**What goes wrong:** JSON stores `batch_size` as int, but setattr does `int(int(v))` which is fine,
but `rnn_hidden` stored as float (e.g. 256.0 in JSON) gets cast to wrong type.
**Why it happens:** JSON doesn't distinguish int vs float for round numbers.
**How to avoid:** When loading --params, explicitly cast to the type of the argparse default:
```python
type(getattr(args, k))(v)  # uses argparse's declared type
```

### Pitfall 5: ClearML Section Prefix for HPO Parameter Names
**What goes wrong:** When using ClearML HPO (if ever revisiting), parameter names must include
section prefix: `hyperparams/lr`, `Args/lr`, or `General/lr` depending on how they were
connected. Wrong prefix means HPO can't override the parameter.
**Why it happens:** `task.connect(dict, name="hyperparams")` stores under `hyperparams/`; bare
`task.connect(dict)` stores under `General/`.
**How to avoid:** This phase uses Optuna standalone (no ClearML HPO), so this pitfall only
applies if switching backends later.

### Pitfall 6: Optuna Study Direction (minimize CER)
**What goes wrong:** Creating study with `direction="maximize"` when optimizing CER (lower is better).
**Why it happens:** Copy-paste from accuracy examples.
**How to avoid:** Always `optuna.create_study(direction="minimize")` for CER.

### Pitfall 7: GPU-Enqueue Path and Pruning Incompatibility
**What goes wrong:** Expecting Optuna to prune GPU-enqueued trials mid-run.
**Why it happens:** When `--enqueue` is set, training happens in the ClearML agent process —
tune.py cannot call `trial.report()` in that remote process.
**How to avoid:** Pruning is local-only. Document this clearly. The enqueue path runs each trial
to full completion; pruning is skipped. Add a comment in tune.py.

---

## Code Examples

### Full tune.py Skeleton (Local Path)
```python
# Source: Optuna 4.8.0 docs + project patterns
import argparse, json
from pathlib import Path
import optuna
from clearml import Task  # noqa: F401  # module-level for test patchability
from src.clearml_utils import init_task

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HPO sweep for CRNN+CTC")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--enqueue", action="store_true", default=False)
    p.add_argument("--queue_name", type=str, default="gpu")
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    return p

def main() -> int:
    args = _build_parser().parse_args()
    orch_task = init_task("handwriting-hebrew-ocr", "hpo_sweep", tags=["phase-5"])
    orch_task.connect(vars(args), name="sweep_config")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    best = study.best_trial
    best_params = {**best.params, "best_val_cer": best.value,
                   "trial_number": best.number, "n_trials_run": len(study.trials)}
    out_path = args.output_dir / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_params, indent=2))
    print(f"Best trial {best.number}: CER={best.value:.4f}")
    print(json.dumps(best_params, indent=2))

    # ClearML summary report (logged to orch_task)
    _report_hpo_results(orch_task, study)
    return 0
```

### Optuna suggest_* for This Project's Search Space
```python
# Source: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html
lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
batch_size  = trial.suggest_categorical("batch_size", [4, 8, 16])
epochs      = trial.suggest_int("epochs", 20, 50)
rnn_hidden  = trial.suggest_categorical("rnn_hidden", [128, 256, 512])
num_layers  = trial.suggest_categorical("num_layers", [1, 2])
aug_copies  = trial.suggest_int("aug_copies", 0, 6)
rotation_max       = trial.suggest_float("rotation_max", 0.0, 15.0)
brightness_delta   = trial.suggest_float("brightness_delta", 0.0, 0.20)
noise_sigma        = trial.suggest_float("noise_sigma", 0.0, 0.05)
```

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| optuna | tune.py HPO | No (not installed) | — | Must install: `uv add optuna==4.8.0` |
| clearml | ClearML task logging | Yes | 2.1.5 | — |
| torch | Training | Yes | 2.11.0 | — |
| ClearML agent (WSL2) | --enqueue GPU path | Unknown (external) | — | Omit --enqueue for local CPU |

**Missing dependencies with no fallback:**
- optuna is not installed in the project venv. Must be added to pyproject.toml and uv.lock before
  any tune.py code can run. Install with: `uv add optuna==4.8.0`

**Missing dependencies with fallback:**
- ClearML agent on WSL2/RTX 5060: if agent is not running, omit `--enqueue` and run locally on
  CPU. Phase 4 established this fallback pattern.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual grid search loops | Optuna TPE sampler | Optuna 1.0 (2019) | 5-10x fewer trials needed |
| suggest_uniform / suggest_loguniform | suggest_float(log=True) | Optuna 3.0 (2023) | API consolidation; old names deprecated |
| optuna.Trial.report at each step | Same (unchanged) | — | Pruning API stable |
| ClearML HPO as wrapper | Optuna standalone preferred for small-N sequential | — | Simpler, no cold-start |

**Deprecated/outdated:**
- `trial.suggest_uniform()`, `trial.suggest_loguniform()`, `trial.suggest_discrete_uniform()`,
  `trial.suggest_int()` with step<1: all deprecated in Optuna 3.x, removed in 4.x. Use
  `suggest_float()` and `suggest_int()` with `log=True` / `step=` arguments instead.

---

## Open Questions

1. **Enqueue path: how does tune.py block until a remote trial completes?**
   - What we know: Phase 4 uses `task.execute_remotely()` which makes the local process exit after
     enqueuing. tune.py cannot simply call execute_remotely and continue.
   - What's unclear: The cleanest pattern for a blocking poll — `task.wait_for_status()` or
     periodic `Task.get_task(id).get_last_scalar_metrics()` polling.
   - Recommendation: For Phase 5, the enqueue path should be documented as "fire and forget for
     individual trials is not supported with pruning." Instead, the --enqueue flag can mean:
     dispatch the entire tune.py sweep to the queue as a single ClearML task, so training inside
     it runs on the GPU. This aligns with Phase 4's pattern: tune.py itself calls
     `task.execute_remotely()`, the ClearML agent picks it up and runs the full Optuna study
     locally on the GPU machine. Pruning works normally in this model.
   - This is the cleanest resolution: tune.py uses execute_remotely the same way train_ctc.py does.

2. **Should tune.py duplicate train_ctc's training loop or import/call it?**
   - What we know: Duplicating creates maintenance burden; importing requires train_ctc's main()
     to be callable with args dict, not subprocess.
   - What's unclear: How deeply refactored train_ctc needs to be.
   - Recommendation: Extract the inner training loop from train_ctc.py into a
     `run_training(args) -> float` helper in ctc_utils.py or as a private function in train_ctc.py
     that tune.py can import. This avoids subprocess and enables in-process pruning.

3. **Optuna study persistence across runs?**
   - What we know: By default, optuna.create_study() uses an in-memory storage — study results
     are lost if tune.py is interrupted.
   - What's unclear: Whether the user wants resume capability.
   - Recommendation: Out of scope for Phase 5. A future phase could add
     `optuna.storages.RDBStorage` with SQLite for persistence. Document the limitation in tune.py.

---

## Sources

### Primary (HIGH confidence)
- PyPI optuna 4.8.0 — version and release date verified 2026-05-05
- [Optuna Efficient Optimization Algorithms docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html) — pruning API, trial.report(), should_prune(), MedianPruner
- [Optuna Pythonic Search Space docs](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html) — suggest_float, suggest_int, suggest_categorical, log=True
- [ClearML HPO SDK docs](https://clear.ml/docs/latest/docs/clearml_sdk/hpo_sdk/) — HyperParameterOptimizer workflow, base_task prerequisite
- [ClearML HyperParameterOptimizer reference](https://clear.ml/docs/latest/docs/references/sdk/hpo_optimization_hyperparameteroptimizer/) — get_top_experiments, parameters
- Existing codebase: src/train_ctc.py, src/ctc_utils.py, src/clearml_utils.py — patterns confirmed by reading

### Secondary (MEDIUM confidence)
- [ClearML + Optuna integration page](https://clear.ml/docs/latest/docs/integrations/optuna/) — OptimizerOptuna setup pattern
- [ClearML Hyperparameters docs](https://clear.ml/docs/latest/docs/fundamentals/hyperparameters/) — section prefix naming (Args/, General/, hyperparams/)
- [clearml/clearml GitHub: optuna.py](https://github.com/clearml/clearml/blob/master/clearml/automation/optuna/optuna.py) — OptimizerOptuna internals

### Tertiary (LOW confidence)
- WebSearch community examples for tune.py + execute_remotely blocking pattern — not found; recommendation is based on reasoning from established Phase 4 patterns

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — optuna 4.8.0 version confirmed from PyPI; clearml imports verified in venv
- Architecture: HIGH — patterns derived from reading existing codebase + official Optuna docs
- Pitfalls: HIGH — CRNN fc size from code reading; execute_remotely order from Phase 4 code;
  Optuna deprecated API from official docs
- Backend decision: HIGH — based on concrete analysis of both options' mechanics

**Research date:** 2026-05-05
**Valid until:** 2026-08-05 (Optuna is stable; ClearML changes more frequently)
