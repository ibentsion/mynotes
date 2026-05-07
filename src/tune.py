"""tune.py — Optuna HPO sweep over CRNN+CTC hyperparameters; logs to ClearML.

Per Phase 5 RESEARCH.md backend decision: standalone Optuna 4.8.0 (not ClearML
HyperParameterOptimizer). Each trial creates its own ClearML task; the orchestrator
task 'hpo_sweep' logs the comparison report.

Note: When --enqueue is set, tune.py calls task.execute_remotely() on the orchestrator
task BEFORE study.optimize(). The entire sweep then runs on the GPU agent in-process,
so Optuna pruning still functions normally (RESEARCH.md Open Question 1 resolution;
Pitfall 7: GPU path pruning incompatibility is avoided by dispatching the whole sweep,
not individual trials).
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import optuna
import pandas as pd
from clearml import Task  # noqa: F401  # module-level for test patchability — established pattern

from src.clearml_utils import init_task
from src.train_ctc import _build_parser as _build_train_parser
from src.train_ctc import run_training

PARAM_KEYS = (
    "lr",
    "batch_size",
    "epochs",
    "rnn_hidden",
    "num_layers",
    "aug_copies",
    "rotation_max",
    "brightness_delta",
    "noise_sigma",
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optuna HPO sweep over CRNN+CTC params (Phase 5).")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    p.add_argument("--n_trials", type=int, default=20, help="Number of trials in the sweep (D-07)")
    p.add_argument(
        "--enqueue",
        action="store_true",
        default=False,
        help="Run the entire sweep on the ClearML GPU agent (D-05; Open Question 1)",
    )
    p.add_argument(
        "--queue_name", type=str, default="gpu", help="ClearML queue name when --enqueue is set"
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("outputs"), help="Where to write best_params.json"
    )
    p.add_argument(
        "--min_labeled", type=int, default=100, help="Passed to train_ctc; matches its default"
    )
    p.add_argument(
        "--n_startup_trials",
        type=int,
        default=5,
        help="MedianPruner: trials before pruning kicks in (D-08)",
    )
    p.add_argument(
        "--n_warmup_steps",
        type=int,
        default=5,
        help="MedianPruner: epochs per trial before pruning kicks in (D-08)",
    )
    return p


def _suggest_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "epochs": trial.suggest_int("epochs", 20, 50),
        "rnn_hidden": trial.suggest_categorical("rnn_hidden", [128, 256, 512]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "aug_copies": trial.suggest_int("aug_copies", 0, 6),
        "rotation_max": trial.suggest_float("rotation_max", 0.0, 15.0),
        "brightness_delta": trial.suggest_float("brightness_delta", 0.0, 0.20),
        "noise_sigma": trial.suggest_float("noise_sigma", 0.0, 0.05),
    }


def _init_trial_task(trial: optuna.Trial, train_args: argparse.Namespace) -> Task:
    trial_task = init_task(
        "handwriting-hebrew-ocr",
        f"hpo_trial_{trial.number}",
        tags=["phase-5", "hpo-trial"],
    )
    trial_task.connect(vars(train_args), name="hyperparams")
    return trial_task


def _make_pruning_callback(
    trial: optuna.Trial,
) -> tuple[list[float], Callable[[int, float], None]]:
    """Return (best_cer_tracker, callback). best_cer_tracker[0] updated each epoch."""
    best_cer: list[float] = [float("inf")]

    def _on_epoch_end(epoch: int, val_cer: float) -> None:
        if val_cer < best_cer[0]:
            best_cer[0] = val_cer
        trial.report(val_cer, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_cer, _on_epoch_end


def _objective(trial: optuna.Trial, sweep_args: argparse.Namespace) -> float:
    params = _suggest_params(trial)

    train_args = _build_train_parser().parse_args(
        [
            "--manifest",
            str(sweep_args.manifest),
            "--min_labeled",
            str(sweep_args.min_labeled),
        ]
    )
    for k, v in params.items():
        setattr(train_args, k, v)
    train_args.output_dir = sweep_args.output_dir / f"trial_{trial.number}"
    train_args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_task = _init_trial_task(trial, train_args)
    _, on_epoch_end = _make_pruning_callback(trial)

    try:
        best_val_cer = run_training(train_args, on_epoch_end=on_epoch_end)
        return best_val_cer
    except optuna.TrialPruned:
        raise
    finally:
        trial_task.close()


def _report_hpo_results(orch_task: Task, study: optuna.Study) -> None:
    logger = orch_task.get_logger()
    rows: list[dict[str, object]] = []
    for t in study.trials:
        if t.value is not None:
            logger.report_scalar(title="cer", series="val", iteration=t.number, value=t.value)
        row: dict[str, object] = {
            "trial_number": t.number,
            "value": t.value,
            "state": str(t.state),
        }
        for k in PARAM_KEYS:
            row[k] = t.params.get(k)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        logger.report_table(title="HPO Results", series="trials", iteration=0, table_plot=df)


def _write_best_params(study: optuna.Study, output_dir: Path) -> Path:
    best = study.best_trial
    best_params = {k: best.params.get(k) for k in PARAM_KEYS}
    best_params["best_val_cer"] = float(best.value) if best.value is not None else None
    best_params["trial_number"] = best.number
    best_params["n_trials_run"] = len(study.trials)
    out_path = output_dir / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_params, indent=2))
    return out_path


def main() -> int:
    args = _build_parser().parse_args()

    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2

    orch_task = init_task("handwriting-hebrew-ocr", "hpo_sweep", tags=["phase-5"])
    orch_task.connect(vars(args), name="sweep_config")
    if args.enqueue:
        orch_task.execute_remotely(queue_name=args.queue_name)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps,
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)  # Pitfall 6
    study.optimize(lambda trial: _objective(trial, args), n_trials=args.n_trials)

    if not study.trials or study.best_trial is None:
        print("ERROR: no trials completed; nothing to report.", file=sys.stderr)
        return 7

    out_path = _write_best_params(study, args.output_dir)
    best_params = json.loads(out_path.read_text())
    print(f"Best trial {best_params['trial_number']}: CER={best_params['best_val_cer']:.4f}")
    print(json.dumps(best_params, indent=2))
    _report_hpo_results(orch_task, study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
