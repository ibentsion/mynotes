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
from src.run_config import load_config, peek_mode, update_config
from src.train_ctc import run_training

# Finetune param keys — kept as module-level tuple for backward compat with tests
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

PRETRAIN_PARAM_KEYS = (
    "pretrain_lr",
    "pretrain_epochs",
    "batch_size",
    "rnn_hidden",
    "num_layers",
)


def _param_keys(mode: str) -> tuple[str, ...]:
    return PRETRAIN_PARAM_KEYS if mode == "pretrain" else PARAM_KEYS


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
        "--dataset_id",
        type=str,
        default=None,
        help="ClearML dataset ID. pretrain=synthetic crops, finetune=real labeled data.",
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
    p.add_argument(
        "--pretrain_checkpoint_path",
        type=Path,
        default=None,
        help="Path to pre-trained checkpoint; forwarded to each finetune HPO trial.",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        default="finetune",
        help="HPO target: 'pretrain' tunes synthetic pretraining; 'finetune' tunes real-data.",
    )
    p.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Local SQLite path for persistent study (e.g. outputs/hpo.db). "
            "Enables resume after abort and optuna-dashboard visualization."
        ),
    )
    p.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name for persistent storage (default: hpo_<mode>).",
    )
    return p


def _suggest_params(trial: optuna.Trial, mode: str = "finetune") -> dict[str, object]:
    if mode == "pretrain":
        return {
            "pretrain_lr": trial.suggest_float("pretrain_lr", 1e-4, 1e-2, log=True),
            "pretrain_epochs": trial.suggest_int("pretrain_epochs", 10, 50),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "rnn_hidden": trial.suggest_categorical("rnn_hidden", [128, 256]),
            "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        }
    return {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "epochs": trial.suggest_int("epochs", 20, 50),
        "rnn_hidden": trial.suggest_categorical("rnn_hidden", [128, 256]),
        "num_layers": trial.suggest_categorical("num_layers", [1, 2]),
        "aug_copies": trial.suggest_int("aug_copies", 0, 6),
        "rotation_max": trial.suggest_float("rotation_max", 0.0, 15.0),
        "brightness_delta": trial.suggest_float("brightness_delta", 0.0, 0.20),
        "noise_sigma": trial.suggest_float("noise_sigma", 0.0, 0.05),
    }


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


def _make_progress_callback(
    n_trials_target: int,
) -> Callable[[optuna.Study, optuna.Trial], None]:
    def _cb(study: optuna.Study, trial: optuna.Trial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_done = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        best = study.best_trial
        best_str = f"{best.value:.4f} (trial {best.number})" if best else "N/A"
        print(
            f"Trial {trial.number}: CER={trial.value:.4f}  "
            f"({n_done}/{n_trials_target} done, best={best_str})"
        )

    return _cb


def _objective(trial: optuna.Trial, sweep_args: argparse.Namespace) -> float:
    mode = str(getattr(sweep_args, "mode", "finetune"))
    params = _suggest_params(trial, mode)

    # Bypass argparse entirely. ClearML monkey-patches both parse_args and
    # parse_known_args, and on the GPU agent it injects tune.py-only flags
    # (--n_trials, --n_startup_trials, --n_warmup_steps) into any parser
    # call inside this process — train_ctc's parser rejects them. Direct
    # Namespace construction cannot be intercepted.
    common = dict(
        output_dir=sweep_args.output_dir / f"trial_{trial.number}",
        num_workers=0,
        params=None,
        enqueue=False,
        queue_name="gpu",
        dataset_id=sweep_args.dataset_id,
        blank_bias_init=-2.0,
        rnn_hidden=params.get("rnn_hidden", 128),
        num_layers=params.get("num_layers", 2),
        batch_size=params.get("batch_size", 16),
        elastic_alpha=0.0,
        elastic_sigma=5.0,
        weight_decay=1e-4,
        patience=0,  # disabled — Optuna MedianPruner is the termination mechanism
        mode=mode,
    )

    if mode == "pretrain":
        train_args = argparse.Namespace(
            **common,
            manifest=sweep_args.manifest,
            min_labeled=0,
            pretrain_lr=params["pretrain_lr"],
            pretrain_epochs=params["pretrain_epochs"],
            pretrain_checkpoint_path=None,
            val_frac=0.2,
            # Unused for pretrain but must exist so _run_pretrain's Namespace copy doesn't error:
            lr=1e-3,
            epochs=0,
            aug_copies=0,
            rotation_max=0.0,
            brightness_delta=0.0,
            noise_sigma=0.0,
            words_file=None,
        )
    else:
        train_args = argparse.Namespace(
            **common,
            manifest=sweep_args.manifest,
            min_labeled=sweep_args.min_labeled,
            val_frac=0.2,
            pretrain_epochs=0,
            pretrain_lr=1e-3,
            pretrain_checkpoint_path=sweep_args.pretrain_checkpoint_path,
            lr=params["lr"],
            epochs=params["epochs"],
            aug_copies=params["aug_copies"],
            rotation_max=params["rotation_max"],
            brightness_delta=params["brightness_delta"],
            noise_sigma=params["noise_sigma"],
            words_file=None,
        )

    train_args.output_dir.mkdir(parents=True, exist_ok=True)

    _, on_epoch_end = _make_pruning_callback(trial)
    return run_training(train_args, on_epoch_end=on_epoch_end)


def _report_hpo_results(
    orch_task: Task,
    study: optuna.Study,
    mode: str = "finetune",
) -> None:
    logger = orch_task.get_logger()
    rows: list[dict[str, object]] = []
    keys = _param_keys(mode)
    for t in study.trials:
        if t.value is not None:
            logger.report_scalar(title="cer", series="val", iteration=t.number, value=t.value)
        row: dict[str, object] = {
            "trial_number": t.number,
            "value": t.value,
            "state": str(t.state),
        }
        for k in keys:
            row[k] = t.params.get(k)
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        logger.report_table(title="HPO Results", series="trials", iteration=0, table_plot=df)


def _write_best_params(
    study: optuna.Study,
    output_dir: Path,
    mode: str = "finetune",
) -> Path:
    best = study.best_trial
    keys = _param_keys(mode)
    best_params = {k: best.params.get(k) for k in keys}
    best_params["best_val_cer"] = float(best.value) if best.value is not None else None
    best_params["trial_number"] = best.number
    best_params["n_trials_run"] = len(study.trials)
    out_path = output_dir / "best_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best_params, indent=2))
    return out_path


def main() -> int:
    _config = load_config(mode=peek_mode())
    parser = _build_parser()
    if _config.get("datasets"):
        parser.set_defaults(dataset_id=_config["datasets"].get("id"))  # ty: ignore[unresolved-attribute]
    if _config.get("queue_name"):
        parser.set_defaults(queue_name=_config["queue_name"])
    if _config.get("manifest"):
        parser.set_defaults(manifest=Path(str(_config["manifest"])))
    if _config.get("hpo_storage"):
        parser.set_defaults(storage=str(_config["hpo_storage"]))
    args = parser.parse_args()

    orch_task = init_task(
        "handwriting-hebrew-ocr",
        f"hpo_{args.mode}",
        tags=["phase-5"],
    )
    orch_task.connect(vars(args), name="sweep_config")
    if args.enqueue:
        orch_task.execute_remotely(queue_name=args.queue_name)

    # Resolve manifest from ClearML dataset when not available locally (agent path)
    if args.dataset_id and not args.manifest.exists():
        from clearml import Dataset  # noqa: PLC0415

        alias = "real" if args.mode == "finetune" else "synthetic"
        local_root = Path(Dataset.get(dataset_id=args.dataset_id, alias=alias).get_local_copy())
        args.manifest = local_root / "manifest.csv"

    if not args.manifest.exists():
        print(f"ERROR: --manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    storage_url = f"sqlite:///{args.storage}" if args.storage else None
    study_name = args.study_name or f"hpo_{args.mode}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps,
    )
    study = optuna.create_study(  # Pitfall 6
        direction="minimize",
        pruner=pruner,
        storage=storage_url,
        study_name=study_name,
        load_if_exists=bool(storage_url),
    )

    prev_complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    to_run = max(0, args.n_trials - len(prev_complete))
    if prev_complete:
        print(
            f"Resuming study '{study_name}': {len(prev_complete)} previous trials completed, "
            f"running {to_run} more (target: {args.n_trials})"
        )

    study.optimize(
        lambda trial: _objective(trial, args),
        n_trials=to_run,
        callbacks=[_make_progress_callback(args.n_trials)],
    )

    if not study.trials or study.best_trial is None:
        print("ERROR: no trials completed; nothing to report.", file=sys.stderr)
        return 7

    out_path = _write_best_params(study, args.output_dir, mode=args.mode)
    best_params = json.loads(out_path.read_text())
    keys = _param_keys(args.mode)
    tunable = {f"hyperparams.{k}": best_params[k] for k in keys if k in best_params}
    update_config(mode=args.mode, **tunable)
    best_cer = best_params["best_val_cer"]
    cer_str = f"{best_cer:.4f}" if best_cer is not None else "N/A"
    print(f"Best trial {best_params['trial_number']}: CER={cer_str}")
    print(json.dumps(best_params, indent=2))
    _report_hpo_results(orch_task, study, mode=args.mode)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) >= 2:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importances (fANOVA):")
        for param, score in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {param}: {score:.4f}")
        imp_df = pd.DataFrame(
            [{"param": k, "importance": v} for k, v in importances.items()]
        ).sort_values("importance", ascending=False)
        orch_task.get_logger().report_table(
            title="Parameter Importances", series="fANOVA", iteration=0, table_plot=imp_df
        )

    if args.storage:
        storage_path = Path(args.storage)
        if storage_path.exists():
            orch_task.upload_artifact("optuna_study_db", artifact_object=str(storage_path))
        print(f"\nOptuna dashboard: optuna-dashboard sqlite:///{args.storage}")
        print("(Install: uv add --dev optuna-dashboard)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
