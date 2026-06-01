"""hpo_inspect.py — Inspect Optuna HPO study results from a local SQLite DB or ClearML artifact."""

import argparse
import subprocess
import sys
from pathlib import Path

import optuna

_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "finetune": (
        "lr",
        "batch_size",
        "epochs",
        "rnn_hidden",
        "num_layers",
        "aug_copies",
        "rotation_max",
        "brightness_delta",
        "noise_sigma",
    ),
    "pretrain": (
        "pretrain_lr",
        "pretrain_epochs",
        "batch_size",
        "rnn_hidden",
        "num_layers",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Inspect Optuna HPO study: trial summary, best params, fANOVA importances."
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--storage",
        type=str,
        help="Local SQLite path (e.g. outputs/hpo_finetune.db)",
    )
    source.add_argument(
        "--task_id",
        type=str,
        help="ClearML task ID — downloads the optuna_study_db artifact locally",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["pretrain", "finetune"],
        default="finetune",
        help="HPO mode (selects param key list for importances display; default: finetune)",
    )
    p.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Optuna study name in the DB (default: hpo_<mode>)",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        default=False,
        help="Launch optuna-dashboard after printing summary (uv add --dev optuna-dashboard)",
    )
    return p


def _download_from_clearml(task_id: str) -> Path:
    try:
        from clearml import Task  # noqa: PLC0415
    except ImportError:
        print("ERROR: clearml not installed. Run: uv add clearml", file=sys.stderr)
        sys.exit(1)
    task = Task.get_task(task_id=task_id)
    artifact = task.artifacts.get("optuna_study_db")
    if artifact is None:
        print(
            f"ERROR: ClearML task {task_id} has no 'optuna_study_db' artifact.\n"
            "Re-run the HPO sweep with --storage to persist and upload the DB.",
            file=sys.stderr,
        )
        sys.exit(1)
    return Path(artifact.get_local_copy())


def _print_summary(storage_path: Path, mode: str, study_name: str) -> None:
    storage_url = f"sqlite:///{storage_path}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        available = optuna.get_all_study_names(storage=storage_url)
        print(f"ERROR: cannot load study '{study_name}': {e}", file=sys.stderr)
        if available:
            print(f"Available studies in {storage_path}: {available}", file=sys.stderr)
        sys.exit(1)

    all_trials = study.trials
    complete = [t for t in all_trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in all_trials if t.state == optuna.trial.TrialState.PRUNED]
    running = [t for t in all_trials if t.state == optuna.trial.TrialState.RUNNING]

    stale = f", {len(running)} stale-running (aborted mid-trial)" if running else ""
    print(f"Study:    {study_name}")
    print(
        f"Trials:   {len(all_trials)} total — "
        f"{len(complete)} complete, {len(pruned)} pruned{stale}"
    )

    if not complete:
        print("No completed trials yet.")
        return

    best = study.best_trial
    param_keys = _PARAM_KEYS.get(mode, _PARAM_KEYS["finetune"])
    print(f"\nBest:     trial #{best.number}  CER={best.value:.4f}")
    for k in param_keys:
        if k in best.params:
            v = best.params[k]
            print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    if len(complete) >= 2:
        print("\nParam importances (fANOVA):")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, score in sorted(importances.items(), key=lambda x: -x[1]):
                bar = "█" * max(1, int(score * 20))
                print(f"  {param:<22} {score:.4f}  {bar}")
        except Exception as e:
            print(f"  (unavailable: {e})")
    else:
        print(f"\nParam importances: need ≥2 complete trials (have {len(complete)})")

    print(f"\nDB:       sqlite:///{storage_path}")
    print(f"Dashboard: optuna-dashboard sqlite:///{storage_path}")


def main() -> int:
    args = _build_parser().parse_args()
    study_name = args.study_name or f"hpo_{args.mode}"

    if args.task_id:
        storage_path = _download_from_clearml(args.task_id)
    else:
        storage_path = Path(args.storage)
        if not storage_path.exists():
            print(f"ERROR: not found: {storage_path}", file=sys.stderr)
            return 1

    _print_summary(storage_path, args.mode, study_name)

    if args.dashboard:
        subprocess.run(["optuna-dashboard", f"sqlite:///{storage_path}"], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
