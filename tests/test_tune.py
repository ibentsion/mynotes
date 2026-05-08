"""Tests for src/tune.py — Optuna HPO sweep CLI."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import optuna
import pytest


@pytest.fixture(autouse=True)
def _offline_clearml(monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")


def test_build_parser_defaults():
    from src.tune import _build_parser

    args = _build_parser().parse_args([])
    assert args.n_trials == 20
    assert args.queue_name == "gpu"
    assert args.output_dir == Path("outputs")
    assert args.min_labeled == 100
    assert args.n_startup_trials == 5
    assert args.n_warmup_steps == 5
    assert args.enqueue is False
    assert args.manifest == Path("data/manifest.csv")


def test_param_keys_tuple_order_and_count():
    from src.tune import PARAM_KEYS

    assert PARAM_KEYS == (
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


def test_suggest_params_returns_all_keys_with_valid_ranges():
    from src.tune import PARAM_KEYS, _suggest_params

    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.side_effect = lambda name, lo, hi, **kw: (lo + hi) / 2
    trial.suggest_int.side_effect = lambda name, lo, hi: (lo + hi) // 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    params = _suggest_params(trial)
    assert set(params.keys()) == set(PARAM_KEYS)
    # cast to concrete types; _suggest_params returns dict[str, object] by design
    assert 1e-4 <= cast(float, params["lr"]) <= 1e-2
    assert params["batch_size"] in (4, 8, 16)
    assert 20 <= cast(int, params["epochs"]) <= 50
    assert params["rnn_hidden"] in (128, 256, 512)
    assert params["num_layers"] in (1, 2)
    assert 0 <= cast(int, params["aug_copies"]) <= 6
    assert 0.0 <= cast(float, params["rotation_max"]) <= 15.0
    assert 0.0 <= cast(float, params["brightness_delta"]) <= 0.20
    assert 0.0 <= cast(float, params["noise_sigma"]) <= 0.05


def test_write_best_params_writes_all_required_keys(tmp_path: Path):
    from src.tune import PARAM_KEYS, _write_best_params

    study = MagicMock()
    study.best_trial.number = 7
    study.best_trial.value = 0.42
    study.best_trial.params = {
        "lr": 0.005,
        "batch_size": 8,
        "epochs": 30,
        "rnn_hidden": 256,
        "num_layers": 2,
        "aug_copies": 4,
        "rotation_max": 7.5,
        "brightness_delta": 0.08,
        "noise_sigma": 0.025,
    }
    study.trials = [MagicMock() for _ in range(20)]
    out_path = _write_best_params(study, tmp_path)
    assert out_path == tmp_path / "best_params.json"
    loaded = json.loads(out_path.read_text())
    for k in PARAM_KEYS:
        assert k in loaded, f"missing key: {k}"
    assert loaded["best_val_cer"] == 0.42
    assert loaded["trial_number"] == 7
    assert loaded["n_trials_run"] == 20


def test_objective_pruning_callback_raises_trial_pruned():
    from src.tune import _objective

    sweep_args = MagicMock()
    sweep_args.manifest = Path("data/manifest.csv")
    sweep_args.min_labeled = 100
    sweep_args.dataset_id = None
    sweep_args.output_dir = Path("/tmp/test_outputs_prune")
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_float.side_effect = lambda *a, **kw: 0.001
    trial.suggest_int.side_effect = lambda *a, **kw: 30
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    trial.should_prune.return_value = True

    with patch("src.tune.run_training") as mock_run, patch("src.tune.init_task") as mock_init:

        def fake_run(args, on_epoch_end=None):
            assert on_epoch_end is not None
            on_epoch_end(0, 0.9)
            # If we reach here, pruning didn't fire on the first call — test should fail
            on_epoch_end(1, 0.9)
            return 0.9

        mock_run.side_effect = fake_run
        mock_init.return_value = MagicMock()
        with pytest.raises(optuna.TrialPruned):
            _objective(trial, sweep_args)


def test_objective_closes_trial_task_on_failure():
    from src.tune import _objective

    sweep_args = MagicMock()
    sweep_args.manifest = Path("data/manifest.csv")
    sweep_args.min_labeled = 100
    sweep_args.dataset_id = None
    sweep_args.output_dir = Path("/tmp/test_outputs_fail")
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_float.side_effect = lambda *a, **kw: 0.001
    trial.suggest_int.side_effect = lambda *a, **kw: 30
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

    trial_task_mock = MagicMock()
    with (
        patch("src.tune.run_training") as mock_run,
        patch("src.tune.init_task", return_value=trial_task_mock),
    ):
        mock_run.side_effect = ValueError("split produced empty set")
        with pytest.raises(ValueError):
            _objective(trial, sweep_args)
        assert trial_task_mock.close.called


def test_orchestrator_task_tagged_phase_5_only(tmp_path: Path):
    from src.tune import main

    manifest = tmp_path / "m.csv"
    manifest.write_text("x")
    with (
        patch("src.tune.init_task") as mock_init,
        patch("src.tune.optuna.create_study") as mock_create,
    ):
        study = MagicMock()
        study.optimize = MagicMock()
        study.trials = []
        study.best_trial = None
        mock_create.return_value = study
        mock_init.return_value = MagicMock()
        sys.argv = ["tune.py", "--manifest", str(manifest), "--n_trials", "0"]
        ret = main()
        # First init_task call is the orchestrator
        first_call = mock_init.call_args_list[0]
        tags = first_call.kwargs.get("tags") or (
            first_call.args[2] if len(first_call.args) > 2 else None
        )
        assert tags == ["phase-5"]
        assert ret == 7  # no trials → exit 7


def test_per_trial_task_tagged_phase_5_and_hpo_trial():
    from src.tune import _objective

    sweep_args = MagicMock()
    sweep_args.manifest = Path("data/manifest.csv")
    sweep_args.min_labeled = 100
    sweep_args.dataset_id = None
    sweep_args.output_dir = Path("/tmp/test_outputs_tags")
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 3
    trial.suggest_float.side_effect = lambda *a, **kw: 0.001
    trial.suggest_int.side_effect = lambda *a, **kw: 30
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    trial.should_prune.return_value = False
    with patch("src.tune.run_training", return_value=0.5), patch("src.tune.init_task") as mock_init:
        mock_init.return_value = MagicMock()
        _objective(trial, sweep_args)
        call_args = mock_init.call_args
        tags = call_args.kwargs.get("tags") or (
            call_args.args[2] if len(call_args.args) > 2 else None
        )
        assert tags == ["phase-5", "hpo-trial"]


def test_objective_ignores_extra_sys_argv_args(monkeypatch, tmp_path: Path):
    from src.tune import _objective

    monkeypatch.setattr(
        sys,
        "argv",
        ["tune.py", "--n_trials", "20", "--n_startup_trials", "5", "--n_warmup_steps", "5"],
    )
    sweep_args = MagicMock()
    sweep_args.manifest = Path("data/manifest.csv")
    sweep_args.min_labeled = 100
    sweep_args.dataset_id = None
    sweep_args.output_dir = tmp_path / "out_argv"
    trial = MagicMock(spec=optuna.Trial)
    trial.number = 0
    trial.suggest_float.side_effect = lambda *a, **kw: 0.001
    trial.suggest_int.side_effect = lambda *a, **kw: 30
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
    trial.should_prune.return_value = False
    with patch("src.tune.run_training", return_value=0.5), patch("src.tune.init_task") as mock_init:
        mock_init.return_value = MagicMock()
        result = _objective(trial, sweep_args)
    assert result == 0.5


def test_e2e_one_trial_no_enqueue(monkeypatch, tmp_path: Path):
    from src.tune import PARAM_KEYS, main

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("crop_path,label,status,page_id\nfake/crop.png,א,labeled,p1\n")
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tune.py",
            "--manifest",
            str(manifest),
            "--n_trials",
            "1",
            "--min_labeled",
            "1",
            "--output_dir",
            str(out_dir),
        ],
    )
    with patch("src.tune.run_training", return_value=0.5), patch("src.tune.init_task") as mock_init:
        mock_init.return_value = MagicMock()
        ret = main()
    assert ret == 0
    best_params_path = out_dir / "best_params.json"
    assert best_params_path.exists()
    loaded = json.loads(best_params_path.read_text())
    for k in PARAM_KEYS:
        assert k in loaded, f"missing key: {k}"


def test_enqueue_calls_execute_remotely_before_optimize(tmp_path: Path):
    from src.tune import main

    manifest = tmp_path / "m.csv"
    manifest.write_text("x")
    orch_task = MagicMock()
    call_order: list[str] = []

    def track_execute_remotely(**kwargs):
        call_order.append("execute_remotely")

    def track_optimize(fn, n_trials):
        call_order.append("optimize")

    orch_task.execute_remotely.side_effect = track_execute_remotely
    with (
        patch("src.tune.init_task", return_value=orch_task),
        patch("src.tune.optuna.create_study") as mock_create,
    ):
        study = MagicMock()
        study.trials = []
        study.best_trial = None
        study.optimize.side_effect = track_optimize
        mock_create.return_value = study
        sys.argv = [
            "tune.py",
            "--manifest",
            str(manifest),
            "--enqueue",
            "--queue_name",
            "gpu",
            "--n_trials",
            "0",
        ]
        main()
        orch_task.execute_remotely.assert_called_once_with(queue_name="gpu")
        # Verify execute_remotely is called before study.optimize
        assert call_order.index("execute_remotely") < call_order.index("optimize")


def test_missing_manifest_exits_2(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tune",
            "--manifest",
            str(tmp_path / "nope.csv"),
            "--n_trials",
            "0",
        ],
        capture_output=True,
        text=True,
        env={"CLEARML_OFFLINE_MODE": "1", "PATH": ""},
        timeout=30,
    )
    import os

    if result.returncode != 2:
        env = os.environ.copy()
        env["CLEARML_OFFLINE_MODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.tune",
                "--manifest",
                str(tmp_path / "nope.csv"),
                "--n_trials",
                "0",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
    assert result.returncode == 2


def test_gitignore_excludes_best_params_json():
    result = subprocess.run(
        ["git", "check-ignore", "-v", "outputs/best_params.json"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"best_params.json not ignored: {result.stdout!r}"


def test_smoke_imports():
    from src.tune import (  # noqa: F401
        PARAM_KEYS,
        _build_parser,
        _objective,
        _report_hpo_results,
        _suggest_params,
        _write_best_params,
        main,
    )

    assert PARAM_KEYS
    assert callable(main)
