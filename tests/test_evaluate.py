"""Tests for src/evaluate.py — runs with CLEARML_OFFLINE_MODE=1."""

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.manifest_schema import MANIFEST_COLUMNS


def _make_grayscale_png(path: Path, h: int = 64, w: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _row(crop_path, page_path, page_num, y, h, status="labeled", label="א"):
    return {
        "crop_path": crop_path,
        "pdf_path": "x.pdf",
        "page_path": page_path,
        "page_num": page_num,
        "x": 0,
        "y": y,
        "w": 128,
        "h": h,
        "area": 128 * h,
        "is_flagged": False,
        "flag_reasons": "",
        "status": status,
        "label": label,
        "notes": "",
    }


def _run_cli(module, args_list):
    """Run a CLI module in a subprocess.

    Uses a new session (setsid) so we can kill the entire process group —
    including ClearML's forked background monitor — after the main process exits.
    """
    env = {**os.environ, "CLEARML_OFFLINE_MODE": "1"}
    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(
            [sys.executable, "-m", module, *args_list],
            stdout=out_f,
            stderr=err_f,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        finally:
            out_f.seek(0)
            err_f.seek(0)
            stdout = out_f.read().decode(errors="replace")
            stderr = err_f.read().decode(errors="replace")

    return subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _build_12_crop_manifest(tmp_path: Path) -> tuple[Path, Path]:
    """Build a 12-crop fixture with 4 half-page units; return (manifest_path, output_dir)."""
    page1 = tmp_path / "p1.png"
    page2 = tmp_path / "p2.png"
    _make_grayscale_png(page1, h=200, w=128)
    _make_grayscale_png(page2, h=200, w=128)
    labels = ["אב", "בג", "גד", "דה", "הו", "וז", "זח", "חט", "טי", "יכ", "כל", "לם"]
    rows = []
    for i, lab in enumerate(labels[:3]):
        crop = tmp_path / f"p1_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, i * 10, 8, label=lab))
    for i, lab in enumerate(labels[3:6]):
        crop = tmp_path / f"p1_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, 140 + i * 10, 8, label=lab))
    for i, lab in enumerate(labels[6:9]):
        crop = tmp_path / f"p2_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, i * 10, 8, label=lab))
    for i, lab in enumerate(labels[9:12]):
        crop = tmp_path / f"p2_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, 140 + i * 10, 8, label=lab))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    return manifest, tmp_path / "out"


# ---------------------------------------------------------------------------
# Test 1: CLI defaults
# ---------------------------------------------------------------------------


def test_build_parser_defaults():
    from src.evaluate import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--manifest", "m.csv"])
    assert args.output_dir == Path("outputs/model")
    assert args.val_frac == pytest.approx(0.2)
    assert args.batch_size == 8
    assert args.num_workers == 0


# ---------------------------------------------------------------------------
# Tests 2-4: Guard exit codes (in-process to avoid ClearML init overhead)
# ---------------------------------------------------------------------------


@patch("src.evaluate.Task")
def test_missing_manifest_exits_2(mock_task_cls, tmp_path):
    import io

    from src import evaluate as eval_mod

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.evaluate",
        "--manifest", str(tmp_path / "nope.csv"),
        "--output_dir", str(tmp_path / "out"),
    ]
    try:
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            rc = eval_mod.main()
    finally:
        sys.argv = argv_backup

    assert rc == 2
    assert "--manifest does not exist" in captured.getvalue()


@patch("src.evaluate.Task")
def test_missing_checkpoint_exits_3(mock_task_cls, tmp_path):
    import io

    from src import evaluate as eval_mod

    manifest, out_dir = _build_12_crop_manifest(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    argv_backup = sys.argv[:]
    sys.argv = [
        "src.evaluate",
        "--manifest", str(manifest),
        "--output_dir", str(out_dir),
    ]
    try:
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            rc = eval_mod.main()
    finally:
        sys.argv = argv_backup

    assert rc == 3
    assert "checkpoint not found" in captured.getvalue()


@patch("src.evaluate.Task")
def test_missing_charset_exits_4(mock_task_cls, tmp_path):
    import io

    from src import evaluate as eval_mod

    manifest, out_dir = _build_12_crop_manifest(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint.pt").write_bytes(b"")
    argv_backup = sys.argv[:]
    sys.argv = [
        "src.evaluate",
        "--manifest", str(manifest),
        "--output_dir", str(out_dir),
    ]
    try:
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            rc = eval_mod.main()
    finally:
        sys.argv = argv_backup

    assert rc == 4
    assert "charset not found" in captured.getvalue()


# ---------------------------------------------------------------------------
# Test 5 + 7: End-to-end smoke — schema verification after train + evaluate
# ---------------------------------------------------------------------------


def test_evaluate_after_train_writes_report_with_correct_schema(tmp_path):
    manifest, out_dir = _build_12_crop_manifest(tmp_path)

    # Step 1: train for 1 epoch to produce checkpoint.pt + charset.json
    train_result = _run_cli(
        "src.train_ctc",
        [
            "--manifest",
            str(manifest),
            "--output_dir",
            str(out_dir),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--lr",
            "1e-3",
            "--min_labeled",
            "12",
        ],
    )
    assert train_result.returncode == 0, f"train stderr={train_result.stderr}"
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "charset.json").exists()

    # Step 2: evaluate
    eval_result = _run_cli(
        "src.evaluate",
        ["--manifest", str(manifest), "--output_dir", str(out_dir)],
    )
    assert eval_result.returncode == 0, f"eval stderr={eval_result.stderr}"
    assert "cer=" in eval_result.stdout
    assert "exact_match_rate=" in eval_result.stdout

    report_path = out_dir / "eval_report.csv"
    assert report_path.exists()
    report = pd.read_csv(report_path)
    assert list(report.columns) == ["image_path", "target", "prediction", "is_exact"]
    assert len(report) >= 1


# ---------------------------------------------------------------------------
# Test 6: exact_match_rate formula (lightweight, no model)
# ---------------------------------------------------------------------------


def test_exact_match_rate_formula():
    df = pd.DataFrame({"is_exact": [True, True, False, False]})
    assert df["is_exact"].sum() / len(df) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 8: split reproducibility — val_frac flows through unchanged
# ---------------------------------------------------------------------------


@patch("src.evaluate.Task")
def test_split_units_called_with_cli_val_frac(mock_task_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    manifest, out_dir = _build_12_crop_manifest(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stub artifacts to get past the file-existence guards
    (out_dir / "checkpoint.pt").write_bytes(b"")
    (out_dir / "charset.json").write_text(json.dumps([]), encoding="utf-8")

    captured: dict[str, float] = {}

    def fake_split_units(units, val_frac=0.2):
        captured["val_frac"] = val_frac
        # Return empty val to short-circuit before model load
        return list(units.keys()), []

    from src import evaluate as eval_mod

    with patch("src.evaluate.split_units", side_effect=fake_split_units):
        argv_backup = sys.argv[:]
        sys.argv = [
            "src.evaluate",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(out_dir),
            "--val_frac",
            "0.4",
        ]
        try:
            rc = eval_mod.main()
        finally:
            sys.argv = argv_backup

    assert captured["val_frac"] == pytest.approx(0.4)
    assert rc == 5  # forced empty-val path
