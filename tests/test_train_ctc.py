"""Tests for src/train_ctc.py — run with CLEARML_OFFLINE_MODE=1."""

import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from src.manifest_schema import MANIFEST_COLUMNS

HEBREW_ALPHABET = "אבגדהוזחטיךכלםמןנסעףפץצקרשת"


def _make_grayscale_png(path: Path, h: int = 64, w: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _row(
    crop_path: str,
    page_path: str,
    page_num: int,
    y: int,
    h: int,
    status: str = "labeled",
    label: str = "א",
) -> dict:
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


def _run_cli(args_list: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env["CLEARML_OFFLINE_MODE"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "src.train_ctc", *args_list],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


# ---------------------------------------------------------------------------
# Test 1: CLI parser defaults
# ---------------------------------------------------------------------------


def test_build_parser_has_documented_defaults():
    from src.train_ctc import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--manifest", "m.csv", "--output_dir", "out"])
    assert args.epochs == 30
    assert args.batch_size == 8
    assert args.lr == pytest.approx(1e-3)
    assert args.val_frac == pytest.approx(0.2)
    assert args.min_labeled == 100
    assert args.num_workers == 0


# ---------------------------------------------------------------------------
# Test 2: Status filter — only labeled rows reach build_charset
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_status_filter_keeps_only_labeled(mock_task_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [
        _row(str(tmp_path / "a.png"), str(page_path), 1, 10, 8, status="unlabeled", label=""),
        _row(str(tmp_path / "b.png"), str(page_path), 1, 30, 8, status="labeled", label="ש"),
        _row(str(tmp_path / "c.png"), str(page_path), 1, 50, 8, status="skip", label=""),
    ]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]), h=8, w=128)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    captured: list[list[str]] = []

    def fake_build_charset(labels):
        captured.append(list(labels))
        return ["ש"]

    with (
        patch("src.ctc_utils.build_charset", side_effect=fake_build_charset),
        patch("src.ctc_utils.split_units", return_value=([], [])),  # force exit 5
    ):
        from src.train_ctc import main

        argv_backup = sys.argv[:]
        sys.argv = [
            "src.train_ctc",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(tmp_path / "out"),
            "--min_labeled",
            "1",
        ]
        try:
            rc = main()
        finally:
            sys.argv = argv_backup

    # Filter MUST have already happened by the time build_charset is called
    assert captured == [["ש"]]
    assert rc == 5  # forced empty split path


# ---------------------------------------------------------------------------
# Test 3: min_labeled guard exits 3
# ---------------------------------------------------------------------------


def test_min_labeled_guard_exits_3(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [_row(str(tmp_path / f"c{i}.png"), str(page_path), 1, 10 * i, 8) for i in range(5)]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    result = _run_cli(
        [
            "--manifest",
            str(manifest),
            "--output_dir",
            str(tmp_path / "out"),
            "--min_labeled",
            "10",
        ]
    )
    assert result.returncode == 3, result.stderr
    assert "labeled crops" in result.stderr


# ---------------------------------------------------------------------------
# Test 4: empty label guard exits 4
# ---------------------------------------------------------------------------


def test_empty_label_guard_exits_4(tmp_path):
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [_row(str(tmp_path / f"c{i}.png"), str(page_path), 1, 10 * i, 8) for i in range(10)]
    rows[0]["label"] = ""
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]))
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    result = _run_cli(
        [
            "--manifest",
            str(manifest),
            "--output_dir",
            str(tmp_path / "out"),
            "--min_labeled",
            "10",
        ]
    )
    assert result.returncode == 4, result.stderr


# ---------------------------------------------------------------------------
# Test 5: charset build delegation — wiring only (TRAN-02)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_charset_build_receives_labeled_labels(mock_task_cls, tmp_path, monkeypatch):
    """Verify build_charset is called with labels from labeled rows only."""
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [
        _row(str(tmp_path / "a.png"), str(page_path), 1, 10, 8, status="labeled", label="אב"),
        _row(str(tmp_path / "b.png"), str(page_path), 1, 30, 8, status="unlabeled", label=""),
        _row(str(tmp_path / "c.png"), str(page_path), 1, 50, 8, status="labeled", label="גד"),
    ]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]), h=8, w=128)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    received_labels: list[list[str]] = []

    def capture_build_charset(labels):
        received_labels.append(list(labels))
        return ["א", "ב", "ג", "ד"]

    with (
        patch("src.ctc_utils.build_charset", side_effect=capture_build_charset),
        patch("src.ctc_utils.split_units", return_value=([], [])),
    ):
        from src.train_ctc import main

        argv_backup = sys.argv[:]
        sys.argv = [
            "src.train_ctc",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(tmp_path / "out"),
            "--min_labeled",
            "1",
        ]
        try:
            main()
        finally:
            sys.argv = argv_backup

    assert len(received_labels) == 1
    assert set(received_labels[0]) == {"אב", "גד"}


# ---------------------------------------------------------------------------
# Test 6: Page-leakage check (TRAN-03)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_no_page_leakage_between_train_and_val(mock_task_cls, tmp_path, monkeypatch):
    """Verify that no half-page unit appears in both train and val splits (in-process)."""
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page1 = tmp_path / "p1.png"
    page2 = tmp_path / "p2.png"
    _make_grayscale_png(page1, h=200, w=128)
    _make_grayscale_png(page2, h=200, w=128)

    rows = []
    # 4 half-page units across 2 pages, 3 crops each = 12 labeled
    for i in range(3):
        crop = tmp_path / f"p1_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, i * 10, 8, label="אב"))
    for i in range(3):
        crop = tmp_path / f"p1_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, 140 + i * 10, 8, label="בג"))
    for i in range(3):
        crop = tmp_path / f"p2_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, i * 10, 8, label="גד"))
    for i in range(3):
        crop = tmp_path / f"p2_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, 140 + i * 10, 8, label="דה"))

    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    from src import ctc_utils
    from src.train_ctc import main

    captured_split: list[tuple] = []
    original_split_units = ctc_utils.split_units

    def spy_split_units(units, val_frac=0.2):
        train_keys, val_keys = original_split_units(units, val_frac=val_frac)
        captured_split.append((list(train_keys), list(val_keys)))
        return train_keys, val_keys

    with patch("src.ctc_utils.split_units", side_effect=spy_split_units):
        argv_backup = sys.argv[:]
        sys.argv = [
            "src.train_ctc",
            "--manifest", str(manifest),
            "--output_dir", str(out_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--min_labeled", "12",
        ]
        try:
            rc = main()
        finally:
            sys.argv = argv_backup

    assert rc == 0
    assert len(captured_split) == 1
    train_keys, val_keys = captured_split[0]
    assert set(train_keys).isdisjoint(set(val_keys)), "Page units appear in both train and val!"


# ---------------------------------------------------------------------------
# Test 7: End-to-end smoke test (subprocess) — TRAN-04..08
# ---------------------------------------------------------------------------


def test_train_one_epoch_writes_checkpoint_and_charset(tmp_path):
    page1 = tmp_path / "p1.png"
    page2 = tmp_path / "p2.png"
    _make_grayscale_png(page1, h=200, w=128)
    _make_grayscale_png(page2, h=200, w=128)

    labels = ["אב", "בג", "גד", "דה", "הו", "וז", "זח", "חט", "טי", "יכ", "כל", "לם"]
    rows = []
    # page 1 top half (y centers 5..25)
    for i, lab in enumerate(labels[:3]):
        crop = tmp_path / f"p1_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, i * 10, 8, label=lab))
    # page 1 bottom (y centers 145..165)
    for i, lab in enumerate(labels[3:6]):
        crop = tmp_path / f"p1_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page1), 1, 140 + i * 10, 8, label=lab))
    # page 2 top
    for i, lab in enumerate(labels[6:9]):
        crop = tmp_path / f"p2_top_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, i * 10, 8, label=lab))
    # page 2 bottom
    for i, lab in enumerate(labels[9:12]):
        crop = tmp_path / f"p2_bot_{i}.png"
        _make_grayscale_png(crop, h=8, w=128)
        rows.append(_row(str(crop), str(page2), 2, 140 + i * 10, 8, label=lab))

    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    result = _run_cli(
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
        ]
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "charset.json").exists()
    charset = json.loads((out_dir / "charset.json").read_text(encoding="utf-8"))
    used = set()
    for lab in labels:
        used.update(unicodedata.normalize("NFC", lab))
    assert sorted(used) == charset
    assert "epoch=0" in result.stdout
    assert "best_val_cer=" in result.stdout


# ---------------------------------------------------------------------------
# Test 8: Missing manifest exits 2
# ---------------------------------------------------------------------------


def test_missing_manifest_exits_2(tmp_path):
    result = _run_cli(
        [
            "--manifest",
            str(tmp_path / "nope.csv"),
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )
    assert result.returncode == 2
    assert "--manifest does not exist" in result.stderr


# ---------------------------------------------------------------------------
# Test 9: Augmentation CLI parser defaults
# ---------------------------------------------------------------------------


def test_build_parser_aug_defaults():
    from src.train_ctc import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--manifest", "m.csv", "--output_dir", "out"])
    assert args.aug_copies == 4
    assert args.rotation_max == pytest.approx(7.0)
    assert args.brightness_delta == pytest.approx(0.10)
    assert args.noise_sigma == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Test 10: aug_copies=0 backward-compat (subprocess smoke test)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_aug_copies_zero_backward_compat(mock_task_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
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
    out_dir = tmp_path / "out"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    from src.train_ctc import main

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest", str(manifest),
        "--output_dir", str(out_dir),
        "--epochs", "1",
        "--batch_size", "2",
        "--min_labeled", "12",
        "--aug_copies", "0",
    ]
    try:
        rc = main()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    assert (out_dir / "checkpoint.pt").exists()


# ---------------------------------------------------------------------------
# Test 11: aug_copies>0 prints effective dataset size (Pitfall 4 guard)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_aug_copies_nonzero_prints_effective_size(mock_task_cls, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
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
    out_dir = tmp_path / "out"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    from src.train_ctc import main

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest", str(manifest),
        "--output_dir", str(out_dir),
        "--epochs", "1",
        "--batch_size", "2",
        "--min_labeled", "12",
        "--aug_copies", "2",
    ]
    try:
        rc = main()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    out, _ = capsys.readouterr()
    assert "effective dataset size" in out


# ---------------------------------------------------------------------------
# Test 12: val dataset is always constructed without augment (D-04)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
def test_val_dataset_has_no_augment(mock_task_cls, tmp_path, monkeypatch):
    """Verify val_ds is created with augment=None regardless of --aug_copies."""
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page1 = tmp_path / "p1.png"
    page2 = tmp_path / "p2.png"
    _make_grayscale_png(page1, h=200, w=128)
    _make_grayscale_png(page2, h=200, w=128)

    rows = []
    labels = ["אב", "בג", "גד", "דה", "הו", "וז", "זח", "חט", "טי", "יכ", "כל", "לם"]
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
    out_dir = tmp_path / "out"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    from src import ctc_utils
    from src.train_ctc import main

    captured_calls: list[dict] = []
    original_crop_dataset = ctc_utils.CropDataset

    class SpyCropDataset(original_crop_dataset):
        def __init__(self, df, charset, augment=None, aug_copies=0):
            captured_calls.append({"augment": augment, "aug_copies": aug_copies})
            super().__init__(df, charset, augment=augment, aug_copies=aug_copies)

    with patch("src.ctc_utils.CropDataset", SpyCropDataset):
        argv_backup = sys.argv[:]
        sys.argv = [
            "src.train_ctc",
            "--manifest", str(manifest),
            "--output_dir", str(out_dir),
            "--epochs", "1",
            "--batch_size", "2",
            "--min_labeled", "12",
            "--aug_copies", "2",
        ]
        try:
            rc = main()
        finally:
            sys.argv = argv_backup

    assert rc == 0
    # Exactly 2 CropDataset calls: train_ds and val_ds
    assert len(captured_calls) == 2
    # val_ds is the second call — must have augment=None (D-04)
    assert captured_calls[1]["augment"] is None


# ---------------------------------------------------------------------------
# Test 13: Parser has --enqueue, --queue_name, --dataset_id flags (Plan 02)
# ---------------------------------------------------------------------------


def test_build_parser_has_enqueue_and_dataset_id_flags():
    from src.train_ctc import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--manifest", "m.csv", "--output_dir", "out"])
    assert args.enqueue is False
    assert args.queue_name == "gpu"
    assert args.dataset_id is None


# ---------------------------------------------------------------------------
# Test 14: --enqueue calls execute_remotely AFTER connect (ordering constraint)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.init_task")
def test_enqueue_calls_execute_remotely_after_connect(mock_init_task, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [_row(str(tmp_path / f"c{i}.png"), str(page_path), i + 1, 10, 8) for i in range(3)]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]), h=8, w=128)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    call_order: list[str] = []
    mock_task = MagicMock()
    mock_init_task.return_value = mock_task
    mock_task.connect.side_effect = lambda *a, **kw: call_order.append("connect")
    mock_task.execute_remotely.side_effect = lambda **kw: call_order.append("execute_remotely")

    from src.train_ctc import main

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest", str(manifest),
        "--output_dir", str(tmp_path / "out"),
        "--min_labeled", "1",
        "--enqueue",
    ]
    try:
        main()
    finally:
        sys.argv = argv_backup

    assert "connect" in call_order
    assert "execute_remotely" in call_order
    assert call_order.index("connect") < call_order.index("execute_remotely")


# ---------------------------------------------------------------------------
# Test 15: --enqueue causes task to be created with "gpu" tag (D-08)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.init_task")
def test_enqueue_uses_gpu_tag(mock_init_task, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [_row(str(tmp_path / f"c{i}.png"), str(page_path), i + 1, 10, 8) for i in range(3)]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]), h=8, w=128)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    mock_task = MagicMock()
    mock_init_task.return_value = mock_task

    from src.train_ctc import main

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest", str(manifest),
        "--output_dir", str(tmp_path / "out"),
        "--min_labeled", "1",
        "--enqueue",
    ]
    try:
        main()
    finally:
        sys.argv = argv_backup

    mock_init_task.assert_called_once()
    _, kwargs = mock_init_task.call_args
    tags = kwargs.get("tags", mock_init_task.call_args[0][2] if len(mock_init_task.call_args[0]) > 2 else [])
    assert "gpu" in tags


# ---------------------------------------------------------------------------
# Test 16: --dataset_id calls remap_dataset_paths with correct id (D-09)
# ---------------------------------------------------------------------------


@patch("src.train_ctc.Task")
@patch("src.train_ctc.remap_dataset_paths")
def test_dataset_id_calls_remap(mock_remap, mock_task_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    page_path = tmp_path / "page.png"
    _make_grayscale_png(page_path)
    rows = [_row(str(tmp_path / f"c{i}.png"), str(page_path), i + 1, 10, 8) for i in range(3)]
    for r in rows:
        _make_grayscale_png(Path(r["crop_path"]), h=8, w=128)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)

    # remap returns the original df unchanged so training can continue; force exit via split
    mock_remap.side_effect = lambda df, dataset_id: df

    from src.train_ctc import main

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest", str(manifest),
        "--output_dir", str(tmp_path / "out"),
        "--min_labeled", "1",
        "--dataset_id", "my-dataset-id",
    ]
    with patch("src.ctc_utils.split_units", return_value=([], [])):
        try:
            main()
        finally:
            sys.argv = argv_backup

    mock_remap.assert_called_once()
    _, call_kwargs = mock_remap.call_args
    # dataset_id may be positional
    called_id = call_kwargs.get("dataset_id", mock_remap.call_args[0][1] if len(mock_remap.call_args[0]) > 1 else None)
    assert called_id == "my-dataset-id"

