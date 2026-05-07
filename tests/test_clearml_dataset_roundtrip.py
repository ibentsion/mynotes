"""Integration test: real ClearML dataset upload->download round-trip.

Verifies the target_folder fix in maybe_create_dataset produces datasets
whose extracted files live where remap_dataset_paths expects them.
"""

import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
from clearml import Dataset

from src.clearml_utils import maybe_create_dataset, remap_dataset_paths
from src.manifest_schema import MANIFEST_COLUMNS

# Credentials are in ~/clearml.conf; tests run unconditionally locally.
# In CI without clearml.conf, mark with CLEARML_OFFLINE_MODE=1 or skip via -m.


def _png(path: Path, h: int = 200, w: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w), 200, dtype=np.uint8))


def _row(crop: Path, page: Path, page_num: int, y: int, label: str) -> dict:
    return {
        "crop_path": str(crop),
        "pdf_path": "synthetic.pdf",
        "page_path": str(page),
        "page_num": page_num,
        "x": 0,
        "y": y,
        "w": 128,
        "h": 8,
        "area": 1024,
        "is_flagged": False,
        "flag_reasons": "",
        "status": "labeled",
        "label": label,
        "notes": "",
    }


def _build_minimal_dataset(tmp: Path) -> tuple[str, pd.DataFrame]:
    """Create synthetic pages/crops/manifest and upload to ClearML. Returns (ds_id, df)."""
    pages = tmp / "pages"
    crops = tmp / "crops"
    p1 = pages / "p1.png"
    p2 = pages / "p2.png"
    # 200x128 pages so top half is y<100, bottom half is y>=100
    _png(p1)
    _png(p2)

    labels = ["אב", "בג", "גד", "דה", "הו", "וז", "זח", "חט", "טי", "יכ", "כל", "לם"]
    rows = []
    # page 1 top (center_y: 4, 14, 24 — all < 100)
    for i, lab in enumerate(labels[:3]):
        c = crops / f"p1_top_{i}.png"
        _png(c, h=8)
        rows.append(_row(c, p1, 1, i * 10, lab))
    # page 1 bottom (center_y: 144, 154, 164 — all >= 100)
    for i, lab in enumerate(labels[3:6]):
        c = crops / f"p1_bot_{i}.png"
        _png(c, h=8)
        rows.append(_row(c, p1, 1, 140 + i * 10, lab))
    # page 2 top
    for i, lab in enumerate(labels[6:9]):
        c = crops / f"p2_top_{i}.png"
        _png(c, h=8)
        rows.append(_row(c, p2, 2, i * 10, lab))
    # page 2 bottom
    for i, lab in enumerate(labels[9:12]):
        c = crops / f"p2_bot_{i}.png"
        _png(c, h=8)
        rows.append(_row(c, p2, 2, 140 + i * 10, lab))

    manifest = tmp / "manifest.csv"
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.to_csv(manifest, index=False)

    ds_id = maybe_create_dataset(
        project="handwriting-hebrew-ocr-test",
        dataset_name=f"oa8-roundtrip-{uuid.uuid4().hex[:8]}",
        folders=[(pages, "pages"), (crops, "crops")],
        files=[manifest],
    )
    return ds_id, df


def test_dataset_roundtrip_extracts_under_pages_and_crops(tmp_path):
    ds_id, _ = _build_minimal_dataset(tmp_path)
    root = Path(Dataset.get(dataset_id=ds_id).get_local_copy())
    assert (root / "pages" / "p1.png").exists()
    assert (root / "crops" / "p1_top_0.png").exists()
    assert (root / "manifest.csv").exists()


def test_dataset_roundtrip_remap_paths_resolve(tmp_path):
    ds_id, df = _build_minimal_dataset(tmp_path)
    remapped = remap_dataset_paths(df, ds_id)
    for p in remapped["crop_path"]:
        assert Path(p).exists(), p
    for p in remapped["page_path"]:
        assert Path(p).exists(), p


@patch("src.train_ctc.init_task")
@patch("src.train_ctc.Task")
def test_dataset_roundtrip_one_epoch_training(mock_task_cls, mock_init_task, tmp_path):
    """Full end-to-end: upload synthetic dataset, download via ClearML, train 1 epoch."""
    mock_task = MagicMock()
    mock_init_task.return_value = mock_task
    mock_task_cls.current_task.return_value = mock_task

    ds_id, _ = _build_minimal_dataset(tmp_path)
    out = tmp_path / "out"
    manifest_in_ds = Path(Dataset.get(dataset_id=ds_id).get_local_copy()) / "manifest.csv"

    argv_backup = sys.argv[:]
    sys.argv = [
        "src.train_ctc",
        "--manifest",
        str(manifest_in_ds),
        "--output_dir",
        str(out),
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--min_labeled",
        "12",
        "--aug_copies",
        "0",
        "--rnn_hidden",
        "128",
        "--num_layers",
        "1",
        "--dataset_id",
        ds_id,
    ]
    try:
        from src.train_ctc import main

        rc = main()
    finally:
        sys.argv = argv_backup

    assert rc == 0
    assert (out / "checkpoint.pt").exists()
    assert (out / "charset.json").exists()
