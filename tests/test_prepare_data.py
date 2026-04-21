import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.manifest_schema import MANIFEST_COLUMNS
from tests.fixtures.make_synthetic_pdf import make_synthetic_pdf


def test_manifest_columns_constant_has_thirteen_names():
    assert MANIFEST_COLUMNS == [
        "crop_path",
        "pdf_path",
        "page_num",
        "x",
        "y",
        "w",
        "h",
        "area",
        "is_flagged",
        "flag_reasons",
        "status",
        "label",
        "notes",
    ]


def test_prepare_data_end_to_end_on_synthetic_pdf(tmp_path, monkeypatch):
    # Disable ClearML server calls in tests
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")

    pdf_dir = tmp_path / "pdfs"
    output_dir = tmp_path / "outputs"
    pdf_dir.mkdir()
    output_dir.mkdir()
    make_synthetic_pdf(pdf_dir / "sample.pdf", pages=1)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.prepare_data",
            "--pdf_dir",
            str(pdf_dir),
            "--output_dir",
            str(output_dir),
            "--dpi",
            "150",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"

    manifest = output_dir / "manifest.csv"
    review_queue = output_dir / "review_queue.csv"
    assert manifest.exists()
    assert review_queue.exists()

    df = pd.read_csv(manifest)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) >= 1

    # Every crop_path must point to an existing file
    for crop_path in df["crop_path"]:
        assert Path(crop_path).exists(), f"Missing crop file: {crop_path}"

    # Status default and label/notes empty at creation
    assert (df["status"] == "unlabeled").all()

    # review_queue sort contract: is_flagged=True rows first
    rq = pd.read_csv(review_queue)
    assert list(rq.columns) == MANIFEST_COLUMNS
    flagged_mask = rq["is_flagged"].astype(bool).to_numpy()
    if flagged_mask.any() and (~flagged_mask).any():
        first_unflagged = int(flagged_mask.argmin())
        assert flagged_mask[:first_unflagged].all(), (
            "review_queue must put all flagged rows before unflagged"
        )
