from unittest.mock import patch

import pandas as pd
import pytest

from src.manifest_schema import MANIFEST_COLUMNS
from src.review_app import update_manifest_row, write_manifest_atomic


def _row(crop_path: str, status: str = "unlabeled", label: str = "", notes: str = "") -> dict:
    return {
        "crop_path": crop_path,
        "pdf_path": "x.pdf",
        "page_num": 1,
        "x": 0, "y": 0, "w": 10, "h": 10,
        "area": 100,
        "is_flagged": False,
        "flag_reasons": "",
        "status": status,
        "label": label,
        "notes": notes,
    }


def test_update_manifest_row_changes_only_target_row():
    df = pd.DataFrame([_row("a.png"), _row("b.png")], columns=MANIFEST_COLUMNS)
    out = update_manifest_row(df, "a.png", label="שלום", status="labeled")
    a = out.loc[out["crop_path"] == "a.png"].iloc[0]
    b = out.loc[out["crop_path"] == "b.png"].iloc[0]
    assert a["label"] == "שלום"
    assert a["status"] == "labeled"
    assert b["label"] == ""
    assert b["status"] == "unlabeled"


def test_update_manifest_row_preserves_column_order():
    df = pd.DataFrame([_row("a.png")], columns=MANIFEST_COLUMNS)
    out = update_manifest_row(df, "a.png", notes="needs review")
    assert list(out.columns) == MANIFEST_COLUMNS


def test_update_manifest_row_raises_on_missing_path():
    df = pd.DataFrame([_row("a.png")], columns=MANIFEST_COLUMNS)
    with pytest.raises(KeyError, match="crop_path"):
        update_manifest_row(df, "nope.png", label="x")


def test_write_manifest_atomic_round_trip(tmp_path):
    df = pd.DataFrame([_row("a.png", status="labeled", label="שלום")], columns=MANIFEST_COLUMNS)
    target = tmp_path / "manifest.csv"
    write_manifest_atomic(target, df)
    assert target.exists()
    loaded = pd.read_csv(target)
    assert list(loaded.columns) == MANIFEST_COLUMNS
    assert loaded.iloc[0]["label"] == "שלום"
    assert loaded.iloc[0]["status"] == "labeled"


def test_write_manifest_atomic_preserves_existing_on_failure(tmp_path):
    target = tmp_path / "manifest.csv"
    original = pd.DataFrame([_row("a.png", label="orig")], columns=MANIFEST_COLUMNS)
    write_manifest_atomic(target, original)
    original_bytes = target.read_bytes()

    broken = pd.DataFrame([_row("a.png", label="broken")], columns=MANIFEST_COLUMNS)
    with (
        patch("src.review_app.os.replace", side_effect=OSError("simulated crash")),
        pytest.raises(OSError),
    ):
        write_manifest_atomic(target, broken)

    # Original file still intact
    assert target.read_bytes() == original_bytes
    # No leftover .tmp files
    leftovers = list(tmp_path.glob(".manifest.*.csv.tmp"))
    assert leftovers == [], f"Leftover temp files: {leftovers}"
