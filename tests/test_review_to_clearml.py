import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.manifest_schema import MANIFEST_COLUMNS


@pytest.fixture
def valid_manifest(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "crop_path": str(tmp_path / "crops" / "a.png"),
                "pdf_path": "x.pdf", "page_num": 1, "x": 0, "y": 0, "w": 10, "h": 10,
                "area": 100, "is_flagged": True, "flag_reasons": "margin",
                "status": "unlabeled", "label": "", "notes": "",
            },
            {
                "crop_path": str(tmp_path / "crops" / "b.png"),
                "pdf_path": "x.pdf", "page_num": 1, "x": 0, "y": 0, "w": 10, "h": 10,
                "area": 100, "is_flagged": False, "flag_reasons": "",
                "status": "labeled", "label": "שלום", "notes": "",
            },
            {
                "crop_path": str(tmp_path / "crops" / "c.png"),
                "pdf_path": "x.pdf", "page_num": 1, "x": 0, "y": 0, "w": 10, "h": 10,
                "area": 100, "is_flagged": False, "flag_reasons": "",
                "status": "skip", "label": "", "notes": "blurry",
            },
        ]
    )
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    return path


def test_summarize_status_counts_empty_dataframe():
    from src.review_to_clearml import summarize_status_counts

    df = pd.DataFrame(columns=MANIFEST_COLUMNS)
    result = summarize_status_counts(df)
    assert result == {
        "unlabeled": 0,
        "labeled": 0,
        "skip": 0,
        "bad_seg": 0,
        "merge_needed": 0,
    }


def test_summarize_status_counts_known_statuses(valid_manifest):
    from src.review_to_clearml import summarize_status_counts

    df = pd.read_csv(valid_manifest)
    result = summarize_status_counts(df)
    assert result == {
        "unlabeled": 1,
        "labeled": 1,
        "skip": 1,
        "bad_seg": 0,
        "merge_needed": 0,
    }


def test_summarize_status_counts_ignores_unknown_status(tmp_path):
    from src.review_to_clearml import summarize_status_counts

    df = pd.DataFrame(
        [{"status": "unlabeled"}, {"status": "labeled"}, {"status": "garbage"}]
    )
    result = summarize_status_counts(df)
    assert result["unlabeled"] == 1
    assert result["labeled"] == 1
    assert "garbage" not in result
    assert set(result.keys()) == {"unlabeled", "labeled", "skip", "bad_seg", "merge_needed"}


@patch("src.review_to_clearml.init_task")
@patch("src.review_to_clearml.upload_file_artifact")
def test_sync_review_to_clearml_uploads_manifest_and_logs_counts(
    mock_upload, mock_init_task, valid_manifest
):
    from src.review_to_clearml import sync_review_to_clearml

    mock_task = MagicMock()
    mock_init_task.return_value = mock_task
    logger = mock_task.get_logger.return_value

    sync_review_to_clearml(valid_manifest)

    mock_init_task.assert_called_once_with(
        "handwriting-hebrew-ocr", "manual_review_summary", tags=["phase-2"]
    )
    mock_upload.assert_called_once_with(mock_task, "manifest", valid_manifest)

    # SYNC-01: per-status scalar logging
    logged_series = {
        call.kwargs.get("series") for call in logger.report_scalar.call_args_list
    }
    assert logged_series == {"unlabeled", "labeled", "skip", "bad_seg", "merge_needed"}
    for call in logger.report_scalar.call_args_list:
        assert call.kwargs.get("title") == "status_counts"
        assert call.kwargs.get("iteration") == 0


@patch("src.review_to_clearml.init_task")
@patch("src.review_to_clearml.upload_file_artifact")
def test_sync_review_to_clearml_rejects_bad_schema(mock_upload, mock_init_task, tmp_path):
    from src.review_to_clearml import sync_review_to_clearml

    bad = tmp_path / "manifest.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="schema"):
        sync_review_to_clearml(bad)
    mock_upload.assert_not_called()


def test_main_missing_manifest_returns_nonzero(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.review_to_clearml",
            "--manifest",
            str(tmp_path / "does_not_exist.csv"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "manifest" in (result.stdout + result.stderr).lower()
