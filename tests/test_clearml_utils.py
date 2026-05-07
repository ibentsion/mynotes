from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd

from src.clearml_utils import (
    init_task,
    maybe_create_dataset,
    remap_dataset_paths,
    report_manifest_stats,
    upload_file_artifact,
)


@patch("src.clearml_utils.Task")
def test_init_task_calls_task_init_with_correct_args(mock_task_cls):
    init_task("handwriting-hebrew-ocr", "data_prep", tags=["v1"])
    mock_task_cls.init.assert_called_once_with(
        project_name="handwriting-hebrew-ocr",
        task_name="data_prep",
        tags=["v1"],
        reuse_last_task_id=False,
    )


@patch("src.clearml_utils.Task")
def test_init_task_default_tags_is_empty_list(mock_task_cls):
    init_task("p", "t")
    mock_task_cls.init.assert_called_once_with(
        project_name="p", task_name="t", tags=[], reuse_last_task_id=False
    )


def test_upload_file_artifact_stringifies_path():
    task = MagicMock()
    upload_file_artifact(task, "manifest", Path("/tmp/manifest.csv"))
    task.upload_artifact.assert_called_once_with(
        name="manifest", artifact_object="/tmp/manifest.csv"
    )


def test_report_manifest_stats_logs_total_and_flagged():
    df = pd.DataFrame({"is_flagged": [True, False, True, False, False]})
    task = MagicMock()
    logger = task.get_logger.return_value
    report_manifest_stats(task, df)
    logger.report_scalar.assert_any_call(title="crops", series="total", iteration=0, value=5)
    logger.report_scalar.assert_any_call(title="crops", series="flagged", iteration=0, value=2)
    assert logger.report_scalar.call_count == 2


@patch("src.clearml_utils.Dataset")
def test_maybe_create_dataset_full_lifecycle(mock_dataset_cls):
    mock_ds = MagicMock()
    mock_ds.id = "abc123"
    mock_dataset_cls.create.return_value = mock_ds

    result = maybe_create_dataset("proj", "ds_v1", [Path("/tmp/a"), Path("/tmp/b")])

    assert result == "abc123"
    mock_dataset_cls.create.assert_called_once_with(
        dataset_name="ds_v1", dataset_project="proj", use_current_task=True
    )
    assert mock_ds.add_files.call_args_list == [call("/tmp/a"), call("/tmp/b")]
    mock_ds.upload.assert_called_once_with()
    mock_ds.finalize.assert_called_once_with()


# ---------------------------------------------------------------------------
# remap_dataset_paths tests
# ---------------------------------------------------------------------------


def _make_remap_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crop_path": [
                "/original/machine/crops/img_001.png",
                "/original/machine/crops/img_002.png",
            ],
            "page_path": [
                "/original/machine/pages/p1.png",
                "/original/machine/pages/p2.png",
            ],
        }
    )


@patch("src.clearml_utils.Dataset")
def test_remap_dataset_paths_replaces_crop_and_page_paths(mock_dataset_cls):
    mock_dataset_cls.get.return_value.get_local_copy.return_value = "/cache/root"
    df = _make_remap_df()
    result = remap_dataset_paths(df, "abc123")
    assert result["crop_path"].tolist() == [
        "/cache/root/crops/img_001.png",
        "/cache/root/crops/img_002.png",
    ]
    assert result["page_path"].tolist() == [
        "/cache/root/pages/p1.png",
        "/cache/root/pages/p2.png",
    ]


@patch("src.clearml_utils.Dataset")
def test_remap_dataset_paths_does_not_modify_original_df(mock_dataset_cls):
    mock_dataset_cls.get.return_value.get_local_copy.return_value = "/cache/root"
    df = _make_remap_df()
    original_crop = df["crop_path"].tolist()
    remap_dataset_paths(df, "abc123")
    assert df["crop_path"].tolist() == original_crop


@patch("src.clearml_utils.Dataset")
def test_remap_dataset_paths_calls_dataset_get_with_correct_id(mock_dataset_cls):
    mock_dataset_cls.get.return_value.get_local_copy.return_value = "/cache/root"
    remap_dataset_paths(_make_remap_df(), "abc123")
    mock_dataset_cls.get.assert_called_once_with(dataset_id="abc123")
