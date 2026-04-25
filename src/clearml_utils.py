from pathlib import Path

import pandas as pd
from clearml import Dataset, Task


def init_task(
    project: str,
    task_name: str,
    tags: list[str] | None = None,
) -> Task:
    """Initialize a ClearML task. MUST be called before argparse.parse_args()."""
    return Task.init(
        project_name=project,
        task_name=task_name,
        tags=tags if tags is not None else [],
    )


def upload_file_artifact(task: Task, name: str, path: Path) -> None:
    task.upload_artifact(name=name, artifact_object=str(path))


def report_manifest_stats(task: Task, df: pd.DataFrame) -> None:
    logger = task.get_logger()
    logger.report_scalar(title="crops", series="total", iteration=0, value=len(df))
    logger.report_scalar(
        title="crops",
        series="flagged",
        iteration=0,
        value=int(df["is_flagged"].sum()),
    )


def maybe_create_dataset(
    project: str,
    dataset_name: str,
    folders: list[Path],
    files: list[Path] | None = None,
) -> str:
    """Create, populate, upload, and finalize a ClearML dataset. Returns dataset id."""
    ds = Dataset.create(
        dataset_name=dataset_name, dataset_project=project, use_current_task=True
    )
    for folder in folders:
        ds.add_files(str(folder))
    for file in files or []:
        ds.add_files(str(file))
    ds.upload()
    ds.finalize()
    return ds.id
