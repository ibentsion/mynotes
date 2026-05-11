from pathlib import Path

import pandas as pd
from clearml import Dataset, Task


def register_requirements(path: str = "requirements.txt") -> None:
    """Register task requirements from file, skipping pip option lines.

    Must be called BEFORE Task.init() (i.e., before init_task()).
    Filters lines starting with '-' (e.g. --extra-index-url) which
    pkg_resources cannot parse.
    """
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            Task.add_requirements(line)


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
        reuse_last_task_id=False,
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
    folders: list[Path | tuple[Path, str]],
    files: list[Path] | None = None,
) -> str:
    """Create, populate, upload, and finalize a ClearML dataset. Returns dataset id."""
    # use_current_task=True attaches to the running task (avoids Task.init conflict, per D-01).
    # When no task is running, pass False so finalize() calls mark_completed() instead of
    # just flush() — otherwise is_final() returns False and get_local_copy() raises.
    use_current_task = Task.current_task() is not None
    ds = Dataset.create(
        dataset_name=dataset_name, dataset_project=project, use_current_task=use_current_task
    )
    for entry in folders:
        if isinstance(entry, tuple):
            folder, dataset_path = entry
            ds.add_files(str(folder), dataset_path=dataset_path)
        else:
            ds.add_files(str(entry))
    for file in files or []:
        ds.add_files(str(file))
    ds.upload()
    ds.finalize()
    return ds.id


def remap_dataset_paths(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Remap manifest crop_path and page_path to ClearML dataset cache root.

    Downloads (or uses cached) dataset identified by dataset_id.
    Returns a copy of df — original is not modified (D-10).
    Cache is reused on subsequent calls with same dataset_id (D-11).
    """
    ds = Dataset.get(dataset_id=dataset_id)
    root = Path(ds.get_local_copy())
    df = df.copy()
    df["crop_path"] = df["crop_path"].apply(
        lambda p: str(root / "crops" / Path(p).name)
    )
    df["page_path"] = df["page_path"].apply(
        lambda p: str(root / "pages" / Path(p).name)
    )
    return df
