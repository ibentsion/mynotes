from pathlib import Path

import pandas as pd
from clearml import Dataset, Task


def _fix_flat_symlinks(cache_root: Path) -> None:
    """Create flat→prefixed symlinks for datasets with the upload duplication bug.

    Some datasets were uploaded with files registered at both `subdir/name.ext` (prefixed)
    and `name.ext` (flat). ClearML extracts only the prefixed form but the merge system
    also checks for the flat form. Symlinks from flat paths to subdirs make merges succeed.
    Idempotent — only creates links that don't already exist.
    """
    for subdir in ("crops", "pages", "pdfs"):
        d = cache_root / subdir
        if d.exists():
            for f in d.iterdir():
                flat = cache_root / f.name
                if not flat.exists():
                    flat.symlink_to(f.resolve())


def get_dataset_root(dataset_id: str) -> Path:
    """Return dataset local copy root, pre-fixing parent caches for the flat/prefixed symlink bug.

    For child datasets, downloads each parent's local copy first and applies the
    flat→prefixed symlink fix before triggering the child merge. This ensures the
    merge succeeds even if the parent was uploaded with the duplication bug.
    """
    ds = Dataset.get(dataset_id=dataset_id)
    parent_ids: list[str] = list(getattr(ds, "_dependency_chunk_lookup", {}).keys())
    for parent_id in parent_ids:
        parent_root = Path(Dataset.get(dataset_id=parent_id).get_local_copy())
        _fix_flat_symlinks(parent_root)
    return Path(ds.get_local_copy())


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
    root = get_dataset_root(dataset_id)
    df = df.copy()
    df["crop_path"] = df["crop_path"].apply(
        lambda p: str(root / "crops" / Path(p).name)
    )
    df["page_path"] = df["page_path"].apply(
        lambda p: str(root / "pages" / Path(p).name)
    )
    return df


def remap_synthetic_paths(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Remap crop_path only — synthetic rows have no page_path (D-10)."""
    root = get_dataset_root(dataset_id)
    df = df.copy()
    df["crop_path"] = df["crop_path"].apply(
        lambda p: str(root / "crops" / Path(p).name)
    )
    return df
