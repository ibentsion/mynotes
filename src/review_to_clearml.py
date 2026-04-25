"""review_to_clearml.py — sync manifest.csv to ClearML task manual_review_summary.

SYNC-01: per-status counts logged as scalars on the ClearML task.
SYNC-02: manifest.csv uploaded as a task artifact named "manifest".

Reused in-process by the Streamlit sync button (Plan 03) via sync_review_to_clearml(...).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.clearml_utils import init_task, upload_file_artifact
from src.manifest_schema import MANIFEST_COLUMNS

PROJECT_NAME = "handwriting-hebrew-ocr"
TASK_NAME = "manual_review_summary"
KNOWN_STATUSES: tuple[str, ...] = (
    "unlabeled",
    "labeled",
    "skip",
    "bad_seg",
    "merge_needed",
)


def summarize_status_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return a dict with one entry per known status (zero-filled for missing)."""
    counts: dict[str, int] = {status: 0 for status in KNOWN_STATUSES}
    if "status" not in df.columns or len(df) == 0:
        return counts
    observed = df["status"].astype(str).value_counts().to_dict()
    for status in KNOWN_STATUSES:
        counts[status] = int(observed.get(status, 0))
    return counts


def _validate_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"manifest schema mismatch.\nExpected: {MANIFEST_COLUMNS}\nGot: {list(df.columns)}"
        )


def sync_review_to_clearml(manifest_path: Path) -> dict[str, int]:
    """Upload manifest + per-status counts to ClearML task `manual_review_summary`.

    Returns the status counts dict so callers (e.g. the Streamlit sync button) can render
    confirmation in-app.
    """
    df = pd.read_csv(manifest_path)
    _validate_schema(df)

    task = init_task(PROJECT_NAME, TASK_NAME, tags=["phase-2"])
    upload_file_artifact(task, "manifest", manifest_path)

    counts = summarize_status_counts(df)
    logger = task.get_logger()
    for status, value in counts.items():
        logger.report_scalar(
            title="status_counts", series=status, iteration=0, value=int(value)
        )
    return counts


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Upload manifest.csv + status summary to ClearML task manual_review_summary."
    )
    p.add_argument("--manifest", type=Path, required=True)
    return p


def main() -> int:
    # Pitfall 2: ClearML must patch parse_args BEFORE we call it — but we cannot init_task
    # until we know there is a manifest to upload. Resolve by parsing the path FIRST with a
    # throw-away parser that only reads --manifest, validate the path, then init_task, then
    # do the actual sync. This mirrors prepare_data.py where init_task happens before the
    # main parse but AFTER any sanity-checks that don't require ClearML.
    # However, prepare_data.py's exact rule is "init_task before parse_args()", so to stay
    # consistent we init_task first and let it short-circuit only if the file is missing.
    parser = _build_parser()
    # Pre-flight: check manifest path BEFORE Task.init so an obvious typo doesn't create a
    # spurious ClearML task. This trades strict Pitfall-2 ordering for cheaper failure mode;
    # ClearML still auto-logs args because we call parse_args() between init_task and use.
    # A first parse just for the path:
    ns_preview, _ = parser.parse_known_args()
    manifest_path: Path = ns_preview.manifest
    if not manifest_path.exists():
        print(f"ERROR: --manifest does not exist: {manifest_path}", file=sys.stderr)
        return 2

    # Now init the task so subsequent parse_args() is auto-logged.
    init_task(PROJECT_NAME, TASK_NAME, tags=["phase-2"])
    args = parser.parse_args()
    # Re-resolve manifest path from the auto-logged parse (in case it differs).
    manifest_path = args.manifest

    try:
        counts = sync_review_to_clearml(manifest_path)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    print(f"Synced {manifest_path} to ClearML. Counts: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
