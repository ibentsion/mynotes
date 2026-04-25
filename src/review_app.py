"""review_app.py — Streamlit review interface with edit fields, autosave, and ClearML sync.

Run with: `uv run streamlit run src/review_app.py -- --manifest outputs/manifest.csv`
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.manifest_schema import MANIFEST_COLUMNS
from src.review_state import (
    VALID_FILTERS,
    ReviewState,
    load_state,
    save_state,
    with_filter,
)
from src.review_to_clearml import (  # noqa: F401 — used in main() (Task 2)
    KNOWN_STATUSES,
    summarize_status_counts,
    sync_review_to_clearml,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamlit review app.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(argv)


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != MANIFEST_COLUMNS:
        st.error(
            f"{label} schema mismatch.\nExpected: {MANIFEST_COLUMNS}\nGot: {list(df.columns)}"
        )
        st.stop()
    return df


def _filter_queue(df: pd.DataFrame, filter_name: str) -> list[str]:
    if filter_name == "unlabeled":
        mask = df["status"] == "unlabeled"
    elif filter_name == "flagged":
        mask = df["is_flagged"].astype(bool)
    elif filter_name == "labeled":
        mask = df["status"] == "labeled"
    else:
        mask = pd.Series(True, index=df.index)
    return df.loc[mask, "crop_path"].astype(str).tolist()


def _resolve_queue(manifest_df: pd.DataFrame, queue_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a manifest-rows DataFrame ordered the same as review_queue (D-02).

    Falls back to manifest order if review_queue is unavailable.
    """
    if queue_df is None:
        return manifest_df.reset_index(drop=True)
    ordered = queue_df[["crop_path"]].merge(
        manifest_df, on="crop_path", how="left", sort=False
    )
    # Drop any queue rows whose crop_path isn't present in the manifest (defensive)
    return ordered.dropna(subset=["pdf_path"]).reset_index(drop=True)


def update_manifest_row(
    df: pd.DataFrame,
    crop_path: str,
    *,
    label: str | None = None,
    status: str | None = None,
    notes: str | None = None,
) -> pd.DataFrame:
    """Return a new DataFrame with the named row's fields updated.

    Raises KeyError if crop_path is not present.
    """
    mask = df["crop_path"] == crop_path
    if not mask.any():
        raise KeyError(f"crop_path not in manifest: {crop_path}")
    new_df = df.copy()
    if label is not None:
        new_df.loc[mask, "label"] = label
    if status is not None:
        new_df.loc[mask, "status"] = status
    if notes is not None:
        new_df.loc[mask, "notes"] = notes
    return new_df[MANIFEST_COLUMNS]


def write_manifest_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write df to path via tempfile + os.replace so partial writes can't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the SAME dir guarantees os.replace stays on the same filesystem.
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".csv.tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df[MANIFEST_COLUMNS].to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    # streamlit forwards args after `--` on the command line
    args = _parse_args(sys.argv[1:])
    manifest_path: Path = args.manifest
    review_queue_path = manifest_path.with_name("review_queue.csv")
    state_path = manifest_path.with_name(".review_state.json")

    st.set_page_config(page_title="Hebrew OCR Review", layout="wide")
    st.title("Hebrew OCR — Review Queue")

    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest_df = _load_csv(manifest_path, "manifest.csv")
    if review_queue_path.exists():
        queue_df = _load_csv(review_queue_path, "review_queue.csv")
    else:
        st.warning(f"review_queue.csv not found at {review_queue_path}; using manifest order.")
        queue_df = None

    ordered_df = _resolve_queue(manifest_df, queue_df)

    # Restore prior session state on first render
    if "filter" not in st.session_state or "index" not in st.session_state:
        persisted: ReviewState = load_state(state_path)
        st.session_state["filter"] = persisted.filter
        st.session_state["index"] = persisted.index

    # Sidebar — filter
    st.sidebar.title("Review Queue")
    prior_filter = st.session_state["filter"]
    chosen = st.sidebar.selectbox(
        "Filter",
        VALID_FILTERS,
        index=VALID_FILTERS.index(prior_filter) if prior_filter in VALID_FILTERS else 0,
        key="filter_selectbox",
    )
    if chosen != prior_filter:
        new_state = with_filter(
            ReviewState(filter=prior_filter, index=st.session_state["index"]), chosen
        )
        st.session_state["filter"] = new_state.filter
        st.session_state["index"] = new_state.index
        save_state(state_path, new_state)
        st.rerun()

    # Compute filtered queue from ordered_df
    filtered_paths = _filter_queue(ordered_df, st.session_state["filter"])

    # Clamp index
    if st.session_state["index"] >= len(filtered_paths):
        st.session_state["index"] = max(0, len(filtered_paths) - 1)

    # Sidebar — Prev/Next
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button(
        "◀ Prev", use_container_width=True, disabled=st.session_state["index"] <= 0
    ):
        st.session_state["index"] = max(0, st.session_state["index"] - 1)
        save_state(
            state_path, ReviewState(st.session_state["filter"], st.session_state["index"])
        )
        st.rerun()
    if col_next.button(
        "Next ▶",
        use_container_width=True,
        disabled=st.session_state["index"] >= len(filtered_paths) - 1,
    ):
        st.session_state["index"] = min(
            len(filtered_paths) - 1, st.session_state["index"] + 1
        )
        save_state(
            state_path, ReviewState(st.session_state["filter"], st.session_state["index"])
        )
        st.rerun()

    # Empty queue
    if not filtered_paths:
        st.info(f"No crops match filter: {st.session_state['filter']}")
        return

    # Position indicator (D-06)
    st.caption(
        f"Crop {st.session_state['index'] + 1} of {len(filtered_paths)}"
        f" ({st.session_state['filter']})"
    )

    current_path = filtered_paths[st.session_state["index"]]
    current_row = manifest_df.loc[manifest_df["crop_path"] == current_path].iloc[0]

    if Path(current_path).exists():
        st.image(current_path, use_container_width=True)
    else:
        st.error(f"Crop image missing on disk: {current_path}")

    with st.expander("Crop metadata"):
        st.write(
            {
                "page_num": int(current_row["page_num"]),
                "x": int(current_row["x"]),
                "y": int(current_row["y"]),
                "w": int(current_row["w"]),
                "h": int(current_row["h"]),
                "area": int(current_row["area"]),
                "is_flagged": bool(current_row["is_flagged"]),
                "flag_reasons": (
                    str(current_row["flag_reasons"])
                    if pd.notna(current_row["flag_reasons"])
                    else ""
                ),
            }
        )


if __name__ == "__main__":
    main()
