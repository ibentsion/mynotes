"""review_app.py — Streamlit review interface with edit fields, autosave, and ClearML sync.

Run with: `uv run streamlit run src/review_app.py -- --manifest outputs/manifest.csv`
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Streamlit runs this file as a script, adding src/ to sys.path instead of the project root.
# Re-insert the project root so `src.*` absolute imports resolve correctly.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.manifest_schema import MANIFEST_COLUMNS  # noqa: E402
from src.review_state import (
    VALID_FILTERS,
    ReviewState,
    load_state,
    save_state,
    with_filter,
)
from src.review_to_clearml import (
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
    # D-04: RTL CSS for all textareas (Hebrew transcription input)
    st.markdown(
        "<style>textarea { direction: rtl; text-align: right; unicode-bidi: plaintext; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Hebrew OCR — Review Queue")

    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest_df = _load_csv(manifest_path, "manifest.csv")
    # Session-state cache of the manifest (so edits persist across Streamlit reruns)
    if "manifest_df" not in st.session_state:
        st.session_state["manifest_df"] = manifest_df
    manifest_df = st.session_state["manifest_df"]

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

    # D-05: live status counts in sidebar
    counts = summarize_status_counts(manifest_df)
    flagged_count = int(manifest_df["is_flagged"].astype(bool).sum())
    st.sidebar.divider()
    st.sidebar.subheader("Status")
    st.sidebar.metric("unlabeled", counts["unlabeled"])
    st.sidebar.metric("flagged", flagged_count)
    st.sidebar.metric("labeled", counts["labeled"])
    st.sidebar.metric("skip", counts["skip"])
    st.sidebar.metric("bad_seg", counts["bad_seg"])
    st.sidebar.metric("merge_needed", counts["merge_needed"])

    # D-07: sync to ClearML button
    st.sidebar.divider()
    if st.sidebar.button("Sync to ClearML", width="stretch"):
        # Persist any pending edits first
        write_manifest_atomic(manifest_path, manifest_df)
        try:
            synced = sync_review_to_clearml(manifest_path)
            st.sidebar.success(f"Synced — {synced}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Sync failed: {exc}")

    # Compute filtered queue from ordered_df
    filtered_paths = _filter_queue(ordered_df, st.session_state["filter"])

    # Clamp index
    if st.session_state["index"] >= len(filtered_paths):
        st.session_state["index"] = max(0, len(filtered_paths) - 1)

    # Sidebar — Prev/Next
    col_prev, col_next = st.sidebar.columns(2)
    if col_prev.button(
        "◀ Prev", width="stretch", disabled=st.session_state["index"] <= 0
    ):
        st.session_state["index"] = max(0, st.session_state["index"] - 1)
        save_state(
            state_path, ReviewState(st.session_state["filter"], st.session_state["index"])
        )
        st.rerun()
    if col_next.button(
        "Next ▶",
        width="stretch",
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
        st.image(current_path, width="stretch")
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

    # REVW-04: status selector
    current_status = str(current_row["status"])
    statuses = list(KNOWN_STATUSES)
    status_index = statuses.index(current_status) if current_status in KNOWN_STATUSES else 0
    new_status = st.selectbox(
        "Status",
        statuses,
        index=status_index,
        key=f"status_{current_path}",
    )

    # REVW-03: Hebrew transcription text area (RTL via global CSS)
    current_label = str(current_row["label"]) if pd.notna(current_row["label"]) else ""
    new_label = st.text_area(
        "Transcription (Hebrew)",
        value=current_label,
        height=120,
        key=f"label_{current_path}",
    )

    # REVW-05: free-text review notes
    current_notes = str(current_row["notes"]) if pd.notna(current_row["notes"]) else ""
    new_notes = st.text_area(
        "Notes",
        value=current_notes,
        height=80,
        key=f"notes_{current_path}",
    )

    # REVW-06 + D-09: autosave on any change
    changed = (
        new_status != current_status
        or new_label != current_label
        or new_notes != current_notes
    )
    if changed:
        manifest_df = update_manifest_row(
            manifest_df,
            current_path,
            label=new_label,
            status=new_status,
            notes=new_notes,
        )
        st.session_state["manifest_df"] = manifest_df
        write_manifest_atomic(manifest_path, manifest_df)
        st.toast("Saved", icon="✅")
        st.rerun()


if __name__ == "__main__":
    main()
