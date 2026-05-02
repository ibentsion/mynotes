"""review_app.py — Streamlit review interface with edit fields, autosave, and ClearML sync.

Run with: `uv run streamlit run src/review_app.py -- --manifest data/manifest.csv`
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as st_components  # noqa: E402
import torch  # noqa: E402

from src.ctc_utils import (  # noqa: E402
    CRNN,
    load_charset,
    predict_single,
    resolve_device,
)
from src.manifest_schema import MANIFEST_COLUMNS  # noqa: E402
from src.review_state import (  # noqa: E402
    VALID_FILTERS,
    ReviewState,
    load_state,
    save_state,
    with_filter,
)
from src.review_to_clearml import (  # noqa: E402
    KNOWN_STATUSES,
    summarize_status_counts,
    sync_review_to_clearml,
)


@st.cache_resource
def _load_model(model_dir: str) -> tuple[CRNN, list[str], torch.device] | None:
    """Load CRNN + charset from model_dir. Returns None if either artifact is missing.

    model_dir is str (not Path) so Streamlit can hash it for the cache key.
    """
    md = Path(model_dir)
    ckpt = md / "checkpoint.pt"
    cs = md / "charset.json"
    if not ckpt.exists() or not cs.exists():
        return None
    charset = load_charset(cs)
    device = resolve_device()
    model = CRNN(num_classes=len(charset) + 1).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, charset, device


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streamlit review app.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Optional outputs/model dir; enables prediction suggestions on unlabeled crops.",
    )
    return parser.parse_args(argv)


_STR_COLS = ("label", "notes", "flag_reasons")


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: object for c in _STR_COLS})
    # Backfill page_path for manifests generated before it was added to the schema
    if "page_path" not in df.columns:
        df.insert(df.columns.get_loc("page_num"), "page_path", "")
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
    elif filter_name == "auto_labeled":
        mask = df["notes"].astype(str).str.startswith("auto:")
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


def _save_manifest(path: Path, df: pd.DataFrame) -> None:
    """Write manifest and update session-state mtime to prevent stale-cache reload."""
    write_manifest_atomic(path, df)
    st.session_state["manifest_mtime"] = path.stat().st_mtime


def _render_context(
    page_path: str, x: int, y: int, w: int, h: int, scale: int
) -> np.ndarray | None:
    """Return an RGB image of the page region surrounding (x,y,w,h) at `scale` times crop size.

    The crop bounding box is highlighted with a red rectangle.
    """
    page = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    if page is None:
        return None
    page_h, page_w = page.shape
    pad_x = w * (scale - 1) // 2
    pad_y = h * (scale - 1) // 2
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(page_w, x + w + pad_x)
    y2 = min(page_h, y + h + pad_y)
    region = cv2.cvtColor(page[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
    rx, ry = x - x1, y - y1
    cv2.rectangle(region, (rx, ry), (rx + w, ry + h), (0, 0, 255), 3)
    return cv2.cvtColor(region, cv2.COLOR_BGR2RGB)


def main() -> None:
    # streamlit forwards args after `--` on the command line
    args = _parse_args(sys.argv[1:])
    manifest_path: Path = args.manifest
    model_bundle = None
    if args.model_dir is not None:
        model_bundle = _load_model(str(args.model_dir))
        if model_bundle is None:
            st.sidebar.warning(
                f"Model not loaded — checkpoint.pt / charset.json missing in {args.model_dir}"
            )
    review_queue_path = manifest_path.with_name("review_queue.csv")
    state_path = manifest_path.with_name(".review_state.json")

    st.set_page_config(page_title="Hebrew OCR Review", layout="wide")
    st.markdown(
        "<style>textarea { direction: rtl; text-align: right; unicode-bidi: plaintext; }"
        " #MainMenu, footer { visibility: hidden; }</style>",
        unsafe_allow_html=True,
    )

    if not manifest_path.exists():
        st.error(f"Manifest not found: {manifest_path}")
        st.stop()

    manifest_df = _load_csv(manifest_path, "manifest.csv")
    # Reload session-state cache if the file was modified externally (e.g. by auto-label).
    file_mtime = manifest_path.stat().st_mtime
    if ("manifest_df" not in st.session_state
            or st.session_state.get("manifest_mtime", 0) < file_mtime):
        st.session_state["manifest_df"] = manifest_df
        st.session_state["manifest_mtime"] = file_mtime
        st.session_state.pop("_undo_stack", None)
        st.session_state.pop("_redo_stack", None)
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

    # Compute filtered queue early so action buttons can use it
    filtered_paths = _filter_queue(ordered_df, st.session_state["filter"])

    # Clamp index
    if st.session_state["index"] >= len(filtered_paths):
        st.session_state["index"] = max(0, len(filtered_paths) - 1)

    # Sidebar — Prev/Next (top, after filter)
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

    # D-07: sync to ClearML button
    st.sidebar.divider()
    if st.sidebar.button("Sync to ClearML", width="stretch"):
        # Persist any pending edits first
        _save_manifest(manifest_path, manifest_df)
        try:
            synced = sync_review_to_clearml(manifest_path)
            st.sidebar.success(f"Synced — {synced}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Sync failed: {exc}")

    # D-05: live status counts in sidebar (bottom, table layout)
    counts = summarize_status_counts(manifest_df)
    flagged_count = int(manifest_df["is_flagged"].astype(bool).sum())
    st.sidebar.divider()
    st.sidebar.subheader("Status")
    rows = [
        ("unlabeled", counts["unlabeled"]),
        ("flagged", flagged_count),
        ("labeled", counts["labeled"]),
        ("skip", counts["skip"]),
        ("bad_seg", counts["bad_seg"]),
        ("merge_needed", counts["merge_needed"]),
    ]
    table_md = "| | |\n|---|---|\n" + "\n".join(f"| {lbl} | {val} |" for lbl, val in rows)
    st.sidebar.markdown(table_md)

    # Empty queue
    if not filtered_paths:
        st.info(f"No crops match filter: {st.session_state['filter']}")
        return

    current_path = filtered_paths[st.session_state["index"]]
    current_row = manifest_df.loc[manifest_df["crop_path"] == current_path].iloc[0]

    current_status = str(current_row["status"])
    current_label = str(current_row["label"]) if pd.notna(current_row["label"]) else ""
    current_notes = str(current_row["notes"]) if pd.notna(current_row["notes"]) else ""
    label_key = f"label_{current_path}"
    _hist_key = f"_lhist_{current_path}"
    _hist_pos_key = f"_lhist_pos_{current_path}"
    if _hist_key not in st.session_state:
        st.session_state[_hist_key] = [current_label]
        st.session_state[_hist_pos_key] = 0

    # --- Fast-path: Enter on label → save, mark labeled, advance ---
    def _on_label_enter() -> None:
        st.session_state["_label_submitted"] = st.session_state[label_key]

    if st.session_state.get("_label_submitted") is not None:
        submitted = st.session_state.pop("_label_submitted")
        if submitted:
            _hist = st.session_state[_hist_key]
            _pos = st.session_state[_hist_pos_key]
            _new_hist = _hist[: _pos + 1]
            if not _new_hist or _new_hist[-1] != submitted:
                _new_hist.append(submitted)
            st.session_state[_hist_key] = _new_hist
            st.session_state[_hist_pos_key] = len(_new_hist) - 1
            manifest_df = update_manifest_row(
                manifest_df, current_path, label=submitted, status="labeled"
            )
            st.session_state["manifest_df"] = manifest_df
            _save_manifest(manifest_path, manifest_df)
            next_idx = min(len(filtered_paths) - 1, st.session_state["index"] + 1)
            st.session_state["index"] = next_idx
            save_state(state_path, ReviewState(st.session_state["filter"], next_idx))
            st.session_state["_focus_label"] = True
            st.rerun()

    # Re-focus the label input after advancing (zero-height, outside columns)
    if st.session_state.pop("_focus_label", False):
        st_components.html(
            "<script>setTimeout(function(){"
            "var inputs=window.parent.document.querySelectorAll('input[type=\"text\"]');"
            "if(inputs.length>0)inputs[0].focus();"
            "},120);</script>",
            height=0,
        )

    col_interact, col_image = st.columns([2, 3])

    # ── Interaction column ──────────────────────────────────────────────────
    with col_interact:
        st.caption(
            f"Crop {st.session_state['index'] + 1} of {len(filtered_paths)}"
            f" ({st.session_state['filter']})"
        )

        _hist = st.session_state[_hist_key]
        _pos = st.session_state[_hist_pos_key]
        _bcol, _fcol, _ = st.columns([1, 1, 6])
        if _bcol.button("◀", disabled=_pos <= 0, key=f"lhist_back_{current_path}",
                        help="Undo label"):
            _new_pos = _pos - 1
            _prev = _hist[_new_pos]
            st.session_state[_hist_pos_key] = _new_pos
            st.session_state[label_key] = _prev
            manifest_df = update_manifest_row(manifest_df, current_path, label=_prev)
            st.session_state["manifest_df"] = manifest_df
            _save_manifest(manifest_path, manifest_df)
            st.rerun()
        if _fcol.button("▶", disabled=_pos >= len(_hist) - 1,
                        key=f"lhist_fwd_{current_path}", help="Redo label"):
            _new_pos = _pos + 1
            _next = _hist[_new_pos]
            st.session_state[_hist_pos_key] = _new_pos
            st.session_state[label_key] = _next
            manifest_df = update_manifest_row(manifest_df, current_path, label=_next)
            st.session_state["manifest_df"] = manifest_df
            _save_manifest(manifest_path, manifest_df)
            st.rerun()

        if model_bundle is not None and current_status == "unlabeled":
            model, charset, device = model_bundle
            with torch.no_grad():
                suggestion = predict_single(model, charset, device, current_path)
            st.info(f"Model suggests: {suggestion}")
            if st.button("Accept ->", key=f"accept_pred_{current_path}"):
                # Reuse existing save/advance flow — same path as Enter on the text_input.
                st.session_state["_label_submitted"] = suggestion
                st.rerun()

        st.text_input(
            "Label — press Enter to save & next",
            value=current_label,
            key=label_key,
            on_change=_on_label_enter,
        )

        # Status buttons (one-click)
        statuses = list(KNOWN_STATUSES)
        current_status_safe = current_status if current_status in KNOWN_STATUSES else statuses[0]

        def _save_status(new: str) -> None:
            if new == current_status:
                return
            updated = update_manifest_row(manifest_df, current_path, status=new)
            st.session_state["manifest_df"] = updated
            _save_manifest(manifest_path, updated)
            st.toast("Saved", icon="✅")
            st.rerun()

        st.write("**Status**")
        for row_statuses in (statuses[:3], statuses[3:]):
            btn_cols = st.columns(len(row_statuses))
            for btn_col, s in zip(btn_cols, row_statuses, strict=True):
                if btn_col.button(
                    s,
                    key=f"status_btn_{s}_{current_path}",
                    type="primary" if s == current_status_safe else "secondary",
                    use_container_width=True,
                ):
                    _save_status(s)

        current_notes = str(current_row["notes"]) if pd.notna(current_row["notes"]) else ""
        new_notes = st.text_area(
            "Notes", value=current_notes, height=100, key=f"notes_{current_path}"
        )
        if new_notes != current_notes:
            manifest_df = update_manifest_row(manifest_df, current_path, notes=new_notes)
            st.session_state["manifest_df"] = manifest_df
            _save_manifest(manifest_path, manifest_df)
            st.toast("Saved", icon="✅")

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

    # ── Image column ────────────────────────────────────────────────────────
    with col_image:
        if Path(current_path).exists():
            st.image(current_path, width=240)
        else:
            st.error(f"Crop image missing on disk: {current_path}")

        page_path = str(current_row["page_path"]) if pd.notna(current_row["page_path"]) else ""
        if page_path and Path(page_path).exists():
            ctx_key = f"ctx_{current_path}"
            if ctx_key not in st.session_state:
                st.session_state[ctx_key] = None
            c2x, c4x, _ = st.columns([1, 1, 4])
            if c2x.button("2× context", key=f"ctx2_{current_path}"):
                st.session_state[ctx_key] = None if st.session_state[ctx_key] == 2 else 2
                st.session_state.pop("_label_submitted", None)
            if c4x.button("4× context", key=f"ctx4_{current_path}"):
                st.session_state[ctx_key] = None if st.session_state[ctx_key] == 4 else 4
                st.session_state.pop("_label_submitted", None)
            if st.session_state[ctx_key] is not None:
                ctx_img = _render_context(
                    page_path,
                    int(current_row["x"]), int(current_row["y"]),
                    int(current_row["w"]), int(current_row["h"]),
                    st.session_state[ctx_key],
                )
                if ctx_img is not None:
                    st.image(ctx_img, use_container_width=True)
        elif not page_path:
            st.caption("Context view unavailable — re-run prepare-data to enable.")


if __name__ == "__main__":
    main()
