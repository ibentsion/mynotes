---
phase: 02-review-annotation
plan: "03"
subsystem: review-app
tags: [streamlit, autosave, rtl, clearml, ux, checkpoint]
status: complete
self_check: PASSED
---

## What Was Built

Extended `review_app.py` with the full edit surface and fast-path labeling:

- **Enter-to-advance label flow** — `st.text_input` with `on_change`; pressing Enter saves the label, sets status=`labeled`, advances to next crop
- **RTL Hebrew input** — CSS injection (`direction:rtl`) on all textareas/inputs
- **Status selectbox** — manual override for `skip/bad_seg/merge_needed`; autosaves on change
- **Notes field** — free-text, autosaves on change
- **Atomic autosave** — `write_manifest_atomic()` via tempfile+os.replace on every edit
- **Sidebar live counts** — per-status metrics update after every save
- **Sync to ClearML button** — calls `sync_review_to_clearml()` from Plan 02-02
- **sys.path fix** — project root inserted before third-party imports so `src.*` resolves under Streamlit
- **Deprecation fixes** — `use_container_width` → `width=`; image capped at 240px

## Files Modified

- `src/review_app.py` — full edit surface, autosave, sync button, layout (interactions above image)
- `tests/test_review_app_io.py` — headless tests for `update_manifest_row` and `write_manifest_atomic`

## Key Links Verified

- `review_app.py` → `manifest.csv` via `write_manifest_atomic` ✓
- `review_app.py` → `sync_review_to_clearml()` via sidebar button ✓
- `review_app.py` → RTL CSS injection ✓
- `review_app.py` → `MANIFEST_COLUMNS` column ordering on every save ✓

## Human Verification

Approved by user 2026-04-25. Confirmed:
- App loads without errors
- Enter on label saves, sets status=labeled, advances to next crop
- RTL input, status selectbox, notes autosave working
- Image displayed at 240px below edit surface
