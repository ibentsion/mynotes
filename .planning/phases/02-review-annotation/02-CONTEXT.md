# Phase 2: Review & Annotation - Context

**Gathered:** 2026-04-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a local Streamlit review app (`review_app.py`) that lets the reviewer work through
cropped regions, transcribe Hebrew text, set statuses, and add notes — with all edits
persisted immediately to `manifest.csv`. A companion CLI script (`review_to_clearml.py`)
syncs the updated manifest to ClearML; a "Sync to ClearML" button in the app also
triggers this sync.

Out of scope for this phase: model training, evaluation, split/merge editing, FiftyOne.

</domain>

<decisions>
## Implementation Decisions

### Review Flow
- **D-01:** Sequential queue navigation — one crop at a time, Prev/Next buttons to move
  through the current filter. No list/gallery view.
- **D-02:** Filter selector drives the queue (unlabeled / flagged / labeled / all). The
  queue is ordered as review_queue.csv defines (flagged first, then large/mixed, etc.).
- **D-03:** Session position is persisted. On close/reopen, the app resumes at the last
  filter + crop index via a small state file (e.g., `.review_state.json` next to
  manifest.csv). Position resets to 0 when the user changes the filter.

### Hebrew Text Input
- **D-04:** RTL direction enforced on the transcription field via `st.markdown` CSS
  injection (`direction: rtl; text-align: right`). Use `st.text_area` (not `text_input`)
  so multi-line Hebrew text displays correctly right-to-left.

### Progress Visibility
- **D-05:** Sidebar shows live per-status counts: unlabeled / flagged / labeled / skip /
  bad_seg / merge_needed. Updates after every edit. No progress bar needed — counts are
  sufficient.
- **D-06:** Above the crop image, show current position: "Crop N of M (filter_name)".

### ClearML Sync
- **D-07:** `review_to_clearml.py` is a standalone CLI script (satisfies SYNC-01/02).
  Additionally, a "Sync to ClearML" button appears in the Streamlit sidebar so sync can
  be triggered without leaving the app. The button calls the same logic as the CLI script.
- **D-08:** The app imports from `clearml_utils.py` (existing module) for the sync button.
  ClearML task name: `manual_review_summary` (per SYNC-01).

### Save Behavior
- **D-09:** Auto-save on every field change (status, label, notes). Edits are written to
  manifest.csv immediately — no explicit Save button. This satisfies REVW-06.

### Claude's Discretion
- Exact CSS snippet for RTL injection — Claude chooses what works reliably with Streamlit
- State file format and location (`.review_state.json` suggested but Claude can adjust)
- Sidebar layout and widget ordering within the constraints above
- Whether to use `st.rerun()` or `st.session_state` for position tracking (Streamlit idiom)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §Review App (REVW-01 through REVW-06) — acceptance criteria for the app
- `.planning/REQUIREMENTS.md` §ClearML Sync (SYNC-01, SYNC-02) — sync script requirements

### Existing Code
- `src/manifest_schema.py` — canonical manifest column list (crop_path, pdf_path, page_num, x/y/w/h, area, is_flagged, flag_reasons, status, label, notes)
- `src/clearml_utils.py` — reusable helpers: `init_task()`, `upload_file_artifact()`, `report_manifest_stats()`, `maybe_create_dataset()`
- `src/prepare_data.py` — reference for ClearML task init pattern and argparse style

### Project Constraints
- `.planning/PROJECT.md` §Constraints — CPU-only, no extra heavy deps, privacy-sensitive data stays local

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/clearml_utils.py`: `init_task()`, `upload_file_artifact()`, `report_manifest_stats()` — all usable directly in review_to_clearml.py and the app's sync button
- `src/manifest_schema.py`: `MANIFEST_COLUMNS` list — use as reference when reading/writing manifest.csv to guarantee column order
- `src/prepare_data.py`: argparse + ClearML task init pattern — follow the same style in review_to_clearml.py

### Established Patterns
- Module-level ClearML imports (`from clearml import Task, Dataset`) required for test patchability (Phase 1 decision)
- `init_task()` must be called before `argparse.parse_args()` in CLI scripts
- pandas for manifest I/O (used in prepare_data.py)

### Integration Points
- `manifest.csv` is the shared data contract between Phase 1 output and Phase 2 input — app reads and writes it in place
- `review_queue.csv` provides the ordered crop list the app's queue should follow
- Phase 3 (training) will read `manifest.csv` filtering on `status == "labeled"` — the app must not corrupt the CSV schema

</code_context>

<specifics>
## Specific Ideas

- Use `review_queue.csv` row order as the queue order (not re-sorted in app) so the
  priority established by Phase 1 (flagged first) is preserved
- `.review_state.json` sits next to manifest.csv so it's automatically collocated with
  the data and easy to delete to reset position

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-review-annotation*
*Context gathered: 2026-04-25*
