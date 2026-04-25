# Phase 2: Review & Annotation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-25
**Phase:** 02-review-annotation
**Areas discussed:** Review flow, Hebrew text input, Progress visibility, ClearML sync trigger

---

## Review Flow

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential queue | One crop at a time, Prev/Next buttons, position persisted | ✓ |
| Scrollable list | All crops in a list, click to edit on the right | |

**User's choice:** Sequential queue

---

### Position persistence

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, persist position | Store last filter + crop index in .review_state.json | ✓ |
| No, always start fresh | App always loads at position 0 | |

**User's choice:** Persist position across sessions

---

## Hebrew Text Input

| Option | Description | Selected |
|--------|-------------|----------|
| RTL via CSS injection | st.markdown CSS sets direction:rtl on transcription field | ✓ |
| Leave LTR | No direction change — Hebrew appears reversed | |
| You decide | Claude handles RTL approach | |

**User's choice:** RTL via CSS injection

---

## Progress Visibility

| Option | Description | Selected |
|--------|-------------|----------|
| Sidebar status counts | Live per-status counts in sidebar | ✓ |
| Current position only | "Crop 4 of 12 (flagged)" above image | |
| Progress bar + counts | Full progress bar + per-status breakdown | |

**User's choice:** Sidebar status counts (live, updates after each edit)

---

## ClearML Sync Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| CLI only | review_to_clearml.py as standalone script | |
| CLI + app button | CLI script + Sync button in Streamlit sidebar | ✓ |

**User's choice:** Both CLI script and in-app Sync button

---

## Claude's Discretion

- Exact CSS for RTL injection
- State file format/location
- Streamlit session_state vs. st.rerun() for position tracking
- Sidebar widget ordering

## Deferred Ideas

None.
