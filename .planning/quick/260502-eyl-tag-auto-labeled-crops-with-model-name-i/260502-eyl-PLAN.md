---
phase: quick-260502-eyl
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/auto_label.py
  - src/review_state.py
  - src/review_app.py
autonomous: true
requirements:
  - QUICK-EYL-01
  - QUICK-EYL-02
must_haves:
  truths:
    - "Auto-labeled crops are tagged with `auto:{model}` in the notes column"
    - "Existing human-authored notes are never overwritten by auto-labeling"
    - "Review app sidebar exposes an `auto_labeled` filter that shows only crops whose notes start with `auto:`"
    - "ReviewState accepts and persists the `auto_labeled` filter value"
  artifacts:
    - path: "src/auto_label.py"
      provides: "Auto-labeler that writes `auto:{model}` to notes alongside the label"
      contains: "auto:"
    - path: "src/review_state.py"
      provides: "VALID_FILTERS tuple including `auto_labeled`"
      contains: "auto_labeled"
    - path: "src/review_app.py"
      provides: "_filter_queue branch matching notes starting with `auto:`"
      contains: "startswith(\"auto:\")"
  key_links:
    - from: "src/auto_label.py:_run"
      to: "manifest.csv notes column"
      via: "df.at[idx, 'notes'] = f'auto:{args.model}' before write_manifest_atomic"
      pattern: "df\\.at\\[idx, \"notes\"\\] = f\"auto:"
    - from: "src/review_app.py:_filter_queue"
      to: "src/review_state.py:VALID_FILTERS"
      via: "filter_name == \"auto_labeled\" branch matches notes prefix"
      pattern: "filter_name == \"auto_labeled\""
---

<objective>
Tag auto-labeled crops with the model name in the manifest `notes` field, and add an `auto_labeled` filter to the review app sidebar so users can isolate model-labeled crops for verification.

Purpose: Distinguish OpenAI-generated labels from human labels in the manifest so reviewers can audit auto-labels without losing trust in the dataset.
Output: Updated `src/auto_label.py`, `src/review_state.py`, `src/review_app.py`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md

<interfaces>
<!-- Existing contracts the executor must preserve. Extracted from codebase. -->

From src/review_state.py:
```python
DEFAULT_FILTER = "unlabeled"
VALID_FILTERS: tuple[str, ...] = ("unlabeled", "flagged", "labeled", "all")

@dataclass
class ReviewState:
    filter: str = DEFAULT_FILTER
    index: int = 0
```

From src/review_app.py (_filter_queue, current shape):
```python
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
```

From src/auto_label.py (_run loop, current shape):
```python
text = _label_one(client, args.model, crop_path)
df.at[idx, "label"] = text
df.at[idx, "status"] = "labeled"
write_manifest_atomic(args.manifest, df)
```
Note: `notes` column is part of `MANIFEST_COLUMNS` (string dtype per `STRING_DTYPES`).
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Tag auto-labeled crops with model name in notes</name>
  <files>src/auto_label.py</files>
  <action>
    In `_run()`, modify the per-crop success path inside the `for i, idx in enumerate(indices, start=1):` loop. After `text = _label_one(client, args.model, crop_path)` and BEFORE `write_manifest_atomic(args.manifest, df)`, set the notes field to `f"auto:{args.model}"` only when the existing notes value is empty/NaN — never overwrite human-authored notes.

    Resulting block:
    ```python
    text = _label_one(client, args.model, crop_path)
    df.at[idx, "label"] = text
    df.at[idx, "status"] = "labeled"
    if pd.isna(df.at[idx, "notes"]) or str(df.at[idx, "notes"]).strip() == "":
        df.at[idx, "notes"] = f"auto:{args.model}"
    write_manifest_atomic(args.manifest, df)
    ```

    Do NOT introduce any helper function — keep the change inline (matches existing style; no premature abstraction). Do NOT add a comment explaining the guard; the condition is self-documenting.
  </action>
  <verify>
    <automated>cd ~/git/mynotes &amp;&amp; uv run ruff check src/auto_label.py &amp;&amp; uv run ty check src/auto_label.py</automated>
  </verify>
  <done>auto_label.py writes `auto:{model}` to notes alongside the label on success, and skips the assignment when notes already contain a non-empty human value. Lint and type checks pass.</done>
</task>

<task type="auto">
  <name>Task 2: Add auto_labeled to VALID_FILTERS</name>
  <files>src/review_state.py</files>
  <action>
    Update the `VALID_FILTERS` tuple to include `"auto_labeled"`:
    ```python
    VALID_FILTERS: tuple[str, ...] = ("unlabeled", "flagged", "labeled", "auto_labeled", "all")
    ```
    No other changes — `load_state` and `with_filter` already validate via `VALID_FILTERS` so the new value is accepted automatically.
  </action>
  <verify>
    <automated>cd ~/git/mynotes &amp;&amp; uv run ruff check src/review_state.py &amp;&amp; uv run ty check src/review_state.py</automated>
  </verify>
  <done>`auto_labeled` is a recognized filter; `load_state` accepts it and `with_filter` does not coerce it back to the default.</done>
</task>

<task type="auto">
  <name>Task 3: Add auto_labeled branch to _filter_queue</name>
  <files>src/review_app.py</files>
  <action>
    In `_filter_queue()` (around line 94), insert a new `elif` branch for `"auto_labeled"` between the existing `"labeled"` branch and the `else` fallback:
    ```python
    elif filter_name == "auto_labeled":
        mask = df["notes"].astype(str).str.startswith("auto:")
    ```
    Final shape:
    ```python
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
    ```
    Do NOT modify the sidebar radio/selectbox wiring — Streamlit picks up the new option automatically from `VALID_FILTERS` if it iterates over them; if the sidebar hardcodes the option list, also extend that list to include `"auto_labeled"` in the same display order (between `labeled` and `all`). Inspect the sidebar code in `review_app.py` and update only if it hardcodes filter names.
  </action>
  <verify>
    <automated>cd ~/git/mynotes &amp;&amp; uv run ruff check src/review_app.py &amp;&amp; uv run ty check src/review_app.py</automated>
  </verify>
  <done>`_filter_queue(df, "auto_labeled")` returns only crop_paths whose notes start with `auto:`; sidebar exposes the new filter option to users.</done>
</task>

</tasks>

<verification>
Run from repo root:
```bash
cd ~/git/mynotes
uv run ruff check src/auto_label.py src/review_state.py src/review_app.py
uv run ty check src/auto_label.py src/review_state.py src/review_app.py
uv run pytest -q tests/ -k "auto_label or review_state or review_app" 2>&1 | tail -20
```
All checks must pass with zero warnings.

Manual smoke (only if any tests touch these modules and pass): run `uv run python -m src.auto_label --dry-run --manifest data/manifest.csv` to confirm the dry-run path still exits 0 (the notes assignment is in the wet path so dry-run is unaffected).
</verification>

<success_criteria>
- `src/auto_label.py`: notes set to `auto:{model}` on success, only when notes are empty/NaN
- `src/review_state.py`: `VALID_FILTERS` includes `"auto_labeled"`
- `src/review_app.py`: `_filter_queue` has `auto_labeled` branch matching `notes` prefix `auto:`
- `ruff check` and `ty check` pass with zero warnings on all three files
- Existing tests for these modules still pass
</success_criteria>

<output>
After completion, create `.planning/quick/260502-eyl-tag-auto-labeled-crops-with-model-name-i/260502-eyl-SUMMARY.md`
</output>
