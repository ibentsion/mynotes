---
phase: quick-260503-pht
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/train_ctc.py
autonomous: true
requirements:
  - QUICK-260503-pht-01
must_haves:
  truths:
    - "After each training epoch, ClearML logs show predictions for ~5 fixed val crops"
    - "The same 5 crops are tracked across all epochs (deterministic selection)"
    - "Each logged block shows crop filename, ground-truth label, and current prediction"
    - "Console training output is unchanged in shape (existing per-epoch line still prints)"
  artifacts:
    - path: "src/train_ctc.py"
      provides: "Training loop with per-epoch debug-sample logging"
      contains: "DEBUG_SAMPLES"
  key_links:
    - from: "src/train_ctc.py"
      to: "src/ctc_utils.py::predict_single"
      via: "import + call inside val torch.no_grad() block"
      pattern: "predict_single\\("
    - from: "src/train_ctc.py"
      to: "ClearML logger"
      via: "logger.report_text"
      pattern: "logger\\.report_text\\("
---

<objective>
Add per-epoch debug-sample logging to the CTC training loop so we can watch
predictions evolve for a fixed set of validation crops in ClearML.

Purpose: Currently the only training signal is val_loss / val_cer scalars. There
is no way to visually inspect *what* the model is predicting on real crops as it
learns. A small fixed set of "debug samples" makes regressions and progress
obvious at a glance.

Output: `src/train_ctc.py` reports a text block per epoch via
`logger.report_text(title="debug_samples", series="val", iteration=epoch, ...)`
showing 5 deterministic val-split crops with ground truth and current prediction.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@.planning/STATE.md
@src/train_ctc.py
@src/ctc_utils.py

<interfaces>
<!-- Existing function in src/ctc_utils.py — already used by evaluate.py. -->

```python
def predict_single(
    model: CRNN,
    charset: list[str],
    device: torch.device,
    crop_path: str,
) -> str:
    """Greedy-decode a single crop. Caller manages model.eval() + torch.no_grad()."""
```

Manifest columns relevant to debug samples (from labeled DataFrame in main()):
- `crop_path: str`  — path to crop image on disk
- `label: str`      — ground-truth Hebrew label

ClearML logger API used:
```python
logger.report_text(
    title="debug_samples",
    series="val",
    iteration=epoch,
    print_console=False,
    body=text_block,
)
```
Note: `report_text` signature in ClearML uses `msg` as the body parameter in some
versions. If `body=` raises TypeError at runtime, fall back to positional:
`logger.report_text(text_block, level=logging.INFO, ...)` — verify against the
installed ClearML version before committing.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Add debug-sample selection and per-epoch ClearML text logging in train_ctc.py</name>
  <files>src/train_ctc.py</files>
  <behavior>
    - DEBUG_SAMPLES module-level constant = 5
    - Selection: first min(DEBUG_SAMPLES, len(val_ds)) rows from
      labeled.iloc[val_idx].reset_index(drop=True). Captured ONCE before the
      epoch loop as a list of (crop_path: str, gt_label: str) tuples.
    - After computing val_cer each epoch, while still inside the val
      `torch.no_grad()` context (or immediately after, model still in eval()),
      call predict_single(model, charset, device, crop_path) for each debug sample.
    - Build a text block in this exact shape:
        === debug samples epoch={epoch} ===
        [0] {crop_path} | gt={gt} | pred={pred}
        [1] {crop_path} | gt={gt} | pred={pred}
        ...
    - Log via logger.report_text(title="debug_samples", series="val",
      iteration=epoch, print_console=False, body=text_block).
    - If the val split has 0 rows the existing return-code-5 branch already
      aborts; debug-sample list is therefore guaranteed non-empty when reached.
    - If labeled has fewer than DEBUG_SAMPLES val rows, log whatever is
      available (no padding, no error).
  </behavior>
  <action>
    1. Add `predict_single` to the existing import from `src.ctc_utils` (alphabetical position keeps the block sorted).
    2. Add module-level constant `DEBUG_SAMPLES = 5` near the top of the file (after imports, before `_build_parser`). This is intentionally NOT a CLI arg — keep it as a constant per quick-task scope.
    3. In `main()`, AFTER `val_ds = CropDataset(...)` is constructed and BEFORE `model = CRNN(...)`, materialize the debug list from the val DataFrame:
       ```python
       val_df = labeled.iloc[val_idx].reset_index(drop=True)
       n_debug = min(DEBUG_SAMPLES, len(val_df))
       debug_samples = [
           (str(val_df.iloc[i]["crop_path"]), str(val_df.iloc[i]["label"]))
           for i in range(n_debug)
       ]
       ```
       Reuse this `val_df` for `val_ds = CropDataset(val_df, charset)` instead of recomputing — keeps the source of truth single.
    4. Inside the epoch loop, AFTER `val_cer = cer_total / max(cer_count, 1)` and AFTER the three existing `logger.report_scalar(...)` calls, add the debug-sample logging block. Keep model in eval() mode (it already is from the val pass) and wrap the prediction calls in `with torch.no_grad():`.
       ```python
       with torch.no_grad():
           lines = [f"=== debug samples epoch={epoch} ==="]
           for i, (crop_path, gt) in enumerate(debug_samples):
               pred = predict_single(model, charset, device, crop_path)
               lines.append(f"[{i}] {crop_path} | gt={gt} | pred={pred}")
           text_block = "\n".join(lines)
       logger.report_text(
           title="debug_samples",
           series="val",
           iteration=epoch,
           print_console=False,
           body=text_block,
       )
       ```
    5. Do NOT modify the per-epoch `print(...)` line, the scalar reporting, or the checkpoint-save logic. Do NOT add a CLI flag.
    6. Run `ruff check src/train_ctc.py && ruff format src/train_ctc.py && ty check src/train_ctc.py`. Fix any warnings.
    7. Run the existing training tests to confirm no regression:
       `uv run pytest tests/ -k train_ctc -q` (or the closest existing pattern — check `tests/` first).
  </action>
  <verify>
    <automated>uv run ruff check src/train_ctc.py && uv run ruff format --check src/train_ctc.py && uv run ty check src/train_ctc.py && uv run pytest tests/ -k "train_ctc or ctc" -q</automated>
  </verify>
  <done>
    - `src/train_ctc.py` imports `predict_single` from `src.ctc_utils`.
    - Module-level `DEBUG_SAMPLES = 5` constant exists.
    - `main()` materializes a `debug_samples` list of (crop_path, label) tuples from the first N val rows, ONCE, before the epoch loop.
    - Inside the epoch loop, after val_cer + scalar reporting, predictions are computed for each debug sample under `torch.no_grad()` and logged via `logger.report_text(title="debug_samples", series="val", iteration=epoch, ...)`.
    - Existing scalar reporting, checkpoint save, console print, and return codes are unchanged.
    - `ruff check`, `ruff format --check`, and `ty check` all pass with zero warnings.
    - All existing tests still pass.
  </done>
</task>

</tasks>

<verification>
- Static: `ruff check src/train_ctc.py` zero warnings, `ty check src/train_ctc.py` zero errors.
- Tests: existing train_ctc / ctc tests still green.
- Smoke (manual, optional — only if a manifest with ≥100 labeled rows is available):
  `uv run python -m src.train_ctc --epochs 2` should print the per-epoch line as before. The ClearML task page should show a text panel titled `debug_samples / val` with iterations 0 and 1, each containing 5 lines in the documented format.
</verification>

<success_criteria>
- After each epoch, ClearML logs contain exactly one `debug_samples / val` text block at iteration=epoch.
- Each block shows ≤5 entries with `[i] {crop_path} | gt={gt} | pred={pred}`.
- The same 5 crops appear in every epoch's block (deterministic).
- No new CLI flags, no new files, no changes to scalar reporting or checkpoints.
- Zero linter / type-checker warnings.
</success_criteria>

<output>
After completion, create `.planning/quick/260503-pht-add-debug-samples-to-training-so-i-can-s/260503-pht-SUMMARY.md`
</output>
