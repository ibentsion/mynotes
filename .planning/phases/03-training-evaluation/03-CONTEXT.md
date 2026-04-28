# Phase 3: Training & Evaluation - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Build two independent CLI scripts:
- `train_ctc.py` — loads labeled crops from `manifest.csv`, builds a Hebrew charset, performs
  page-level train/val split, trains a CRNN+CTC model on CPU, saves checkpoint and charset,
  logs all metrics and artifacts to ClearML task `train_baseline_ctc`
- `evaluate.py` — loads the saved checkpoint and charset, runs greedy CTC inference on the
  validation set, computes CER, exports `eval_report.csv`, logs to ClearML task `evaluate_model`

Out of scope: beam search decoding, active learning, pseudo-labeling, LLM post-correction,
FiftyOne integration, cloud/GPU training.

</domain>

<decisions>
## Implementation Decisions

### Model Input Sizing
- **D-01:** Fixed height **64px**, proportional width. Each crop image is resized to 64px height
  while preserving aspect ratio. The DataLoader collates batches by padding each sample's width
  to the longest width in the batch (right-pad with zeros). 64px (over the common 32px) gives
  more vertical resolution for Hebrew letters with potential diacritics.

### Output Organization
- **D-02:** Both scripts accept `--output_dir` with a default of `outputs/model/`. Artifacts
  written there: `checkpoint.pt`, `charset.json` (from training), `eval_report.csv` (from
  evaluation). Consistent with the `outputs/` directory already in the repo.

### Train/Val Page Split
- **D-03:** Split unit is a **half-page**. For each page, the midpoint is the page's pixel
  height read from the `page_path` image (stored in `manifest.csv`). Crops whose center-y
  (`y + h/2`) is above the midpoint belong to the top half (`{page_num}.0`); crops below
  belong to the bottom half (`{page_num}.1`).
- **D-04:** **20% of half-page units, rounded up** (minimum 1 unit), are allocated to
  validation. Units are sorted deterministically before splitting so runs are reproducible.
  No random seed needed — the split is purely structural.

### CTC Decoding
- **D-05:** **Greedy decode** for evaluation — argmax at each timestep, collapse consecutive
  repeats, remove blank tokens. Built into PyTorch; no extra dependencies.

### Inherited Patterns (from prior phases)
- `init_task()` called before `argparse.parse_args()` — required by ClearML SDK
- Module-level ClearML imports (`from clearml import Task`) for test patchability
- `clearml_utils.py` helpers (`init_task`, `upload_file_artifact`) reused directly
- argparse style follows `prepare_data.py` — all hyperparameters as explicit CLI flags,
  connected to ClearML via `task.connect(vars(args), name="hyperparams")`
- PyTorch must be added to `pyproject.toml` (CPU-only wheel) — concrete task for the planner

### Claude's Discretion
- Specific CNN topology (number of conv layers, filter counts, pooling) — standard small CRNN
  for document OCR is fine
- Unicode normalization for Hebrew charset (NFC is standard)
- Minimum labeled-crops guard (fail with a clear message if fewer than N labeled crops found)
- Batch collate padding implementation detail

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §Training (TRAN-01 through TRAN-08) — acceptance criteria for train_ctc.py
- `.planning/REQUIREMENTS.md` §Evaluation (EVAL-01 through EVAL-04) — acceptance criteria for evaluate.py
- `.planning/REQUIREMENTS.md` §ClearML Infrastructure (CLML-01 through CLML-05) — ClearML task/artifact requirements

### Existing Code
- `src/manifest_schema.py` — canonical manifest columns; phase 3 uses `crop_path`, `page_path`,
  `page_num`, `y`, `h`, `status`, `label`
- `src/clearml_utils.py` — reusable helpers: `init_task()`, `upload_file_artifact()`, `maybe_create_dataset()`
- `src/prepare_data.py` — reference for argparse + ClearML task init pattern to follow

### Project Constraints
- `.planning/PROJECT.md` §Constraints — CPU-only, no extra heavy deps, privacy-sensitive data local
- `.planning/PROJECT.md` §Context — target 50–120 labeled crops for MVP; CPU training ~20–60 min

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/clearml_utils.py`: `init_task()`, `upload_file_artifact()`, `maybe_create_dataset()` —
  use directly in train_ctc.py and evaluate.py
- `src/manifest_schema.py`: `MANIFEST_COLUMNS` — reference when reading manifest to confirm
  expected columns are present
- `src/prepare_data.py`: argparse + `task.connect()` pattern — follow exactly for CLI hyperparams

### Established Patterns
- Module-level ClearML imports required for test patchability (patch `src.train_ctc.Task`, not `clearml.Task`)
- `init_task()` before `argparse.parse_args()` — enforced by ClearML SDK design
- pandas for manifest I/O (all prior scripts use it)
- Tests in `tests/` mirroring src structure; mock ClearML at module level

### Integration Points
- `manifest.csv` is the sole input to training — filter on `status == "labeled"`
- `page_path` column provides the image path needed to determine page height for half-page split
- `outputs/` directory already exists in repo — `outputs/model/` is a natural subdirectory
- Phase 2 must have run (labeled crops exist) before Phase 3 is useful, but train_ctc.py
  should fail gracefully with a clear message if no labeled crops are found

### New Dependency
- `torch` (CPU-only) not yet in `pyproject.toml` — planner must add it as first task

</code_context>

<specifics>
## Specific Ideas

- Half-page split notation: `{page_num}.0` (top half) and `{page_num}.1` (bottom half) — makes
  split units human-readable if logged or printed
- 64px height chosen over 32px standard to preserve diacritic detail in Hebrew handwriting
- Greedy decode is sufficient for MVP CER benchmarking; beam search deferred to v2

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 03-training-evaluation*
*Context gathered: 2026-04-28*
