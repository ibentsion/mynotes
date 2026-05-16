# Phase 6: Synthetic Generation - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning

<domain>
## Phase Boundary

A CLI tool (`src/generate_synthetic.py`) that:
1. Renders Hebrew text crop images via TRDG with handwriting-style fonts and distortions
2. Assembles a text corpus from existing labeled crops (+ optional wordlist) with inverse-frequency weighting for rare characters
3. Writes a `manifest.csv` consumable by `CropDataset` without any code changes
4. Validates character coverage and exits non-zero when chars fall below `--min_char_count`
5. Logs all runs to ClearML task `generate_synthetic`

Out of scope: changes to `train_ctc.py`, `ctc_utils.py`, augmentation (Phase 7), two-stage training (Phase 7).

</domain>

<decisions>
## Implementation Decisions

### TRDG Rendering Style

- **D-01:** Use handwriting-like Hebrew fonts with TRDG distortions (skew, blur, noise). Aim for visual similarity to real scanned handwriting, not printed text. This is the primary lever for synthetic data quality.
- **D-02:** Plain white background. Real crops are preprocessed grayscale from CamScanner-denoised scans — white background is the appropriate approximation.
- **D-03:** Bundle a list of recommended Hebrew handwriting font names; download them on first use (lazy download into `assets/fonts/` or a cache dir). `--fonts_dir` overrides the downloaded defaults. No font bundled directly in the repo (keeps repo lean).

### Text Generation Strategy

- **D-04:** Word-level corpus sampling. Extract individual words from existing labeled crops (split on whitespace after NFC normalization). Sample 1–N words per synthetic crop. Words are the sampling unit, not characters.
- **D-05:** Match the real labeled crop character-count distribution when determining text length per crop. Read character counts from the existing manifest's `label` column, sample from that empirical distribution to pick how many characters each synthetic crop will contain.
- **D-06:** Inverse-frequency weighting on words. Compute character frequency across all existing labels. Words containing rarer characters get proportionally higher sampling probability. This satisfies SYN-03 without requiring targeted per-character generation.

### ClearML Integration

- **D-07:** Log generation runs to ClearML — initialize a task, connect CLI args, upload the output `manifest.csv` as an artifact. Follows the established pattern (`init_task()` from `clearml_utils.py`, `task.connect(vars(args))`).
- **D-08:** ClearML task name: `generate_synthetic`. Consistent with existing short-name convention (`data_prep`, `train_baseline_ctc`, `hpo_sweep`).

### Claude's Discretion

- Exact TRDG distortion parameter values (skew angle, blur radius, noise level) — pick conservative defaults that produce legible but imperfect text
- Font download mechanism (requests/urllib, target URL, cache location)
- Whether to also log a per-character count scalar to ClearML alongside the manifest artifact
- Exact word-splitting strategy for extracting corpus words (whitespace split + filter single chars, or include single-char words)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Training Pipeline (compatibility targets)
- `src/ctc_utils.py` — `CropDataset.__init__` and `__getitem__` (lines ~214–250): defines exact fields read from manifest (`crop_path`, `label`, `status`); `load_crop` defines the expected image format (grayscale, 64px height)
- `src/manifest_schema.py` — `MANIFEST_COLUMNS`: full real manifest schema; synthetic manifest needs at minimum `crop_path`, `label`, `status`

### Existing CLI Patterns (replicate these)
- `src/train_ctc.py` — argparse + `task.connect(vars(args))` pattern (TRAN-07); `--enqueue` / `--queue_name` pattern (Phase 4)
- `src/clearml_utils.py` — `init_task()`, `upload_file_artifact()` helpers to reuse

### Phase Requirements
- `.planning/REQUIREMENTS.md` §v1.1 Requirements — Synthetic Generation (SYN-01 through SYN-04): exact acceptance criteria
- `.planning/ROADMAP.md` §Phase 6: Synthetic Generation — success criteria (4 items)

### Project Constraints
- `.planning/PROJECT.md` §Constraints — no new heavy dependencies; stack is PyTorch, ClearML, argparse, OpenCV
- `.planning/PROJECT.md` §Context — GPU is RTX 5060 via ClearML agent on WSL2; data stays local

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/clearml_utils.py::init_task()` — reuse for ClearML task initialization in `generate_synthetic`
- `src/clearml_utils.py::upload_file_artifact()` — upload output `manifest.csv` as ClearML artifact
- `src/ctc_utils.py::build_charset()` — NFC normalization pattern; reuse for corpus character frequency counting
- `src/manifest_schema.py::MANIFEST_COLUMNS` — reference for which columns are optional vs required in synthetic manifest

### Established Patterns
- `argparse` + `task.connect(vars(args))` for hyperparameter tracking (TRAN-07) — replicate exactly
- Module-level ClearML imports (`from clearml import Task, Dataset`) for test patchability — follow this pattern
- `outputs/` directory for artifacts — `manifest.csv` goes in `--output_dir` (user-specified)
- `src/` layout with console script entry points in `pyproject.toml` — add `generate-synthetic` script

### Integration Points
- `CropDataset` reads the manifest via `pd.read_csv` filtering `status == "labeled"` — synthetic manifest must pass this filter
- `load_crop` in `ctc_utils.py` resizes to 64px height and returns a grayscale tensor — TRDG must produce images that survive this transform
- Real manifest has many optional columns (`pdf_path`, `page_num`, etc.) — synthetic manifest can omit them (CropDataset only reads `crop_path`, `label`, `status`)

</code_context>

<specifics>
## Specific Ideas

- Font strategy: bundle font names, download on first use into a local cache. `--fonts_dir` overrides. This mirrors how TRDG handles its own font lists internally.
- Text length: sample from empirical character-count distribution of existing labeled crops (not a fixed range) so synthetic crop widths statistically match real data.
- Rare-char weighting: inverse frequency at the word level — no need for explicit per-character generation passes.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 06-synthetic-generation*
*Context gathered: 2026-05-16*
