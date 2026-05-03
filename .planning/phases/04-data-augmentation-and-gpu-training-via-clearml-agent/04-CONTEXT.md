# Phase 4: Data Augmentation and GPU Training via ClearML Agent - Context

**Gathered:** 2026-05-03
**Status:** Ready for planning

<domain>
## Phase Boundary

Two independent improvements to the CRNN+CTC training pipeline:

1. **Data augmentation** — Apply transforms to labeled crops at train time to improve
   CRNN+CTC generalization without collecting new data. Training only; val crops remain clean.

2. **GPU training via ClearML agent** — Configure a ClearML agent in WSL2 on a Windows
   RTX 5060 host. Add an `--enqueue` flag to `train_ctc.py` so training jobs can be
   dispatched to the GPU queue instead of running locally on CPU.

Out of scope: new model architecture, beam search, active learning, FiftyOne, cloud GPU,
changes to `evaluate.py` or `review_app.py`.

</domain>

<decisions>
## Implementation Decisions

### Augmentation Types

- **D-01:** Apply three transforms: **rotation ±5-10°**, **brightness/contrast jitter**,
  **Gaussian noise/blur**. No horizontal flip (reverses RTL text), no elastic distortion
  (risks distorting Hebrew diacritics), no affine shear.
- **D-02:** **Conservative parameters** — small magnitude, prioritize legibility. Expose
  all augmentation parameters as CLI flags (rotation_max, brightness_delta, noise_sigma)
  so the user can tune per run without code changes.

### Augmentation Integration

- **D-03:** **Online augmentation** — transforms applied inside `CropDataset.__getitem__`
  on each epoch pass. No new files written to disk, no manifest changes.
- **D-04:** Augmentation applies to the **training split only**. The val split always
  receives clean original crops to produce honest CER measurements.
- **D-05:** Augmentation multiplier (`--aug_copies`) defaults to Claude's discretion based
  on dataset size and expected training time. Exposed as a CLI flag. When `--aug_copies 0`
  or flag is absent, augmentation is off — backward-compatible default.

### ClearML Agent Deployment

- **D-06:** ClearML agent runs in **WSL2 with CUDA** on the Windows RTX 5060 host.
  WSL2 CUDA is well-supported for RTX 5060 via NVIDIA's WSL GPU driver.
- **D-07:** `train_ctc.py` gets an `--enqueue` flag. When set, the script creates the
  ClearML task and enqueues it to a named queue (e.g., `"gpu"`) instead of running
  training locally. The agent polls the queue and executes the task.
- **D-08:** The ClearML task name for GPU runs: `train_baseline_ctc` (same as CPU),
  differentiated by tags (`"gpu"`) and hyperparams. No separate script.

### Data Access on Agent Host

- **D-09:** Training data is already versioned in ClearML (`maybe_create_dataset` in
  Phase 1). Add `--dataset_id` arg to `train_ctc.py`. When provided, use
  `Dataset.get(dataset_id=...).get_local_copy()` to download and cache the dataset to
  the agent host's local ClearML cache directory (default `~/clearml/cache/`).
- **D-10:** Crop paths in `manifest.csv` are absolute paths from the original machine.
  When loading via `--dataset_id`, remap `crop_path` and `page_path` columns: replace
  the original directory prefix with the downloaded dataset root. Pattern:
  `new_path = dataset_root / "crops" / Path(original).name` (and `pages/` for page paths).
  This remapping runs in-memory; manifest.csv is not modified.
- **D-11:** ClearML SDK caches datasets by dataset ID; subsequent runs with the same
  `--dataset_id` skip the download and use the cached copy. No extra work needed beyond
  using `get_local_copy()`.

### Inherited Patterns (unchanged)

- `init_task()` called before `argparse.parse_args()` — required by ClearML SDK
- Module-level `from clearml import Task` for test patchability
- `resolve_device()` in `ctc_utils.py` already returns CUDA if available — no changes needed
- PyTorch CUDA wheel must be available on the agent host (handled via ClearML agent venv
  config, not pyproject.toml — the project pyproject.toml remains CPU-only for local dev)

### Claude's Discretion

- Default `--aug_copies` value (suggested: 2, producing ~2x effective dataset)
- Exact conservative parameter defaults for rotation, brightness, noise (suggested: ±7°, ±10%, sigma=5)
- Whether augmentation is a standalone `AugmentTransform` class or inline in `__getitem__`
- ClearML queue name (suggested: `"gpu"`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Training Code
- `src/train_ctc.py` — script to extend with `--enqueue`, `--dataset_id`, and augmentation
- `src/ctc_utils.py` — `load_crop()`, `CropDataset` (online aug goes here), `resolve_device()`
- `src/clearml_utils.py` — `init_task()`, `maybe_create_dataset()` patterns to follow

### Project Constraints
- `.planning/PROJECT.md` §Constraints — privacy-sensitive data stays local; stack constraints
- `.planning/PROJECT.md` §Context — local machine CPU-only (dev); RTX 5060 is the target GPU host

### Prior Phase Context
- `.planning/phases/03-training-evaluation/03-CONTEXT.md` — D-01 through D-05 (sizing,
  split, decode, patterns) all still apply; Phase 4 extends Phase 3, not replaces it

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/ctc_utils.py::load_crop()` — loads and resizes crops to 64px height; augmentation
  should be applied to the returned tensor (or numpy array before tensor conversion)
- `src/ctc_utils.py::resolve_device()` — already handles CUDA auto-detection, no changes
- `src/clearml_utils.py::init_task()` — reuse for enqueued task creation
- `src/clearml_utils.py::maybe_create_dataset()` — companion to the new `Dataset.get()` path

### Established Patterns
- CLI flags via `argparse` + `task.connect(vars(args))` for ClearML hyperparameter logging
- Module-level `from clearml import Task, Dataset` for test patchability
- `parse_known_args` pre-flight check before `Task.init` (Phase 2 pattern, Phase 3 used it)
- Conservative augmentation: transforms should be deterministic per image+index combo
  (e.g., seeded per index) to allow debug reproduction

### Integration Points
- `CropDataset.__getitem__` is the natural insertion point for online augmentation
- `train_ctc.py::main()` handles the `--enqueue` logic (task creation + enqueue + exit)
- `manifest.csv` crop_path/page_path columns need remapping when loading from ClearML dataset

</code_context>

<specifics>
## Specific Ideas

- The user wants the agent setup to use WSL2 — can install WSL if needed
- Researcher should look up: how to install ClearML agent in WSL2 with CUDA support for RTX
  5060, and the specific `clearml-agent` config for GPU queues
- Researcher should look up: ClearML `Dataset.get().get_local_copy()` caching behavior —
  confirm that re-calling with same dataset ID hits local cache rather than re-downloading

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-data-augmentation-and-gpu-training-via-clearml-agent*
*Context gathered: 2026-05-03*
