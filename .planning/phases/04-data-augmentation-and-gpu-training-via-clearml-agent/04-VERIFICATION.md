---
phase: 04-data-augmentation-and-gpu-training-via-clearml-agent
verified: 2026-05-04T00:00:00Z
status: passed
score: 10/10 must-haves verified
re_verification: false
gaps: []
---

# Phase 04: Data Augmentation and GPU Training via ClearML Agent — Verification Report

**Phase Goal:** Online data augmentation for training crops (rotation, brightness, noise) and
ClearML agent setup for GPU training on Windows RTX 5060 via WSL2.
**Verified:** 2026-05-04
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Running train_ctc.py with --aug_copies 2 produces an effective dataset 3x the original size | VERIFIED | CropDataset.__len__ returns len(df)*(1+copies); test_crop_dataset_aug_copies_2_triples_length passes |
| 2  | Augmented images produced only from training crops; val crops always clean | VERIFIED | val_ds constructed as CropDataset(val_df, charset) with no augment= arg; test_val_dataset_has_no_augment passes |
| 3  | Running with --aug_copies 0 produces identical behaviour to Phase 3 (backward compatible) | VERIFIED | test_aug_copies_zero_backward_compat and test_no_enqueue_no_dataset_id_backward_compat pass |
| 4  | AugmentTransform applies rotation, brightness jitter, Gaussian noise; no horizontal flip | VERIFIED | AugmentTransform class at src/ctc_utils.py:147; padding_mode="border" present; test_augment_transform_no_horizontal_flip passes |
| 5  | All augmentation parameters tunable via CLI flags | VERIFIED | --aug_copies, --rotation_max, --brightness_delta, --noise_sigma all in _build_parser(); test_build_parser_aug_defaults passes |
| 6  | Effective dataset size is printed and logged to ClearML as a hyperparam | VERIFIED | "effective dataset size" printed at train_ctc.py:126; task.connect called with effective_train_size; test_aug_copies_nonzero_prints_effective_size passes |
| 7  | Running train_ctc.py --enqueue calls task.execute_remotely() after task.connect() and exits local process | VERIFIED | connect at line 91, execute_remotely at line 94; test_enqueue_calls_execute_remotely_after_connect passes with ordering assertion |
| 8  | Running train_ctc.py --dataset_id remaps manifest crop_path and page_path in-memory | VERIFIED | remap_dataset_paths called at train_ctc.py:86; original df not modified (df.copy() in clearml_utils.py:63); test_dataset_id_calls_remap passes |
| 9  | Running without --enqueue and without --dataset_id behaves identically to Phase 3 | VERIFIED | test_no_enqueue_no_dataset_id_backward_compat passes; rc=0 and checkpoint.pt written |
| 10 | docs/clearml-agent-setup.md contains actionable WSL2 setup instructions including cu128 index URL | VERIFIED | File exists at docs/clearml-agent-setup.md; cu128 URL present at line 56; clearml-agent==3.0.0 pinned; "Do NOT" apt warning at line 70 |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/ctc_utils.py` | AugmentTransform class; updated CropDataset with augment/aug_copies params | VERIFIED | class AugmentTransform at line 147; CropDataset at line 205 with augment/aug_copies params |
| `src/train_ctc.py` | Augmentation CLI flags + --enqueue/--queue_name/--dataset_id wired to main() | VERIFIED | All 7 flags present; AugmentTransform and remap_dataset_paths imported and used |
| `src/clearml_utils.py` | remap_dataset_paths() returning df copy with remapped paths | VERIFIED | Function at line 54; df.copy() at line 63; get_local_copy() at line 62 |
| `docs/clearml-agent-setup.md` | Step-by-step WSL2 agent setup guide with cu128 | VERIFIED | 158-line doc; cu128 in 3 places; apt warning; version table |
| `tests/test_ctc_utils.py` | AugmentTransform and CropDataset augmentation tests | VERIFIED | 5 AugmentTransform tests + 4 CropDataset aug tests; all pass |
| `tests/test_train_ctc.py` | aug_copies tests + enqueue/dataset_id tests | VERIFIED | 2 aug_copies tests, 1 val_dataset test, 2 enqueue tests, 1 dataset_id test; all pass |
| `tests/test_clearml_utils.py` | remap_dataset_paths unit tests | VERIFIED | 3 remap_dataset_paths tests; all pass |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/train_ctc.py` | `src/ctc_utils.py::CropDataset` | `CropDataset(train_df, charset, augment=augment, aug_copies=args.aug_copies)` | WIRED | train_ctc.py lines 131-136 |
| `src/ctc_utils.py::CropDataset.__getitem__` | `src/ctc_utils.py::AugmentTransform.__call__` | `self._augment(image, seed=index) when copy_idx > 0` | WIRED | ctc_utils.py line 229 |
| `src/train_ctc.py::main()` | `clearml Task::execute_remotely()` | `task.execute_remotely(queue_name=args.queue_name) when args.enqueue is True` | WIRED | train_ctc.py lines 93-95; connect (line 91) precedes execute_remotely (line 94) |
| `src/train_ctc.py::main()` | `src/clearml_utils.py::remap_dataset_paths()` | `labeled = remap_dataset_paths(labeled, args.dataset_id) when args.dataset_id is not None` | WIRED | train_ctc.py lines 85-86 |
| `src/clearml_utils.py::remap_dataset_paths()` | `clearml Dataset::get_local_copy()` | `Dataset.get(dataset_id=dataset_id).get_local_copy()` | WIRED | clearml_utils.py lines 61-62 |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/ctc_utils.py::CropDataset.__getitem__` | image, label_ids | load_crop() from disk + encode_label() | Yes — reads actual PNG files | FLOWING |
| `src/ctc_utils.py::AugmentTransform.__call__` | tensor (augmented) | torch operations on input tensor | Yes — rotation/brightness/noise applied | FLOWING |
| `src/clearml_utils.py::remap_dataset_paths` | df (remapped) | Dataset.get().get_local_copy() + df.copy() | Yes — real ClearML SDK call | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| AugmentTransform class importable and callable | `uv run pytest tests/test_ctc_utils.py -q` | 33 passed | PASS |
| CropDataset aug_copies multiplies length correctly | `test_crop_dataset_aug_copies_2_triples_length` | passed | PASS |
| train_ctc.py --aug_copies 0 backward compat | `test_aug_copies_zero_backward_compat` subprocess | rc=0, checkpoint.pt written | PASS |
| train_ctc.py --aug_copies 2 prints effective size | `test_aug_copies_nonzero_prints_effective_size` subprocess | "effective dataset size" in stdout | PASS |
| --enqueue connect-before-execute_remotely ordering | `test_enqueue_calls_execute_remotely_after_connect` | connect index < execute_remotely index | PASS |
| remap_dataset_paths does not mutate original df | `test_remap_dataset_paths_does_not_modify_original_df` | passed | PASS |
| Full phase test suite | `uv run pytest tests/test_ctc_utils.py tests/test_train_ctc.py tests/test_clearml_utils.py -q` | 65 passed | PASS |
| Full repo test suite (all tests) | `uv run pytest tests/ -q` | 120 passed, 1 flake (test_prepare_data_end_to_end_on_synthetic_pdf — passes in isolation, intermittent timing failure in parallel run) | PASS |

---

### Requirements Coverage

No explicit requirement IDs were declared in either plan's `requirements:` field (both are empty
lists). The phase was self-contained with success criteria specified directly in the plan
`must_haves` and `success_criteria` sections. All success criteria verified above.

---

### Anti-Patterns Found

None. No TODO/FIXME/placeholder comments found in modified files. No empty handlers or
stub returns in production code paths. No commented-out code.

---

### Human Verification Required

#### 1. Live ClearML Agent Enqueue Flow

**Test:** Commit Phase 4 changes, start a ClearML agent in WSL2 (`clearml-agent daemon --queue gpu --gpus 0 --foreground`), then run `uv run python -m src.train_ctc --manifest data/manifest.csv --output_dir outputs/model --enqueue --queue_name gpu`.
**Expected:** Task appears in ClearML web UI with status "queued"; agent picks it up and begins training on RTX 5060; PyTorch uses CUDA device.
**Why human:** Requires physical WSL2 setup, live ClearML server, and RTX 5060 GPU hardware. Cannot be verified programmatically in this environment.

#### 2. RTX 5060 / sm_120 CUDA Wheel Compatibility

**Test:** After agent setup per docs/clearml-agent-setup.md Section 4, run `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` in the agent task venv.
**Expected:** `True` and `NVIDIA GeForce RTX 5060` (or similar Blackwell device name). Verifies cu128 wheel installs correctly for sm_120.
**Why human:** Requires physical RTX 5060 hardware with CUDA 12.8 driver. Cannot be verified without the GPU.

---

### Gaps Summary

No gaps. All 10 observable truths are verified against the codebase:

- Plan 01 (augmentation): AugmentTransform class with rotation/brightness/noise is implemented in `src/ctc_utils.py`, CropDataset extended with augment/aug_copies params, 7 new augmentation CLI flags in train_ctc.py, val dataset always receives augment=None, effective dataset size logged. All 33 ctc_utils tests pass.

- Plan 02 (ClearML agent): remap_dataset_paths() in clearml_utils.py returns a copy with remapped paths, train_ctc.py has --enqueue/--queue_name/--dataset_id flags, execute_remotely called at line 94 (after connect at line 91), docs/clearml-agent-setup.md provides complete WSL2 setup guide with cu128 index URL, apt warning, and version table. All 32 train_ctc + clearml_utils tests pass.

The one failing test in the full suite (test_prepare_data_end_to_end_on_synthetic_pdf) is unrelated to this phase and passes in isolation — intermittent timing issue in parallel test execution.

---

_Verified: 2026-05-04_
_Verifier: Claude (gsd-verifier)_
