---
phase: quick/260512-uk2
plan: 01
subsystem: training-diagnostics
tags: [training, clearml, grad-cam, saliency, debug-visualization]
dependency_graph:
  requires: [src/train_ctc.py, src/ctc_utils.py]
  provides: [annotated-debug-samples, saliency-panel-per-epoch]
  affects: [ClearML training run visualization]
tech_stack:
  added: [Grad-CAM (custom), matplotlib figure reporting]
  patterns: [lazy import pattern (noqa PLC0415), private helpers for 100-line compliance]
key_files:
  created: []
  modified:
    - src/train_ctc.py
    - src/ctc_utils.py
    - tests/test_ctc_utils.py
decisions:
  - "_report_annotated_crop uses Any for logger/crop_hw types — ClearML Logger has no stubs; existing pattern from _report_prob_heatmap"
  - "compute_char_saliency manages eval mode + autograd internally — callers don't need to set no_grad"
  - "_register_gradcam_hooks and _pad_to_multiple_of_4 extracted to keep compute_char_saliency under 100-line limit"
  - "per_sample_cer list captured during val loop; saliency panel called OUTSIDE no_grad — backward pass requires autograd"
  - "F.affine_grid pre-existing ty error fixed: list(tensor.shape) cast added in AugmentTransform"
metrics:
  duration: ~20 minutes
  completed: 2026-05-12
  tasks: 2
  files_modified: 3
---

# Quick Task 260512-uk2: Training Results Analysis and Debug Visualization Summary

**One-liner:** Diagnose overfitting in the CPU training run, then add GT/pred annotated debug samples and per-epoch Grad-CAM saliency panel at 5 CER percentiles to ClearML.

## Objective

Two outcomes:
1. Written analysis of the recent CPU training run (train_loss 1.05->0.20, val_loss 3.06->4.28, best_val_cer 0.5966, blank_frac ~0.94) with ranked experiment suggestions.
2. Code changes that make the next training run's ClearML diagnostics actionable.

## Requirements Completed

- **UK2-01:** Training-Run Analysis section written in PLAN.md — diagnoses capacity >> data overfitting, residual blank collapse, inf_loss from CTC infeasibility, first-character prediction bias. Includes ranked A-H experiment table.
- **UK2-02:** `_report_annotated_crop` renders GT label and predicted text as figure title (green=match, red=mismatch) on every `debug_samples/sample_i` image. Replaces the old `logger.report_image` + `np.stack` call.
- **UK2-03:** `compute_char_saliency` (Grad-CAM on last CNN conv block) + `_report_saliency_panel` log a 5-row panel per epoch under `saliency_chars/percentiles` — one row per CER percentile (0/25/50/75/100).

## Tasks Completed

### Task 1: Annotated debug samples (commit 3598aa9)

Added `_report_annotated_crop` helper to `train_ctc.py`. Replaced `logger.report_image` + dead `np.stack` code with a matplotlib figure call. Removed unused `numpy` import from `run_training`. Fixed pre-existing ty issue: changed `logger: object` to `logger: Any` in both heatmap helpers (ClearML Logger has no type stubs).

### Task 2: Per-character Grad-CAM saliency (TDD, commits 7d4e2af + 98aa375)

**RED:** Added `test_compute_char_saliency_shape_and_range` — confirmed ImportError before implementation.

**GREEN:** Implemented `compute_char_saliency` in `ctc_utils.py` with two extracted helpers:
- `_pad_to_multiple_of_4` — width padding for CRNN compatibility
- `_register_gradcam_hooks` — forward + backward hook registration on last Conv2d

The function hooks the last Conv2d in `model.cnn`, runs a forward pass, sums log-probs at non-blank greedy argmax timesteps, backprops, and computes `relu((alpha * A).sum(channels))` upsampled to crop H×W and normalized to [0,1].

Added `_report_saliency_panel` to `train_ctc.py` — calls `compute_char_saliency` for each of 5 percentile picks and logs a stacked matplotlib figure to ClearML.

Modified the val loop to capture `per_sample_cer: list[tuple[str, str, str, float]]` and `dataset_idx` counter. The saliency panel is called outside `torch.no_grad()` so autograd is available for the backward pass.

**REFACTOR:** Fixed pre-existing `ty` error in `ctc_utils.py` — `F.affine_grid` stub expects `list[int]` but `tensor.shape` returns `Size`; fixed with `list(tensor.shape)` cast in `AugmentTransform`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pre-existing ty error in ctc_utils.py**
- **Found during:** Task 2 REFACTOR
- **Issue:** `F.affine_grid(theta, tensor.unsqueeze(0).shape, ...)` — ty reports `Expected list[int], found Size` (pre-existing, not caused by this task)
- **Fix:** Changed `tensor.unsqueeze(0).shape` to `list(tensor.unsqueeze(0).shape)` in `AugmentTransform`
- **Files modified:** `src/ctc_utils.py:185`
- **Commit:** 98aa375

**2. [Rule 1 - Bug] Pre-existing ty error in train_ctc.py**
- **Found during:** Task 1 verification
- **Issue:** `logger: object` in `_report_prob_heatmap` causes `unresolved-attribute` for `.report_matplotlib_figure` — same issue would appear in new `_report_annotated_crop`
- **Fix:** Added `from typing import Any` and changed `logger: object` to `logger: Any` in both helpers
- **Files modified:** `src/train_ctc.py`
- **Commit:** 3598aa9

## Known Stubs

None — all new helpers are wired and functional.

## Self-Check: PASSED

Files created/modified:
- `src/train_ctc.py` — FOUND (contains `_report_annotated_crop`, `_report_saliency_panel`, `saliency_chars` call)
- `src/ctc_utils.py` — FOUND (contains `def compute_char_saliency`)
- `tests/test_ctc_utils.py` — FOUND (contains `test_compute_char_saliency_shape_and_range`)

Commits verified:
- 3598aa9 — FOUND (Task 1: annotated crop)
- 7d4e2af — FOUND (RED test)
- 98aa375 — FOUND (Task 2: saliency implementation)

All 77 tests pass. ruff clean. ty clean.
