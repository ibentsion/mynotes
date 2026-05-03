# Phase 4: Data Augmentation and GPU Training via ClearML Agent - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-03
**Phase:** 04-data-augmentation-and-gpu-training-via-clearml-agent
**Areas discussed:** Augmentation types, Augmentation integration, ClearML agent deployment, Data path bridging, Eval scope

---

## Augmentation Types

| Option | Description | Selected |
|--------|-------------|----------|
| Rotation ±5-10° | Compensates for slightly tilted writing; safe for Hebrew character identity | ✓ |
| Brightness/contrast jitter | Simulates different scan qualities, ink density, page aging | ✓ |
| Gaussian noise / blur | Simulates scan artifacts, paper texture, faint ink | ✓ |
| Affine shear / elastic distortion | Simulates handwriting slant/stretch; riskier for Hebrew diacritics | |

**User's choice:** Rotation, brightness/contrast, Gaussian noise. No elastic distortion.
**Notes:** Explicitly excluded horizontal flip (reverses RTL text) and elastic distortion (could corrupt Hebrew diacritics).

---

## Augmentation Strength

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative | Small parameters: ±5°, ±10%, sigma 5 | ✓ |
| Moderate | Medium parameters: ±10°, ±20%, sigma 10-15 | |
| Let Claude decide | Claude picks defaults, exposes as CLI flags | |

**User's choice:** Conservative. Expose all parameters as CLI flags for tuning.

---

## Augmentation Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Online in Dataset (Recommended) | Applied in CropDataset.__getitem__ per epoch; no new files | ✓ |
| Offline pre-generation | augment_crops.py generates copies on disk before training | |
| Offline with --augment flag | One-command offline generation triggered from train_ctc.py | |

**User's choice:** Online augmentation in Dataset.
**Notes:** Cleaner, fits existing code pattern, avoids disk bloat.

---

## Augmentation Copies

| Option | Description | Selected |
|--------|-------------|----------|
| 2x (Recommended) | Each crop produces 2 augmented variants per epoch | |
| 3–4x | More variety but slower per epoch | |
| Let Claude decide | Exposed as --aug_copies flag | ✓ |

**User's choice:** Let Claude decide, expose as `--aug_copies` flag.

---

## ClearML Agent Deployment

| Option | Description | Selected |
|--------|-------------|----------|
| WSL2 with CUDA (Recommended) | Linux env on Windows, CUDA via WSL GPU driver; RTX 5060 well-supported | ✓ |
| Windows native Python | Native GPU, but env diverges from Linux dev environment | |

**User's choice:** WSL2 with CUDA.

---

## Agent Dispatch

| Option | Description | Selected |
|--------|-------------|----------|
| ClearML queue + --enqueue flag (Recommended) | train_ctc.py creates task and enqueues; agent picks up | ✓ |
| Manual clone + enqueue | No code change; two-step workflow via ClearML UI | |

**User's choice:** `--enqueue` flag in `train_ctc.py`.

---

## Data Path Bridging

**User's answer (free text):** "data is in clearml, which supports moving data to reside locally on agent's host. make sure it is cached locally on that host to avoid downloading it every experiment. search web on how to do that."

**Captured decision:** Add `--dataset_id` arg. Use `Dataset.get().get_local_copy()` which caches by dataset ID. Remap manifest crop_path/page_path to downloaded root in-memory.

---

## Eval Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Training split only (Recommended) | Augmentation after train/val split; val sees clean crops | ✓ |
| All labeled crops | Would augment val crops — inflates CER; not recommended | |

**User's choice:** Training split only.

---

## Claude's Discretion

- Default `--aug_copies` value
- Exact conservative parameter defaults
- Whether augmentation is a class or inline in `__getitem__`
- ClearML queue name

## Deferred Ideas

None.
