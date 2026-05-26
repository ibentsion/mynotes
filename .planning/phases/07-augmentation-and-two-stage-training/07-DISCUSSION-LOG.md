# Phase 7: Augmentation & Two-Stage Training - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-26
**Phase:** 07-augmentation-and-two-stage-training
**Areas discussed:** Elastic implementation, Two-stage loop design, Pre-training val strategy, ClearML reporting

---

## Elastic Implementation

| Option | Description | Selected |
|--------|-------------|----------|
| albumentations | Add dep, apply ElasticTransform + GridDistortion, tensor↔numpy conversion inside __call__ | ✓ |
| Pure PyTorch | Implement via torch displacement fields, no new dependency | |
| You decide | Leave to Claude | |

**User's choice:** albumentations
**Notes:** None — straightforward acceptance of the recommended approach.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fine-tuning only | Elastic applies only to real train split; synthetic pre-training uses TRDG distortions already | ✓ |
| Both stages | Elastic applies to synthetic pre-training and real fine-tuning | |
| You decide | Leave to Claude | |

**User's choice:** Fine-tuning only
**Notes:** TRDG already applies distortions to synthetic crops, so elastic on top is redundant.

---

## Two-Stage Loop Design

| Option | Description | Selected |
|--------|-------------|----------|
| Extract _run_loop helper | Refactor epoch loop into _run_loop(), called per stage with series_prefix | ✓ |
| Add block at top of run_training | Pre-training inline before main loop, minimal refactoring | |
| You decide | Whichever keeps functions under 100 lines | |

**User's choice:** Extract _run_loop helper
**Notes:** Cleaner separation, respects CLAUDE.md 100-line limit.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Reset optimizer with --lr | Fresh AdamW + ReduceLROnPlateau after pre-training | ✓ |
| Continue from pre-train state | Same optimizer across both stages | |
| You decide | Leave to Claude | |

**User's choice:** Reset optimizer with --lr

---

### Stage flow question (user-initiated)

User raised: "Are we going to even perform fine-tuning if pre-training didn't succeed? What about HPO — tune.py calls run_training per trial, would each trial pre-train? Checkpoints can be stored in ClearML."

Key concern: embedding pre-training inside run_training() would make every HPO trial pre-train, which is prohibitively expensive.

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential in one call, no gate | --pretrain_manifest triggers pre-train then fine-tune in one call | |
| Pre-train once, load checkpoint (two-call approach) | Separate CLI calls; HPO trials load pre-trained checkpoint | ✓ |
| Sequential with val_cer gate | Auto-abort fine-tuning if pre-train val_cer too high | |

**User's choice:** Two-call approach
**Notes:** HPO-aware design — pre-train once outside of HPO, all trials fine-tune from the same pre-trained weights.

---

### ROADMAP reconciliation

| Option | Description | Selected |
|--------|-------------|----------|
| Two-call approach, update ROADMAP | Reflect two-call interface in success criteria | ✓ |
| Keep one-call, skip HPO integration | Preserve ROADMAP as-is; HPO always fine-tunes from scratch | |

**User's choice:** Update ROADMAP
**Notes:** ROADMAP success criteria currently imply one call does both stages — needs updating.

---

## Pre-Training Val Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse --val_frac | Same fraction (default 0.15) random split on synthetic rows | ✓ |
| Separate --pretrain_val_frac flag | Independent tuning for synthetic val fraction | |
| Fixed 10% holdout | Hardcoded fraction, no flag | |

**User's choice:** Reuse --val_frac
**Notes:** Simple, consistent. Synthetic rows have no page structure so build_half_page_units() doesn't apply — random split only.

---

## ClearML Reporting / Checkpoint Management

| Option | Description | Selected |
|--------|-------------|----------|
| Same task, prefixed series | One task; pretrain/val_loss + finetune/val_loss series prefixes | ✓ |
| Two separate tasks | Separate ClearML tasks per stage | |
| You decide | Leave to Claude | |

**User's choice:** Same task, prefixed series

---

| Option | Description | Selected |
|--------|-------------|----------|
| Fine-tuned checkpoint only | Only checkpoint.pt saved; pre-trained weights are intermediate | |
| Both checkpoints | checkpoint_pretrain.pt + checkpoint.pt both saved | |

**User's choice:** User redirected this question to broader flow concern (see Stage flow above). Resolved as: checkpoint_pretrain.pt uploaded as ClearML artifact on pre-training task; fine-tuning task receives --pretrain_checkpoint_path.

---

| Option | Description | Selected |
|--------|-------------|----------|
| ClearML artifact (pre-trained checkpoint) | Upload checkpoint_pretrain.pt to pre-training task | ✓ |
| Local disk only | Save to --output_dir, not tracked in ClearML | |

**User's choice:** ClearML artifact

---

## Claude's Discretion

- Exact albumentations ElasticTransform + GridDistortion parameter defaults
- Whether _run_loop() returns float or richer result struct
- Exact series name casing (pretrain/val_cer vs pretrain_val_cer)
- ClearML artifact name for pre-trained checkpoint

## Deferred Ideas

None.
