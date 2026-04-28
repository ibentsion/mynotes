# Phase 3: Training & Evaluation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-28
**Phase:** 03-training-evaluation
**Areas discussed:** Model input sizing, Output directory structure, Train/val page split ratio, CTC decode strategy

---

## Model Input Sizing

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed height 32px, proportional width | Standard CRNN approach, batch-padded | |
| Fixed height 32px, fixed width 256px | Simpler collation, distorts aspect ratio | |
| Fixed height 64px, proportional width | More vertical resolution for diacritics | ✓ |

**User's choice:** Fixed height 64px, proportional width
**Notes:** 64px chosen over 32px standard to better preserve Hebrew diacritics (vowel marks). DataLoader pads to longest width in batch.

---

## Output Directory Structure

| Option | Description | Selected |
|--------|-------------|----------|
| `--output_dir`, no default | Fully explicit, matches prepare_data.py exactly | |
| `--output_dir` with default `outputs/model/` | Convenient default, still overridable | ✓ |
| Separate `--model_dir` and `--eval_dir` | More granular, extra flags for little MVP gain | |

**User's choice:** `--output_dir` with default `outputs/model/`
**Notes:** Consistent with existing `outputs/` directory in repo.

---

## Train/Val Page Split Ratio

| Option | Description | Selected |
|--------|-------------|----------|
| 1 page held out for val | Deterministic, works at smallest scales | |
| 20% of pages, rounded up | Scales as data grows | ✓ (extended) |
| Claude's discretion | Runtime-adaptive | |

**User's choice:** 20% of half-page units, rounded up
**Notes:** User extended the decision: split pages into halves (`{page_num}.0` top, `{page_num}.1` bottom) using page pixel height from `page_path`. Crops assigned by center-y vs page midpoint. This gives finer split granularity without needing more pages.

---

## CTC Decode Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Greedy decode | Argmax + collapse + remove blank; built into PyTorch | ✓ |
| Beam search | Better CER; requires `ctcdecode` or custom impl | |

**User's choice:** Greedy decode for MVP
**Notes:** No extra dependencies. Sufficient for CER benchmarking at MVP scale. Beam search deferred to v2.

---

## Claude's Discretion

- CNN topology (number of conv layers, filter counts, pooling)
- Unicode normalization for Hebrew charset (NFC)
- Minimum labeled-crops guard implementation
- Batch collate padding implementation detail

## Deferred Ideas

None.
