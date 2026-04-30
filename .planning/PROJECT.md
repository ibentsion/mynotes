# Hebrew Handwriting OCR Pipeline

## What This Is

A local Python MVP pipeline for personal Hebrew handwritten OCR, built around a human-in-the-loop workflow. It converts scanned PDF pages into region crops, flags suspicious segmentations for manual review, trains a CRNN+CTC baseline model on minimal labeled data, and logs all data versions, metrics, and artifacts to ClearML.

## Core Value

A reviewable, labeled dataset of personal Hebrew handwriting that a baseline OCR model can train on — getting the data pipeline and annotation workflow right matters more than model accuracy at MVP.

## Requirements

### Validated

Validated in Phase 1: PDF ingestion, region detection, heuristic flagging, ClearML task infrastructure
Validated in Phase 2: Streamlit review app, ClearML sync, manifest persistence
Validated in Phase 3: CRNN+CTC training CLI, CER evaluation, eval_report.csv export, ClearML metric logging

### Active

(All v1 requirements complete — milestone shipped)

### Out of Scope

- Active learning / model uncertainty sampling — future extension, not MVP
- Split/merge editor on full page view — manual review covers bad_seg via status flag
- Pseudo-labeling on high-confidence crops — needs a working model first
- LLM-based post-correction — post-MVP
- FiftyOne integration — nice-to-have, deferred
- TrOCR / Kraken comparison — deferred until baseline is established
- Perfect OCR accuracy — MVP optimizes for dataset quality, not model performance

## Context

- Input: scanned PDFs exported from Gmail; pre-processed by CamScanner (denoised)
- Handwriting characteristics: not always on straight lines, may include diagonal text, margin notes, overlapping regions, text added later, irregular layouts
- Segmentation must be reviewable — messy writing makes automated segmentation unreliable
- Target labeled dataset size: 50–120 crops for MVP, growing to 150–300
- Local machine: Intel UHD 620 (integrated GPU, no CUDA) — CPU-only training
- CPU training estimate: ~20–60 min for MVP dataset (50–120 crops); ~1.5–3 hours at 300 crops
- Cloud GPU (Colab T4) is an option when iteration speed matters at scale, but not needed for MVP
- ClearML project: `handwriting-hebrew-ocr`; tasks: `data_prep`, `manual_review_summary`, `train_baseline_ctc`, `evaluate_model`

## Constraints

- **Runtime**: Python 3.13, CPU-only for MVP — model must train without CUDA
- **Stack**: pdf2image + Poppler, OpenCV, PyTorch, Streamlit, ClearML — no additional heavy dependencies
- **Data**: Personal Hebrew notes only; privacy-sensitive — stays local
- **Reproducibility**: Git commit, package versions, and all configs stored in ClearML per run
- **Modularity**: Scripts are independent CLI tools, not a monolithic app — easy to extend or replace individual steps
- **Poppler**: Required system dependency for pdf2image on Linux

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Region-first segmentation (not line-only) | Hebrew notes have diagonal text, margin notes, overlapping regions — strict line segmentation would miss too much | — Pending |
| CRNN+CTC over transformer OCR (TrOCR) | Simpler, lighter, trains on CPU with <300 samples; transformers need more data and compute | — Pending |
| Page-level train/val split | Prevents crop-level leakage where crops from the same page appear in both train and val | — Pending |
| Streamlit for review UI | Local-first, no server needed, quick to build; no need for a web backend for personal use | — Pending |
| ClearML for experiment tracking | Already part of the user's ML workflow; handles dataset versioning + artifact storage in one tool | — Pending |
| Label hard/flagged regions first | Maximizes model improvement per label; easy crops add less signal for a hard domain | — Pending |
| CPU-only for MVP | No local CUDA GPU available; dataset is small enough that CPU training is feasible | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-30 after Phase 3 completion — v1.0 milestone complete*
