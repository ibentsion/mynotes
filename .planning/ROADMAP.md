# Roadmap: Hebrew Handwriting OCR Pipeline

## Overview

Three phases deliver the complete MVP pipeline: first build the data extraction and flagging
system that turns scanned PDFs into reviewable crops; then wire up the Streamlit annotation
app so crops can be transcribed and synced to ClearML; finally train and evaluate the baseline
CRNN+CTC model on the resulting labeled dataset. Each phase produces something independently
verifiable before the next begins.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Data Pipeline** - PDF ingestion, region detection, heuristic flagging, and ClearML infrastructure (completed 2026-04-21)
- [ ] **Phase 2: Review & Annotation** - Streamlit review app, labeling workflow, and ClearML sync
- [ ] **Phase 3: Training & Evaluation** - CRNN+CTC model training and CER evaluation

## Phase Details

### Phase 1: Data Pipeline
**Goal**: Scanned PDFs are converted into reviewable, flagged region crops with full ClearML tracking
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, FLAG-01, FLAG-02, FLAG-03, FLAG-04, FLAG-05, FLAG-06, CLML-01, CLML-02, CLML-03, CLML-04, CLML-05
**Success Criteria** (what must be TRUE):
  1. Running prepare_data.py on a PDF folder produces a manifest.csv and review_queue.csv with one row per crop
  2. Each crop image exists on disk as a grayscale file with bounding box metadata in the manifest
  3. Regions in review_queue.csv are sorted with flagged/suspicious crops first (angle, overlap, size, faintness, margin)
  4. Flag reasons are stored per crop in manifest.csv so a reviewer knows why each region was flagged
  5. ClearML task data_prep is created with PDF list, parameters, manifest, and a versioned dataset logged
**Plans**: TBD

### Phase 2: Review & Annotation
**Goal**: A reviewer can work through flagged crops, transcribe Hebrew text, set statuses, and sync results to ClearML
**Depends on**: Phase 1
**Requirements**: REVW-01, REVW-02, REVW-03, REVW-04, REVW-05, REVW-06, SYNC-01, SYNC-02
**Success Criteria** (what must be TRUE):
  1. Streamlit app loads and displays crops from manifest.csv, filterable by status (unlabeled, flagged, labeled, all)
  2. Reviewer can type a Hebrew transcription, set a status, add notes, and see changes persisted to manifest.csv immediately
  3. Running review_to_clearml.py uploads the updated manifest and a labeling status summary to ClearML task manual_review_summary
**Plans:** 3 plans
- [x] 02-01-PLAN.md — Streamlit review app skeleton: streamlit dep, review_state helper, manifest load + filter + Prev/Next navigation + position display
- [x] 02-02-PLAN.md — review_to_clearml.py CLI: status counts + manifest artifact upload to ClearML task manual_review_summary
- [x] 02-03-PLAN.md — Wire edit fields (RTL transcription, status, notes), autosave, sidebar status counts, sync button + human-verify checkpoint
**UI hint**: yes

### Phase 3: Training & Evaluation
**Goal**: A CRNN+CTC model trains on labeled crops and reports CER on a held-out validation set
**Depends on**: Phase 2
**Requirements**: TRAN-01, TRAN-02, TRAN-03, TRAN-04, TRAN-05, TRAN-06, TRAN-07, TRAN-08, EVAL-01, EVAL-02, EVAL-03, EVAL-04
**Success Criteria** (what must be TRUE):
  1. train_ctc.py reads only status=labeled crops, builds a dynamic Hebrew charset, and completes training on CPU
  2. Train/val split is done by page, not by crop, so no page appears in both sets
  3. Best model checkpoint and charset.json are saved to disk after training
  4. evaluate.py loads the checkpoint and writes eval_report.csv with per-crop predictions, CER, and exact match rate
  5. ClearML tasks train_baseline_ctc and evaluate_model log all metrics, artifacts, and CLI hyperparameters
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 4/4 | Complete   | 2026-04-21 |
| 2. Review & Annotation | 0/? | Not started | - |
| 3. Training & Evaluation | 0/? | Not started | - |
