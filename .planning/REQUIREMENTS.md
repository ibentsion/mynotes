# Requirements: Hebrew Handwriting OCR Pipeline

**Defined:** 2026-04-21
**Core Value:** A reviewable, labeled dataset of personal Hebrew handwriting that a baseline OCR model can train on — getting the data pipeline and annotation workflow right matters more than model accuracy at MVP.

## v1 Requirements

### Data Preparation

- [ ] **DATA-01**: Pipeline converts scanned PDFs to per-page images (pdf2image + Poppler)
- [x] **DATA-02**: Pipeline preprocesses pages (grayscale, normalize contrast) before region detection
- [x] **DATA-03**: Region detector extracts handwriting regions using a region-first approach (not strict line segmentation)
- [ ] **DATA-04**: Each detected region is saved as a grayscale image crop with metadata (page, bounding box, dimensions)
- [ ] **DATA-05**: Pipeline produces manifest.csv with one row per crop (path, page, bbox, flags, status)
- [ ] **DATA-06**: Pipeline produces review_queue.csv sorted by labeling priority (flagged → large/mixed → diverse → easy)

### Flagging

- [x] **FLAG-01**: Regions are scored and flagged using heuristics: strong angle/diagonal text
- [x] **FLAG-02**: Regions are flagged for: overlap with other detected regions
- [x] **FLAG-03**: Regions are flagged for: very tall region, tiny region, unusual aspect ratio
- [x] **FLAG-04**: Regions are flagged for: margin note candidate (near page edge)
- [x] **FLAG-05**: Regions are flagged for: faint/low-contrast content
- [ ] **FLAG-06**: Flag reasons are stored per region in manifest.csv

### Review App

- [x] **REVW-01**: Streamlit app loads manifest.csv and displays crops for review
- [x] **REVW-02**: User can filter crops by status: unlabeled, flagged, labeled, all
- [ ] **REVW-03**: User can transcribe Hebrew text for a crop (edit label field)
- [ ] **REVW-04**: User can set crop status: unlabeled / labeled / skip / bad_seg / merge_needed
- [ ] **REVW-05**: User can add free-text review notes per crop
- [ ] **REVW-06**: App saves changes back to manifest.csv on each update

### ClearML Sync

- [x] **SYNC-01**: review_to_clearml script summarizes current labeling status (counts per status) and uploads to ClearML task `manual_review_summary`
- [x] **SYNC-02**: Updated manifest.csv is uploaded as artifact to `manual_review_summary` task

### Training

- [x] **TRAN-01**: train_ctc.py loads only crops with status=labeled from manifest.csv
- [x] **TRAN-02**: Charset is built dynamically from labeled Hebrew text with Unicode normalization
- [x] **TRAN-03**: Train/val split is done by page (not random crop) to prevent leakage
- [x] **TRAN-04**: Model is CRNN: CNN feature extractor → BiLSTM → CTC loss
- [x] **TRAN-05**: Training runs on CPU (device auto-detected via torch.device)
- [x] **TRAN-06**: Best model checkpoint and charset.json are saved to disk
- [x] **TRAN-07**: Training hyperparameters are passed via CLI and connected to ClearML via Task.connect()
- [x] **TRAN-08**: ClearML task `train_baseline_ctc` logs: train loss, val loss, val CER per epoch; uploads checkpoint, charset.json, training config

### Evaluation

- [ ] **EVAL-01**: evaluate.py runs inference on the validation set using the saved checkpoint
- [x] **EVAL-02**: CER is computed on validation set
- [ ] **EVAL-03**: eval_report.csv is exported: image_path, target, prediction, is_exact
- [ ] **EVAL-04**: ClearML task `evaluate_model` logs final CER, exact match rate, and uploads eval_report.csv

### ClearML Infrastructure

- [ ] **CLML-01**: prepare_data.py initializes ClearML task `data_prep`, logs PDF list and prep parameters, uploads manifest.csv and review_queue.csv as artifacts
- [ ] **CLML-02**: ClearML dataset is created/versioned containing page images, crop images, and manifest files
- [x] **CLML-03**: clearml_utils.py provides shared helpers: init_task(), upload_file_artifact(), report_manifest_stats(), maybe_create_dataset()
- [x] **CLML-04**: All scripts save git commit hash and log package versions for reproducibility
- [ ] **CLML-05**: All scripts accept explicit CLI arguments that are tracked in ClearML

## v2 Requirements

### Active Learning

- **ACTV-01**: Model uncertainty sampling to prioritize next crops to label
- **ACTV-02**: Pseudo-labeling on high-confidence unlabeled crops

### Advanced Review

- **ADVR-01**: Split/merge editor on full page view for bad_seg corrections
- **ADVR-02**: FiftyOne integration for visual error analysis of crops and predictions

### Model Expansion

- **MODX-01**: Comparison with TrOCR / Kraken-based recognizers
- **MODX-02**: LLM-based post-correction of OCR output

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time / server-based review UI | Personal local tool — Streamlit local-first is sufficient |
| Multi-user collaboration | Single user (personal notes) |
| Language support beyond Hebrew | Out of scope for this dataset |
| Perfect OCR accuracy at MVP | MVP goal is dataset quality, not model performance |
| Cloud training infrastructure | CPU training is feasible at MVP scale; cloud is manual opt-in |
| Automatic bad_seg correction | Requires human judgment given irregular handwriting |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Pending |
| DATA-06 | Phase 1 | Pending |
| FLAG-01 | Phase 1 | Complete |
| FLAG-02 | Phase 1 | Complete |
| FLAG-03 | Phase 1 | Complete |
| FLAG-04 | Phase 1 | Complete |
| FLAG-05 | Phase 1 | Complete |
| FLAG-06 | Phase 1 | Pending |
| CLML-01 | Phase 1 | Pending |
| CLML-02 | Phase 1 | Pending |
| CLML-03 | Phase 1 | Complete |
| CLML-04 | Phase 1 | Complete (01-01) |
| CLML-05 | Phase 1 | Pending |
| REVW-01 | Phase 2 | Complete |
| REVW-02 | Phase 2 | Complete |
| REVW-03 | Phase 2 | Pending |
| REVW-04 | Phase 2 | Pending |
| REVW-05 | Phase 2 | Pending |
| REVW-06 | Phase 2 | Pending |
| SYNC-01 | Phase 2 | Complete |
| SYNC-02 | Phase 2 | Complete |
| TRAN-01 | Phase 3 | Complete |
| TRAN-02 | Phase 3 | Complete |
| TRAN-03 | Phase 3 | Complete |
| TRAN-04 | Phase 3 | Complete |
| TRAN-05 | Phase 3 | Complete |
| TRAN-06 | Phase 3 | Complete |
| TRAN-07 | Phase 3 | Complete |
| TRAN-08 | Phase 3 | Complete |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Complete |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-21 after roadmap creation*
