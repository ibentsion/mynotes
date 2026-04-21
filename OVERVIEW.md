Build an MVP pipeline for personal handwritten Hebrew OCR from scanned PDF pages, with minimal labeling, manual review for bad segmentation, and ClearML for data management + experiment tracking.

Context:

* Input files are scanned PDFs exported from Gmail.
* PDFs already had basic denoise/cleanup from CamScanner.
* Handwriting is not always on straight lines.
* There may be diagonal text, margin notes, added text written later, overlapping regions, and irregular layouts.
* The system must detect likely segmentation problems and let me correct them before training.

Goal:
Create a local Python project that:

1. Converts PDFs to page images
2. Detects handwriting regions on each page
3. Flags suspicious regions for review
4. Lets me manually label/fix segmentation issues
5. Trains a baseline OCR model with minimal labeled data
6. Evaluates performance and exports an error report
7. Logs all dataset versions, manifests, metrics, and model artifacts to ClearML

Recommended project structure:

* requirements.txt
* prepare_data.py
* review_app.py
* train_ctc.py
* evaluate.py
* clearml_utils.py
* data/

  * raw_pdfs/
  * pages/
  * crops/
  * manifests/
  * labels/
  * models/

Technical direction:

* Use a region-first segmentation approach, not strict line-only segmentation.
* Start from page-level preprocessing, then detect text regions that may contain a line, note, angled text, or mixed block.
* Save each region as an image crop plus metadata in a manifest CSV.
* Flag likely problematic regions using heuristics such as:

  * strong angle / diagonal text
  * overlap with other regions
  * very tall region
  * margin note candidate
  * tiny region
  * faint region
  * unusual aspect ratio

Manual review UI should allow:

* viewing the crop
* editing/transcribing text
* setting status:

  * unlabeled
  * labeled
  * skip
  * bad_seg
  * merge_needed
* adding review notes

Minimal-labeling strategy:

* Do not label random crops first.
* Prioritize:

  1. flagged / hard regions
  2. large or mixed regions
  3. diverse handwriting examples
  4. only then easy regions
* Initial target: label about 50–120 regions for MVP.
* Later grow to 150–300 labeled regions.

Baseline model:

* Build a simple OCR baseline using grayscale region crops and labeled text.
* Use a CRNN-style model:

  * CNN feature extractor
  * BiLSTM sequence model
  * CTC loss
* Build charset dynamically from the labeled Hebrew text.
* Normalize Hebrew text before training.
* Split train/validation by page, not random crop, to reduce leakage.
* Save best model checkpoint and charset.

Evaluation:

* Compute CER on validation set.
* Export eval_report.csv with:

  * image_path
  * target
  * prediction
  * is_exact

ClearML integration requirements:

1. ClearML project names:

   * project: handwriting-hebrew-ocr
   * tasks:

     * data_prep
     * manual_review_summary
     * train_baseline_ctc
     * evaluate_model

2. In prepare_data.py:

   * initialize ClearML Task
   * log input PDF list
   * log dataset preparation parameters
   * upload manifest.csv and review_queue.csv as artifacts
   * optionally create/version a ClearML dataset containing:

     * page images
     * crop images
     * manifest files

3. In review workflow:

   * keep Streamlit local-first
   * add an optional script or button to summarize current review status into ClearML:

     * total regions
     * unlabeled count
     * labeled count
     * bad_seg count
     * merge_needed count
     * skip count
     * flagged count
   * upload updated manifest.csv as artifact

4. In train_ctc.py:

   * initialize ClearML Task
   * log:

     * model hyperparameters
     * number of labeled samples
     * charset size
     * train/val split sizes
   * report training loss per epoch
   * report validation loss per epoch
   * report validation CER per epoch
   * upload:

     * best checkpoint
     * charset.json
     * training config
   * connect hyperparameters via Task.connect()

5. In evaluate.py:

   * initialize ClearML Task
   * log:

     * final CER
     * exact match rate
   * upload eval_report.csv as artifact

6. Reproducibility:

   * save git commit if available
   * log package versions
   * store all configs in ClearML
   * keep CLI arguments explicit and tracked

Implementation notes:

* Python stack can include:

  * pdf2image
  * pillow
  * opencv-python
  * numpy
  * pandas
  * streamlit
  * torch
  * torchvision
  * torchmetrics
  * scikit-learn
  * clearml
* On Linux, pdf2image requires Poppler installed.
* Keep the code modular and easy to extend.

Suggested helper module:

* clearml_utils.py

  * init_task(task_name, params)
  * upload_file_artifact(task, name, path)
  * report_manifest_stats(task, df)
  * maybe_create_dataset(dataset_name, paths)

Important constraints:

* This is for handwritten Hebrew, personal notes.
* Segmentation must be reviewable because the writing is messy and not always line-based.
* The MVP should optimize for building a good personal dataset, not for perfect OCR on day one.

Expected deliverables:

1. prepare_data.py

   * convert PDFs to images
   * preprocess pages
   * detect regions
   * save crops
   * create manifest.csv and review_queue.csv
   * log dataset/manifests to ClearML

2. review_app.py

   * Streamlit review tool
   * load manifest
   * filter by unlabeled / flagged / all
   * edit text and status
   * save changes back to manifest

3. review_to_clearml.py or equivalent helper

   * summarize current labeling status
   * upload updated manifest and counters to ClearML

4. train_ctc.py

   * load labeled regions
   * build charset
   * train CRNN+CTC baseline
   * save best checkpoint
   * log metrics/artifacts to ClearML

5. evaluate.py

   * run validation inference
   * compute CER
   * export eval report
   * log final metrics/artifacts to ClearML

6. clearml_utils.py

   * small wrappers for Task init, artifact upload, metrics reporting, dataset versioning

Future extension points:

* active learning based on model uncertainty
* split/merge editor on full page view
* pseudo-labeling on high-confidence unlabeled crops
* LLM-based post-correction
* later comparison with TrOCR / Kraken-based recognizers
* optional FiftyOne integration for visual error analysis of crops and predictions

Please implement a clean first version with comments, reasonable defaults, and CLI-friendly usage.
