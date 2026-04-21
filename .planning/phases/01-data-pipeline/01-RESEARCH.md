# Phase 1: Data Pipeline - Research

**Researched:** 2026-04-21
**Domain:** PDF ingestion, image preprocessing, region detection, heuristic flagging, ClearML tracking
**Confidence:** HIGH

## Summary

Phase 1 builds the core data pipeline: convert scanned PDFs to page images, detect handwriting
regions using a region-first approach, apply heuristic flagging for suspicious regions, and emit
two output files (manifest.csv and review_queue.csv) with full ClearML experiment tracking.

The technology choices are all well-established. pdf2image wraps poppler's pdftoppm, which is
already installed on the machine (v24.02.0). OpenCV provides all the primitives needed for
preprocessing and region detection — CLAHE for contrast normalization, Otsu/adaptive thresholding
for binarization, connectedComponentsWithStats for region extraction, minAreaRect for angle
scoring, and bounding-box arithmetic for overlap and margin proximity checks. ClearML 2.1.5 is
already installed and handles task init, argparse auto-connection, git-commit capture, artifact
upload, and dataset versioning in one SDK.

The main planning risk is Python 3.13 (required by project constraints) not being installed — it
must be fetched via `uv python install 3.13` before the venv is created. Everything else is
available. The pipeline script (prepare_data.py) should be a single CLI tool that runs all steps
sequentially and exits with a non-zero code on failure.

**Primary recommendation:** Use connectedComponentsWithStats (with morphological dilation to merge
adjacent ink blobs into regions) as the region detector. Apply CLAHE + Gaussian blur preprocessing
before thresholding. Score each region with five independent heuristic checks and store the union
of triggered checks in manifest.csv flag_reasons column.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DATA-01 | Pipeline converts scanned PDFs to per-page images (pdf2image + Poppler) | pdf2image 1.17.0 + poppler already installed; `convert_from_path(dpi=300, grayscale=True, output_folder=...)` |
| DATA-02 | Preprocesses pages (grayscale, normalize contrast) before region detection | CLAHE + GaussianBlur + Otsu threshold pipeline documented below |
| DATA-03 | Region detector extracts regions using a region-first approach | connectedComponentsWithStats on dilated threshold image; morphological dilation merges nearby strokes |
| DATA-04 | Each region saved as grayscale crop with metadata (page, bbox, dimensions) | Slice numpy array, cv2.imwrite; metadata captured from stats array |
| DATA-05 | Pipeline produces manifest.csv (path, page, bbox, flags, status) | pandas DataFrame → to_csv; schema defined below |
| DATA-06 | Pipeline produces review_queue.csv sorted by labeling priority | Sort by (is_flagged DESC, area DESC); write separate CSV |
| FLAG-01 | Regions flagged for strong angle/diagonal text | minAreaRect on region contour; flag if abs(angle) > threshold |
| FLAG-02 | Regions flagged for overlap with other regions | Pairwise IoU or containment check on bounding boxes |
| FLAG-03 | Regions flagged for very tall, tiny, or unusual aspect ratio | From stats: width/height ratio, area thresholds |
| FLAG-04 | Regions flagged for margin note candidate (near page edge) | Bounding box proximity to image border (configurable margin_px) |
| FLAG-05 | Regions flagged for faint/low-contrast content | Mean pixel intensity of crop on preprocessed grayscale image |
| FLAG-06 | Flag reasons stored per region in manifest.csv | Comma-separated string in flag_reasons column; empty if not flagged |
| CLML-01 | prepare_data.py initializes ClearML task `data_prep`, logs PDF list + params, uploads manifests | Task.init(project_name="handwriting-hebrew-ocr", task_name="data_prep") + upload_artifact() |
| CLML-02 | ClearML dataset versioned with page images, crop images, manifest files | Dataset.create() → add_files() → upload() → finalize() |
| CLML-03 | clearml_utils.py provides shared helpers | Module-level functions: init_task(), upload_file_artifact(), report_manifest_stats(), maybe_create_dataset() |
| CLML-04 | All scripts save git commit hash and log package versions | ClearML captures git hash automatically; log pip versions via task.connect() |
| CLML-05 | All scripts accept explicit CLI arguments tracked in ClearML | argparse + auto_connect_arg_parser=True (default) in Task.init() |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Runtime**: Python 3.13, CPU-only — model must train without CUDA
- **Stack**: pdf2image + Poppler, OpenCV, PyTorch, Streamlit, ClearML — no additional heavy dependencies
- **Data**: Personal Hebrew notes; privacy-sensitive — stays local
- **Reproducibility**: Git commit, package versions, and all configs stored in ClearML per run
- **Modularity**: Scripts are independent CLI tools, not a monolithic app
- **Poppler**: Required system dependency for pdf2image on Linux (already installed: v24.02.0)
- **Code style**: No docstrings on internal functions; Google-style on public APIs; no inline comments for obvious logic
- **Global CLAUDE.md**: uv for deps, ruff for lint/format, ty for type checking, pytest for tests
- **Error handling**: Fail fast with clear actionable messages; never swallow exceptions

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pdf2image | 1.17.0 | PDF page → PIL Image via poppler/pdftoppm | Thin poppler wrapper; no JVM; `grayscale` param built in |
| opencv-python-headless | 4.13.0.92 | Image preprocessing, region detection, geometry | Headless (no GUI deps); all required primitives in one package |
| numpy | 2.4.4 | Array math, region stat extraction | Installed; required by opencv |
| pandas | 3.0.2 | manifest.csv / review_queue.csv generation | Standard for tabular data |
| clearml | 2.1.5 | Experiment tracking, artifact upload, dataset versioning | Already installed; project-mandated |
| Pillow | (pdf2image dep) | PIL image returned by pdf2image | Automatic transitive dep |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pathlib | stdlib | File path manipulation | Always — avoid os.path strings |
| argparse | stdlib | CLI argument parsing with ClearML auto-connect | All scripts must use argparse |
| subprocess | stdlib | Not needed (pdf2image handles poppler calls) | Only if direct poppler CLI needed |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| opencv-python-headless | opencv-python (full) | Full version pulls in GUI deps (Qt); unnecessary on a headless pipeline script |
| connectedComponentsWithStats | MSER | MSER detects character-level blobs, not word/line regions; needs post-merge step; more complex |
| connectedComponentsWithStats | watershed | Watershed better for touching objects; overhead not needed at region level |
| pandas CSV | csv stdlib | pandas handles dtypes and NaN cleanly; CSV module is fine for trivial cases |

**Installation (Wave 0 — into uv venv with Python 3.13):**
```bash
uv python install 3.13
uv venv --python 3.13
uv pip install pdf2image==1.17.0 opencv-python-headless==4.13.0.92 numpy==2.4.4 pandas==3.0.2 clearml==2.1.5
```

**Note:** Python 3.13 is not installed on this machine. `uv python install 3.13` downloads it.
Poppler is already present (v24.02.0 system package).

## Architecture Patterns

### Recommended Project Structure

```
src/
├── prepare_data.py      # Main CLI: PDF → manifest.csv + review_queue.csv + ClearML task
├── clearml_utils.py     # Shared helpers: init_task(), upload_file_artifact(), etc.
└── region_detector.py   # (optional module) Region detection logic if prepare_data.py > 100 lines

data/
├── pdfs/                # Input PDFs (gitignored — privacy-sensitive)
├── pages/               # Per-page PNG images output
└── crops/               # Per-region grayscale crop images

outputs/
├── manifest.csv         # One row per crop; uploaded to ClearML as artifact
└── review_queue.csv     # Sorted by labeling priority; uploaded to ClearML
```

### Pattern 1: PDF-to-Page Conversion

**What:** Convert each PDF to per-page images, write to disk, return paths.
**When to use:** DATA-01 — always first step.
**Key params:** `dpi=300` (higher = better region detection; 200 is minimum), `grayscale=True`, `output_folder=pages_dir`, `paths_only=True` (avoids loading all pages into RAM).

```python
# Source: https://github.com/Belval/pdf2image
from pdf2image import convert_from_path
from pathlib import Path

def pdf_to_pages(pdf_path: Path, output_dir: Path, dpi: int = 300) -> list[Path]:
    paths = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        grayscale=True,
        output_folder=str(output_dir),
        paths_only=True,
        fmt="png",
    )
    return [Path(p) for p in paths]
```

### Pattern 2: Page Preprocessing

**What:** CLAHE contrast normalization + Gaussian blur before thresholding.
**When to use:** DATA-02 — before any region detection.
**Why CLAHE over plain equalize:** Handles uneven scan illumination without over-amplifying noise.

```python
# Source: https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
import cv2
import numpy as np

def preprocess_page(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary
```

### Pattern 3: Region-First Detection via Connected Components

**What:** Morphological dilation merges nearby ink strokes into word/line blobs; connectedComponentsWithStats extracts one bounding box per region.
**When to use:** DATA-03 — the core region extraction step.
**Why dilation first:** Direct CCL on raw ink gives character-level blobs; dilation merges adjacent ink into coherent text regions.

```python
# Source: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html
import cv2
import numpy as np

def detect_regions(binary: np.ndarray, dilation_iters: int = 3) -> np.ndarray:
    """Return stats array: [x, y, w, h, area] per component (excluding background)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, kernel, iterations=dilation_iters)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    # stats shape: (num_labels, 5) — CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA
    return stats[1:]  # skip background (label 0)
```

### Pattern 4: Heuristic Flagging

**What:** Five independent checks per region; results stored as comma-separated reasons.
**When to use:** FLAG-01 through FLAG-06.

```python
# Source: OpenCV minAreaRect docs + PyImageSearch skew correction guide
import cv2
import numpy as np

def flag_region(
    gray_crop: np.ndarray,
    x: int, y: int, w: int, h: int,
    all_boxes: list[tuple[int,int,int,int]],
    page_shape: tuple[int, int],
    *,
    angle_thresh: float = 15.0,
    min_area: int = 500,
    margin_px: int = 30,
    faint_thresh: int = 200,
    aspect_lo: float = 0.1,
    aspect_hi: float = 10.0,
) -> list[str]:
    reasons = []
    ph, pw = page_shape

    # FLAG-01: angle
    coords = np.column_stack(np.where(gray_crop < 128))
    if len(coords) > 10:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > angle_thresh:
            reasons.append("angle")

    # FLAG-02: overlap with any other box (IoU > 0)
    for ox, oy, ow, oh in all_boxes:
        ix1, iy1 = max(x, ox), max(y, oy)
        ix2, iy2 = min(x+w, ox+ow), min(y+h, oy+oh)
        if ix1 < ix2 and iy1 < iy2:
            reasons.append("overlap")
            break

    # FLAG-03: size / aspect ratio
    area = w * h
    ratio = w / h if h > 0 else 0
    if area < min_area or ratio < aspect_lo or ratio > aspect_hi:
        reasons.append("size_aspect")

    # FLAG-04: margin proximity
    if x < margin_px or y < margin_px or (pw - x - w) < margin_px or (ph - y - h) < margin_px:
        reasons.append("margin")

    # FLAG-05: faint content (high mean = mostly white = little ink)
    if gray_crop.mean() > faint_thresh:
        reasons.append("faint")

    return reasons
```

### Pattern 5: ClearML Task Initialization and Artifact Upload

**What:** Shared helpers in clearml_utils.py consumed by all pipeline scripts.
**When to use:** CLML-01, CLML-02, CLML-03.

```python
# Source: https://clear.ml/docs/latest/docs/references/sdk/task/
# Source: https://clear.ml/docs/latest/docs/clearml_data/clearml_data_sdk/
from clearml import Task, Dataset
from pathlib import Path

def init_task(project: str, task_name: str, tags: list[str] | None = None) -> Task:
    return Task.init(
        project_name=project,
        task_name=task_name,
        tags=tags or [],
        # auto_connect_arg_parser=True is default — argparse params auto-logged
    )

def upload_file_artifact(task: Task, name: str, path: Path) -> None:
    task.upload_artifact(name=name, artifact_object=str(path))

def report_manifest_stats(task: Task, df) -> None:
    logger = task.get_logger()
    logger.report_scalar("crops", "total", iteration=0, value=len(df))
    logger.report_scalar("crops", "flagged", iteration=0, value=(df["is_flagged"] == True).sum())

def maybe_create_dataset(
    project: str,
    dataset_name: str,
    folders: list[Path],
) -> str:
    ds = Dataset.create(dataset_name=dataset_name, dataset_project=project)
    for folder in folders:
        ds.add_files(str(folder))
    ds.upload()
    ds.finalize()
    return ds.id
```

### Pattern 6: manifest.csv Schema

```python
# One row per crop
MANIFEST_COLUMNS = [
    "crop_path",      # str  — relative path from project root
    "pdf_path",       # str  — source PDF
    "page_num",       # int  — 1-indexed
    "x", "y",         # int  — top-left corner in page image coords
    "w", "h",         # int  — crop dimensions in pixels
    "area",           # int  — w * h
    "is_flagged",     # bool — True if any flag reason triggered
    "flag_reasons",   # str  — comma-separated list of triggered reasons (empty if none)
    "status",         # str  — always "unlabeled" at creation time
    "label",          # str  — empty at creation time (filled by review app)
    "notes",          # str  — empty at creation time
]
```

### Anti-Patterns to Avoid

- **Loading all PDF pages into RAM at once:** Use `paths_only=True` and `output_folder` in pdf2image. Without these, large PDFs will exhaust memory.
- **Running CCL on raw grayscale (non-binary):** connectedComponentsWithStats operates on binary images. Pass thresholded image, not grayscale.
- **Using `cv2.threshold` with a fixed value for scanned docs:** Uneven scan lighting means a fixed threshold will fail. Always use `THRESH_OTSU` or `adaptiveThreshold`.
- **Calling `Task.init()` after argparse `parse_args()`:** ClearML must intercept `parse_args()` to auto-log arguments. Call `Task.init()` before argparse parses.
- **Leaving `Dataset.finalize()` uncalled:** An unfinalized dataset is in Draft state and cannot be reliably consumed by downstream tasks.
- **Hardcoding thresholds:** All heuristic thresholds (angle_thresh, min_area, margin_px, faint_thresh, aspect ratios) must be CLI arguments so ClearML logs them per run.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PDF-to-image conversion | Custom pdftoppm subprocess wrapper | pdf2image 1.17.0 | Handles multi-page, memory, threading, and format options |
| Connected component stats | Manual flood-fill or contour grouping | `cv2.connectedComponentsWithStats` | Returns (x, y, w, h, area) per component in one call |
| Contrast normalization | Manual histogram stretching | `cv2.createCLAHE` | Handles tile-local normalization with noise clipping |
| Automatic threshold | Manual threshold tuning per image | `cv2.THRESH_OTSU` | Bimodal histogram analysis; works across different scan conditions |
| CSV output | Custom file writer | pandas `DataFrame.to_csv()` | Handles NaN, dtypes, encoding; consistent quoting |
| Experiment reproducibility | Manual git hash capture | ClearML `Task.init()` | Auto-captures git hash, uncommitted diff, pip freeze per run |
| Dataset versioning | Manual file hash tracking | `clearml.Dataset` | Immutable finalized versions with parent inheritance |

**Key insight:** The image processing primitives (CLAHE, Otsu, morphological ops, CCL) are all
available in a single OpenCV install. There is no need to add scikit-image or any separate
thresholding library.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Poppler (pdftoppm) | pdf2image / DATA-01 | YES | 24.02.0 | — |
| ClearML | CLML-01 to CLML-05 | YES | 2.1.5 | — |
| Python 3.13 | Project constraint | NO (3.12 present) | — | `uv python install 3.13` (download required) |
| pdf2image | DATA-01 | NO (not installed) | — | Install via uv pip |
| opencv-python-headless | DATA-02, DATA-03, FLAG-01 to FLAG-06 | NO (not installed) | — | Install via uv pip |
| numpy | All array ops | NO in project venv | 2.4.4 on system | Install via uv pip |
| pandas | DATA-05, DATA-06 | NO (not installed) | — | Install via uv pip |

**Missing dependencies with no fallback:**
- Python 3.13 (project constraint) — must download via `uv python install 3.13` before venv creation
- pdf2image, opencv-python-headless, pandas — must be installed; no alternative within stack constraint

**Missing dependencies with fallback:**
- None — all fallbacks are "install the required library"

**Note:** A Wave 0 task must create the venv and install all dependencies before any pipeline
code can execute.

## Common Pitfalls

### Pitfall 1: Memory Exhaustion on Large PDFs

**What goes wrong:** `convert_from_path()` without `output_folder` and `paths_only=True` loads all
page images into a Python list in RAM. A 20-page PDF at 300 DPI is ~500MB.
**Why it happens:** Default behavior returns PIL Image objects.
**How to avoid:** Always pass `output_folder=str(pages_dir)` and `paths_only=True`.
**Warning signs:** OOM kill during conversion of multi-page PDFs.

### Pitfall 2: ClearML Task.init() Called After argparse.parse_args()

**What goes wrong:** CLI arguments are not captured in ClearML hyperparameters section despite
`auto_connect_arg_parser=True`.
**Why it happens:** ClearML patches `parse_args()` at import time; calling `Task.init()` after
`parse_args()` means the patch fires after the values are already read.
**How to avoid:** Call `Task.init()` at the top of `if __name__ == "__main__":`, before
constructing the argparse parser.
**Warning signs:** Empty `Args` section in ClearML task hyperparameters.

### Pitfall 3: Region Explosion from Fine-Grained CCL

**What goes wrong:** Running connectedComponentsWithStats on an undilated binary image produces
thousands of character-level components, not word/line regions.
**Why it happens:** Each disconnected ink stroke is its own component.
**How to avoid:** Apply morphological dilation (horizontal kernel ~15px wide, 3px tall) before CCL.
Tune dilation kernel size to the expected stroke density.
**Warning signs:** manifest.csv with 2000+ rows from a single page.

### Pitfall 4: minAreaRect Angle Convention Confusion

**What goes wrong:** Regions with clearly diagonal text are not flagged because the angle check
uses the raw minAreaRect output without the -45° convention correction.
**Why it happens:** minAreaRect returns angles in [-90, 0); for near-vertical rectangles the
reported angle must be corrected by adding 90°.
**How to avoid:** Apply the standard heuristic: `if angle < -45: angle = -(90 + angle)`.
**Warning signs:** No regions flagged with "angle" even on obviously tilted crops.

### Pitfall 5: Dataset Not Finalized Before Downstream Use

**What goes wrong:** A downstream task calling `Dataset.get()` on an unfinalized dataset gets
an error or retrieves an incomplete version.
**Why it happens:** Datasets in Draft state are mutable and may be in inconsistent state.
**How to avoid:** Always call `dataset.finalize()` after `dataset.upload()`. Check return code.
**Warning signs:** ClearML dataset shown as "Draft" in web UI after pipeline completes.

### Pitfall 6: Fixed Binarization Threshold on Variable Scans

**What goes wrong:** Regions appear blank or have huge blobs because thresholding fails on a
page that was scanned differently (different brightness, paper color, scan angle).
**Why it happens:** CamScanner-processed PDFs still have local illumination variance; a single
Otsu threshold on the whole page may miss low-contrast areas.
**How to avoid:** Apply CLAHE before Otsu. If Otsu still fails, fall back to adaptiveThreshold
with a blockSize proportional to expected character size.
**Warning signs:** manifest.csv rows with `faint` flag covering >50% of crops on a page.

## Code Examples

### End-to-End prepare_data.py skeleton

```python
# Source: ClearML Task SDK docs + pdf2image + OpenCV docs
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
from clearml_utils import init_task, upload_file_artifact, report_manifest_stats, maybe_create_dataset

def main() -> None:
    # Task.init MUST come before argparse.parse_args()
    from clearml_utils import init_task
    task = init_task("handwriting-hebrew-ocr", "data_prep")

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--angle_thresh", type=float, default=15.0)
    parser.add_argument("--min_area", type=int, default=500)
    parser.add_argument("--margin_px", type=int, default=30)
    parser.add_argument("--faint_thresh", type=int, default=200)
    args = parser.parse_args()
    # argparse params are auto-logged by ClearML at this point

    # ... pipeline steps ...

    upload_file_artifact(task, "manifest", Path(args.output_dir) / "manifest.csv")
    upload_file_artifact(task, "review_queue", Path(args.output_dir) / "review_queue.csv")
    report_manifest_stats(task, df)
    maybe_create_dataset(
        "handwriting-hebrew-ocr",
        "data-pipeline-v1",
        [Path(args.output_dir) / "pages", Path(args.output_dir) / "crops"],
    )

if __name__ == "__main__":
    main()
```

### Priority Sorting for review_queue.csv

```python
# Priority: flagged first, then larger area first
df["is_flagged"] = df["flag_reasons"].str.len() > 0
review_queue = df.sort_values(
    ["is_flagged", "area"],
    ascending=[False, False],
).reset_index(drop=True)
review_queue.to_csv(output_dir / "review_queue.csv", index=False)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| cv2.equalizeHist (global) | cv2.createCLAHE (tile-local) | OpenCV 2.x era | CLAHE avoids noise overamplification in homogeneous regions |
| Fixed binary threshold | THRESH_OTSU | OpenCV 2.x era | Automatic bimodal separation; adapts per image |
| character-level segmentation | Region-first (word/line blobs via dilation) | Standard for irregular handwriting | Avoids false splits on diagonal/overlapping text |
| clearml <= 1.x Dataset API | clearml 2.x Dataset.create() + finalize() | clearml 2.0.0 | `get_or_create` replaced; explicit create → upload → finalize lifecycle |

**Deprecated/outdated:**
- `pdf2image.pdfinfo_from_path()` for page count: still works but `convert_from_path(paths_only=True)` is sufficient
- Global histogram equalization (`cv2.equalizeHist`): superseded by CLAHE for document images

## Open Questions

1. **Dilation kernel size for Hebrew handwriting**
   - What we know: horizontal kernel (15×3) merges Latin word-level strokes
   - What's unclear: Hebrew letter spacing and stroke width may need smaller kernel (characters are dense); diagonal text needs a more isotropic kernel
   - Recommendation: Make kernel size a CLI arg (`--dilation_kernel_w`, `--dilation_kernel_h`); default to 12×3 and tune on real data

2. **DPI choice: 200 vs. 300**
   - What we know: pdf2image default is 200; 300 is better for detection but 50% more disk/RAM
   - What's unclear: CamScanner PDFs may already embed at a fixed resolution; re-rendering at 300 DPI may add no information
   - Recommendation: Default 300, expose as `--dpi` arg, note in ClearML run

3. **Overlap detection strategy**
   - What we know: pairwise IoU is O(n²) — fine at <200 regions per page
   - What's unclear: Whether 200+ regions per page ever occurs (would need spatial indexing)
   - Recommendation: Implement naive pairwise for MVP; add note to revisit if pages produce >500 components

## Sources

### Primary (HIGH confidence)
- pdf2image GitHub README — API functions, parameters, known issues
- OpenCV 4.x official docs (`docs.opencv.org`) — connectedComponentsWithStats, CLAHE, minAreaRect, adaptive threshold
- ClearML official docs (`clear.ml/docs/latest`) — Task.init(), Task.connect(), upload_artifact(), Dataset lifecycle

### Secondary (MEDIUM confidence)
- PyImageSearch text skew correction guide — minAreaRect angle convention (-45° heuristic); verified against OpenCV Q&A forum
- PyImageSearch CLAHE guide — clipLimit parameter range (2–5); consistent with OpenCV docs
- ClearML PyPI release history — confirmed 2.1.5 is current stable

### Tertiary (LOW confidence)
- None — all key findings verified against primary or secondary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified against PyPI registry; poppler confirmed installed
- Architecture: HIGH — pdf2image + OpenCV patterns verified against official docs and GitHub README
- Pitfalls: HIGH — angle convention and Task.init() ordering verified against OpenCV forum and ClearML docs

**Research date:** 2026-04-21
**Valid until:** 2026-07-21 (stable libraries; ClearML API relatively stable at 2.x)
