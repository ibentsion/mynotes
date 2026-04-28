"""prepare_data.py — scan PDFs into flagged grayscale crops + manifest + ClearML task."""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from pdf2image import convert_from_path

from src.clearml_utils import (
    init_task,
    maybe_create_dataset,
    report_manifest_stats,
    upload_file_artifact,
)
from src.flagging import flag_region
from src.manifest_schema import MANIFEST_COLUMNS
from src.region_detector import detect_regions, preprocess_page


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert PDFs into grayscale crops with heuristic flags."
    )
    p.add_argument("--pdf_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--dilation_kernel_w", type=int, default=15)
    p.add_argument("--dilation_kernel_h", type=int, default=3)
    p.add_argument("--dilation_iters", type=int, default=3)
    p.add_argument("--angle_thresh", type=float, default=15.0)
    p.add_argument("--min_area", type=int, default=500)
    p.add_argument("--margin_px", type=int, default=30)
    p.add_argument("--faint_thresh", type=int, default=200)
    p.add_argument("--aspect_lo", type=float, default=0.1)
    p.add_argument("--aspect_hi", type=float, default=10.0)
    p.add_argument("--max_aspect_ratio", type=float, default=8.0,
                   help="Drop regions with w/h above this (filters horizontal lines/underlines)")
    p.add_argument(
        "--dataset_name",
        type=str,
        default="data-pipeline-v1",
        help="ClearML dataset name",
    )
    return p


def _process_pdf(
    pdf_path: Path,
    pages_dir: Path,
    crops_dir: Path,
    args: argparse.Namespace,
) -> list[dict]:
    rows: list[dict] = []
    page_paths = convert_from_path(
        str(pdf_path),
        dpi=args.dpi,
        grayscale=True,
        output_folder=str(pages_dir),
        paths_only=True,
        fmt="png",
    )
    for page_idx, page_path_str in enumerate(page_paths, start=1):
        page_path = Path(page_path_str)  # ty: ignore[invalid-argument-type]  # paths_only=True guarantees str
        gray = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Failed to read page image: {page_path}")

        binary = preprocess_page(gray)
        stats = detect_regions(
            binary,
            dilation_kernel_w=args.dilation_kernel_w,
            dilation_kernel_h=args.dilation_kernel_h,
            dilation_iters=args.dilation_iters,
            max_aspect_ratio=args.max_aspect_ratio,
        )
        all_boxes: list[tuple[int, int, int, int]] = [
            (int(x), int(y), int(w), int(h)) for x, y, w, h, _ in stats
        ]

        for region_idx, (x, y, w, h, _area) in enumerate(stats):
            x, y, w, h = int(x), int(y), int(w), int(h)
            gray_crop = gray[y : y + h, x : x + w]
            crop_name = f"{pdf_path.stem}_p{page_idx:03d}_r{region_idx:04d}.png"
            crop_path = crops_dir / crop_name
            if not cv2.imwrite(str(crop_path), gray_crop):
                raise RuntimeError(f"Failed to write crop: {crop_path}")

            reasons = flag_region(
                gray_crop,
                x,
                y,
                w,
                h,
                all_boxes,
                page_shape=gray.shape,
                angle_thresh=args.angle_thresh,
                min_area=args.min_area,
                margin_px=args.margin_px,
                faint_thresh=args.faint_thresh,
                aspect_lo=args.aspect_lo,
                aspect_hi=args.aspect_hi,
            )
            rows.append(
                {
                    "crop_path": str(crop_path),
                    "pdf_path": str(pdf_path),
                    "page_path": str(page_path),
                    "page_num": page_idx,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": w * h,
                    "is_flagged": len(reasons) > 0,
                    "flag_reasons": ",".join(reasons),
                    "status": "unlabeled",
                    "label": "",
                    "notes": "",
                }
            )
    return rows


def main() -> int:
    # Pitfall 2: Task.init BEFORE argparse.parse_args()
    task = init_task("handwriting-hebrew-ocr", "data_prep", tags=["phase-1"])

    parser = _build_parser()
    args = parser.parse_args()

    pdf_dir: Path = args.pdf_dir
    output_dir: Path = args.output_dir
    if not pdf_dir.is_dir():
        print(f"ERROR: --pdf_dir does not exist: {pdf_dir}", file=sys.stderr)
        return 2

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: no PDFs found in {pdf_dir}", file=sys.stderr)
        return 3

    pages_dir = output_dir / "pages"
    crops_dir = output_dir / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # CLML-01: log PDF list
    task.connect({"pdf_list": [p.name for p in pdfs]}, name="inputs")

    all_rows: list[dict] = []
    for pdf in pdfs:
        print(f"Processing {pdf.name}")
        all_rows.extend(_process_pdf(pdf, pages_dir, crops_dir, args))

    df = pd.DataFrame(all_rows, columns=MANIFEST_COLUMNS)
    manifest_path = output_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    # DATA-06: priority sort
    review_queue = df.sort_values(
        ["is_flagged", "area"], ascending=[False, False]
    ).reset_index(drop=True)
    review_queue_path = output_dir / "review_queue.csv"
    review_queue.to_csv(review_queue_path, index=False)

    # CLML-01: upload artifacts + CLML-02: version dataset
    upload_file_artifact(task, "manifest", manifest_path)
    upload_file_artifact(task, "review_queue", review_queue_path)
    report_manifest_stats(task, df)
    maybe_create_dataset(
        "handwriting-hebrew-ocr",
        args.dataset_name,
        folders=[pages_dir, crops_dir],
        files=[manifest_path, review_queue_path],
    )

    print(
        f"Done. {len(df)} crops written. "
        f"Flagged: {int(df['is_flagged'].sum())}. "
        f"manifest={manifest_path} review_queue={review_queue_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
