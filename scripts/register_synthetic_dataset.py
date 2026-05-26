"""Register outputs/synthetic/ as a ClearML dataset and print the dataset ID."""

import sys
from pathlib import Path

from src.clearml_utils import maybe_create_dataset
from src.run_config import update_config

SYNTHETIC_DIR = Path("outputs/synthetic")
DATASET_PROJECT = "handwriting-hebrew-ocr"
DATASET_NAME = "synthetic_hebrew"


def main() -> int:
    crops_dir = SYNTHETIC_DIR / "crops"
    if not crops_dir.is_dir() or not any(crops_dir.glob("*.png")):
        print(f"ERROR: {crops_dir} missing or contains no PNGs", file=sys.stderr)
        return 1

    manifest = SYNTHETIC_DIR / "manifest.csv"
    if not manifest.is_file():
        print(f"ERROR: {manifest} not found", file=sys.stderr)
        return 1

    dataset_id = maybe_create_dataset(
        project=DATASET_PROJECT,
        dataset_name=DATASET_NAME,
        folders=[(crops_dir, "crops")],
        files=[manifest],
    )
    print(dataset_id)
    update_config(**{"datasets.synthetic_id": dataset_id})  # ty: ignore[invalid-argument-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
