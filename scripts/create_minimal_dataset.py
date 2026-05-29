"""Create minimal ClearML datasets for local end-to-end testing.

Creates two standalone datasets (no parent chain):
  - minimal_real: 5 labeled rows from 5 different pages + crop + page images
  - minimal_synth: 5 rows from outputs/synthetic/manifest.csv + crop images

Writes config_minimal.yaml with the new IDs and conservative hyperparams.
Run this once; then verify with:
  uv run python -m src.train_ctc --params /dev/null
  (the config is picked up automatically from config_minimal.yaml if set as CONFIG_PATH)
Or pass explicitly:
  CONFIG_PATH=config_minimal.yaml uv run python -m src.train_ctc
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd

from src.clearml_utils import maybe_create_dataset
from src.run_config import update_config

REAL_MANIFEST = Path("data/manifest.csv")
SYNTH_MANIFEST = Path("outputs/synthetic/manifest.csv")
MINIMAL_N = 5
CONFIG_OUT = Path("config_minimal.yaml")
DATASET_PROJECT = "handwriting-hebrew-ocr"


def create_minimal_real(n: int) -> str:
    df = pd.read_csv(REAL_MANIFEST)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)
    if len(labeled) < n:
        raise ValueError(f"Only {len(labeled)} labeled rows in {REAL_MANIFEST}; need {n}")

    # One crop per unique page so half-page split produces non-empty train+val sets
    unique_pages = labeled["page_path"].unique()
    if len(unique_pages) < n:
        subset = labeled.head(n).copy()
    else:
        subset = pd.concat(
            [labeled[labeled["page_path"] == p].head(1) for p in unique_pages[:n]],
            ignore_index=True,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        crops_dir = tmp / "crops"
        pages_dir = tmp / "pages"
        crops_dir.mkdir()
        pages_dir.mkdir()

        for _, row in subset.iterrows():
            crop_src = Path(row["crop_path"])
            page_src = Path(row["page_path"])
            if not crop_src.exists():
                raise FileNotFoundError(f"Crop not found: {crop_src}")
            if not page_src.exists():
                raise FileNotFoundError(f"Page not found: {page_src}")
            shutil.copy(crop_src, crops_dir / crop_src.name)
            shutil.copy(page_src, pages_dir / page_src.name)

        subset.to_csv(tmp / "manifest.csv", index=False)

        return maybe_create_dataset(
            project=DATASET_PROJECT,
            dataset_name="minimal_real",
            folders=[(crops_dir, "crops"), (pages_dir, "pages")],
            files=[tmp / "manifest.csv"],
        )


def create_minimal_synth(n: int) -> str:
    df = pd.read_csv(SYNTH_MANIFEST)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)
    if len(labeled) < n:
        raise ValueError(f"Only {len(labeled)} labeled rows in {SYNTH_MANIFEST}; need {n}")

    subset = labeled.head(n).copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        crops_dir = tmp / "crops"
        crops_dir.mkdir()

        for _, row in subset.iterrows():
            crop_src = Path(row["crop_path"])
            if not crop_src.exists():
                # manifest may store relative paths like "outputs/synthetic/crops/syn_000001.png"
                crop_src = Path("outputs/synthetic") / Path(row["crop_path"]).name
            if not crop_src.exists():
                raise FileNotFoundError(f"Synthetic crop not found: {row['crop_path']}")
            shutil.copy(crop_src, crops_dir / crop_src.name)

        subset.to_csv(tmp / "manifest.csv", index=False)

        return maybe_create_dataset(
            project=DATASET_PROJECT,
            dataset_name="minimal_synth",
            folders=[(crops_dir, "crops")],
            files=[tmp / "manifest.csv"],
        )


def main() -> int:
    print(f"Creating minimal_real dataset ({MINIMAL_N} crops)…")
    real_id = create_minimal_real(MINIMAL_N)
    print(f"  dataset_id: {real_id}")

    print(f"Creating minimal_synth dataset ({MINIMAL_N} crops)…")
    synth_id = create_minimal_synth(MINIMAL_N)
    print(f"  dataset_id: {synth_id}")

    update_config(
        path=CONFIG_OUT,
        **{
            "datasets.real_id": real_id,
            "datasets.synthetic_id": synth_id,
            "hyperparams.min_labeled": MINIMAL_N,
            "hyperparams.epochs": 3,
            "hyperparams.aug_copies": 1,
            "hyperparams.batch_size": 4,
            "hyperparams.patience": 0,
        },
    )
    print(f"Config written: {CONFIG_OUT}")
    print()
    print("To test:")
    print(f"  CONFIG_PATH={CONFIG_OUT} uv run python -m src.train_ctc")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
