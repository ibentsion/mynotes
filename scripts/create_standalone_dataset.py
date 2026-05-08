"""Create a standalone ClearML v3 dataset (no parent) and enqueue GPU training."""

import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from clearml import Dataset, Task

MANIFEST_CACHE = Path(
    "~/.clearml/cache/storage_manager/datasets"
    "/ds_02210b47a0534666b0462a175bf2af9d/manifest.csv"
).expanduser()
CROPS_DIR = Path("data/crops")
PAGES_DIR = Path("data/pages")
DATASET_PROJECT = "handwriting-hebrew-ocr"
DATASET_NAME = "data_prep"
QUEUE_NAME = "ofek"
POLL_INTERVAL_S = 30
POLL_TIMEOUT_S = 7200
TERMINAL_STATES = {"completed", "failed", "stopped", "closed"}


def build_dataset() -> str:
    if not MANIFEST_CACHE.is_file():
        raise FileNotFoundError(f"Cached manifest not found: {MANIFEST_CACHE}")
    if not CROPS_DIR.is_dir() or not any(CROPS_DIR.glob("*.png")):
        raise FileNotFoundError(f"Crops dir missing or empty: {CROPS_DIR}")
    if not PAGES_DIR.is_dir() or not any(PAGES_DIR.glob("*.png")):
        raise FileNotFoundError(f"Pages dir missing or empty: {PAGES_DIR}")

    use_current_task = Task.current_task() is not None
    ds = Dataset.create(
        dataset_name=DATASET_NAME,
        dataset_project=DATASET_PROJECT,
        use_current_task=use_current_task,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(MANIFEST_CACHE, Path(tmpdir) / "manifest.csv")
        ds.add_files(str(Path(tmpdir) / "manifest.csv"))
        ds.add_files(path=str(CROPS_DIR), dataset_path="crops")
        ds.add_files(path=str(PAGES_DIR), dataset_path="pages")
        ds.upload()

    ds.finalize()
    print(f"Dataset ID: {ds.id}")
    return ds.id


def verify_roundtrip(dataset_id: str) -> None:
    import csv

    root = Path(Dataset.get(dataset_id=dataset_id).get_local_copy())
    manifest = root / "manifest.csv"
    assert manifest.is_file(), f"manifest.csv missing at {manifest}"

    crops = list((root / "crops").glob("*.png"))
    assert crops, f"No PNGs found under {root / 'crops'}"

    pages = list((root / "pages").glob("*.png"))
    assert pages, f"No PNGs found under {root / 'pages'}"

    # Verify every page referenced in the manifest resolves to a real file
    missing_pages: list[str] = []
    with open(manifest) as f:
        for row in csv.DictReader(f):
            page_name = Path(row["page_path"]).name
            if not (root / "pages" / page_name).is_file():
                missing_pages.append(page_name)
    assert not missing_pages, (
        f"{len(missing_pages)} manifest pages missing in dataset: {missing_pages[:3]}"
    )

    manifest_rows = sum(1 for _ in open(manifest)) - 1  # noqa: SIM115
    print(f"Roundtrip OK — {manifest_rows} manifest rows, {len(crops)} crops, {len(pages)} pages")


def enqueue_training(dataset_id: str) -> str:
    result = subprocess.run(
        [
            "uv", "run", "python", "-m", "src.train_ctc",
            "--enqueue", "--queue_name", QUEUE_NAME,
            "--dataset_id", dataset_id,
        ],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    print(combined)

    match = re.search(r"task[_ ]id[=:\s]+([0-9a-f]{32})", combined, re.IGNORECASE)
    if match:
        task_id = match.group(1)
    else:
        tasks = Task.get_tasks(project_name=DATASET_PROJECT, task_name="train_baseline_ctc")
        if not tasks:
            raise RuntimeError("Could not find training task after enqueue")
        task_id = sorted(tasks, key=lambda t: t.data.created)[-1].id

    print(f"Training task ID: {task_id}")
    return task_id


def poll_until_done(task_id: str) -> str:
    task = Task.get_task(task_id=task_id)
    last_status = None
    start = time.monotonic()

    while True:
        elapsed = int(time.monotonic() - start)
        status = task.get_status()

        if status != last_status:
            print(f"[{elapsed}s] status={status}")
            last_status = status

        if status in TERMINAL_STATES:
            return status

        if elapsed >= POLL_TIMEOUT_S:
            print(f"WARNING: poll timeout after {POLL_TIMEOUT_S}s — last status={status}")
            raise SystemExit(1)

        time.sleep(POLL_INTERVAL_S)


def main() -> None:
    dataset_id = build_dataset()
    verify_roundtrip(dataset_id)

    subprocess.run(["git", "push"], check=True)

    task_id = enqueue_training(dataset_id)
    final_status = poll_until_done(task_id)

    if final_status in {"failed", "stopped"}:
        task = Task.get_task(task_id=task_id)
        lines = task.get_reported_console_output(number_of_reports=50)
        print("\n--- Last console output ---")
        for line in lines:
            print(line)
        raise SystemExit(1)

    print(f"Training {final_status}. Dataset: {dataset_id}, Task: {task_id}")


if __name__ == "__main__":
    main()
