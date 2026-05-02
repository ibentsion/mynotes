import argparse
import base64
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from openai import OpenAI

from src.manifest_schema import MANIFEST_COLUMNS

PROMPT = (
    "Transcribe the Hebrew handwriting in this image exactly as written. "
    "Return only the transcribed text, nothing else."
)
STRING_DTYPES: dict[str, type] = {"label": object, "notes": object, "flag_reasons": object}


def write_manifest_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".csv.tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df[MANIFEST_COLUMNS].to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _encode_png_b64(crop_path: Path) -> str:
    return base64.b64encode(crop_path.read_bytes()).decode("ascii")


def _label_one(client: OpenAI, model: str, crop_path: Path) -> str:
    b64 = _encode_png_b64(crop_path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                }},
            ],
        }],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty response from model")
    return text


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label unlabeled crops via OpenAI vision")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument(
        "--model", default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o"]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def _collect_unlabeled(df: pd.DataFrame, limit: int | None) -> list:
    indices = df.index[df["status"] == "unlabeled"].tolist()
    return indices[:limit] if limit is not None else indices


def _resolve_crop(crop_path: str) -> Path:
    return Path(crop_path)


def main() -> int:
    """Batch-label unlabeled crops in manifest.csv using OpenAI vision API.

    Returns:
        Exit code: 0 on success, 2 on usage error, 130 on keyboard interrupt.
    """
    try:
        return _run()
    except KeyboardInterrupt:
        print("Interrupted — progress saved.", file=sys.stderr)
        return 130


def _run() -> int:
    args = _parse_args()
    df = pd.read_csv(args.manifest, dtype=STRING_DTYPES)  # type: ignore[arg-type]

    if list(df.columns) != MANIFEST_COLUMNS:
        print(f"error: unexpected manifest columns: {list(df.columns)}", file=sys.stderr)
        return 2

    indices = _collect_unlabeled(df, args.limit)
    total = len(indices)
    print(f"Found {total} unlabeled crops")

    if args.dry_run:
        print("DRY RUN — no API calls made")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set in environment", file=sys.stderr)
        return 2

    client = OpenAI()
    success = 0
    failures = 0

    for i, idx in enumerate(indices, start=1):
        crop_path = _resolve_crop(str(df.at[idx, "crop_path"]))
        print(f"[{i}/{total}] labeling {crop_path}", flush=True)
        try:
            text = _label_one(client, args.model, crop_path)
            df.at[idx, "label"] = text
            df.at[idx, "status"] = "labeled"
            if pd.isna(df.at[idx, "notes"]) or str(df.at[idx, "notes"]).strip() == "":
                df.at[idx, "notes"] = f"auto:{args.model}"
            write_manifest_atomic(args.manifest, df)
            success += 1
        except Exception as exc:
            print(f"[WARN] {crop_path}: {exc}", file=sys.stderr)
            failures += 1

    print(f"Done. Labeled {success} / {total}, failed {failures}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
