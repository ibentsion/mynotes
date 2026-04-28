"""backfill_page_paths.py — patch page_path column in manifests generated before it was added.

Strategy: prepare_data.py processes PDFs in sorted filename order and pdf2image writes page
files sequentially, so sorting UUID groups by their earliest mtime reconstructs the original
PDF → pages mapping. Page-count validation confirms each match.

Run:
    uv run backfill-page-paths --manifest data/manifest.csv --dry-run
    uv run backfill-page-paths --manifest data/manifest.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Entry points run without the project root in sys.path; re-insert it so src.* imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.manifest_schema import MANIFEST_COLUMNS  # noqa: E402

_PAGE_FILE_RE = re.compile(r"^(.+)-(\d+)\.(png|jpe?g|tiff?)$", re.IGNORECASE)


def _group_pages_by_uuid(pages_dir: Path) -> dict[str, dict[int, Path]]:
    """Return {uuid_prefix: {page_num: path}} for all page images in pages_dir."""
    groups: dict[str, dict[int, Path]] = {}
    for f in pages_dir.iterdir():
        m = _PAGE_FILE_RE.match(f.name)
        if not m:
            continue
        prefix, page_num = m.group(1), int(m.group(2))
        groups.setdefault(prefix, {})[page_num] = f
    return groups


def _match_pdfs_to_uuid_groups(
    pdf_page_counts: dict[str, int],
    uuid_groups: dict[str, dict[int, Path]],
) -> tuple[dict[str, str], list[str]]:
    """Match each PDF to a UUID group using processing order + page-count validation.

    prepare_data.py uses sorted(pdf_dir.glob("*.pdf")), so sort both sides the same way
    and match positionally. Returns (pdf_path → uuid_prefix, warning_messages).
    """
    sorted_pdfs = sorted(pdf_page_counts, key=lambda p: Path(p).name)

    def _earliest_mtime(pages: dict[int, Path]) -> float:
        return min(p.stat().st_mtime for p in pages.values())

    sorted_uuids = sorted(uuid_groups, key=lambda u: _earliest_mtime(uuid_groups[u]))

    warnings: list[str] = []
    if len(sorted_pdfs) != len(sorted_uuids):
        warnings.append(
            f"PDF count ({len(sorted_pdfs)}) ≠ UUID group count ({len(sorted_uuids)}). "
            "Extra groups may be from previous runs — matches after position "
            f"{min(len(sorted_pdfs), len(sorted_uuids))} are skipped."
        )

    mapping: dict[str, str] = {}
    for i, pdf_path in enumerate(sorted_pdfs):
        if i >= len(sorted_uuids):
            warnings.append(f"  no UUID group for {Path(pdf_path).name}")
            continue
        uuid = sorted_uuids[i]
        expected = pdf_page_counts[pdf_path]
        actual = len(uuid_groups[uuid])
        status = "✓" if expected == actual else "✗"
        warnings.append(
            f"  {status} {Path(pdf_path).name} ({expected}p) → {uuid} ({actual}p)"
            + ("" if expected == actual else "  ← PAGE COUNT MISMATCH")
        )
        mapping[pdf_path] = uuid

    return mapping, warnings


def backfill(manifest_path: Path) -> tuple[pd.DataFrame, list[str]]:
    _STR_COLS = ("label", "notes", "flag_reasons")
    df = pd.read_csv(manifest_path, dtype={c: object for c in _STR_COLS})

    # Insert page_path column at the right position if missing
    if "page_path" not in df.columns:
        insert_at = list(df.columns).index("page_num")
        df.insert(insert_at, "page_path", "")

    missing_mask = df["page_path"].isna() | (df["page_path"].astype(str) == "")
    if not missing_mask.any():
        return df, ["All rows already have page_path — nothing to do."]

    pages_dir = manifest_path.parent / "pages"
    if not pages_dir.exists():
        return df, [f"ERROR: pages directory not found: {pages_dir}"]

    uuid_groups = _group_pages_by_uuid(pages_dir)
    if not uuid_groups:
        return df, [f"ERROR: no page images found in {pages_dir}"]

    # Page counts only from rows that still need filling
    pdf_page_counts: dict[str, int] = (
        df[missing_mask].groupby("pdf_path")["page_num"].max().astype(int).to_dict()
    )

    messages: list[str] = [f"Matching {len(pdf_page_counts)} PDFs to {len(uuid_groups)} UUID groups:"]
    mapping, match_msgs = _match_pdfs_to_uuid_groups(pdf_page_counts, uuid_groups)
    messages.extend(match_msgs)

    # Apply mapping
    filled = 0
    unresolved: list[str] = []
    for pdf_path, uuid_prefix in mapping.items():
        pages_map = uuid_groups[uuid_prefix]
        pdf_rows = missing_mask & (df["pdf_path"] == pdf_path)
        for idx in df[pdf_rows].index:
            page_num = int(df.at[idx, "page_num"])
            if page_num in pages_map:
                df.at[idx, "page_path"] = str(pages_map[page_num])
                filled += 1
            else:
                unresolved.append(
                    f"{df.at[idx, 'crop_path']}: page {page_num} missing from UUID group {uuid_prefix}"
                )

    for crop in df[missing_mask & (df["page_path"].isna() | (df["page_path"].astype(str) == ""))]["crop_path"]:
        unresolved.append(f"{crop}: PDF had no matching UUID group")

    messages.append(f"\nFilled {filled} / {int(missing_mask.sum())} missing entries.")
    if unresolved:
        messages.append(f"{len(unresolved)} unresolved:")
        messages.extend(f"  {u}" for u in unresolved)

    return df[MANIFEST_COLUMNS], messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill page_path column in manifests generated before it was added."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: {args.manifest} not found", file=sys.stderr)
        return 2

    df, messages = backfill(args.manifest)
    print("\n".join(messages))

    if not args.dry_run:
        df.to_csv(args.manifest, index=False)
        print(f"Written: {args.manifest}")
    else:
        print("(dry-run — manifest not modified)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
