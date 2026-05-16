"""generate_synthetic.py — render Hebrew text crops via TRDG, write manifest.csv, log to ClearML."""

import argparse
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from clearml import Task  # noqa: F401  # module-level for test patchability

from src.clearml_utils import init_task, upload_file_artifact  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Hebrew Unicode block: U+05D0 (א) through U+05EA (ת)
_HEBREW_START = 0x05D0
_HEBREW_END = 0x05EA

# Font lazy-download URLs (RESEARCH.md Pattern 5 — OFL licensed Hebrew fonts)
FONT_URLS: dict[str, str] = {
    "GveretLevin-Regular.ttf": (
        "https://raw.githubusercontent.com/AlefAlefAlef/gveret-levin"
        "/master/fonts/ttf/GveretLevin-Regular.ttf"
    ),
    "FrankRuhlLibre-Regular.ttf": (
        "https://fonts.gstatic.com/s/frankruhllibre/v23"
        "/j8_96_fAw7jrcalD7oKYNX0QfAnPcbzNEEB7OoicBw7FYVqQ.ttf"
    ),
    "FrankRuhlLibre-Bold.ttf": (
        "https://fonts.gstatic.com/s/frankruhllibre/v23"
        "/j8_96_fAw7jrcalD7oKYNX0QfAnPcbzNEEB7OoicBw4iZlqQ.ttf"
    ),
}

# Lazy TRDG import — populated on first call to render_crops.
# Module-level name allows test patching via patch("src.generate_synthetic._GeneratorFromStrings").
# The ACTUAL import happens inside render_crops (never at module level) to avoid the
# transitive `wikipedia` import from trdg/generators/__init__.py (RESEARCH.md Pattern 1).
_GeneratorFromStrings: Any = None


def _contains_hebrew(word: str) -> bool:
    """Return True if word contains at least one Hebrew character (U+05D0–U+05EA)."""
    return any(_HEBREW_START <= ord(ch) <= _HEBREW_END for ch in word)


# ---------------------------------------------------------------------------
# Pure functions — tested in tests/test_generate_synthetic.py
# ---------------------------------------------------------------------------


def build_word_corpus(
    labels: list[str],
    extra_words: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Build Hebrew word list and inverse-frequency sampling weights from labels.

    NFC-normalizes all input, splits on whitespace, keeps only words containing
    at least one Hebrew character (U+05D0–U+05EA), merges optional extra_words,
    deduplicates, then computes per-word score as sum of inverse character
    frequencies (Assumption A1 from RESEARCH.md).

    Args:
        labels: Existing crop labels (plain text strings).
        extra_words: Optional additional words to merge into the pool (SYN-03).

    Returns:
        Tuple of (words, weights) where weights sums to 1.0 and words containing
        rarer characters have higher sampling probability (D-06).

    Raises:
        ValueError: If no Hebrew words remain after filtering.
    """
    normalized_labels = [unicodedata.normalize("NFC", lbl) for lbl in labels]

    # Extract and filter words (D-04: whitespace split; Pitfall 5: Hebrew filter)
    seen: dict[str, None] = {}  # ordered dedup via dict keys
    for lbl in normalized_labels:
        for w in lbl.split():
            if w and _contains_hebrew(w):
                seen[w] = None

    # Merge extra words (NFC-normalized; Hebrew filter applied)
    if extra_words:
        for w in extra_words:
            nw = unicodedata.normalize("NFC", w)
            if nw and _contains_hebrew(nw):
                seen[nw] = None

    words = list(seen.keys())
    if not words:
        raise ValueError("build_word_corpus: no labeled Hebrew words found in corpus")

    # Character frequency across all normalized labels (not words — use all label text)
    char_freq: Counter[str] = Counter()
    for lbl in normalized_labels:
        char_freq.update(lbl)

    # Word score = sum of inverse char frequencies (RESEARCH.md Pattern 3 / Assumption A1)
    total_chars = sum(char_freq.values())
    word_scores = np.array(
        [
            sum(1.0 / (char_freq.get(c, 1) / total_chars + 1e-8) for c in word)
            for word in words
        ],
        dtype=np.float64,
    )
    weights = word_scores / word_scores.sum()
    return words, weights


def sample_text(
    words: list[str],
    weights: np.ndarray,
    target_chars: int,
    rng: np.random.Generator,
) -> str:
    """Sample words until approximately target_chars characters are accumulated.

    Always selects at least one word (guards RESEARCH.md Pitfall 4 narrow-image risk).

    Args:
        words: Pool of Hebrew words to sample from.
        weights: Probability weights matching words length, summing to 1.0.
        target_chars: Minimum total character count to reach.
        rng: NumPy random Generator for reproducible sampling.

    Returns:
        Space-joined string of sampled words.
    """
    selected: list[str] = []
    total = 0
    while total < target_chars:
        word = str(rng.choice(words, p=weights))
        selected.append(word)
        # Track as joined length: first word has no leading space
        total = sum(len(w) for w in selected) + len(selected) - 1
    return " ".join(selected)


def build_char_count_distribution(labels: list[str]) -> np.ndarray:
    """Return array of NFC character counts for each label.

    Used to sample empirical text-length distribution (D-05).

    Args:
        labels: Existing crop labels.

    Returns:
        1-D integer array with one element per label.
    """
    return np.array([len(unicodedata.normalize("NFC", lbl)) for lbl in labels])


def check_coverage(labels: list[str], min_char_count: int) -> dict[str, int]:
    """Return characters below min_char_count threshold across all labels.

    NFC-normalizes every label before counting. Pure function — no side effects,
    no sys.exit (caller owns exit-code decision per 06-PATTERNS.md).

    Args:
        labels: Label strings to analyze.
        min_char_count: Minimum acceptable character occurrence count.

    Returns:
        Dict mapping under-represented characters to their current counts.
        Empty dict when all characters meet the threshold.
    """
    char_counts: Counter[str] = Counter()
    for lbl in labels:
        char_counts.update(unicodedata.normalize("NFC", lbl))
    return {ch: cnt for ch, cnt in char_counts.items() if cnt < min_char_count}


# ---------------------------------------------------------------------------
# Font management
# ---------------------------------------------------------------------------


def ensure_fonts(fonts_dir: Path) -> list[str]:
    """Download missing fonts to fonts_dir; return list of .ttf paths.

    If fonts_dir already contains any .ttf files (user-supplied override via
    --fonts_dir, D-03, or previously downloaded defaults), those files are returned
    as-is without any network request. Only when the directory is empty of .ttf files
    are the defaults downloaded from FONT_URLS.

    Args:
        fonts_dir: Directory to store/find .ttf font files.

    Returns:
        List of absolute paths to .ttf files (strings).
    """
    import requests  # local import — only needed on first run with no cached fonts

    fonts_dir.mkdir(parents=True, exist_ok=True)

    # If any .ttf already present, use those (cached or user-supplied override)
    existing_ttfs = sorted(fonts_dir.glob("*.ttf"))
    if existing_ttfs:
        return [str(p) for p in existing_ttfs]

    # No fonts found — download defaults from FONT_URLS
    paths: list[str] = []
    for name, url in FONT_URLS.items():
        dest = fonts_dir / name
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        paths.append(str(dest))
    return paths


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_crops(
    texts: list[str],
    font_paths: list[str],
    out_crops_dir: Path,
) -> list[tuple[str, str]]:
    """Render each text as a grayscale TRDG crop; save PNGs to out_crops_dir.

    Skips None images returned by TRDG when text/background contrast is too low
    (RESEARCH.md Pitfall 3). The caller must supply enough texts (len(texts) >= count
    + estimated None rate) to reach the desired crop count.

    The TRDG import is deferred inside this function to avoid the transitive
    `wikipedia` module-level import (RESEARCH.md Pattern 1 / Anti-Pattern 1).

    Args:
        texts: Pre-sampled text strings to render (NFC-normalized).
        font_paths: Absolute paths to .ttf font files.
        out_crops_dir: Directory to write PNG files.

    Returns:
        List of (crop_path_str, original_text) for each successfully saved crop.
    """
    global _GeneratorFromStrings
    if _GeneratorFromStrings is None:
        from trdg.generators.from_strings import GeneratorFromStrings

        _GeneratorFromStrings = GeneratorFromStrings

    out_crops_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[str, str]] = []
    idx = 0
    for text in texts:
        gen = _GeneratorFromStrings(
            strings=[text],
            count=1,
            fonts=font_paths,
            size=64,
            skewing_angle=3,
            random_skew=True,
            blur=1,
            random_blur=True,
            background_type=1,
            distorsion_type=1,
            distorsion_orientation=2,
            image_mode="L",
            margins=(0, 0, 0, 0),
            rtl=True,
            fit=True,
        )
        img, _ = next(gen)
        if img is None:
            # RESEARCH.md Pitfall 3: skip rather than save
            continue
        idx += 1
        crop_name = f"syn_{idx:06d}.png"
        crop_path = out_crops_dir / crop_name
        img.save(str(crop_path))
        saved.append((str(crop_path), text))
    return saved


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def write_manifest(rows: list[tuple[str, str]], output_dir: Path) -> Path:
    """Write a 3-column manifest CSV for generated synthetic crops.

    Columns are exactly ["crop_path", "label", "status"] with status="labeled"
    so CropDataset's status==labeled filter passes (06-PATTERNS.md).

    Args:
        rows: List of (crop_path, label) tuples from render_crops.
        output_dir: Parent directory for manifest.csv.

    Returns:
        Path to the written manifest.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(parents=True, exist_ok=True)
    manifest_df = pd.DataFrame(
        [{"crop_path": path, "label": label, "status": "labeled"} for path, label in rows],
        columns=["crop_path", "label", "status"],
    )
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_path


# ---------------------------------------------------------------------------
# CLI entry point (main() implemented in Plan 03)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render synthetic Hebrew crops via TRDG; write manifest.csv; log to ClearML."
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Existing labeled manifest to extract corpus from",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/synthetic"),
        help="Directory for output crops/ and manifest.csv",
    )
    p.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of synthetic crops to generate (SYN-01)",
    )
    p.add_argument(
        "--fonts_dir",
        type=Path,
        default=Path("assets/fonts"),
        help="Directory containing .ttf fonts; downloads defaults on first use (D-03)",
    )
    p.add_argument(
        "--wordlist",
        type=Path,
        default=None,
        help="Optional extra word list (one word per line) merged into corpus (SYN-03)",
    )
    p.add_argument(
        "--min_char_count",
        type=int,
        default=5,
        help="Exit non-zero if any char appears fewer times after generation (SYN-04)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible corpus sampling",
    )
    return p


def main() -> int:
    """CLI entry point — rendering loop implemented in Plan 03."""
    task = init_task("handwriting-hebrew-ocr", "generate_synthetic")
    args = _build_parser().parse_args()
    task.connect(vars(args), name="hyperparams")

    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)
    if labeled.empty:
        print("ERROR: no labeled rows in manifest — cannot build corpus.", file=sys.stderr)
        return 3

    print("INFO: Plan 03 rendering loop not yet implemented.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
