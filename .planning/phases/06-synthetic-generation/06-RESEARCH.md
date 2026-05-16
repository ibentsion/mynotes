# Phase 6: Synthetic Generation - Research

**Researched:** 2026-05-16
**Domain:** Synthetic Hebrew OCR image generation (TRDG + PIL + BiDi)
**Confidence:** HIGH (core API verified from wheel source; dependency conflict verified live)

---

## Summary

This phase adds a CLI tool (`src/generate_synthetic.py`) that renders Hebrew text as crop images using TRDG, assembles a word corpus from existing labeled crops with inverse-frequency weighting for rare-character coverage, writes a `manifest.csv` compatible with `CropDataset`, and logs to ClearML.

The primary technical risk is the **TRDG dependency conflict**: trdg 1.8.0's pinned `arabic-reshaper==2.1.3` has bad metadata (yanked; pip 24.1+ rejects it), causing uv to fall back to trdg 1.7.0 which pins `numpy<1.17` — catastrophically incompatible with the project's numpy 2.4.4. The fix is a single `[tool.uv] override-dependencies` entry in `pyproject.toml`.

Hebrew rendering in TRDG requires `rtl=True` and a custom font list (`fonts=` parameter). TRDG ships no Hebrew fonts; fonts must be downloaded lazily on first use. Two verified OFL-licensed fonts are available with direct stable URLs (Gveret Levin, Frank Ruhl Libre).

The inverse-frequency weighting is pure Python using `collections.Counter` and `numpy.random.choice` with a `p=` probability array — no external library needed.

**Primary recommendation:** Use `GeneratorFromStrings` imported directly from `trdg.generators.from_strings` (not `trdg.generators`) to avoid the transitive `wikipedia` module-level import. Use `image_mode="L"` for grayscale output, `background_type=1` for plain white, `rtl=True` for Hebrew BiDi reordering.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Use handwriting-like Hebrew fonts with TRDG distortions (skew, blur, noise). Aim for visual similarity to real scanned handwriting, not printed text.
- **D-02:** Plain white background. `background_type=1` in TRDG.
- **D-03:** Bundle a list of recommended Hebrew handwriting font names; download them on first use (lazy download into `assets/fonts/` or a cache dir). `--fonts_dir` overrides the downloaded defaults. No font bundled directly in the repo.
- **D-04:** Word-level corpus sampling. Extract individual words from existing labeled crops (split on whitespace after NFC normalization). Sample 1–N words per synthetic crop.
- **D-05:** Match the real labeled crop character-count distribution when determining text length per crop. Read character counts from the existing manifest's `label` column, sample from that empirical distribution to pick how many characters each synthetic crop will contain.
- **D-06:** Inverse-frequency weighting on words. Compute character frequency across all existing labels. Words containing rarer characters get proportionally higher sampling probability.
- **D-07:** Log generation runs to ClearML — initialize a task, connect CLI args, upload the output `manifest.csv` as an artifact.
- **D-08:** ClearML task name: `generate_synthetic`.

### Claude's Discretion

- Exact TRDG distortion parameter values (skew angle, blur radius, noise level) — pick conservative defaults that produce legible but imperfect text
- Font download mechanism (requests/urllib, target URL, cache location)
- Whether to also log a per-character count scalar to ClearML alongside the manifest artifact
- Exact word-splitting strategy for extracting corpus words (whitespace split + filter single chars, or include single-char words)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SYN-01 | `generate_synthetic` CLI with `--count`, `--output_dir`, `--fonts_dir`, `--wordlist`; writes `manifest.csv` with `crop_path`, `label`, `status="labeled"` | TRDG `GeneratorFromStrings` + argparse pattern from `train_ctc.py` |
| SYN-02 | Synthetic crops: grayscale, 64px height, variable width — readable by `CropDataset` without changes | `image_mode="L"`, `size=64`, TRDG output is PIL Image; save as PNG; `load_crop` resizes to 64px |
| SYN-03 | Text corpus from existing labeled crops + optional `--wordlist`; rare chars get higher rendering weight | `Counter`-based char frequency + `numpy.random.choice(p=weights)` |
| SYN-04 | Coverage validation: report chars below `--min_char_count` before and after; exit non-zero if gaps remain | Simple `Counter` on all labels in generated manifest; compare before/after |
</phase_requirements>

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Image rendering | Script (CPU) | — | TRDG is pure PIL/CV2, no network needed at render time |
| Text corpus assembly | Script (CPU) | Filesystem | Read existing manifest CSV, split labels |
| Rare-char weighting | Script (CPU) | — | Pure Python Counter + numpy probability array |
| Font management | Filesystem (`assets/fonts/`) | Network (lazy download) | Fonts live on disk after first use |
| Coverage validation | Script (CPU) | — | Counter comparison, no external service |
| ClearML logging | ClearML | — | Same pattern as `train_ctc.py` |
| Manifest writing | Filesystem (`--output_dir`) | — | `pd.DataFrame.to_csv()` |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| trdg | 1.8.0 | Synthetic text image rendering | The decision (D-01); provides `GeneratorFromStrings` with RTL support |
| arabic-reshaper | 3.0.1 | BiDi text reshaping (transitive trdg dep) | Required by trdg; 3.0.1 fixes the bad-metadata issue of 2.1.3 |
| python-bidi | 0.6.10 | Unicode BiDi algorithm for Hebrew RTL rendering | Required by trdg for Hebrew visual reordering |
| Pillow | 12.2.0 | Image I/O (already installed) | TRDG returns PIL Image objects |
| requests | 2.33.1 | Font lazy-download (already installed) | Simple, already in project deps |

[VERIFIED: PyPI JSON API, wheel inspection] Versions checked 2026-05-16.

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| collections.Counter | stdlib | Character frequency counting | Corpus assembly + coverage validation |
| unicodedata | stdlib | NFC normalization (already used in ctc_utils.py) | Normalize labels before word splitting |
| numpy | 2.4.4 | `np.random.choice(p=weights)` for weighted sampling | Already installed |
| pathlib | stdlib | Path handling | Consistent with existing code style |
| tqdm | stdlib-ish | Progress bar during generation | Optional, TRDG already uses it |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TRDG | Hand-rolled PIL rendering | TRDG encapsulates font rendering, distortions, background. Building equivalent is weeks of work. |
| Lazy font download | Bundle fonts in repo | Repo size grows ~200KB per font; user's D-03 explicitly chose lazy download. |
| `np.random.choice(p=weights)` | `random.choices(k=1, weights=weights)` | Either works; numpy is already imported; both are O(N) |

**Installation (pyproject.toml changes required):**

```toml
# In [project] dependencies: add trdg==1.8.0
# In [tool.uv]: add override-dependencies to fix transitive dep conflict

[project]
dependencies = [
    # ... existing deps ...
    "trdg==1.8.0",
    "arabic-reshaper==3.0.1",   # explicit pin — trdg requires ==2.1.3 which is yanked
    "python-bidi==0.6.10",      # trdg requires ==0.4.2; this has compat import path
]

[tool.uv]
override-dependencies = [
    "arabic-reshaper==3.0.1",   # override trdg's ==2.1.3 pin (bad metadata, yanked)
    "python-bidi==0.6.10",      # override trdg's ==0.4.2 pin
]
```

Then: `uv sync` — no numpy downgrade, no opencv conflict (opencv-python-headless satisfies cv2).

[VERIFIED: uv pip install --dry-run shows 1.7.0+numpy downgrade without override; uv docs confirm override-dependencies syntax]

---

## Architecture Patterns

### System Architecture Diagram

```
Existing manifest.csv ──► corpus_builder()
       │                        │
       │                   [extract words]
       │                   [NFC normalize]
       │                   [char freq Counter]
       │                   [compute word weights]
       │                        │
       │                   word_pool: list[str]
       │                   weights: np.ndarray
       │
Optional --wordlist ──────────► merge into word_pool
       │
       │
       ├── sample_text(weights) ──► text_string (1-N words, length ~ empirical dist)
       │           │
       │     GeneratorFromStrings(
       │       strings=[text_string],
       │       fonts=[path1, path2, ...],  ← from assets/fonts/ or --fonts_dir
       │       rtl=True,
       │       image_mode="L",
       │       background_type=1,          ← plain white
       │       skewing_angle=3,
       │       random_skew=True,
       │       blur=1,
       │       random_blur=True,
       │       distorsion_type=1,          ← sin wave
       │       size=64,
       │       margins=(0,0,0,0),
       │     )
       │           │
       │     PIL Image (mode="L", h≈64px, variable W)
       │           │
       │     save as PNG in --output_dir/crops/
       │           │
       │     manifest row: {crop_path, label, status="labeled"}
       │
       ▼
coverage_validator()
    [count chars in manifest labels]
    [compare before vs after generation]
    [print chars below --min_char_count]
    [exit(1) if gaps remain]
       │
       ▼
ClearML: init_task("generate_synthetic")
         task.connect(vars(args))
         upload_file_artifact("manifest", manifest_path)
         [optional: log per-char count scalar]
```

### Recommended Project Structure

```
src/
├── generate_synthetic.py   # new CLI entry point
assets/
└── fonts/                  # created on first run (gitignored contents)
    ├── GveretLevin-Regular.ttf
    └── FrankRuhlLibre-Regular.ttf
tests/
└── test_generate_synthetic.py  # new test file
```

The `assets/fonts/` directory contents are gitignored (font files, OFL license). The font name list (`FONT_URLS` dict) is hardcoded in the script.

### Pattern 1: TRDG GeneratorFromStrings — Direct Import

Import `GeneratorFromStrings` directly from the submodule, bypassing `trdg/generators/__init__.py` which triggers a module-level `import wikipedia` via `from_wikipedia.py`.

```python
# Source: inspected trdg-1.8.0-py3-none-any.whl
# CORRECT — avoids transitive wikipedia import
from trdg.generators.from_strings import GeneratorFromStrings

# WRONG — triggers wikipedia import at module level
from trdg.generators import GeneratorFromStrings
```

### Pattern 2: GeneratorFromStrings Configuration for Hebrew

```python
# Source: trdg 1.8.0 from_strings.py constructor (inspected from wheel)
from trdg.generators.from_strings import GeneratorFromStrings

gen = GeneratorFromStrings(
    strings=["שלום"],          # list of text strings to render
    count=1,                    # one image per call to next()
    fonts=["/path/to/font.ttf"], # explicit font paths (no 'he' lang dir in TRDG)
    language="en",              # fallback; fonts= overrides lang font loading
    size=64,                    # total image height in pixels (including margins)
    skewing_angle=3,            # max skew degrees (D-01: conservative)
    random_skew=True,           # randomize within [-angle, +angle]
    blur=1,                     # blur radius (D-01: conservative)
    random_blur=True,           # randomize blur
    background_type=1,          # 1 = plain white (D-02)
    distorsion_type=1,          # 1 = sine distortion (handwriting-like)
    distorsion_orientation=2,   # 2 = both vertical+horizontal
    image_mode="L",             # grayscale output (SYN-02)
    margins=(0, 0, 0, 0),       # no margin → full 64px height
    rtl=True,                   # Hebrew BiDi visual reordering via python-bidi
    fit=True,                   # crop to text bounding box (removes whitespace)
)

img, label = next(gen)
# img: PIL Image, mode="L", height≈64px, width varies with text length
# label: the original string (orig_strings preserved when rtl=True)
```

**Critical:** When `rtl=True`, TRDG calls `arabic_reshaper.reshape()` then `bidi.algorithm.get_display()` on the input string. For Hebrew this is correct: arabic_reshaper is a no-op on Hebrew (no Arabic ligatures), and `get_display()` applies Unicode BiDi visual reordering so PIL draws the characters left-to-right visually as they appear right-to-left logically. The returned `label` is the **original** input string (not the reshaped/reordered one) — this is the intended behavior for RTL generators.

### Pattern 3: Inverse-Frequency Word Weighting

```python
# Source: [ASSUMED] standard Python pattern; verify in implementation
import unicodedata
from collections import Counter
import numpy as np

def build_word_corpus(
    labels: list[str],
    extra_words: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Build word list and inverse-frequency sampling weights from labels.

    Returns (words, weights) where weights sums to 1.0 and words containing
    rarer characters have higher probability (D-06).
    """
    # NFC normalize — matches ctc_utils.py build_charset() pattern
    normalized = [unicodedata.normalize("NFC", lbl) for lbl in labels]

    # Extract words (D-04: whitespace split)
    words: list[str] = []
    for lbl in normalized:
        words.extend(w for w in lbl.split() if w)  # filter empty strings
    if extra_words:
        words.extend(unicodedata.normalize("NFC", w) for w in extra_words if w)
    words = list(set(words))  # deduplicate

    # Character frequency across all labels
    char_freq: Counter[str] = Counter()
    for lbl in normalized:
        char_freq.update(lbl)

    # Word score = sum of inverse char frequencies
    # Rare chars → low frequency → high inverse → word gets high weight
    total_chars = sum(char_freq.values())
    word_scores = np.array([
        sum(1.0 / (char_freq.get(c, 1) / total_chars + 1e-8) for c in word)
        for word in words
    ], dtype=np.float64)
    weights = word_scores / word_scores.sum()

    return words, weights


def sample_text(
    words: list[str],
    weights: np.ndarray,
    target_chars: int,
    rng: np.random.Generator,
) -> str:
    """Sample words until approximately target_chars characters reached."""
    selected: list[str] = []
    total = 0
    while total < target_chars:
        word = rng.choice(words, p=weights)
        selected.append(word)
        total += len(word) + 1  # +1 for space
    return " ".join(selected)
```

### Pattern 4: Empirical Character-Count Distribution Sampling (D-05)

```python
# Source: [ASSUMED] standard numpy pattern
import numpy as np

def build_char_count_distribution(labels: list[str]) -> np.ndarray:
    """Return array of character counts from existing crop labels."""
    return np.array([len(unicodedata.normalize("NFC", lbl)) for lbl in labels])

# Usage during generation:
char_counts = build_char_count_distribution(existing_labels)
target_chars = int(rng.choice(char_counts))  # sample from empirical distribution
```

### Pattern 5: Font Lazy-Download

```python
# Source: [ASSUMED] standard requests pattern matching existing project style
import requests
from pathlib import Path

FONT_URLS: dict[str, str] = {
    # Verified URLs 2026-05-16
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

def ensure_fonts(fonts_dir: Path) -> list[str]:
    """Download missing fonts to fonts_dir. Returns list of .ttf paths."""
    fonts_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name, url in FONT_URLS.items():
        dest = fonts_dir / name
        if not dest.exists():
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        paths.append(str(dest))
    return paths
```

### Pattern 6: ClearML Integration (replicating train_ctc.py)

```python
# Source: src/clearml_utils.py + src/train_ctc.py (verified)
from clearml import Task  # noqa: F401 — required for @patch('src.generate_synthetic.Task')
from src.clearml_utils import init_task, upload_file_artifact

def main() -> None:
    task = init_task("Hebrew OCR", "generate_synthetic")
    # parse args AFTER init_task (ClearML may override from server)
    args = parse_args()
    task.connect(vars(args))
    # ... generate images ...
    upload_file_artifact(task, "manifest", manifest_path)
    # optional: log per-char counts
    logger = task.get_logger()
    for char, count in sorted(char_counts.items()):
        logger.report_scalar("char_coverage", char, iteration=0, value=count)
```

### Anti-Patterns to Avoid

- **`from trdg.generators import GeneratorFromStrings`:** Triggers module-level `import wikipedia` via `__init__.py` chain. Use `from trdg.generators.from_strings import GeneratorFromStrings` instead.
- **`count=-1` in GeneratorFromStrings:** Runs forever (infinite generator). Always pass explicit `count=1` and call `next()` in a loop.
- **Installing trdg without override-dependencies:** uv falls back to trdg 1.7.0 which pins `numpy<1.17` — catastrophic conflict with project's numpy 2.4.4.
- **Using `is_handwritten=True`:** TRDG's handwriting mode requires TensorFlow (not installed). Use font-based simulation with skew/blur/distortion instead (D-01).
- **Horizontal flip augmentation in corpus sampling:** Reverses Hebrew text direction. Consistent with existing `AugmentTransform` which explicitly avoids horizontal flip for RTL.
- **Setting `language="he"` in TRDG:** TRDG has no `he` language font directory (only `cn`, `ja`, `latin`). Must pass explicit `fonts=` list.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Text → image rendering | Custom PIL text drawing with RTL | `GeneratorFromStrings(rtl=True)` | TRDG handles font metrics, line height, distortions, BiDi reordering in ~5 params |
| BiDi text reordering | Custom character reordering | `python-bidi` (via TRDG `rtl=True`) | Unicode BiDi algorithm has 20+ rules; python-bidi is the reference implementation |
| Weighted sampling | Custom rejection sampling | `np.random.choice(p=weights)` | Single call, correct, O(N) |
| Image format conversion | Manual numpy array manipulation | `img.convert("L")` → `np.array(img)` | PIL conversion is correct and tested |

**Key insight:** The hard part of synthetic generation is not the rendering loop — it's getting Hebrew BiDi direction, font metrics, and image height right. TRDG encapsulates all of this.

---

## Common Pitfalls

### Pitfall 1: trdg Dependency Conflict (CRITICAL)

**What goes wrong:** `uv sync` after adding `trdg==1.8.0` resolves to trdg 1.7.0 and downgrades numpy from 2.4.4 to 1.16.6, breaking PyTorch and CV2.
**Why it happens:** trdg 1.8.0 depends on `arabic-reshaper==2.1.3` which was yanked from PyPI due to malformed metadata (pip 24.1+ rejects it). uv falls back to trdg 1.7.0 which has `numpy (<1.17, >=1.16.4)`.
**How to avoid:** Add to `pyproject.toml`:
```toml
[tool.uv]
override-dependencies = [
    "arabic-reshaper==3.0.1",
    "python-bidi==0.6.10",
]
```
Then add `trdg==1.8.0` to `[project] dependencies` normally.
**Warning signs:** `uv sync` output shows `- numpy==2.4.4` and `+ numpy==1.16.6`.

[VERIFIED: uv pip install --dry-run on 2026-05-16 confirmed downgrade to numpy 1.16.6 without override]

### Pitfall 2: `bidi.algorithm` Import Path in python-bidi 0.5+

**What goes wrong:** `from bidi.algorithm import get_display` raises `ModuleNotFoundError` with python-bidi >= 0.5.0.
**Why it happens:** python-bidi 0.5 switched to a Rust backend and changed the public import to `from bidi import get_display`. trdg 1.8.0 uses `from bidi.algorithm import get_display`.
**How to avoid:** python-bidi 0.5.1 added backward-compatibility shim for the old import path. Pin `python-bidi>=0.5.1` (covered by `==0.6.10` in the override). No code change needed.
**Warning signs:** `ImportError: cannot import name 'get_display' from 'bidi.algorithm'` at import time.

[VERIFIED: python-bidi CHANGELOG.rst — 0.5.1 entry: "Added compat for older import, closes #23"]

### Pitfall 3: TRDG Returns `None` When Text/Background Are Too Similar

**What goes wrong:** `next(gen)` returns `(None, label)` instead of a PIL Image.
**Why it happens:** `data_generator.py` returns `None` when `abs(resized_img_px_mean - background_img_px_mean) < 15` (pixel similarity guard). Also returns `None` on `ImageStat` exceptions.
**How to avoid:** Always check `if img is None: continue` in the generation loop. Use dark text color (default `#282828`) on plain white background to ensure sufficient contrast.
**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'save'` when blindly calling `img.save(...)`.

[VERIFIED: inspected `trdg/data_generator.py` from wheel — explicit `return` (None) with print warning]

### Pitfall 4: TRDG `fit=True` May Produce Very Narrow Images

**What goes wrong:** Single Hebrew characters with `fit=True` produce images too narrow for CRNN (CRNN requires width ≥ (label_len + 2) × 4 per `crnn_collate`).
**Why it happens:** `fit=True` trims to text bounding box. A single character like `י` may render at 8–12px wide.
**How to avoid:** Ensure `sample_text` always produces multi-character text (enforce minimum word count or minimum char target). Alternatively use `fit=False` and accept fixed margins. Or apply `crnn_collate`'s existing padding logic at load time.
**Warning signs:** High CTC loss spikes on synthetic data; crops with width < 20px in the output directory.

### Pitfall 5: Hebrew Chars Lost in Corpus Extraction from Noisy Labels

**What goes wrong:** Word splitting produces empty strings, numbers, or punctuation as "words", diluting the corpus.
**Why it happens:** Some labels may contain spaces, digits, or punctuation mixed with Hebrew. Naive `lbl.split()` includes these.
**How to avoid:** Filter words to only those containing at least one Hebrew character (Unicode block U+05D0–U+05EA) before adding to the pool.
**Warning signs:** Generated crops contain digits or Latin characters not present in real data.

[ASSUMED — based on knowledge of typical OCR label noise]

### Pitfall 6: opencv-python vs opencv-python-headless Conflict

**What goes wrong:** trdg's declared dependency `opencv-python>=4.2.0.32` conflicts with the project's `opencv-python-headless`.
**Why it happens:** Both packages provide the `cv2` namespace; pip/uv treats them as conflicting.
**How to avoid:** At runtime they are compatible (both provide `cv2`). The conflict is metadata-level only. The `override-dependencies` in `[tool.uv]` approach does not solve this; instead add `opencv-python-headless` to `override-dependencies` as well to force uv to prefer it:
```toml
override-dependencies = [
    "arabic-reshaper==3.0.1",
    "python-bidi==0.6.10",
    "opencv-python-headless==4.13.0.92",  # override trdg's opencv-python dep
]
```
**Warning signs:** `uv sync` fails with "conflicting dependencies" between opencv-python and opencv-python-headless.

[VERIFIED: uv dry-run showed `+ opencv-python==4.5.3.56` being added even with trdg==1.8.0 install attempt]

---

## Code Examples

### Complete Generator Usage (no file saving)

```python
# Source: inspected trdg-1.8.0-py3-none-any.whl data_generator.py
from trdg.generators.from_strings import GeneratorFromStrings
import numpy as np
from PIL import Image

gen = GeneratorFromStrings(
    strings=["שלום עולם"],
    count=1,
    fonts=["assets/fonts/GveretLevin-Regular.ttf"],
    size=64,
    skewing_angle=3,
    random_skew=True,
    blur=1,
    random_blur=True,
    background_type=1,      # plain white
    distorsion_type=1,      # sine
    distorsion_orientation=2,
    image_mode="L",         # grayscale
    margins=(0, 0, 0, 0),
    rtl=True,
    fit=True,
)

img, label = next(gen)    # img: PIL Image or None; label: str
if img is not None:
    # Save to disk
    img.save("path/to/crop.png")
    # Or convert to numpy for inspection
    arr = np.array(img)   # shape: (H, W), dtype=uint8, range [0, 255]
```

### Coverage Validation

```python
# Source: [ASSUMED] standard Counter pattern
from collections import Counter
import unicodedata
import sys

def check_coverage(
    labels: list[str],
    min_char_count: int,
) -> dict[str, int]:
    """Return {char: count} for chars below threshold."""
    char_counts: Counter[str] = Counter()
    for lbl in labels:
        char_counts.update(unicodedata.normalize("NFC", lbl))
    return {ch: cnt for ch, cnt in char_counts.items() if cnt < min_char_count}

# Before generation:
gaps_before = check_coverage(existing_labels, args.min_char_count)

# After generation:
all_labels = existing_labels + synthetic_labels
gaps_after = check_coverage(all_labels, args.min_char_count)

if gaps_after:
    for ch, cnt in sorted(gaps_after.items()):
        print(f"WARN: char {ch!r} has only {cnt} examples (min={args.min_char_count})")
    sys.exit(1)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TRDG `is_handwritten=True` | Font-based with distortions | TRDG design — TF model | TF not installed; font-based is simpler and controllable |
| `from bidi.algorithm import get_display` | `from bidi import get_display` | python-bidi 0.5 | Handled by 0.5.1 compat shim; no code change needed |
| trdg 1.7.0 (numpy<1.17) | trdg 1.8.0 + override | 2022 | Only safe install path for this project |
| `arabic-reshaper==2.1.3` | `arabic-reshaper>=3.0.1` | PyPI yank | 3.0.1 has no hard deps, clean install |

**Deprecated/outdated:**
- trdg 1.7.0: Pins `numpy<1.17` — unusable with this project
- `arabic-reshaper==2.1.3`: Yanked from PyPI, bad metadata

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Inverse-frequency word weight formula: `1 / (freq/total + epsilon)`, summed over chars in word | Pattern 3 | Weight distribution may not give desired rare-char boost; easy to tune in implementation |
| A2 | `arabic_reshaper.reshape()` is a no-op for Hebrew characters | TRDG RTL notes | If reshaper corrupts Hebrew chars, `rtl=True` would produce garbled images; workaround: pre-apply `bidi.get_display()` directly without reshaper |
| A3 | Font download URLs remain stable (Google Fonts CDN + GitHub raw) | Pattern 5 | URLs break → download step fails; fix by updating URLs in `FONT_URLS` dict |
| A4 | Gveret Levin OFL 1.1 license permits ML training use without restriction | Font section | Low risk; OFL is permissive for all use except selling the font itself |
| A5 | `fit=True` is safe with multi-character text (width will be ≥ min CRNN input) | Pattern 2 | Narrow crops cause CTC loss spikes; mitigation: enforce min char count in sampling |
| A6 | Word-splitting on whitespace + filtering empties is sufficient for corpus extraction | Pitfall 5 | Mixed-script labels could include Latin/digits; add Hebrew char filter to be safe |

---

## Open Questions

1. **Font `--fonts_dir` override vs bundled download**
   - What we know: D-03 says `--fonts_dir` overrides downloaded defaults; default is lazy-download to `assets/fonts/`
   - What's unclear: Should `assets/fonts/` be gitignored entirely, or should the font name list (not files) be committed?
   - Recommendation: Gitignore `assets/fonts/*.ttf` (binary files); commit `assets/fonts/.gitkeep`; FONT_URLS dict stays in source code.

2. **Single-character Hebrew words**
   - What we know: D-04 says filter single chars OR include them — Claude's discretion
   - What's unclear: Hebrew has many meaningful single-character words (prepositions like ב, ל, מ)
   - Recommendation: Include single-character words; they're legitimate Hebrew and improve rare-char coverage.

3. **`--min_char_count` default value**
   - What we know: CLI should accept it; exit non-zero if gaps remain after generation
   - What's unclear: What's a useful default for the current dataset size?
   - Recommendation: Default `--min_char_count=5`; low enough not to fail on first run with small dataset, high enough to surface real gaps.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | All | ✓ | 3.13.13 | — |
| Pillow | TRDG image output | ✓ | 12.2.0 | — |
| OpenCV (headless) | TRDG background_generator | ✓ | 4.13.0 | — |
| numpy | Weighted sampling | ✓ | 2.4.4 | — |
| requests | Font download | ✓ | 2.33.1 | urllib.request (stdlib) |
| trdg | Image rendering | ✗ | — (1.8.0 available, needs install) | None — core requirement |
| arabic-reshaper | trdg RTL (transitive) | ✗ | — (3.0.1 needed) | None |
| python-bidi | trdg RTL (transitive) | ✗ | — (0.6.10 needed) | None |
| Hebrew fonts (.ttf) | TRDG font rendering | ✗ | — (no Hebrew fonts system-wide) | Download on first use |
| ClearML | Run logging | ✓ | 2.1.5 | — |
| Internet access | Font download (first run only) | ✓ | — | Manual font placement via `--fonts_dir` |

**Missing dependencies with no fallback:**
- `trdg==1.8.0` + `arabic-reshaper==3.0.1` + `python-bidi==0.6.10` — must be installed via `uv sync` with `override-dependencies` in place.

**Missing dependencies with fallback:**
- Hebrew fonts: Manual placement via `--fonts_dir` if internet unavailable.

---

## Sources

### Primary (HIGH confidence)

- trdg-1.8.0-py3-none-any.whl (inspected with zipfile) — full source of `from_strings.py`, `data_generator.py`, `utils.py`, `computer_text_generator.py`
- PyPI JSON API: `https://pypi.org/pypi/trdg/{version}/json` — dependency constraints for 1.7.0 and 1.8.0
- PyPI JSON API: `https://pypi.org/pypi/arabic-reshaper/json` — latest version 3.0.1
- PyPI JSON API: `https://pypi.org/pypi/python-bidi/0.6.10/json` — requires-python >=3.9
- python-bidi CHANGELOG.rst (GitHub) — 0.5.1 backward-compat shim for `bidi.algorithm` import
- uv docs `https://docs.astral.sh/uv/reference/settings/#override-dependencies` — `[tool.uv] override-dependencies` syntax
- `src/ctc_utils.py` (local codebase) — `load_crop`, `build_charset`, `CropDataset` integration points
- `src/clearml_utils.py` (local codebase) — `init_task`, `upload_file_artifact` signatures
- `src/manifest_schema.py` (local codebase) — `MANIFEST_COLUMNS`, required fields
- `uv pip install trdg --dry-run` live on 2026-05-16 — confirmed numpy downgrade to 1.16.6 without override
- Google Fonts CSS API verified: Frank Ruhl Libre v23 TTF URLs (66KB files confirmed valid)
- GitHub API: Gveret Levin `fonts/ttf/GveretLevin-Regular.ttf` (59KB TTF confirmed valid, OFL 1.1)

### Secondary (MEDIUM confidence)

- TextRecognitionDataGenerator GitHub README — generator class list, CLI parameter overview
- python-bidi readthedocs — `bidi.algorithm.get_display` still accessible via compat in 0.6.x

### Tertiary (LOW confidence)

- WebSearch re: TRDG Arabic/Hebrew RTL workarounds — confirms known label-reversal bug exists, `rtl=True` is correct fix

---

## Metadata

**Confidence breakdown:**
- TRDG API: HIGH — verified from wheel source
- Dependency conflict: HIGH — verified live with `uv pip install --dry-run`
- Font URLs: HIGH — verified with curl (file size, `file` command TrueType confirmed)
- Hebrew rendering correctness: MEDIUM — rtl=True code path verified in source, actual rendering not tested
- Inverse-frequency weighting: MEDIUM — pattern is standard; specific formula is ASSUMED (A1)
- Font license (OFL): HIGH — LICENSE file confirmed OFL 1.1 in gveret-levin repo

**Research date:** 2026-05-16
**Valid until:** 2026-08-16 (font URLs may change; trdg is unmaintained since 2022)
