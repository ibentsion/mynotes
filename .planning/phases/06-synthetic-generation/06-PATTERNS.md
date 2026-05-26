# Phase 6: Synthetic Generation - Pattern Map

**Mapped:** 2026-05-16
**Files analyzed:** 2 (1 new script + 1 pyproject.toml modification)
**Analogs found:** 2 / 2

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/generate_synthetic.py` | CLI script / service | batch + file-I/O | `src/train_ctc.py` | exact (same role: argparse CLI + ClearML task + file output) |
| `pyproject.toml` | config | — | `pyproject.toml` (self) | self-modification |

---

## Pattern Assignments

### `src/generate_synthetic.py` (CLI script, batch + file-I/O)

**Primary analog:** `src/train_ctc.py`
**Supporting analogs:** `src/tune.py` (init_task call sequence), `src/clearml_utils.py` (helpers)

---

#### Imports pattern

Copy the module-level `from clearml import Task` import for test patchability. All heavy
imports (trdg, PIL, torch, etc.) belong inside functions to avoid import-time side effects.

**Source:** `src/train_ctc.py` lines 1–13
```python
"""generate_synthetic.py — render Hebrew text crops via TRDG, write manifest.csv, log to ClearML."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Task  # noqa: F401  # module-level for test patchability — RESEARCH.md Pattern 6

from src.clearml_utils import init_task, upload_file_artifact
```

**TRDG import (inside function, not module-level):**
```python
# Inside generate() or main() — avoids transitive wikipedia import at module level
from trdg.generators.from_strings import GeneratorFromStrings  # NOT trdg.generators
```

---

#### Argument parser pattern

**Source:** `src/train_ctc.py` lines 38–134 (`_build_parser`)

Copy the `_build_parser() -> argparse.ArgumentParser` private function convention with
`p.add_argument(... help="...")` on every argument, `type=Path` for path args.

```python
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render synthetic Hebrew crops via TRDG; write manifest.csv; log to ClearML."
    )
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"),
                   help="Existing labeled manifest to extract corpus from")
    p.add_argument("--output_dir", type=Path, default=Path("outputs/synthetic"),
                   help="Directory for output crops/ and manifest.csv")
    p.add_argument("--count", type=int, default=500,
                   help="Number of synthetic crops to generate (SYN-01)")
    p.add_argument("--fonts_dir", type=Path, default=Path("assets/fonts"),
                   help="Directory containing .ttf fonts; downloads defaults on first use (D-03)")
    p.add_argument("--wordlist", type=Path, default=None,
                   help="Optional extra word list (one word per line) merged into corpus (SYN-03)")
    p.add_argument("--min_char_count", type=int, default=5,
                   help="Exit non-zero if any char appears fewer times after generation (SYN-04)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible corpus sampling")
    return p
```

---

#### ClearML task init and connect pattern

**Source:** `src/train_ctc.py` lines 565–581 (`main`) and `src/tune.py` lines 197–203

The invariant: `init_task()` BEFORE `parse_args()`, then `task.connect(vars(args))` BEFORE
any real work. This is the same sequence in every script.

```python
def main() -> int:
    # init_task BEFORE parse_args — ClearML must capture args before parse (D-07)
    task = init_task("handwriting-hebrew-ocr", "generate_synthetic")

    args = _build_parser().parse_args()
    task.connect(vars(args), name="hyperparams")
    # ... do work ...
    upload_file_artifact(task, "manifest", manifest_path)
    return 0
```

**Source:** `src/clearml_utils.py` lines 1–22
```python
# init_task signature — call with project + task_name + optional tags
task = init_task("handwriting-hebrew-ocr", "generate_synthetic")

# upload_file_artifact signature — wraps task.upload_artifact(name, artifact_object=str(path))
upload_file_artifact(task, "manifest", args.output_dir / "manifest.csv")
```

---

#### Error handling / early-exit pattern

**Source:** `src/train_ctc.py` lines 594–614

Return integer exit codes; print to `sys.stderr` on error; never raise from `main()` itself.

```python
def main() -> int:
    # ...
    if not args.manifest.exists():
        print(f"ERROR: --manifest does not exist: {args.manifest}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.manifest)
    labeled = df[df["status"] == "labeled"].reset_index(drop=True)

    if labeled.empty:
        print("ERROR: no labeled rows in manifest — cannot build corpus.", file=sys.stderr)
        return 3

    # SYN-04: exit non-zero when coverage gaps remain after generation
    if gaps_after:
        for ch, cnt in sorted(gaps_after.items()):
            print(f"WARN: char {ch!r} has only {cnt} examples (min={args.min_char_count})")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

#### Manifest reading and filtering pattern

**Source:** `src/train_ctc.py` lines 350–351 and 599–600

Always filter `status == "labeled"` and reset index. `CropDataset` relies on this — synthetic
manifest must write `status="labeled"` so downstream consumers filter identically.

```python
df = pd.read_csv(args.manifest)
labeled = df[df["status"] == "labeled"].reset_index(drop=True)
existing_labels = labeled["label"].tolist()
```

---

#### NFC normalization pattern

**Source:** `src/ctc_utils.py` lines 18–28 (`build_charset`)

All label text must be NFC-normalized before character frequency counting or word splitting.
This matches `build_charset` exactly.

```python
import unicodedata

normalized = [unicodedata.normalize("NFC", lbl) for lbl in existing_labels]
```

---

#### Output directory creation pattern

**Source:** `src/train_ctc.py` line 613

```python
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "crops").mkdir(parents=True, exist_ok=True)
```

---

#### Manifest writing — required columns

**Source:** `src/manifest_schema.py` lines 1–16 and `src/ctc_utils.py` lines 232–240

`CropDataset.__getitem__` reads only `crop_path` and `label` from each row (after the caller
filters `status == "labeled"`). The synthetic manifest must contain at minimum:

| Column | Value |
|--------|-------|
| `crop_path` | absolute or relative path to the saved PNG |
| `label` | the text string rendered (NFC-normalized; original, not BiDi-reordered) |
| `status` | `"labeled"` — hard-coded so CropDataset filter passes |

Omit all other `MANIFEST_COLUMNS` columns (`pdf_path`, `page_path`, `page_num`, `x`, `y`,
`w`, `h`, `area`, `is_flagged`, `flag_reasons`, `notes`) — they are optional and absent
from synthetic rows.

```python
rows = []
for crop_path, label in generated:
    rows.append({"crop_path": str(crop_path), "label": label, "status": "labeled"})
manifest_df = pd.DataFrame(rows, columns=["crop_path", "label", "status"])
manifest_df.to_csv(args.output_dir / "manifest.csv", index=False)
```

---

#### ClearML scalar logging pattern (optional per-char counts)

**Source:** `src/train_ctc.py` lines 496–507 (`logger.report_scalar`)

```python
logger = task.get_logger()
for char, count in sorted(char_counts.items()):
    logger.report_scalar("char_coverage", char, iteration=0, value=count)
```

---

### `pyproject.toml` (config, self-modification)

**Analog:** `pyproject.toml` itself (existing `[project.scripts]` block)

#### Console script entry point

**Source:** `pyproject.toml` lines 37–45

Follow the existing `verb-noun = "src.module:main"` convention:

```toml
[project.scripts]
generate-synthetic = "src.generate_synthetic:main"
```

#### Dependency additions

Add to `[project] dependencies` and add `[tool.uv] override-dependencies`.
RESEARCH.md Pitfall 1 and 6 make the overrides mandatory.

```toml
[project]
dependencies = [
    # ... existing deps ...
    "trdg==1.8.0",
    "arabic-reshaper==3.0.1",
    "python-bidi==0.6.10",
]

[tool.uv]
override-dependencies = [
    "arabic-reshaper==3.0.1",
    "python-bidi==0.6.10",
    "opencv-python-headless==4.13.0.92",
]
```

---

## Shared Patterns

### ClearML task initialization order
**Source:** `src/train_ctc.py` lines 565–584, `src/tune.py` lines 197–203
**Apply to:** `src/generate_synthetic.py`

The mandatory order is: `init_task()` → `parse_args()` → `task.connect(vars(args))` → work.
Never call `parse_args()` before `init_task()`.

### Module-level `from clearml import Task` import
**Source:** `src/train_ctc.py` line 11, `src/tune.py` line 22
**Apply to:** `src/generate_synthetic.py`

```python
from clearml import Task  # noqa: F401  # module-level for test patchability
```

Every CLI script does this even when `Task` is not referenced directly in module scope,
so tests can patch `src.generate_synthetic.Task`.

### NFC normalization before any character-level operation
**Source:** `src/ctc_utils.py` lines 26–27 (`build_charset`), line 36 (`encode_label`)
**Apply to:** corpus building, coverage validation, word splitting in `generate_synthetic.py`

```python
unicodedata.normalize("NFC", text)
```

### `main() -> int` + `raise SystemExit(main())` entry point
**Source:** `src/train_ctc.py` lines 565, 627; `src/tune.py` lines 197, 236
**Apply to:** `src/generate_synthetic.py`

```python
def main() -> int:
    # ... returns int exit code ...


if __name__ == "__main__":
    raise SystemExit(main())
```

### Manifest filtering: `df[df["status"] == "labeled"]`
**Source:** `src/train_ctc.py` lines 351, 600
**Apply to:** Reading the existing manifest in `generate_synthetic.py`

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| TRDG image rendering logic | utility | batch | No image synthesis exists in the codebase — use RESEARCH.md Pattern 2 (GeneratorFromStrings config) |
| Font lazy-download logic | utility | file-I/O | No font download pattern in codebase — use RESEARCH.md Pattern 5 (requests.get + dest.write_bytes) |
| Inverse-frequency word weighting | utility | transform | No weighted sampling in codebase — use RESEARCH.md Pattern 3 (Counter + np.random.choice(p=weights)) |

---

## Metadata

**Analog search scope:** `src/`, `pyproject.toml`
**Files scanned:** `src/train_ctc.py`, `src/tune.py`, `src/clearml_utils.py`, `src/ctc_utils.py`, `src/manifest_schema.py`, `pyproject.toml`
**Key line references:**
- `src/train_ctc.py:1–13` — imports
- `src/train_ctc.py:38–134` — `_build_parser` pattern
- `src/train_ctc.py:565–627` — `main()` init/connect/exit pattern
- `src/tune.py:197–203` — init_task call order
- `src/clearml_utils.py:1–22` — `init_task`, `upload_file_artifact` signatures
- `src/ctc_utils.py:18–28` — NFC normalization in `build_charset`
- `src/ctc_utils.py:101–114` — `load_crop` (image format contract for synthetic outputs)
- `src/ctc_utils.py:214–240` — `CropDataset.__init__` and `__getitem__` (manifest field contract)
- `src/manifest_schema.py:1–16` — full column list; only `crop_path`, `label`, `status` required
- `pyproject.toml:37–45` — `[project.scripts]` console entry point convention
**Pattern extraction date:** 2026-05-16
