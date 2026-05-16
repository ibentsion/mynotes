---
phase: 06-synthetic-generation
reviewed: 2026-05-16T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - src/generate_synthetic.py
  - tests/test_generate_synthetic.py
  - pyproject.toml
  - .gitignore
findings:
  critical: 2
  warning: 6
  info: 4
  total: 12
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-05-16
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Reviewed `src/generate_synthetic.py` (new TRDG rendering pipeline), `tests/test_generate_synthetic.py`,
`pyproject.toml` (dependency changes), and `.gitignore`. The implementation is largely well-structured:
ClearML init order is correct, the None-image guard is in place, the manifest CSV columns match the
`CropDataset` filter contract, and the core corpus/sampling functions are individually correct.

Two blockers were found: a `ZeroDivisionError` in `build_word_corpus` reachable when existing labels
are empty strings and `--wordlist` is supplied, and an infinite loop in `_generate_until_count`
triggered when any label has zero characters (causing `sample_text` to return `""` which TRDG
likely returns `None` for — the loop then spins indefinitely). Both are latent bugs that survive
normal use but will surface with real-world manifest data that has blank or missing label cells.

Six warnings cover unhandled `ValueError` propagation from `build_word_corpus` through `main()`,
font download URL fragility, missing `requests` declaration in `pyproject.toml`, loose `>=` version
pins on production dependencies, a dead `monkeypatch.setattr` block in one test, and the `Task`
import that is silently dead code kept alive by a `noqa` suppressor without an active test-side
reason.

---

## Critical Issues

### CR-01: ZeroDivisionError in `build_word_corpus` when all labels are empty and `--wordlist` is supplied

**File:** `src/generate_synthetic.py:105-113`

**Issue:** `total_chars = sum(char_freq.values())` is `0` when every label in `existing_labels`
is an empty string (or whitespace-only, so `char_freq` is never populated). This path is reachable
in production: `main()` does `labeled["label"].astype(str).tolist()` — a CSV row with an empty
label cell produces `""`. When `extra_words` (from `--wordlist`) contains Hebrew words, those words
pass the `ValueError` guard and `words` is non-empty, but the subsequent `char_freq.get(c, 1) / total_chars`
divides by zero:

```python
word_scores = np.array(
    [
        sum(1.0 / (char_freq.get(c, 1) / total_chars + 1e-8) for c in word)  # ZeroDivisionError!
        for word in words
    ],
    ...
)
```

**Fix:** Guard `total_chars` before the comprehension, and raise a clear `ValueError` if
`char_freq` is empty:

```python
total_chars = sum(char_freq.values())
if total_chars == 0:
    raise ValueError(
        "build_word_corpus: all labels are empty — cannot compute character frequencies. "
        "Ensure labeled rows contain non-empty Hebrew text."
    )
```

---

### CR-02: Potential infinite loop in `_generate_until_count` when any label has zero characters

**File:** `src/generate_synthetic.py:381-386`

**Issue:** `build_char_count_distribution` returns `[len(NFC(lbl)) for lbl in labels]`. If any
label is `""`, the distribution contains `0`. `rng.choice(char_counts)` can then select `0`,
which is passed as `target_chars=0` to `sample_text`. With `target_chars=0` the `while total <
target_chars` loop never runs and `sample_text` returns `""`. `render_crops([""], ...)` is then
called — TRDG renders an empty string, which it typically returns `None` for (matching RESEARCH.md
Pitfall 3). The None-skip causes `render_crops` to return `[]`. `all_rows` is unchanged and the
outer `while len(all_rows) < target` loop repeats forever with `text = ""`:

```python
# _generate_until_count loop — spins forever when sample_text returns ''
while len(all_rows) < target:
    text = sample_text(words, weights, int(rng.choice(char_counts)), rng)  # can be ''
    rows = render_crops([text], font_paths, out_crops_dir, start_idx=len(all_rows))
    all_rows.extend(rows)  # rows=[] when TRDG returns None for ''
```

**Fix:** Filter zero-length entries from `char_counts` before sampling, or clamp the selected
target to at least 1:

```python
char_counts = build_char_count_distribution(existing_labels)
char_counts = char_counts[char_counts > 0]  # drop empty-label lengths
if char_counts.size == 0:
    raise ValueError("_generate_until_count: all existing labels are empty strings")
```

Alternatively, clamp inside `sample_text`:

```python
target_chars = max(1, int(rng.choice(char_counts)))
```

---

## Warnings

### WR-01: `ValueError` from `build_word_corpus` propagates as raw traceback through `main()`

**File:** `src/generate_synthetic.py:476-483`

**Issue:** `_generate_until_count` calls `build_word_corpus`, which raises `ValueError` when no
Hebrew words exist in the corpus. `main()` has no `try/except` around the
`_generate_until_count` call, so any `ValueError` (or other exception from the render path)
prints a full traceback and exits with code `1` — not one of the documented exit codes (2, 3, 4).
This breaks the contract advertised in the docstring and makes the error unscriptable.

**Fix:**
```python
try:
    rows = _generate_until_count(...)
except ValueError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 5  # or reuse an existing code
```

---

### WR-02: `requests` is not declared in `pyproject.toml` — implicit transitive dependency

**File:** `pyproject.toml:6-21`

**Issue:** `ensure_fonts` does `import requests` at runtime but `requests` is not listed in
`[project.dependencies]`. It is available only as a transitive dependency of `clearml`. If `clearml`
drops `requests` in a future release, `ensure_fonts` silently breaks on first-run font download
with an `ImportError`.

**Fix:** Add `requests` as an explicit pinned dependency:
```toml
"requests==2.33.1",
```

---

### WR-03: Font download URLs are not commit-pinned and have no integrity verification

**File:** `src/generate_synthetic.py:25-38` and `src/generate_synthetic.py:210-215`

**Issue:** Two concerns:

1. `GveretLevin-Regular.ttf` is fetched from a GitHub `master` branch URL. If the upstream repo
   reorganises its file tree, the URL silently 404s on first run (no retry, just `raise_for_status`).
2. Neither URL is verified against a known SHA-256 hash after download. A corrupted download
   (truncated response, CDN fault) is written to disk and cached as-is. On subsequent runs
   `existing_ttfs` is non-empty so the corrupt file is returned without re-download, and TRDG
   silently fails or produces garbage images.

**Fix:** Pin the GitHub URL to a specific commit SHA and add hash verification after download:

```python
import hashlib

FONT_SHA256: dict[str, str] = {
    "GveretLevin-Regular.ttf": "<sha256-of-file>",
    ...
}

# after dest.write_bytes(resp.content):
digest = hashlib.sha256(resp.content).hexdigest()
expected = FONT_SHA256[name]
if digest != expected:
    dest.unlink()
    raise RuntimeError(f"font {name}: hash mismatch (got {digest}, expected {expected})")
```

---

### WR-04: `sample_text` docstring falsely claims "always selects at least one word"

**File:** `src/generate_synthetic.py:125-126`

**Issue:** The docstring states "Always selects at least one word (guards RESEARCH.md Pitfall 4
narrow-image risk)." This is false when `target_chars=0`: the `while total < target_chars` loop
does not execute and the function returns `""`. This contract violation contributes directly to
CR-02 and creates a misleading API.

**Fix:** Either enforce the invariant in code (clamp `target_chars` to at least 1 at the top of
`sample_text`) or correct the docstring to document the zero-length case:

```python
def sample_text(..., target_chars: int, ...) -> str:
    target_chars = max(1, target_chars)  # guard against zero-length distribution entries
    ...
```

---

### WR-05: Dead `monkeypatch.setattr` block in `test_main_clearml_order_init_before_parse`

**File:** `tests/test_generate_synthetic.py:480-484`

**Issue:** Lines 480–484 set `sys.argv` to one value, then lines 487–494 immediately override it
with a different `monkeypatch.setattr` call. The first `setattr` (lines 480–484) is dead code —
its effect is never visible because it is overwritten before the function under test runs.

```python
# Lines 480-484 — overwritten by the identical call below; remove these lines
monkeypatch.setattr("sys.argv", [
    "generate-synthetic",
    "--manifest", str(manifest_path),
    "--output_dir", str(tmp_path / "out"),
])
```

**Fix:** Delete lines 480–484 (the first `monkeypatch.setattr` call in this test).

---

### WR-06: Three production dependencies use `>=` instead of exact pins

**File:** `pyproject.toml:13,15,17`

**Issue:** `scikit-learn>=1.5`, `openai>=1.0`, and `matplotlib>=3.9` use minimum-version
constraints instead of exact pins. CLAUDE.md requires `==` pins for Python dependencies
(`"pin exact versions (== not >=)"`). These allow silent upgrades that can change behavior
across environments.

**Fix:** Pin each to the version currently resolved in `uv.lock` (inspect with `uv pip list`):
```toml
"scikit-learn==<resolved-version>",
"openai==<resolved-version>",
"matplotlib==<resolved-version>",
```

---

## Info

### IN-01: `from clearml import Task` is genuinely unused (suppressed with `noqa: F401`)

**File:** `src/generate_synthetic.py:12`

**Issue:** `Task` is imported at module level with `# noqa: F401  # module-level for test patchability`,
but no test patches `src.generate_synthetic.Task`. Tests patch `src.generate_synthetic.init_task`
and `src.generate_synthetic.upload_file_artifact` (both imported from `clearml_utils`). The `Task`
import and its `noqa` suppressor are dead code.

**Fix:** Delete line 12. If future tests need to patch `Task` directly, add it back with a
test-side justification.

---

### IN-02: Dev dependencies also use `>=` pins (`pytest`, `ruff`, `ty`)

**File:** `pyproject.toml:25-27`

**Issue:** `pytest>=8.0`, `ruff>=0.6`, and `ty>=0.0.1a1` use loose constraints. `ty>=0.0.1a1`
in particular can resolve alpha releases, which may have breaking API changes between runs.
CLAUDE.md's pin policy applies to dev deps too.

**Fix:** Pin to exact versions (e.g., `ty==0.0.1a8`) after confirming the currently resolved
version in `uv.lock`.

---

### IN-03: `write_manifest` creates `output_dir/crops/` unnecessarily

**File:** `src/generate_synthetic.py:307`

**Issue:** `write_manifest` creates `(output_dir / "crops")` even though that directory is
already created by `main()` (line 472) before `_generate_until_count` is called. This is
redundant coupling: `write_manifest` has no documented responsibility for crops directory
creation, and the extra `mkdir` is surprising to readers of the function.

**Fix:** Remove line 307 from `write_manifest`:
```python
# Remove this line:
(output_dir / "crops").mkdir(parents=True, exist_ok=True)
```

---

### IN-04: Missing test for `_resolve_extra_words`

**File:** `tests/test_generate_synthetic.py`

**Issue:** `_resolve_extra_words` has no direct test. It is exercised only transitively through
`test_build_word_corpus_merges_extra_words` (which does not call it). The function has
non-trivial behavior: it strips whitespace, skips blank lines, reads with UTF-8 encoding, and
returns `None` vs a list. Edge cases (non-existent file, file with blank lines, all-blank file)
are untested.

**Fix:** Add a parameterized test:
```python
def test_resolve_extra_words_skips_blank_lines(tmp_path: Path) -> None:
    from src.generate_synthetic import _resolve_extra_words
    wl = tmp_path / "words.txt"
    wl.write_text("שלום\n\n  \nעולם\n", encoding="utf-8")
    result = _resolve_extra_words(wl)
    assert result == ["שלום", "עולם"]

def test_resolve_extra_words_none_when_not_provided() -> None:
    from src.generate_synthetic import _resolve_extra_words
    assert _resolve_extra_words(None) is None
```

---

_Reviewed: 2026-05-16_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
