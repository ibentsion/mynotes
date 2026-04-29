---
phase: "03"
plan: "01"
subsystem: training-evaluation
tags: [torch, ctc, crnn, charset, collate, split]
dependency_graph:
  requires: []
  provides: [src/ctc_utils.py, torch-cpu-wheel]
  affects: [03-02-PLAN.md, 03-03-PLAN.md]
tech_stack:
  added: ["torch==2.11.0+cpu via pytorch-cpu index"]
  patterns:
    - "CRNN: Conv2d×3 -> BiLSTM(2,bidirectional) -> Linear; raw logits (no log_softmax inside)"
    - "CTC blank at index 0; charset chars at indices 1..N"
    - "Width padded to multiple of 4 to avoid input_lengths mismatch (Pitfall 1)"
    - "NFC normalization for Hebrew charset (TRAN-02)"
    - "Half-page unit split: sorted keys, ceil(N*0.2) min 1 to val (D-03, D-04)"
    - "ty: ignore[invalid-argument-type] on int(idx) in iterrows() — pandas Hashable index"
key_files:
  created:
    - pyproject.toml (modified — torch dep + pytorch-cpu index routing)
    - src/ctc_utils.py
    - tests/test_ctc_utils.py
  modified:
    - uv.lock
decisions:
  - "NFC normalization chosen for Hebrew charset (standard; TRAN-02)"
  - "ty: ignore suppression on int(idx) from df.iterrows() — pandas index typed as Hashable, not SupportsInt, despite being int at runtime for default RangeIndex"
  - "Test assertion corrected: 'שלום' ends with ם (final mem U+05DD), not מ (regular mem U+05DE)"
metrics:
  duration: "6 minutes"
  completed: "2026-04-29"
  tasks_completed: 2
  files_changed: 4
---

# Phase 3 Plan 1: torch CPU Install + ctc_utils Shared Module Summary

CRNN model, charset I/O, greedy CTC decode, CER, image loading, collate, and half-page split consolidated into `src/ctc_utils.py` with torch 2.11.0+cpu installed via the pytorch-cpu uv index.

## What Was Built

### Task 1: torch 2.11.0+cpu dependency (commit `59029c8`)

- Added `"torch==2.11.0"` to `[project] dependencies` in pyproject.toml
- Added `[[tool.uv.index]]` pytorch-cpu pointing to `https://download.pytorch.org/whl/cpu` with `explicit = true`
- Added `[tool.uv.sources]` routing torch to pytorch-cpu index
- Ran `uv sync` — downloaded and installed `torch==2.11.0+cpu` (181.5 MiB)
- All 54 prior tests continue to pass

### Task 2: src/ctc_utils.py + 30 unit tests (commit `0d9ff52`)

Implemented every public function in the exact interface specified by the plan:

| Function | Requirement | Key detail |
|----------|-------------|------------|
| `build_charset` | TRAN-02 | NFC normalization before set union; sorted |
| `encode_label` | TRAN-02 | blank=0 reserved; charset index +1 |
| `save_charset` / `load_charset` | — | JSON UTF-8 round-trip |
| `greedy_decode` | D-05 | argmax → collapse repeats → remove blank |
| `cer` | EVAL-02 | hand-rolled Levenshtein; empty reference → len(hypothesis) |
| `load_crop` | D-01 | 64px target height, proportional width, [0,1] float32 |
| `crnn_collate` | — | pad to nearest multiple of 4 (Pitfall 1 fix) |
| `build_half_page_units` | D-03 | reads page height from page_path with cache |
| `split_units` | D-04 | sorted keys, ceil(N*0.2) min 1 to val |
| `CRNN` | TRAN-04 | Conv2d×3 → BiLSTM(2,bidirectional) → Linear; T = W//4 |
| `resolve_device` | TRAN-05 | cuda if available else cpu |

## Test Coverage

30 tests in `tests/test_ctc_utils.py`:
- 8 charset tests (build_charset, encode_label, save/load_charset)
- 7 decode+CER tests (greedy_decode, cer)
- 6 image I/O + collate tests (load_crop, crnn_collate)
- 5 split tests (build_half_page_units, split_units)
- 4 model+device tests (CRNN forward shape, resolve_device)

Full suite: 84 tests passed (54 prior + 30 new).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected test assertion for Hebrew charset**
- **Found during:** Task 2 TDD RED→GREEN
- **Issue:** Test expected 'מ' (regular mem U+05DE) in `build_charset(["שלום", "שם"])` but "שלום" ends with 'ם' (final mem U+05DD). The expected set should be {ו, ל, ם, ש} not {ו, ל, מ, ם, ש}.
- **Fix:** Corrected `set(result) == {"ש", "ל", "ו", "מ", "ם"}` to `set(result) == {"ש", "ל", "ו", "ם"}` with an explanatory comment.
- **Files modified:** `tests/test_ctc_utils.py`
- **Commit:** `0d9ff52`

**2. [Rule 2 - Missing functionality] Added ty: ignore for pandas iterrows() index type**
- **Found during:** Task 2 ty check
- **Issue:** `ty` reports `int(idx)` where `idx: Hashable` from `df.iterrows()` as `invalid-argument-type`. Pandas types its index labels as `Hashable` in stubs, but at runtime a default RangeIndex yields `int`. Cannot fix without changing the iteration pattern.
- **Fix:** Added `# ty: ignore[invalid-argument-type]` comment with justification (CLAUDE.md: "If a warning truly can't be fixed, add an inline ignore with a justification comment").
- **Files modified:** `src/ctc_utils.py:177`
- **Commit:** `0d9ff52`

### Pre-existing Issues (out of scope, logged to deferred-items.md)

`uv run ruff check src/ tests/` and `uv run ty check src/` were NOT clean before this plan ran. Pre-existing issues in `backfill_page_paths.py`, `cluster_sampler.py`, `review_app.py`, and test files were verified by `git stash` check. These are out of scope and logged to `.planning/phases/03-training-evaluation/deferred-items.md`.

## Known Stubs

None — all public functions are fully implemented and wired.

## Self-Check: PASSED

Files created:
- `src/ctc_utils.py` — exists
- `tests/test_ctc_utils.py` — exists

Commits:
- `59029c8` — chore(03-01): add torch 2.11.0 CPU-only dependency
- `0d9ff52` — feat(03-01): implement src/ctc_utils.py + 30-test suite
