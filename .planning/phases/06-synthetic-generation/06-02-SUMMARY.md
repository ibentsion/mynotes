---
phase: 06-synthetic-generation
plan: 02
subsystem: synthetic-generation
tags: [tdd, corpus-building, inverse-frequency-weighting, nfc, hebrew-filter, coverage-validation]

# Dependency graph
requires:
  - phase: 06-synthetic-generation
    plan: 01
    provides: trdg==1.8.0 installed, generate-synthetic console script registered
provides:
  - build_word_corpus: NFC-normalized Hebrew word corpus with inverse-frequency sampling weights
  - sample_text: empirical-length text sampler from weighted word pool
  - build_char_count_distribution: character count distribution array from labels
  - check_coverage: pure coverage gap detector returning chars below threshold
  - src/generate_synthetic.py module skeleton with final import block (no trdg/PIL at module level)
  - tests/test_generate_synthetic.py: 12 passing unit tests
affects: [06-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inverse-frequency word weighting: sum(1/(char_freq[c]/total + epsilon) for c in word); word_scores / sum"
    - "Hebrew-only filter: U+05D0-U+05EA presence check; includes single-char Hebrew words (prepositions)"
    - "sample_text tracks len(joined)+len(spaces) to guarantee len(result) >= target_chars"
    - "check_coverage is pure — no sys.exit, no print; caller owns exit-code decision"
    - "Stable dedup via dict insertion order (Python 3.7+ guarantee) instead of set()"

key-files:
  created:
    - src/generate_synthetic.py
    - tests/test_generate_synthetic.py
  modified: []

key-decisions:
  - "All 12 tests (Tasks 1+2) committed in a single RED commit; single GREEN feat commit covers all 4 functions — plan called for separate RED/GREEN per task but single-file implementation makes combined approach cleaner"
  - "sample_text tracks actual joined string length (not cumulative word+space count) to guarantee len(result) >= target_chars"
  - "Stable word deduplication via dict insertion order (not set) to preserve deterministic ordering for reproducible weighted sampling"
  - "upload_file_artifact imported with noqa F401 — unused in stub main(), used by Plan 03 rendering loop"

requirements-completed: [SYN-03, SYN-04]

# Metrics
duration: 3min
completed: 2026-05-16
---

# Phase 6 Plan 02: Corpus Builder and Coverage Validator Summary

**Inverse-frequency Hebrew word corpus and NFC coverage validation implemented as 4 pure functions with 12 TDD-verified unit tests**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-16T09:59:35Z
- **Completed:** 2026-05-16T10:03:07Z
- **Tasks:** 2
- **Files created:** 2 (src/generate_synthetic.py, tests/test_generate_synthetic.py)

## Accomplishments

- Implemented `build_word_corpus`: NFC-normalize labels, whitespace-split, Hebrew-only filter (U+05D0–U+05EA), stable dedup, merge extra_words, inverse-frequency weights via Counter; raises ValueError on empty corpus
- Implemented `sample_text`: weighted word sampling until `len(" ".join(selected)) >= target_chars`; always selects >= 1 word (guards Pitfall 4 narrow-image risk)
- Implemented `build_char_count_distribution`: NFC character counts per label as np.ndarray
- Implemented `check_coverage`: Counter over NFC-normalized labels returning chars below `min_char_count`; pure function with no side effects
- Module skeleton with final import block: `from clearml import Task  # noqa: F401`, no trdg/PIL at module level
- Stub `main()` + `_build_parser()` for Plan 03 extension
- 12 unit tests covering all specified behaviors plus edge cases; ruff + ty clean

## Task Commits

1. **Task 1 RED: Failing tests for all behaviors** - `f39aa37` (test)
2. **Task 1+2 GREEN: All 4 functions implemented** - `d755a93` (feat)

## Files Created

- `src/generate_synthetic.py` - Module with 4 pure functions + argparse skeleton + stub main()
- `tests/test_generate_synthetic.py` - 12 unit tests covering corpus, sampling, distribution, coverage

## Decisions Made

- **Combined RED commit**: All 12 tests (Task 1 + Task 2) written in a single RED commit since they target functions in the same module and the plan's acceptance criteria is satisfied by the test(06-02): → feat(06-02): sequence
- **Stable dedup via dict**: `dict` insertion order (Python 3.7+) replaces `set()` for deterministic word ordering — reproducible weighted sampling with `rng.choice(p=weights)`
- **sample_text length tracking**: Tracks actual joined string length (`len(joined) = sum(len(w)) + len(selected) - 1`) instead of cumulative `len(w)+1` per word; guarantees `len(result) >= target_chars` invariant
- **check_coverage pure**: No `sys.exit`, no `print` — exit-code decision deferred to Plan 03 `main()` per 06-PATTERNS.md error handling pattern

## Deviations from Plan

### Minor Implementation Differences

**1. [Rule 1 - Bug] Fixed off-by-one in sample_text length tracking**
- **Found during:** Task 1 GREEN phase (first test run)
- **Issue:** Original `total += len(word) + 1` overcounts by 1 (last word has no trailing space), causing `len(" ".join(selected))` to be `target_chars - 1` for single-word result
- **Fix:** Track `total = sum(len(w) for w in selected) + len(selected) - 1` — actual joined string length
- **Files modified:** src/generate_synthetic.py
- **Commit:** d755a93 (included in GREEN commit)

**2. Combined RED commit for Tasks 1 and 2**: Both tasks' tests live in the same file. Writing them in a single RED commit is cleaner than two separate RED commits; the RED→GREEN sequence is preserved in git log.

## TDD Gate Compliance

- RED gate: `test(06-02):` commit `f39aa37` — 12 failing tests
- GREEN gate: `feat(06-02):` commit `d755a93` — all 12 tests passing

## Threat Surface Scan

No new network endpoints, auth paths, or schema changes introduced. Pure in-memory functions only.

## Known Stubs

`main()` in `src/generate_synthetic.py` is a skeleton returning early ("Plan 03 rendering loop not yet implemented"). This is intentional — Plan 03 extends the function with TRDG rendering, font download, and ClearML logging. The stub is not blocking Plan 02's goal (pure function correctness).

## Self-Check: PASSED

- src/generate_synthetic.py: FOUND
- tests/test_generate_synthetic.py: FOUND
- 06-02-SUMMARY.md: FOUND
- commit f39aa37 (RED): FOUND
- commit d755a93 (GREEN): FOUND
