---
phase: 06-synthetic-generation
plan: 03
subsystem: synthetic-generation
tags: [trdg, tdd, clearml, manifest, hebrew-fonts, lazy-download, coverage-validation, cli]

# Dependency graph
requires:
  - phase: 06-synthetic-generation
    plan: 01
    provides: trdg==1.8.0 installed, generate-synthetic console script registered
  - phase: 06-synthetic-generation
    plan: 02
    provides: build_word_corpus, sample_text, build_char_count_distribution, check_coverage

provides:
  - FONT_URLS dict with 3 OFL Hebrew fonts (Gveret Levin, Frank Ruhl Libre Regular/Bold)
  - ensure_fonts: lazy download to assets/fonts; returns existing .ttf paths without download
  - render_crops: TRDG GeneratorFromStrings loop; skips None results; start_idx for sequential naming
  - write_manifest: exactly [crop_path, label, status=labeled] 3-column manifest CSV
  - _generate_until_count: one-text-at-a-time generation loop compensating for TRDG None results
  - main(): complete CLI — init_task→parse_args→connect order, exit codes 0/2/3/4
  - 23 TDD-verified tests covering all behaviors and ClearML order invariant
  - generate-synthetic CLI produces N PNG crops + manifest.csv compatible with CropDataset

affects: [07-pretraining]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "_GeneratorFromStrings module-level alias: TRDG deferred inside render_crops (avoids wikipedia import), patchable for tests"
    - "start_idx parameter for sequential PNG naming across one-at-a-time render_crops calls"
    - "_generate_until_count: compensates for TRDG None results by rendering one text at a time"
    - "ensure_fonts: any existing .ttf = skip all downloads (cached OR user-supplied override)"

key-files:
  created: []
  modified:
    - src/generate_synthetic.py
    - tests/test_generate_synthetic.py

key-decisions:
  - "_GeneratorFromStrings imported inside render_crops but aliased at module level for test patchability"
  - "render_crops takes start_idx param for sequential numbering; _generate_until_count calls one text at a time to avoid excess PNG files"
  - "ensure_fonts: any existing .ttf in fonts_dir = use as-is (no download); this handles both cached defaults and user --fonts_dir overrides"
  - "Coverage gap test uses deterministic render_crops patch rather than probabilistic TRDG mock"
  - "ClearML order test spies on _build_parser (proxy for parse_args) to avoid ArgumentParser.parse_args recursion"

patterns-established:
  - "Pattern: TRDG import inside function + module-level alias pattern for lazy import with test patchability"

requirements-completed: [SYN-01, SYN-02, SYN-03, SYN-04]

# Metrics
duration: 20min
completed: 2026-05-16
---

# Phase 6 Plan 03: Synthetic Generation — CLI Integration Summary

**Full generate-synthetic CLI: TRDG Hebrew rendering, lazy font download, 3-column manifest, ClearML logging, and coverage-gated exit codes (0/2/3/4) — 23 TDD-verified tests, CropDataset-compatible PNG output**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-05-16T10:07:42Z
- **Completed:** 2026-05-16T10:27:55Z
- **Tasks:** 2 (each with RED+GREEN TDD commits)
- **Files modified:** 2

## Accomplishments

- `FONT_URLS` dict with 3 OFL-licensed Hebrew handwriting fonts (Gveret Levin + Frank Ruhl Libre)
- `ensure_fonts`: idempotent lazy download; skips download for any pre-existing .ttf (both cached and user-supplied `--fonts_dir` override)
- `render_crops`: TRDG `GeneratorFromStrings` with RTL/grayscale/64px config per RESEARCH.md Pattern 2; `if img is None: continue` guard (Pitfall 3); `start_idx` for sequential PNG naming across multiple calls
- `write_manifest`: exactly `["crop_path", "label", "status"]` columns with `status="labeled"` — verified loadable by unmodified `CropDataset`
- `_generate_until_count`: one-text-at-a-time loop that compensates for TRDG None results without writing excess files
- `main()`: complete CLI with ClearML init→parse→connect order, lazy font download, rendering loop, 3-column manifest, per-char ClearML scalars, coverage-gated exit codes
- 23 TDD-verified tests; `ctc_utils.py`/`train_ctc.py` unmodified (Phase 6 boundary respected)

## Task Commits

1. **Task 1 RED: Failing tests for ensure_fonts, render_crops, write_manifest** - `4c1eebe` (test)
2. **Task 1 GREEN: implement ensure_fonts, render_crops, write_manifest** - `db1fd34` (feat)
3. **Task 2 RED: Failing tests for main() CLI** - `630aa7c` (test)
4. **Task 2 GREEN: implement main() CLI** - `4c5710d` (feat)

## Files Created/Modified

- `src/generate_synthetic.py` - Complete CLI: FONT_URLS, ensure_fonts, render_crops, write_manifest, helpers, main()
- `tests/test_generate_synthetic.py` - 23 tests (12 Plan 02 + 11 Plan 03); full TDD coverage

## Decisions Made

- **_GeneratorFromStrings module-level alias**: TRDG import inside `render_crops` (RESEARCH.md Pattern 1 — avoids transitive wikipedia import), but aliased at module level as `_GeneratorFromStrings = None` so tests can patch `src.generate_synthetic._GeneratorFromStrings`
- **start_idx parameter**: `render_crops` accepts `start_idx` for sequential zero-padded PNG naming when called one-text-at-a-time by `_generate_until_count`. Without this, every call overwrites `syn_000001.png`
- **ensure_fonts directory-level check**: Any `.ttf` file present in `fonts_dir` → skip all downloads. Covers both the "fonts already downloaded" case and the `--fonts_dir` user override (D-03)
- **Deterministic coverage gap test**: Patching `render_crops` directly (not `_GeneratorFromStrings`) to control which labels appear in synthetic crops, making `test_main_coverage_gap_returns_4` deterministic and independent of TRDG sampling behavior
- **ClearML order test spy approach**: Patches `_build_parser` (proxy for parse_args call) rather than `ArgumentParser.parse_args` (which causes infinite recursion via `patch.object`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed render_crops test asserting wrong label**
- **Found during:** Task 1 GREEN (first test run)
- **Issue:** `test_render_crops_skips_none_images` asserted `r[1] == "שלום"` but the text "שלום" caused a None result (skipped); the saved crop used text "עולם" as its label
- **Fix:** Updated test assertion to `rows[0][1] == "עולם"` — the input text is the label, not the TRDG-returned string
- **Files modified:** tests/test_generate_synthetic.py
- **Verification:** 17 tests pass after fix
- **Committed in:** db1fd34 (Task 1 GREEN commit)

**2. [Rule 1 - Bug] Fixed excess PNG files from batch generation**
- **Found during:** Task 2 GREEN (first test run — 25 PNGs written instead of 5)
- **Issue:** Original `_generate_until_count` generated a large batch and sliced the returned rows, but wrote all batch PNGs to disk. Test counted 25 PNGs on disk, expected 5
- **Fix:** Changed to one-text-at-a-time loop — each call to `render_crops` generates exactly 1 text, no excess files written
- **Files modified:** src/generate_synthetic.py
- **Verification:** test_main_happy_path_returns_0_and_writes_outputs passes with exactly N PNGs
- **Committed in:** 4c5710d (Task 2 GREEN commit)

**3. [Rule 1 - Bug] Fixed non-deterministic coverage gap test**
- **Found during:** Task 2 GREEN (coverage gap test returned 0 instead of 4)
- **Issue:** `test_main_coverage_gap_returns_4` relied on probabilistic TRDG behavior — `sample_text` could pick words containing 'ת' in synthetic crops, pushing count above threshold
- **Fix:** Patched `render_crops` directly to always return labels of "שש" only — ensuring 'ת' count stays at 1 (below min_char_count=3) regardless of sampling
- **Files modified:** tests/test_generate_synthetic.py
- **Verification:** Coverage gap test reliably returns 4
- **Committed in:** 4c5710d (Task 2 GREEN commit)

**4. [Rule 1 - Bug] Fixed ClearML order test recursion**
- **Found during:** Task 2 GREEN (RecursionError on `ArgumentParser.parse_args`)
- **Issue:** Patching `ArgumentParser.parse_args` and then calling `_ap.ArgumentParser.parse_args(self)` hit the patch again → infinite recursion
- **Fix:** Patched `_build_parser` instead (proxy for parse_args call) and tracked its call position in `call_order`
- **Files modified:** tests/test_generate_synthetic.py
- **Verification:** test_main_clearml_order_init_before_parse passes
- **Committed in:** 4c5710d (Task 2 GREEN commit)

---

**Total deviations:** 4 auto-fixed (all Rule 1 — test/implementation bugs caught during TDD GREEN phase)
**Impact on plan:** All fixes necessary for test correctness. No scope creep — no new behaviors added.

## TDD Gate Compliance

- Task 1 RED gate: `test(06-03):` commit `4c1eebe` — 5 failing tests (ImportError on missing functions)
- Task 1 GREEN gate: `feat(06-03):` commit `db1fd34` — 17 tests passing
- Task 2 RED gate: `test(06-03):` commit `630aa7c` — 6 new failing tests (stub main())
- Task 2 GREEN gate: `feat(06-03):` commit `4c5710d` — 23 tests passing

## Threat Surface Scan

| Flag | File | Description |
|------|------|-------------|
| threat_flag: network | src/generate_synthetic.py | ensure_fonts downloads from GitHub raw / Google Fonts CDN → assets/fonts/; mitigated per T-06-07/T-06-08 (HTTPS, raise_for_status) |
| threat_flag: file-write | src/generate_synthetic.py | PNG crops written to --output_dir/crops/; manifest to --output_dir/manifest.csv; consistent with T-06-10 threat register |

All surface items are covered by the plan's threat model (T-06-07 through T-06-12).

## Known Stubs

None — `main()` is fully implemented. The prior Plan 02 stub ("Plan 03 rendering loop not yet implemented") has been replaced with the complete implementation.

## Issues Encountered

None beyond the TDD bug-fix deviations documented above.

## User Setup Required

None — no external service configuration required. Fonts download lazily on first CLI invocation.

## Next Phase Readiness

- `generate-synthetic` CLI is complete and ready for use in Phase 7 (pre-training)
- Output manifest schema (`crop_path`, `label`, `status=labeled`) is compatible with `CropDataset` with zero changes
- ClearML task `generate_synthetic` logs args, per-char scalars, and manifest artifact per D-07/D-08
- `--fonts_dir` override allows offline use without internet access
- Phase 6 boundary respected: `ctc_utils.py` and `train_ctc.py` unmodified

## Self-Check: PASSED

- src/generate_synthetic.py: FOUND
- tests/test_generate_synthetic.py: FOUND
- 06-03-SUMMARY.md: FOUND
- commit 4c1eebe (Task 1 RED): FOUND
- commit db1fd34 (Task 1 GREEN): FOUND
- commit 630aa7c (Task 2 RED): FOUND
- commit 4c5710d (Task 2 GREEN): FOUND

---
*Phase: 06-synthetic-generation*
*Completed: 2026-05-16*
