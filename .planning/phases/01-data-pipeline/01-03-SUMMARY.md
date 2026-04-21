---
phase: 01-data-pipeline
plan: 03
subsystem: computer-vision
tags: [opencv, numpy, region-detection, heuristic-flagging, tdd]

# Dependency graph
requires:
  - phase: 01-01
    provides: project scaffolding, pyproject.toml with opencv-python-headless dependency
provides:
  - preprocess_page: CLAHE + GaussianBlur + Otsu binarization (DATA-02)
  - detect_regions: morphological dilation + connectedComponentsWithStats (DATA-03)
  - flag_region: five independent heuristic checks FLAG-01 through FLAG-05
  - FLAG_NAMES constant for downstream consumers
affects:
  - 01-04 (pipeline script that composes these modules with pdf2image and pandas)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure CV modules: no pandas/PDF knowledge; numpy arrays in, numpy arrays/lists out"
    - "TDD with synthetic numpy inputs — no real image fixtures needed"
    - "Dilation kernel dimensions exposed as kwargs for CLI tuning"
    - "minAreaRect angle corrected with -45 heuristic (if raw_angle < -45: -(90+angle))"

key-files:
  created:
    - src/region_detector.py
    - src/flagging.py
    - tests/test_region_detector.py
    - tests/test_flagging.py
  modified: []

key-decisions:
  - "Dilation kernel w/h exposed as kwargs (not hardcoded) so Plan 04 CLI can tune per dataset"
  - "Self-box excluded from overlap check via identity comparison on (x, y, w, h) tuple"
  - "minAreaRect angle convention corrected per Pitfall 4: if raw_angle < -45 then -(90+angle)"

patterns-established:
  - "Pattern: CV modules are pure functions (np.ndarray in, np.ndarray/list out)"
  - "Pattern: TDD with synthetic numpy inputs eliminates image fixture dependencies"

requirements-completed: [DATA-02, DATA-03, FLAG-01, FLAG-02, FLAG-03, FLAG-04, FLAG-05]

# Metrics
duration: 2min
completed: 2026-04-21
---

# Phase 1 Plan 3: CV Modules Summary

**OpenCV region extraction (CLAHE+Otsu+CCL) and five-check heuristic flagging implemented as pure numpy modules, covered by 19 TDD tests with synthetic inputs**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-21T14:10:19Z
- **Completed:** 2026-04-21T14:12:04Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `preprocess_page`: CLAHE contrast normalization + Gaussian blur + Otsu inverted binarization
- `detect_regions`: morphological dilation with configurable kernel + connectedComponentsWithStats, returns (N, 5) int32 array excluding background
- `flag_region`: five independent checks (angle via minAreaRect, overlap pairwise, size/aspect, margin proximity, faint via mean intensity), all thresholds as kwargs
- 6 tests for region_detector, 13 tests for flagging — all passing; ruff and ty clean

## Task Commits

1. **Task 1: Implement region_detector with tests** - `59feabe` (feat)
2. **Task 2: Implement flagging module with per-check tests** - `1e65a3b` (feat)

## Files Created/Modified

- `src/region_detector.py` - preprocess_page and detect_regions using cv2 primitives
- `src/flagging.py` - flag_region with FLAG-01 through FLAG-05 checks and FLAG_NAMES constant
- `tests/test_region_detector.py` - 6 tests: binary output, ink survival, 2-blob detection, background exclusion, column order, kernel tuning
- `tests/test_flagging.py` - 13 tests: each flag triggered/not-triggered independently, clean region, constant check

## Decisions Made

- Dilation kernel dimensions (`dilation_kernel_w`, `dilation_kernel_h`) exposed as kwargs per RESEARCH.md Open Question 1 and CLML-05 requirement
- Self-box exclusion in overlap check via tuple identity comparison — simple and correct for bounding-box deduplication
- minAreaRect angle correction applied as documented in Pitfall 4 to avoid false negatives on tilted crops

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 04 (prepare_data.py pipeline) can import `preprocess_page`, `detect_regions`, `flag_region`, and `FLAG_NAMES` directly
- All thresholds are kwargs, ready for argparse CLI wiring
- No image fixtures or external data needed to run tests

---
*Phase: 01-data-pipeline*
*Completed: 2026-04-21*

## Self-Check: PASSED

- src/region_detector.py: FOUND
- src/flagging.py: FOUND
- tests/test_region_detector.py: FOUND
- tests/test_flagging.py: FOUND
- .planning/phases/01-data-pipeline/01-03-SUMMARY.md: FOUND
- commit 59feabe: FOUND
- commit 1e65a3b: FOUND
