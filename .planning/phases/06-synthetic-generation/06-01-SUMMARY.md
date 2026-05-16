---
phase: 06-synthetic-generation
plan: 01
subsystem: infra
tags: [trdg, uv, pyproject, dependency-management, arabic-reshaper, python-bidi]

# Dependency graph
requires:
  - phase: 05-hyperparameter-tuning-system
    provides: baseline pyproject.toml with numpy==2.4.4, torch, opencv-python-headless
provides:
  - trdg==1.8.0 installed and importable without breaking numpy/torch/cv2
  - override-dependencies resolving arabic-reshaper and opencv-python-headless conflicts
  - generate-synthetic console script registered in pyproject.toml
  - assets/fonts/.gitkeep tracked; assets/fonts/*.ttf gitignored
affects: [06-02, 06-03]

# Tech tracking
tech-stack:
  added:
    - trdg==1.8.0
    - arabic-reshaper==3.0.1
    - python-bidi==0.6.10
  patterns:
    - "[tool.uv] override-dependencies to force resolver past yanked/conflicting transitive deps"
    - "assets/fonts/ pattern: .gitkeep tracked, binary .ttf gitignored, FONT_URLS in source"

key-files:
  created:
    - assets/fonts/.gitkeep
  modified:
    - pyproject.toml
    - .gitignore
    - uv.lock

key-decisions:
  - "override-dependencies pins arabic-reshaper==3.0.1 (trdg's 2.1.3 is yanked) and opencv-python-headless==4.13.0.92 (resolves cv2 namespace conflict)"
  - "python-bidi==0.6.10 override covers Pitfall 2: 0.5.1 compat shim for bidi.algorithm import path"
  - "assets/fonts/*.ttf gitignored (OFL binaries); FONT_URLS dict lives in source code"
  - "generate-synthetic console script registered now to keep all pyproject.toml edits in one plan"

patterns-established:
  - "Pattern: Import trdg via from trdg.generators.from_strings import GeneratorFromStrings to bypass wikipedia module-level import in __init__.py"

requirements-completed: [SYN-01]

# Metrics
duration: 2min
completed: 2026-05-16
---

# Phase 6 Plan 01: Synthetic Generation — TRDG Dependency Install Summary

**trdg==1.8.0 installed with uv override-dependencies, numpy==2.4.4 and torch intact, fonts directory scaffolded**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-05-16T09:49:48Z
- **Completed:** 2026-05-16T09:51:45Z
- **Tasks:** 2
- **Files modified:** 4 (pyproject.toml, .gitignore, assets/fonts/.gitkeep, uv.lock)

## Accomplishments

- Added trdg==1.8.0 + arabic-reshaper==3.0.1 + python-bidi==0.6.10 to project dependencies with exact `==` pins
- Added `[tool.uv] override-dependencies` to prevent trdg's yanked arabic-reshaper==2.1.3 from causing numpy downgrade (RESEARCH.md Pitfall 1 + 6)
- Verified: `uv sync` installs cleanly with numpy==2.4.4, torch==2.11.0+cpu, cv2==4.13.0 all intact
- Created assets/fonts/.gitkeep (tracked directory for lazy font downloads) with .ttf binaries gitignored
- Registered `generate-synthetic = "src.generate_synthetic:main"` console script (module created in Plan 03)

## Task Commits

1. **Task 1: Add TRDG dependency stack and override-dependencies** - `0dad05e` (feat)
2. **Task 2: Gitignore fonts, track assets/fonts/, verify uv sync** - `d6ac85e` (feat)

## Files Created/Modified

- `pyproject.toml` - Added trdg/arabic-reshaper/python-bidi deps, override-dependencies block, generate-synthetic script
- `.gitignore` - Added assets/fonts/*.ttf ignore rule with !.gitkeep exception
- `assets/fonts/.gitkeep` - Empty tracked file establishing the fonts directory
- `uv.lock` - Updated lock file with all 9 new packages including trdg==1.8.0

## Decisions Made

- Used `[tool.uv] override-dependencies` with 3 pins (arabic-reshaper, python-bidi, opencv-python-headless) to resolve all trdg transitive conflicts in a single mechanism
- Placed the `generate-synthetic` script entry in pyproject.toml now (Plan 01) to avoid cross-plan file contention with Plans 02-03 which also touch pyproject.toml

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - uv sync resolved without errors. Both opencv-python and opencv-python-headless appear installed (both at 4.13.0.92 providing the same cv2 namespace); this is the expected behavior when the override forces headless preference but trdg's metadata still declares opencv-python as a dep.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 02 can now import `from trdg.generators.from_strings import GeneratorFromStrings` and `from bidi import get_display` directly
- Plans 02-03 should use `assets/fonts/` as the default font cache directory (populated by lazy download in Plan 03)
- No blockers

---
*Phase: 06-synthetic-generation*
*Completed: 2026-05-16*
