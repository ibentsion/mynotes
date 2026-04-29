# Deferred Items — Phase 03

## Pre-existing Lint/Type Warnings (out of scope for 03-01)

These issues existed before Phase 03 started. They are not caused by any Phase 03 changes.
They should be addressed in a separate cleanup task.

### ruff E501 / B905 / F401 / F841

- `src/backfill_page_paths.py:109,126,129` — lines exceeding 100 chars
- `src/cluster_sampler.py:97,105` — lines exceeding 100 chars
- `src/review_app.py:337` — `zip()` without `strict=` parameter (B905)
- `tests/test_cluster_sampler.py:3` — unused `pytest` import (F401)
- `tests/test_region_detector.py:53` — unused `stats` variable (F841)
- `tests/test_review_to_clearml.py:18,24,30` — lines exceeding 100 chars

### ty unresolved-attribute

- `src/cluster_sampler.py:44` — `_HOG.compute(resized).ravel()`: object type `Sequence[int | float]`
  has no attribute `ravel` (OpenCV HOG compute return type lacks proper stubs)

All above were present before the Phase 03-01 work began (verified by `git stash` check).
