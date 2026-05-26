---
slug: cdf-char-dist-plot-to-plots-tab
date: 2026-05-15
status: in-progress
---

# Fix char distribution plot appearing in Debug Samples instead of Plots tab

## Problem
`_report_char_distribution` calls `logger.report_matplotlib_figure(..., report_image=True)`.
`report_image=True` routes the figure to ClearML's "Debug Samples" tab.
It should appear under "Plots".

## Fix
Set `report_image=False` in the `report_matplotlib_figure` call at `src/train_ctc.py:292`.

## File
`src/train_ctc.py` — `_report_char_distribution` function
