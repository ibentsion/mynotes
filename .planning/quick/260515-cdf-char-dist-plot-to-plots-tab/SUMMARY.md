---
slug: cdf-char-dist-plot-to-plots-tab
date: 2026-05-15
status: complete
---

## What changed
`src/train_ctc.py:292` — `report_image=True` → `report_image=False` in `_report_char_distribution`.

## Why
`report_image=True` sends the figure to ClearML's "Debug Samples" tab. `False` sends it to "Plots".

## Commit
b8f9bb8
