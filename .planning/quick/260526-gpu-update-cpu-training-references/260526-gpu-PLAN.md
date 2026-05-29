---
quick_id: 260526-gpu
slug: update-cpu-training-references
date: 2026-05-26
description: Update all CPU-only training references to reflect GPU training via ClearML agent (queue ofek)
---

# Quick Task 260526-gpu: Update CPU Training References

## Task

All project documentation still references CPU-only training (a now-obsolete constraint). The user exclusively trains on a remote GPU via ClearML agent (queue: ofek, RTX 5060, WSL2). Update all mentions to reflect the current setup.

## Files

1. `CLAUDE.md` — Constraints section: remove "CPU-only for MVP — model must train without CUDA"
2. `.planning/PROJECT.md` — Context block (CPU estimate lines), Constraints runtime line, Key Decisions table
3. `.planning/REQUIREMENTS.md` — TRAN-05 description, Out of Scope table row
4. `.planning/ROADMAP.md` — Phase 3 success criteria #1, plan descriptions for 03-01 and 03-02

## Tasks

### Task 1: Update CLAUDE.md constraints
- File: `CLAUDE.md`
- Action: Replace CPU-only runtime constraint with GPU via ClearML agent (queue: ofek)

### Task 2: Update PROJECT.md
- File: `.planning/PROJECT.md`
- Action: Remove CPU training estimate lines from Context; update Constraints runtime; update Key Decisions table

### Task 3: Update REQUIREMENTS.md
- File: `.planning/REQUIREMENTS.md`
- Action: Update TRAN-05; update Out of Scope row

### Task 4: Update ROADMAP.md
- File: `.planning/ROADMAP.md`
- Action: Update Phase 3 success criteria and plan descriptions
