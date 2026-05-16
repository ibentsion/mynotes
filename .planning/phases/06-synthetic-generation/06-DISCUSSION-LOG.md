# Phase 6: Synthetic Generation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-16
**Phase:** 06-synthetic-generation
**Areas discussed:** TRDG rendering style, Text generation strategy, ClearML integration

---

## TRDG Rendering Style

| Option | Description | Selected |
|--------|-------------|----------|
| Handwriting-like fonts + distortions | Hebrew handwriting/cursive fonts with skew, blur, noise | ✓ |
| Clean printed fonts only | Standard printed Hebrew fonts, no distortions | |
| You decide | Leave font/distortion choices to Claude | |

**User's choice:** Handwriting-like fonts + distortions
**Notes:** Closer to real data — better for generalization.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Plain white | Matches real crop preprocessing (grayscale, normalized) | ✓ |
| Gaussian noise background | Adds texture mimicking paper grain | |
| You decide | Let Claude pick | |

**User's choice:** Plain white background
**Notes:** Real crops come from CamScanner-denoised scans — white is the correct approximation.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Optional with bundled default | Bundle one font, --fonts_dir overrides | |
| Required — user always provides | No bundled font, must pass --fonts_dir | |
| Bundle font names, download on first use | (free text) | ✓ |

**User's choice:** Bundle font names, download upon first use
**Notes:** `--fonts_dir` overrides. No font stored in repo directly. Lazy download pattern.

---

## Text Generation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Word-level sampling | Extract words from labels, sample 1–N words per crop | ✓ |
| Character n-gram sampling | Sample character sequences directly | |
| You decide | Let Claude pick | |

**User's choice:** Word-level sampling
**Notes:** Natural Hebrew text; rare chars get weighted words rather than isolated character sequences.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Match real crop length distribution | Sample from empirical char-count distribution of existing labels | ✓ |
| Fixed range: 2–6 words | Simple, predictable | |
| You decide | Let Claude pick | |

**User's choice:** Match real crop length distribution
**Notes:** Synthetic crop widths statistically match real data.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Inverse-frequency weighting on words | Words with rarer chars get higher sampling weight | ✓ |
| Guarantee-based: targeted crops per rare char | Generate dedicated crops for chars below threshold | |
| You decide | Let Claude implement simpler approach | |

**User's choice:** Inverse-frequency weighting on words
**Notes:** Satisfies SYN-03 without targeted per-character generation passes.

---

## ClearML Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — log to ClearML | Init task, connect args, upload manifest as artifact | ✓ |
| No — pure local tool | No ClearML task | |
| Optional via --clearml flag | Opt-in | |

**User's choice:** Yes — log generation runs to ClearML
**Notes:** Follows established pattern from train_ctc.py, tune.py, etc.

---

| Option | Description | Selected |
|--------|-------------|----------|
| generate_synthetic | Follows short-name convention | ✓ |
| synthetic_data_generation | More descriptive but breaks convention | |
| You decide | | |

**User's choice:** `generate_synthetic`
**Notes:** Consistent with `data_prep`, `train_baseline_ctc`, `hpo_sweep`.

---

## Claude's Discretion

- Exact TRDG distortion parameter values (skew angle, blur radius, noise level)
- Font download mechanism and cache location
- Whether to log per-character count scalars to ClearML alongside manifest artifact
- Word-splitting strategy for corpus extraction (whether to include single-char words)

## Deferred Ideas

None — discussion stayed within phase scope.
