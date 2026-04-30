---
phase: 03-training-evaluation
verified: 2026-04-28T00:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
human_verification:
  - test: "Run train_ctc.py on a real labeled manifest.csv and confirm ClearML task appears in UI with scalars and artifacts"
    expected: "ClearML project 'handwriting-hebrew-ocr' shows a 'train_baseline_ctc' task with train/val loss + val CER scalars and checkpoint+charset artifacts"
    why_human: "Requires a live ClearML server or inspection of offline task storage; cannot verify server-side logging programmatically without a running instance"
  - test: "Run evaluate.py after a real training run and confirm ClearML 'evaluate_model' task appears with CER scalar and eval_report.csv artifact"
    expected: "Task visible in ClearML UI with cer and exact_match scalars, eval_report artifact attached"
    why_human: "Same ClearML server dependency; cannot verify UI display or artifact upload to a real server programmatically"
---

# Phase 3: Training Evaluation Verification Report

**Phase Goal:** A CRNN+CTC model trains on labeled crops and reports CER on a held-out validation set
**Verified:** 2026-04-28
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | torch 2.11.0+cpu installs via uv pytorch-cpu index | VERIFIED | `uv run python -c "import torch; print(torch.__version__)"` prints `2.11.0+cpu`; pyproject.toml has `[[tool.uv.index]]` pytorch-cpu routing |
| 2 | src/ctc_utils.py exposes CRNN with Conv2d×3 → BiLSTM → Linear topology | VERIFIED | `CRNN` class at line 200 with `nn.LSTM(128*8, 256, num_layers=2, bidirectional=True)`; forward shape test passes: `(4,2,10)` for input `(2,1,64,16)` |
| 3 | ctc_utils exposes build_charset with NFC normalization | VERIFIED | `unicodedata.normalize("NFC", ...)` at lines 24 and 34; test `test_build_charset_normalizes_to_nfc` passes |
| 4 | ctc_utils exposes encode_label, save/load_charset, greedy_decode, cer, load_crop, crnn_collate, build_half_page_units, split_units, resolve_device | VERIFIED | All 13 public functions present and substantive; 30 tests green in 1.81s |
| 5 | train_ctc.py is a runnable CLI that trains CRNN+CTC on labeled crops | VERIFIED | `test_train_one_epoch_writes_checkpoint_and_charset` passes; writes checkpoint.pt + charset.json on 1-epoch run |
| 6 | train_ctc.py filters only status==labeled rows (TRAN-01) | VERIFIED | Line 70: `labeled = df[df["status"] == "labeled"]`; spy test `test_status_filter_keeps_only_labeled` confirms only labeled labels reach build_charset |
| 7 | train_ctc.py uses half-page split, no leakage (TRAN-03) | VERIFIED | `build_half_page_units` + `split_units` called at lines 93-94; `test_no_page_leakage_between_train_and_val` asserts train/val key sets are disjoint |
| 8 | train_ctc.py logs train loss, val loss, val CER per epoch and uploads artifacts (TRAN-08) | VERIFIED | `report_scalar` calls at lines 177-179; `upload_file_artifact` at lines 192-193; stdout contains `epoch=0` and `best_val_cer=` confirmed by smoke test |
| 9 | evaluate.py is a runnable CLI that loads checkpoint and runs greedy decode on val split (EVAL-01) | VERIFIED | `model.load_state_dict(torch.load(checkpoint_path, weights_only=True))` at lines 84-85; `greedy_decode(log_probs[:, 0, :])` at line 107; end-to-end smoke test passes |
| 10 | evaluate.py computes CER on validation set (EVAL-02) | VERIFIED | `cer_values = [cer(t, p) ...]` at line 128; aggregate CER logged to ClearML at line 134 |
| 11 | eval_report.csv has exactly columns image_path, target, prediction, is_exact in order (EVAL-03) | VERIFIED | `columns=["image_path", "target", "prediction", "is_exact"]` explicit at line 122; schema assertion in smoke test passes |
| 12 | evaluate.py logs CER + exact_match_rate to ClearML task evaluate_model and uploads artifact (EVAL-04) | VERIFIED | `init_task("handwriting-hebrew-ocr", "evaluate_model")` at line 42; `report_scalar` at lines 134-135; `upload_file_artifact(task, "eval_report", ...)` at line 136 |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | torch CPU-only dependency + pytorch-cpu index | VERIFIED | Contains `"torch==2.11.0"`, `[[tool.uv.index]] name="pytorch-cpu"`, `[tool.uv.sources]` routing; `torch.__version__ == "2.11.0+cpu"` confirmed at runtime |
| `src/ctc_utils.py` | Shared CRNN model + CTC utilities + charset I/O + half-page split + collate | VERIFIED | 229 lines, 13 public functions, 0 stubs; all functions substantive and tested |
| `tests/test_ctc_utils.py` | Unit tests for every public function in ctc_utils | VERIFIED | 30 tests covering charset (8), decode+CER (7), image I/O+collate (6), split (5), model+device (4); all pass |
| `src/train_ctc.py` | CLI training script wiring ctc_utils + ClearML | VERIFIED | 204 lines; `def main`, `def _build_parser`, `class CropDataset` all present; full training loop with CTCLoss, val CER, best-checkpoint save |
| `tests/test_train_ctc.py` | Unit + smoke tests for train_ctc | VERIFIED | 8 tests covering parser defaults, status filter, 4 guard exit codes, charset delegation, page-leakage check, 1-epoch smoke; all pass |
| `src/evaluate.py` | CLI evaluation script: load checkpoint, greedy decode, write eval_report.csv, log to ClearML | VERIFIED | 147 lines; `def main`, `def _build_parser` present; reproduces val split, pads crops to width%4, writes 4-column report |
| `tests/test_evaluate.py` | Unit + smoke tests for evaluate | VERIFIED | 7 tests covering parser defaults, 3 guard exit codes, end-to-end smoke (train then eval), exact_match_rate formula, split val_frac flow-through; all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `pyproject.toml` | pytorch-cpu wheel index | `[[tool.uv.index]]` + `[tool.uv.sources]` | WIRED | Pattern `download.pytorch.org/whl/cpu` confirmed; `2.11.0+cpu` installed |
| `src/ctc_utils.py` | `torch.nn.LSTM(bidirectional=True)` | CRNN class composition | WIRED | `bidirectional=True` at line 216 |
| `src/ctc_utils.py` | `unicodedata.normalize('NFC', ...)` | `build_charset` + `encode_label` | WIRED | At lines 24 and 34 |
| `tests/test_ctc_utils.py` | `src/ctc_utils.py` | Direct import of all 13 public functions | WIRED | `from src.ctc_utils import CRNN, build_charset, ...` at lines 9-22 |
| `src/train_ctc.py` | `src.ctc_utils.CRNN` | Import + `CRNN(num_classes=len(charset)+1)` | WIRED | Line 124: `model = CRNN(num_classes=len(charset) + 1).to(device)` |
| `src/train_ctc.py` | `torch.nn.CTCLoss` | `ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)` over `log_softmax(2)` | WIRED | Lines 126 and 140-141 |
| `src/train_ctc.py` | `src.clearml_utils.init_task + upload_file_artifact` | ClearML task lifecycle | WIRED | Lines 60, 192-193 |
| `src/train_ctc.py` | `task.connect(vars(args), name='hyperparams')` | Hyperparameter capture | WIRED | Line 86 |
| `src/train_ctc.py` | `checkpoint.pt` + `charset.json` | `torch.save(model.state_dict())` + `save_charset` on val-CER improvement | WIRED | Lines 90 and 183 |
| `src/evaluate.py` | `src.ctc_utils.CRNN + load_charset + greedy_decode + cer` | Direct import | WIRED | Lines 16-25 |
| `src/evaluate.py` | `checkpoint.pt` | `torch.load(path, weights_only=True)` + `model.load_state_dict` | WIRED | Lines 84-85 |
| `src/evaluate.py` | `eval_report.csv` | `pd.DataFrame(...).to_csv` with 4-column explicit order | WIRED | Lines 115-125 |
| `src/evaluate.py` | `src.clearml_utils.init_task + upload_file_artifact` | ClearML evaluate_model lifecycle | WIRED | Lines 42, 136 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `src/train_ctc.py` (training loop) | `labeled` DataFrame | `df[df["status"] == "labeled"]` from `pd.read_csv(args.manifest)` | Yes — CSV from disk; no static fallback | FLOWING |
| `src/train_ctc.py` (val CER) | `val_cer` | Per-sample `cer(tgt_text, pred_text)` where `pred_text` comes from `greedy_decode(log_probs[:, n, :])` on real model output | Yes — model forward pass on actual image tensors | FLOWING |
| `src/evaluate.py` (eval_report.csv) | `image_paths`, `targets`, `predictions` | `load_crop(row["crop_path"])` → model forward → `greedy_decode` | Yes — reads crop image files from disk, runs actual model inference | FLOWING |
| `src/evaluate.py` (val split) | `val_df` | `build_half_page_units(labeled)` reading page image heights from disk + `split_units` | Yes — reads page images to compute midpoints | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| CRNN forward shape `(T=4, N=2, C=10)` for input `(2,1,64,16)` | `uv run python -c "from src.ctc_utils import CRNN; import torch; m=CRNN(10); out=m(torch.zeros(2,1,64,16)); assert out.shape==(4,2,10)"` | Shape `(4, 2, 10)` confirmed | PASS |
| resolve_device returns cpu | `uv run python -c "from src.ctc_utils import resolve_device; import torch; assert resolve_device()==torch.device('cpu')"` | `cpu` | PASS |
| 30 ctc_utils tests pass | `uv run pytest tests/test_ctc_utils.py -q` | `30 passed in 1.81s` | PASS |
| 8 train_ctc tests pass | `uv run pytest tests/test_train_ctc.py -q` | `8 passed in 88.92s` | PASS |
| 7 evaluate tests pass | `uv run pytest tests/test_evaluate.py -q` | `7 passed in 104.52s` | PASS |
| ruff clean on phase 3 files | `uv run ruff check src/ctc_utils.py src/train_ctc.py src/evaluate.py` | All checks passed | PASS |
| ty clean on phase 3 files | `uv run ty check src/ctc_utils.py src/train_ctc.py src/evaluate.py` | All checks passed | PASS |
| torch CPU wheel installed | `uv run python -c "import torch; print(torch.__version__)"` | `2.11.0+cpu` | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| TRAN-01 | 03-02 | train_ctc.py loads only status=labeled crops | SATISFIED | Line 70: `df[df["status"] == "labeled"]`; spy test confirms only labeled labels reach build_charset |
| TRAN-02 | 03-01, 03-02 | Charset built dynamically with Unicode NFC normalization | SATISFIED | `build_charset` at ctc_utils.py:16-26 does NFC; `train_ctc.py` calls `build_charset(labeled["label"].tolist())` at line 89 |
| TRAN-03 | 03-01, 03-02 | Train/val split by page (no crop leakage) | SATISFIED | `build_half_page_units` + `split_units` implement half-page grouping; page-leakage test verifies disjoint key sets |
| TRAN-04 | 03-01, 03-02 | CRNN: CNN → BiLSTM → CTC loss | SATISFIED | `CRNN` class: 3 Conv2d + BiLSTM(2) + Linear; `CTCLoss(blank=0, zero_infinity=True)` over `log_softmax(2)` at train_ctc.py:126-141 |
| TRAN-05 | 03-01, 03-02 | Training on CPU via torch.device auto-detect | SATISFIED | `resolve_device()` at ctc_utils.py:227-229 returns cpu on this host; confirmed by test and runtime check |
| TRAN-06 | 03-02 | Best checkpoint + charset.json saved to disk | SATISFIED | `torch.save(model.state_dict(), checkpoint_path)` at line 183 on `val_cer < best_val_cer`; `save_charset` at line 90; smoke test confirms both files written |
| TRAN-07 | 03-02 | All hyperparameters as CLI flags, connected to ClearML | SATISFIED | `_build_parser` exposes --epochs, --batch_size, --lr, --val_frac, --min_labeled, --num_workers; `task.connect(vars(args), name="hyperparams")` at line 86 |
| TRAN-08 | 03-02 | ClearML task `train_baseline_ctc` logs scalars + uploads artifacts | SATISFIED | `report_scalar` at lines 177-179 (train loss, val loss, val CER); `upload_file_artifact` for checkpoint and charset at lines 192-193; task name confirmed at line 60 |
| EVAL-01 | 03-03 | evaluate.py runs inference on val set using saved checkpoint | SATISFIED | `torch.load(checkpoint_path, weights_only=True)` + `model.load_state_dict` + `greedy_decode` loop over val_df at lines 84-107 |
| EVAL-02 | 03-01, 03-03 | CER computed on validation set | SATISFIED | `cer_values = [cer(t, p) ...]` at line 128; `avg_cer` logged; `cer` function in ctc_utils verified by 4 unit tests |
| EVAL-03 | 03-03 | eval_report.csv: image_path, target, prediction, is_exact | SATISFIED | Explicit `columns=["image_path", "target", "prediction", "is_exact"]` at line 122; column order asserted in smoke test |
| EVAL-04 | 03-03 | ClearML task `evaluate_model` logs CER + exact_match, uploads eval_report.csv | SATISFIED | Task name at line 42; `report_scalar` at lines 134-135; `upload_file_artifact(task, "eval_report", report_path)` at line 136 |

**Note on REQUIREMENTS.md stale status:** The traceability table in `.planning/REQUIREMENTS.md` still shows EVAL-01, EVAL-03, EVAL-04 as "Pending" and their checkboxes are unchecked. This is a documentation maintenance issue — the implementations are complete and verified in code. REQUIREMENTS.md was last updated before Plan 03-03 landed. The file should be updated to mark these as complete.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/train_ctc.py` | 9 | `from clearml import Task  # noqa: F401` | Info | Intentional: required at module level for `@patch("src.train_ctc.Task")` test patchability; documented with `# noqa: F401` and explanation comment |
| `src/evaluate.py` | 12-13 | `from clearml import Task  # noqa: F401` | Info | Same as above — required for test patchability; documented |
| `src/ctc_utils.py` | 177 | `# ty: ignore[invalid-argument-type]` on `int(idx)` from `df.iterrows()` | Info | pandas types index as `Hashable` in stubs; runtime value is `int`; suppress is justified and documented |

No stubs, no placeholder returns, no empty implementations. All anti-patterns are intentional and documented suppressions.

### Human Verification Required

#### 1. ClearML Task: train_baseline_ctc

**Test:** Run `CLEARML_OFFLINE_MODE=0 uv run python -m src.train_ctc --manifest data/manifest.csv --output_dir outputs/model --epochs 5` against a real manifest with ≥10 labeled crops and a live ClearML server
**Expected:** ClearML project `handwriting-hebrew-ocr` shows task `train_baseline_ctc` with hyperparams connected, per-epoch scalars for train loss / val loss / val CER, and checkpoint + charset.json artifacts uploaded
**Why human:** Cannot verify ClearML server-side storage or UI display without a running ClearML instance; offline mode (CLEARML_OFFLINE_MODE=1) bypasses actual server calls

#### 2. ClearML Task: evaluate_model

**Test:** After a successful training run, run `uv run python -m src.evaluate --manifest data/manifest.csv --output_dir outputs/model`
**Expected:** ClearML task `evaluate_model` shows cer and exact_match scalars, eval_report.csv uploaded as artifact; stdout shows `cer=X.XXXX exact_match_rate=Y.YYYY`
**Why human:** Same ClearML server dependency

### Gaps Summary

No gaps. All 12 must-have truths are verified. All 12 requirement IDs (TRAN-01..08, EVAL-01..04) are satisfied by substantive, wired, data-flowing implementations. The full test suite (30 + 8 + 7 = 45 phase 3 tests, all green) exercises every requirement. The only pending item is a documentation update to REQUIREMENTS.md marking EVAL-01, EVAL-03, and EVAL-04 as complete.

---

_Verified: 2026-04-28_
_Verifier: Claude (gsd-verifier)_
