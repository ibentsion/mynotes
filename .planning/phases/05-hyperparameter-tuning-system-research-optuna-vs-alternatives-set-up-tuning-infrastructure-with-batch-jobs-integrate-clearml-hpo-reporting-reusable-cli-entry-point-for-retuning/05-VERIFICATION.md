---
phase: 05-hyperparameter-tuning-system
verified: 2026-05-07T10:45:00Z
status: passed
score: 10/10 must-haves verified
gaps: []
human_verification:
  - test: "Run `tune-hpo --manifest data/manifest.csv --n_trials 3` against a real labeled dataset"
    expected: "3 ClearML tasks appear in the project tagged 'hpo-trial'; outputs/best_params.json is written with 9 PARAM_KEYS + metadata"
    why_human: "Requires a real labeled manifest; live ClearML connectivity not testable offline"
  - test: "Run `tune-hpo --manifest data/manifest.csv --n_trials 20 --enqueue` with a running GPU agent"
    expected: "Entire sweep dispatches to GPU queue; pruned trials are terminated early; winning config written to outputs/best_params.json"
    why_human: "Requires GPU agent online and real training runtime"
  - test: "Retrain with `train-ctc --manifest data/manifest.csv --params outputs/best_params.json`"
    expected: "ClearML run is tagged 'phase-5'; training uses the architecture/hyperparams from the JSON; CER is at least as good as baseline"
    why_human: "Requires real labeled manifest and end-to-end GPU training run"
---

# Phase 5: Hyperparameter Tuning System — Verification Report

**Phase Goal:** Standalone Optuna-driven hyperparameter sweep over training, architecture, and
augmentation parameters; each trial is a ClearML task on the GPU queue; winning config is
written to outputs/best_params.json and consumable by train_ctc.py via --params for
zero-friction retuning.

**Verified:** 2026-05-07T10:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CRNN can be instantiated with arbitrary rnn_hidden ∈ {128,256,512} and num_layers ∈ {1,2} | VERIFIED | `CRNN.__init__` at ctc_utils.py:298 accepts `rnn_hidden: int = 256, num_layers: int = 2`; `CRNN(10, rnn_hidden=512, num_layers=1)` produces `hidden_size=512, fc.in_features=1024` |
| 2 | fc layer output size scales correctly with rnn_hidden (rnn_hidden * 2 for BiLSTM) | VERIFIED | ctc_utils.py:314 — `self.fc = nn.Linear(rnn_hidden * 2, num_classes)` |
| 3 | Default `CRNN(num_classes=N)` is identical to the pre-Phase-5 hardcoded architecture | VERIFIED | Defaults `rnn_hidden=256, num_layers=2` replicate the old `nn.LSTM(128*8, 256, num_layers=2)` + `nn.Linear(512, …)`; test `test_crnn_default_kwargs_match_legacy_architecture` passes |
| 4 | optuna 4.8.0 is importable from the project venv | VERIFIED | `optuna.__version__ == '4.8.0'`; pinned in pyproject.toml:16 and uv.lock |
| 5 | `python -m src.train_ctc --rnn_hidden 128 --num_layers 1 …` constructs CRNN with those values | VERIFIED | train_ctc.py:214-218 — CRNN built with `rnn_hidden=args.rnn_hidden, num_layers=args.num_layers`; argparse flags at lines 74-88 with choices enforcement |
| 6 | `python -m src.train_ctc --params best_params.json …` loads JSON hyperparams BEFORE task.connect() | VERIFIED | `_apply_params_file(args)` (train_ctc.py:17-30) called at line 311 before `task.connect` at line 339; type-casting via `type(existing)(v)` per Pitfall 4 |
| 7 | External script (tune.py) can import `run_training(args, on_epoch_end=…)` and receive per-epoch val_cer via callback | VERIFIED | tune.py:26 `from src.train_ctc import run_training`; called at tune.py:135 in-process with `on_epoch_end=on_epoch_end`; callback at train_ctc.py:295-296 fires after each val pass |
| 8 | `python -m src.tune --manifest … --n_trials N` runs N trials, each producing a ClearML task tagged 'hpo-trial'; outputs/best_params.json is written | VERIFIED | tune.py:93 `tags=["phase-5", "hpo-trial"]`; `_write_best_params` at tune.py:162 writes all 9 PARAM_KEYS + metadata; 12 passing tests in test_tune.py |
| 9 | Optuna study direction is 'minimize' (lower CER is better) with MedianPruner (n_startup_trials=5, n_warmup_steps=5 by default) | VERIFIED | tune.py:190 `optuna.create_study(direction="minimize", pruner=pruner)`; tune.py:186-189 `MedianPruner(n_startup_trials=…, n_warmup_steps=…)` |
| 10 | `--enqueue` dispatches the entire sweep to the GPU agent; outputs/best_params.json is gitignored; `tune-hpo` console script works | VERIFIED | tune.py:183-184 `orch_task.execute_remotely(queue_name=args.queue_name)` before `study.optimize`; `.gitignore:16` covers `outputs/best_params.json` (confirmed with `git check-ignore -v`); `/home/ido/git/mynotes/.venv/bin/tune-hpo` exists after `uv sync` |

**Score:** 10/10 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/ctc_utils.py` | CRNN.__init__ with rnn_hidden and num_layers params | VERIFIED | Line 298: `def __init__(self, num_classes: int, rnn_hidden: int = 256, num_layers: int = 2)` |
| `pyproject.toml` | optuna==4.8.0 in dependencies + tune-hpo script | VERIFIED | Line 16: `"optuna==4.8.0"`; Line 42: `tune-hpo = "src.tune:main"` |
| `tests/test_ctc_utils.py` | 5 new CRNN parameterization tests | VERIFIED | Tests at lines 276, 283, 290, 296, 303 — all pass |
| `src/train_ctc.py` | CLI flags --rnn_hidden, --num_layers, --params + run_training() + _apply_params_file() | VERIFIED | Lines 74-88 (flags), 17-30 (_apply_params_file), 114-300 (run_training) |
| `tests/test_train_ctc.py` | 13 new tests (8 parser/JSON + 5 run_training contract) | VERIFIED | 30 total tests; new ones at lines 843-1172; 13 new tests confirmed |
| `src/tune.py` | HPO orchestrator CLI (min 100 lines) | VERIFIED | 206 lines; all required functions present |
| `tests/test_tune.py` | 12 tests covering all behavioral contracts (min 80 lines) | VERIFIED | 301 lines; 12 tests all pass |
| `.gitignore` | Explicit comment for outputs/best_params.json | VERIFIED | Line 18: comment per D-11; `outputs/*` on line 16 already covers it |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ctc_utils.py::CRNN.__init__` | `self.rnn = nn.LSTM(128*8, rnn_hidden, num_layers=num_layers, …)` | constructor arg plumbing | WIRED | ctc_utils.py:311-314 |
| `ctc_utils.py::CRNN.__init__` | `self.fc = nn.Linear(rnn_hidden * 2, num_classes)` | BiLSTM doubling | WIRED | ctc_utils.py:314 |
| `train_ctc.py CLI args` | `CRNN(num_classes=…, rnn_hidden=args.rnn_hidden, num_layers=args.num_layers)` | constructor argument plumbing | WIRED | train_ctc.py:214-218 |
| `train_ctc.py --params loading` | args mutated BEFORE task.connect(vars(args)) | load JSON, setattr with type-cast, then connect | WIRED | `_apply_params_file` at line 311, `task.connect` at line 339 |
| `train_ctc.py::run_training` | `on_epoch_end(epoch, val_cer)` called per epoch | optional Callable invoked at end of each val pass | WIRED | train_ctc.py:295-296 |
| `tune.py::objective` | `run_training(train_args, on_epoch_end=on_epoch_end)` | in-process call, NOT subprocess | WIRED | tune.py:25-26 (imports), tune.py:135 (call site) |
| `tune.py::objective on_epoch_end callback` | `trial.report(val_cer, epoch) + trial.should_prune()` | Optuna pruning API | WIRED | tune.py:108-110 |
| `tune.py::main` | `optuna.create_study(direction='minimize', pruner=MedianPruner(…))` | Optuna study factory | WIRED | tune.py:186-190 |
| `tune.py::main` | `outputs/best_params.json` | json.dumps(best.params + metadata) | WIRED | tune.py:162-171 (`_write_best_params`) |
| `tune.py per-trial init_task` | `tags=["phase-5", "hpo-trial"]` | init_task tags arg | WIRED | tune.py:93 |

---

### Data-Flow Trace (Level 4)

Not applicable — tune.py and train_ctc.py are training pipeline scripts, not UI rendering components with data source / render separation. The data flows are verified through key link verification above (callback propagates val_cer from training loop through Optuna trial reporting; best_params.json is populated from real Optuna study results).

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| tune.py imports all required names | `uv run python -c "from src.tune import _build_parser, _suggest_params, _objective, _write_best_params, _report_hpo_results, main, PARAM_KEYS; print('ok')"` | `ok` | PASS |
| tune.py --help lists all 8 flags | `uv run python -m src.tune --help` | All flags listed including n_startup_trials, n_warmup_steps | PASS |
| optuna 4.8.0 importable | `uv run python -c "import optuna; print(optuna.__version__)"` | `4.8.0` | PASS |
| CRNN rnn_hidden=512, num_layers=1 produces correct shapes | `uv run python -c "from src.ctc_utils import CRNN; m = CRNN(10, rnn_hidden=512, num_layers=1); print(m.rnn.hidden_size, m.fc.in_features)"` | `512 1024` | PASS |
| run_training signature has on_epoch_end param | `uv run python -c "from src.train_ctc import run_training; import inspect; print(list(inspect.signature(run_training).parameters))"` | `['args', 'on_epoch_end']` | PASS |
| train_ctc --help shows new CLI flags | `uv run python -m src.train_ctc --help \| grep -E "rnn_hidden\|num_layers\|params"` | All 3 flags present with choices enforcement | PASS |
| tune-hpo console script installed | `ls /home/ido/git/mynotes/.venv/bin/tune-hpo` | File exists | PASS |
| best_params.json gitignored | `git check-ignore -v outputs/best_params.json` | `.gitignore:16:outputs/*` — exit code 0 | PASS |
| 12 test_tune.py tests pass | `uv run pytest tests/test_tune.py -q` | 12 passed | PASS |
| 12 CRNN tests pass | `uv run pytest tests/test_ctc_utils.py -k "crnn" -q` | 12 passed | PASS |
| ruff clean on all modified files | `uv run ruff check src/tune.py src/train_ctc.py src/ctc_utils.py` | All checks passed | PASS |

---

### Requirements Coverage

The HPO-01 through HPO-12 requirements are declared in ROADMAP.md for Phase 5 but are not present in `.planning/REQUIREMENTS.md` (which covers v1 requirements through Phase 4 only). They are defined implicitly by the plan must_haves. Cross-reference by plan:

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| HPO-01 | 05-01 | CRNN architecture parameterized (rnn_hidden, num_layers) | SATISFIED | ctc_utils.py:298 — parameterized constructor with defaults |
| HPO-02 | 05-01 | optuna 4.8.0 dependency added | SATISFIED | pyproject.toml:16; importable from venv |
| HPO-03 | 05-02 | train_ctc.py exposes --rnn_hidden and --num_layers CLI flags | SATISFIED | train_ctc.py:74-87 with choices=[128,256,512] and [1,2] |
| HPO-04 | 05-02 | train_ctc.py --params best_params.json round-trip | SATISFIED | `_apply_params_file` at train_ctc.py:17-30; 8 tests covering round-trip and type casting |
| HPO-05 | 05-02 | Reusable run_training(args, on_epoch_end=…) helper | SATISFIED | train_ctc.py:114; callback wired at line 295-296; 5 contract tests |
| HPO-06 | 05-03 | tune.py CLI: --manifest, --n_trials, --enqueue, --queue_name, --output_dir, --min_labeled | SATISFIED | tune.py:41-72 — all flags present with documented defaults |
| HPO-07 | 05-03 | Optuna study with TPE sampler + MedianPruner; minimize CER | SATISFIED | tune.py:186-190 — `direction="minimize"`, `MedianPruner(n_startup_trials=5, n_warmup_steps=5)` |
| HPO-08 | 05-03 | Per-trial ClearML task, tagged ["phase-5","hpo-trial"] | SATISFIED | tune.py:89-96 (`_init_trial_task`), tags at line 93 |
| HPO-09 | 05-03 | outputs/best_params.json written on completion + stdout summary | SATISFIED | `_write_best_params` at tune.py:162-171; stdout at lines 199-200 |
| HPO-10 | 05-03 | Orchestrator ClearML task "hpo_sweep" with CER-per-trial scalar + trial table | SATISFIED | tune.py:181 (`hpo_sweep` task), `_report_hpo_results` at tune.py:143-159 |
| HPO-11 | 05-03 | outputs/best_params.json gitignored | SATISFIED | .gitignore covers via `outputs/*` (line 16) + explicit comment (line 18) |
| HPO-12 | 05-03 | `tune-hpo` console script entry point | SATISFIED | pyproject.toml:42; `.venv/bin/tune-hpo` exists |

**Note:** HPO-01 through HPO-12 are not in `.planning/REQUIREMENTS.md` (which was last updated at Phase 4). This is a documentation lag — the requirements exist in ROADMAP.md and the plan must_haves, but REQUIREMENTS.md was not updated to include the Phase 5 HPO requirements. This does not affect the implementation — all 12 requirements are verified in code.

**Secondary note:** ROADMAP.md still marks plan 05-03 as `[ ]` (incomplete), but `src/tune.py` and `tests/test_tune.py` fully exist and pass. This is a ROADMAP tracking lag from the execution agent — the implementation is complete.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| tests/test_train_ctc.py (pre-existing) | multiple | Patches `src.train_ctc.build_charset` / `split_units` / `CropDataset` which are deferred imports inside run_training — patches miss | Info (pre-existing) | 6 tests (`test_status_filter_keeps_only_labeled`, `test_charset_build_receives_labeled_labels`, `test_no_page_leakage_between_train_and_val`, `test_val_dataset_has_no_augment`, `test_dataset_id_calls_remap`, `test_aug_copies_zero_backward_compat`) are effectively testing the wrong patch target; pre-dates Phase 5 and documented in 05-02-SUMMARY.md |
| tests/test_train_ctc.py | — | `test_main_still_writes_checkpoint_via_helper` flakes when run alongside other training tests due to ClearML singleton collision | Info (pre-existing) | Passes in isolation; pre-existing flakiness documented in 05-03-SUMMARY.md |

No blockers or warnings introduced by Phase 5. Both items are pre-existing and documented.

---

### Human Verification Required

#### 1. Live HPO sweep (end-to-end)

**Test:** Run `tune-hpo --manifest data/manifest.csv --n_trials 3` against a real labeled dataset
**Expected:** 3 ClearML tasks appear in the project tagged `["phase-5", "hpo-trial"]`; `outputs/best_params.json` is written containing all 9 PARAM_KEYS plus `best_val_cer`, `trial_number`, `n_trials_run`; stdout shows `Best trial N: CER=X.XXXX` followed by the JSON
**Why human:** Requires a real labeled manifest (privacy-sensitive local data); live ClearML connectivity for task creation; cannot be mocked in an offline test

#### 2. GPU-agent dispatch sweep

**Test:** With a running ClearML GPU agent, run `tune-hpo --manifest data/manifest.csv --n_trials 20 --enqueue`
**Expected:** Entire sweep dispatches to GPU queue; MedianPruner terminates underperforming trials before all epochs complete; `outputs/best_params.json` written with the winning config
**Why human:** Requires GPU agent online and real training wall time

#### 3. End-to-end round-trip retrain

**Test:** Run `train-ctc --manifest data/manifest.csv --params outputs/best_params.json` after a sweep
**Expected:** ClearML run is tagged `"phase-5"`; training uses the architecture and hyperparams from the JSON file; CER is comparable to or better than the baseline Phase 4 run
**Why human:** Requires real labeled manifest, real best_params.json from a previous sweep, and GPU training runtime to measure CER

---

### Gaps Summary

No gaps found. All 10 observable truths are verified, all artifacts exist and are substantive, all key links are wired, all 12 HPO requirements are satisfied in code.

The only open items are:
1. REQUIREMENTS.md was not updated to include HPO-01..HPO-12 (documentation lag, not a code gap)
2. ROADMAP.md still marks plan 05-03 as incomplete (tracking lag from execution agent, not a code gap)
3. 6 pre-existing tests have broken patch targets due to Phase 4's deferred-import refactor (pre-existing, documented)
4. `test_main_still_writes_checkpoint_via_helper` is flaky under parallel execution due to ClearML singleton (pre-existing, documented)

---

_Verified: 2026-05-07T10:45:00Z_
_Verifier: Claude (gsd-verifier)_
