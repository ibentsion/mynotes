---
phase: quick-260508-hqp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/tune.py
  - tests/test_tune.py
autonomous: true
requirements:
  - HQP-01  # Bypass argparse in _objective so ClearML cannot inject sweep-only args into train_ctc parser
  - HQP-02  # Smoke test: simulate ClearML agent sys.argv injection and verify _objective survives
  - HQP-03  # End-to-end smoke test: main() with --n_trials=1 and tiny manifest writes best_params.json

must_haves:
  truths:
    - "tune.py runs on the ClearML GPU agent without 'unrecognized arguments' errors when n_trials/n_startup_trials/n_warmup_steps are injected into sys.argv"
    - "_objective constructs train_ctc args without invoking argparse (so ClearML monkey-patches cannot interfere)"
    - "Existing tune.py tests still pass after the refactor"
    - "A new test simulates the ClearML agent failure mode by mutating sys.argv and confirms _objective returns successfully"
    - "A new fast end-to-end test exercises main() with n_trials=1 and writes best_params.json"
  artifacts:
    - path: "src/tune.py"
      provides: "Optuna HPO sweep orchestrator; _objective builds train args via direct Namespace, not argparse"
      contains: "argparse.Namespace("
    - path: "tests/test_tune.py"
      provides: "Adds test_objective_ignores_extra_sys_argv_args and test_e2e_one_trial_no_enqueue"
      contains: "test_objective_ignores_extra_sys_argv_args"
  key_links:
    - from: "src/tune.py::_objective"
      to: "src.train_ctc.run_training"
      via: "direct argparse.Namespace construction (no parser.parse_*)"
      pattern: "argparse\\.Namespace\\("
    - from: "tests/test_tune.py::test_objective_ignores_extra_sys_argv_args"
      to: "src.tune._objective"
      via: "monkeypatch sys.argv with tune-only flags, mock run_training and init_task"
      pattern: "sys\\.argv.*--n_trials"
---

<objective>
Fix the HPO sweep API error reported on the ClearML GPU agent:

```
tune.py: error: unrecognized arguments: --n_trials 20 --n_startup_trials 5 --n_warmup_steps 5
```

Root cause (per root_cause_analysis): ClearML monkey-patches both `parse_args` and `parse_known_args`. When the agent runs the orchestrator task, it injects all stored hyperparams into `sys.argv`. Inside `_objective`, the call `_build_train_parser().parse_known_args(cli)` is intercepted by ClearML, which tries to inject tune.py-only flags (`--n_trials`, `--n_startup_trials`, `--n_warmup_steps`) into `train_ctc`'s parser — which rejects them.

Fix: bypass argparse in `_objective` entirely by constructing an `argparse.Namespace` directly with all fields `run_training` reads. ClearML cannot intercept direct attribute assignment.

Add two fast smoke tests so this regression is caught locally before any future remote run.

Purpose: Restore the GPU sweep happy path so HPO can run on the agent.
Output: src/tune.py with _objective rewritten + 2 new tests in tests/test_tune.py.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@src/tune.py
@src/train_ctc.py
@tests/test_tune.py

<interfaces>
<!-- run_training reads ONLY these attributes from args. The replacement Namespace must populate them all. -->
<!-- Source: src/train_ctc.py::run_training and CropDataset/AugmentTransform usage -->

train_ctc.run_training reads from args:
- args.manifest          : Path           (from sweep_args.manifest)
- args.output_dir        : Path           (sweep_args.output_dir / f"trial_{trial.number}")
- args.epochs            : int            (from params)
- args.batch_size        : int            (from params)
- args.lr                : float          (from params)
- args.val_frac          : float          (default 0.2)
- args.min_labeled       : int            (from sweep_args.min_labeled — read by main(), not run_training, but kept for parity)
- args.num_workers       : int            (default 0)
- args.aug_copies        : int            (from params)
- args.rotation_max      : float          (from params)
- args.brightness_delta  : float          (from params)
- args.noise_sigma       : float          (from params)
- args.rnn_hidden        : int            (from params)
- args.num_layers        : int            (from params)
- args.dataset_id        : str | None     (from sweep_args.dataset_id)

train_ctc._build_parser defaults (for fields NOT set by params loop):
  val_frac=0.2, num_workers=0, params=None, enqueue=False, queue_name="gpu", dataset_id=None

_suggest_params returns these keys (PARAM_KEYS):
  lr, batch_size, epochs, rnn_hidden, num_layers,
  aug_copies, rotation_max, brightness_delta, noise_sigma
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Replace argparse with direct Namespace in _objective and remove unused import</name>
  <files>src/tune.py, tests/test_tune.py</files>
  <behavior>
    - test_objective_ignores_extra_sys_argv_args:
      - Save original sys.argv; monkeypatch.setattr(sys, "argv", ["tune.py", "--n_trials", "20", "--n_startup_trials", "5", "--n_warmup_steps", "5"]).
      - Build sweep_args via MagicMock with: manifest=Path("data/manifest.csv"), min_labeled=100, dataset_id=None, output_dir=Path(tmp_path / "out_argv").
      - Build trial via MagicMock(spec=optuna.Trial) with number=0; trial.suggest_float side_effect returns 0.001; trial.suggest_int returns 30; trial.suggest_categorical returns choices[0]; trial.should_prune returns False.
      - Patch src.tune.run_training to return 0.5; patch src.tune.init_task to return MagicMock.
      - Call _objective(trial, sweep_args); assert result == 0.5.
      - Importantly: with the OLD code (parse_known_args path) this test would raise SystemExit/argparse error because ClearML's offline mode still uses real argparse; with the NEW direct-Namespace code it must pass cleanly.
    - test_e2e_one_trial_no_enqueue:
      - Write tmp_path / "manifest.csv" with header "crop_path,label,status,page_id" and one labeled row (any path; run_training is mocked).
      - monkeypatch.setattr(sys, "argv", ["tune.py", "--manifest", str(manifest), "--n_trials", "1", "--min_labeled", "1", "--output_dir", str(tmp_path / "out")]).
      - Patch src.tune.run_training to return 0.5; patch src.tune.init_task to return MagicMock; do NOT patch optuna (let it run for 1 trial).
      - Call main(); assert it returns 0 and (tmp_path / "out" / "best_params.json") exists with PARAM_KEYS subset present in the JSON.
  </behavior>
  <action>
    Step 1 — Edit src/tune.py:

    1a. Remove the now-unused import line:
        `from src.train_ctc import _build_parser as _build_train_parser`
        Keep `from src.train_ctc import run_training` as-is.

    1b. Rewrite `_objective` to build the Namespace directly. The new function body
        (replacing lines ~119-142, keeping the same signature and same return/raise
        semantics including the `finally: trial_task.close()` and TrialPruned re-raise):

        ```python
        def _objective(trial: optuna.Trial, sweep_args: argparse.Namespace) -> float:
            params = _suggest_params(trial)

            # Bypass argparse entirely. ClearML monkey-patches both parse_args and
            # parse_known_args, and on the GPU agent it injects tune.py-only flags
            # (--n_trials, --n_startup_trials, --n_warmup_steps) into any parser
            # call inside this process — train_ctc's parser rejects them. Direct
            # Namespace construction cannot be intercepted.
            train_args = argparse.Namespace(
                manifest=sweep_args.manifest,
                output_dir=sweep_args.output_dir / f"trial_{trial.number}",
                val_frac=0.2,
                min_labeled=sweep_args.min_labeled,
                num_workers=0,
                params=None,
                enqueue=False,
                queue_name="gpu",
                dataset_id=sweep_args.dataset_id,
                # Populated from Optuna trial.suggest_*:
                lr=params["lr"],
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                rnn_hidden=params["rnn_hidden"],
                num_layers=params["num_layers"],
                aug_copies=params["aug_copies"],
                rotation_max=params["rotation_max"],
                brightness_delta=params["brightness_delta"],
                noise_sigma=params["noise_sigma"],
            )
            train_args.output_dir.mkdir(parents=True, exist_ok=True)

            trial_task = _init_trial_task(trial, train_args)
            _, on_epoch_end = _make_pruning_callback(trial)

            try:
                best_val_cer = run_training(train_args, on_epoch_end=on_epoch_end)
                return best_val_cer
            except optuna.TrialPruned:
                raise
            finally:
                trial_task.close()
        ```

        Notes:
        - Drop the `cli = [...]` list; no longer needed.
        - Drop the `for k, v in params.items(): setattr(...)` loop; values now go straight into the Namespace constructor.
        - Drop the parse_known_args call and its preceding comment (the comment is now stale).
        - Keep the `_init_trial_task(trial, train_args)` call — it still calls `task.connect(vars(train_args), name="hyperparams")` which works on plain Namespaces.

    Step 2 — Edit tests/test_tune.py:

    2a. Add test_objective_ignores_extra_sys_argv_args (uses monkeypatch + tmp_path).
        Place after test_per_trial_task_tagged_phase_5_and_hpo_trial. Use the existing
        autouse `_offline_clearml` fixture (no extra setup needed). Use
        `monkeypatch.setattr(sys, "argv", [...])` so sys.argv restores automatically.

    2b. Add test_e2e_one_trial_no_enqueue (uses monkeypatch + tmp_path). Place after
        test_objective_ignores_extra_sys_argv_args. Also use monkeypatch for sys.argv.

    Both tests follow the patterns already in the file (see
    test_orchestrator_task_tagged_phase_5_only and
    test_per_trial_task_tagged_phase_5_and_hpo_trial). Import names from src.tune at
    test scope, mirror existing `with patch("src.tune.run_training") ...` style.

    Self-checks before finishing:
    - `argparse.Namespace(` appears in src/tune.py.
    - `from src.train_ctc import _build_parser` is GONE from src/tune.py.
    - tests/test_tune.py contains both new test functions.
    - No new top-level imports in tune.py (argparse is already imported).
    - Existing tests (test_objective_pruning_callback_raises_trial_pruned,
      test_objective_closes_trial_task_on_failure, test_per_trial_task_tagged_*)
      continue to pass — they already set sweep_args.dataset_id=None, .manifest,
      .min_labeled, .output_dir explicitly, which is exactly what the new code reads.
  </action>
  <verify>
    <automated>uv run pytest tests/test_tune.py -q</automated>
  </verify>
  <done>
    - src/tune.py imports list no longer references _build_parser as _build_train_parser.
    - _objective builds train_args via argparse.Namespace(...) — no parse_args / parse_known_args calls inside _objective.
    - tests/test_tune.py contains test_objective_ignores_extra_sys_argv_args and test_e2e_one_trial_no_enqueue, both passing.
    - All previously passing tests in tests/test_tune.py still pass.
    - `uv run ruff check src/tune.py tests/test_tune.py` reports no warnings.
    - `uv run ty check src/tune.py tests/test_tune.py` reports no errors.
  </done>
</task>

</tasks>

<verification>
Run after task completes:

```bash
uv run pytest tests/test_tune.py -q
uv run ruff check src/tune.py tests/test_tune.py
uv run ruff format --check src/tune.py tests/test_tune.py
uv run ty check src/tune.py tests/test_tune.py
```

All four must pass with no warnings.

Manual confirmation (optional, user-driven): trigger an actual GPU agent run with
`uv run python -m src.tune --enqueue --queue_name gpu --n_trials 2 --dataset_id <id>`
and confirm the agent reaches the first trial without the "unrecognized arguments"
error. Not required for plan completion — automated tests cover the regression.
</verification>

<success_criteria>
- HPO sweep no longer raises `unrecognized arguments: --n_trials ...` on the agent (proven by test_objective_ignores_extra_sys_argv_args, which simulates the agent's sys.argv injection in-process).
- A 1-trial end-to-end run of main() succeeds locally and writes best_params.json with all PARAM_KEYS present.
- Existing 11 tests in test_tune.py still pass; net delta is +2 tests.
- No new dependencies, no new top-level imports.
- Lint and type checks clean.
</success_criteria>

<output>
After completion, create `.planning/quick/260508-hqp-fix-hpo-sweep-api-error-and-add-smoke-te/260508-hqp-SUMMARY.md`
documenting:
- What changed in src/tune.py (with before/after of _objective).
- The two new tests and what they prove.
- Confirmation that the ClearML monkey-patch interaction is now bypassed.
- Pointer to the next manual step (user triggers a real GPU sweep to confirm).
</output>
