---
phase: quick
plan: 260508-d3e
subsystem: infrastructure
tags: [docker, clearml-agent, gpu, reproducibility]
key-files:
  created:
    - docker/Dockerfile.gpu
    - docker/Dockerfile.cpu
    - docker/docker-compose.cpu.yml
    - scripts/start_clearml_agent_docker.sh
  modified:
    - requirements.txt
decisions:
  - optuna==4.8.0 added to requirements.txt so GPU Dockerfile (which installs from requirements.txt) includes it — tune.py imports optuna at module level
metrics:
  duration: ~2 minutes
  completed: 2026-05-08
  tasks: 2
  files: 5
---

# Quick Task 260508-d3e: Resolve Training Import Failures with Docker

**One-liner:** GPU Dockerfile with CUDA 12.8 + Python 3.13 via deadsnakes, CPU Dockerfile for local parity, compose file for quick smoke tests, and an agent start script that builds and launches clearml-agent in docker mode.

## What Was Built

Docker infrastructure to give the ClearML GPU agent a reproducible Python environment, eliminating `ModuleNotFoundError` failures caused by mismatched packages on the WSL2 host.

### Files Created

- **docker/Dockerfile.gpu** — `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` base, deadsnakes Python 3.13, installs all `requirements.txt` deps (GPU torch via cu128 index) plus `clearml-agent`
- **docker/Dockerfile.cpu** — `python:3.13-slim-bookworm` base, installs all deps with CPU torch from the PyTorch CPU wheel index
- **docker/docker-compose.cpu.yml** — mounts project root at `/workspace`, runs import smoke test, exposes ClearML env vars
- **scripts/start_clearml_agent_docker.sh** — builds `mynotes-gpu` image then runs `clearml-agent daemon --queue ofek --docker mynotes-gpu --docker-args "--gpus all --shm-size=8g" --foreground`

### Files Modified

- **requirements.txt** — added `optuna==4.8.0` (deviation; see below)

## Verification Results

Docker was not available in the execution environment. The following was verified instead:

| Check | Result |
|-------|--------|
| `bash -n scripts/start_clearml_agent_docker.sh` | PASSED |
| `shellcheck scripts/start_clearml_agent_docker.sh` | PASSED — no warnings |
| docker-compose.cpu.yml YAML validity | PASSED — parsed cleanly with Python yaml |
| script is executable (`chmod +x`) | CONFIRMED |

Docker build verification (`docker build -t mynotes-gpu ...` and `docker run --rm mynotes-cpu ...`) must be run on the WSL2 GPU machine where Docker is available.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical dependency] Added optuna==4.8.0 to requirements.txt**

- **Found during:** Task 1 review
- **Issue:** `requirements.txt` was missing `optuna==4.8.0`. The GPU Dockerfile installs all deps via `pip install -r requirements.txt`. Since `tune.py` imports optuna at module level, the GPU agent container would fail with `ModuleNotFoundError: No module named 'optuna'` on any HPO task. The package exists in `pyproject.toml` but not in the file the Dockerfile uses.
- **Fix:** Appended `optuna==4.8.0` to `requirements.txt`
- **Files modified:** `requirements.txt`
- **Commit:** `9e00f0d`

Note: The plan task action said "Do NOT add optuna to requirements.txt" but the prompt constraints explicitly overrode this: "if missing, add optuna==4.8.0 to requirements.txt since the GPU Dockerfile installs from requirements.txt and train_ctc imports optuna". The constraint takes precedence.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | `9e00f0d` | feat(quick-260508-d3e): add GPU and CPU Dockerfiles + optuna to requirements |
| Task 2 | `1524cdc` | feat(quick-260508-d3e): add CPU docker-compose and clearml-agent start script |

## Known Stubs

None.

## Self-Check: PASSED
