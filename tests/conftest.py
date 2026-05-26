"""Pytest plugin: record per-test durations and warn about slow tests at session start."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_DURATIONS_FILE = Path(".pytest_durations.json")
_SLOW_THRESHOLD = 10.0
_durations: dict[str, float] = {}


def pytest_configure(config):
    if not _DURATIONS_FILE.exists():
        return
    try:
        data: dict[str, float] = json.loads(_DURATIONS_FILE.read_text())
    except Exception:
        return
    slow = sorted([(k, v) for k, v in data.items() if v > _SLOW_THRESHOLD], key=lambda x: -x[1])
    if slow:
        print(f"\nSlow tests from last run (>{_SLOW_THRESHOLD:.0f}s):")
        for name, dur in slow:
            print(f"  {dur:6.1f}s  {name}")


@pytest.fixture(autouse=True)
def _no_config_yaml():
    """Prevent config.yaml at project root from affecting tests.

    Tests that call train_ctc.main() or tune.main() should not pick up dataset IDs
    or hyperparams from the developer's local config.yaml — each test constructs its
    own args via sys.argv. Tests that explicitly need config behavior patch load_config
    themselves and override this autouse fixture.
    """
    with (
        patch("src.train_ctc.load_config", return_value={}),
        patch("src.tune.load_config", return_value={}),
    ):
        yield


@pytest.fixture(autouse=True)
def _clearml_env_cleanup():
    """Remove CLEARML_PROC_MASTER_ID after each test.

    ClearML writes this env var when task.close() is called in-process. Subprocesses
    spawned afterwards inherit it and get Task.init() returning a StubObject, causing
    Task.current_task() to return None and downstream AttributeErrors.
    """
    yield
    os.environ.pop("CLEARML_PROC_MASTER_ID", None)


def pytest_runtest_logreport(report):
    if report.when == "call":
        _durations[report.nodeid] = report.duration


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    if not _durations:
        return
    try:
        existing: dict[str, float] = {}
        if _DURATIONS_FILE.exists():
            existing = json.loads(_DURATIONS_FILE.read_text())
        existing.update(_durations)
        _DURATIONS_FILE.write_text(json.dumps(existing, indent=2, sort_keys=True))
    except Exception:
        pass
