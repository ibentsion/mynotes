"""Pytest plugin: record per-test durations and warn about slow tests at session start."""

import json
from pathlib import Path

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
