#!/usr/bin/env python3
"""PreToolUse hook: warn before pytest runs if any tests are expected to take >10s."""

import json
import sys
from pathlib import Path

DURATIONS_FILE = Path(".pytest_durations.json")
SLOW_THRESHOLD = 10.0

tool_input = json.load(sys.stdin)
cmd = tool_input.get("command", "")

if "pytest" not in cmd:
    sys.exit(0)

if not DURATIONS_FILE.exists():
    sys.exit(0)

try:
    data: dict[str, float] = json.loads(DURATIONS_FILE.read_text())
except Exception:
    sys.exit(0)

slow = sorted([(k, v) for k, v in data.items() if v > SLOW_THRESHOLD], key=lambda x: -x[1])
if slow:
    print(f"Slow tests expected (>{SLOW_THRESHOLD:.0f}s based on last run):")
    for name, dur in slow:
        print(f"  {dur:6.1f}s  {name}")
