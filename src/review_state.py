"""review_state.py — filesystem-backed session position for the review app."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

DEFAULT_FILTER = "unlabeled"
VALID_FILTERS: tuple[str, ...] = ("unlabeled", "flagged", "labeled", "auto_labeled", "all")


@dataclass
class ReviewState:
    filter: str = DEFAULT_FILTER
    index: int = 0


def load_state(path: Path) -> ReviewState:
    """Load ReviewState from JSON. Missing or malformed files yield defaults."""
    if not path.exists():
        return ReviewState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ReviewState()
    filter_value = data.get("filter", DEFAULT_FILTER)
    if filter_value not in VALID_FILTERS:
        filter_value = DEFAULT_FILTER
    index_value = data.get("index", 0)
    if not isinstance(index_value, int) or index_value < 0:
        index_value = 0
    return ReviewState(filter=filter_value, index=index_value)


def save_state(path: Path, state: ReviewState) -> None:
    """Persist ReviewState as JSON. Creates parent directory if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state)), encoding="utf-8")


def with_filter(state: ReviewState, new_filter: str) -> ReviewState:
    """Return a new state with filter changed and index reset to 0 (D-03)."""
    if new_filter not in VALID_FILTERS:
        new_filter = DEFAULT_FILTER
    return replace(state, filter=new_filter, index=0)
