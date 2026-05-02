import json

from src.review_state import (
    DEFAULT_FILTER,
    VALID_FILTERS,
    ReviewState,
    load_state,
    save_state,
    with_filter,
)


def test_load_state_missing_file_returns_defaults(tmp_path):
    state = load_state(tmp_path / "missing.json")
    assert state == ReviewState(filter="unlabeled", index=0)


def test_load_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    original = ReviewState(filter="flagged", index=42)
    save_state(path, original)
    loaded = load_state(path)
    assert loaded == original


def test_load_state_malformed_json_returns_defaults(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    state = load_state(path)
    assert state == ReviewState(filter="unlabeled", index=0)


def test_load_state_invalid_filter_falls_back_to_default(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"filter": "bogus", "index": 3}), encoding="utf-8")
    state = load_state(path)
    assert state.filter == DEFAULT_FILTER
    assert state.index == 3


def test_load_state_negative_index_clamped_to_zero(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"filter": "labeled", "index": -5}), encoding="utf-8")
    state = load_state(path)
    assert state.index == 0


def test_save_state_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "state.json"
    save_state(nested, ReviewState(filter="all", index=7))
    assert nested.exists()
    assert json.loads(nested.read_text()) == {"filter": "all", "index": 7}


def test_with_filter_resets_index_and_validates(tmp_path):
    s = ReviewState(filter="unlabeled", index=15)
    assert with_filter(s, "labeled") == ReviewState(filter="labeled", index=0)
    assert with_filter(s, "bogus") == ReviewState(filter=DEFAULT_FILTER, index=0)


def test_valid_filters_constant():
    assert VALID_FILTERS == ("unlabeled", "flagged", "labeled", "auto_labeled", "all")
