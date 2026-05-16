"""Tests for src/generate_synthetic.py — corpus, sampling, distribution, coverage."""

import unicodedata

import numpy as np
import pytest

from src.generate_synthetic import (
    build_char_count_distribution,
    build_word_corpus,
    check_coverage,
    sample_text,
)

# ---------------------------------------------------------------------------
# build_word_corpus
# ---------------------------------------------------------------------------


def test_build_word_corpus_extracts_and_dedups_nfc_words():
    labels = ["שלום עולם", "שלום"]
    words, weights = build_word_corpus(labels)
    # NFC-normalized unique words
    assert set(words) == {"שלום", "עולם"}
    assert len(words) == 2
    assert np.isclose(weights.sum(), 1.0)


def test_build_word_corpus_filters_non_hebrew_words():
    labels = ["שלום 123 hello עולם"]
    words, weights = build_word_corpus(labels)
    word_set = set(words)
    assert "שלום" in word_set
    assert "עולם" in word_set
    assert "123" not in word_set
    assert "hello" not in word_set


def test_build_word_corpus_rare_char_gets_higher_weight():
    # "א" appears many times; "ק" appears once only in "קר"
    # "שש" contains only common chars (ש appears often)
    labels = ["אא אא אא אא אא שש קר"]
    words, weights = build_word_corpus(labels)
    rare_word_idx = words.index("קר")
    common_word_idx = words.index("שש")
    assert weights[rare_word_idx] > weights[common_word_idx]


def test_build_word_corpus_merges_extra_words():
    labels = ["שלום עולם"]
    words, weights = build_word_corpus(labels, extra_words=["נדיר"])
    nfc_extra = unicodedata.normalize("NFC", "נדיר")
    assert nfc_extra in words
    assert np.isclose(weights.sum(), 1.0)


def test_build_word_corpus_empty_labels_raises():
    with pytest.raises(ValueError, match="(?i)no labeled|corpus"):
        build_word_corpus([])


def test_build_word_corpus_all_noise_raises():
    with pytest.raises(ValueError, match="(?i)no labeled|corpus"):
        build_word_corpus(["123 hello foo"])


# ---------------------------------------------------------------------------
# sample_text
# ---------------------------------------------------------------------------


def test_sample_text_reaches_target_length_and_is_rtl_safe():
    labels = ["שלום עולם", "בוקר טוב"]
    words, weights = build_word_corpus(labels)
    rng = np.random.default_rng(0)
    target = 5
    result = sample_text(words, weights, target_chars=target, rng=rng)
    # Returned string length >= target_chars
    assert len(result) >= target
    # All tokens in result are from the word pool
    for token in result.split():
        assert token in words


# ---------------------------------------------------------------------------
# build_char_count_distribution
# ---------------------------------------------------------------------------


def test_build_char_count_distribution_returns_nfc_lengths():
    labels = ["אב", "אבג"]
    dist = build_char_count_distribution(labels)
    np.testing.assert_array_equal(dist, np.array([2, 3]))


# ---------------------------------------------------------------------------
# check_coverage
# ---------------------------------------------------------------------------


def test_check_coverage_returns_chars_below_threshold():
    labels = ["אאא", "ב"]
    gaps = check_coverage(labels, min_char_count=2)
    assert gaps == {"ב": 1}


def test_check_coverage_nfc_normalizes_before_counting():
    # Decomposed form of 'é' (NFD) and its NFC equivalent — same character after normalization
    nfd_label = "café"  # e + combining accent
    nfc_label = "café"  # precomposed é
    # Both labels should produce count 1 for é (same after NFC)
    gaps_nfd = check_coverage([nfd_label], min_char_count=2)
    gaps_nfc = check_coverage([nfc_label], min_char_count=2)
    # Both should have é below threshold (count=1 < 2)
    assert "é" in gaps_nfd
    assert "é" in gaps_nfc
    # Should NOT double-count the combining mark
    assert "́" not in gaps_nfd


def test_check_coverage_empty_when_all_meet_threshold():
    labels = ["אאא", "בב"]
    gaps = check_coverage(labels, min_char_count=2)
    assert gaps == {}


def test_check_coverage_before_after_gap_shrinks():
    existing = ["אאא"]
    # ב appears 0 times in existing → below threshold
    gaps_before = check_coverage(existing, min_char_count=2)
    assert "ב" not in gaps_before  # ב not counted at all

    # Add synthetic labels that include ב twice
    synthetic = ["בב"]
    gaps_after = check_coverage(existing + synthetic, min_char_count=2)
    # ב now has count 2 — at threshold, so not in gap dict
    assert "ב" not in gaps_after
