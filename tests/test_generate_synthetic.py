"""Tests for src/generate_synthetic.py — corpus, sampling, distribution, coverage."""

import unicodedata
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.generate_synthetic import (
    build_char_count_distribution,
    build_word_corpus,
    check_coverage,
    ensure_fonts,
    render_crops,
    sample_text,
    write_manifest,
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


# ---------------------------------------------------------------------------
# Task 1: ensure_fonts, render_crops, write_manifest
# ---------------------------------------------------------------------------


def test_ensure_fonts_skips_existing(tmp_path: Path) -> None:
    """Pre-existing .ttf files must not trigger a network request."""
    # Create one fake .ttf in the fonts dir matching a name in FONT_URLS
    from src.generate_synthetic import FONT_URLS

    first_name = next(iter(FONT_URLS))
    (tmp_path / first_name).write_bytes(b"FAKE-TTF")

    # patch requests.get to fail immediately if called
    with patch("requests.get", side_effect=RuntimeError("should not download")):
        paths = ensure_fonts(tmp_path)

    # The existing file path must be in the returned list
    assert str(tmp_path / first_name) in paths


def test_ensure_fonts_downloads_missing(tmp_path: Path) -> None:
    """Missing .ttf files are downloaded and written to fonts_dir."""
    fake_content = b"FAKE-TTF-CONTENT"
    mock_resp = MagicMock()
    mock_resp.content = fake_content

    with patch("requests.get", return_value=mock_resp) as mock_get:
        paths = ensure_fonts(tmp_path)

    from src.generate_synthetic import FONT_URLS

    # requests.get called once per FONT_URLS entry
    assert mock_get.call_count == len(FONT_URLS)
    mock_resp.raise_for_status.assert_called()
    for name in FONT_URLS:
        dest = tmp_path / name
        assert dest.exists()
        assert dest.read_bytes() == fake_content
        assert str(dest) in paths


def test_render_crops_skips_none_images(tmp_path: Path) -> None:
    """render_crops must skip None images and keep generating until --count crops saved."""
    real_img = Image.new("L", (40, 64), 255)

    call_count = 0

    def fake_next(gen_instance: object) -> tuple[Image.Image | None, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, "x"  # first call: None — must be skipped
        return real_img, "שלום"

    out_crops = tmp_path / "crops"
    out_crops.mkdir()

    fake_gen = MagicMock()
    fake_gen.__next__ = fake_next

    fake_cls = MagicMock(return_value=fake_gen)

    with patch("src.generate_synthetic._GeneratorFromStrings", fake_cls):
        from src.generate_synthetic import render_crops

        texts = ["שלום", "עולם"]  # 2 texts — enough after 1 None skip
        rows = render_crops(texts, ["fakefont.ttf"], out_crops)

    # Exactly 2 PNGs saved (len(texts) minus the one None)
    saved_pngs = list(out_crops.glob("*.png"))
    assert len(saved_pngs) == len(rows)
    assert all(r[1] == "שלום" for r in rows)
    # None result is not counted — len(rows) == 1 (one text skipped due to None)
    assert len(rows) == 1


def test_written_manifest_has_exact_columns(tmp_path: Path) -> None:
    """manifest.csv must have exactly [crop_path, label, status] with status=labeled."""
    rows = [
        (str(tmp_path / "crops" / "syn_000001.png"), "שלום"),
        (str(tmp_path / "crops" / "syn_000002.png"), "עולם"),
    ]
    manifest_path = write_manifest(rows, tmp_path)
    df = pd.read_csv(manifest_path)

    assert list(df.columns) == ["crop_path", "label", "status"]
    assert (df["status"] == "labeled").all()
    assert len(df) == 2


def test_rendered_png_loads_as_grayscale_64px(tmp_path: Path) -> None:
    """A PNG saved by render_crops must load via load_crop as (1, 64, W)."""
    from src.ctc_utils import load_crop

    # Create a real 64px-tall grayscale image and save it
    img = Image.new("L", (80, 64), 128)
    png_path = tmp_path / "test_crop.png"
    img.save(str(png_path))

    tensor = load_crop(str(png_path))
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 64
    assert tensor.shape[2] >= 1
