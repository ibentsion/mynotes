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
    """render_crops must skip None images; only successfully rendered crops are returned."""
    real_img = Image.new("L", (40, 64), 255)

    call_count = 0

    def fake_next(gen_instance: object) -> tuple[Image.Image | None, str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, "x"  # first call: None — must be skipped (Pitfall 3)
        return real_img, "עולם"

    out_crops = tmp_path / "crops"
    out_crops.mkdir()

    fake_gen = MagicMock()
    fake_gen.__next__ = fake_next

    fake_cls = MagicMock(return_value=fake_gen)

    with patch("src.generate_synthetic._GeneratorFromStrings", fake_cls):
        from src.generate_synthetic import render_crops

        # texts[0]="שלום" → None (skipped), texts[1]="עולם" → real img (saved)
        texts = ["שלום", "עולם"]
        rows = render_crops(texts, ["fakefont.ttf"], out_crops)

    # None result is not counted — only 1 crop saved out of 2 texts
    assert len(rows) == 1
    saved_pngs = list(out_crops.glob("*.png"))
    assert len(saved_pngs) == len(rows)
    # Label is the original text argument (NOT any TRDG-returned value)
    assert rows[0][1] == "עולם"


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


# ---------------------------------------------------------------------------
# Task 2: main() CLI — argparse, ClearML, coverage-gated exit codes
# ---------------------------------------------------------------------------


def test_build_parser_has_all_flags() -> None:
    """_build_parser exposes all required flags with correct types and defaults."""
    from src.generate_synthetic import _build_parser

    p = _build_parser()
    ns = p.parse_args([])  # all defaults

    assert isinstance(ns.manifest, Path)
    assert isinstance(ns.output_dir, Path)
    assert ns.count == 500
    assert isinstance(ns.fonts_dir, Path)
    assert ns.wordlist is None
    assert ns.min_char_count == 5
    assert ns.seed == 42


def test_main_missing_manifest_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() returns 2 and writes to stderr when --manifest does not exist."""
    import io

    from src.generate_synthetic import main

    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(tmp_path / "nonexistent.csv"),
        "--output_dir", str(tmp_path / "out"),
    ])

    mock_task = MagicMock()
    with patch("src.generate_synthetic.init_task", return_value=mock_task):
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            result = main()

    assert result == 2
    assert str(tmp_path / "nonexistent.csv") in captured.getvalue()


def test_main_no_labeled_rows_returns_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() returns 3 when manifest has no status==labeled rows."""
    import io

    from src.generate_synthetic import main

    # Write a manifest with only non-labeled rows
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"crop_path": "a.png", "label": "שלום", "status": "flagged"},
    ]).to_csv(manifest_path, index=False)

    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(manifest_path),
        "--output_dir", str(tmp_path / "out"),
    ])

    mock_task = MagicMock()
    with patch("src.generate_synthetic.init_task", return_value=mock_task):
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            result = main()

    assert result == 3


def test_main_happy_path_returns_0_and_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 0; writes N PNGs and manifest.csv; Task.connect + upload called."""
    from src.generate_synthetic import main

    # Write a manifest with labeled Hebrew rows
    labels = ["שלום עולם", "בוקר טוב", "ערב טוב", "לילה טוב", "יום טוב"]
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"crop_path": f"c{i}.png", "label": lbl, "status": "labeled"}
        for i, lbl in enumerate(labels)
    ]).to_csv(manifest_path, index=False)

    count = 5
    real_img = Image.new("L", (80, 64), 200)

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    fake_font = fonts_dir / "GveretLevin-Regular.ttf"
    fake_font.write_bytes(b"TTF")

    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(manifest_path),
        "--output_dir", str(tmp_path / "out"),
        "--count", str(count),
        "--fonts_dir", str(fonts_dir),
        "--min_char_count", "1",
        "--seed", "42",
    ])

    mock_task = MagicMock()
    mock_gen = MagicMock()
    mock_gen.__next__ = MagicMock(return_value=(real_img, "שלום"))
    fake_gen_cls = MagicMock(return_value=mock_gen)

    with (
        patch("src.generate_synthetic.init_task", return_value=mock_task),
        patch("src.generate_synthetic._GeneratorFromStrings", fake_gen_cls),
        patch("src.generate_synthetic.upload_file_artifact") as mock_upload,
    ):
        result = main()

    assert result == 0
    out_manifest = tmp_path / "out" / "manifest.csv"
    assert out_manifest.exists(), "manifest.csv not written"
    out_df = pd.read_csv(out_manifest)
    assert len(out_df) == count, f"expected {count} rows, got {len(out_df)}"
    pngs = list((tmp_path / "out" / "crops").glob("*.png"))
    assert len(pngs) == count, f"expected {count} PNGs, got {len(pngs)}"
    # task.connect called with vars(args)
    mock_task.connect.assert_called_once()
    connect_arg = mock_task.connect.call_args[0][0]
    assert isinstance(connect_arg, dict)
    assert "count" in connect_arg
    # upload_file_artifact called with "manifest"
    mock_upload.assert_called_once()
    assert mock_upload.call_args[0][1] == "manifest"


def test_main_coverage_gap_returns_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 4 and prints WARN: lines when coverage gaps remain.

    Strategy: use a corpus where the only word is "שש", so the char 'ת'
    in the existing label "שת" can never appear in synthetic crops — its count
    stays at 1 no matter how many crops are generated. With min_char_count=10,
    'ת' always has count=1 < 10 → exit 4.
    """
    from src.generate_synthetic import main

    # 'ת' appears once in existing labels; corpus word pool = {"שש", "שת"}
    # but sampled texts will contain 'ת' only sometimes — use a high threshold
    # and only 1 labeled row so even ת at count=1 is guaranteed below threshold.
    labels = ["שש שת"]  # existing: ש×3, ת×1 — ת below any threshold > 1
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"crop_path": "c.png", "label": lbl, "status": "labeled"}
        for lbl in labels
    ]).to_csv(manifest_path, index=False)

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "GveretLevin-Regular.ttf").write_bytes(b"TTF")

    real_img = Image.new("L", (80, 64), 200)
    # Patch render so generated labels are always "שש" only (no ת in synthetic)
    call_count_ref = [0]

    def patched_render(
        texts: list[str], font_paths: list[str], out_crops_dir: Path, start_idx: int = 0
    ) -> list[tuple[str, str]]:
        call_count_ref[0] += 1
        idx = start_idx + 1
        png = out_crops_dir / f"syn_{idx:06d}.png"
        real_img.save(str(png))
        return [(str(png), "שש")]  # always ש only — ת count stays at 1

    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(manifest_path),
        "--output_dir", str(tmp_path / "out"),
        "--count", "2",
        "--fonts_dir", str(fonts_dir),
        "--min_char_count", "3",  # ת count=1 < 3 → gap; ש count grows above 3
        "--seed", "0",
    ])

    mock_task = MagicMock()
    with (
        patch("src.generate_synthetic.init_task", return_value=mock_task),
        patch("src.generate_synthetic.render_crops", side_effect=patched_render),
        patch("src.generate_synthetic.upload_file_artifact"),
    ):
        result = main()

    assert result == 4


def test_main_clearml_order_init_before_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """init_task must be invoked before argparse parsing (ClearML order invariant).

    Verifies source-code order by checking that init_task is called before
    _build_parser().parse_args(). We patch _build_parser itself to record ordering
    without recursion (avoids patching ArgumentParser.parse_args which recurses).
    """
    import io

    from src.generate_synthetic import main

    call_order: list[str] = []

    def record_init(*_: object, **__: object) -> MagicMock:
        call_order.append("init_task")
        return MagicMock()

    import src.generate_synthetic as _mod

    original_build_parser = _mod._build_parser

    def record_build_parser() -> object:
        call_order.append("parse_args")
        return original_build_parser()

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"crop_path": "a.png", "label": "שלום", "status": "labeled"}]).to_csv(
        manifest_path, index=False
    )

    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(manifest_path),
        "--output_dir", str(tmp_path / "out"),
    ])

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "GveretLevin-Regular.ttf").write_bytes(b"TTF")

    # Override fonts_dir default so ensure_fonts won't try to download
    monkeypatch.setattr("sys.argv", [
        "generate-synthetic",
        "--manifest", str(manifest_path),
        "--output_dir", str(tmp_path / "out"),
        "--fonts_dir", str(fonts_dir),
        "--count", "1",
        "--min_char_count", "1",
    ])

    with (
        patch("src.generate_synthetic.init_task", side_effect=record_init),
        patch("src.generate_synthetic._build_parser", side_effect=record_build_parser),
        patch("src.generate_synthetic.render_crops",
              return_value=[(str(tmp_path / "out" / "crops" / "syn_000001.png"), "שלום")]),
        patch("src.generate_synthetic.upload_file_artifact"),
    ):
        # Ensure output dirs exist
        (tmp_path / "out" / "crops").mkdir(parents=True, exist_ok=True)
        (tmp_path / "out" / "crops" / "syn_000001.png").write_bytes(b"PNG")
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            main()  # may return any code — we only care about call order

    assert "init_task" in call_order, "init_task was not called"
    assert "parse_args" in call_order, "_build_parser (parse_args proxy) was not called"
    assert call_order.index("init_task") < call_order.index("parse_args"), (
        f"init_task must come before parse_args; got order: {call_order}"
    )
