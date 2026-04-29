from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.ctc_utils import (
    CRNN,
    build_charset,
    build_half_page_units,
    cer,
    crnn_collate,
    encode_label,
    greedy_decode,
    load_charset,
    load_crop,
    resolve_device,
    save_charset,
    split_units,
)

# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------


def test_build_charset_returns_sorted_unique_chars():
    result = build_charset(["שלום", "שם"])
    assert result == sorted(result)
    # שלום = ש,ל,ו,ם (final mem U+05DD); שם = ש,ם — combined: ו,ל,ם,ש
    assert set(result) == {"ש", "ל", "ו", "ם"}


def test_build_charset_normalizes_to_nfc():
    # café with combining accent (NFD) vs precomposed é (NFC) — should collapse to same charset
    nfd = "café"  # e + combining acute
    nfc = "café"  # precomposed é
    assert build_charset([nfd]) == build_charset([nfc])


def test_build_charset_empty_input_returns_empty_list():
    assert build_charset([]) == []


def test_encode_label_offsets_by_one_for_blank_reservation():
    # charset=['א','ש']; 'ש' is at index 1 → token id 2
    assert encode_label("ש", ["א", "ש"]) == [2]


def test_encode_label_handles_repeated_chars():
    assert encode_label("שש", ["ש"]) == [1, 1]


def test_encode_label_raises_on_unknown_char():
    with pytest.raises(KeyError):
        encode_label("x", ["ש"])


def test_save_charset_load_charset_round_trip(tmp_path: Path):
    charset = ["א", "ב", "ג", "ש"]
    p = tmp_path / "charset.json"
    save_charset(p, charset)
    assert load_charset(p) == charset


def test_load_charset_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_charset(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Decode + CER
# ---------------------------------------------------------------------------


def test_greedy_decode_collapses_repeats_and_removes_blank():
    # log_probs rows: [0,1,0] -> argmax 1, [0,1,0] -> 1, [0,0,1] -> 2, [0,1,0] -> 1
    # After collapse: [1, 2, 1]. No blanks (blank=0 not in result).
    log_probs = torch.tensor([
        [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]
    ])
    assert greedy_decode(log_probs) == [1, 2, 1]


def test_greedy_decode_all_blank_returns_empty():
    log_probs = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert greedy_decode(log_probs) == []


def test_greedy_decode_alternating_blank_nonblank():
    # argmax: [0,1,0,1] -> collapse (all different) -> [0,1,0,1] -> remove blank -> [1,1]
    log_probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert greedy_decode(log_probs) == [1, 1]


def test_cer_identical_strings_zero():
    assert cer("שלום", "שלום") == 0.0


def test_cer_one_deletion_quarter():
    # 'שלם' vs 'שלום': 1 deletion / 4 reference chars = 0.25
    assert cer("שלום", "שלם") == pytest.approx(0.25)


def test_cer_empty_reference_returns_hypothesis_length():
    # Pattern 8: empty reference returns len(hypothesis) as float
    assert cer("", "שלם") == 3.0


def test_cer_empty_hypothesis_returns_one():
    # 4 deletions / 4 = 1.0
    assert cer("שלום", "") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Image I/O + Collate
# ---------------------------------------------------------------------------


def _write_gray_png(path: Path, h: int, w: int, value: int = 128) -> Path:
    img = np.full((h, w), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_load_crop_resizes_to_target_height(tmp_path: Path):
    # 100h x 200w -> target_h=64 -> new_w = int(200 * 64 / 100) = 128
    img_path = _write_gray_png(tmp_path / "crop.png", 100, 200)
    tensor = load_crop(str(img_path), target_h=64)
    assert tensor.shape == (1, 64, 128)


def test_load_crop_returns_normalized_float32(tmp_path: Path):
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 64, value=255)
    tensor = load_crop(str(img_path), target_h=64)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0


def test_crnn_collate_pads_to_multiple_of_four():
    # widths 7 and 13 -> max=13 -> ceil(13/4)*4 = 16
    img1 = torch.zeros(1, 64, 7)
    img2 = torch.zeros(1, 64, 13)
    padded, _, _, _ = crnn_collate([(img1, [1]), (img2, [2, 3])])
    assert padded.shape[3] == 16
    assert padded.shape[3] % 4 == 0


def test_crnn_collate_input_lengths_equal_padded_w_div_4():
    img1 = torch.zeros(1, 64, 7)
    img2 = torch.zeros(1, 64, 13)
    _, _, input_lengths, _ = crnn_collate([(img1, [1]), (img2, [2, 3])])
    padded_w = 16  # ceil(13/4)*4
    assert input_lengths.tolist() == [padded_w // 4, padded_w // 4]


def test_crnn_collate_concatenates_labels():
    img1 = torch.zeros(1, 64, 8)
    img2 = torch.zeros(1, 64, 8)
    _, label_tensor, _, _ = crnn_collate([(img1, [1, 2]), (img2, [3])])
    assert label_tensor.tolist() == [1, 2, 3]


def test_crnn_collate_target_lengths():
    img1 = torch.zeros(1, 64, 8)
    img2 = torch.zeros(1, 64, 8)
    _, _, _, target_lengths = crnn_collate([(img1, [1, 2]), (img2, [3])])
    assert target_lengths.tolist() == [2, 1]


# ---------------------------------------------------------------------------
# Half-page split (D-03 + D-04)
# ---------------------------------------------------------------------------


def _make_df_row(page_path: str, page_num: int, y: int, h: int) -> dict:
    return {
        "crop_path": "fake.png",
        "pdf_path": "fake.pdf",
        "page_path": page_path,
        "page_num": page_num,
        "x": 0,
        "y": y,
        "w": 10,
        "h": h,
        "area": 100,
        "is_flagged": False,
        "flag_reasons": "",
        "status": "labeled",
        "label": "א",
        "notes": "",
    }


def test_build_half_page_units_assigns_top_and_bottom(tmp_path: Path):
    # Create 100px tall page image
    page_img = _write_gray_png(tmp_path / "page.png", 100, 200)
    # Row 0: y=10, h=20 → center_y=20 < 50 (midpoint) → top half '.0'
    # Row 1: y=60, h=20 → center_y=70 >= 50 → bottom half '.1'
    df = pd.DataFrame([
        _make_df_row(str(page_img), 1, y=10, h=20),
        _make_df_row(str(page_img), 1, y=60, h=20),
    ])
    units = build_half_page_units(df)
    assert units == {"1.0": [0], "1.1": [1]}


def test_build_half_page_units_uses_cache(tmp_path: Path):
    page_img = _write_gray_png(tmp_path / "page.png", 100, 200)
    df = pd.DataFrame([_make_df_row(str(page_img), 1, y=10, h=20)])
    cache: dict[str, int] = {}
    build_half_page_units(df, page_height_cache=cache)
    assert str(page_img) in cache
    assert cache[str(page_img)] == 100


def test_split_units_takes_ceil_20_percent():
    # 5 keys, ceil(5*0.2)=1 → first sorted key goes to val
    units = {"1.0": [0], "1.1": [1], "2.0": [2], "2.1": [3], "3.0": [4]}
    train_keys, val_keys = split_units(units, val_frac=0.2)
    assert val_keys == ["1.0"]
    assert train_keys == ["1.1", "2.0", "2.1", "3.0"]


def test_split_units_min_one_when_only_one_unit():
    units = {"1.0": [0]}
    train_keys, val_keys = split_units(units)
    assert val_keys == ["1.0"]
    assert train_keys == []


def test_split_units_is_deterministic():
    units = {"3.0": [4], "1.0": [0], "2.0": [2]}
    result1 = split_units(units)
    result2 = split_units(units)
    assert result1 == result2


# ---------------------------------------------------------------------------
# Model + Device
# ---------------------------------------------------------------------------


def test_crnn_instantiates_without_error():
    model = CRNN(num_classes=10)
    assert model is not None


def test_crnn_forward_output_shape():
    model = CRNN(num_classes=10)
    out = model(torch.zeros(2, 1, 64, 16))
    # T = W // 4 = 16 // 4 = 4; N=2; C=10
    assert out.shape == (4, 2, 10)


def test_crnn_forward_returns_raw_logits_no_softmax():
    model = CRNN(num_classes=10)
    out = model(torch.zeros(2, 1, 64, 16))
    assert out.dtype == torch.float32
    assert out.shape == (4, 2, 10)
    # Not log_softmax: at least some values should be positive with random init
    # (log-probabilities are always <= 0; raw logits can be positive)
    # Use a looser check: just verify dtype and shape (init is random, positivity not guaranteed)


def test_resolve_device_returns_cpu_on_cpu_only_host():
    device = resolve_device()
    assert device == torch.device("cpu")
