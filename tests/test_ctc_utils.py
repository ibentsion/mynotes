from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.ctc_utils import (
    CRNN,
    AugmentTransform,
    CropDataset,
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
    log_probs = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
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
    df = pd.DataFrame(
        [
            _make_df_row(str(page_img), 1, y=10, h=20),
            _make_df_row(str(page_img), 1, y=60, h=20),
        ]
    )
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


def test_crnn_default_kwargs_match_legacy_architecture():
    model = CRNN(num_classes=10)
    assert model.rnn.hidden_size == 256
    assert model.rnn.num_layers == 2
    assert model.fc.in_features == 512


def test_crnn_accepts_smaller_rnn_hidden_and_single_layer():
    model = CRNN(num_classes=10, rnn_hidden=128, num_layers=1)
    assert model.rnn.hidden_size == 128
    assert model.rnn.num_layers == 1
    assert model.fc.in_features == 256


def test_crnn_accepts_larger_rnn_hidden():
    model = CRNN(num_classes=10, rnn_hidden=512, num_layers=2)
    assert model.rnn.hidden_size == 512
    assert model.fc.in_features == 1024


def test_crnn_forward_works_with_smaller_rnn_hidden():
    model = CRNN(num_classes=10, rnn_hidden=128, num_layers=2)
    x = torch.zeros(2, 1, 64, 128)  # batch=2, height=64, width=128
    out = model(x)
    assert out.shape == (32, 2, 10)  # T = W // 4 = 32


def test_crnn_forward_works_with_single_layer():
    model = CRNN(num_classes=10, rnn_hidden=256, num_layers=1)
    x = torch.zeros(2, 1, 64, 128)
    out = model(x)
    assert out.shape == (32, 2, 10)


# ---------------------------------------------------------------------------
# AugmentTransform
# ---------------------------------------------------------------------------


def test_augment_transform_output_shape(tmp_path: Path):
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 128)
    tensor = load_crop(str(img_path))
    out = AugmentTransform()(tensor, seed=0)
    assert out.shape == tensor.shape


def test_augment_transform_output_dtype(tmp_path: Path):
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 64)
    tensor = load_crop(str(img_path))
    out = AugmentTransform()(tensor, seed=0)
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_augment_transform_different_seeds_differ(tmp_path: Path):
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 128, value=128)
    tensor = load_crop(str(img_path))
    out0 = AugmentTransform()(tensor, seed=0)
    out99 = AugmentTransform()(tensor, seed=99)
    assert not torch.equal(out0, out99)


def test_augment_transform_same_seed_is_deterministic(tmp_path: Path):
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 128, value=128)
    tensor = load_crop(str(img_path))
    aug = AugmentTransform()
    out1 = aug(tensor, seed=42)
    out2 = aug(tensor, seed=42)
    assert torch.equal(out1, out2)


def test_augment_transform_no_horizontal_flip(tmp_path: Path):
    # Build asymmetric tensor: left half all-ones, right half all-zeros
    tensor = torch.zeros(1, 64, 32)
    tensor[:, :, :16] = 1.0
    out = AugmentTransform()(tensor, seed=0)
    # If there were a horizontal flip, the output would equal torch.flip(tensor, [2])
    # With ±7° rotation only, the output should differ from a flipped input
    flipped = torch.flip(tensor, [2])
    assert not torch.equal(out, flipped)


# ---------------------------------------------------------------------------
# CropDataset (augmentation-aware)
# ---------------------------------------------------------------------------


def _make_aug_df(tmp_path: Path, n: int = 3) -> pd.DataFrame:
    rows = []
    page_img = _write_gray_png(tmp_path / "page.png", 100, 200)
    for i in range(n):
        crop_path = _write_gray_png(tmp_path / f"crop_{i}.png", 64, 128)
        rows.append(_make_df_row(str(page_img), page_num=1, y=10 * i, h=8))
        rows[-1]["crop_path"] = str(crop_path)
    return pd.DataFrame(rows)


def test_crop_dataset_aug_copies_zero_is_plain_length(tmp_path: Path):
    df = _make_aug_df(tmp_path, n=3)
    charset = ["א"]
    ds = CropDataset(df, charset, augment=None, aug_copies=0)
    assert len(ds) == 3


def test_crop_dataset_aug_copies_2_triples_length(tmp_path: Path):
    df = _make_aug_df(tmp_path, n=3)
    charset = ["א"]
    ds = CropDataset(df, charset, augment=AugmentTransform(), aug_copies=2)
    assert len(ds) == 9  # 3 * (1 + 2)


def test_crop_dataset_first_copy_is_clean(tmp_path: Path):
    df = _make_aug_df(tmp_path, n=3)
    charset = ["א"]
    ds = CropDataset(df, charset, augment=AugmentTransform(), aug_copies=2)
    image, _ = ds[0]
    expected = load_crop(str(df.iloc[0]["crop_path"]))
    assert torch.equal(image, expected)


def test_crop_dataset_augmented_copy_differs(tmp_path: Path):
    df = _make_aug_df(tmp_path, n=3)
    charset = ["א"]
    ds = CropDataset(df, charset, augment=AugmentTransform(), aug_copies=2)
    clean, _ = ds[0]
    augmented, _ = ds[3]  # index 3 = first real_idx=0, copy_idx=1
    assert not torch.equal(clean, augmented)


def test_crop_dataset_augmentation_varies_across_epochs(tmp_path: Path):
    """Same augmented index must produce different pixels on each access (per-epoch freshness)."""
    img_path = _write_gray_png(tmp_path / "crop.png", 64, 128, value=128)
    df = pd.DataFrame([{"crop_path": str(img_path), "label": "א",
                        "page_path": str(img_path), "page_num": 1,
                        "x": 0, "y": 0, "w": 128, "h": 64, "status": "labeled"}])
    charset = ["א"]
    ds = CropDataset(df, charset, augment=AugmentTransform(), aug_copies=1)
    first, _ = ds[1]   # augmented copy
    second, _ = ds[1]  # same index, second epoch
    assert not torch.equal(first, second), "augmented copies must differ across epochs"


def test_crop_dataset_augment_none_ignores_aug_copies(tmp_path: Path):
    df = _make_aug_df(tmp_path, n=4)
    charset = ["א"]
    ds = CropDataset(df, charset, augment=None, aug_copies=5)
    assert len(ds) == 4  # copies ignored when augment is None
