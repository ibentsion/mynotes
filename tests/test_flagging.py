import numpy as np

from src.flagging import FLAG_NAMES, flag_region


def _diagonal_crop(size: int = 60) -> np.ndarray:
    img = np.full((size, size), 240, dtype=np.uint8)
    for i in range(10, 50):
        img[i - 2 : i + 2, i - 2 : i + 2] = 30  # thick diagonal line
    return img


def test_flag_angle_triggers_on_diagonal_crop():
    crop = _diagonal_crop()
    reasons = flag_region(crop, 100, 100, 60, 60, [(100, 100, 60, 60)], (500, 500))
    assert "angle" in reasons


def test_flag_angle_not_triggered_on_horizontal():
    crop = np.full((60, 60), 240, dtype=np.uint8)
    crop[28:32, 10:50] = 30
    reasons = flag_region(crop, 100, 100, 60, 60, [(100, 100, 60, 60)], (500, 500))
    assert "angle" not in reasons


def test_flag_overlap_triggers_when_boxes_intersect():
    crop = np.full((50, 50), 100, dtype=np.uint8)
    reasons = flag_region(
        crop, 10, 10, 50, 50, [(10, 10, 50, 50), (30, 30, 50, 50)], (500, 500)
    )
    assert "overlap" in reasons


def test_flag_overlap_not_triggered_when_disjoint():
    crop = np.full((20, 20), 100, dtype=np.uint8)
    reasons = flag_region(
        crop, 10, 10, 20, 20, [(10, 10, 20, 20), (100, 100, 20, 20)], (500, 500)
    )
    assert "overlap" not in reasons


def test_flag_overlap_ignores_self_box():
    crop = np.full((50, 50), 100, dtype=np.uint8)
    reasons = flag_region(crop, 10, 10, 50, 50, [(10, 10, 50, 50)], (500, 500))
    assert "overlap" not in reasons


def test_flag_size_aspect_triggers_when_area_below_min():
    crop = np.full((5, 5), 100, dtype=np.uint8)
    reasons = flag_region(crop, 10, 10, 5, 5, [(10, 10, 5, 5)], (500, 500))
    assert "size_aspect" in reasons


def test_flag_size_aspect_triggers_when_aspect_extreme():
    crop = np.full((5, 300), 100, dtype=np.uint8)
    reasons = flag_region(crop, 10, 10, 300, 5, [(10, 10, 300, 5)], (500, 500))
    assert "size_aspect" in reasons


def test_flag_margin_triggers_near_edge():
    crop = np.full((80, 80), 100, dtype=np.uint8)
    reasons = flag_region(
        crop, 5, 100, 80, 80, [(5, 100, 80, 80)], (500, 500), margin_px=30
    )
    assert "margin" in reasons


def test_flag_margin_not_triggered_in_center():
    crop = np.full((80, 80), 100, dtype=np.uint8)
    reasons = flag_region(
        crop, 200, 200, 80, 80, [(200, 200, 80, 80)], (500, 500), margin_px=30
    )
    assert "margin" not in reasons


def test_flag_faint_triggers_on_mostly_white_crop():
    crop = np.full((60, 60), 240, dtype=np.uint8)
    reasons = flag_region(crop, 200, 200, 60, 60, [(200, 200, 60, 60)], (500, 500))
    assert "faint" in reasons


def test_flag_faint_not_triggered_on_dark_crop():
    crop = np.full((60, 60), 50, dtype=np.uint8)
    reasons = flag_region(crop, 200, 200, 60, 60, [(200, 200, 60, 60)], (500, 500))
    assert "faint" not in reasons


def test_flag_no_flags_on_clean_region():
    crop = np.full((80, 120), 180, dtype=np.uint8)
    crop[30:50, 20:100] = 50  # horizontal ink band
    reasons = flag_region(crop, 200, 200, 120, 80, [(200, 200, 120, 80)], (500, 500))
    # clean region: centered, not small, horizontal, dark enough
    assert "margin" not in reasons
    assert "size_aspect" not in reasons
    assert "overlap" not in reasons


def test_flag_names_constant_exports_five_names():
    assert FLAG_NAMES == ("angle", "overlap", "size_aspect", "margin", "faint")
