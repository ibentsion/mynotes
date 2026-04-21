import numpy as np

from src.region_detector import detect_regions, preprocess_page


def test_preprocess_page_returns_binary_values_only():
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
    out = preprocess_page(gray)
    assert out.dtype == np.uint8
    assert out.shape == gray.shape
    assert set(np.unique(out).tolist()) <= {0, 255}


def test_preprocess_page_ink_survives_threshold():
    img = np.full((100, 100), 200, dtype=np.uint8)
    img[30:50, 30:50] = 30  # dark square = ink
    out = preprocess_page(img)
    # THRESH_BINARY_INV means ink (dark) -> 255
    assert (out[30:50, 30:50] == 255).sum() >= 300


def test_detect_regions_finds_two_components_when_two_blobs():
    img = np.zeros((200, 200), dtype=np.uint8)
    img[20:40, 20:60] = 255
    img[20:40, 140:180] = 255
    stats = detect_regions(img)
    assert stats.shape == (2, 5)
    assert (stats[:, 4] > 0).all()


def test_detect_regions_excludes_background():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:60, 30:60] = 255
    stats = detect_regions(img)
    assert stats.shape == (1, 5)


def test_detect_regions_columns_are_x_y_w_h_area():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:60, 30:60] = 255
    stats = detect_regions(img)
    x, y, w, h, area = stats[0]
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0
    assert area > 0


def test_detect_regions_respects_dilation_kernel():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:25, 40:60] = 255
    img[40:45, 40:60] = 255  # 15px vertical gap
    thin = detect_regions(img, dilation_kernel_w=1, dilation_kernel_h=1, dilation_iters=1)
    assert thin.shape[0] == 2
    thick = detect_regions(img, dilation_kernel_w=1, dilation_kernel_h=20, dilation_iters=5)
    assert thick.shape[0] == 1
