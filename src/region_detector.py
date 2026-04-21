import cv2
import numpy as np


def preprocess_page(
    gray: np.ndarray,
    *,
    clahe_clip: float = 2.0,
    clahe_tile: tuple[int, int] = (8, 8),
    blur_ksize: tuple[int, int] = (5, 5),
) -> np.ndarray:
    if gray.ndim != 2:
        raise ValueError(f"preprocess_page expects 2D grayscale; got shape {gray.shape}")
    if gray.dtype != np.uint8:
        raise ValueError(f"preprocess_page expects uint8; got {gray.dtype}")
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, blur_ksize, 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_regions(
    binary: np.ndarray,
    *,
    dilation_kernel_w: int = 15,
    dilation_kernel_h: int = 3,
    dilation_iters: int = 3,
) -> np.ndarray:
    if binary.ndim != 2:
        raise ValueError(f"detect_regions expects 2D binary; got shape {binary.shape}")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_kernel_w, dilation_kernel_h))
    dilated = cv2.dilate(binary, kernel, iterations=dilation_iters)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if num_labels <= 1:
        return np.empty((0, 5), dtype=np.int32)
    # stats cols: CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA; skip background (label 0)
    return stats[1:].astype(np.int32)
