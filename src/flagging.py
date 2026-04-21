import cv2
import numpy as np

FLAG_NAMES = ("angle", "overlap", "size_aspect", "margin", "faint")


def flag_region(
    gray_crop: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    all_boxes: list[tuple[int, int, int, int]],
    page_shape: tuple[int, int],  # (height, width)
    *,
    angle_thresh: float = 15.0,
    min_area: int = 500,
    margin_px: int = 30,
    faint_thresh: int = 200,
    aspect_lo: float = 0.1,
    aspect_hi: float = 10.0,
) -> list[str]:
    reasons: list[str] = []
    page_h, page_w = page_shape

    # FLAG-01: angle (Pitfall 4: correct minAreaRect convention)
    coords = np.column_stack(np.where(gray_crop < 128))
    if len(coords) > 10:
        raw_angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + raw_angle) if raw_angle < -45 else -raw_angle
        if abs(angle) > angle_thresh:
            reasons.append("angle")

    # FLAG-02: overlap with any OTHER box
    self_box = (x, y, w, h)
    for ox, oy, ow, oh in all_boxes:
        if (ox, oy, ow, oh) == self_box:
            continue
        ix1, iy1 = max(x, ox), max(y, oy)
        ix2, iy2 = min(x + w, ox + ow), min(y + h, oy + oh)
        if ix1 < ix2 and iy1 < iy2:
            reasons.append("overlap")
            break

    # FLAG-03: size / aspect ratio
    area = w * h
    ratio = w / h if h > 0 else 0.0
    if area < min_area or ratio < aspect_lo or ratio > aspect_hi:
        reasons.append("size_aspect")

    # FLAG-04: margin proximity
    if (
        x < margin_px
        or y < margin_px
        or (page_w - x - w) < margin_px
        or (page_h - y - h) < margin_px
    ):
        reasons.append("margin")

    # FLAG-05: faint (high mean intensity = mostly white = little ink)
    if float(gray_crop.mean()) > faint_thresh:
        reasons.append("faint")

    return reasons
