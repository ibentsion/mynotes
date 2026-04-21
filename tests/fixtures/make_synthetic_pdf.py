from pathlib import Path

from PIL import Image, ImageDraw


def make_synthetic_pdf(path: Path, pages: int = 1) -> Path:
    """Write a minimal grayscale PDF with dark rectangles on each page.

    The rectangles are sized so detect_regions + default thresholds yield >=2 crops per page,
    with at least one crop near the margin (so FLAG-04 triggers).
    """
    imgs: list[Image.Image] = []
    for _ in range(pages):
        img = Image.new("L", (1200, 1600), color=240)  # light gray page
        draw = ImageDraw.Draw(img)
        # Three dark blobs: center, center-right, near left margin
        draw.rectangle([400, 400, 800, 480], fill=30)
        draw.rectangle([400, 600, 900, 680], fill=30)
        draw.rectangle([10, 800, 150, 870], fill=30)  # near margin
        imgs.append(img)
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:] if len(imgs) > 1 else [],
        format="PDF",
        resolution=100.0,
    )
    return path
