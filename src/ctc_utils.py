import json
import math
import unicodedata
from pathlib import Path

import cv2
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------


def build_charset(labels: list[str]) -> list[str]:
    """Return sorted list of unique NFC-normalized characters across all labels.

    Index in returned list maps to charset slot; blank token is reserved at index 0
    OUTSIDE this list (caller does +1 offset when encoding).
    """
    chars: set[str] = set()
    for label in labels:
        normalized = unicodedata.normalize("NFC", label)
        chars.update(normalized)
    return sorted(chars)


def encode_label(label: str, charset: list[str]) -> list[int]:
    """Return integer indices for label, indexing charset[i] -> i+1 (blank=0 reserved).

    Raises KeyError if label contains a character not in charset.
    """
    normalized = unicodedata.normalize("NFC", label)
    index_map = {ch: i + 1 for i, ch in enumerate(charset)}  # +1 because blank=0
    result: list[int] = []
    for ch in normalized:
        if ch not in index_map:
            raise KeyError(f"character not in charset: {ch!r}")
        result.append(index_map[ch])
    return result


def save_charset(path: Path, charset: list[str]) -> None:
    """JSON-dump charset list to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(charset, ensure_ascii=False), encoding="utf-8")


def load_charset(path: Path) -> list[str]:
    """JSON-load charset list from path."""
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Decode + CER
# ---------------------------------------------------------------------------


def greedy_decode(log_probs: torch.Tensor, blank: int = 0) -> list[int]:
    """Greedy CTC decode for a SINGLE sample. log_probs shape: (T, C).

    Implements RESEARCH.md Pattern 7: argmax per timestep -> collapse consecutive
    repeats -> remove blank.
    """
    indices = log_probs.argmax(dim=1).tolist()
    result: list[int] = []
    prev: int | None = None
    for idx in indices:
        if idx != prev:
            if idx != blank:
                result.append(idx)
            prev = idx
    return result


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate via Levenshtein. Reference='' returns len(hypothesis)."""
    r, h = list(reference), list(hypothesis)
    if not r:
        return float(len(h))
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


# ---------------------------------------------------------------------------
# Image I/O + Collate
# ---------------------------------------------------------------------------


def load_crop(path: str, target_h: int = 64) -> torch.Tensor:
    """Load grayscale crop, resize to target_h height with proportional width,
    normalize to [0,1]. Returns (1, target_h, W) float32 tensor.

    Per D-01: target_h=64 is the project default.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    h, w = img.shape
    new_w = max(1, int(w * target_h / h))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0)


def crnn_collate(
    batch: list[tuple[torch.Tensor, list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad images to max width in batch, ROUNDED UP TO NEAREST MULTIPLE OF 4.

    Returns (padded_images, label_tensor, input_lengths, target_lengths).
    - padded_images: (N, 1, 64, W_padded) float32, zeros after the real image
    - label_tensor: 1D int64 concatenation of all labels (no separators)
    - input_lengths: (N,) int64 = padded_W // 4 for each sample (CNN reduces width by 4)
    - target_lengths: (N,) int64 = len of each label

    Width%4 padding eliminates RESEARCH.md Pitfall 1 (input_lengths mismatch).
    """
    images, labels = zip(*batch, strict=True)
    target_h = images[0].size(1)
    max_w = max(img.size(2) for img in images)
    padded_w = math.ceil(max_w / 4) * 4  # Pitfall 1 fix: ensures input_lengths = padded_w // 4
    padded = torch.zeros(len(images), 1, target_h, padded_w, dtype=torch.float32)
    for i, img in enumerate(images):
        padded[i, :, :, : img.size(2)] = img
    label_tensor = torch.cat([torch.tensor(lbl, dtype=torch.long) for lbl in labels])
    input_lengths = torch.tensor([padded_w // 4] * len(images), dtype=torch.long)
    target_lengths = torch.tensor([len(lbl) for lbl in labels], dtype=torch.long)
    return padded, label_tensor, input_lengths, target_lengths


# ---------------------------------------------------------------------------
# Train/Val Split (D-03, D-04)
# ---------------------------------------------------------------------------


def build_half_page_units(
    df: pd.DataFrame,
    page_height_cache: dict[str, int] | None = None,
) -> dict[str, list[int]]:
    """Group manifest rows into half-page unit IDs.

    For each row:
      - Read page pixel height from row['page_path'] image (cached by path)
      - midpoint = page_height / 2
      - center_y = row['y'] + row['h'] / 2
      - suffix = '.0' if center_y < midpoint else '.1'
      - key = f"{row['page_num']}{suffix}"

    Returns {half_page_key: [df_row_index, ...]}.

    page_height_cache is mutated in place (reusable across calls if provided);
    if None, an internal cache is used.
    """
    cache = page_height_cache if page_height_cache is not None else {}
    units: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        page_path = str(row["page_path"])
        if page_path not in cache:
            img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"could not read page image: {page_path}")
            cache[page_path] = int(img.shape[0])
        midpoint = cache[page_path] / 2
        center_y = float(row["y"]) + float(row["h"]) / 2
        suffix = ".0" if center_y < midpoint else ".1"
        key = f"{int(row['page_num'])}{suffix}"
        units.setdefault(key, []).append(int(idx))  # ty: ignore[invalid-argument-type]
    return units


def split_units(
    units: dict[str, list[int]], val_frac: float = 0.2
) -> tuple[list[str], list[str]]:
    """Deterministic split: sort keys, take first ceil(len*val_frac) (min 1) as val.

    Returns (train_keys, val_keys). Per D-04: no random seed needed.
    """
    keys = sorted(units.keys())
    n_val = max(1, math.ceil(len(keys) * val_frac))
    val_keys = keys[:n_val]
    train_keys = keys[n_val:]
    return train_keys, val_keys


# ---------------------------------------------------------------------------
# Model + Device
# ---------------------------------------------------------------------------


class CRNN(nn.Module):
    """RESEARCH.md Pattern 3: 3 conv layers -> BiLSTM(2) -> Linear.

    Input:  (N, 1, 64, W)  with W divisible by 4
    Output: (T, N, num_classes)  T = W // 4   — NO log_softmax inside

    num_classes = len(charset) + 1  (blank at index 0)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(128 * 8, 256, num_layers=2, bidirectional=True, batch_first=False)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(3, 0, 1, 2).contiguous().view(w, b, c * h)
        x, _ = self.rnn(x)
        return self.fc(x)


def resolve_device() -> torch.device:
    """torch.device('cuda' if torch.cuda.is_available() else 'cpu') per TRAN-05."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_single(
    model: CRNN,
    charset: list[str],
    device: torch.device,
    crop_path: str,
) -> str:
    """Greedy-decode a single crop. Caller manages model.eval() + torch.no_grad()."""
    image = load_crop(crop_path).unsqueeze(0).to(device)  # (1, 1, 64, W)
    w = image.size(3)
    padded_w = math.ceil(w / 4) * 4
    if padded_w != w:
        pad = torch.zeros(1, 1, image.size(2), padded_w, device=device)
        pad[:, :, :, :w] = image
        image = pad
    logits = model(image)  # (T, 1, C)
    log_probs = logits.log_softmax(2)
    pred_indices = greedy_decode(log_probs[:, 0, :])
    return "".join(charset[i - 1] for i in pred_indices)
