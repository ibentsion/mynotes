"""cluster_sampler.py — reorder review_queue.csv by visual diversity clusters.

Extracts HOG features from unlabeled crops, clusters with KMeans, and rewrites
review_queue.csv so one representative per cluster comes first. Label those
first to maximize information per annotation.

Run: uv run python src/cluster_sampler.py --manifest outputs/manifest.csv --n_clusters 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Entry points run without the project root in sys.path; re-insert it so src.* imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from src.manifest_schema import MANIFEST_COLUMNS

_WIN = (64, 64)
_HOG = cv2.HOGDescriptor(
    _winSize=_WIN,
    _blockSize=(16, 16),
    _blockStride=(8, 8),
    _cellSize=(8, 8),
    _nbins=9,
)
_FEAT_DIM: int = _HOG.getDescriptorSize()


def extract_hog_features(paths: list[str]) -> np.ndarray:
    feats = np.zeros((len(paths), _FEAT_DIM), dtype=np.float32)
    for i, p in enumerate(paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        resized = cv2.resize(img, _WIN)
        feats[i] = _HOG.compute(resized).ravel()
    return feats


def _medoid_indices(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> list[int]:
    medoids: list[int] = []
    for c in range(n_clusters):
        member_mask = labels == c
        if not member_mask.any():
            continue
        cluster_feats = features[member_mask]
        centroid = cluster_feats.mean(axis=0)
        local_best = int(np.linalg.norm(cluster_feats - centroid, axis=1).argmin())
        medoids.append(int(np.where(member_mask)[0][local_best]))
    return medoids


def build_priority_queue(manifest_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Return manifest rows reordered: cluster medoids first, then other unlabeled, then rest."""
    unlabeled = manifest_df[manifest_df["status"] == "unlabeled"].reset_index(drop=True)
    rest = manifest_df[manifest_df["status"] != "unlabeled"]

    if len(unlabeled) == 0:
        return manifest_df.reset_index(drop=True)

    actual_k = min(n_clusters, len(unlabeled))
    features = extract_hog_features(unlabeled["crop_path"].tolist())
    labels = MiniBatchKMeans(n_clusters=actual_k, random_state=42, n_init=3).fit_predict(features)

    medoid_idx = _medoid_indices(features, labels, actual_k)
    non_medoid_idx = sorted(set(range(len(unlabeled))) - set(medoid_idx))

    return pd.concat(
        [unlabeled.iloc[medoid_idx], unlabeled.iloc[non_medoid_idx], rest],
        ignore_index=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Reorder review queue by visual diversity.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=200,
        help="Number of visual clusters (default: 200)",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest_df = pd.read_csv(args.manifest, dtype={c: object for c in ("label", "notes", "flag_reasons")})
    queue_df = build_priority_queue(manifest_df, args.n_clusters)

    out_path = args.manifest.with_name("review_queue.csv")
    queue_df[MANIFEST_COLUMNS].to_csv(out_path, index=False)

    n_unlabeled = int((manifest_df["status"] == "unlabeled").sum())
    n_reps = min(args.n_clusters, n_unlabeled)
    print(f"Done. {n_reps} cluster representatives leading {n_unlabeled} unlabeled crops → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
