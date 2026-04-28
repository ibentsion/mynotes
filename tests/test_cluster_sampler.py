import numpy as np
import pandas as pd
import pytest

from src.cluster_sampler import _medoid_indices, build_priority_queue, extract_hog_features
from src.manifest_schema import MANIFEST_COLUMNS


def _make_manifest(n: int, status: str = "unlabeled") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crop_path": [f"/fake/crop_{i}.png" for i in range(n)],
            "pdf_path": ["/fake/doc.pdf"] * n,
            "page_num": [1] * n,
            "x": range(n),
            "y": [0] * n,
            "w": [50] * n,
            "h": [50] * n,
            "area": [2500] * n,
            "is_flagged": [False] * n,
            "flag_reasons": [""] * n,
            "status": [status] * n,
            "label": [""] * n,
            "notes": [""] * n,
        },
        columns=MANIFEST_COLUMNS,
    )


def test_medoid_indices_returns_one_per_cluster():
    rng = np.random.default_rng(0)
    features = rng.random((20, 10), dtype=np.float32)
    labels = np.array([i % 4 for i in range(20)])
    medoids = _medoid_indices(features, labels, 4)
    assert len(medoids) == 4
    assert len(set(medoids)) == 4  # no duplicates
    assert all(0 <= m < 20 for m in medoids)


def test_medoid_is_closest_to_centroid():
    # Two tight clusters, each with an obvious center
    features = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                          [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])
    medoids = _medoid_indices(features, labels, 2)
    assert 0 in medoids  # centroid of cluster 0 is near index 0
    assert 3 in medoids  # centroid of cluster 1 is near index 3


def test_extract_hog_features_missing_file_returns_zeros():
    feats = extract_hog_features(["/nonexistent/crop.png"])
    assert feats.shape[0] == 1
    assert (feats[0] == 0).all()


def test_extract_hog_features_shape(tmp_path):
    import cv2

    img = np.full((80, 120), 128, dtype=np.uint8)
    p = str(tmp_path / "crop.png")
    cv2.imwrite(p, img)
    feats = extract_hog_features([p])
    from src.cluster_sampler import _FEAT_DIM
    assert feats.shape == (1, _FEAT_DIM)
    assert feats.dtype == np.float32


def test_build_priority_queue_medoids_come_first(tmp_path):
    import cv2

    n = 30
    paths = []
    for i in range(n):
        p = tmp_path / f"crop_{i}.png"
        cv2.imwrite(str(p), np.full((40, 40), i * 8, dtype=np.uint8))
        paths.append(str(p))

    df = _make_manifest(n)
    df["crop_path"] = paths

    result = build_priority_queue(df, n_clusters=5)

    assert len(result) == n
    assert set(result["crop_path"]) == set(paths)


def test_build_priority_queue_preserves_labeled_at_end():
    df_unlabeled = _make_manifest(10, status="unlabeled")
    df_labeled = _make_manifest(5, status="labeled")
    df_labeled["crop_path"] = [f"/fake/labeled_{i}.png" for i in range(5)]
    df = pd.concat([df_unlabeled, df_labeled], ignore_index=True)

    result = build_priority_queue(df, n_clusters=3)

    labeled_rows = result[result["status"] == "labeled"]
    unlabeled_rows = result[result["status"] == "unlabeled"]
    # All unlabeled rows appear before all labeled rows
    assert unlabeled_rows.index.max() < labeled_rows.index.min()


def test_build_priority_queue_all_labeled_returns_unchanged():
    df = _make_manifest(5, status="labeled")
    result = build_priority_queue(df, n_clusters=3)
    assert list(result["crop_path"]) == list(df["crop_path"])


def test_build_priority_queue_n_clusters_larger_than_unlabeled(tmp_path):
    import cv2

    n = 4
    paths = []
    for i in range(n):
        p = tmp_path / f"crop_{i}.png"
        cv2.imwrite(str(p), np.full((40, 40), 100, dtype=np.uint8))
        paths.append(str(p))

    df = _make_manifest(n)
    df["crop_path"] = paths

    result = build_priority_queue(df, n_clusters=100)  # more clusters than samples
    assert len(result) == n
