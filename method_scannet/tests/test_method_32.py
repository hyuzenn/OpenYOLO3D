"""Unit tests for METHOD_32 — HungarianMerger. CPU-only mock data."""
from __future__ import annotations

import numpy as np
import torch

from method_scannet.method_32_hungarian_merging import HungarianMerger


def _make_clustered_instances(num_clusters: int = 5, per_cluster: int = 2, dim: int = 16, seed: int = 0):
    """5 well-separated cluster centers, 2 instances each. Within a cluster:
    centroids are within ~0.05m and visual features are nearly identical
    (cosine sim ~1). Across clusters: centroids are ~10m apart and features
    are independent (cosine sim ~0)."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    instances = []
    features = {}
    for c in range(num_clusters):
        center = np.array([c * 10.0, 0.0, 0.0])
        cluster_feat = torch.randn(dim, generator=g)
        for k in range(per_cluster):
            iid = c * per_cluster + k
            jitter = rng.normal(scale=0.05, size=3)
            instances.append(
                {
                    "id": iid,
                    "label": f"obj_{c}",
                    "centroid": center + jitter,
                    "bbox_3d": np.stack(
                        [center + jitter - 0.5, center + jitter + 0.5], axis=0
                    ),
                }
            )
            # Per-instance features differ only by tiny noise within cluster
            features[iid] = cluster_feat + 0.01 * torch.randn(dim, generator=g)
    return instances, features


def test_cost_matrix_diagonal_inf_and_symmetry():
    instances, features = _make_clustered_instances(num_clusters=3, per_cluster=2)
    merger = HungarianMerger(spatial_alpha=0.5, distance_threshold=2.0, semantic_threshold=0.3)
    centroids = np.stack([inst["centroid"] for inst in instances], axis=0)
    feats = torch.stack([features[inst["id"]] for inst in instances], dim=0)
    cost = merger.build_cost_matrix(centroids, feats)
    n = cost.shape[0]
    assert cost.shape == (n, n)
    # Diagonal must be inf (self-pair forbidden)
    assert np.all(np.isinf(np.diag(cost)))
    # Off-diagonal must be symmetric
    finite = np.isfinite(cost)
    triu = np.triu(np.ones_like(cost, dtype=bool), k=1)
    pairs = finite & triu
    if pairs.any():
        rs, cs = np.where(pairs)
        assert np.allclose(cost[rs, cs], cost[cs, rs])


def test_far_pairs_are_masked_inf():
    """Two instances 100m apart must produce inf cost regardless of features."""
    instances = [
        {"id": 0, "label": "a", "centroid": np.array([0.0, 0.0, 0.0])},
        {"id": 1, "label": "b", "centroid": np.array([100.0, 0.0, 0.0])},
    ]
    f = torch.tensor([1.0, 0.0, 0.0, 0.0])
    features = {0: f.clone(), 1: f.clone()}
    merger = HungarianMerger(distance_threshold=2.0)
    cost = merger.build_cost_matrix(
        np.stack([inst["centroid"] for inst in instances]),
        torch.stack([features[i] for i in (0, 1)]),
    )
    assert np.isinf(cost[0, 1]) and np.isinf(cost[1, 0])


def test_clustered_pairs_collapse_to_one_per_cluster():
    instances, features = _make_clustered_instances(num_clusters=5, per_cluster=2)
    merger = HungarianMerger(spatial_alpha=0.5, distance_threshold=2.0, semantic_threshold=0.3)
    merged = merger.merge(instances, features)
    assert len(merged) == 5, f"expected 5 merged clusters, got {len(merged)}"
    # Surviving ids should be the smallest in each cluster: 0, 2, 4, 6, 8
    surviving_ids = sorted(m["id"] for m in merged)
    assert surviving_ids == [0, 2, 4, 6, 8]
    # Each merged group should contain exactly the two source ids
    for m in merged:
        assert sorted(m["merged_from"]) == [m["id"], m["id"] + 1]
        assert m["bbox_3d"].shape == (2, 3)


def test_no_merge_when_thresholds_block_everything():
    """Set distance_threshold tiny; nothing should merge."""
    instances, features = _make_clustered_instances(num_clusters=3, per_cluster=2)
    merger = HungarianMerger(spatial_alpha=0.5, distance_threshold=1e-6, semantic_threshold=0.3)
    merged = merger.merge(instances, features)
    assert len(merged) == len(instances)
    # All ids preserved, every group is size 1
    for m in merged:
        assert m["merged_from"] == [m["id"]]


def test_singleton_instance_returns_single_copy():
    inst = [{"id": 42, "label": "x", "centroid": np.array([1.0, 2.0, 3.0])}]
    merged = HungarianMerger().merge(inst, {42: torch.ones(8)})
    assert len(merged) == 1
    assert merged[0]["id"] == 42
