"""METHOD_32 — Hungarian-assignment-based merging that combines spatial
distance and visual-feature similarity into a single cost.

Per-pair cost (i != j):

    cost(i, j) = alpha * ||c_i - c_j||_2  +  (1 - alpha) * (1 - cos(f_i, f_j))

Pairs whose spatial distance exceeds `distance_threshold` (in meters) or
whose semantic similarity falls below `semantic_threshold` are masked to
`+inf`, and so are self-pairs. The masked, symmetric N x N matrix is fed to
`scipy.optimize.linear_sum_assignment`; every (row, col) the solver returns
with a *finite* cost is treated as a merge edge. Merge edges feed a
union-find pass to produce final clusters.

Each cluster collapses to a single instance:
    - id      : min over the cluster (matches the brief: "smallest id wins")
    - centroid: mean of the cluster's centroids
    - bbox_3d : axis-aligned envelope when bboxes are provided, else None
    - label   : the kept (smallest-id) instance's label

This module assumes the visual embeddings come from FeatureFusionEMA (or any
upstream feature provider) and that they live on the same device. CPU is
fine. Inputs are not mutated.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _cosine_similarity_matrix(features: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Return (N, N) cosine-similarity numpy array. Zero-vectors get 0 similarity."""
    f = features.detach().float()
    norms = f.norm(dim=1, keepdim=True).clamp_min(eps)
    fn = f / norms
    sim = (fn @ fn.t()).cpu().numpy()
    # Rows where the original feature was effectively zero shouldn't claim
    # high similarity to anything. Detect by comparing the un-clamped norm.
    raw_norms = f.norm(dim=1).cpu().numpy()
    zero_mask = raw_norms < eps
    if zero_mask.any():
        sim[zero_mask, :] = 0.0
        sim[:, zero_mask] = 0.0
    return sim


class _UnionFind:
    """Tiny union-find for N up to a few thousand."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


class HungarianMerger:
    def __init__(
        self,
        spatial_alpha: float = 0.5,
        distance_threshold: float = 2.0,
        semantic_threshold: float = 0.3,
    ) -> None:
        if not (0.0 <= spatial_alpha <= 1.0):
            raise ValueError(f"spatial_alpha must be in [0, 1], got {spatial_alpha}")
        self.spatial_alpha = float(spatial_alpha)
        self.distance_threshold = float(distance_threshold)
        self.semantic_threshold = float(semantic_threshold)

    # --- cost matrix ---------------------------------------------------------

    def build_cost_matrix(
        self,
        centroids: np.ndarray,
        features: torch.Tensor,
    ) -> np.ndarray:
        """Build the N x N masked cost matrix used by the Hungarian solver.

        Args:
            centroids: (N, 3) float array.
            features:  (N, D) torch tensor.
        """
        n = centroids.shape[0]
        if n == 0:
            return np.zeros((0, 0), dtype=np.float64)
        if features.shape[0] != n:
            raise ValueError(
                f"centroids and features count mismatch: {n} vs {features.shape[0]}"
            )

        # Pairwise Euclidean distance
        diff = centroids[:, None, :] - centroids[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)  # (N, N)

        # Cosine similarity
        sim = _cosine_similarity_matrix(features)

        cost = self.spatial_alpha * dist + (1.0 - self.spatial_alpha) * (1.0 - sim)

        # Mask: self-pairs, far pairs, dissimilar pairs -> +inf
        mask_far = dist > self.distance_threshold
        mask_dissim = sim < self.semantic_threshold
        invalid = mask_far | mask_dissim
        np.fill_diagonal(invalid, True)
        cost[invalid] = np.inf
        return cost

    # --- Hungarian + union-find merge ---------------------------------------

    def merge(
        self,
        instance_list: list,
        instance_features: dict,
    ) -> list:
        """Merge instances based on combined spatial/semantic cost.

        Args:
            instance_list: list of dicts with at minimum:
                - 'id'      : int
                - 'label'   : Any (str preferred)
                - 'centroid': array-like [x, y, z]
                Optional:
                - 'bbox_3d' : (2, 3) array-like  (min_xyz, max_xyz)
            instance_features: {id: 1D torch.Tensor} (typically EMA features
                from METHOD_22).

        Returns:
            New list of merged instance dicts, ordered by surviving id.
        """
        n = len(instance_list)
        if n == 0:
            return []
        if n == 1:
            return [dict(instance_list[0])]

        ids = [int(inst["id"]) for inst in instance_list]
        if len(set(ids)) != n:
            raise ValueError("instance_list contains duplicate ids")

        centroids = np.asarray(
            [np.asarray(inst["centroid"], dtype=np.float64).reshape(-1)
             for inst in instance_list],
            dtype=np.float64,
        )
        if centroids.shape[1] != 3:
            raise ValueError(
                f"each centroid must be length-3, got shape {centroids.shape}"
            )

        # Stack features in instance_list order. Missing features are treated
        # as zero-vectors, which automatically score 0 cosine sim and are
        # therefore excluded by the semantic threshold.
        feat_dim = None
        for f in instance_features.values():
            feat_dim = int(f.reshape(-1).shape[0])
            break
        if feat_dim is None:
            # No features provided at all → no semantic component possible;
            # only spatial-distance threshold can drive merges.
            feat_dim = 1
        feats_rows = []
        for iid in ids:
            f = instance_features.get(int(iid))
            if f is None:
                feats_rows.append(torch.zeros(feat_dim, dtype=torch.float32))
            else:
                feats_rows.append(f.detach().float().reshape(-1))
        features = torch.stack(feats_rows, dim=0)

        cost = self.build_cost_matrix(centroids, features)

        # Hungarian solver requires finite values. Replace +inf with a large
        # sentinel — any pair the solver picks at the sentinel value is a
        # forced fill, so we'll filter them out after assignment.
        finite_mask = np.isfinite(cost)
        if not finite_mask.any():
            # No merge edges possible — every instance survives as a singleton.
            singletons = []
            for i in range(n):
                inst = instance_list[i]
                out = {
                    "id": ids[i],
                    "label": inst.get("label"),
                    "centroid": np.asarray(inst["centroid"], dtype=np.float64).reshape(-1),
                    "merged_from": [ids[i]],
                }
                if inst.get("bbox_3d") is not None:
                    out["bbox_3d"] = np.asarray(inst["bbox_3d"], dtype=np.float64)
                singletons.append(out)
            singletons.sort(key=lambda d: d["id"])
            return singletons

        large = float(cost[finite_mask].max()) * 10.0 + 1.0
        cost_filled = np.where(finite_mask, cost, large)

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_filled)

        uf = _UnionFind(n)
        for r, c in zip(row_ind, col_ind):
            if r == c:
                continue
            if not finite_mask[r, c]:
                continue
            uf.union(int(r), int(c))

        # Cluster instances by root
        clusters: dict = {}
        for i in range(n):
            root = uf.find(i)
            clusters.setdefault(root, []).append(i)

        merged: list = []
        for members in clusters.values():
            members_sorted = sorted(members, key=lambda i: ids[i])
            keep_local = members_sorted[0]
            keep_inst = instance_list[keep_local]
            merged_centroid = centroids[members_sorted].mean(axis=0)
            out = {
                "id": ids[keep_local],
                "label": keep_inst.get("label"),
                "centroid": merged_centroid,
                "merged_from": [ids[i] for i in members_sorted],
            }
            # Combine bboxes if every member has one
            bboxes = [
                np.asarray(instance_list[i].get("bbox_3d"), dtype=np.float64)
                for i in members_sorted
                if instance_list[i].get("bbox_3d") is not None
            ]
            if bboxes and len(bboxes) == len(members_sorted):
                stacked = np.stack(bboxes, axis=0)  # (M, 2, 3)
                out["bbox_3d"] = np.stack(
                    [stacked[:, 0, :].min(axis=0), stacked[:, 1, :].max(axis=0)],
                    axis=0,
                )
            merged.append(out)

        merged.sort(key=lambda d: d["id"])
        return merged
