"""METHOD_31 — class-aware 3D IoU merging applied to OpenYOLO3D's
post-classification per-scene predictions.

After OpenYolo3D.label_3d_masks_from_2d_bboxes, the pipeline returns up to
topk_per_image (mask, class, score) triples. Two predictions of the same class
that overlap heavily on the 3D mesh should collapse to one. This module
performs class-aware NMS:

    1. Optional KDTree pre-filter on per-mask centroids (radius
       kdtree_neighbor_radius meters) to limit the IoU computation to nearby
       pairs.
    2. Sort predictions by score (descending). For each, suppress all later
       predictions with the same class label and mask-IoU >= iou_threshold.

The mesh masks are vertex-level boolean masks; we use vertex-set IoU
(|A ∩ B| / |A ∪ B|) as the 3D IoU. This avoids fitting axis-aligned bboxes,
which is lossy for elongated / curved objects in indoor scenes.

The merge keeps the **higher-score** prediction in each cluster (a strict
NMS), which matches the pipeline's existing scoring conventions and is what
the ScanNet evaluator uses to rank predictions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class IoUMerger:
    def __init__(
        self,
        iou_threshold: float = 0.5,
        use_kdtree: bool = True,
        kdtree_neighbor_radius: float = 2.0,
        same_class_only: bool = True,
    ) -> None:
        self.iou_threshold = float(iou_threshold)
        self.use_kdtree = bool(use_kdtree)
        self.kdtree_neighbor_radius = float(kdtree_neighbor_radius)
        self.same_class_only = bool(same_class_only)

    def merge(
        self,
        predicted_masks: torch.Tensor,
        pred_classes: torch.Tensor,
        pred_scores: torch.Tensor,
        vertex_coords: Optional[np.ndarray] = None,
    ):
        """Apply class-aware IoU NMS.

        Args:
            predicted_masks: (n_vertices, K) bool tensor (matches OpenYOLO3D's
                output convention from label_3d_masks_from_2d_bboxes).
            pred_classes: (K,) long tensor.
            pred_scores: (K,) float tensor.
            vertex_coords: optional (n_vertices, 3) numpy array for KDTree
                pre-filtering. If None or use_kdtree=False, all pairs are
                considered.

        Returns:
            (kept_masks, kept_classes, kept_scores) with the same conventions.
        """
        if predicted_masks.numel() == 0 or pred_classes.numel() == 0:
            return predicted_masks, pred_classes, pred_scores

        if not self.same_class_only:
            return self._merge_global(
                predicted_masks, pred_classes, pred_scores, vertex_coords
            )

        # Same-class fast path: split predictions by class label, run
        # per-class NMS, recombine. Each class group is small (~3-30 entries
        # for ScanNet200 top-k=600), so per-class work is cheap.
        K = pred_classes.shape[0]
        if K <= 1:
            return predicted_masks, pred_classes, pred_scores

        keep_flag = np.zeros(K, dtype=bool)
        classes_np = pred_classes.cpu().numpy()
        unique_classes = np.unique(classes_np)
        verts = np.asarray(vertex_coords, dtype=np.float64) if vertex_coords is not None else None

        for c in unique_classes:
            idx = np.where(classes_np == c)[0]
            if idx.size == 0:
                continue
            kept_local = self._nms_within_group(
                predicted_masks, pred_scores, idx, verts
            )
            for j in kept_local:
                keep_flag[j] = True

        keep_idx = torch.from_numpy(np.where(keep_flag)[0]).long().to(predicted_masks.device)
        return predicted_masks[:, keep_idx], pred_classes[keep_idx], pred_scores[keep_idx]

    def _nms_within_group(
        self,
        predicted_masks: torch.Tensor,
        pred_scores: torch.Tensor,
        global_idx: np.ndarray,
        verts: Optional[np.ndarray],
    ) -> list:
        """Greedy NMS over a same-class subset of predictions. Returns the
        global indices of kept predictions.
        """
        if global_idx.size == 0:
            return []
        if global_idx.size == 1:
            sub = predicted_masks[:, int(global_idx[0])]
            return [int(global_idx[0])] if bool(sub.any().item()) else []

        # Subset of masks (V, n) bool, transposed to (n, V) for row-wise ops
        sub_masks = predicted_masks[:, torch.from_numpy(global_idx).long()].permute(1, 0).contiguous().bool()
        n = sub_masks.shape[0]
        sub_sizes = sub_masks.sum(dim=1).cpu().numpy()
        sub_scores = pred_scores[torch.from_numpy(global_idx).long()].cpu().numpy()

        # Optional KDTree pre-filter on centroids (only useful when n is large).
        centroids: Optional[np.ndarray] = None
        if (
            self.use_kdtree
            and verts is not None
            and n > 8  # below this threshold, full-pair compare is faster
        ):
            sub_masks_np = sub_masks.cpu().numpy()
            cents = np.zeros((n, 3), dtype=np.float64)
            for k in range(n):
                if sub_sizes[k] == 0:
                    continue
                cents[k] = verts[sub_masks_np[k]].mean(axis=0)
            centroids = cents

        tree = None
        if centroids is not None:
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(centroids)
            except Exception:
                tree = None

        order = np.argsort(-sub_scores)  # descending
        keep_local = np.ones(n, dtype=bool)
        keep_local[sub_sizes == 0] = False  # never keep empty masks

        for rank_i, i in enumerate(order.tolist()):
            if not keep_local[i]:
                continue
            if tree is not None:
                neighbors = tree.query_ball_point(centroids[i], self.kdtree_neighbor_radius)
                later = [j for j in neighbors if keep_local[j] and j != i]
            else:
                later = [int(j) for j in order[rank_i + 1 :] if keep_local[int(j)]]

            if not later:
                continue

            j_idx = torch.tensor(later, dtype=torch.long, device=sub_masks.device)
            mi = sub_masks[i]  # (V,) bool
            mj = sub_masks[j_idx]  # (n_neighbors, V) bool
            # Bool & / | are 1-byte ops; sum returns int — much cheaper than
            # float matmul or repeated allocation of float32 batches.
            inter = (mj & mi).sum(dim=1)
            uni = (mj | mi).sum(dim=1)
            uni_f = uni.float().clamp_min_(1.0)
            ious = inter.float() / uni_f
            ious[uni == 0] = 0.0

            for jj, iou_v in zip(later, ious.cpu().tolist()):
                if iou_v >= self.iou_threshold and sub_scores[jj] <= sub_scores[i]:
                    keep_local[jj] = False

        return [int(global_idx[k]) for k in range(n) if keep_local[k]]

    def _merge_global(
        self,
        predicted_masks: torch.Tensor,
        pred_classes: torch.Tensor,
        pred_scores: torch.Tensor,
        vertex_coords: Optional[np.ndarray],
    ):
        """Class-agnostic NMS — kept for completeness (same_class_only=False)."""
        K = pred_classes.shape[0]
        idx_all = np.arange(K)
        verts = np.asarray(vertex_coords, dtype=np.float64) if vertex_coords is not None else None
        kept = self._nms_within_group(predicted_masks, pred_scores, idx_all, verts)
        keep_idx = torch.tensor(kept, dtype=torch.long, device=predicted_masks.device)
        return predicted_masks[:, keep_idx], pred_classes[keep_idx], pred_scores[keep_idx]
