"""Streaming baseline label assignment — direct port of OpenYOLO3D's
``OpenYolo3D.label_3d_masks_from_label_maps`` (utils/__init__.py:158-262).

The baseline accumulator stores raw per-frame data (label_maps, mesh
projections, vertex visibility, YOLO bboxes) and reconstructs the offline
algorithm on the accumulated state. At ``t = final``, the output should
match offline OpenYOLO3D bit-for-bit (modulo float ordering on CUDA vs
CPU).

This file does **not** modify OpenYOLO3D core or the 5월 method modules.
It imports two helper functions from utils/__init__.py via the regular
public path; everything else is a self-contained port.

For Task 1.4 (M11/12/21/22/31/32) the algorithm here is the baseline that
each method axis replaces; this module is its frozen reference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from utils import compute_iou, get_visibility_mat


def construct_label_map_single_frame(
    bbox_pred: dict,
    height: int,
    width: int,
    scaling_w: float = 1.0,
    scaling_h: float = 1.0,
) -> torch.Tensor:
    """Per-frame label map (painted bbox interiors with class id, -1 elsewhere).

    Replicates the inner body of ``OpenYolo3D.construct_label_maps``
    (utils/__init__.py:264-281) for a single frame so the streaming
    wrapper can call it per frame without needing the full OY3D state.

    Args:
        bbox_pred: dict with keys ``"bbox"`` (N, 4) and ``"labels"`` (N,);
            same shape as ``Network_2D.inference_detector`` output.
        height, width: target label_map dimensions (depth resolution).
        scaling_w, scaling_h: pixel ratio from image to depth resolution.

    Returns:
        torch.int16 tensor (height, width). Background pixels = -1; bbox
        interiors set to the bbox's class id (smaller bbox overwrites
        larger if they overlap, matching offline ordering).
    """
    label_map = torch.full((height, width), -1, dtype=torch.int16)
    if "bbox" not in bbox_pred or len(bbox_pred["bbox"]) == 0:
        return label_map

    bboxes = bbox_pred["bbox"].long().clone()
    labels = bbox_pred["labels"].to(torch.int16)

    bboxes[:, 0] = (bboxes[:, 0].float() * scaling_w).long()
    bboxes[:, 2] = (bboxes[:, 2].float() * scaling_w).long()
    bboxes[:, 1] = (bboxes[:, 1].float() * scaling_h).long()
    bboxes[:, 3] = (bboxes[:, 3].float() * scaling_h).long()

    bbox_weights = (bboxes[:, 2] - bboxes[:, 0]) + (bboxes[:, 3] - bboxes[:, 1])
    sort_idx = bbox_weights.sort(descending=True).indices
    bboxes = bboxes[sort_idx]
    labels = labels[sort_idx]

    for i in range(len(bboxes)):
        x_l, y_t, x_r, y_b = bboxes[i].tolist()
        x_l = max(0, min(width, x_l))
        x_r = max(0, min(width, x_r))
        y_t = max(0, min(height, y_t))
        y_b = max(0, min(height, y_b))
        if x_r > x_l and y_b > y_t:
            label_map[y_t:y_b, x_l:x_r] = labels[i]
    return label_map


class BaselineLabelAccumulator:
    """Per-frame accumulator for offline-equivalent streaming label assignment.

    Usage:
        acc = BaselineLabelAccumulator(prediction_3d_masks, num_classes, ...)
        for t in range(F):
            acc.add_frame(projection_t, visible_mask_t, bbox_pred_t, height, width)
        masks, classes, scores = acc.compute_predictions()

    ``compute_predictions()`` may be called at any t with the data
    accumulated so far. It is O(F · V · K) — heavy. Intended call sites:
    final frame + sparse checkpoints (every 10 frames or so).
    """

    def __init__(
        self,
        prediction_3d_masks: torch.Tensor,
        num_classes: int,
        topk: int = 40,
        topk_per_image: int = 600,
        scaling_w: float = 1.0,
        scaling_h: float = 1.0,
        depth_height: int = 0,
        depth_width: int = 0,
        is_gt: bool = False,
        device: str = "cpu",
    ) -> None:
        if prediction_3d_masks.dim() != 2:
            raise ValueError(
                f"prediction_3d_masks must be (V, K); got {prediction_3d_masks.shape}"
            )
        self.prediction_3d_masks = prediction_3d_masks.bool()
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.topk_per_image = int(topk_per_image)
        self.scaling_w = float(scaling_w)
        self.scaling_h = float(scaling_h)
        self.depth_height = int(depth_height)
        self.depth_width = int(depth_width)
        self.is_gt = bool(is_gt)
        self.device = device

        # Per-frame accumulators.
        self._projections: list[torch.Tensor] = []  # each (V, 2) int16
        self._visible_masks: list[torch.Tensor] = []  # each (V,) bool
        self._bbox_preds: list[dict] = []  # each {"bbox", "labels", ...}
        self._label_maps: list[torch.Tensor] = []  # each (H, W) int16
        # Mapping output-column k → original mask3d proposal index, set by
        # compute_predictions after the topk-flatten step. Method axes use
        # this to filter / re-rank predictions on the proposal level.
        self._last_mask_idx: Optional[torch.Tensor] = None

    @property
    def n_frames(self) -> int:
        return len(self._projections)

    @property
    def n_instances(self) -> int:
        return int(self.prediction_3d_masks.shape[1])

    def add_frame(
        self,
        projection: np.ndarray,
        visible_mask: np.ndarray,
        bbox_pred: dict,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> None:
        """Append per-frame data."""
        proj_t = torch.as_tensor(projection, dtype=torch.int32)
        vis_t = torch.as_tensor(visible_mask, dtype=torch.bool)
        if vis_t.dim() != 1:
            raise ValueError(f"visible_mask must be (V,); got {vis_t.shape}")
        if proj_t.dim() != 2 or proj_t.shape[1] != 2:
            raise ValueError(f"projection must be (V, 2); got {proj_t.shape}")

        h = int(height) if height is not None else self.depth_height
        w = int(width) if width is not None else self.depth_width
        if h <= 0 or w <= 0:
            raise ValueError(
                "depth_height/width must be set before add_frame() (or pass height/width)."
            )

        label_map = construct_label_map_single_frame(
            bbox_pred, h, w, scaling_w=self.scaling_w, scaling_h=self.scaling_h
        )

        self._projections.append(proj_t)
        self._visible_masks.append(vis_t)
        self._bbox_preds.append(bbox_pred)
        self._label_maps.append(label_map)

    def _stack(self) -> tuple:
        """Stack frame buffers into (F, ...) tensors on CPU."""
        projections = torch.stack(self._projections, dim=0)  # (F, V, 2) int32
        visible_masks = torch.stack(self._visible_masks, dim=0)  # (F, V) bool
        label_maps = torch.stack(self._label_maps, dim=0)  # (F, H, W) int16
        return projections, visible_masks, label_maps

    def stack_for_methods(self) -> dict:
        """Public access to per-frame buffers for the method-axis adapters
        (M21 WeightedVoting, M22 FeatureFusionEMA). Returns CPU numpy arrays
        + the raw bbox-pred dicts in frame order.
        """
        projections, visible_masks, label_maps = self._stack()
        return {
            "projections": projections.numpy(),       # (F, V, 2) int32
            "visible_masks": visible_masks.numpy(),   # (F, V) bool
            "label_maps": label_maps.numpy(),         # (F, H, W) int16
            "bbox_preds": list(self._bbox_preds),     # list of dicts
            "prediction_3d_masks": self.prediction_3d_masks,  # (V, K) bool
        }

    def compute_predictions(self) -> tuple:
        """Run offline-equivalent label assignment on accumulated state.

        Returns:
            (pred_masks: torch.Tensor (V, K_out) bool,
             pred_classes: torch.Tensor (K_out,) long,
             pred_scores: torch.Tensor (K_out,) float)

        Mirrors ``OpenYolo3D.label_3d_masks_from_label_maps`` (utils/__init__.py:158-262).
        """
        if self.n_frames == 0:
            # Degenerate: no frames seen → no predictions.
            return (
                torch.zeros((self.prediction_3d_masks.shape[0], 0), dtype=torch.bool),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float32),
            )

        # ---- stack frame buffers -----------------------------------
        projections, visible_masks, label_maps = self._stack()  # (F,V,2),(F,V),(F,H,W)

        # ---- top-K representative frames per instance --------------
        pred_masks_VK = self.prediction_3d_masks  # (V, K) bool

        device = torch.device(self.device)
        pred_masks_KV = pred_masks_VK.permute(1, 0).to(device)  # (K, V)
        visible_masks_dev = visible_masks.to(device)  # (F, V)

        topk_for_call = 25 if self.is_gt else self.topk
        visibility_matrix = get_visibility_mat(
            pred_masks_KV, visible_masks_dev, topk=topk_for_call
        )  # (K, F) bool

        # Frames with at least one instance well-visible.
        valid_frames = visibility_matrix.sum(dim=0) >= 1  # (F,)
        valid_frames_cpu = valid_frames.cpu()
        if not valid_frames_cpu.any():
            # No usable frame — every instance predicted as background.
            K = self.n_instances
            pred_classes = torch.full((K,), self.num_classes - 1, dtype=torch.long)
            pred_scores = torch.zeros(K, dtype=torch.float32)
            return pred_masks_VK, pred_classes, pred_scores

        prediction_3d_masks_np = pred_masks_VK.permute(1, 0).cpu().numpy()  # (K, V)
        projections_np = projections[valid_frames_cpu].cpu().numpy()  # (F', V, 2)
        visibility_matrix_np = (
            visibility_matrix[:, valid_frames].cpu().numpy()
        )  # (K, F')
        keep_visible_points_np = (
            visible_masks_dev[valid_frames]
            .cpu()
            .numpy()
        )  # (F', V)
        label_maps_np = label_maps[valid_frames_cpu].numpy()  # (F', H, W)

        bbox_preds_valid = [
            self._bbox_preds[i]
            for i in range(self.n_frames)
            if bool(valid_frames_cpu[i].item())
        ]

        class_labels: list[int] = []
        class_probs: list[float] = []
        distributions: list[torch.Tensor] = []

        for mask_id, mask in enumerate(prediction_3d_masks_np):
            representative_frame_ids = np.where(visibility_matrix_np[mask_id])[0]
            labels_distribution = []
            iou_vals: list[float] = []
            prob_normalizer = 0
            for rep_frame_id in representative_frame_ids:
                visible_points_mask = (
                    keep_visible_points_np[rep_frame_id].squeeze() * mask
                ).astype(bool)
                prob_normalizer += int(visible_points_mask.sum())
                instance_xy = (
                    projections_np[rep_frame_id][np.where(visible_points_mask)]
                    .astype(np.int64)
                )

                boxes = bbox_preds_valid[rep_frame_id]["bbox"].long()
                if len(boxes) > 0 and len(instance_xy) > 10:
                    x_l = int(instance_xy[:, 0].min())
                    x_r = int(instance_xy[:, 0].max()) + 1
                    y_t = int(instance_xy[:, 1].min())
                    y_b = int(instance_xy[:, 1].max()) + 1
                    box = torch.tensor(
                        [
                            x_l / self.scaling_w,
                            y_t / self.scaling_h,
                            x_r / self.scaling_w,
                            y_b / self.scaling_h,
                        ],
                        dtype=torch.float32,
                    )
                    iou_values = compute_iou(box, boxes.float())
                    iou_vals.append(float(iou_values.max().item()))

                selected_labels = label_maps_np[
                    rep_frame_id, instance_xy[:, 1], instance_xy[:, 0]
                ]
                labels_distribution.append(selected_labels)

            labels_distribution = (
                np.concatenate(labels_distribution)
                if len(labels_distribution) > 0
                else np.array([-1])
            )

            distribution = (
                torch.zeros(self.num_classes) if self.topk_per_image != -1 else None
            )
            if (labels_distribution != -1).sum() != 0:
                if distribution is not None:
                    all_labels = torch.from_numpy(
                        labels_distribution[labels_distribution != -1].astype(np.int64)
                    )
                    for lb in all_labels.unique():
                        distribution[int(lb.item())] = (all_labels == lb).sum()
                    distribution = distribution / distribution.max()
                class_label = int(
                    torch.mode(
                        torch.from_numpy(
                            labels_distribution[labels_distribution != -1].astype(
                                np.int64
                            )
                        )
                    ).values.item()
                )
                class_prob = float(
                    (labels_distribution == class_label).sum()
                ) / max(prob_normalizer, 1)
            else:
                if distribution is not None:
                    distribution[-1] = 1.0
                class_label = self.num_classes - 1
                class_prob = 0.0

            iou_vals_t = torch.tensor(iou_vals, dtype=torch.float32)
            class_labels.append(class_label)
            if (iou_vals_t != 0).sum():
                iou_prob = float(iou_vals_t[iou_vals_t != 0].mean().item())
            else:
                iou_prob = 0.0

            class_probs.append(float(class_prob * iou_prob))
            if distribution is not None:
                distributions.append(distribution)

        pred_classes_t = torch.tensor(class_labels, dtype=torch.long)
        pred_scores_t = torch.tensor(class_probs, dtype=torch.float32)
        pred_masks_KV_cpu = pred_masks_VK.permute(1, 0).cpu()  # (K, V)

        if self.topk_per_image != -1 and not self.is_gt and distributions:
            distributions_stacked = torch.stack(distributions)  # (K, num_classes)
            n_inst = distributions_stacked.shape[0]
            distributions_flat = distributions_stacked.reshape(-1)
            labels_flat = (
                torch.arange(self.num_classes)
                .unsqueeze(0)
                .repeat(n_inst, 1)
                .flatten(0, 1)
            )

            cur_topk = min(self.topk_per_image, int(distributions_flat.numel()))
            _, idx = torch.topk(distributions_flat, k=cur_topk, largest=True)
            mask_idx = torch.div(idx, self.num_classes, rounding_mode="floor")

            pred_classes_t = labels_flat[idx]
            pred_scores_t = distributions_flat[idx]
            pred_masks_KV_cpu = pred_masks_KV_cpu[mask_idx]
            self._last_mask_idx = mask_idx.cpu().long()
        else:
            self._last_mask_idx = torch.arange(pred_masks_KV_cpu.shape[0], dtype=torch.long)

        # Return (V, K_out)
        return pred_masks_KV_cpu.permute(1, 0).contiguous(), pred_classes_t, pred_scores_t
