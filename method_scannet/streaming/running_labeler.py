"""Per-frame running label snapshot for temporal-metric instrumentation.

Maintains an incremental per-instance label histogram so the streaming
wrapper can answer "what label does the model currently assign to
instance i?" at every frame, not just at scene end. The cumulative
mode (or weighted argmax for M21) is what
:mod:`method_scannet.streaming.metrics` needs to compute
``label_switch_count`` and ``time_to_confirm``.

Design notes:
- The labeler does **not** replace ``BaselineLabelAccumulator`` —
  that one still computes the offline-equivalent scene-end prediction
  in one batched pass. The running labeler is a sibling whose only job
  is to expose a per-frame snapshot for metric collection.
- Two modes:
    1. ``baseline``: uniform-weighted histogram (matches the underlying
       offline mode-vote semantics).
    2. ``m21``: weighted histogram using the WeightedVoting class's
       ``frame_weight`` formula (alpha · exp(-d3/D) + (1-alpha) ·
       exp(-d2/C)) × per-frame IoU confidence. Hyperparameters come
       from the live ``WeightedVoting`` instance.
- For M22 the wrapper bypasses this class and calls
  ``method_22.predict_label`` directly (the EMA fusion already keeps
  cumulative state, and its label space differs from the baseline
  mode-vote space).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch


class RunningInstanceLabeler:
    """Per-instance cumulative label histogram + per-frame snapshot.

    Args:
        num_classes: ScanNet label space (e.g. 199 for ScanNet200 inference).
        instance_vertex_masks: (K, V) bool array — Mask3D's K proposals.
            Held as a reference; not copied.
        scene_vertices: (V, 3) float — used by the M21 weighting fn to
            compute per-instance centroid → world-frame distance to camera.
        depth_height/width: depth-map dimensions (matches label_map shape).
        voter: optional ``WeightedVoting`` instance. When provided, the
            labeler uses its hyperparameters (and the per-frame IoU
            confidence vs YOLO bboxes) to weight the histogram.
        image_height/width: image resolution (M21 center-distance uses
            this; only consulted when ``voter`` is not None).
        scaling_w/h: depth↔image scaling for AABB→bbox match (M21 only).
    """

    def __init__(
        self,
        num_classes: int,
        instance_vertex_masks: np.ndarray,
        scene_vertices: np.ndarray,
        depth_height: int,
        depth_width: int,
        voter=None,
        image_height: int = 0,
        image_width: int = 0,
        scaling_w: float = 1.0,
        scaling_h: float = 1.0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.instance_vertex_masks = instance_vertex_masks  # (K, V)
        self.scene_vertices = scene_vertices                # (V, 3)
        self.depth_h = int(depth_height)
        self.depth_w = int(depth_width)
        self.voter = voter
        self.image_h = int(image_height)
        self.image_w = int(image_width)
        self.scaling_w = float(scaling_w)
        self.scaling_h = float(scaling_h)
        # Per-instance cumulative class histogram.
        self.counts: dict[int, np.ndarray] = {}
        # Optional per-instance cached centroid (only needed for M21).
        self._centroid_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Per-frame ingestion
    # ------------------------------------------------------------------

    def update_frame(
        self,
        visible_instances: Sequence[int],
        projection: np.ndarray,           # (V, 2) int — vertex→depth px
        inside_mask: np.ndarray,          # (V,) bool — D3 visibility for the frame
        label_map: np.ndarray,            # (H, W) int16 — frame's painted bboxes
        bbox_pred: dict,                  # YOLO output for this frame
        camera_position: Optional[np.ndarray] = None,  # (3,) world cam pos
    ) -> None:
        """Accumulate one frame's pixel-labels into per-instance histograms."""
        if len(visible_instances) == 0:
            return
        # Safe pixel coords clip — projection can be out of frame for
        # near-edge vertices even if inside_mask=True (depth-test passed but
        # the (x, y) clamping logic in compute_vertex_projection is local).
        proj = np.asarray(projection, dtype=np.int64)
        H, W = label_map.shape

        boxes_yolo = bbox_pred.get("bbox")
        if boxes_yolo is not None and hasattr(boxes_yolo, "long"):
            boxes_yolo = boxes_yolo.long()
        else:
            boxes_yolo = None

        img_cx = self.image_w / 2.0
        img_cy = self.image_h / 2.0

        for iid in visible_instances:
            iid = int(iid)
            inst_mask = self.instance_vertex_masks[iid].astype(bool)
            visible_pts = inside_mask & inst_mask
            if not visible_pts.any():
                continue
            xy = proj[visible_pts]
            if xy.shape[0] == 0:
                continue
            # Clip to label_map bounds.
            xs = np.clip(xy[:, 0], 0, W - 1)
            ys = np.clip(xy[:, 1], 0, H - 1)
            labels = label_map[ys, xs]
            valid = labels[labels != -1]
            if valid.size == 0:
                continue

            # Weight: baseline = 1.0; M21 = WeightedVoting.frame_weight.
            if self.voter is not None and camera_position is not None:
                w = self._compute_m21_weight(
                    iid=iid,
                    inst_mask=inst_mask,
                    xy=xy,
                    boxes_yolo=boxes_yolo,
                    camera_position=camera_position,
                    img_cx=img_cx,
                    img_cy=img_cy,
                )
                if w <= 0.0:
                    continue
            else:
                w = 1.0

            hist = self.counts.setdefault(iid, np.zeros(self.num_classes, dtype=np.float64))
            vals, cnts = np.unique(valid.astype(np.int64), return_counts=True)
            for v, c in zip(vals.tolist(), cnts.tolist()):
                if 0 <= v < self.num_classes:
                    hist[int(v)] += float(w) * float(c)

    def _compute_m21_weight(
        self,
        iid: int,
        inst_mask: np.ndarray,
        xy: np.ndarray,
        boxes_yolo,
        camera_position: np.ndarray,
        img_cx: float,
        img_cy: float,
    ) -> float:
        """Per-frame weight matching WeightedVoting.frame_weight / the
        inlined formula in May's offline hooks."""
        voter = self.voter
        centroid = self._centroid_cache.get(iid)
        if centroid is None:
            if inst_mask.any():
                centroid = self.scene_vertices[inst_mask].mean(axis=0)
            else:
                centroid = np.zeros(3, dtype=np.float64)
            self._centroid_cache[iid] = centroid

        d3 = float(
            math.sqrt(
                (camera_position[0] - centroid[0]) ** 2
                + (camera_position[1] - centroid[1]) ** 2
                + (camera_position[2] - centroid[2]) ** 2
            )
        )
        w_dist = math.exp(-d3 / float(voter.distance_weight_decay))

        # 2D bbox center + IoU-vs-YOLO confidence.
        x_l = int(xy[:, 0].min())
        x_r = int(xy[:, 0].max()) + 1
        y_t = int(xy[:, 1].min())
        y_b = int(xy[:, 1].max()) + 1
        confidence = 1.0
        bbox_cx = (x_l + x_r) / 2.0
        bbox_cy = (y_t + y_b) / 2.0

        if boxes_yolo is not None and len(boxes_yolo) > 0 and xy.shape[0] > 10:
            try:
                from utils import compute_iou

                box_img = torch.tensor(
                    [
                        x_l / self.scaling_w,
                        y_t / self.scaling_h,
                        x_r / self.scaling_w,
                        y_b / self.scaling_h,
                    ],
                    dtype=torch.float32,
                )
                ious = compute_iou(box_img, boxes_yolo.float())
                confidence = float(ious.max().item())
            except Exception:
                confidence = 1.0

        d2 = float(math.sqrt((bbox_cx - img_cx) ** 2 + (bbox_cy - img_cy) ** 2))
        w_center = math.exp(-d2 / float(voter.center_weight_decay))
        alpha = float(voter.spatial_alpha)
        return (alpha * w_dist + (1.0 - alpha) * w_center) * max(confidence, 0.0)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, instance_ids: Sequence[int]) -> dict[int, int]:
        """Return cumulative argmax label for each requested instance id.

        Instances never observed (or with all-zero histogram) get ``-1``.
        """
        out: dict[int, int] = {}
        for iid in instance_ids:
            iid_i = int(iid)
            h = self.counts.get(iid_i)
            if h is None or float(h.max()) == 0.0:
                out[iid_i] = -1
            else:
                out[iid_i] = int(np.argmax(h))
        return out

    def snapshot_method22(
        self,
        instance_ids: Sequence[int],
        fusion,
        class_name_to_idx: Optional[dict] = None,
    ) -> dict[int, int]:
        """Snapshot using FeatureFusionEMA's running EMA feature.

        ``fusion`` is the populated ``FeatureFusionEMA``; we call
        ``predict_label(iid)`` which returns (class_name_or_idx, conf).
        If the fusion was built with ``prompt_class_names``, the result
        is the class index in the inference-class subset — which is
        already the correct id for ``metrics.label_switch_count``.
        """
        out: dict[int, int] = {}
        for iid in instance_ids:
            iid_i = int(iid)
            feat = fusion.get_feature(iid_i)
            if feat is None:
                out[iid_i] = -1
                continue
            try:
                pred = fusion.predict_label(iid_i)
                # pred is (class_name_or_idx, confidence)
                cls = pred[0]
                if isinstance(cls, str):
                    if class_name_to_idx is not None and cls in class_name_to_idx:
                        out[iid_i] = int(class_name_to_idx[cls])
                    else:
                        out[iid_i] = -1
                else:
                    out[iid_i] = int(cls)
            except Exception:
                out[iid_i] = -1
        return out

    def reset(self) -> None:
        self.counts.clear()
        self._centroid_cache.clear()
