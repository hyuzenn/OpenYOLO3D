"""METHOD_21 — Weighted label voting (replaces OpenYOLO3D's uniform pixel-mode
aggregation in OpenYolo3D.label_3d_masks_from_label_maps).

Each (instance, frame) pair contributes its pixel-label histogram weighted by:

    w = (alpha * exp(-d3D / D) + (1 - alpha) * exp(-d2D_center / C)) * conf

where:
    d3D     : ||camera_pos - instance_centroid|| in meters.
    d2D_center : ||bbox_2d_center - image_center|| in pixels.
    conf    : per-frame confidence (max 2D-IoU between the projected instance
              AABB and any 2D detection in that frame, falling back to 1.0).

Defaults match the brief: D = 10.0 m, C = 300.0 px, alpha = 0.5.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class WeightedVoting:
    def __init__(
        self,
        distance_weight_decay: float = 10.0,
        center_weight_decay: float = 300.0,
        spatial_alpha: float = 0.5,
    ) -> None:
        self.distance_weight_decay = float(distance_weight_decay)
        self.center_weight_decay = float(center_weight_decay)
        self.spatial_alpha = float(spatial_alpha)

    def frame_weight(
        self,
        camera_pos,
        instance_centroid,
        bbox_2d_center,
        image_size,
        confidence: float = 1.0,
    ) -> float:
        cam = np.asarray(camera_pos, dtype=np.float64).reshape(-1)
        ctr = np.asarray(instance_centroid, dtype=np.float64).reshape(-1)
        d3 = float(np.linalg.norm(cam - ctr))
        w_dist = math.exp(-d3 / self.distance_weight_decay)

        bc = np.asarray(bbox_2d_center, dtype=np.float64).reshape(-1)
        img_c = np.array([image_size[0] / 2.0, image_size[1] / 2.0], dtype=np.float64)
        d2 = float(np.linalg.norm(bc - img_c))
        w_center = math.exp(-d2 / self.center_weight_decay)

        spatial = self.spatial_alpha * w_dist + (1.0 - self.spatial_alpha) * w_center
        return spatial * max(float(confidence), 0.0)

    def vote_distribution(
        self,
        per_instance_frame_labels: list,
        per_instance_frame_meta: list,
        num_classes: int,
    ) -> torch.Tensor:
        """Return a (n_inst, num_classes) tensor of weighted, max-normalized
        per-class votes. Instances with no valid pixel-labels get an all-zero
        row with the last bin (background) set to 1.0, matching the original
        OpenYOLO3D fall-back semantics.
        """
        n_inst = len(per_instance_frame_labels)
        out = torch.zeros(n_inst, num_classes, dtype=torch.float32)
        for i in range(n_inst):
            frames = per_instance_frame_labels[i]
            metas = per_instance_frame_meta[i]
            saw_valid = False
            for labels, meta in zip(frames, metas):
                if labels is None:
                    continue
                arr = np.asarray(labels).reshape(-1)
                arr = arr[arr != -1]
                if arr.size == 0:
                    continue
                w = self.frame_weight(**meta)
                if w <= 0.0:
                    continue
                vals, counts = np.unique(arr, return_counts=True)
                for v, c in zip(vals.tolist(), counts.tolist()):
                    if 0 <= v < num_classes:
                        out[i, int(v)] += float(w) * float(c)
                saw_valid = True
            if not saw_valid:
                out[i, -1] = 1.0
            else:
                m = float(out[i].max().item())
                if m > 0.0:
                    out[i] = out[i] / m
        return out

    def vote_label(
        self,
        per_instance_frame_labels: list,
        per_instance_frame_meta: list,
        num_classes: int,
    ) -> list:
        """Return [final_label_per_instance]. Argmax of weighted distribution."""
        dist = self.vote_distribution(per_instance_frame_labels, per_instance_frame_meta, num_classes)
        return [int(x.item()) for x in dist.argmax(dim=1)]
