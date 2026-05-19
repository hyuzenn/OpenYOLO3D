"""2D detection bbox → LiDAR ego-frame frustum extraction.

For each YOLO-World 2D detection in any of the 6 cameras, we cut a 3D frustum
out of the ego-frame point cloud:
  1. expand the 2D bbox by ``expand_ratio`` (10% default)
  2. transform LiDAR points into that camera's frame
  3. keep points whose camera-frame depth is in [min_depth, max_depth] AND
     whose image-plane projection lies inside the (expanded) bbox

This is the geometric-from-detection step that breaks the 36% geometry-only
ceiling — clustering is later run only inside these frustums, so the
proposal pool is constrained to regions a 2D detector already considered
foreground.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


@dataclass
class FrustumConfig:
    expand_ratio: float = 0.10
    min_depth: float = 1.0
    max_depth: float = 80.0
    use_z_filter: bool = True   # informational; z-filter is downstream's job
                                 # (the pillar foreground extractor handles ground removal)


class FrustumExtractor:
    def __init__(
        self,
        expand_ratio: float = 0.10,
        min_depth: float = 1.0,
        max_depth: float = 80.0,
        use_z_filter: bool = True,
    ):
        self.config = FrustumConfig(
            expand_ratio=float(expand_ratio),
            min_depth=float(min_depth),
            max_depth=float(max_depth),
            use_z_filter=bool(use_z_filter),
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    @staticmethod
    def _project_pc_to_cam(pc_ego_xyz: np.ndarray, K: np.ndarray,
                            T_cam_to_ego: np.ndarray):
        """Returns (uv_all_pts, depth_cam, in_front_mask). uv_all_pts has shape
        (N, 2) and is finite only where in_front_mask is True. Depth is
        cam-frame z."""
        N = pc_ego_xyz.shape[0]
        if N == 0:
            return (np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool))
        pts_h = np.concatenate([pc_ego_xyz, np.ones((N, 1))], axis=1)
        pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]
        depth = pts_cam[:, 2]
        in_front = depth > 0
        uv = np.full((N, 2), np.nan, dtype=np.float64)
        if in_front.any():
            pts_f = pts_cam[in_front]
            uv_h = (K @ pts_f.T).T
            uv[in_front] = uv_h[:, :2] / uv_h[:, 2:3]
        return uv, depth, in_front

    def extract_frustums(
        self,
        detections_per_cam: dict,        # {cam: {"xyxy": [[x1,y1,x2,y2],..],
                                          #         "labels": [str,..],
                                          #         "scores": [float,..]}}
        intrinsics_per_cam: dict,         # {cam: (3,3)}
        cam_to_ego_per_cam: dict,         # {cam: (4,4)}
        image_hw_per_cam: dict,           # {cam: (H,W)}
        point_cloud_ego: np.ndarray,      # (N, ≥3); columns 0:3 are x,y,z
    ) -> list:
        t0 = time.perf_counter()
        pc_xyz = point_cloud_ego[:, :3]
        N = pc_xyz.shape[0]

        frustums = []
        for cam, det_dict in detections_per_cam.items():
            xyxys = det_dict.get("xyxy", [])
            labels = det_dict.get("labels", [])
            scores = det_dict.get("scores", [])
            if len(xyxys) == 0:
                continue
            K = intrinsics_per_cam[cam]
            T = cam_to_ego_per_cam[cam]
            H, W = image_hw_per_cam[cam]
            uv, depth, in_front = self._project_pc_to_cam(pc_xyz, K, T)
            if not in_front.any():
                continue

            in_depth = (depth >= self.config.min_depth) & (depth <= self.config.max_depth)
            base_keep = in_front & in_depth

            for det_idx, bbox in enumerate(xyxys):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                w, h = (x2 - x1), (y2 - y1)
                grow = self.config.expand_ratio
                x1e = cx - w * (1 + grow) / 2
                x2e = cx + w * (1 + grow) / 2
                y1e = cy - h * (1 + grow) / 2
                y2e = cy + h * (1 + grow) / 2
                x1c = max(0.0, x1e)
                y1c = max(0.0, y1e)
                x2c = min(float(W), x2e)
                y2c = min(float(H), y2e)
                in_box = np.zeros(N, dtype=bool)
                if base_keep.any():
                    u = uv[:, 0]; v = uv[:, 1]
                    valid_uv = base_keep & np.isfinite(u) & np.isfinite(v)
                    if valid_uv.any():
                        idx = np.where(valid_uv)[0]
                        match = (u[idx] >= x1c) & (u[idx] < x2c) & (v[idx] >= y1c) & (v[idx] < y2c)
                        in_box[idx[match]] = True

                pts_in = point_cloud_ego[in_box]
                d_in = depth[in_box] if in_box.any() else np.zeros((0,))
                frustums.append({
                    "cam": cam,
                    "det_idx": int(det_idx),
                    "class": str(labels[det_idx]) if det_idx < len(labels) else f"class_{det_idx}",
                    "score": float(scores[det_idx]) if det_idx < len(scores) else 0.0,
                    "bbox_2d_original": [x1, y1, x2, y2],
                    "bbox_2d_expanded_clipped": [x1c, y1c, x2c, y2c],
                    "frustum_point_indices": np.where(in_box)[0],   # caller can drop later
                    "frustum_points_ego": pts_in,
                    "n_points": int(in_box.sum()),
                    "depth_min": float(d_in.min()) if d_in.size else None,
                    "depth_max": float(d_in.max()) if d_in.size else None,
                })

        t_total = time.perf_counter() - t0
        return {
            "frustums": frustums,
            "n_frustums": len(frustums),
            "n_with_points": int(sum(1 for f in frustums if f["n_points"] > 0)),
            "timing": {"total": float(t_total)},
        }
