"""Pillar-based foreground/background separation.

Drops LiDAR returns that look like ground/road/vegetation and keeps points
that sit on objects rising above the ground plane. Inspired by PointPillars'
BEV pillar grid: split the xy plane into squares of size ``pillar_size_xy``,
estimate ground height per scene, and keep all points in any pillar whose
max-height-above-ground exceeds ``z_threshold``.

Two ground estimators:
  - ``ransac``     — fit a single plane via open3d segment_plane (handles
                     scenes with sloped roads better)
  - ``percentile`` — use the lower P percentile of z as a flat-ground
                     estimate (cheap, decent on flat scenes)

Output keeps the original (x, y, z, intensity) columns of the foreground
subset so downstream HDBSCAN sees the same dtype/layout it would on raw input.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import open3d as o3d


@dataclass
class PillarConfig:
    pillar_size_xy: Tuple[float, float] = (0.5, 0.5)
    z_threshold: float = 0.3
    ground_estimation: str = "ransac"        # "ransac" | "percentile"
    ransac_distance_threshold: float = 0.2
    ransac_n: int = 3
    ransac_iterations: int = 1000
    percentile_p: float = 10.0


class PillarForegroundExtractor:
    def __init__(
        self,
        pillar_size_xy: Tuple[float, float] = (0.5, 0.5),
        z_threshold: float = 0.3,
        ground_estimation: str = "ransac",
        ransac_distance_threshold: float = 0.2,
        ransac_n: int = 3,
        ransac_iterations: int = 1000,
        percentile_p: float = 10.0,
    ):
        self.config = PillarConfig(
            pillar_size_xy=tuple(pillar_size_xy),
            z_threshold=z_threshold,
            ground_estimation=ground_estimation,
            ransac_distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            ransac_iterations=ransac_iterations,
            percentile_p=percentile_p,
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def _estimate_ground_z_at_points(self, pts: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Returns (ground_z_at_each_point: (N,), info_dict)."""
        if self.config.ground_estimation == "ransac":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            try:
                plane_model, _ = pcd.segment_plane(
                    distance_threshold=self.config.ransac_distance_threshold,
                    ransac_n=self.config.ransac_n,
                    num_iterations=self.config.ransac_iterations,
                )
                a, b, c, d = plane_model
                if abs(c) < 1e-6:
                    # near-vertical plane: fall back to percentile
                    z_global = float(np.percentile(pts[:, 2], self.config.percentile_p))
                    return np.full(pts.shape[0], z_global), {
                        "method": "ransac_fallback_percentile",
                        "fallback_reason": "near_vertical_plane",
                        "ground_z_global": z_global,
                    }
                ground_z = -(a * pts[:, 0] + b * pts[:, 1] + d) / c
                return ground_z, {
                    "method": "ransac",
                    "plane_abcd": [float(a), float(b), float(c), float(d)],
                }
            except Exception as e:
                z_global = float(np.percentile(pts[:, 2], self.config.percentile_p))
                return np.full(pts.shape[0], z_global), {
                    "method": "ransac_fallback_percentile",
                    "fallback_reason": str(e),
                    "ground_z_global": z_global,
                }
        else:  # percentile
            z_global = float(np.percentile(pts[:, 2], self.config.percentile_p))
            return np.full(pts.shape[0], z_global), {
                "method": "percentile",
                "percentile_p": self.config.percentile_p,
                "ground_z_global": z_global,
            }

    def extract(self, point_cloud_ego: np.ndarray) -> dict:
        """Run the pillar pipeline. Input columns 0:3 are x, y, z; remaining
        columns (intensity, etc.) are preserved on the foreground subset.
        """
        if point_cloud_ego.ndim != 2 or point_cloud_ego.shape[1] < 3:
            raise ValueError(f"point_cloud_ego must be (N, ≥3); got {point_cloud_ego.shape}")

        N = point_cloud_ego.shape[0]
        pts = point_cloud_ego[:, :3].astype(np.float64, copy=False)

        # ---- ground estimation ----
        t0 = time.perf_counter()
        ground_z_at_pts, ground_info = self._estimate_ground_z_at_points(pts)
        height_above_ground = pts[:, 2] - ground_z_at_pts
        t1 = time.perf_counter()

        # ---- pillar assignment ----
        px, py = self.config.pillar_size_xy
        ix = np.floor(pts[:, 0] / px).astype(np.int64)
        iy = np.floor(pts[:, 1] / py).astype(np.int64)
        # Compose a single int64 key. Range guard: ix, iy ∈ [-1000, 1000] for ego
        # frame within ±300 m at smallest pillar 0.3 m → 1000² < 2³¹.
        OFFSET = 1024
        STRIDE = 2048
        key = (ix + OFFSET).clip(0, STRIDE - 1) * STRIDE + (iy + OFFSET).clip(0, STRIDE - 1)
        unique_keys, inverse = np.unique(key, return_inverse=True)
        n_pillars = int(unique_keys.size)

        # ---- per-pillar max height ----
        pillar_max_h = np.full(n_pillars, -np.inf, dtype=np.float64)
        np.maximum.at(pillar_max_h, inverse, height_above_ground)

        is_fg_pillar = pillar_max_h > self.config.z_threshold
        n_fg_pillars = int(is_fg_pillar.sum())
        point_keep = is_fg_pillar[inverse]
        foreground_pcd = point_cloud_ego[point_keep]
        background_mask = ~point_keep
        t2 = time.perf_counter()

        return {
            "foreground_pcd": foreground_pcd,
            "background_mask": background_mask,
            "n_input_points": int(N),
            "n_foreground_points": int(point_keep.sum()),
            "n_pillars_total": n_pillars,
            "n_pillars_foreground": n_fg_pillars,
            "foreground_ratio": float(point_keep.sum() / N) if N else 0.0,
            "ground_info": ground_info,
            "timing": {
                "ground_estimation": float(t1 - t0),
                "pillar_assignment_and_filter": float(t2 - t1),
                "total": float(t2 - t0),
            },
        }
