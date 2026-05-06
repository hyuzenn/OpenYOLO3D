"""HDBSCAN-based LiDAR proposal generator.

Replaces Mask3D's class-agnostic proposal stage for outdoor scenes. This
module is **adapter scope only**: it consumes a point cloud in ego frame
and emits per-cluster geometry. No 2D detection, no fusion, no reliability
scoring — those belong to later W-tasks.

Coordinate convention follows ``dataloaders/nuscenes_loader.py``:
the input ``point_cloud_ego`` is in the EGO frame at the LIDAR_TOP timestamp.
ego z=0 is approximately the LiDAR sensor mount height; nuScenes ground
sits around z ≈ -1.5 to -1.8 m, so a default ground threshold of -1.4 m
removes the bulk of road points without clipping curbs/low obstacles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import open3d as o3d

import hdbscan


@dataclass
class ClusteringConfig:
    min_cluster_size: int = 20
    min_samples: int = 5
    cluster_selection_epsilon: float = 0.5
    # Ground filter mode: "z_threshold" | "ransac" | "percentile" | None
    ground_filter: Optional[str] = "z_threshold"
    # 'z_threshold' params
    ground_z_max: float = -1.4
    # 'ransac' params (open3d segment_plane)
    ransac_distance_threshold: float = 0.2
    ransac_n: int = 3
    ransac_iterations: int = 1000
    # 'percentile' params
    percentile_p: float = 15.0
    # ego-frame xy radius cutoff (m); None = no filter
    max_distance: Optional[float] = 100.0


class LiDARProposalGenerator:
    """Class-agnostic 3D proposal generator using HDBSCAN.

    Pipeline (strict order):
      1. Ground filter (z < ground_z_max removed when 'z_threshold')
      2. Distance filter (xy-radius > max_distance removed when set)
      3. HDBSCAN fit_predict on (x, y, z)
      4. Per-cluster centroid, axis-aligned bbox, point count
      5. Drop clusters whose final size is below min_cluster_size (HDBSCAN
         already enforces this internally — kept here as a guard against
         post-filter shrinkage in future variants)
    """

    def __init__(
        self,
        min_cluster_size: int = 20,
        min_samples: int = 5,
        cluster_selection_epsilon: float = 0.5,
        ground_filter: Optional[str] = "z_threshold",
        ground_z_max: float = -1.4,
        ransac_distance_threshold: float = 0.2,
        ransac_n: int = 3,
        ransac_iterations: int = 1000,
        percentile_p: float = 15.0,
        max_distance: Optional[float] = 100.0,
    ):
        self.config = ClusteringConfig(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            ground_filter=ground_filter,
            ground_z_max=ground_z_max,
            ransac_distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            ransac_iterations=ransac_iterations,
            percentile_p=percentile_p,
            max_distance=max_distance,
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def generate(self, point_cloud_ego: np.ndarray) -> dict:
        """Run the full pipeline on one frame's ego-frame point cloud.

        Args:
            point_cloud_ego: (N, ≥3) array; columns 0:3 are x, y, z. Extra
                columns (e.g. intensity) are ignored.

        Returns:
            Dict with keys ``cluster_ids`` (N,), ``n_clusters``,
            ``cluster_centroids`` (n_clusters, 3), ``cluster_sizes`` (n_clusters,),
            ``cluster_bbox`` (n_clusters, 6), ``timing`` dict, ``noise_ratio``,
            ``ground_filtered_ratio``, ``distance_filtered_ratio``.
        """
        if point_cloud_ego.ndim != 2 or point_cloud_ego.shape[1] < 3:
            raise ValueError(f"point_cloud_ego must be (N, ≥3); got {point_cloud_ego.shape}")

        N = point_cloud_ego.shape[0]
        pts = point_cloud_ego[:, :3].astype(np.float64, copy=False)

        # ---- pre-process ----
        t0 = time.perf_counter()
        keep_mask = np.ones(N, dtype=bool)

        ground_drop = 0
        if self.config.ground_filter == "z_threshold":
            ground_keep = pts[:, 2] >= self.config.ground_z_max
            ground_drop = int((~ground_keep).sum())
            keep_mask &= ground_keep
        elif self.config.ground_filter == "ransac":
            # open3d plane RANSAC. Inliers are treated as ground and removed.
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            try:
                _plane_model, inliers = pcd.segment_plane(
                    distance_threshold=self.config.ransac_distance_threshold,
                    ransac_n=self.config.ransac_n,
                    num_iterations=self.config.ransac_iterations,
                )
                inlier_mask = np.zeros(N, dtype=bool)
                inlier_mask[np.asarray(inliers, dtype=np.int64)] = True
                ground_drop = int(inlier_mask.sum())
                keep_mask &= ~inlier_mask
            except Exception:
                # Plane fit can fail on near-degenerate inputs — fall through
                # leaving keep_mask unmodified. ground_drop stays 0.
                pass
        elif self.config.ground_filter == "percentile":
            z_threshold = float(np.percentile(pts[:, 2], self.config.percentile_p))
            ground_keep = pts[:, 2] >= z_threshold
            ground_drop = int((~ground_keep).sum())
            keep_mask &= ground_keep

        distance_drop = 0
        if self.config.max_distance is not None:
            xy_r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            dist_keep = xy_r <= self.config.max_distance
            distance_drop = int(((~dist_keep) & keep_mask).sum())
            keep_mask &= dist_keep

        pts_f = pts[keep_mask]
        t1 = time.perf_counter()

        # ---- HDBSCAN ----
        full_labels = np.full(N, -1, dtype=np.int64)
        if pts_f.shape[0] >= self.config.min_cluster_size:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
                cluster_selection_epsilon=self.config.cluster_selection_epsilon,
                core_dist_n_jobs=1,  # avoid oversubscription on shared nodes
            )
            labels_f = clusterer.fit_predict(pts_f)
            full_labels[keep_mask] = labels_f
        t2 = time.perf_counter()

        # ---- post-process: per-cluster stats ----
        unique_ids = sorted(int(c) for c in set(full_labels.tolist()) if c != -1)
        centroids = np.zeros((len(unique_ids), 3), dtype=np.float64)
        bboxes = np.zeros((len(unique_ids), 6), dtype=np.float64)
        sizes = np.zeros((len(unique_ids),), dtype=np.int64)
        # Re-index clusters densely (0..K-1) so downstream code can use
        # cluster_id as an array index without sparsity surprises.
        relabel = {old: new for new, old in enumerate(unique_ids)}
        for old, new in relabel.items():
            mask = full_labels == old
            cluster_pts = pts[mask]
            centroids[new] = cluster_pts.mean(axis=0)
            bboxes[new] = np.concatenate([cluster_pts.min(axis=0), cluster_pts.max(axis=0)])
            sizes[new] = int(mask.sum())
            full_labels[mask] = new

        # Guard: drop any cluster below min_cluster_size after re-index.
        if (sizes < self.config.min_cluster_size).any():
            keep_clusters = sizes >= self.config.min_cluster_size
            kept_indices = np.where(keep_clusters)[0]
            id_remap = {int(old): int(new) for new, old in enumerate(kept_indices)}
            new_labels = np.full(N, -1, dtype=np.int64)
            for old, new in id_remap.items():
                new_labels[full_labels == old] = new
            full_labels = new_labels
            centroids = centroids[keep_clusters]
            bboxes = bboxes[keep_clusters]
            sizes = sizes[keep_clusters]
        t3 = time.perf_counter()

        n_in_filtered = int(keep_mask.sum())
        n_noise = int((full_labels == -1).sum() - (~keep_mask).sum())
        noise_ratio = (n_noise / n_in_filtered) if n_in_filtered > 0 else 0.0

        return {
            "cluster_ids": full_labels,
            "n_clusters": int(centroids.shape[0]),
            "cluster_centroids": centroids,
            "cluster_sizes": sizes,
            "cluster_bbox": bboxes,
            "timing": {
                "preprocess": float(t1 - t0),
                "hdbscan": float(t2 - t1),
                "postprocess": float(t3 - t2),
                "total": float(t3 - t0),
            },
            "noise_ratio": float(noise_ratio),
            "ground_filtered_ratio": float(ground_drop / N) if N > 0 else 0.0,
            "distance_filtered_ratio": float(distance_drop / N) if N > 0 else 0.0,
            "n_input_points": int(N),
            "n_clustered_points": int((full_labels != -1).sum()),
        }
