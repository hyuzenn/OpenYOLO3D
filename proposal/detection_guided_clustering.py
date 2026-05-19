"""DetectionGuidedClusterer — composes FrustumExtractor + PillarForegroundExtractor
+ HDBSCAN to produce per-frustum 3D proposals.

Each 2D detection becomes a frustum; per-frustum we apply β1 best pillar
foreground (so ground/road points inside the frustum don't dominate
clustering), then HDBSCAN. Each cluster within a frustum is one 3D proposal.

The proposal index space is global (across cameras), with a back-reference
to (cam, det_idx, class, score). Proposals from different cameras can
overlap in 3D — we don't dedup here; that decision belongs to a downstream
fusion step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from preprocessing.detection_frustum import FrustumExtractor
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator


@dataclass
class ClustererConfig:
    min_points_per_frustum: int = 10


class DetectionGuidedClusterer:
    def __init__(
        self,
        frustum_extractor: FrustumExtractor,
        pillar_extractor: PillarForegroundExtractor,
        hdbscan_generator: LiDARProposalGenerator,
        min_points_per_frustum: int = 10,
    ):
        self.frustum_extractor = frustum_extractor
        self.pillar_extractor = pillar_extractor
        self.hdbscan_generator = hdbscan_generator
        self.config = ClustererConfig(min_points_per_frustum=int(min_points_per_frustum))

    @property
    def config_dict(self) -> dict:
        return {
            "min_points_per_frustum": self.config.min_points_per_frustum,
            "frustum": self.frustum_extractor.config_dict,
            "pillar": self.pillar_extractor.config_dict,
        }

    def generate(
        self,
        point_cloud_ego: np.ndarray,
        detections_per_cam: dict,
        intrinsics_per_cam: dict,
        cam_to_ego_per_cam: dict,
        image_hw_per_cam: dict,
    ) -> dict:
        N_pts = point_cloud_ego.shape[0]
        t0 = time.perf_counter()

        fr = self.frustum_extractor.extract_frustums(
            detections_per_cam, intrinsics_per_cam, cam_to_ego_per_cam,
            image_hw_per_cam, point_cloud_ego,
        )
        t_frustum = time.perf_counter() - t0

        proposals = []
        n_frustums_skipped_low = 0
        n_frustums_skipped_no_cluster = 0
        n_frustums_with_clusters = 0
        per_frustum_records = []
        t_cluster_start = time.perf_counter()
        for f in fr["frustums"]:
            n_pts = f["n_points"]
            if n_pts < self.config.min_points_per_frustum:
                n_frustums_skipped_low += 1
                per_frustum_records.append({
                    **{k: f[k] for k in ("cam", "det_idx", "class", "score",
                                            "n_points", "depth_min", "depth_max")},
                    "skipped_reason": "low_points",
                    "n_proposals": 0,
                })
                continue

            # β1 pillar foreground inside the frustum
            fg_out = self.pillar_extractor.extract(f["frustum_points_ego"])
            fg_pcd = fg_out["foreground_pcd"]
            if fg_pcd.shape[0] < self.config.min_points_per_frustum:
                n_frustums_skipped_low += 1
                per_frustum_records.append({
                    **{k: f[k] for k in ("cam", "det_idx", "class", "score",
                                            "n_points", "depth_min", "depth_max")},
                    "skipped_reason": "low_foreground",
                    "n_foreground": int(fg_pcd.shape[0]),
                    "n_proposals": 0,
                })
                continue

            # HDBSCAN inside the foreground
            h_out = self.hdbscan_generator.generate(fg_pcd)
            n_clusters = int(h_out["n_clusters"])
            if n_clusters == 0:
                n_frustums_skipped_no_cluster += 1
                per_frustum_records.append({
                    **{k: f[k] for k in ("cam", "det_idx", "class", "score",
                                            "n_points", "depth_min", "depth_max")},
                    "skipped_reason": "no_cluster",
                    "n_foreground": int(fg_pcd.shape[0]),
                    "n_proposals": 0,
                })
                continue

            # Each HDBSCAN cluster within this frustum becomes one proposal.
            n_frustums_with_clusters += 1
            cluster_ids_in_fg = h_out["cluster_ids"]   # (n_fg,) int
            # Map foreground points back to original frustum indices, then to
            # original ego-frame indices.
            # _save: index trail = orig_pc_indices[ frustum_subset ][ foreground_subset ]
            # We just need a bool mask over the *full* PC for matching.
            # Reconstruct via a row-wise mask chain.
            # Step 1: per-frustum point indices (from FrustumExtractor)
            frustum_idx = f["frustum_point_indices"]
            # Step 2: foreground mask over the frustum subset
            # The pillar extractor doesn't expose the keep mask explicitly; we
            # re-derive it by detecting which rows of frustum_points_ego are in
            # foreground_pcd. Cheaper alternative: use the lengths and the
            # extractor's internal keep mask if exposed. For now, call extract
            # again with a wrapper that returns the mask.
            # → For simplicity and speed, we trust the extractor's order is
            #   stable: foreground_pcd = frustum_points_ego[fg_keep]. Pillar
            #   extractor preserves point order on the kept subset, so we can
            #   recover fg_keep by matching coordinates.
            # Cheaper: we ask the extractor to *also* return a keep mask via
            # its public output. PillarForegroundExtractor returns
            # ``background_mask``. So fg_keep = ~background_mask of LENGTH
            # equal to len(frustum_points_ego).
            # The current PillarForegroundExtractor does return background_mask.
            fg_keep = ~fg_out["background_mask"]
            # composite mapping: original-pc-index for each foreground point
            orig_indices = frustum_idx[fg_keep]  # (n_fg,)

            for cid in range(n_clusters):
                cluster_mask_fg = cluster_ids_in_fg == cid
                if not cluster_mask_fg.any():
                    continue
                orig_for_cluster = orig_indices[cluster_mask_fg]
                pts_for_cluster = point_cloud_ego[orig_for_cluster, :3]
                proposals.append({
                    "cam": f["cam"],
                    "det_idx": f["det_idx"],
                    "class": f["class"],
                    "score": f["score"],
                    "n_points": int(cluster_mask_fg.sum()),
                    "centroid_ego": pts_for_cluster.mean(axis=0).tolist() if pts_for_cluster.size else None,
                    "orig_pc_indices": orig_for_cluster,
                })

            per_frustum_records.append({
                **{k: f[k] for k in ("cam", "det_idx", "class", "score",
                                        "n_points", "depth_min", "depth_max")},
                "n_foreground": int(fg_pcd.shape[0]),
                "n_proposals": int(n_clusters),
            })

        t_cluster = time.perf_counter() - t_cluster_start
        t_total = time.perf_counter() - t0

        # Build (N_pts, N_proposals) bool mask for downstream matching.
        if proposals:
            masks = np.zeros((N_pts, len(proposals)), dtype=bool)
            for j, p in enumerate(proposals):
                masks[p["orig_pc_indices"], j] = True
        else:
            masks = np.zeros((N_pts, 0), dtype=bool)

        return {
            "n_frustums": fr["n_frustums"],
            "n_frustums_with_lidar": fr["n_with_points"],
            "n_frustums_with_clusters": n_frustums_with_clusters,
            "n_frustums_skipped_low_points": n_frustums_skipped_low,
            "n_frustums_skipped_no_cluster": n_frustums_skipped_no_cluster,
            "n_proposals_total": len(proposals),
            "proposals_meta": [
                {k: v for k, v in p.items() if k != "orig_pc_indices"} for p in proposals
            ],
            "proposal_masks": masks,
            "per_frustum_records": per_frustum_records,
            "timing": {
                "frustum_extraction_s": float(t_frustum),
                "clustering_s": float(t_cluster),
                "total_s": float(t_total),
            },
        }
