"""Mask3D proposal runner — wraps Network_3D and replicates the OpenYolo3D
post-filter (score threshold + NMS) so the (mask, score) tensors we compare
match what the real pipeline would have produced.

Saves a temp PLY per call, since prepare_data() reads from disk. Uses the
existing _save_ply helper from the nuScenes adapter to keep the on-disk
layout identical to the smoke test.
"""

from __future__ import annotations

import os
import os.path as osp
import time

import numpy as np
import open3d as o3d
import torch

from utils.utils_3d import Network_3D
from utils import apply_nms

from adapters.nuscenes_to_openyolo3d import _save_ply, _color_lidar_via_camera


class Mask3DProposalRunner:
    """Holds a single Network_3D model in memory; one call per sample."""

    def __init__(self, openyolo3d_config: dict, work_root: str):
        self.cfg = openyolo3d_config
        self.score_th = float(openyolo3d_config["network3d"]["th"])
        self.nms_th = float(openyolo3d_config["network3d"]["nms"])
        os.makedirs(work_root, exist_ok=True)
        self.work_root = work_root
        self.network = Network_3D(openyolo3d_config)

    def _write_ply(self, pc_ego_xyz: np.ndarray, sample_token: str,
                   image=None, K=None, T_cam_to_ego=None) -> str:
        """Write a coloured PLY. When (image, K, T_cam_to_ego) are supplied we
        project LiDAR into that camera and copy per-pixel RGB onto the points
        — this matches the Tier-1 / smoke-test pipeline exactly. When omitted,
        all points get gray-128, which is a no-op variant useful for diagnostic
        comparisons (and matches what an upstream caller without camera access
        can do).
        """
        ply_path = osp.join(self.work_root, f"{sample_token}.ply")
        if image is not None and K is not None and T_cam_to_ego is not None:
            colors = _color_lidar_via_camera(pc_ego_xyz, image, K, T_cam_to_ego)
        else:
            colors = np.full((pc_ego_xyz.shape[0], 3), 128, dtype=np.uint8)
        _save_ply(pc_ego_xyz[:, :3], colors, ply_path)
        return ply_path

    def run(self, pc_ego_xyz: np.ndarray, sample_token: str,
            image=None, K=None, T_cam_to_ego=None) -> dict:
        """Run Mask3D + the OpenYolo3D-equivalent score/NMS filter.

        Returns:
            dict with keys 'masks' (N_pts, N_inst) bool, 'scores' (N_inst,) float,
            'n_instances' int, 'timing' dict, 'ply_path' str (caller can clean up).
        """
        t0 = time.perf_counter()
        ply_path = self._write_ply(pc_ego_xyz, sample_token, image, K, T_cam_to_ego)
        t_write = time.perf_counter() - t0

        t1 = time.perf_counter()
        # network.get_class_agnostic_masks returns (masks, scores) where masks
        # is a tensor with shape (N_pts, N_inst_raw) per the OpenYolo3D pattern.
        masks, scores = self.network.get_class_agnostic_masks(ply_path, "point cloud")
        t_infer = time.perf_counter() - t1

        # Replicate OpenYolo3D.predict() filtering exactly.
        t2 = time.perf_counter()
        if scores.numel() == 0 or masks.shape[1] == 0:
            kept_masks = torch.zeros((masks.shape[0], 0), dtype=torch.bool)
            kept_scores = torch.zeros((0,), dtype=torch.float32)
        else:
            keep_score = scores >= self.score_th
            if keep_score.sum() == 0:
                kept_masks = torch.zeros((masks.shape[0], 0), dtype=torch.bool)
                kept_scores = torch.zeros((0,), dtype=torch.float32)
            else:
                m_after_score = masks[:, keep_score].cuda()
                s_after_score = scores[keep_score].cuda()
                keep_nms = apply_nms(m_after_score, s_after_score, self.nms_th)
                # mirrors utils/__init__.py:118
                kept_masks = (
                    masks.cpu().permute(1, 0)[keep_score][keep_nms].permute(1, 0)
                )
                kept_scores = scores.cpu()[keep_score][keep_nms]
        t_filter = time.perf_counter() - t2

        masks_np = kept_masks.cpu().numpy().astype(bool)
        scores_np = kept_scores.cpu().numpy().astype(float)

        return {
            "masks": masks_np,
            "scores": scores_np,
            "n_instances": int(masks_np.shape[1]),
            "timing": {
                "ply_write_s": float(t_write),
                "mask3d_infer_s": float(t_infer),
                "post_filter_s": float(t_filter),
                "total_s": float(time.perf_counter() - t0),
            },
            "ply_path": ply_path,
        }

    def cleanup_ply(self, ply_path: str) -> None:
        try:
            os.remove(ply_path)
        except FileNotFoundError:
            pass
