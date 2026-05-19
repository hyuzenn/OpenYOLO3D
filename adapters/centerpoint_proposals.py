"""CenterPoint LiDAR proposal adapter.

Wraps mmdet3d's CenterPoint (Voxel + SECFPN + CircleNMS, nuScenes-pretrained,
10-class) so it emits the same record shape as the geometric LiDARProposalGenerator:
list of proposals with (cls, score, centroid, points-in-box).

Coordinate convention:
  - Input ``point_cloud_ego`` is in the EGO frame (per nuScenes_loader).
  - CenterPoint was trained in LIDAR_TOP frame, so we transform ego→lidar
    via ``inv(T_lidar_to_ego)`` before inference, run, then transform the
    resulting boxes lidar→ego before returning. Caller's downstream
    matching code is in ego frame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

# mmdet3d / torch loaded lazily so login-node imports don't fail
def _load_mmdet3d():
    from mmdet3d.apis import init_model, inference_detector
    return init_model, inference_detector


# nuScenes 10-class names (CenterPoint trained order — confirmed by sanity)
NUSC_10 = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)


@dataclass
class CenterPointConfig:
    config_path: str
    checkpoint_path: str
    score_threshold: float = 0.10
    nms_iou_threshold: float = 0.20    # informational; CircleNMS already in head


class CenterPointProposalGenerator:
    """One model instance, many ``generate(...)`` calls.

    Performance: model init ~5s on A100, per-sample inference ~0.5-2s.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        score_threshold: float = 0.10,
        nms_iou_threshold: float = 0.20,
        device: str = "cuda:0",
    ):
        self.config = CenterPointConfig(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            score_threshold=float(score_threshold),
            nms_iou_threshold=float(nms_iou_threshold),
        )
        init_model, inference_detector = _load_mmdet3d()
        self._inference_detector = inference_detector
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.device = device

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def update_thresholds(self, score_threshold: float, nms_iou_threshold: float):
        """Same model, different post-filter — used during the score-threshold sweep."""
        self.config.score_threshold = float(score_threshold)
        self.config.nms_iou_threshold = float(nms_iou_threshold)

    def generate(
        self,
        point_cloud_ego: np.ndarray,
        T_lidar_to_ego: np.ndarray,
        tmp_bin_path: str,
    ) -> dict:
        """Run CenterPoint on one sample.

        Args:
            point_cloud_ego: (N, ≥3) array; columns 0:3 = x,y,z, column 3 = intensity.
            T_lidar_to_ego: (4, 4) lidar→ego transform.
            tmp_bin_path: writable .bin path; nuScenes mmdet3d inference path
                expects a file. Caller manages cleanup.
        """
        N = point_cloud_ego.shape[0]
        t0 = time.perf_counter()

        # ego → lidar
        T_inv = np.linalg.inv(T_lidar_to_ego)
        pts_h = np.concatenate([point_cloud_ego[:, :3], np.ones((N, 1))], axis=1)
        pts_lidar_xyz = (T_inv @ pts_h.T).T[:, :3]
        intensity = point_cloud_ego[:, 3] if point_cloud_ego.shape[1] >= 4 else np.zeros(N)
        pc_5 = np.zeros((N, 5), dtype=np.float32)
        pc_5[:, :3] = pts_lidar_xyz
        pc_5[:, 3] = intensity
        pc_5[:, 4] = 0.0  # single-keyframe time delta
        pc_5.tofile(tmp_bin_path)
        t1 = time.perf_counter()

        # inference
        result, _data = self._inference_detector(self.model, tmp_bin_path)
        t2 = time.perf_counter()

        # parse
        if isinstance(result, list):
            result = result[0]
        pred = result.pred_instances_3d
        bboxes_lidar = pred.bboxes_3d.tensor.cpu().numpy()  # (M, 7) or (M, 9)
        scores = pred.scores_3d.cpu().numpy()
        labels = pred.labels_3d.cpu().numpy().astype(np.int64)

        # filter by score threshold
        keep = scores >= self.config.score_threshold
        bboxes_lidar = bboxes_lidar[keep]
        scores = scores[keep]
        labels = labels[keep]

        # transform box centers lidar → ego
        if len(bboxes_lidar):
            centers_lidar = bboxes_lidar[:, :3]
            ones = np.ones((centers_lidar.shape[0], 1))
            centers_ego = (T_lidar_to_ego @ np.concatenate(
                [centers_lidar, ones], axis=1).T).T[:, :3]
            # yaw also rotates with extrinsic — for matching we use box
            # CONTAINMENT in ego frame, computed from (center_ego, size, yaw_lidar).
            # nuScenes lidar/ego differ only by a small mounting yaw + translation;
            # box dims (w,l,h) are frame-invariant, the yaw angle does shift.
            # For matching via points-in-box we need an axis-aligned-or-rotated
            # box specified in ego. We rebuild using nuScenes' Box class downstream
            # rather than try to derive yaw_ego analytically here.
        else:
            centers_ego = np.zeros((0, 3))

        # determine which ego-frame points fall inside each box
        # (do this via the matching primitive, not here — kept lazy)
        proposals = []
        for j in range(len(bboxes_lidar)):
            b = bboxes_lidar[j]
            cls_idx = int(labels[j])
            proposals.append({
                "cls_idx": cls_idx,
                "cls_name": NUSC_10[cls_idx] if 0 <= cls_idx < len(NUSC_10) else f"cls_{cls_idx}",
                "score": float(scores[j]),
                "bbox_lidar": b.tolist(),     # [x, y, z, w/dx, l/dy, h/dz, yaw, (vx, vy)]
                "centroid_ego": centers_ego[j].tolist(),
            })
        t3 = time.perf_counter()

        return {
            "proposals": proposals,
            "n_proposals": len(proposals),
            "n_proposals_pre_threshold": int(keep.size),
            "score_threshold_applied": self.config.score_threshold,
            "timing": {
                "preprocess_s": float(t1 - t0),
                "inference_s": float(t2 - t1),
                "postprocess_s": float(t3 - t2),
                "total_s": float(t3 - t0),
            },
        }
