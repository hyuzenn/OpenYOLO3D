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
    from mmdet3d.apis import init_model
    return init_model


# CenterPoint label index → class name, in the EXACT order the checkpoint's
# multi-task CenterHead emits labels_3d. This is the top-level `class_names`
# of centerpoint_voxel0075_..._nus-3d.py (tasks flattened in order:
# car | truck,construction_vehicle | bus,trailer | barrier | motorcycle,bicycle
# | pedestrian,traffic_cone). NOTE: this is NOT the canonical nuScenes-devkit
# alphabetical-ish order — Task 2.5 found the previous tuple was permuted,
# which mislabeled construction_vehicle/trailer/barrier/bicycle/pedestrian/
# traffic_cone and zeroed their AP. Verified against the config head tasks.
NUSC_10 = (
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
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
        init_model = _load_mmdet3d()
        self.model = init_model(config_path, checkpoint_path, device=device)
        self.device = device
        self._pipeline, self._box_type_3d, self._box_mode_3d = self._build_pipeline()

    # -- inference pipeline ------------------------------------------------
    # We do NOT use mmdet3d.apis.inference_detector. It composes the model's
    # *full* configured test pipeline (mmdet3d/apis/inference.py:142-176), which
    # for this checkpoint is
    #     LoadPointsFromFile -> LoadPointsFromMultiSweeps
    #                        -> MultiScaleFlipAug3D -> Pack3DDetInputs
    # and the cloud we hand it has ALREADY been aggregated over 10 sweeps by the
    # nuScenes devkit (dataloaders/nuscenes_loader.py:_load_lidar_ego). Running
    # LoadPointsFromMultiSweeps a second time is wrong twice over: with no
    # 'lidar_sweeps' key in the data dict it (a) zeroes the per-point Δt channel
    # unconditionally (mmdet3d/datasets/transforms/loading.py:413) and
    # (b) appends 9 further copies of the whole cloud via pad_empty_sweeps
    # (loading.py:416-422). Measured: 20,000 points -> 199,892, and the 10
    # distinct Δt values collapse to a single 0.0 — i.e. the checkpoint never
    # received the temporal channel it was trained with.
    #
    # So we compose exactly that pipeline MINUS that one stage and drive
    # model.test_step ourselves, mirroring inference_detector otherwise line for
    # line. Decode, score threshold, circle NMS, top-K, class mapping, yaw and
    # the z convention all stay inside the official model and are untouched.
    _SKIP_TRANSFORMS = ("LoadPointsFromMultiSweeps",)

    def _build_pipeline(self):
        from copy import deepcopy
        from mmengine.dataset import Compose
        from mmdet3d.structures import get_box_type

        cfg_pipeline = self.model.cfg.test_dataloader.dataset.pipeline
        kept = [t for t in deepcopy(cfg_pipeline)
                if t["type"] not in self._SKIP_TRANSFORMS]
        # Recorded so a validation probe can assert on the real runtime pipeline
        # rather than on this comment.
        self.pipeline_steps = [t["type"] for t in kept]
        self.pipeline_dropped = [t["type"] for t in cfg_pipeline
                                 if t["type"] in self._SKIP_TRANSFORMS]
        box_type_3d, box_mode_3d = get_box_type(
            self.model.cfg.test_dataloader.dataset.box_type_3d)
        return Compose(kept), box_type_3d, box_mode_3d

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
        # Time-delta channel. With multi-sweep input the loader supplies a 5th
        # column = per-point Δt (s); CenterPoint's 10-sweep-trained checkpoint
        # uses it. Single-sweep input has no 5th column → Δt=0 (keyframe).
        pc_5[:, 4] = point_cloud_ego[:, 4] if point_cloud_ego.shape[1] >= 5 else 0.0
        pc_5.tofile(tmp_bin_path)
        t1 = time.perf_counter()

        # inference — same steps inference_detector performs, minus the
        # duplicate LoadPointsFromMultiSweeps (see _build_pipeline).
        import torch
        from mmengine.dataset import pseudo_collate

        data = self._pipeline(dict(
            lidar_points=dict(lidar_path=tmp_bin_path),
            timestamp=1,
            axis_align_matrix=np.eye(4),
            box_type_3d=self._box_type_3d,
            box_mode_3d=self._box_mode_3d))
        with torch.no_grad():
            result = self.model.test_step(pseudo_collate([data]))
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

        # mmdet3d LiDARInstance3DBoxes store the z coordinate at the box
        # BOTTOM (origin (0.5, 0.5, 0): bottom_center == tensor[:, :3]). Every
        # downstream consumer treats this value as the geometric centre, so we
        # convert here to the gravity (geometric) centre exactly as
        # LiDARInstance3DBoxes.gravity_center does: z_centre = z_bottom + h/2,
        # where h == z_size == column 5. Only column 2 (z) changes; x, y,
        # dims (3:6), yaw (6) and velocity (7:9) are left byte-identical. This
        # single edit corrects both `centers_lidar` (centroid_ego) and the
        # serialized `bbox_lidar`, since both derive from `bboxes_lidar`.
        if len(bboxes_lidar):
            bboxes_lidar[:, 2] += bboxes_lidar[:, 5] * 0.5

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
