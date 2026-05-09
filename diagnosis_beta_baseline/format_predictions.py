"""Convert OpenYOLO3D mask predictions → nuScenes detection boxes.

OpenYOLO3D predict() returns:
    {scene_name: (predicted_masks, predicated_classes, predicated_scores)}

  predicted_masks: (N_points, N_instances)  bool / float
  predicated_classes: (N_instances,)        int — index into text_prompts
                                              the LAST index (= num_classes-1)
                                              is the catch-all "no class"
  predicated_scores: (N_instances,)         float

This module extracts a 3D box per instance from the masked points (which
live in EGO frame, since the adapter writes lidar.ply in ego frame), then
applies ego_pose to get global-frame DetectionBox dicts compatible with
nuScenes-devkit `DetectionBox.deserialize`.

Heading is set to identity (Quaternion(1,0,0,0)) — OpenYOLO3D produces
masks, not oriented bboxes; mAOE will be poor by construction. mAP@center
distance is heading-independent.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d
import torch
from pyquaternion import Quaternion

from diagnosis_beta_baseline import NUSCENES_10_CLASS


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _read_ply_points(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    return np.asarray(pcd.points, dtype=np.float64)


def _aabb_size(points_xyz: np.ndarray) -> np.ndarray:
    """nuScenes wlh = (width along y, length along x, height along z).

    For an axis-aligned box from a point cluster:
      length = extent in x (forward),
      width  = extent in y (lateral),
      height = extent in z (vertical).
    """
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-3)  # floor to avoid 0-size box
    return np.array([extents[1], extents[0], extents[2]], dtype=np.float64)


def _ego_to_global(center_ego: np.ndarray, ego_pose_4x4: np.ndarray) -> np.ndarray:
    h = np.array([center_ego[0], center_ego[1], center_ego[2], 1.0], dtype=np.float64)
    return (ego_pose_4x4 @ h)[:3]


def predictions_to_detection_boxes(
    sample_token: str,
    pred_tuple,
    ply_points_ego: np.ndarray,
    ego_pose_4x4: np.ndarray,
    text_prompts=NUSCENES_10_CLASS,
    min_points_per_instance: int = 5,
):
    """Convert one sample's OpenYOLO3D prediction tuple to a list of dicts
    matching `DetectionBox.serialize()` shape (so they round-trip through
    `DetectionBox.deserialize`).
    """
    masks, classes, scores = pred_tuple
    masks = _to_numpy(masks).astype(bool)         # (N_points, N_instances)
    classes = _to_numpy(classes).astype(np.int64) # (N_instances,)
    scores = _to_numpy(scores).astype(np.float64) # (N_instances,)

    n_classes_with_bg = len(text_prompts) + 1  # last = "no class"
    boxes = []
    n_dropped_bg = 0
    n_dropped_small = 0

    for i in range(classes.shape[0]):
        cls_idx = int(classes[i])
        if cls_idx >= len(text_prompts):
            n_dropped_bg += 1
            continue
        mask_i = masks[:, i] if masks.ndim == 2 else masks
        if mask_i.shape[0] != ply_points_ego.shape[0]:
            raise RuntimeError(
                f"mask/points size mismatch: mask {mask_i.shape[0]} vs ply {ply_points_ego.shape[0]}"
            )
        pts = ply_points_ego[mask_i]
        if pts.shape[0] < min_points_per_instance:
            n_dropped_small += 1
            continue

        center_ego = pts.mean(axis=0)
        wlh = _aabb_size(pts)
        center_global = _ego_to_global(center_ego, ego_pose_4x4)
        # ego rotation extracted from ego_pose; identity local heading.
        ego_rot_q = Quaternion(matrix=ego_pose_4x4[:3, :3])
        global_rot = ego_rot_q  # local heading = identity → global = ego rotation
        boxes.append({
            "sample_token": sample_token,
            "translation": [float(x) for x in center_global],
            "size": [float(x) for x in wlh],  # w, l, h
            "rotation": [float(x) for x in [global_rot.w, global_rot.x, global_rot.y, global_rot.z]],
            "velocity": [0.0, 0.0],
            "ego_translation": [float(ego_pose_4x4[i_, 3]) for i_ in range(3)],
            "num_pts": int(pts.shape[0]),
            "detection_name": text_prompts[cls_idx],
            "detection_score": float(scores[i]),
            "attribute_name": "",
        })

    return boxes, {
        "n_pred_total": int(classes.shape[0]),
        "n_pred_kept": len(boxes),
        "n_dropped_bg": int(n_dropped_bg),
        "n_dropped_small": int(n_dropped_small),
    }


def gt_to_detection_boxes(sample_token: str, gt_boxes_global: list, ego_pose_4x4=None):
    """Convert NuScenesLoader gt_boxes (already in global frame) →
    DetectionBox-shaped dicts. Filter to the 10 official detection classes.

    `ego_pose_4x4` is the ego-to-global transform at this sample's
    LIDAR_TOP timestamp; if provided, its translation column is written
    into each GT dict's `ego_translation` field, matching the convention
    used by `predictions_to_detection_boxes`. evaluate_nuscenes._filter_by_range
    needs this to compute correct ego-distance for class_range filtering.
    """
    from nuscenes.eval.detection.utils import category_to_detection_name

    if ego_pose_4x4 is not None:
        ego_translation = [float(ego_pose_4x4[i, 3]) for i in range(3)]
    else:
        ego_translation = None

    out = []
    for gt in gt_boxes_global:
        det_name = category_to_detection_name(gt["category"])
        if det_name is None or det_name not in NUSCENES_10_CLASS:
            continue
        rot = gt["rotation"]
        translation = [float(x) for x in gt["translation"]]
        d = {
            "sample_token": sample_token,
            "translation": translation,
            "size": [float(x) for x in gt["size"]],
            "rotation": [float(x) for x in rot],
            "velocity": [0.0, 0.0],
            "num_pts": int(gt.get("num_lidar_pts", 0)),
            "detection_name": det_name,
            "detection_score": -1.0,
            "attribute_name": "",
        }
        if ego_translation is not None:
            d["ego_translation"] = ego_translation
        out.append(d)
    return out
