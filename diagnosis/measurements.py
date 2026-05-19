"""Measurement primitives for nuScenes diagnosis.

Pure geometric functions — no model invocation, no IO. Conventions match
dataloaders/nuscenes_loader.py: point clouds in EGO frame, cam_to_ego
matrices stored without inversion.
"""

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box


DISTANCE_BIN_EDGES = [0.0, 10.0, 20.0, 30.0, 50.0, float("inf")]
DISTANCE_BIN_LABELS = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]
LIFTABLE_K_VALUES = [1, 3, 5, 10, 20]
OVERSEG_THRESHOLDS = [50, 100, 200, 500]


def distance_bin(d):
    if d is None or not np.isfinite(d):
        return None
    for i in range(len(DISTANCE_BIN_LABELS)):
        if DISTANCE_BIN_EDGES[i] <= d < DISTANCE_BIN_EDGES[i + 1]:
            return DISTANCE_BIN_LABELS[i]
    return DISTANCE_BIN_LABELS[-1]


def project_points_to_camera(pc_ego_xyz, K, T_cam_to_ego, image_hw):
    """Project ego-frame points into the camera image.

    Returns dict with:
      - uv: (M, 2) float pixel coords of in-front + in-bounds points
      - depth: (M,) camera-frame z of those points
      - keep_idx: (M,) original indices into pc_ego_xyz
    M can be 0.
    """
    H, W = image_hw
    if pc_ego_xyz.shape[0] == 0:
        return {"uv": np.zeros((0, 2)), "depth": np.zeros((0,)), "keep_idx": np.zeros((0,), dtype=np.int64)}

    pts_h = np.concatenate([pc_ego_xyz, np.ones((pc_ego_xyz.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]

    in_front = pts_cam[:, 2] > 0.1
    in_front_idx = np.where(in_front)[0]
    if in_front_idx.size == 0:
        return {"uv": np.zeros((0, 2)), "depth": np.zeros((0,)), "keep_idx": np.zeros((0,), dtype=np.int64)}

    pts_cam_f = pts_cam[in_front_idx]
    uv_h = (K @ pts_cam_f.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    return {
        "uv": uv[in_bounds],
        "depth": pts_cam_f[in_bounds, 2],
        "keep_idx": in_front_idx[in_bounds],
    }


def points_inside_2d_box(uv, box_xyxy):
    """Boolean mask over uv selecting points inside the axis-aligned box."""
    if uv.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    x1, y1, x2, y2 = box_xyxy
    return (uv[:, 0] >= x1) & (uv[:, 0] < x2) & (uv[:, 1] >= y1) & (uv[:, 1] < y2)


def valid_projection_ratio(uv_inside_box, box_xyxy):
    """Fraction of integer pixels inside the box that have ≥1 point projected.

    Rasterizes uv into a per-box occupancy grid clipped to the box rectangle.
    """
    x1, y1, x2, y2 = box_xyxy
    W_box = max(1, int(round(x2 - x1)))
    H_box = max(1, int(round(y2 - y1)))
    if uv_inside_box.shape[0] == 0:
        return 0.0
    u = (uv_inside_box[:, 0] - x1).astype(np.int64)
    v = (uv_inside_box[:, 1] - y1).astype(np.int64)
    u = np.clip(u, 0, W_box - 1)
    v = np.clip(v, 0, H_box - 1)
    occ = np.zeros((H_box, W_box), dtype=bool)
    occ[v, u] = True
    return float(occ.sum()) / float(W_box * H_box)


def gt_box_to_ego(gt, ego_pose_4x4):
    """Build a nuScenes Box and transform it from global to ego frame.

    gt: dict with 'translation' (3,), 'size' (3, w/l/h), 'rotation' (4, wxyz).
    ego_pose_4x4: ego-to-global homogeneous transform.
    Returns nuscenes.utils.data_classes.Box in ego frame.
    """
    box = Box(
        center=np.array(gt["translation"], dtype=np.float64),
        size=np.array(gt["size"], dtype=np.float64),
        orientation=Quaternion(np.array(gt["rotation"], dtype=np.float64)),
    )
    ego_translation = ego_pose_4x4[:3, 3]
    ego_rotation_q = Quaternion(matrix=ego_pose_4x4[:3, :3])
    box.translate(-ego_translation)
    box.rotate(ego_rotation_q.inverse)
    return box


def points_inside_3d_box(box_ego, pc_ego_xyz):
    """Boolean mask: which points (ego frame) lie inside the 3D box (ego frame)."""
    if pc_ego_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    return points_in_box(box_ego, pc_ego_xyz.T)


def gt_box_visible(box_ego, K, T_cam_to_ego, image_hw):
    """GT box is 'visible' if its center projects inside the image with positive depth."""
    center_ego = box_ego.center.reshape(1, 3)
    proj = project_points_to_camera(center_ego, K, T_cam_to_ego, image_hw)
    return proj["uv"].shape[0] == 1


def project_3d_box_corners_to_image(box_ego, K, T_cam_to_ego, image_hw):
    """Return axis-aligned 2D xyxy box from projected 8 corners (clipped to image).

    Returns None if no corner has positive camera-frame depth.
    """
    H, W = image_hw
    corners = box_ego.corners().T  # (8, 3) in ego frame
    pts_h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return None
    pts_cam_f = pts_cam[in_front]
    uv_h = (K @ pts_cam_f.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    x1 = float(max(0.0, uv[:, 0].min()))
    y1 = float(max(0.0, uv[:, 1].min()))
    x2 = float(min(W - 1.0, uv[:, 0].max()))
    y2 = float(min(H - 1.0, uv[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def iou_2d_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


def is_oversegmented_at(num_instances, threshold):
    return bool(num_instances > threshold)
