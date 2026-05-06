"""Tier-2 measurement primitives — multi-view + uniformity.

No model invocation, no IO. Reuses Tier-1 geometry helpers where possible.
Conventions:
  - point_cloud xyz in EGO frame
  - cam_to_ego stored as 4x4; ego→cam = inv(cam_to_ego)
"""

import math

import numpy as np

from diagnosis.measurements import (
    DISTANCE_BIN_LABELS,
    distance_bin,
    project_points_to_camera,
    points_inside_2d_box,
    iou_2d_xyxy,
    project_3d_box_corners_to_image,
    gt_box_visible,
)


SIX_CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


# -------------------- M1 view diversity --------------------

def per_cam_geom_visibility(box_ego, pc_ego_xyz, K, T_cam_to_ego, image_hw):
    """Return (geom_visible, projected_2d_bbox_or_None, n_inbox_pts).

    geom_visible = (3D box center projects inside image with positive depth)
                   AND (≥1 LiDAR point — projected from full sweep — falls
                        inside the GT's projected 2D bbox in this cam).
    """
    if not gt_box_visible(box_ego, K, T_cam_to_ego, image_hw):
        return False, None, 0
    bbox_2d = project_3d_box_corners_to_image(box_ego, K, T_cam_to_ego, image_hw)
    if bbox_2d is None:
        return False, None, 0
    proj = project_points_to_camera(pc_ego_xyz, K, T_cam_to_ego, image_hw)
    if proj["uv"].shape[0] == 0:
        return False, bbox_2d, 0
    inside = points_inside_2d_box(proj["uv"], bbox_2d)
    n = int(inside.sum())
    return n >= 1, bbox_2d, n


def per_cam_det_visibility(projected_2d_bbox, det_xyxys, iou_threshold=0.3):
    """det_visible = max IoU between projected GT bbox and any cam detection ≥ threshold."""
    if projected_2d_bbox is None or not det_xyxys:
        return False, None, 0.0
    ious = [iou_2d_xyxy(projected_2d_bbox, b) for b in det_xyxys]
    j = int(np.argmax(ious))
    return (ious[j] >= iou_threshold), j, float(ious[j])


# -------------------- M2 multi-view consistency --------------------

def per_cam_inbox_centroid(box_ego, pc_ego_xyz, K, T_cam_to_ego, image_hw):
    """For one cam, find LiDAR points whose projection lands inside the GT's
    projected 2D bbox, then return the *median* ego-frame centroid of those
    points (robust to outliers).

    Returns dict {centroid: (3,) | None, n_points: int, projected_2d_bbox: list | None}.
    """
    bbox_2d = project_3d_box_corners_to_image(box_ego, K, T_cam_to_ego, image_hw)
    if bbox_2d is None:
        return {"centroid": None, "n_points": 0, "projected_2d_bbox": None}
    proj = project_points_to_camera(pc_ego_xyz, K, T_cam_to_ego, image_hw)
    if proj["uv"].shape[0] == 0:
        return {"centroid": None, "n_points": 0, "projected_2d_bbox": bbox_2d}
    inside = points_inside_2d_box(proj["uv"], bbox_2d)
    n = int(inside.sum())
    if n == 0:
        return {"centroid": None, "n_points": 0, "projected_2d_bbox": bbox_2d}
    keep = proj["keep_idx"][inside]
    pts_ego = pc_ego_xyz[keep]
    centroid = np.median(pts_ego, axis=0)
    return {"centroid": centroid.tolist(), "n_points": n, "projected_2d_bbox": bbox_2d}


def pair_distances(centroids):
    """centroids: list of 3-vectors. Returns list of pairwise Euclidean distances."""
    out = []
    n = len(centroids)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(np.asarray(centroids[i]) - np.asarray(centroids[j])))
            out.append(d)
    return out


# -------------------- M3 per-box uniformity --------------------

def quadrant_entropy(uv_inside_box, box_xyxy):
    """4-quadrant Shannon entropy of in-box LiDAR projection counts, normalised to [0, 1].

    Returns (entropy_norm, quadrant_counts). entropy_norm = 1 → uniform across
    4 quadrants; 0 → all in one quadrant. Caller should gate on n_total ≥ 4.
    """
    if uv_inside_box.shape[0] == 0:
        return 0.0, [0, 0, 0, 0]
    x1, y1, x2, y2 = box_xyxy
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    right = uv_inside_box[:, 0] >= mid_x
    bottom = uv_inside_box[:, 1] >= mid_y
    q_idx = right.astype(np.int64) + 2 * bottom.astype(np.int64)
    counts = np.bincount(q_idx, minlength=4).astype(np.int64).tolist()
    total = sum(counts)
    if total == 0:
        return 0.0, counts
    H = 0.0
    for c in counts:
        if c == 0:
            continue
        p = c / total
        H -= p * math.log(p)
    H_norm = float(H / math.log(4))
    return H_norm, counts
