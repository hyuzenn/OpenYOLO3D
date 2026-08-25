"""Hybrid-proposal geometry: project a CenterPoint 3D box onto the nuScenes
camera images and emit a 2D ROI, so an open-vocabulary 2D detector (YOLO-World)
can re-label it.

This is the geometry half of the "CenterPoint = pure geometry proposal generator"
pipeline:

    CenterPoint 3D box  (class DISCARDED)
      -> 8 corners in LIDAR frame (gravity-centre convention)
      -> lidar -> ego -> camera -> image  for each of the 6 cameras
      -> pick the camera that best sees the box
      -> clipped 2D ROI (xyxy)        <-- consumed by YOLO-World

Frame conventions (match dataloaders/nuscenes_loader.py):
  - bbox_lidar = [x, y, z, dx, dy, dz, yaw, (vx, vy)] in the LIDAR_TOP frame.
    z is the GRAVITY centre (the gamma cache `..._single_gravity` already applied
    z += dz/2), so corners are symmetric in z about the centre.
  - T_lidar_to_ego : (4,4) lidar -> ego.
  - cam_to_ego[cam]: (4,4) camera -> ego  (nuScenes calibrated_sensor pose);
    invert for ego -> cam.
  - K[cam]         : (3,3) camera intrinsic. nuScenes camera frame is
    x-right, y-down, z-forward, so a 3D point is visible iff z > 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def box_corners_lidar(bbox_lidar) -> np.ndarray:
    """8 corners of a gravity-centred LiDAR box. Returns (8, 3) in lidar frame."""
    x, y, z, dx, dy, dz, yaw = [float(v) for v in bbox_lidar[:7]]
    # box-local corner signs (x=length, y=width, z=height), centre at origin
    sx = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64) * (dx / 2.0)
    sy = np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64) * (dy / 2.0)
    sz = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float64) * (dz / 2.0)
    corners = np.stack([sx, sy, sz], axis=1)            # (8,3) box frame
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = corners @ R.T
    corners += np.array([x, y, z])
    return corners


def _homog(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)


@dataclass
class ROIResult:
    cam: Optional[str]            # winning camera, or None if box not seen
    roi_xyxy: Optional[tuple]     # clipped ROI in image pixels, or None
    n_corners_in_front: int       # corners with cam-depth > eps (of 8)
    center_depth: float           # box-centre depth in winning cam (m); nan if none
    roi_area_frac: float          # clipped ROI area / image area
    success: bool                 # passes the visibility / min-size test
    per_cam: dict                 # cam -> diagnostic dict (debug / multi-view)


def project_box(
    bbox_lidar,
    T_lidar_to_ego: np.ndarray,
    cam_to_ego: dict,
    intrinsics: dict,
    image_hw: dict,
    *,
    depth_eps: float = 0.1,
    min_side_px: float = 8.0,
    min_corners_in_front: int = 1,
) -> ROIResult:
    """Project one box into every camera, return the best-visible ROI.

    A camera "sees" the box when: the box centre is in front (depth > eps),
    at least ``min_corners_in_front`` corners are in front, and the clipped
    in-image ROI has both sides >= ``min_side_px``. Among cameras that pass,
    the one with the largest clipped ROI area wins.
    """
    corners = box_corners_lidar(bbox_lidar)                       # (8,3) lidar
    corners_ego = (T_lidar_to_ego @ _homog(corners).T).T[:, :3]
    center_lidar = np.asarray(bbox_lidar[:3], dtype=np.float64).reshape(1, 3)
    center_ego = (T_lidar_to_ego @ _homog(center_lidar).T).T[:, :3]

    per_cam = {}
    best = None
    for cam, T_c2e in cam_to_ego.items():
        K = np.asarray(intrinsics[cam], dtype=np.float64)
        H, W = image_hw[cam]
        T_e2c = np.linalg.inv(T_c2e)

        cc = (T_e2c @ _homog(corners_ego).T).T[:, :3]            # corners in cam
        depths = cc[:, 2]
        in_front = depths > depth_eps
        n_front = int(in_front.sum())

        center_cam = (T_e2c @ _homog(center_ego).T).T[0, :3]
        cdepth = float(center_cam[2])

        cam_rec = {
            "n_corners_in_front": n_front,
            "center_depth": cdepth,
            "roi_xyxy": None,
            "roi_area_frac": 0.0,
            "in_image": False,
        }

        if n_front >= min_corners_in_front and cdepth > depth_eps:
            uv = (K @ cc[in_front].T).T
            uv = uv[:, :2] / uv[:, 2:3]
            x0, y0 = float(uv[:, 0].min()), float(uv[:, 1].min())
            x1, y1 = float(uv[:, 0].max()), float(uv[:, 1].max())
            # clip to image
            cx0, cy0 = max(0.0, x0), max(0.0, y0)
            cx1, cy1 = min(float(W), x1), min(float(H), y1)
            w, h = cx1 - cx0, cy1 - cy0
            if w >= min_side_px and h >= min_side_px:
                area_frac = (w * h) / (W * H)
                cam_rec.update({
                    "roi_xyxy": (cx0, cy0, cx1, cy1),
                    "roi_area_frac": area_frac,
                    "in_image": True,
                })
                if best is None or area_frac > best[1]:
                    best = (cam, area_frac, (cx0, cy0, cx1, cy1), n_front, cdepth)
        per_cam[cam] = cam_rec

    if best is None:
        return ROIResult(None, None, 0, float("nan"), 0.0, False, per_cam)
    cam, area_frac, roi, n_front, cdepth = best
    return ROIResult(cam, roi, n_front, cdepth, area_frac, True, per_cam)
