"""Adapter: convert one nuScenes sample → OpenYOLO3D scene directory.

Output layout (matches what `utils.WORLD_2_CAM` reads):

    scene_dir/
    ├── color/0.jpg          # CAM_FRONT image, original resolution
    ├── depth/0.png          # uint16 sparse depth in mm; 0 = invalid pixel
    ├── poses/0.txt          # 4x4 cam-to-world matrix
    ├── intrinsics.txt       # 4x4 padded camera intrinsic
    └── lidar.ply            # LiDAR point cloud (with colors) in "world"

Coordinate convention
---------------------
We choose the EGO frame at the LIDAR_TOP timestamp as OpenYOLO3D's "world"
frame. With this choice:
  - poses/0.txt = cam_to_ego (directly from the dataloader, no inversion).
    OpenYOLO3D inverts this internally to obtain extrinsics (world→cam).
  - lidar.ply contains the point cloud already in ego frame (= world).
  - ego_pose (ego→global) is unused at this stage; using global as world
    would only complicate things and gain nothing for a single-frame
    smoke test.

Sparse depth
------------
We project every LiDAR point into the camera and write its z-value (in
camera frame) to the corresponding pixel. Multiple points landing on the
same pixel → keep the closest (z-buffer). No interpolation, no depth
completion. Empty pixels are 0, which OpenYOLO3D treats as invalid. uint16
mm gives a max representable depth of 65.535 m; points beyond are dropped
rather than clipped (would otherwise appear at exactly the max value and
mislead lifting).
"""

import os
import os.path as osp

import numpy as np
import open3d as o3d
from PIL import Image


CAMERA = "CAM_FRONT"
DEPTH_SCALE_MM = 1000  # png_value / DEPTH_SCALE_MM = depth in meters
DEPTH_MAX_M = 65.535   # 65535 / 1000


def project_lidar_to_depth(pc_ego, K, T_cam_to_ego, H, W):
    """Project a LiDAR cloud (ego frame) onto the camera image plane.

    Returns (depth_uint16_HxW_mm, n_filled_pixels).
    """
    pts_h = np.concatenate([pc_ego[:, :3], np.ones((pc_ego.shape[0], 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]

    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    if pts_cam.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint16), 0

    uv_h = (K @ pts_cam.T).T
    depth_m = uv_h[:, 2].copy()
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth_m > 0) & (depth_m <= DEPTH_MAX_M)
    u, v, depth_m = u[valid], v[valid], depth_m[valid]

    if u.size == 0:
        return np.zeros((H, W), dtype=np.uint16), 0

    depth_mm = np.round(depth_m * DEPTH_SCALE_MM).astype(np.uint16)
    flat_idx = v * W + u

    # z-buffer: for each pixel keep the smallest depth.
    order = np.argsort(depth_mm)
    flat_sorted = flat_idx[order]
    depth_sorted = depth_mm[order]
    _, first_pos = np.unique(flat_sorted, return_index=True)

    depth_map = np.zeros(H * W, dtype=np.uint16)
    depth_map[flat_sorted[first_pos]] = depth_sorted[first_pos]
    return depth_map.reshape(H, W), int(first_pos.size)


def _color_lidar_via_camera(pc_ego, image, K, T_cam_to_ego, default_gray=128):
    """Look up RGB color from the camera image for each visible LiDAR point.
    Points outside the camera FoV get (default_gray,)*3.
    """
    H, W = image.shape[:2]
    N = pc_ego.shape[0]
    colors = np.full((N, 3), default_gray, dtype=np.uint8)

    pts_h = np.concatenate([pc_ego[:, :3], np.ones((N, 1))], axis=1)
    pts_cam = (np.linalg.inv(T_cam_to_ego) @ pts_h.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.1
    if not in_front.any():
        return colors

    uv_h = (K @ pts_cam[in_front].T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    in_front_idx = np.where(in_front)[0]
    visible_idx = in_front_idx[in_bounds]
    colors[visible_idx] = image[v[in_bounds], u[in_bounds]]
    return colors


def _save_ply(points_xyz, colors_rgb_uint8, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb_uint8.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def _save_intrinsics_4x4(K_3x3, path):
    K_4x4 = np.zeros((4, 4))
    K_4x4[:3, :3] = K_3x3
    K_4x4[3, 3] = 1.0
    np.savetxt(path, K_4x4)


def adapt_sample(item, scene_dir, camera=CAMERA):
    """Materialize one NuScenesLoader item dict as an OpenYOLO3D scene dir.

    Returns a stats dict (image size, point count, depth pixel coverage).
    """
    os.makedirs(osp.join(scene_dir, "color"), exist_ok=True)
    os.makedirs(osp.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(osp.join(scene_dir, "poses"), exist_ok=True)

    image = item["images"][camera]
    H, W = image.shape[:2]
    K = item["intrinsics"][camera]
    T_cam_to_ego = item["cam_to_ego"][camera]
    pc_ego = item["point_cloud"]

    Image.fromarray(image).save(osp.join(scene_dir, "color", "0.jpg"), quality=95)

    depth_map, n_filled = project_lidar_to_depth(pc_ego, K, T_cam_to_ego, H, W)
    Image.fromarray(depth_map).save(osp.join(scene_dir, "depth", "0.png"))

    np.savetxt(osp.join(scene_dir, "poses", "0.txt"), T_cam_to_ego)
    _save_intrinsics_4x4(K, osp.join(scene_dir, "intrinsics.txt"))

    colors = _color_lidar_via_camera(pc_ego, image, K, T_cam_to_ego)
    _save_ply(pc_ego[:, :3], colors, osp.join(scene_dir, "lidar.ply"))

    return {
        "image_hw": (H, W),
        "n_lidar_points": int(pc_ego.shape[0]),
        "n_depth_pixels_filled": int(n_filled),
        "depth_pixel_coverage": float(n_filled) / (H * W),
        "depth_scale_mm": DEPTH_SCALE_MM,
    }
