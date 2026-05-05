"""nuScenes per-sample dataloader for OpenYOLO3D.

Frame conventions:
  - point_cloud: (N, 4) — x, y, z, intensity in the EGO frame at the
    LIDAR_TOP sample timestamp.
  - cam_to_ego[cam]: (4, 4) camera-to-ego transform. Multiplying a
    homogeneous point in camera coordinates by this matrix yields its
    position in the ego frame. To project ego-frame points into the
    image, invert this matrix first (see dataloaders/sanity_check.py).
    The matrix comes directly from nuScenes `calibrated_sensor` records,
    which store the sensor's pose IN the ego frame — so without any
    inversion it acts as cam-to-ego.
  - ego_pose: (4, 4) ego-to-global at the LIDAR_TOP timestamp.

This loader is interface-only — it does not invoke OpenYOLO3D inference.
"""

import os.path as osp

import numpy as np
import yaml
from PIL import Image

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def _quat_trans_to_matrix(rotation, translation):
    return transform_matrix(translation=translation, rotation=Quaternion(rotation))


class NuScenesLoader:
    DEFAULT_CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(self, config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["nuscenes"]

        self.dataroot = cfg["dataroot"]
        self.version = cfg["version"]
        self.cameras = cfg.get("cameras") or self.DEFAULT_CAMERAS
        self.multi_sweep = bool(cfg.get("multi_sweep", False))
        self.num_sweeps = int(cfg.get("num_sweeps", 1))
        self.scene_filter = cfg.get("scene_filter")

        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

        if self.scene_filter:
            scene_tokens = {s["token"] for s in self.nusc.scene if s["name"] in self.scene_filter}
            self.sample_tokens = [s["token"] for s in self.nusc.sample if s["scene_token"] in scene_tokens]
        else:
            self.sample_tokens = [s["token"] for s in self.nusc.sample]

    def __len__(self):
        return len(self.sample_tokens)

    def __iter__(self):
        for tok in self.sample_tokens:
            yield self._load(tok)

    def __getitem__(self, idx):
        return self._load(self.sample_tokens[idx])

    def _load_lidar_ego(self, sample, lidar_sd, lidar_cs):
        if self.multi_sweep and self.num_sweeps > 1:
            pc, _ = LidarPointCloud.from_file_multisweep(
                self.nusc,
                sample,
                chan="LIDAR_TOP",
                ref_chan="LIDAR_TOP",
                nsweeps=self.num_sweeps,
            )
            points_lidar = pc.points.T
        else:
            pc = LidarPointCloud.from_file(osp.join(self.dataroot, lidar_sd["filename"]))
            points_lidar = pc.points.T

        T_lidar_to_ego = _quat_trans_to_matrix(lidar_cs["rotation"], lidar_cs["translation"])
        xyz_h = np.concatenate([points_lidar[:, :3], np.ones((points_lidar.shape[0], 1))], axis=1)
        xyz_ego = (T_lidar_to_ego @ xyz_h.T).T[:, :3]
        intensity = points_lidar[:, 3:4] if points_lidar.shape[1] >= 4 else np.zeros((points_lidar.shape[0], 1))
        return np.concatenate([xyz_ego, intensity], axis=1).astype(np.float32)

    def _load_cameras(self, sample):
        images, intrinsics, cam_to_ego = {}, {}, {}
        for cam in self.cameras:
            cam_token = sample["data"][cam]
            cam_sd = self.nusc.get("sample_data", cam_token)
            cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])

            with Image.open(osp.join(self.dataroot, cam_sd["filename"])) as img:
                images[cam] = np.array(img.convert("RGB"))
            intrinsics[cam] = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
            cam_to_ego[cam] = _quat_trans_to_matrix(cam_cs["rotation"], cam_cs["translation"])
        return images, intrinsics, cam_to_ego

    def _load_gt_boxes(self, sample):
        boxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            boxes.append({
                "category": ann["category_name"],
                "translation": np.array(ann["translation"], dtype=np.float64),
                "size": np.array(ann["size"], dtype=np.float64),
                "rotation": np.array(ann["rotation"], dtype=np.float64),
                "instance_token": ann["instance_token"],
                "num_lidar_pts": ann.get("num_lidar_pts", 0),
            })
        return boxes

    def _load(self, sample_token):
        sample = self.nusc.get("sample", sample_token)

        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_sd = self.nusc.get("sample_data", lidar_token)
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        ego_pose_rec = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        point_cloud = self._load_lidar_ego(sample, lidar_sd, lidar_cs)
        images, intrinsics, cam_to_ego = self._load_cameras(sample)
        T_ego_to_global = _quat_trans_to_matrix(ego_pose_rec["rotation"], ego_pose_rec["translation"])
        gt_boxes = self._load_gt_boxes(sample)

        return {
            "point_cloud": point_cloud,
            "images": images,
            "intrinsics": intrinsics,
            "cam_to_ego": cam_to_ego,
            "ego_pose": T_ego_to_global,
            "timestamp": int(sample["timestamp"]),
            "sample_token": sample_token,
            "gt_boxes": gt_boxes,
        }
