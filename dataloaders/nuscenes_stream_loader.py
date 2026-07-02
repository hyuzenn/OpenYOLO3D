"""Hybrid streaming nuScenes loader for OpenYOLO3D.

Heavy sensor data (6 camera JPGs + LIDAR_TOP .pcd.bin) is pulled over the
network from a remote WebDataset stream, while the lightweight calibration,
ego-pose, and GT annotation tables are read **locally** on the server via the
standard nuScenes API (matched by ``sample_token``). This lets the heavy
``samples/``/``sweeps/`` directories be deleted from this box as long as the
``<version>/*.json`` metadata tables are kept.

Output contract is identical to ``dataloaders.nuscenes_loader.NuScenesLoader``
so this can be swapped in wherever that loader is consumed. Frame conventions
(copied verbatim from that loader so downstream projection code is unchanged):

  - point_cloud: (N, 4) — x, y, z, intensity in the EGO frame at the
    LIDAR_TOP sample timestamp.
  - cam_to_ego[cam]: (4, 4) camera-to-ego transform (from nuScenes
    ``calibrated_sensor``; invert to project ego-frame points into the image).
  - ego_pose: (4, 4) ego-to-global at the LIDAR_TOP timestamp.

Differences from the disk loader, by design:
  - Iteration order follows the *stream* (shard order), not nuScenes sample
    order. ``__getitem__`` random access is unsupported (streaming is
    sequential-only) and raises ``NotImplementedError``.
  - ``multi_sweep`` is unsupported: shards carry only the keyframe LIDAR_TOP,
    not the intermediate sweeps, so sweep aggregation cannot be reconstructed
    from the stream. Set ``multi_sweep: false``.

This loader is interface-only — it does not invoke OpenYOLO3D inference.
"""

import io
import json
import os.path as osp

import numpy as np
import yaml
from PIL import Image

import webdataset as wds

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


def _quat_trans_to_matrix(rotation, translation):
    return transform_matrix(translation=translation, rotation=Quaternion(rotation))


# Shard files are named "{token}.cam_front.jpg" etc. (lowercase). WebDataset
# splits on the first dot, so the per-sample dict is keyed by these extensions.
_CAM_EXT = {
    "CAM_FRONT": "cam_front.jpg",
    "CAM_FRONT_LEFT": "cam_front_left.jpg",
    "CAM_FRONT_RIGHT": "cam_front_right.jpg",
    "CAM_BACK": "cam_back.jpg",
    "CAM_BACK_LEFT": "cam_back_left.jpg",
    "CAM_BACK_RIGHT": "cam_back_right.jpg",
}
_LIDAR_EXT = "lidar_top.pcd.bin"
_META_EXT = "meta.json"
_LIDAR_NUM_COLS = 5  # nuScenes standard: x, y, z, intensity, ring


def _curl_wrapped(url: str) -> str:
    """Same curl options proven out in test_wds_loader.py (long timeouts for
    1 GB shards over the reverse tunnel; -f so HTTP errors propagate)."""
    return f"pipe:curl -s -f -L --retry 3 --connect-timeout 60 --max-time 0 '{url}'"


class StreamingNuScenesLoader:
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
            cfg = yaml.safe_load(f)

        nusc_cfg = cfg["nuscenes"]
        stream_cfg = cfg["stream"]

        # --- local metadata side ---
        self.dataroot = nusc_cfg["dataroot"]
        self.version = nusc_cfg["version"]
        self.cameras = nusc_cfg.get("cameras") or self.DEFAULT_CAMERAS
        self.scene_filter = nusc_cfg.get("scene_filter")
        if bool(nusc_cfg.get("multi_sweep", False)):
            raise ValueError(
                "StreamingNuScenesLoader does not support multi_sweep: the "
                "stream carries only keyframe LIDAR_TOP, not intermediate "
                "sweeps. Set multi_sweep: false."
            )

        # Fail loudly *before* the heavy NuScenes() load if the metadata tables
        # are not on this box (the whole point of the 'local metadata' design).
        meta_dir = osp.join(self.dataroot, self.version)
        if not osp.isdir(meta_dir) or not osp.isfile(osp.join(meta_dir, "sample.json")):
            raise FileNotFoundError(
                f"nuScenes metadata tables not found at '{meta_dir}'. This loader "
                f"streams pixels/lidar but reads calibration/pose/GT locally, so the "
                f"'{self.version}/*.json' tables (sample.json, sample_data.json, "
                f"calibrated_sensor.json, ego_pose.json, sample_annotation.json, ...) "
                f"must be present under dataroot='{self.dataroot}'. Copy the metadata "
                f"tables to this server (the heavy samples/ and sweeps/ blobs are NOT "
                f"needed — those come from the stream)."
            )

        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)

        if self.scene_filter:
            scene_tokens = {s["token"] for s in self.nusc.scene if s["name"] in self.scene_filter}
            self.sample_tokens = [s["token"] for s in self.nusc.sample if s["scene_token"] in scene_tokens]
        else:
            self.sample_tokens = [s["token"] for s in self.nusc.sample]

        # --- remote stream side ---
        url_base = stream_cfg["url_base"].rstrip("/")
        num_shards = int(stream_cfg["num_shards"])
        shard_pattern = stream_cfg.get("shard_pattern", "nuscenes-{:06d}.tar")
        self.urls = [
            _curl_wrapped(f"{url_base}/{shard_pattern.format(i)}")
            for i in range(num_shards)
        ]

    def __len__(self):
        # Matches NuScenesLoader: count of (optionally scene-filtered) samples
        # in the local metadata. Assumes the stream covers these samples.
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "StreamingNuScenesLoader is sequential-only (WebDataset stream). "
            "Iterate with `for item in loader:` instead of random indexing."
        )

    # --------------------------------------------------------------------- #
    # heavy-data decode (from streamed bytes)
    # --------------------------------------------------------------------- #
    def _decode_image(self, raw: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(raw)) as img:
            return np.array(img.convert("RGB"))

    def _decode_lidar_sensor_frame(self, raw: bytes) -> np.ndarray:
        pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, _LIDAR_NUM_COLS)
        return pts  # x, y, z, intensity, ring — still in LIDAR sensor frame

    # --------------------------------------------------------------------- #
    # local-metadata enrichment (by sample_token)
    # --------------------------------------------------------------------- #
    def _lidar_to_ego(self, points_sensor: np.ndarray, lidar_cs) -> np.ndarray:
        """Replicates NuScenesLoader single-sweep path: transform sensor-frame
        xyz into the ego frame, keep intensity -> (N, 4) float32."""
        # dtype handling mirrors NuScenesLoader._load_lidar_ego exactly (default
        # float64 ones column -> float64 homogeneous coords) so the streamed
        # result is bit-identical to the disk loader, not just np.allclose.
        T_lidar_to_ego = _quat_trans_to_matrix(lidar_cs["rotation"], lidar_cs["translation"])
        xyz = points_sensor[:, :3]
        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
        xyz_ego = (T_lidar_to_ego @ xyz_h.T).T[:, :3]
        intensity = points_sensor[:, 3:4]
        return np.concatenate([xyz_ego, intensity], axis=1).astype(np.float32)

    def _camera_calib(self, sample):
        intrinsics, cam_to_ego = {}, {}
        for cam in self.cameras:
            cam_sd = self.nusc.get("sample_data", sample["data"][cam])
            cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
            intrinsics[cam] = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
            cam_to_ego[cam] = _quat_trans_to_matrix(cam_cs["rotation"], cam_cs["translation"])
        return intrinsics, cam_to_ego

    def _gt_boxes(self, sample):
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

    def _build_item(self, streamed: dict) -> dict:
        """Merge streamed heavy data with locally-queried metadata into the
        exact dict contract NuScenesLoader produces."""
        meta = json.loads(streamed[_META_EXT])
        sample_token = meta["sample_token"]
        sample = self.nusc.get("sample", sample_token)

        # cameras (pixels from stream, calib from local API)
        images = {cam: self._decode_image(streamed[_CAM_EXT[cam]]) for cam in self.cameras}
        intrinsics, cam_to_ego = self._camera_calib(sample)

        # lidar (points from stream, sensor->ego transform from local API)
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        point_cloud = self._lidar_to_ego(
            self._decode_lidar_sensor_frame(streamed[_LIDAR_EXT]), lidar_cs
        )

        # pose + GT (local API)
        ego_pose_rec = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        T_ego_to_global = _quat_trans_to_matrix(ego_pose_rec["rotation"], ego_pose_rec["translation"])

        return {
            "point_cloud": point_cloud,
            "images": images,
            "intrinsics": intrinsics,
            "cam_to_ego": cam_to_ego,
            "ego_pose": T_ego_to_global,
            "timestamp": int(sample["timestamp"]),
            "sample_token": sample_token,
            "gt_boxes": self._gt_boxes(sample),
        }

    # --------------------------------------------------------------------- #
    # iteration
    # --------------------------------------------------------------------- #
    def __iter__(self):
        # SimpleShardList = deterministic single pass (eval-friendly). Each
        # tar-grouped sample is a dict keyed by file extension after the token.
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.urls),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
        )
        for streamed in pipeline:
            yield self._build_item(streamed)
