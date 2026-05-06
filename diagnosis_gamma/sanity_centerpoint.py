"""γ Stage B.2 — CenterPoint Phase 0.5 sanity.

Load 1 W1.5 sample, run mmdet3d CenterPoint, verify output shape +
inference time. Writes results to
``results/diagnosis_gamma/sanity_centerpoint_check.md``.

Coordinate frame: nuScenes loader emits points in EGO frame; CenterPoint is
trained in LIDAR_TOP frame. We transform ego→lidar via inv(T_lidar_to_ego)
which the loader stores indirectly. For the smoke test we recompute it from
the nuScenes calibrated_sensor record.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import tempfile
import time

import numpy as np
import torch
import yaml
from pyquaternion import Quaternion

from dataloaders.nuscenes_loader import NuScenesLoader


CKPT = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"
CONFIG = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"


def _build_loader(version):
    cfg = yaml.safe_load(open("configs/nuscenes_baseline.yaml"))
    cfg["nuscenes"]["version"] = version
    cfg["nuscenes"]["cameras"] = ["CAM_FRONT"]
    fd, p = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    yaml.safe_dump(cfg, open(p, "w"))
    L = NuScenesLoader(p)
    os.remove(p)
    return L


def _ego_to_lidar(pc_ego, lidar_cs):
    """Reconstruct lidar-frame xyz from ego-frame xyz via inv(T_lidar_to_ego)."""
    from nuscenes.utils.geometry_utils import transform_matrix
    T_lidar_to_ego = transform_matrix(translation=lidar_cs["translation"],
                                        rotation=Quaternion(lidar_cs["rotation"]))
    T_inv = np.linalg.inv(T_lidar_to_ego)
    pts_h = np.concatenate([pc_ego[:, :3], np.ones((pc_ego.shape[0], 1))], axis=1)
    pts_lidar = (T_inv @ pts_h.T).T[:, :3]
    return pts_lidar


def _pad_to_5_feature(pc_xyz, intensity):
    """nuScenes CenterPoint is trained on (x,y,z,intensity,time) where time=0
    on keyframe. Build the 5-channel array in float32."""
    out = np.zeros((pc_xyz.shape[0], 5), dtype=np.float32)
    out[:, :3] = pc_xyz
    out[:, 3] = intensity.flatten() if intensity.ndim > 1 else intensity
    out[:, 4] = 0.0  # time delta — single sweep keyframe
    return out


def main():
    out_md = "/home/rintern16/OpenYOLO3D-nuscenes/results/diagnosis_gamma/sanity_centerpoint_check.md"
    out_dir = "/home/rintern16/OpenYOLO3D-nuscenes/results/diagnosis_gamma"
    os.makedirs(out_dir, exist_ok=True)

    notes = []
    notes.append("# γ Stage B.2 — CenterPoint sanity inference\n")
    notes.append(f"- ckpt: `{CKPT}`")
    notes.append(f"- config: `{CONFIG}`")
    notes.append("")

    # Load 1 mini sample
    print("[B.2] Loading 1 mini sample ...")
    prov = json.load(open("/home/rintern16/OpenYOLO3D-nuscenes/results/diagnosis_step_a/samples_used.json"))
    target_token = prov["tokens_by_source"]["mini"][0]
    notes.append(f"- sample_token: `{target_token}`")

    loader = _build_loader("v1.0-mini")
    tok_to_idx = {t: i for i, t in enumerate(loader.sample_tokens)}
    item = loader[tok_to_idx[target_token]]

    # Transform ego → lidar
    sample = loader.nusc.get("sample", target_token)
    lidar_sd = loader.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = loader.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc_ego = item["point_cloud"]
    pc_lidar_xyz = _ego_to_lidar(pc_ego, lidar_cs)
    pc_5feat = _pad_to_5_feature(pc_lidar_xyz, pc_ego[:, 3])
    notes.append(f"- point cloud: ego shape {pc_ego.shape}, lidar shape {pc_lidar_xyz.shape}, "
                  f"5-feat shape {pc_5feat.shape}")

    # Save to .bin (mmdet3d inference expects file path or array)
    bin_path = osp.join(out_dir, "_sanity_input.bin")
    pc_5feat.astype(np.float32).tofile(bin_path)

    # Init model
    print("[B.2] Loading CenterPoint model ...")
    t0 = time.time()
    from mmdet3d.apis import init_model, inference_detector
    model = init_model(CONFIG, CKPT, device="cuda:0")
    notes.append(f"- model init: {time.time() - t0:.2f}s")
    print(f"  init {time.time() - t0:.1f}s")

    # Inference
    print("[B.2] Running inference ...")
    t1 = time.time()
    result, data = inference_detector(model, bin_path)
    t_infer = time.time() - t1
    notes.append(f"- inference time: {t_infer:.3f}s")
    print(f"  inference {t_infer:.3f}s")

    # Parse output. mmdet3d 1.x returns Det3DDataSample(s); attribute name = pred_instances_3d
    pred = getattr(result, "pred_instances_3d", None)
    if pred is None and isinstance(result, list):
        pred = getattr(result[0], "pred_instances_3d", None)

    if pred is None:
        notes.append("- **INFERENCE_FAIL — could not locate pred_instances_3d on result**")
        with open(out_md, "w") as f:
            f.write("\n".join(notes))
        sys.exit(2)

    bboxes = pred.bboxes_3d.tensor.cpu().numpy()  # (N, 7) usually [x,y,z,w,l,h,yaw] or 9 with vx,vy
    scores = pred.scores_3d.cpu().numpy()
    labels = pred.labels_3d.cpu().numpy()
    notes.append(f"- output bboxes shape: {bboxes.shape}")
    notes.append(f"- output scores shape: {scores.shape}, "
                  f"min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}")
    notes.append(f"- # bboxes (raw, no threshold): {len(bboxes)}")

    # Threshold-applied counts
    for thr in [0.05, 0.10, 0.20, 0.30]:
        n = int((scores >= thr).sum())
        notes.append(f"- # bboxes @ score≥{thr}: {n}")

    # Class distribution at threshold 0.1
    keep = scores >= 0.10
    if keep.any():
        unique, counts = np.unique(labels[keep], return_counts=True)
        # mmdet3d nuScenes class names (10 classes, in order):
        class_names = (
            "car", "truck", "trailer", "bus", "construction_vehicle",
            "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
        )
        dist_str = ", ".join(
            f"{class_names[u] if 0 <= u < len(class_names) else f'cls_{u}'}={c}"
            for u, c in zip(unique.tolist(), counts.tolist())
        )
        notes.append(f"- class dist @ score≥0.10: {dist_str}")

    notes.append("")

    # Verdict
    n_at_01 = int((scores >= 0.10).sum())
    if 5 <= n_at_01 <= 30 and t_infer < 5.0:
        verdict = "**B.2 SUCCESS**"
        proceed = True
    else:
        verdict = (f"**B.2 ATTENTION** — n@0.1={n_at_01} (target 5-30), "
                    f"timing={t_infer:.2f}s (target <5s). Out of band but not blocking.")
        proceed = (n_at_01 > 0)

    notes.append(f"## Verdict: {verdict}")
    notes.append("")
    if proceed:
        notes.append("Stage C will use CenterPoint as proposal source.")
    else:
        notes.append("STOP — output looks wrong. Investigate before Stage C.")

    with open(out_md, "w") as f:
        f.write("\n".join(notes))

    # Print summary to stdout
    print("\n".join(notes[-12:]))
    sys.exit(0 if proceed else 2)


if __name__ == "__main__":
    main()
