"""γ measurement helpers.

For each sample we have CenterPoint boxes (in lidar frame), the original
ego-frame point cloud, and the GT box list. To evaluate using the W1
M/L/D/miss matching primitive, we need a (N_pts, N_proposals) bool mask
where each column is "points belonging to one proposal". We construct that
mask by transforming each CenterPoint box back to the ego frame as a
nuScenes ``Box`` and asking which ego-frame points fall inside it.

This is identical in spirit to W1's matching but applied to learned
detections instead of HDBSCAN clusters.
"""

from __future__ import annotations

import time
import signal
import tempfile
import os

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box, transform_matrix

from diagnosis_step1.matching import match_gt_to_instances
from diagnosis.measurements import distance_bin


PER_SAMPLE_TIMEOUT_S = 90

# nuScenes 10-class names (order matches CenterPoint training)
NUSC_10 = (
    "car", "truck", "trailer", "bus", "construction_vehicle",
    "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier",
)


# Map nuScenes GT category strings → 10-class group A membership.
# nuScenes GT category names look like "vehicle.car", "human.pedestrian.adult", etc.
def gt_category_to_group(category: str) -> str:
    """'A' if GT belongs to one of CenterPoint's 10 trained classes, else 'B'."""
    if category is None:
        return "B"
    c = category.lower()
    # vehicle.*
    if "vehicle.car" in c:
        return "A"
    if "vehicle.truck" in c:
        return "A"
    if "vehicle.trailer" in c:
        return "A"
    if "vehicle.bus" in c:
        return "A"
    if "vehicle.construction" in c:
        return "A"
    if "vehicle.bicycle" in c:
        return "A"
    if "vehicle.motorcycle" in c:
        return "A"
    # pedestrian
    if "human.pedestrian" in c:
        return "A"
    # static / movable
    if "movable_object.trafficcone" in c or "movable_object.traffic_cone" in c:
        return "A"
    if "movable_object.barrier" in c:
        return "A"
    return "B"


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


def _box_lidar_to_ego_box(b_lidar: list, T_lidar_to_ego: np.ndarray) -> Box:
    """CenterPoint box (lidar frame) → nuScenes Box (ego frame).

    bbox_lidar can be 7 or 9 dims:
      [x, y, z, w(dx), l(dy), h(dz), yaw, (vx, vy)]
    nuScenes Box wants ``size=(w, l, h)`` and yaw orientation.
    """
    x, y, z = b_lidar[0], b_lidar[1], b_lidar[2]
    w, l, h = b_lidar[3], b_lidar[4], b_lidar[5]
    yaw = b_lidar[6]

    # yaw → quaternion around z-axis
    rot = Quaternion(axis=[0, 0, 1], angle=float(yaw))
    box = Box(center=np.array([x, y, z], dtype=np.float64),
              size=np.array([w, l, h], dtype=np.float64),
              orientation=rot)
    # apply lidar→ego = rotate by R(T) then translate by t(T)
    box.rotate(Quaternion(matrix=T_lidar_to_ego[:3, :3]))
    box.translate(T_lidar_to_ego[:3, 3])
    return box


def _proposals_to_masks(proposals, T_lidar_to_ego, pc_ego_xyz):
    """Build (N_pts, N_prop) bool mask via points_in_box (ego frame)."""
    N_pts = pc_ego_xyz.shape[0]
    N_prop = len(proposals)
    if N_prop == 0 or N_pts == 0:
        return np.zeros((N_pts, N_prop), dtype=bool)
    masks = np.zeros((N_pts, N_prop), dtype=bool)
    pts_T = pc_ego_xyz.T  # (3, N) for nuScenes points_in_box
    for j, p in enumerate(proposals):
        try:
            box = _box_lidar_to_ego_box(p["bbox_lidar"], T_lidar_to_ego)
            inside = points_in_box(box, pts_T)
            masks[:, j] = inside
        except Exception:
            pass
    return masks


def measure_one(
    cp_generator,                  # CenterPointProposalGenerator
    cached_record: dict,
    work_root: str,
) -> dict:
    pc_ego = cached_record["pc_ego"]
    T_l2e = cached_record["T_lidar_to_ego"]
    pc_xyz = pc_ego[:, :3]
    n_pts = pc_xyz.shape[0]

    t0 = time.perf_counter()
    bin_path = os.path.join(work_root, f"{cached_record['sample_token']}.bin")
    out = cp_generator.generate(pc_ego, T_l2e, bin_path)
    proposals = out["proposals"]
    try:
        os.remove(bin_path)
    except FileNotFoundError:
        pass
    t1 = time.perf_counter()

    masks = _proposals_to_masks(proposals, T_l2e, pc_xyz)
    per_gt_all, cases_all = match_gt_to_instances(
        cached_record["gt_boxes"], cached_record["ego_pose"], pc_xyz, masks,
    )

    # Per-GT group classification (A = trained 10-class, B = unseen)
    group_of_gt = [gt_category_to_group(g.get("category"))
                   for g in cached_record["gt_boxes"]]
    cases_A = {"M": 0, "L": 0, "D": 0, "miss": 0}
    cases_B = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for grp, g in zip(group_of_gt, per_gt_all):
        case = g["case"]
        if grp == "A":
            cases_A[case] += 1
        else:
            cases_B[case] += 1
    n_gt = len(cached_record["gt_boxes"])
    n_A = sum(1 for g in group_of_gt if g == "A")
    n_B = sum(1 for g in group_of_gt if g == "B")

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "config": cp_generator.config_dict,
        "n_proposals": int(out["n_proposals"]),
        "n_proposals_pre_threshold": out["n_proposals_pre_threshold"],
        "score_threshold_applied": out["score_threshold_applied"],
        "timing_total_s": float(t1 - t0),
        "timing_breakdown": out["timing"],
        "n_gt_total": n_gt,
        "n_gt_A": int(n_A),
        "n_gt_B": int(n_B),
        "case_counts_all": cases_all,
        "case_counts_A": cases_A,
        "case_counts_B": cases_B,
        "M_rate_all": (cases_all["M"] / n_gt) if n_gt else 0.0,
        "L_rate_all": (cases_all["L"] / n_gt) if n_gt else 0.0,
        "D_rate_all": (cases_all["D"] / n_gt) if n_gt else 0.0,
        "miss_rate_all": (cases_all["miss"] / n_gt) if n_gt else 0.0,
        "M_rate_A": (cases_A["M"] / n_A) if n_A else 0.0,
        "miss_rate_A": (cases_A["miss"] / n_A) if n_A else 0.0,
        "M_rate_B": (cases_B["M"] / n_B) if n_B else 0.0,
        "miss_rate_B": (cases_B["miss"] / n_B) if n_B else 0.0,
        "per_gt": [
            {**g, "group": grp} for g, grp in zip(per_gt_all, group_of_gt)
        ],
        "proposal_classes": [p["cls_name"] for p in proposals],
        "proposal_scores": [p["score"] for p in proposals],
        "status": "ok",
    }


def measure_with_timeout(cp_generator, cached_record, work_root,
                          timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    _set_alarm(timeout_s)
    try:
        return measure_one(cp_generator, cached_record, work_root)
    finally:
        _clear_alarm()
