"""Option 5 measurement helpers — wraps DetectionGuidedClusterer + matching.

Adds two diagnostic axes on top of W1's M/L/D/miss:
  - per-GT 2D detection coverage (was the GT detected in any cam?). H13.
  - M_within_detected vs M_outside_detected (does the method recover M only
    when 2D found the GT, or does it also work on geometry-only-visible GTs?)
"""

from __future__ import annotations

import time
import signal

import numpy as np

from preprocessing.detection_frustum import FrustumExtractor
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from proposal.detection_guided_clustering import DetectionGuidedClusterer
from diagnosis_step1.matching import match_gt_to_instances
from diagnosis.measurements import (
    distance_bin, gt_box_to_ego, points_inside_3d_box,
    project_3d_box_corners_to_image,
)
from diagnosis_tier2.measurements_tier2 import iou_2d_xyxy


PER_SAMPLE_TIMEOUT_S = 90


# Locked configs — frozen by spec.
BETA1_BEST_PILLAR = {
    "pillar_size_xy": (0.5, 0.5),
    "z_threshold": 0.3,
    "ground_estimation": "percentile",
    "percentile_p": 10.0,
}
HDBSCAN_BEST = {
    "min_cluster_size": 3,
    "min_samples": 3,
    "cluster_selection_epsilon": 1.0,
    "ground_filter": "z_threshold",
    "ground_z_max": -1.4,
}
SIX_CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
DET_RECALL_IOU = 0.3


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


def compute_2d_detection_recall_per_gt(
    gt_boxes, ego_pose_4x4, intrinsics_per_cam, cam_to_ego_per_cam,
    image_hw_per_cam, detections_per_cam, iou_threshold=DET_RECALL_IOU,
):
    """Per-GT: was it detected in any cam (best IoU ≥ threshold against
    projected GT 2D bbox)? Returns list parallel to gt_boxes.

    Each entry: dict with keys 'detected_in_any_cam', 'best_cam', 'best_iou',
    'distance_m', 'distance_bin'.
    """
    out = []
    for gt in gt_boxes:
        try:
            box_ego = gt_box_to_ego(gt, ego_pose_4x4)
        except Exception:
            out.append({"detected_in_any_cam": False, "best_cam": None,
                         "best_iou": 0.0, "distance_m": None, "distance_bin": None})
            continue
        d = float(np.linalg.norm(box_ego.center))
        d_bin = distance_bin(d)
        detected = False
        best_cam = None
        best_iou = 0.0
        for cam in SIX_CAMERAS:
            if cam not in intrinsics_per_cam:
                continue
            bbox_2d = project_3d_box_corners_to_image(
                box_ego, intrinsics_per_cam[cam], cam_to_ego_per_cam[cam],
                image_hw_per_cam[cam],
            )
            if bbox_2d is None:
                continue
            xyxys = detections_per_cam.get(cam, {}).get("xyxy", [])
            if not xyxys:
                continue
            for det_box in xyxys:
                iou = iou_2d_xyxy(bbox_2d, det_box)
                if iou > best_iou:
                    best_iou = iou
                    if iou >= iou_threshold:
                        detected = True
                        best_cam = cam
        out.append({
            "detected_in_any_cam": detected,
            "best_cam": best_cam,
            "best_iou": float(best_iou),
            "distance_m": d,
            "distance_bin": d_bin,
        })
    return out


def measure_one(
    clusterer: DetectionGuidedClusterer,
    cached_record: dict,
    cached_detections: dict,         # {cam: {"xyxy", "labels", "scores"}}
) -> dict:
    pc = cached_record["pc_ego"]
    pc_xyz = pc[:, :3]
    intrinsics = cached_record["intrinsics_per_cam"]
    cam_to_ego = cached_record["cam_to_ego_per_cam"]
    image_hw = cached_record["image_hw_per_cam"]

    t0 = time.perf_counter()
    out = clusterer.generate(pc, cached_detections, intrinsics, cam_to_ego, image_hw)
    t_pipeline = time.perf_counter() - t0

    masks = out["proposal_masks"]   # (N_pts, N_proposals)
    per_gt, cases = match_gt_to_instances(
        cached_record["gt_boxes"], cached_record["ego_pose"], pc_xyz, masks,
    )
    n_gt = len(cached_record["gt_boxes"])

    # 2D detection recall per GT
    det_per_gt = compute_2d_detection_recall_per_gt(
        cached_record["gt_boxes"], cached_record["ego_pose"],
        intrinsics, cam_to_ego, image_hw, cached_detections,
    )

    # Split M/miss by 2D detection coverage
    n_det = sum(1 for d in det_per_gt if d["detected_in_any_cam"])
    n_undet = n_gt - n_det
    M_in_det = M_in_undet = 0
    miss_in_det = miss_in_undet = 0
    for g, d in zip(per_gt, det_per_gt):
        if d["detected_in_any_cam"]:
            if g["case"] == "M":
                M_in_det += 1
            elif g["case"] == "miss":
                miss_in_det += 1
        else:
            if g["case"] == "M":
                M_in_undet += 1
            elif g["case"] == "miss":
                miss_in_undet += 1

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "config": clusterer.config_dict,
        "n_proposals_total": int(out["n_proposals_total"]),
        "n_frustums": out["n_frustums"],
        "n_frustums_with_lidar": out["n_frustums_with_lidar"],
        "n_frustums_with_clusters": out["n_frustums_with_clusters"],
        "n_frustums_skipped_low_points": out["n_frustums_skipped_low_points"],
        "n_frustums_skipped_no_cluster": out["n_frustums_skipped_no_cluster"],
        "n_gt_total": n_gt,
        "n_gt_2d_detected": n_det,
        "case_counts": cases,
        "M_rate": (cases["M"] / n_gt) if n_gt else 0.0,
        "L_rate": (cases["L"] / n_gt) if n_gt else 0.0,
        "D_rate": (cases["D"] / n_gt) if n_gt else 0.0,
        "miss_rate": (cases["miss"] / n_gt) if n_gt else 0.0,
        "two_d_recall": (n_det / n_gt) if n_gt else 0.0,
        "M_in_detected_count": int(M_in_det),
        "M_in_undetected_count": int(M_in_undet),
        "miss_in_detected_count": int(miss_in_det),
        "miss_in_undetected_count": int(miss_in_undet),
        "M_rate_within_detected": (M_in_det / n_det) if n_det else 0.0,
        "M_rate_within_undetected": (M_in_undet / n_undet) if n_undet else 0.0,
        "n_detections_total_6cam": sum(len(cached_detections.get(c, {}).get("xyxy", []))
                                          for c in SIX_CAMERAS),
        "timing_total_s": float(t_pipeline),
        "timing_breakdown": out["timing"],
        "per_gt": [
            {**g, **{"detected_in_any_cam": d["detected_in_any_cam"],
                       "best_cam": d["best_cam"], "best_iou": d["best_iou"],
                       "distance_m": d["distance_m"], "distance_bin": d["distance_bin"]}}
            for g, d in zip(per_gt, det_per_gt)
        ],
        "per_frustum_records": out["per_frustum_records"],
        "status": "ok",
    }


def measure_with_timeout(clusterer, cached_record, cached_detections,
                          timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    _set_alarm(timeout_s)
    try:
        return measure_one(clusterer, cached_record, cached_detections)
    finally:
        _clear_alarm()
