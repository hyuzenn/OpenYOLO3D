"""α measurement helpers.

Per sample we run β1 (pillar foreground + HDBSCAN) and γ (CenterPoint) once
each, then translate both proposal sets into a unified ``(N_pts, n_proposals)``
mask basis so the four union strategies can act on the same artifacts:

    artifacts = {
        "gt_boxes", "ego_pose", "pc_xyz",
        "beta1_masks":   (N, n_b)  bool,
        "beta1_sizes":   (n_b,)    int (cluster point count),
        "beta1_aabbs":   (n_b, 4)  float (xmin, ymin, xmax, ymax in ego),
        "gamma_masks":   (N, n_g)  bool,
        "gamma_scores":  (n_g,)    float,
        "gamma_aabbs":   (n_g, 4)  float (BEV AABB derived from rotated box),
    }

Hard-locked configs (per α spec):
    β1: pillar=(0.5,0.5), z_thr=0.3, ground=percentile p=10,
         HDBSCAN min_cluster_size=3, min_samples=3, eps=1.0
    γ:  score_threshold=0.2, nms_iou_threshold=0.1
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box

from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from adapters.centerpoint_proposals import CenterPointProposalGenerator
from diagnosis_step1.matching import (
    match_gt_to_instances,
    cluster_ids_to_masks,
)
from diagnosis_w1.measurements import match_gt_to_clusters
from diagnosis.measurements import distance_bin


# Locked configs
BETA1_PILLAR = {
    "pillar_size_xy": (0.5, 0.5),
    "z_threshold": 0.3,
    "ground_estimation": "percentile",
    "percentile_p": 10.0,
}
BETA1_HDBSCAN = {
    "min_cluster_size": 3,
    "min_samples": 3,
    "cluster_selection_epsilon": 1.0,
    "ground_filter": "z_threshold",
    "ground_z_max": -1.4,
}
GAMMA_THRESHOLDS = {
    "score_threshold": 0.20,
    "nms_iou_threshold": 0.10,
}

PER_SAMPLE_TIMEOUT_S = 90    # β1 ≤1s + γ ≤0.16s + 4 strategies — buffered


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


# -- β1 path ----------------------------------------------------------------

def run_beta1(extractor: PillarForegroundExtractor,
              hdbscan_gen: LiDARProposalGenerator,
              pc_ego: np.ndarray) -> dict:
    """β1 best-config inference. Returns cluster_ids over the *full* pc_ego.

    The foreground extractor returns a reduced point cloud; we re-expand
    cluster_ids back to the original index space so both sources share the
    same row basis. Points not in foreground (or noise) get id = -1.
    """
    t0 = time.perf_counter()
    fg = extractor.extract(pc_ego)
    fg_pcd = fg["foreground_pcd"]
    fg_mask = ~fg["background_mask"]    # bool over original pc_ego rows
    t1 = time.perf_counter()

    full_ids = np.full(pc_ego.shape[0], -1, dtype=np.int64)
    cluster_centroids = np.zeros((0, 3))
    cluster_sizes = np.zeros((0,), dtype=np.int64)
    cluster_bbox = np.zeros((0, 6))
    n_clusters = 0
    if fg_pcd.shape[0] > 0:
        h_out = hdbscan_gen.generate(fg_pcd)
        n_clusters = int(h_out["n_clusters"])
        cluster_centroids = h_out["cluster_centroids"]
        cluster_sizes = h_out["cluster_sizes"]
        cluster_bbox = h_out["cluster_bbox"]
        # h_out["cluster_ids"] is over foreground_pcd rows; expand to full pc_ego
        # via the foreground mask so β1 + γ share the same row basis.
        full_ids[fg_mask] = h_out["cluster_ids"]
    t2 = time.perf_counter()

    return {
        "full_cluster_ids": full_ids,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.astype(np.int64),
        "cluster_centroids": cluster_centroids,
        "cluster_bbox": cluster_bbox,
        "timing": {"foreground_s": float(t1 - t0),
                   "hdbscan_s": float(t2 - t1),
                   "total_s": float(t2 - t0)},
        "n_foreground": int(fg["n_foreground_points"]),
        "foreground_ratio": float(fg["foreground_ratio"]),
    }


def beta1_to_masks_and_meta(beta1_out: dict, pc_xyz: np.ndarray) -> dict:
    """Convert β1 cluster_ids → (N_pts, n_clusters) bool + sizes + BEV AABB."""
    n_clusters = int(beta1_out["n_clusters"])
    full_ids = beta1_out["full_cluster_ids"]
    if n_clusters == 0:
        return {
            "masks": np.zeros((pc_xyz.shape[0], 0), dtype=bool),
            "sizes": np.zeros((0,), dtype=np.int64),
            "aabbs": np.zeros((0, 4), dtype=np.float64),
        }
    masks = cluster_ids_to_masks(full_ids, n_clusters)
    bbox = beta1_out["cluster_bbox"]    # (n_clusters, 6) xmin..xmax
    aabbs = np.stack([bbox[:, 0], bbox[:, 1], bbox[:, 3], bbox[:, 4]], axis=1)
    return {
        "masks": masks,
        "sizes": beta1_out["cluster_sizes"].astype(np.int64),
        "aabbs": aabbs.astype(np.float64),
    }


# -- γ path ------------------------------------------------------------------

def _box_lidar_to_ego_box(b_lidar, T_lidar_to_ego):
    x, y, z = b_lidar[0], b_lidar[1], b_lidar[2]
    w, l, h = b_lidar[3], b_lidar[4], b_lidar[5]
    yaw = b_lidar[6]
    rot = Quaternion(axis=[0, 0, 1], angle=float(yaw))
    box = Box(center=np.array([x, y, z], dtype=np.float64),
              size=np.array([w, l, h], dtype=np.float64),
              orientation=rot)
    box.rotate(Quaternion(matrix=T_lidar_to_ego[:3, :3]))
    box.translate(T_lidar_to_ego[:3, 3])
    return box


def gamma_proposals_to_artifacts(proposals: list,
                                  T_lidar_to_ego: np.ndarray,
                                  pc_ego_xyz: np.ndarray) -> dict:
    """Per-proposal points-in-3D-box mask + BEV AABB + score.

    Identical containment logic to ``diagnosis_gamma.measurements`` so the
    γ-alone regression matches the saved per-sample numbers bit-for-bit.
    """
    N = pc_ego_xyz.shape[0]
    n_prop = len(proposals)
    masks = np.zeros((N, n_prop), dtype=bool)
    scores = np.zeros((n_prop,), dtype=np.float64)
    aabbs = np.zeros((n_prop, 4), dtype=np.float64)
    pts_T = pc_ego_xyz.T
    for j, p in enumerate(proposals):
        scores[j] = float(p.get("score", 0.0))
        try:
            box = _box_lidar_to_ego_box(p["bbox_lidar"], T_lidar_to_ego)
            inside = points_in_box(box, pts_T)
            masks[:, j] = inside
            corners = box.corners()           # (3, 8)
            aabbs[j, 0] = float(corners[0].min())
            aabbs[j, 1] = float(corners[1].min())
            aabbs[j, 2] = float(corners[0].max())
            aabbs[j, 3] = float(corners[1].max())
        except Exception:
            pass
    return {"masks": masks, "scores": scores, "aabbs": aabbs}


# -- single-sample assembly --------------------------------------------------

def run_sources_once(extractor: PillarForegroundExtractor,
                     hdbscan_gen: LiDARProposalGenerator,
                     cp_gen: CenterPointProposalGenerator,
                     cached_record: dict,
                     work_root: str) -> dict:
    """Run β1 and γ once; return the unified ``sample_artifacts`` dict.

    Also returns β1-alone and γ-alone matching results so the regression
    check (β1 M=0.3612, γ M=0.3519) can be compared per-sample without
    re-running inference.
    """
    pc_ego = cached_record["pc_ego"]
    pc_xyz = pc_ego[:, :3].astype(np.float64, copy=False)
    T_l2e = cached_record["T_lidar_to_ego"]

    # ---- β1 ----
    t_b0 = time.perf_counter()
    b1_out = run_beta1(extractor, hdbscan_gen, pc_ego)
    b1_meta = beta1_to_masks_and_meta(b1_out, pc_xyz)
    t_b1 = time.perf_counter()

    # β1-alone matching (regression)
    per_gt_b1, cases_b1 = match_gt_to_clusters(
        cached_record["gt_boxes"], cached_record["ego_pose"], pc_xyz,
        b1_out["full_cluster_ids"],
    )
    n_gt = len(cached_record["gt_boxes"])

    # ---- γ ----
    t_g0 = time.perf_counter()
    bin_path = os.path.join(work_root, f"{cached_record['sample_token']}_alpha.bin")
    cp_out = cp_gen.generate(pc_ego, T_l2e, bin_path)
    try:
        os.remove(bin_path)
    except FileNotFoundError:
        pass
    g_arts = gamma_proposals_to_artifacts(cp_out["proposals"], T_l2e, pc_xyz)
    t_g1 = time.perf_counter()

    per_gt_g, cases_g = match_gt_to_instances(
        cached_record["gt_boxes"], cached_record["ego_pose"], pc_xyz, g_arts["masks"],
    )

    # GT distance + bin (precompute once for downstream stratification)
    gt_distances = []
    gt_bins = []
    for g in per_gt_b1:
        gt_distances.append(g.get("distance_m"))
        gt_bins.append(g.get("distance_bin"))

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "n_gt_total": n_gt,
        "gt_distances": gt_distances,
        "gt_bins": gt_bins,
        # unified artifacts (handed to union_strategies.apply_strategy)
        "artifacts": {
            "gt_boxes": cached_record["gt_boxes"],
            "ego_pose": cached_record["ego_pose"],
            "pc_xyz": pc_xyz,
            "beta1_masks": b1_meta["masks"],
            "beta1_sizes": b1_meta["sizes"],
            "beta1_aabbs": b1_meta["aabbs"],
            "gamma_masks": g_arts["masks"],
            "gamma_scores": g_arts["scores"],
            "gamma_aabbs": g_arts["aabbs"],
        },
        # source-alone matching (used for regression + complementarity)
        "beta1_alone": {
            "case_counts": cases_b1,
            "M_rate": (cases_b1["M"] / n_gt) if n_gt else 0.0,
            "L_rate": (cases_b1["L"] / n_gt) if n_gt else 0.0,
            "D_rate": (cases_b1["D"] / n_gt) if n_gt else 0.0,
            "miss_rate": (cases_b1["miss"] / n_gt) if n_gt else 0.0,
            "n_clusters": int(b1_out["n_clusters"]),
            "per_gt": per_gt_b1,
            "timing_s": float(t_b1 - t_b0),
        },
        "gamma_alone": {
            "case_counts": cases_g,
            "M_rate": (cases_g["M"] / n_gt) if n_gt else 0.0,
            "L_rate": (cases_g["L"] / n_gt) if n_gt else 0.0,
            "D_rate": (cases_g["D"] / n_gt) if n_gt else 0.0,
            "miss_rate": (cases_g["miss"] / n_gt) if n_gt else 0.0,
            "n_proposals": int(cp_out["n_proposals"]),
            "per_gt": per_gt_g,
            "timing_s": float(t_g1 - t_g0),
        },
    }


def build_per_sample_strategy_record(strategy_combo: dict,
                                      strategy_out: dict,
                                      sample_pack: dict,
                                      strategy_timing_s: float) -> dict:
    """Distill one (sample × strategy) record for JSON dump + aggregation."""
    n_gt = sample_pack["n_gt_total"]
    cases = strategy_out["case_counts"]
    per_gt = strategy_out["per_gt"]
    pgb1 = sample_pack["beta1_alone"]["per_gt"]
    pgg = sample_pack["gamma_alone"]["per_gt"]

    # coverage breakdown — independent of strategy but reported alongside
    cov = {"both": 0, "beta1_only": 0, "gamma_only": 0, "neither": 0}
    for gb1, gg in zip(pgb1, pgg):
        b_hit = gb1["case"] != "miss"
        g_hit = gg["case"] != "miss"
        if b_hit and g_hit:
            cov["both"] += 1
        elif b_hit:
            cov["beta1_only"] += 1
        elif g_hit:
            cov["gamma_only"] += 1
        else:
            cov["neither"] += 1

    # per-distance-bin
    BIN_LABELS = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+", "unknown"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0} for b in BIN_LABELS}
    for g in per_gt:
        b = g.get("distance_bin") or "unknown"
        if b not in by_bin:
            b = "unknown"
        by_bin[b]["n_GT"] += 1
        by_bin[b][g["case"]] += 1

    return {
        "sample_token": sample_pack["sample_token"],
        "source": sample_pack["source"],
        "strategy_combo_id": strategy_combo["combo_id"],
        "strategy": strategy_combo["strategy"],
        "params": strategy_combo["params"],
        "n_gt_total": n_gt,
        "case_counts": cases,
        "M_rate": (cases["M"] / n_gt) if n_gt else 0.0,
        "L_rate": (cases["L"] / n_gt) if n_gt else 0.0,
        "D_rate": (cases["D"] / n_gt) if n_gt else 0.0,
        "miss_rate": (cases["miss"] / n_gt) if n_gt else 0.0,
        "n_proposals_total": int(strategy_out["n_proposals_total"]),
        "n_proposals_beta1": int(strategy_out["n_proposals_beta1"]),
        "n_proposals_gamma": int(strategy_out["n_proposals_gamma"]),
        "strategy_timing_s": float(strategy_timing_s),
        "coverage": cov,
        "by_distance_bin": by_bin,
        "beta1_alone_M_rate": sample_pack["beta1_alone"]["M_rate"],
        "gamma_alone_M_rate": sample_pack["gamma_alone"]["M_rate"],
        "beta1_alone_n_proposals": sample_pack["beta1_alone"]["n_clusters"],
        "gamma_alone_n_proposals": sample_pack["gamma_alone"]["n_proposals"],
        "extras": {
            k: v for k, v in strategy_out.items()
            if k not in ("per_gt", "case_counts",
                         "n_proposals_total", "n_proposals_beta1", "n_proposals_gamma")
        },
        "per_gt": per_gt,
        "status": "ok",
    }
