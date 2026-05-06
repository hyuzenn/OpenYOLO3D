"""Step A measurement helpers.

Adds two analyses on top of the β1 pipeline (pillar foreground + HDBSCAN):

  1. ``analyze_components`` — for the same BEV connected components that
     β1.5's verticality_filter would inspect, mark each component as
     too_small (size < 3 pillars) or kept, and as inside_gt (any of its
     foreground points lies inside any GT 3D box) or outside_gt. We don't
     call VerticalityFilter — we replicate the connected-component step
     so β1.5 stays untouched.

  2. ``pillars_per_gt`` — count the unique pillars covered by foreground
     points falling inside each GT box. Diagnoses whether smaller pillars
     give small/far GTs more "pillars to live in" before the size-min
     filter fires.
"""

from __future__ import annotations

import time
import signal

import numpy as np
from scipy.ndimage import label as scipy_label, find_objects

from adapters.lidar_proposals import LiDARProposalGenerator
from preprocessing.pillar_foreground import PillarForegroundExtractor
from diagnosis_w1.measurements import match_gt_to_clusters
from diagnosis.measurements import gt_box_to_ego, points_inside_3d_box, distance_bin


PER_SAMPLE_TIMEOUT_S = 35
SIZE_MIN_FOR_TOO_SMALL = 3   # mirrors β1.5 default size_min so analysis is
                              # directly comparable to that experiment


# Locked HDBSCAN config = W1.5 best
HDBSCAN_BEST = {
    "min_cluster_size": 3,
    "min_samples": 3,
    "cluster_selection_epsilon": 1.0,
    "ground_filter": "z_threshold",
    "ground_z_max": -1.4,
}


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


def _bev_components(foreground_pcd: np.ndarray, pillar_size_xy):
    """4-connectivity BEV connected components on the foreground point cloud.

    Returns a dict with point→component mapping plus a per-component table.
    """
    pts = foreground_pcd[:, :3]
    M = pts.shape[0]
    if M == 0:
        return {
            "n_components": 0,
            "point_component": np.zeros((0,), dtype=np.int64),
            "components": [],
        }

    px, py = pillar_size_xy
    ix = np.floor(pts[:, 0] / px).astype(np.int64)
    iy = np.floor(pts[:, 1] / py).astype(np.int64)
    ix_min, iy_min = int(ix.min()), int(iy.min())
    H = int(ix.max()) - ix_min + 1
    W = int(iy.max()) - iy_min + 1
    grid = np.zeros((H, W), dtype=bool)
    ix_o = ix - ix_min
    iy_o = iy - iy_min
    grid[ix_o, iy_o] = True
    labeled, n_components = scipy_label(grid)
    if n_components == 0:
        return {
            "n_components": 0,
            "point_component": np.zeros((M,), dtype=np.int64),
            "components": [],
        }
    point_component = labeled[ix_o, iy_o]
    slices = find_objects(labeled)
    comps = []
    for cid in range(1, n_components + 1):
        sl = slices[cid - 1]
        if sl is None:
            comps.append({"cid": cid, "n_pillars": 0})
            continue
        sub = labeled[sl] == cid
        comps.append({
            "cid": cid,
            "n_pillars": int(sub.sum()),
        })
    return {
        "n_components": int(n_components),
        "point_component": point_component,
        "components": comps,
        "pillar_key": (ix_o, iy_o),  # for downstream pillars_per_gt
    }


def analyze_components_vs_gt(foreground_pcd, gt_boxes, ego_pose_4x4, pillar_size_xy,
                              size_min: int = SIZE_MIN_FOR_TOO_SMALL):
    """Classify every BEV component as (too_small | kept) × (inside_gt | outside_gt).

    Component is "inside_gt" if at least one of its foreground points lies
    inside any GT's 3D box (in ego frame). "too_small" if its pillar count
    is below ``size_min`` (default 3, matching β1.5).
    """
    bev = _bev_components(foreground_pcd, pillar_size_xy)
    n_comp = bev["n_components"]
    if n_comp == 0:
        return {
            "n_components": 0,
            "n_too_small": 0,
            "n_kept": 0,
            "too_small_inside_gt": 0,
            "too_small_outside_gt": 0,
            "kept_inside_gt": 0,
            "kept_outside_gt": 0,
            "too_small_inside_gt_ratio": 0.0,
            "too_small_outside_gt_ratio": 0.0,
            "kept_inside_gt_ratio": 0.0,
            "kept_outside_gt_ratio": 0.0,
        }

    point_component = bev["point_component"]
    components = bev["components"]
    M = foreground_pcd.shape[0]

    # Per-component: which foreground points belong to it?
    # Use bincount-like inverse mapping. point_component values are 1..n_comp.

    # Per-GT: which foreground points fall inside the 3D box?
    if gt_boxes:
        in_any_gt = np.zeros(M, dtype=bool)
        for gt in gt_boxes:
            try:
                box_ego = gt_box_to_ego(gt, ego_pose_4x4)
                in_any_gt |= points_inside_3d_box(box_ego, foreground_pcd[:, :3])
            except Exception:
                continue
    else:
        in_any_gt = np.zeros(M, dtype=bool)

    # Per-component, ask: any in-GT foreground point belongs to this component?
    n_too_small = n_kept = 0
    too_small_in = too_small_out = 0
    kept_in = kept_out = 0
    for comp in components:
        cid = comp["cid"]
        n_pillars = comp["n_pillars"]
        if n_pillars == 0:
            continue
        comp_mask = point_component == cid
        inside_gt = bool((comp_mask & in_any_gt).any())
        too_small = n_pillars < size_min
        if too_small:
            n_too_small += 1
            if inside_gt:
                too_small_in += 1
            else:
                too_small_out += 1
        else:
            n_kept += 1
            if inside_gt:
                kept_in += 1
            else:
                kept_out += 1

    return {
        "n_components": int(n_comp),
        "n_too_small": int(n_too_small),
        "n_kept": int(n_kept),
        "too_small_inside_gt": int(too_small_in),
        "too_small_outside_gt": int(too_small_out),
        "kept_inside_gt": int(kept_in),
        "kept_outside_gt": int(kept_out),
        "too_small_inside_gt_ratio": (too_small_in / n_too_small) if n_too_small else 0.0,
        "too_small_outside_gt_ratio": (too_small_out / n_too_small) if n_too_small else 0.0,
        "kept_inside_gt_ratio": (kept_in / n_kept) if n_kept else 0.0,
        "kept_outside_gt_ratio": (kept_out / n_kept) if n_kept else 0.0,
    }


def pillars_per_gt(foreground_pcd, gt_boxes, ego_pose_4x4, pillar_size_xy):
    """For each GT, count the unique BEV pillars covered by foreground points
    inside its 3D box. Returns list parallel to gt_boxes; 0 means the GT had
    no foreground points inside its box.
    """
    if foreground_pcd.shape[0] == 0 or not gt_boxes:
        return []
    px, py = pillar_size_xy
    ix = np.floor(foreground_pcd[:, 0] / px).astype(np.int64)
    iy = np.floor(foreground_pcd[:, 1] / py).astype(np.int64)
    # composite key — values fit in int64
    OFFSET = 1024
    STRIDE = 2048
    key = (ix + OFFSET).clip(0, STRIDE - 1) * STRIDE + (iy + OFFSET).clip(0, STRIDE - 1)
    counts = []
    for gt in gt_boxes:
        try:
            box_ego = gt_box_to_ego(gt, ego_pose_4x4)
            in3d = points_inside_3d_box(box_ego, foreground_pcd[:, :3])
        except Exception:
            counts.append(0)
            continue
        if in3d.sum() == 0:
            counts.append(0)
        else:
            counts.append(int(np.unique(key[in3d]).size))
    return counts


def measure_one(extractor: PillarForegroundExtractor,
                hdbscan_gen: LiDARProposalGenerator,
                cached_record: dict) -> dict:
    pc = cached_record["pc_ego"]
    t0 = time.perf_counter()

    fg = extractor.extract(pc)
    foreground_pcd = fg["foreground_pcd"]
    t1 = time.perf_counter()

    if foreground_pcd.shape[0] == 0:
        n_gt = len(cached_record["gt_boxes"])
        return _empty_record(cached_record, t0, fg, n_gt)

    h_out = hdbscan_gen.generate(foreground_pcd)
    cluster_ids = h_out["cluster_ids"]
    fg_xyz = foreground_pcd[:, :3]
    per_gt, cases = match_gt_to_clusters(
        cached_record["gt_boxes"], cached_record["ego_pose"], fg_xyz, cluster_ids,
    )
    n_gt = len(cached_record["gt_boxes"])
    t2 = time.perf_counter()

    # Step-A specific: component spatial analysis + per-GT pillar count
    cmp_stats = analyze_components_vs_gt(
        foreground_pcd, cached_record["gt_boxes"], cached_record["ego_pose"],
        extractor.config.pillar_size_xy,
    )
    pillars_counts = pillars_per_gt(
        foreground_pcd, cached_record["gt_boxes"], cached_record["ego_pose"],
        extractor.config.pillar_size_xy,
    )
    t3 = time.perf_counter()

    avg_pillars_per_gt = float(np.mean(pillars_counts)) if pillars_counts else 0.0
    median_pillars_per_gt = float(np.median(pillars_counts)) if pillars_counts else 0.0

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "foreground": {
            "n_input": fg["n_input_points"],
            "n_foreground": fg["n_foreground_points"],
            "ratio": fg["foreground_ratio"],
            "n_pillars_total": fg["n_pillars_total"],
            "n_pillars_foreground": fg["n_pillars_foreground"],
            "timing": fg["timing"],
        },
        "n_clusters": int(h_out["n_clusters"]),
        "noise_ratio": float(h_out["noise_ratio"]),
        "hdbscan_timing": dict(h_out["timing"]),
        "n_gt_total": n_gt,
        "case_counts": cases,
        "M_rate": (cases["M"] / n_gt) if n_gt else 0.0,
        "L_rate": (cases["L"] / n_gt) if n_gt else 0.0,
        "D_rate": (cases["D"] / n_gt) if n_gt else 0.0,
        "miss_rate": (cases["miss"] / n_gt) if n_gt else 0.0,
        "components": cmp_stats,
        "pillars_per_gt_box": pillars_counts,
        "avg_pillars_per_gt_box": avg_pillars_per_gt,
        "median_pillars_per_gt_box": median_pillars_per_gt,
        "timing_total": float(time.perf_counter() - t0),
        "timing_breakdown": {
            "foreground": float(t1 - t0),
            "hdbscan_match": float(t2 - t1),
            "analysis": float(t3 - t2),
        },
        "per_gt": per_gt,
        "status": "ok",
    }


def _empty_record(cached_record, t0, fg, n_gt):
    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "foreground": {
            "n_input": fg["n_input_points"],
            "n_foreground": 0,
            "ratio": 0.0,
            "timing": fg["timing"],
        },
        "n_clusters": 0,
        "noise_ratio": 0.0,
        "n_gt_total": n_gt,
        "case_counts": {"M": 0, "L": 0, "D": 0, "miss": n_gt},
        "M_rate": 0.0, "L_rate": 0.0, "D_rate": 0.0,
        "miss_rate": 1.0 if n_gt else 0.0,
        "components": {"n_components": 0, "n_too_small": 0, "n_kept": 0,
                        "too_small_inside_gt": 0, "too_small_outside_gt": 0,
                        "kept_inside_gt": 0, "kept_outside_gt": 0,
                        "too_small_inside_gt_ratio": 0.0,
                        "too_small_outside_gt_ratio": 0.0,
                        "kept_inside_gt_ratio": 0.0,
                        "kept_outside_gt_ratio": 0.0},
        "pillars_per_gt_box": [],
        "avg_pillars_per_gt_box": 0.0,
        "median_pillars_per_gt_box": 0.0,
        "timing_total": float(time.perf_counter() - t0),
        "per_gt": [],
        "status": "ok",
    }


def measure_with_timeout(extractor, hdbscan_gen, cached_record,
                         timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    _set_alarm(timeout_s)
    try:
        return measure_one(extractor, hdbscan_gen, cached_record)
    finally:
        _clear_alarm()
