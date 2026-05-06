"""β1.5 measurement helpers — pillar foreground (β1 best, fixed) → verticality
filter (sweep variable) → HDBSCAN (W1.5 best, fixed) → GT matching.
"""

from __future__ import annotations

import time
import signal

import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from preprocessing.pillar_foreground import PillarForegroundExtractor
from preprocessing.verticality_filter import VerticalityFilter
from diagnosis_w1.measurements import match_gt_to_clusters


PER_SAMPLE_TIMEOUT_S = 40


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


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


def measure_one(vert_filter: VerticalityFilter,
                hdbscan_gen: LiDARProposalGenerator,
                cached_record: dict,
                foreground_pcd: np.ndarray) -> dict:
    """Run verticality + HDBSCAN + matching on a pre-computed β1 foreground PC."""
    t0 = time.perf_counter()
    if foreground_pcd.shape[0] == 0:
        return _empty_record(cached_record, t0, vert_filter)
    vf = vert_filter.filter(foreground_pcd)
    kept_pcd = vf["kept_pcd"]
    t1 = time.perf_counter()

    if kept_pcd.shape[0] == 0:
        n_gt = len(cached_record["gt_boxes"])
        return {
            "sample_token": cached_record["sample_token"],
            "source": cached_record["source"],
            "verticality": {**vf, "kept_pcd": None, "kept_mask": None},
            "n_clusters": 0,
            "noise_ratio": 0.0,
            "hdbscan_timing": {"total": 0.0},
            "n_gt_total": n_gt,
            "case_counts": {"M": 0, "L": 0, "D": 0, "miss": n_gt},
            "M_rate": 0.0, "L_rate": 0.0, "D_rate": 0.0,
            "miss_rate": 1.0 if n_gt else 0.0,
            "timing_total": float(t1 - t0),
            "per_gt": [],
            "status": "ok",
        }

    h_out = hdbscan_gen.generate(kept_pcd)
    cluster_ids = h_out["cluster_ids"]
    kept_xyz = kept_pcd[:, :3]
    per_gt, cases = match_gt_to_clusters(
        cached_record["gt_boxes"], cached_record["ego_pose"], kept_xyz, cluster_ids,
    )
    n_gt = len(cached_record["gt_boxes"])
    t2 = time.perf_counter()

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "verticality": {k: v for k, v in vf.items() if k not in ("kept_pcd", "kept_mask")},
        "n_clusters": int(h_out["n_clusters"]),
        "noise_ratio": float(h_out["noise_ratio"]),
        "hdbscan_timing": dict(h_out["timing"]),
        "n_gt_total": n_gt,
        "case_counts": cases,
        "M_rate": (cases["M"] / n_gt) if n_gt else 0.0,
        "L_rate": (cases["L"] / n_gt) if n_gt else 0.0,
        "D_rate": (cases["D"] / n_gt) if n_gt else 0.0,
        "miss_rate": (cases["miss"] / n_gt) if n_gt else 0.0,
        "timing_total": float(t2 - t0),
        "per_gt": per_gt,
        "status": "ok",
    }


def _empty_record(cached_record, t0, vf_obj):
    n_gt = len(cached_record["gt_boxes"])
    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "verticality": {"n_components_total": 0, "n_components_kept": 0,
                         "n_input_points": 0, "n_kept_points": 0,
                         "removed_point_ratio": 0.0,
                         "timing": {"total": 0.0}},
        "n_clusters": 0,
        "noise_ratio": 0.0,
        "hdbscan_timing": {"total": 0.0},
        "n_gt_total": n_gt,
        "case_counts": {"M": 0, "L": 0, "D": 0, "miss": n_gt},
        "M_rate": 0.0, "L_rate": 0.0, "D_rate": 0.0,
        "miss_rate": 1.0 if n_gt else 0.0,
        "timing_total": float(time.perf_counter() - t0),
        "per_gt": [],
        "status": "ok",
    }


def measure_with_timeout(vert_filter, hdbscan_gen, cached_record, foreground_pcd,
                         timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    _set_alarm(timeout_s)
    try:
        return measure_one(vert_filter, hdbscan_gen, cached_record, foreground_pcd)
    finally:
        _clear_alarm()
