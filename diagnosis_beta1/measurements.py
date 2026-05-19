"""β1 measurement helpers — wraps W1's GT-cluster matching for the
foreground-only point cloud.

The foreground subset shares the ego frame with the original PC (it's just a
row-subset), so W1's `match_gt_to_clusters` works directly when fed
``foreground_pcd[:, :3]`` and the cluster_ids returned by HDBSCAN on that
subset.
"""

from __future__ import annotations

import time
import signal

import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from preprocessing.pillar_foreground import PillarForegroundExtractor
from diagnosis_w1.measurements import match_gt_to_clusters


PER_SAMPLE_TIMEOUT_S = 35


class SampleTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SampleTimeout()


def _set_alarm(s):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(s)


def _clear_alarm():
    signal.alarm(0)


# Locked HDBSCAN config = W1.5 best (foreground hypothesis isolation).
HDBSCAN_BEST = {
    "min_cluster_size": 3,
    "min_samples": 3,
    "cluster_selection_epsilon": 1.0,
    "ground_filter": "z_threshold",
    "ground_z_max": -1.4,
}


def measure_one(extractor: PillarForegroundExtractor,
                hdbscan_gen: LiDARProposalGenerator,
                cached_record: dict) -> dict:
    """Run pillar foreground → HDBSCAN → GT matching for one sample."""
    pc = cached_record["pc_ego"]

    t0 = time.perf_counter()
    fg = extractor.extract(pc)
    foreground_pcd = fg["foreground_pcd"]
    t_fg = time.perf_counter() - t0

    if foreground_pcd.shape[0] == 0:
        # Empty foreground — short-circuit to all-miss
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
            "hdbscan_timing": {"total": 0.0},
            "n_gt_total": len(cached_record["gt_boxes"]),
            "case_counts": {"M": 0, "L": 0, "D": 0, "miss": len(cached_record["gt_boxes"])},
            "M_rate": 0.0, "L_rate": 0.0, "D_rate": 0.0,
            "miss_rate": 1.0 if cached_record["gt_boxes"] else 0.0,
            "timing_total": float(t_fg),
            "per_gt": [],
            "status": "ok",
        }

    t1 = time.perf_counter()
    h_out = hdbscan_gen.generate(foreground_pcd)
    t_h = time.perf_counter() - t1

    cluster_ids = h_out["cluster_ids"]
    fg_xyz = foreground_pcd[:, :3]
    per_gt, cases = match_gt_to_clusters(
        cached_record["gt_boxes"], cached_record["ego_pose"], fg_xyz, cluster_ids,
    )
    n_gt = len(cached_record["gt_boxes"])

    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "foreground": {
            "n_input": fg["n_input_points"],
            "n_foreground": fg["n_foreground_points"],
            "ratio": fg["foreground_ratio"],
            "n_pillars_total": fg["n_pillars_total"],
            "n_pillars_foreground": fg["n_pillars_foreground"],
            "ground_info": fg["ground_info"],
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
        "timing_total": float(t_fg + t_h),
        "per_gt": per_gt,
        "status": "ok",
    }


def measure_with_timeout(extractor, hdbscan_gen, cached_record,
                         timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    _set_alarm(timeout_s)
    try:
        return measure_one(extractor, hdbscan_gen, cached_record)
    finally:
        _clear_alarm()
