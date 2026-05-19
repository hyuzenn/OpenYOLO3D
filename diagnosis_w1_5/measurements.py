"""W1.5 measurement helpers.

Mostly re-exports of W1 / Tier-1 primitives plus a small wrapper that runs
HDBSCAN on a cached sample and returns a compact per-sample record (no IO).
This keeps phase modules thin and ensures they all use the same matching code.
"""

from __future__ import annotations

import time
import signal

import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_w1.measurements import match_gt_to_clusters, cluster_extents
from diagnosis.measurements import distance_bin


PER_SAMPLE_TIMEOUT_S = 60


class SampleTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise SampleTimeout()


def measure_sample(generator: LiDARProposalGenerator, cached_record: dict) -> dict:
    """Run HDBSCAN on one cached sample and compute matching/case stats.

    cached_record carries the keys produced by diagnosis_w1.run_clustering_check._cache_samples:
      'pc_ego' (N,4), 'gt_boxes', 'ego_pose', 'sample_token', 'source'.
    """
    pc = cached_record["pc_ego"]
    out = generator.generate(pc)
    cluster_ids = out["cluster_ids"]
    per_gt, cases = match_gt_to_clusters(
        cached_record["gt_boxes"], cached_record["ego_pose"], pc[:, :3], cluster_ids,
    )
    n_total = len(cached_record["gt_boxes"])
    miss_rate = (cases["miss"] / n_total) if n_total else 0.0
    M_rate = (cases["M"] / n_total) if n_total else 0.0
    L_rate = (cases["L"] / n_total) if n_total else 0.0
    D_rate = (cases["D"] / n_total) if n_total else 0.0
    return {
        "sample_token": cached_record["sample_token"],
        "source": cached_record["source"],
        "n_clusters": int(out["n_clusters"]),
        "cluster_sizes": out["cluster_sizes"].tolist(),
        "noise_ratio": float(out["noise_ratio"]),
        "ground_filtered_ratio": float(out["ground_filtered_ratio"]),
        "timing": dict(out["timing"]),
        "n_gt_total": int(n_total),
        "case_counts": cases,
        "miss_rate": float(miss_rate),
        "M_rate": float(M_rate),
        "L_rate": float(L_rate),
        "D_rate": float(D_rate),
        "per_gt": per_gt,
    }


def run_with_timeout(generator: LiDARProposalGenerator, cached_record: dict,
                     timeout_s: int = PER_SAMPLE_TIMEOUT_S) -> dict:
    """Wrap measure_sample in signal.alarm. On timeout, raises SampleTimeout."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    try:
        out = measure_sample(generator, cached_record)
    finally:
        signal.alarm(0)
    return out
