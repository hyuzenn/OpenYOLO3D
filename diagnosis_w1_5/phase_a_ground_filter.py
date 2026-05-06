"""Phase A — Ground filter sanity.

9 ground-filter configs × cached samples, HDBSCAN nominal config (W1 best:
min_cluster=50, min_samples=10, eps=1.0) held fixed. Verdict picks a winner
when one exists in the operational band; otherwise records that ground filter
is not the root cause and leaves Phase B to run on W1 default.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time
import traceback
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_w1_5.measurements import (
    SampleTimeout, run_with_timeout, PER_SAMPLE_TIMEOUT_S,
)


# Phase A grid
GROUND_CONFIGS = [
    # (label, kwargs to LiDARProposalGenerator)
    ("z_threshold_-1.0", {"ground_filter": "z_threshold", "ground_z_max": -1.0}),
    ("z_threshold_-1.2", {"ground_filter": "z_threshold", "ground_z_max": -1.2}),
    ("z_threshold_-1.4", {"ground_filter": "z_threshold", "ground_z_max": -1.4}),
    ("z_threshold_-1.6", {"ground_filter": "z_threshold", "ground_z_max": -1.6}),
    ("z_threshold_-1.8", {"ground_filter": "z_threshold", "ground_z_max": -1.8}),
    ("ransac_default",   {"ground_filter": "ransac"}),
    ("percentile_10",    {"ground_filter": "percentile", "percentile_p": 10.0}),
    ("percentile_15",    {"ground_filter": "percentile", "percentile_p": 15.0}),
    ("percentile_20",    {"ground_filter": "percentile", "percentile_p": 20.0}),
]

# HDBSCAN nominal config — W1 best
HDBSCAN_NOMINAL = {"min_cluster_size": 50, "min_samples": 10, "cluster_selection_epsilon": 1.0}

# Verdict thresholds
N_BAND = (5, 30)
MISS_RATE_BAND = 0.40
TIMING_BAND_S = 1.5
MISS_RATE_FAIL_FLOOR = 0.50  # all configs at this or above → ground filter not root cause


def run_phase_a(cache: dict, out_dir: str) -> dict:
    """Run Phase A on cached samples; write per-config JSONs + figures + verdict."""
    phase_dir = osp.join(out_dir, "phase_a")
    per_config_dir = osp.join(phase_dir, "per_config")
    figures_dir = osp.join(phase_dir, "figures")
    os.makedirs(per_config_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    tokens = sorted(cache.keys())
    print(f"\n=== Phase A — {len(GROUND_CONFIGS)} ground filter configs × {len(tokens)} samples ===")

    config_summaries = []
    for label, gf_kwargs in GROUND_CONFIGS:
        gen = LiDARProposalGenerator(**HDBSCAN_NOMINAL, **gf_kwargs)
        per_sample = []
        timeouts = 0
        errors = 0
        for tok in tokens:
            try:
                rec = run_with_timeout(gen, cache[tok])
                per_sample.append(rec)
            except SampleTimeout:
                timeouts += 1
            except Exception as e:
                errors += 1
                print(f"  [Phase A | {label} | {tok}] {e}")
        if not per_sample:
            print(f"  {label}: NO successful samples")
            continue

        n_cl = [r["n_clusters"] for r in per_sample]
        miss = [r["miss_rate"] for r in per_sample]
        M = [r["M_rate"] for r in per_sample]
        ground_ratio = [r["ground_filtered_ratio"] for r in per_sample]
        timing = [r["timing"]["total"] for r in per_sample]
        summary = {
            "label": label,
            "ground_filter_kwargs": gf_kwargs,
            "n_samples": len(per_sample),
            "n_timeouts": timeouts,
            "n_errors": errors,
            "mean_n_clusters": float(np.mean(n_cl)),
            "median_n_clusters": float(np.median(n_cl)),
            "mean_miss_rate": float(np.mean(miss)),
            "median_miss_rate": float(np.median(miss)),
            "mean_M_rate": float(np.mean(M)),
            "mean_ground_filtered_ratio": float(np.mean(ground_ratio)),
            "median_timing_s": float(np.median(timing)),
            "p95_timing_s": float(np.percentile(timing, 95)),
        }
        with open(osp.join(per_config_dir, f"{label}.json"), "w") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, "item") else str(o))
        config_summaries.append(summary)
        print(f"  {label}: n_cl={summary['mean_n_clusters']:.1f}, "
              f"miss={summary['mean_miss_rate']*100:.1f}%, "
              f"timing={summary['median_timing_s']:.3f}s, "
              f"ground_drop={summary['mean_ground_filtered_ratio']*100:.1f}%")

    # ---- verdict ----
    in_band = [c for c in config_summaries
               if N_BAND[0] <= c["mean_n_clusters"] <= N_BAND[1]
               and c["mean_miss_rate"] < MISS_RATE_BAND
               and c["median_timing_s"] < TIMING_BAND_S]
    all_high_miss = all(c["mean_miss_rate"] >= MISS_RATE_FAIL_FLOOR for c in config_summaries) \
                    if config_summaries else False

    if in_band:
        best = min(in_band, key=lambda c: c["mean_miss_rate"])
        verdict = "GROUND_FILTER_HELPS"
        verdict_text = (f"Ground filter strongly suspected as root cause. "
                        f"Best config: {best['label']} "
                        f"(n_cl={best['mean_n_clusters']:.1f}, "
                        f"miss={best['mean_miss_rate']*100:.1f}%, "
                        f"timing={best['median_timing_s']:.3f}s).")
    elif all_high_miss:
        # Fallback to W1 default
        best = next((c for c in config_summaries if c["label"] == "z_threshold_-1.4"),
                    config_summaries[0] if config_summaries else None)
        verdict = "GROUND_FILTER_NOT_ROOT_CAUSE"
        verdict_text = ("All ground filter configs land at miss_rate ≥ 50%. "
                        "Ground filter is not the root cause — Phase B will use the W1 default.")
    else:
        # mixed — neither in-band nor uniformly failing
        best = next((c for c in config_summaries if c["label"] == "z_threshold_-1.4"),
                    config_summaries[0] if config_summaries else None)
        verdict = "INCONCLUSIVE"
        verdict_text = ("Phase A inconclusive — no config met the in-band criteria but not all "
                        "configs failed at the floor either. Phase B will use the W1 default; "
                        "interpret with caveat.")

    verdict_record = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "best_config": best,
        "thresholds": {
            "n_band": list(N_BAND),
            "miss_rate_band": MISS_RATE_BAND,
            "timing_band_s": TIMING_BAND_S,
            "miss_rate_fail_floor": MISS_RATE_FAIL_FLOOR,
        },
        "config_summaries": config_summaries,
    }
    with open(osp.join(phase_dir, "verdict.json"), "w") as f:
        json.dump(verdict_record, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    _plot_phase_a(config_summaries, figures_dir)

    print(f"  → Phase A verdict: {verdict}")
    if best:
        print(f"  → Phase B will use: {best['label']}")
    return verdict_record


def _plot_phase_a(summaries, figures_dir):
    if not summaries:
        return
    labels = [s["label"] for s in summaries]
    ground_ratio = [s["mean_ground_filtered_ratio"] * 100 for s in summaries]
    miss = [s["mean_miss_rate"] * 100 for s in summaries]
    n_cl = [s["mean_n_clusters"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), ground_ratio, color="#117733")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean ground-filtered ratio (%)")
    ax.set_title("Phase A — Ground filter removal ratio per config")
    for bar, r in zip(bars, ground_ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.5, f"{r:.1f}%",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "ground_filter_ratio_vs_threshold.png"), dpi=120)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax1.bar(x - 0.18, miss, 0.36, color="#882255", label="mean miss_rate (%)")
    ax2 = ax1.twinx()
    line = ax2.plot(x + 0.18, n_cl, "o-", color="#4477AA", label="mean n_clusters")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("miss rate (%)", color="#882255")
    ax2.set_ylabel("n_clusters", color="#4477AA")
    ax2.axhspan(N_BAND[0], N_BAND[1], color="green", alpha=0.10)
    ax1.set_title("Phase A — miss_rate and n_clusters per ground-filter config")
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "miss_rate_by_ground_mode.png"), dpi=120)
    plt.close(fig)
