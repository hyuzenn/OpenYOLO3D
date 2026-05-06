"""Phase C — Distance-stratified analysis.

Re-runs Phase B's best config on the cached samples and bins per-GT case
labels by GT distance. Verdict picks C1/C2/C3 based on cross-bin M_rate
distribution. Phase C is informational regardless of B verdict.
"""

from __future__ import annotations

import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis.measurements import DISTANCE_BIN_LABELS
from diagnosis_w1_5.measurements import SampleTimeout, run_with_timeout


M_RATE_OK = 0.20    # all bins ≥ 0.20 → uniform OK
NEAR_M_OK = 0.30    # near (0–20m) ≥ 0.30 ...
FAR_M_FAIL = 0.10   # ... and far (30+m) < 0.10 → distance gap
M_RATE_FAIL = 0.10  # all bins < 0.10 → uniform fail


def run_phase_c(cache: dict, phase_b_verdict: dict, out_dir: str) -> dict:
    phase_dir = osp.join(out_dir, "phase_c")
    figures_dir = osp.join(phase_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    best = phase_b_verdict["best_config"]
    ground_kwargs = phase_b_verdict["ground_filter_kwargs"]
    print(f"\n=== Phase C — distance stratification with Phase B best "
          f"(min_cluster={best['min_cluster_size']}, "
          f"min_samples={best['min_samples']}, "
          f"eps={best['cluster_selection_epsilon']}, "
          f"ground={ground_kwargs}) ===")

    gen = LiDARProposalGenerator(
        min_cluster_size=best["min_cluster_size"],
        min_samples=best["min_samples"],
        cluster_selection_epsilon=best["cluster_selection_epsilon"],
        **ground_kwargs,
    )

    tokens = sorted(cache.keys())
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}

    overall = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
    succeeded = 0
    timeouts = 0
    for tok in tokens:
        try:
            rec = run_with_timeout(gen, cache[tok])
        except SampleTimeout:
            timeouts += 1
            continue
        succeeded += 1
        for g in rec["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in by_bin:
                b = "unknown"
            by_bin[b][g["case"]] += 1
            by_bin[b]["n_GT"] += 1
            overall[g["case"]] += 1
            overall["n_GT"] += 1

    # Per-bin rates
    bin_rates = {}
    for b, rec in by_bin.items():
        n = rec["n_GT"]
        bin_rates[b] = {
            "n_GT": n,
            "M_rate": (rec["M"] / n) if n else 0.0,
            "L_rate": (rec["L"] / n) if n else 0.0,
            "D_rate": (rec["D"] / n) if n else 0.0,
            "miss_rate": (rec["miss"] / n) if n else 0.0,
        }
    overall_M = (overall["M"] / overall["n_GT"]) if overall["n_GT"] else 0.0

    # ---- verdict ----
    bins_with_data = [b for b in DISTANCE_BIN_LABELS if by_bin[b]["n_GT"] > 0]
    M_rates = [bin_rates[b]["M_rate"] for b in bins_with_data]
    near_bins = [b for b in ["0-10m", "10-20m"] if by_bin[b]["n_GT"] > 0]
    far_bins = [b for b in ["30-50m", "50m+"] if by_bin[b]["n_GT"] > 0]
    near_M = (sum(by_bin[b]["M"] for b in near_bins) /
              max(1, sum(by_bin[b]["n_GT"] for b in near_bins)))
    far_M = (sum(by_bin[b]["M"] for b in far_bins) /
             max(1, sum(by_bin[b]["n_GT"] for b in far_bins)))

    if M_rates and all(M >= M_RATE_OK for M in M_rates):
        verdict = "C1_UNIFORM_OK"
        verdict_text = "All distance bins clear M≥0.20. Distance-aware variant unnecessary."
    elif near_M >= NEAR_M_OK and far_M < FAR_M_FAIL:
        verdict = "C2_DISTANCE_GAP"
        verdict_text = (f"Near-range OK (M={near_M*100:.1f}%), far-range fails "
                        f"(M={far_M*100:.1f}%). Distance-aware clustering is the "
                        f"natural next experiment.")
    elif M_rates and all(M < M_RATE_FAIL for M in M_rates):
        verdict = "C3_UNIFORM_FAIL"
        verdict_text = ("All distance bins fail (M<0.10). HDBSCAN as proposal generator "
                        "does not work outdoors at any distance — CenterPoint is the "
                        "responsible escalation.")
    else:
        verdict = "C_MIXED"
        verdict_text = ("Mixed across distance bins — no clean structural read. "
                        "Inspect bin table and decide manually.")

    record = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "best_config_used": {**best, **ground_kwargs},
        "succeeded": succeeded,
        "timeouts": timeouts,
        "overall": overall,
        "overall_M_rate": overall_M,
        "by_bin": {b: {**by_bin[b], **bin_rates[b]} for b in by_bin},
        "near_M_rate": near_M,
        "far_M_rate": far_M,
        "thresholds": {
            "M_rate_ok": M_RATE_OK,
            "near_M_ok": NEAR_M_OK,
            "far_M_fail": FAR_M_FAIL,
            "M_rate_fail": M_RATE_FAIL,
        },
    }
    with open(osp.join(phase_dir, "distance_stratified.json"), "w") as f:
        json.dump(record, f, indent=2)
    with open(osp.join(phase_dir, "verdict.json"), "w") as f:
        json.dump({"verdict": verdict, "verdict_text": verdict_text,
                   "near_M_rate": near_M, "far_M_rate": far_M,
                   "by_bin_rates": bin_rates}, f, indent=2)

    _plot_phase_c(by_bin, bin_rates, overall_M, figures_dir)

    print(f"  → Phase C verdict: {verdict}  (near_M={near_M*100:.1f}%, far_M={far_M*100:.1f}%)")
    return record


def _plot_phase_c(by_bin, bin_rates, overall_M, figures_dir):
    bins = DISTANCE_BIN_LABELS
    M_vals = [bin_rates[b]["M_rate"] * 100 for b in bins]
    miss_vals = [bin_rates[b]["miss_rate"] * 100 for b in bins]
    n_GT_per = [by_bin[b]["n_GT"] for b in bins]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(bins, M_vals, color="#117733")
    ax.axhline(overall_M * 100, color="red", ls="--", label=f"overall M={overall_M*100:.1f}%")
    for bar, v, n in zip(bars, M_vals, n_GT_per):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("M_rate (%)")
    ax.set_title("Phase C — M_rate by distance bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "M_rate_by_distance.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(bins, miss_vals, color="#882255")
    overall_miss = sum(by_bin[b]["miss"] for b in bins) / max(1, sum(by_bin[b]["n_GT"] for b in bins))
    ax.axhline(overall_miss * 100, color="red", ls="--",
               label=f"overall miss={overall_miss*100:.1f}%")
    for bar, v, n in zip(bars, miss_vals, n_GT_per):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("miss_rate (%)")
    ax.set_title("Phase C — miss_rate by distance bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "miss_rate_by_distance.png"), dpi=120)
    plt.close(fig)

    # stacked bar (M/L/D/miss) per bin
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cases = ["M", "L", "D", "miss"]
    colors = {"M": "#117733", "L": "#DDCC77", "D": "#882255", "miss": "#999999"}
    bottoms = np.zeros(len(bins))
    for c in cases:
        vals = np.array([bin_rates[b][f"{c}_rate"] * 100 for b in bins])
        ax.bar(bins, vals, bottom=bottoms, color=colors[c], label=c)
        bottoms += vals
    ax.set_ylabel("rate (%)")
    ax.set_title("Phase C — case distribution by distance bin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "case_distribution_by_distance.png"), dpi=120)
    plt.close(fig)
