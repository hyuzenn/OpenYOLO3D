"""Phase B — Foreground-aware extended sweep.

84 (min_cluster_size, min_samples, eps) combos under the ground filter
selected by Phase A. Tiered selection rule (B1 strict / B2 soft / B3 fail)
fires from measured rates only.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_w1_5.measurements import (
    SampleTimeout, run_with_timeout,
)


# Phase B grid (spec)
GRID = {
    "min_cluster_size": [3, 5, 8, 12, 20, 30, 50],
    "min_samples":      [3, 5, 10],
    "cluster_selection_epsilon": [0.0, 0.3, 0.5, 1.0],
}


def _ground_kwargs_from_phase_a(phase_a_verdict: dict) -> dict:
    """Resolve which ground filter Phase B should use from Phase A verdict."""
    v = phase_a_verdict.get("verdict")
    best = phase_a_verdict.get("best_config")
    if v == "GROUND_FILTER_HELPS" and best:
        return dict(best["ground_filter_kwargs"])
    # GROUND_FILTER_NOT_ROOT_CAUSE / INCONCLUSIVE → W1 default
    return {"ground_filter": "z_threshold", "ground_z_max": -1.4}


def run_phase_b(cache: dict, phase_a_verdict: dict, out_dir: str) -> dict:
    phase_dir = osp.join(out_dir, "phase_b")
    per_combo_dir = osp.join(phase_dir, "per_combo")
    figures_dir = osp.join(phase_dir, "figures")
    os.makedirs(per_combo_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    ground_kwargs = _ground_kwargs_from_phase_a(phase_a_verdict)
    tokens = sorted(cache.keys())

    n_combos = (len(GRID["min_cluster_size"]) * len(GRID["min_samples"])
                * len(GRID["cluster_selection_epsilon"]))
    print(f"\n=== Phase B — {n_combos} combos × {len(tokens)} samples "
          f"(ground filter: {ground_kwargs}) ===")

    combo_summaries = []
    t_start = time.time()
    for mcs in GRID["min_cluster_size"]:
        for ms in GRID["min_samples"]:
            for eps in GRID["cluster_selection_epsilon"]:
                gen = LiDARProposalGenerator(
                    min_cluster_size=mcs, min_samples=ms,
                    cluster_selection_epsilon=eps, **ground_kwargs,
                )
                ns, miss, M, L, D, timing = [], [], [], [], [], []
                for tok in tokens:
                    try:
                        rec = run_with_timeout(gen, cache[tok])
                        ns.append(rec["n_clusters"])
                        miss.append(rec["miss_rate"])
                        M.append(rec["M_rate"])
                        L.append(rec["L_rate"])
                        D.append(rec["D_rate"])
                        timing.append(rec["timing"]["total"])
                    except SampleTimeout:
                        pass
                if not ns:
                    continue
                summary = {
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                    "cluster_selection_epsilon": eps,
                    "n_samples": len(ns),
                    "mean_n_clusters": float(np.mean(ns)),
                    "median_n_clusters": float(np.median(ns)),
                    "mean_miss_rate": float(np.mean(miss)),
                    "mean_M_rate": float(np.mean(M)),
                    "mean_L_rate": float(np.mean(L)),
                    "mean_D_rate": float(np.mean(D)),
                    "median_timing_s": float(np.median(timing)),
                }
                combo_summaries.append(summary)
    elapsed = time.time() - t_start
    print(f"  sweep finished in {elapsed:.1f}s ({len(combo_summaries)}/{n_combos} combos succeeded)")

    # Persist full sweep data
    sweep_record = {
        "ground_filter_kwargs": ground_kwargs,
        "grid": GRID,
        "results": combo_summaries,
    }
    with open(osp.join(out_dir, "phase_b", "parameter_sweep_extended.json"), "w") as f:
        json.dump(sweep_record, f, indent=2)

    # ---- selection ----
    filt1 = [c for c in combo_summaries
             if 5 <= c["mean_n_clusters"] <= 15 and c["mean_M_rate"] >= 0.30]
    filt2 = [c for c in combo_summaries
             if 5 <= c["mean_n_clusters"] <= 30 and c["mean_miss_rate"] < 0.30]
    if filt1:
        best = min(filt1, key=lambda c: c["mean_miss_rate"])
        verdict = "B1_PASS"
        verdict_text = ("Strict PASS — config in [5,15] band with M_rate ≥ 0.30; "
                        "lock as method config.")
    elif filt2:
        best = min(filt2, key=lambda c: c["mean_miss_rate"])
        verdict = "B2_SOFT_PASS"
        verdict_text = ("Soft PASS — config in [5,30] band with miss_rate < 0.30; "
                        "M_rate did not clear 0.30 strict bar.")
    else:
        best = min(combo_summaries, key=lambda c: c["mean_miss_rate"])
        verdict = "B3_FAIL"
        verdict_text = ("FAIL — no config met either selection band. HDBSCAN tuning alone "
                        "does not solve the over-segmentation / matching gap.")

    verdict_record = {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "ground_filter_kwargs": ground_kwargs,
        "best_config": best,
        "selection_rule": ("filt1: 5≤n≤15 and M≥0.30; "
                           "filt2: 5≤n≤30 and miss<0.30; else argmin miss."),
    }
    with open(osp.join(phase_dir, "verdict.json"), "w") as f:
        json.dump(verdict_record, f, indent=2)

    _plot_phase_b(combo_summaries, best, figures_dir)

    print(f"  → Phase B verdict: {verdict}")
    print(f"  → best: min_cluster={best['min_cluster_size']}, "
          f"min_samples={best['min_samples']}, eps={best['cluster_selection_epsilon']}, "
          f"n_cl={best['mean_n_clusters']:.2f}, miss={best['mean_miss_rate']*100:.1f}%, "
          f"M={best['mean_M_rate']*100:.1f}%")
    return verdict_record


def _plot_heatmap_panels(combos, metric_key, title, vmin, vmax, cmap, out_path):
    eps_vals = GRID["cluster_selection_epsilon"]
    mcs_vals = GRID["min_cluster_size"]
    ms_vals = GRID["min_samples"]
    fig, axes = plt.subplots(1, len(eps_vals), figsize=(3.5 * len(eps_vals), 4.0), squeeze=False)
    table = {(c["min_cluster_size"], c["min_samples"], c["cluster_selection_epsilon"]):
             c[metric_key] for c in combos}
    for i, eps in enumerate(eps_vals):
        ax = axes[0][i]
        Z = np.full((len(mcs_vals), len(ms_vals)), np.nan)
        for r, mcs in enumerate(mcs_vals):
            for cidx, ms in enumerate(ms_vals):
                if (mcs, ms, eps) in table:
                    Z[r, cidx] = table[(mcs, ms, eps)]
        im = ax.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ms_vals)))
        ax.set_xticklabels(ms_vals)
        ax.set_yticks(range(len(mcs_vals)))
        ax.set_yticklabels(mcs_vals)
        ax.set_xlabel("min_samples")
        if i == 0:
            ax.set_ylabel("min_cluster_size")
        ax.set_title(f"eps={eps}")
        for r in range(len(mcs_vals)):
            for cidx in range(len(ms_vals)):
                v = Z[r, cidx]
                if not np.isnan(v):
                    colour = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(cidx, r, f"{v:.1f}" if v >= 1 else f"{v:.2f}",
                            ha="center", va="center", color=colour, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_phase_b(combos, best, figures_dir):
    if not combos:
        return
    n_vals = [c["mean_n_clusters"] for c in combos]
    miss_vals = [c["mean_miss_rate"] for c in combos]
    M_vals = [c["mean_M_rate"] for c in combos]

    _plot_heatmap_panels(combos, "mean_n_clusters",
                         "Phase B — mean n_clusters across grid", min(n_vals), max(n_vals),
                         "viridis", osp.join(figures_dir, "n_clusters_heatmap.png"))
    _plot_heatmap_panels(combos, "mean_miss_rate",
                         "Phase B — mean miss_rate across grid", 0.0, max(miss_vals),
                         "Reds", osp.join(figures_dir, "miss_rate_heatmap.png"))
    _plot_heatmap_panels(combos, "mean_M_rate",
                         "Phase B — mean M_rate across grid", 0.0, max(0.4, max(M_vals)),
                         "Greens", osp.join(figures_dir, "M_rate_heatmap.png"))

    # Pareto front: n_clusters vs miss_rate
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n_vals, [v * 100 for v in miss_vals], s=14, alpha=0.5, color="#666666")
    ax.scatter([best["mean_n_clusters"]], [best["mean_miss_rate"] * 100], s=120,
               edgecolor="red", facecolor="none", linewidth=2, label="best")
    ax.axvspan(5, 15, color="green", alpha=0.10, label="strict band [5,15]")
    ax.axvspan(5, 30, color="green", alpha=0.05, label="soft band [5,30]")
    ax.axhline(30, color="orange", ls="--", lw=1, label="miss=30% (filt2 cutoff)")
    ax.set_xlabel("mean n_clusters per frame")
    ax.set_ylabel("mean miss_rate (%)")
    ax.set_title(f"Phase B — n_clusters vs miss_rate (n={len(combos)} combos)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, "pareto_front.png"), dpi=120)
    plt.close(fig)
