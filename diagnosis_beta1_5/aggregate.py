"""β1.5 aggregator + report writer."""

from __future__ import annotations

import glob
import json
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# Decision thresholds (β1.5 spec — note WEAK band added on top of β1's grid)
M_STRONG = 0.50
M_PARTIAL = 0.40
M_WEAK = 0.36   # β1 baseline ~0.361
N_CL_PASS = (10, 50)
N_CL_WARN = (50, 100)
TIMING_PASS = 3.0
TIMING_WARN = 3.5

# β1 best baseline (saved Step β1 results)
BETA1_M = 0.3612
BETA1_L = 0.044
BETA1_D = 0.230
BETA1_MISS = 0.364
BETA1_N_CL = 182.12
BETA1_TIMING = 0.829

# W1.5 raw baseline (for cumulative W1.5 → β1 → β1.5 chain)
W1_5_M = 0.285
W1_5_MISS = 0.305
W1_5_N_CL = 189.7


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _per_sample_iter(out_dirs, combo_id):
    pat = osp.join(out_dirs["per_sample_per_config"], combo_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            yield json.load(f)


def aggregate_beta1_5(summaries, best, sweep_record, regression, cache, out_dirs):
    best_id = best["combo_id"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}

    removed_breakdown = {"too_small": 0, "extended": 0, "too_large_compact": 0,
                         "kept": 0, "total": 0}
    removed_pt_ratios = []
    n_components_kept = []
    n_components_total = []

    for rec in _per_sample_iter(out_dirs, best_id):
        v = rec["verticality"]
        removed_breakdown["too_small"] += v.get("n_components_removed_too_small", 0)
        removed_breakdown["extended"] += v.get("n_components_removed_extended", 0)
        removed_breakdown["too_large_compact"] += v.get("n_components_removed_too_large_compact", 0)
        removed_breakdown["kept"] += v.get("n_components_kept", 0)
        removed_breakdown["total"] += v.get("n_components_total", 0)
        removed_pt_ratios.append(v.get("removed_point_ratio", 0.0))
        n_components_kept.append(v.get("n_components_kept", 0))
        n_components_total.append(v.get("n_components_total", 0))
        for g in rec["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in by_bin:
                b = "unknown"
            by_bin[b]["n_GT"] += 1
            by_bin[b][g["case"]] += 1

    bin_rates = {b: {
        "n_GT": rec["n_GT"],
        "M_rate": rec["M"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "L_rate": rec["L"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "D_rate": rec["D"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "miss_rate": rec["miss"] / rec["n_GT"] if rec["n_GT"] else 0.0,
    } for b, rec in by_bin.items()}

    # Hypotheses (measurement-based)
    H4_verdict = ("CONFIRMED" if best["mean_n_clusters"] <= 50
                  else "PARTIAL" if best["mean_n_clusters"] <= 100
                  else "REJECTED")
    H5_verdict = ("CONFIRMED" if (H4_verdict == "CONFIRMED" and best["mean_M_rate"] >= 0.50)
                  else "PARTIAL" if best["mean_M_rate"] >= 0.40
                  else "REJECTED")
    far_M = (bin_rates["30-50m"]["M_rate"] * by_bin["30-50m"]["n_GT"]
             + bin_rates["50m+"]["M_rate"] * by_bin["50m+"]["n_GT"])
    far_n = by_bin["30-50m"]["n_GT"] + by_bin["50m+"]["n_GT"]
    far_M_rate = (far_M / far_n) if far_n else 0.0
    H6_verdict = ("CONFIRMED" if far_M_rate >= 0.30
                  else "PARTIAL" if far_M_rate >= 0.20
                  else "REJECTED")

    return {
        "n_samples": len(cache),
        "n_combos": len(summaries),
        "selection_verdict": sweep_record["selection_verdict"],
        "regression": regression,
        "best": best,
        "best_per_sample_distance": {"by_bin": by_bin, "by_bin_rates": bin_rates},
        "best_components": {
            "breakdown": removed_breakdown,
            "mean_n_components_total": _safe_mean(n_components_total),
            "mean_n_components_kept": _safe_mean(n_components_kept),
            "mean_removed_point_ratio": _safe_mean(removed_pt_ratios),
        },
        "hypotheses": {
            "H4_n_clusters_under_50": {"verdict": H4_verdict,
                                         "mean_n_clusters": best["mean_n_clusters"]},
            "H5_M_rate_above_50": {"verdict": H5_verdict,
                                    "best_M_rate": best["mean_M_rate"]},
            "H6_far_M_above_30": {"verdict": H6_verdict,
                                   "far_M_rate": far_M_rate},
        },
        "comparison": {
            "M_rate":     {"w1_5": W1_5_M, "beta1": BETA1_M, "beta1_5": best["mean_M_rate"]},
            "L_rate":     {"w1_5": 0.054, "beta1": BETA1_L, "beta1_5": best["mean_L_rate"]},
            "D_rate":     {"w1_5": 0.356, "beta1": BETA1_D, "beta1_5": best["mean_D_rate"]},
            "miss_rate":  {"w1_5": W1_5_MISS, "beta1": BETA1_MISS, "beta1_5": best["mean_miss_rate"]},
            "n_clusters": {"w1_5": W1_5_N_CL, "beta1": BETA1_N_CL, "beta1_5": best["mean_n_clusters"]},
            "timing":     {"w1_5": 1.480, "beta1": BETA1_TIMING, "beta1_5": best["median_timing_total"]},
        },
    }


# ---------- decision ----------

def _evaluate_decision(agg):
    best = agg["best"]
    M = best["mean_M_rate"]
    if M >= M_STRONG:
        m_v = "STRONG"
    elif M >= M_PARTIAL:
        m_v = "PARTIAL"
    elif M >= M_WEAK:
        m_v = "WEAK"
    else:
        m_v = "FAIL"

    n = best["mean_n_clusters"]
    if N_CL_PASS[0] <= n <= N_CL_PASS[1]:
        n_v = "PASS"
    elif N_CL_WARN[0] < n <= N_CL_WARN[1]:
        n_v = "WARN"
    else:
        n_v = "FAIL"

    t = best["median_timing_total"]
    if t < TIMING_PASS:
        t_v = "PASS"
    elif t < TIMING_WARN:
        t_v = "WARN"
    else:
        t_v = "FAIL"

    if m_v == "STRONG" and n_v == "PASS":
        branch = ("β1.5 STRONG. Verticality 가설 confirmed. "
                  "W2 (Coverage reliability) 진행 가능. config locked.")
    elif m_v == "PARTIAL":
        branch = ("β1.5 PARTIAL. 의미 있는 개선이지만 50% 미달. "
                  "W2 진행 가능 (baseline으로) 또는 β2 escalate 결정.")
    elif m_v == "WEAK":
        branch = ("β1.5 WEAK. β1 대비 큰 차이 없음. Geometry-only 한계. "
                  "β2 (PointNet++) 또는 γ (CenterPoint) escalate.")
    elif m_v == "FAIL":
        branch = ("β1.5 FAIL. 알고리즘 한계 확정. "
                  "β2 또는 γ 즉시 escalate.")
    else:
        branch = "unknown verdict combination — review tables"

    return {
        "cond_M_rate": {"value": M, "verdict": m_v,
                         "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL,
                                         "WEAK": M_WEAK}},
        "cond_n_clusters": {"value": n, "verdict": n_v,
                             "thresholds": {"PASS": N_CL_PASS, "WARN": N_CL_WARN}},
        "cond_timing": {"value": t, "verdict": t_v,
                         "thresholds": {"PASS": TIMING_PASS, "WARN": TIMING_WARN}},
        "branch": branch,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_M_rate_heatmap(summaries, out):
    """3 panels (size_max), each panel = size_min × aspect_max heatmap of M_rate."""
    size_mins = sorted({s["combo"]["size_min"] for s in summaries})
    size_maxs = sorted({s["combo"]["size_max"] for s in summaries})
    aspects = sorted({s["combo"]["aspect_max"] for s in summaries})
    table = {(c["combo"]["size_min"], c["combo"]["size_max"],
              c["combo"]["aspect_max"]): c["mean_M_rate"] for c in summaries}
    vmin = min(s["mean_M_rate"] for s in summaries) * 100
    vmax = max(s["mean_M_rate"] for s in summaries) * 100
    fig, axes = plt.subplots(1, len(size_maxs), figsize=(4.5 * len(size_maxs), 4),
                              squeeze=False)
    for i, smax in enumerate(size_maxs):
        ax = axes[0][i]
        Z = np.full((len(size_mins), len(aspects)), np.nan)
        for r, smin in enumerate(size_mins):
            for c, asp in enumerate(aspects):
                if (smin, smax, asp) in table:
                    Z[r, c] = table[(smin, smax, asp)] * 100
        im = ax.imshow(Z, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(aspects)))
        ax.set_xticklabels([f"{a:g}" for a in aspects])
        ax.set_yticks(range(len(size_mins)))
        ax.set_yticklabels(size_mins)
        ax.set_xlabel("aspect_max")
        if i == 0:
            ax.set_ylabel("size_min")
        ax.set_title(f"size_max={smax}")
        for r in range(len(size_mins)):
            for c in range(len(aspects)):
                v = Z[r, c]
                if not np.isnan(v):
                    colour = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            color=colour, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"β1.5 — mean M_rate (%) across 27 combos")
    _save(fig, out)


def fig_n_clusters_reduction(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    n_vals = [s["mean_n_clusters"] for s in summaries]
    M_vals = [s["mean_M_rate"] * 100 for s in summaries]
    ax.scatter(n_vals, M_vals, s=24, alpha=0.6, color="#666666", label="combos")
    ax.scatter([best["mean_n_clusters"]], [best["mean_M_rate"] * 100],
               s=160, edgecolor="red", facecolor="none", linewidth=2, label="best")
    ax.scatter([BETA1_N_CL], [BETA1_M * 100], s=140, marker="x",
               color="orange", label=f"β1 baseline ({BETA1_M*100:.1f}% / {BETA1_N_CL:.0f})")
    ax.scatter([W1_5_N_CL], [W1_5_M * 100], s=140, marker="x",
               color="blue", label=f"W1.5 baseline ({W1_5_M*100:.1f}% / {W1_5_N_CL:.0f})")
    ax.axvspan(N_CL_PASS[0], N_CL_PASS[1], color="green", alpha=0.10,
               label=f"n_cl PASS [{N_CL_PASS[0]}, {N_CL_PASS[1]}]")
    ax.axhline(M_STRONG * 100, color="green", ls="--", lw=1, label=f"M STRONG ≥{M_STRONG*100:.0f}%")
    ax.axhline(M_PARTIAL * 100, color="orange", ls="--", lw=1, label=f"M PARTIAL ≥{M_PARTIAL*100:.0f}%")
    ax.axhline(M_WEAK * 100, color="red", ls=":", lw=1, label=f"M WEAK ≥{M_WEAK*100:.0f}% (β1)")
    ax.set_xlabel("mean n_clusters")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title(f"β1.5 sweep ({len(summaries)} combos) vs β1/W1.5 baselines")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_component_breakdown(agg, out):
    rb = agg["best_components"]["breakdown"]
    cats = ["kept", "too_small", "extended", "too_large_compact"]
    vals = [rb[c] for c in cats]
    colors = ["#117733", "#999999", "#882255", "#CC6677"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, vals, color=colors)
    total = sum(vals)
    for b, v in zip(bars, vals):
        if total:
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.01,
                    f"{v}\n({v/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# components (summed across 50 samples)")
    ax.set_title(f"β1.5 best component classification (total {total} components)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_distance_stratified_best(by_bin_rates, out):
    bins = DISTANCE_BIN_LABELS
    M = [by_bin_rates[b]["M_rate"] * 100 for b in bins]
    miss = [by_bin_rates[b]["miss_rate"] * 100 for b in bins]
    n_GT = [by_bin_rates[b]["n_GT"] for b in bins]
    x = np.arange(len(bins))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, M, width, color="#117733", label="M_rate (%)")
    ax.bar(x + width / 2, miss, width, color="#882255", label="miss_rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(bins, n_GT)])
    ax.set_ylabel("rate (%)")
    ax.set_title("β1.5 best — distance-stratified")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_before_after_chain(comparison, out):
    metrics = ["M_rate", "miss_rate", "L_rate", "D_rate"]
    stages = ["W1.5", "β1", "β1.5"]
    keys = ["w1_5", "beta1", "beta1_5"]
    colors = ["#88CCEE", "#DDCC77", "#117733"]
    x = np.arange(len(metrics))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (st, k, col) in enumerate(zip(stages, keys, colors)):
        vals = [comparison[m][k] * 100 for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width, color=col, label=st)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("rate (%)")
    ax.set_title("W1.5 → β1 → β1.5 progression")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_distribution(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [s["median_timing_total"] for s in summaries]
    ax.hist(ts, bins=12, color="#CC6677", edgecolor="white")
    ax.axvline(best["median_timing_total"], color="red", ls="--",
               label=f"best={best['median_timing_total']:.2f}s")
    ax.axvline(TIMING_PASS, color="green", ls=":", label=f"PASS<{TIMING_PASS}s")
    ax.axvline(TIMING_WARN, color="orange", ls=":", label=f"WARN<{TIMING_WARN}s")
    ax.set_xlabel("median timing (s)")
    ax.set_ylabel("# combos")
    ax.set_title(f"β1.5 sweep timing ({len(ts)} combos)")
    ax.legend(fontsize=8)
    _save(fig, out)


# ---------- top-3 ----------

def _top3(agg):
    cmp = agg["comparison"]
    obs = []
    obs.append(
        f"M_rate W1.5 raw {cmp['M_rate']['w1_5']*100:.1f}% → "
        f"β1 {cmp['M_rate']['beta1']*100:.1f}% → "
        f"**β1.5 {cmp['M_rate']['beta1_5']*100:.1f}%**. "
        f"Verticality marginal gain over β1 = "
        f"{(cmp['M_rate']['beta1_5'] - cmp['M_rate']['beta1'])*100:+.2f} pts."
    )
    obs.append(
        f"n_clusters W1.5 {cmp['n_clusters']['w1_5']:.0f} → "
        f"β1 {cmp['n_clusters']['beta1']:.0f} → "
        f"**β1.5 {cmp['n_clusters']['beta1_5']:.0f}**. "
        f"Components classification: "
        f"{agg['best_components']['breakdown']['kept']} kept / "
        f"{agg['best_components']['breakdown']['too_small']} too_small / "
        f"{agg['best_components']['breakdown']['extended']} extended / "
        f"{agg['best_components']['breakdown']['too_large_compact']} too_large_compact."
    )
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    near = (rates["0-10m"]["M_rate"] + rates["10-20m"]["M_rate"]) / 2 * 100
    far = (rates["30-50m"]["M_rate"] + rates["50m+"]["M_rate"]) / 2 * 100
    obs.append(
        f"Distance breakdown of M_rate (β1.5 best): "
        f"0–20m mean {near:.1f}%, 30m+ mean {far:.1f}%. "
        + ("Near range still leads — verticality affects dense scenes more." if near > far + 5
           else "Far range now competitive — verticality recovered some sparse-far GTs." if far > near + 5
           else "Similar near/far M_rate.")
    )
    return obs


# ---------- report ----------

def _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, out_path):
    L = []
    L.append("# β1.5 — Verticality filter (BEV connected component + aspect ratio)")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}** (β1 50-sample set)")
    L.append(f"- Sweep: **{agg['n_combos']}** combos (3 size_min × 3 size_max × 3 aspect_max)")
    L.append(f"- Selection verdict: **{agg['selection_verdict']}**")
    L.append(f"- β1 regression on cached foreground: M_rate={regression['mean_M_rate']:.4f} "
             f"(expected 0.3612, ΔM={regression['delta_M']:.6f}), "
             f"n_cl={regression['mean_n_clusters']:.4f} (expected 182.12, "
             f"Δn={regression['delta_n_clusters']:.6f}) → "
             f"**{'PASS' if regression['passes'] else 'FAIL'}**")
    L.append(f"- Locked configs: pillar `{sweep_record['fixed_pillar_config']}`, "
             f"HDBSCAN `{sweep_record['fixed_hdbscan_config']}`")
    L.append("")

    L.append("## Hypothesis check")
    L.append("")
    h = agg["hypotheses"]
    L.append(f"- **H4** — verticality filter brings n_clusters ≤ 50: "
             f"**{h['H4_n_clusters_under_50']['verdict']}** "
             f"(mean n_clusters = {h['H4_n_clusters_under_50']['mean_n_clusters']:.1f})")
    L.append(f"- **H5** — H4 holds → M_rate ≥ 0.50: "
             f"**{h['H5_M_rate_above_50']['verdict']}** "
             f"(best M_rate = {h['H5_M_rate_above_50']['best_M_rate']*100:.1f}%)")
    L.append(f"- **H6** — 30m+ M_rate ≥ 0.30: "
             f"**{h['H6_far_M_above_30']['verdict']}** "
             f"(far M_rate = {h['H6_far_M_above_30']['far_M_rate']*100:.1f}%)")
    L.append("")

    L.append("## Sweep — 27 combos")
    L.append("")
    L.append("![M_rate_heatmap](figures/M_rate_heatmap_size_aspect.png)")
    L.append("")
    L.append("Three side-by-side heatmaps (one per size_max), with size_min × aspect_max axes. Bright = higher M_rate.")
    L.append("")
    L.append("![n_clusters_reduction](figures/n_clusters_reduction.png)")
    L.append("")
    L.append("Each combo plotted in (n_clusters, M_rate) space alongside W1.5 raw and β1 baselines. The best combo is circled red; the green band is the n_clusters PASS gate.")
    L.append("")

    # Sweep table
    L.append("### Sweep results (sorted by M_rate)")
    L.append("")
    L.append("| size_min | size_max | aspect_max | M% | miss% | n_cl | removed_pts% | timing s | n_ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: -x["mean_M_rate"]):
        c = s["combo"]
        L.append(f"| {c['size_min']} | {c['size_max']} | {c['aspect_max']:g} | "
                 f"{s['mean_M_rate']*100:.1f} | {s['mean_miss_rate']*100:.1f} | "
                 f"{s['mean_n_clusters']:.1f} | "
                 f"{s['mean_removed_point_ratio']*100:.1f} | "
                 f"{s['median_timing_total']:.2f} | "
                 f"{s['n_samples_succeeded']}/{agg['n_samples']} |")
    L.append("")

    # Best
    bc = best["combo"]
    L.append("## Best combo")
    L.append("")
    L.append(f"- size_min: **{bc['size_min']}** pillars")
    L.append(f"- size_max: **{bc['size_max']}** pillars")
    L.append(f"- aspect_max: **{bc['aspect_max']:g}**")
    L.append("")
    L.append("![component_breakdown](figures/component_removal_breakdown.png)")
    L.append("")
    rb = agg["best_components"]["breakdown"]
    L.append(f"Component classification across the 50-sample run: "
             f"{rb['kept']} kept, {rb['too_small']} too_small, "
             f"{rb['extended']} extended, {rb['too_large_compact']} too_large_compact "
             f"(total {rb['total']}).")
    L.append("")
    L.append(f"- mean # components per sample: total {agg['best_components']['mean_n_components_total']:.1f}, "
             f"kept {agg['best_components']['mean_n_components_kept']:.1f}")
    L.append(f"- mean removed-point fraction: "
             f"{agg['best_components']['mean_removed_point_ratio']*100:.1f}%")
    L.append("")

    # Comparison
    L.append("## W1.5 → β1 → β1.5 progression")
    L.append("")
    cmp = agg["comparison"]
    L.append("| metric | W1.5 raw | β1 best | β1.5 best | Δ vs β1 |")
    L.append("|---|---|---|---|---|")
    for m in ["M_rate", "L_rate", "D_rate", "miss_rate"]:
        L.append(f"| {m} | {cmp[m]['w1_5']*100:.1f}% | {cmp[m]['beta1']*100:.1f}% | "
                 f"{cmp[m]['beta1_5']*100:.1f}% | "
                 f"{(cmp[m]['beta1_5'] - cmp[m]['beta1'])*100:+.1f} |")
    L.append(f"| n_clusters | {cmp['n_clusters']['w1_5']:.1f} | "
             f"{cmp['n_clusters']['beta1']:.1f} | "
             f"{cmp['n_clusters']['beta1_5']:.1f} | "
             f"{cmp['n_clusters']['beta1_5'] - cmp['n_clusters']['beta1']:+.1f} |")
    L.append(f"| timing (median s) | {cmp['timing']['w1_5']:.3f} | "
             f"{cmp['timing']['beta1']:.3f} | "
             f"{cmp['timing']['beta1_5']:.3f} | "
             f"{cmp['timing']['beta1_5'] - cmp['timing']['beta1']:+.3f} |")
    L.append("")
    L.append("![before_after](figures/before_after_β1_β1.5.png)")
    L.append("")
    L.append("Three-stage progression on the M/L/D/miss axes.")
    L.append("")

    # Distance
    L.append("## Distance-stratified (best combo)")
    L.append("")
    L.append("![distance_stratified](figures/distance_stratified_best.png)")
    L.append("")
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    L.append("| bin | n_GT | M% | L% | D% | miss% |")
    L.append("|---|---|---|---|---|---|")
    for b in DISTANCE_BIN_LABELS:
        r = rates[b]
        if r["n_GT"] == 0:
            continue
        L.append(f"| {b} | {r['n_GT']} | {r['M_rate']*100:.1f} | "
                 f"{r['L_rate']*100:.1f} | {r['D_rate']*100:.1f} | "
                 f"{r['miss_rate']*100:.1f} |")
    L.append("")
    L.append("![timing](figures/timing_distribution.png)")
    L.append("")

    # Decision
    L.append("## Decision")
    L.append("")
    a = decision["cond_M_rate"]
    b = decision["cond_n_clusters"]
    c = decision["cond_timing"]
    L.append(f"- **cond_M_rate** = {a['value']*100:.2f}% → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"PARTIAL ≥ {a['thresholds']['PARTIAL']*100:.0f}%, "
             f"WEAK ≥ {a['thresholds']['WEAK']*100:.0f}%)")
    L.append(f"- **cond_n_clusters** = {b['value']:.2f} → **{b['verdict']}** "
             f"(PASS in {b['thresholds']['PASS']}, WARN in {b['thresholds']['WARN']})")
    L.append(f"- **cond_timing** = {c['value']:.3f}s → **{c['verdict']}** "
             f"(PASS < {c['thresholds']['PASS']}s, WARN < {c['thresholds']['WARN']}s)")
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_M_rate     = {a['value']:.4f}   → {a['verdict']}")
    L.append(f"cond_n_clusters = {b['value']:.4f}   → {b['verdict']}")
    L.append(f"cond_timing     = {c['value']:.4f}s  → {c['verdict']}")
    L.append(f"→ 분기: {decision['branch']}")
    L.append("```")
    L.append("")

    # Top-3
    L.append("## Top-3 observations")
    L.append("")
    for i, o in enumerate(observations, 1):
        L.append(f"{i}. {o}")
        L.append("")

    # Next-step
    L.append("## Next-step candidates (manual decision required)")
    L.append("")
    L.append("- **STRONG**: lock pillar+verticality config, advance to W2.")
    L.append("- **PARTIAL**: proceed to W2 with caveat, or escalate to β2 (PointNet++).")
    L.append("- **WEAK**: geometry-only family is likely capped — escalate to β2 / γ.")
    L.append("- **FAIL**: HDBSCAN-as-proposal-stage limit confirmed — γ (CenterPoint) is the responsible call.")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_beta1_5(summaries, best, sweep_record, regression, agg,
                       figures_dir, report_path):
    fig_M_rate_heatmap(summaries, osp.join(figures_dir, "M_rate_heatmap_size_aspect.png"))
    fig_n_clusters_reduction(summaries, best, osp.join(figures_dir, "n_clusters_reduction.png"))
    fig_component_breakdown(agg, osp.join(figures_dir, "component_removal_breakdown.png"))
    fig_distance_stratified_best(agg["best_per_sample_distance"]["by_bin_rates"],
                                  osp.join(figures_dir, "distance_stratified_best.png"))
    fig_before_after_chain(agg["comparison"],
                            osp.join(figures_dir, "before_after_β1_β1.5.png"))
    fig_timing_distribution(summaries, best,
                             osp.join(figures_dir, "timing_distribution.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg)
    _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, report_path)
