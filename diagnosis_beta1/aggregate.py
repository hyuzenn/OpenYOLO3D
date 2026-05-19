"""β1 aggregator + report writer.

Builds: aggregate.json, six figures, report.md with the 3-cond Decision
section. The Decision section fires from measured `best` numbers only —
it does not look at hypothesis names.
"""

from __future__ import annotations

import glob
import json
import os.path as osp
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# Decision thresholds
M_STRONG = 0.40
M_PARTIAL = 0.30
N_CL_PASS = (10, 50)
N_CL_WARN = (50, 100)
TIMING_PASS = 2.5
TIMING_WARN = 3.5

# W1.5 baseline (locked numbers)
W1_5_M_RATE = 0.285
W1_5_MISS_RATE = 0.305
W1_5_L_RATE = 0.054
W1_5_D_RATE = 0.356
W1_5_N_CL = 189.70
W1_5_TIMING = 1.480


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


# ---------- aggregate ----------

def _per_sample_iter_for_combo(out_dirs, combo_id):
    pat = osp.join(out_dirs["per_sample_per_config"], combo_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            yield json.load(f)


def aggregate_beta1(summaries, best, sweep_record, regression, cache, out_dirs):
    # Load best combo's per-sample data for distance stratification
    best_id = best["combo_id"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
    fg_ratios = []
    for rec in _per_sample_iter_for_combo(out_dirs, best_id):
        fg_ratios.append(rec["foreground"]["ratio"])
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

    # Hypothesis verdicts (measurement-based)
    delta_n_cl = best["mean_n_clusters"] - W1_5_N_CL
    H1_verdict = ("CONFIRMED" if abs(delta_n_cl) > 100 and best["mean_n_clusters"] < W1_5_N_CL
                  else "REJECTED" if abs(delta_n_cl) < 30
                  else "PARTIAL")
    fg_ratio_mean = float(np.mean(fg_ratios)) if fg_ratios else 0.0
    H2_verdict = ("CONFIRMED" if (delta_n_cl < -50 and 0.05 <= fg_ratio_mean <= 0.40)
                  else "PARTIAL" if (delta_n_cl < -20)
                  else "REJECTED")
    H3_verdict = ("CONFIRMED" if best["mean_M_rate"] >= 0.40
                  else "PARTIAL" if best["mean_M_rate"] >= 0.30
                  else "REJECTED")

    return {
        "n_samples": len(cache),
        "n_combos": len(summaries),
        "selection_verdict": sweep_record["selection_verdict"],
        "regression": regression,
        "best": best,
        "best_per_sample_distance": {
            "by_bin": by_bin,
            "by_bin_rates": bin_rates,
        },
        "best_foreground_ratio_mean": fg_ratio_mean,
        "hypotheses": {
            "H1_road_cluster_explosion": {"verdict": H1_verdict,
                                           "delta_n_cl_vs_w1_5": delta_n_cl},
            "H2_pillar_solves_h1": {"verdict": H2_verdict,
                                     "fg_ratio_mean": fg_ratio_mean},
            "H3_M_rate_above_40": {"verdict": H3_verdict,
                                    "best_M_rate": best["mean_M_rate"]},
        },
        "comparison_to_w1_5": {
            "M_rate":     {"w1_5": W1_5_M_RATE, "beta1": best["mean_M_rate"],
                            "delta": best["mean_M_rate"] - W1_5_M_RATE},
            "miss_rate":  {"w1_5": W1_5_MISS_RATE, "beta1": best["mean_miss_rate"],
                            "delta": best["mean_miss_rate"] - W1_5_MISS_RATE},
            "L_rate":     {"w1_5": W1_5_L_RATE, "beta1": best["mean_L_rate"],
                            "delta": best["mean_L_rate"] - W1_5_L_RATE},
            "D_rate":     {"w1_5": W1_5_D_RATE, "beta1": best["mean_D_rate"],
                            "delta": best["mean_D_rate"] - W1_5_D_RATE},
            "n_clusters": {"w1_5": W1_5_N_CL, "beta1": best["mean_n_clusters"],
                            "delta": best["mean_n_clusters"] - W1_5_N_CL},
            "timing":     {"w1_5": W1_5_TIMING, "beta1": best["median_timing_total"],
                            "delta": best["median_timing_total"] - W1_5_TIMING},
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

    if m_v == "STRONG" and n_v == "PASS" and t_v == "PASS":
        branch = ("β1 STRONG. Foreground hypothesis confirmed. "
                  "pillar config locked. W2 (Coverage reliability) 진행 가능.")
    elif m_v == "PARTIAL":
        branch = ("β1 PARTIAL. 입력 분포 가설은 부분 확정. "
                  "verticality filter 추가 (β1.5) 또는 β2 (PointNet++ pretrained) 결정 필요.")
    elif m_v == "FAIL":
        branch = ("β1 FAIL. 입력 분포만으론 부족. "
                  "β2 또는 다른 알고리즘 (γ CenterPoint, δ 학습된 unsupervised) 검토.")
    else:
        # M STRONG but n_v or t_v not PASS
        branch = (f"β1 mixed (M_rate STRONG but n_clusters {n_v} / timing {t_v}). "
                  "config tweak 또는 본인 검토 후 결정.")

    return {
        "cond_M_rate": {"value": M, "verdict": m_v,
                         "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL}},
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
    """6 sub-panels: pillar_size × ground_estimation, axis = z_threshold"""
    pillar_sizes = sorted({tuple(s["combo"]["pillar_size_xy"]) for s in summaries})
    z_thrs = sorted({s["combo"]["z_threshold"] for s in summaries})
    ges = sorted({s["combo"]["ground_estimation"] for s in summaries})
    table = {(tuple(s["combo"]["pillar_size_xy"]),
              s["combo"]["z_threshold"],
              s["combo"]["ground_estimation"]): s["mean_M_rate"] for s in summaries}
    fig, axes = plt.subplots(len(ges), 1, figsize=(8, 4.5 * len(ges)), squeeze=False)
    vmin = min(s["mean_M_rate"] for s in summaries)
    vmax = max(s["mean_M_rate"] for s in summaries)
    for i, ge in enumerate(ges):
        ax = axes[i][0]
        Z = np.full((len(pillar_sizes), len(z_thrs)), np.nan)
        for r, ps in enumerate(pillar_sizes):
            for c, zt in enumerate(z_thrs):
                if (ps, zt, ge) in table:
                    Z[r, c] = table[(ps, zt, ge)] * 100
        im = ax.imshow(Z, cmap="viridis", vmin=vmin * 100, vmax=vmax * 100, aspect="auto")
        ax.set_xticks(range(len(z_thrs)))
        ax.set_xticklabels([f"{z:g}" for z in z_thrs])
        ax.set_yticks(range(len(pillar_sizes)))
        ax.set_yticklabels([f"{ps[0]:g}×{ps[1]:g}" for ps in pillar_sizes])
        ax.set_xlabel("z_threshold (m)")
        ax.set_ylabel("pillar_size (m)")
        ax.set_title(f"ground={ge}: mean_M_rate (%)")
        for r in range(len(pillar_sizes)):
            for c in range(len(z_thrs)):
                v = Z[r, c]
                if not np.isnan(v):
                    colour = "white" if v > (vmin + vmax) * 50 else "black"
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            color=colour, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out)


def fig_pareto(summaries, best, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    n_vals = [s["mean_n_clusters"] for s in summaries]
    m_vals = [s["mean_M_rate"] * 100 for s in summaries]
    ax.scatter(n_vals, m_vals, s=20, alpha=0.6, color="#666666", label="combos")
    ax.scatter([best["mean_n_clusters"]], [best["mean_M_rate"] * 100],
               s=140, edgecolor="red", facecolor="none", linewidth=2, label="best")
    ax.axvspan(N_CL_PASS[0], N_CL_PASS[1], color="green", alpha=0.10,
               label=f"n_cl PASS [{N_CL_PASS[0]}, {N_CL_PASS[1]}]")
    ax.axhline(M_STRONG * 100, color="green", ls="--", lw=1, label=f"M STRONG ≥{M_STRONG*100:.0f}%")
    ax.axhline(M_PARTIAL * 100, color="orange", ls="--", lw=1, label=f"M PARTIAL ≥{M_PARTIAL*100:.0f}%")
    # also annotate W1.5 baseline
    ax.scatter([W1_5_N_CL], [W1_5_M_RATE * 100], s=140, marker="x", color="blue",
               label=f"W1.5 baseline (n={W1_5_N_CL:.1f}, M={W1_5_M_RATE*100:.1f}%)")
    ax.set_xlabel("mean n_clusters")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title(f"β1 Pareto: n_clusters vs M_rate ({len(summaries)} combos)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_foreground_ratio_distribution(out_dirs, best_id, out):
    ratios = []
    pat = osp.join(out_dirs["per_sample_per_config"], best_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            r = json.load(f)
        ratios.append(r["foreground"]["ratio"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if ratios:
        ax.hist(ratios, bins=20, color="#117733", range=(0, 1), edgecolor="white")
    ax.set_xlabel("foreground_ratio (post-pillar / total)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Best combo foreground retention (n={len(ratios)})")
    _save(fig, out)


def fig_distance_stratified_best(by_bin_rates, out):
    bins = DISTANCE_BIN_LABELS
    M = [by_bin_rates[b]["M_rate"] * 100 for b in bins]
    miss = [by_bin_rates[b]["miss_rate"] * 100 for b in bins]
    n_GT = [by_bin_rates[b]["n_GT"] for b in bins]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    ax.bar(x - width / 2, M, width, color="#117733", label="M_rate (%)")
    ax.bar(x + width / 2, miss, width, color="#882255", label="miss_rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(bins, n_GT)])
    ax.set_ylabel("rate (%)")
    ax.set_title("β1 best — distance-stratified M_rate vs miss_rate")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_before_after(comparison, out):
    metrics = ["M_rate", "miss_rate", "L_rate", "D_rate"]
    w1_5 = [comparison[m]["w1_5"] * 100 for m in metrics]
    beta = [comparison[m]["beta1"] * 100 for m in metrics]
    x = np.arange(len(metrics))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, w1_5, width, color="#88CCEE", label="W1.5 best (raw)")
    ax.bar(x + width / 2, beta, width, color="#117733", label="β1 best (foreground)")
    for i, m in enumerate(metrics):
        d = comparison[m]["delta"] * 100
        col = "green" if (m != "miss_rate" and d > 0) or (m == "miss_rate" and d < 0) else "red"
        ax.text(i, max(w1_5[i], beta[i]) + 1, f"Δ={d:+.1f}",
                ha="center", color=col, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("rate (%)")
    ax.set_title("Before/after — W1.5 raw HDBSCAN vs β1 foreground+HDBSCAN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_distribution(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [s["median_timing_total"] for s in summaries]
    ax.hist(ts, bins=15, color="#CC6677", edgecolor="white")
    ax.axvline(best["median_timing_total"], color="red", ls="--",
               label=f"best={best['median_timing_total']:.2f}s")
    ax.axvline(TIMING_PASS, color="green", ls=":",
               label=f"PASS<{TIMING_PASS}s")
    ax.axvline(TIMING_WARN, color="orange", ls=":",
               label=f"WARN<{TIMING_WARN}s")
    ax.set_xlabel("median timing per sample (s)")
    ax.set_ylabel("# combos")
    ax.set_title(f"β1 sweep timing distribution ({len(ts)} combos)")
    ax.legend(fontsize=8)
    _save(fig, out)


# ---------- top-3 ----------

def _top3(agg, decision):
    cmp = agg["comparison_to_w1_5"]
    obs = []
    obs.append(
        f"M_rate moved from W1.5 baseline {cmp['M_rate']['w1_5']*100:.1f}% to "
        f"**{cmp['M_rate']['beta1']*100:.1f}%** (Δ={cmp['M_rate']['delta']*100:+.1f}). "
        f"miss_rate {cmp['miss_rate']['w1_5']*100:.1f}% → "
        f"{cmp['miss_rate']['beta1']*100:.1f}% "
        f"(Δ={cmp['miss_rate']['delta']*100:+.1f})."
    )
    obs.append(
        f"n_clusters dropped from {cmp['n_clusters']['w1_5']:.1f} to "
        f"**{cmp['n_clusters']['beta1']:.1f}** (Δ={cmp['n_clusters']['delta']:+.1f}). "
        f"Foreground retention = {agg['best_foreground_ratio_mean']*100:.1f}% — "
        f"the rest of the cloud (mostly road/vegetation) was filtered before "
        f"clustering."
    )
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    near = (rates["0-10m"]["M_rate"] + rates["10-20m"]["M_rate"]) / 2 * 100
    far = (rates["30-50m"]["M_rate"] + rates["50m+"]["M_rate"]) / 2 * 100
    obs.append(
        f"Distance breakdown of M_rate: 0–20m mean {near:.1f}%, 30m+ mean {far:.1f}%. "
        + ("Near range improved more — foreground extraction helps where GTs cluster densely." if near > far + 5
           else "Far range improved more — pillars also help recover sparse distant objects." if far > near + 5
           else "Similar improvement near and far.")
    )
    return obs


# ---------- report ----------

def _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, out_path):
    L = []
    L.append("# β1 — Pillar foreground extraction + HDBSCAN re-measurement")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}** (W1.5 50-sample set)")
    L.append(f"- Sweep: **{agg['n_combos']}** combos (3 pillar × 4 z_threshold × 2 ground)")
    L.append(f"- Selection verdict: **{agg['selection_verdict']}**")
    L.append(f"- HDBSCAN regression on raw PC: mean_n_clusters={regression['mean_n_clusters']:.4f} "
             f"(expected 189.7000, Δ={regression['delta']:.6f}) — "
             f"**{'PASS' if regression['passes'] else 'FAIL'}**")
    L.append("")

    # Hypothesis check
    L.append("## Hypothesis check")
    L.append("")
    h = agg["hypotheses"]
    L.append(f"- **H1** — road/vegetation = root cause of cluster explosion: "
             f"**{h['H1_road_cluster_explosion']['verdict']}** "
             f"(Δn_clusters vs W1.5 raw = "
             f"{h['H1_road_cluster_explosion']['delta_n_cl_vs_w1_5']:+.1f})")
    L.append(f"- **H2** — pillar foreground extraction solves H1: "
             f"**{h['H2_pillar_solves_h1']['verdict']}** "
             f"(foreground_ratio mean = "
             f"{h['H2_pillar_solves_h1']['fg_ratio_mean']*100:.1f}%)")
    L.append(f"- **H3** — foreground + W1.5 best HDBSCAN → M ≥ 0.40: "
             f"**{h['H3_M_rate_above_40']['verdict']}** "
             f"(best M_rate = "
             f"{h['H3_M_rate_above_40']['best_M_rate']*100:.1f}%)")
    L.append("")

    # Sweep
    L.append("## Sweep — 24 combos")
    L.append("")
    L.append("![M_rate_heatmap](figures/M_rate_heatmap_pillar_size_z_threshold.png)")
    L.append("")
    L.append("Mean M_rate per combo, separated into top/bottom panels by ground_estimation. Darker = lower M_rate. Locked HDBSCAN config (W1.5 best) means colour variation comes from foreground extraction quality alone.")
    L.append("")
    L.append("![pareto](figures/n_clusters_vs_M_rate_pareto.png)")
    L.append("")
    L.append("Each combo plotted in (n_clusters, M_rate) space. Best combo circled red; W1.5 raw baseline marked × for reference. Green band is the n_clusters PASS gate; horizontal lines mark M STRONG/PARTIAL bars.")
    L.append("")

    # Sweep summary table
    L.append("### Sweep results table")
    L.append("")
    L.append("| pillar | z_thr | ground | M% | miss% | n_cl | fg% | timing s | n_ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: -x["mean_M_rate"]):
        c = s["combo"]
        L.append(f"| {c['pillar_size_xy'][0]:g}x{c['pillar_size_xy'][1]:g} | "
                 f"{c['z_threshold']:g} | {c['ground_estimation']} | "
                 f"{s['mean_M_rate']*100:.1f} | {s['mean_miss_rate']*100:.1f} | "
                 f"{s['mean_n_clusters']:.1f} | "
                 f"{s['mean_foreground_ratio']*100:.1f} | "
                 f"{s['median_timing_total']:.2f} | "
                 f"{s['n_samples_succeeded']}/{agg['n_samples']} |")
    L.append("")

    # Best combo detail
    bc = best["combo"]
    L.append("## Best combo")
    L.append("")
    L.append(f"- pillar_size_xy: **({bc['pillar_size_xy'][0]:g}, {bc['pillar_size_xy'][1]:g})** m")
    L.append(f"- z_threshold: **{bc['z_threshold']:g}** m")
    L.append(f"- ground_estimation: **{bc['ground_estimation']}**")
    L.append("")
    L.append("![foreground_ratio](figures/foreground_ratio_distribution.png)")
    L.append("")
    L.append(f"Foreground retention distribution at the best combo. Mean = {agg['best_foreground_ratio_mean']*100:.1f}% — the rest is dropped as ground/road/vegetation before clustering.")
    L.append("")

    # Comparison table (W1.5 vs β1)
    L.append("## W1.5 raw vs β1 best — direct comparison")
    L.append("")
    cmp = agg["comparison_to_w1_5"]
    L.append("| metric | W1.5 raw | β1 best | Δ |")
    L.append("|---|---|---|---|")
    for m in ["M_rate", "L_rate", "D_rate", "miss_rate"]:
        L.append(f"| {m} | {cmp[m]['w1_5']*100:.1f}% | {cmp[m]['beta1']*100:.1f}% | "
                 f"{cmp[m]['delta']*100:+.1f} |")
    L.append(f"| n_clusters | {cmp['n_clusters']['w1_5']:.1f} | "
             f"{cmp['n_clusters']['beta1']:.1f} | {cmp['n_clusters']['delta']:+.1f} |")
    L.append(f"| timing (median s) | {cmp['timing']['w1_5']:.3f} | "
             f"{cmp['timing']['beta1']:.3f} | {cmp['timing']['delta']:+.3f} |")
    L.append("")
    L.append("![before_after](figures/before_after_comparison.png)")
    L.append("")
    L.append("Side-by-side W1.5 raw HDBSCAN vs β1 (pillar foreground + same HDBSCAN). Δ values show the pillar-extraction effect in isolation.")
    L.append("")

    # Distance-stratified
    L.append("## Distance-stratified (best combo)")
    L.append("")
    L.append("![distance_stratified](figures/distance_stratified_best.png)")
    L.append("")
    L.append("M_rate and miss_rate per distance bin at the best combo.")
    L.append("")
    L.append("| bin | n_GT | M% | L% | D% | miss% |")
    L.append("|---|---|---|---|---|---|")
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
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
    L.append("Distribution of median per-sample timing across the 24 combos. Best combo's timing is marked red; PASS / WARN gates marked too.")
    L.append("")

    # Decision
    L.append("## Decision")
    L.append("")
    a = decision["cond_M_rate"]
    b = decision["cond_n_clusters"]
    c = decision["cond_timing"]
    L.append(f"- **cond_M_rate** = {a['value']*100:.2f}% → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"PARTIAL ≥ {a['thresholds']['PARTIAL']*100:.0f}%)")
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
    L.append(f"cond_M_rate     = {a['value']:.4f}  → {a['verdict']}")
    L.append(f"cond_n_clusters = {b['value']:.4f}  → {b['verdict']}")
    L.append(f"cond_timing     = {c['value']:.4f}s → {c['verdict']}")
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
    L.append("- **STRONG**: lock pillar config, advance to W2 (Coverage reliability).")
    L.append("- **PARTIAL**: try β1.5 (verticality filter) or β2 (PointNet++ pretrained).")
    L.append("- **FAIL**: reconsider — γ (CenterPoint) or δ (learned unsupervised) before retrying input-distribution tweaks.")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_beta1(summaries, best, sweep_record, regression, agg,
                     figures_dir, report_path):
    fig_M_rate_heatmap(summaries, osp.join(figures_dir, "M_rate_heatmap_pillar_size_z_threshold.png"))
    fig_pareto(summaries, best, osp.join(figures_dir, "n_clusters_vs_M_rate_pareto.png"))
    # foreground_ratio_distribution figure needs out_dirs to find best per-sample data
    out_dirs = {"per_sample_per_config": osp.join(osp.dirname(figures_dir),
                                                   "per_sample_per_config")}
    fig_foreground_ratio_distribution(out_dirs, best["combo_id"],
                                      osp.join(figures_dir, "foreground_ratio_distribution.png"))
    fig_distance_stratified_best(agg["best_per_sample_distance"]["by_bin_rates"],
                                  osp.join(figures_dir, "distance_stratified_best.png"))
    fig_before_after(agg["comparison_to_w1_5"],
                     osp.join(figures_dir, "before_after_comparison.png"))
    fig_timing_distribution(summaries, best,
                            osp.join(figures_dir, "timing_distribution.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg, decision)
    _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, report_path)
