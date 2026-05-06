"""Step 1 aggregator + report writer.

Three sources are summarised side-by-side: Mask3D, HDBSCAN-best, Hybrid (union).
Decision section fires from three measured conditions:
  cond_mask3d_M_rate   — does Mask3D alone clear the 0.30 / 0.15 thresholds
  cond_hybrid_gain     — covered_by_either − max(mask3d_cov, hdbscan_cov)
  cond_complementarity — mask3d_only + hdbscan_only fraction
"""

from __future__ import annotations

import json
import os.path as osp
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# --- decision thresholds ---
M_STRONG = 0.30
M_MEDIUM = 0.15
HYBRID_GAIN_STRONG = 0.20
HYBRID_GAIN_MEDIUM = 0.10
COMPLEMENT_STRONG = 0.30
COMPLEMENT_WEAK = 0.15


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _safe_median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


# ---------- aggregate ----------

def aggregate_step1(samples_m, samples_h, samples_y):
    n_gt_total = sum(s.get("n_gt_total", 0) for s in samples_m)

    # M3D
    m_cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for s in samples_m:
        for k in m_cases:
            m_cases[k] += s["case_counts"].get(k, 0)
    m_n_inst = [s["n_instances"] for s in samples_m]
    m_timing = [s["timing"]["total_s"] for s in samples_m]
    m_rates = {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in m_cases.items()}

    # HDBSCAN
    h_cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    for s in samples_h:
        for k in h_cases:
            h_cases[k] += s["case_counts"].get(k, 0)
    h_n_inst = [s["n_instances"] for s in samples_h]
    h_timing = [s["timing"]["total"] for s in samples_h]
    h_rates = {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in h_cases.items()}

    # Hybrid coverage
    y_counts = {"mask3d_only": 0, "hdbscan_only": 0, "both": 0, "neither": 0,
                "covered_by_either": 0}
    y_union_cases = {"M": 0, "L": 0, "D": 0, "miss": 0}
    y_n_total_proposals = []
    for s in samples_y:
        for k in y_counts:
            y_counts[k] += s["counts"].get(k, 0)
        for k in y_union_cases:
            y_union_cases[k] += s["union_case_counts"].get(k, 0)
        y_n_total_proposals.append(s["n_proposals_total"])
    y_rates = {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in y_counts.items()}
    y_union_rates = {k: (v / n_gt_total) if n_gt_total else 0.0
                     for k, v in y_union_cases.items()}

    mask3d_covered_rate = m_rates["M"] + m_rates["L"] + m_rates["D"]
    hdbscan_covered_rate = h_rates["M"] + h_rates["L"] + h_rates["D"]
    hybrid_covered_rate = y_rates["covered_by_either"]
    hybrid_gain = hybrid_covered_rate - max(mask3d_covered_rate, hdbscan_covered_rate)
    complementarity = y_rates["mask3d_only"] + y_rates["hdbscan_only"]

    # Distance-stratified
    bin_breakdown = {b: {
        "n_GT": 0,
        "mask3d": {"M": 0, "L": 0, "D": 0, "miss": 0},
        "hdbscan": {"M": 0, "L": 0, "D": 0, "miss": 0},
        "hybrid_covered_either": 0,
    } for b in DISTANCE_BIN_LABELS}
    bin_breakdown["unknown"] = {**bin_breakdown[DISTANCE_BIN_LABELS[0]]}
    bin_breakdown["unknown"]["mask3d"] = {"M": 0, "L": 0, "D": 0, "miss": 0}
    bin_breakdown["unknown"]["hdbscan"] = {"M": 0, "L": 0, "D": 0, "miss": 0}
    bin_breakdown["unknown"]["n_GT"] = 0
    bin_breakdown["unknown"]["hybrid_covered_either"] = 0

    for s_m in samples_m:
        for g in s_m["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in bin_breakdown:
                b = "unknown"
            bin_breakdown[b]["n_GT"] += 1
            bin_breakdown[b]["mask3d"][g["case"]] += 1
    for s_h in samples_h:
        for g in s_h["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in bin_breakdown:
                b = "unknown"
            bin_breakdown[b]["hdbscan"][g["case"]] += 1
    for s_y in samples_y:
        for g in s_y["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in bin_breakdown:
                b = "unknown"
            if g["covered_tag"] in ("mask3d_only", "hdbscan_only", "both"):
                bin_breakdown[b]["hybrid_covered_either"] += 1

    return {
        "n_samples": len(samples_m),
        "n_gt_total": n_gt_total,
        "mask3d": {
            "case_counts": m_cases, "case_rates": m_rates,
            "mean_n_instances": _safe_mean(m_n_inst),
            "median_n_instances": _safe_median(m_n_inst),
            "mean_timing_s": _safe_mean(m_timing),
            "median_timing_s": _safe_median(m_timing),
            "covered_rate": mask3d_covered_rate,
        },
        "hdbscan": {
            "case_counts": h_cases, "case_rates": h_rates,
            "mean_n_instances": _safe_mean(h_n_inst),
            "median_n_instances": _safe_median(h_n_inst),
            "mean_timing_s": _safe_mean(h_timing),
            "median_timing_s": _safe_median(h_timing),
            "covered_rate": hdbscan_covered_rate,
        },
        "hybrid": {
            "counts": y_counts, "rates": y_rates,
            "union_case_counts": y_union_cases,
            "union_case_rates": y_union_rates,
            "mean_n_total_proposals": _safe_mean(y_n_total_proposals),
            "median_n_total_proposals": _safe_median(y_n_total_proposals),
            "covered_rate": hybrid_covered_rate,
            "gain_over_best_single": hybrid_gain,
            "complementarity": complementarity,
        },
        "by_distance_bin": bin_breakdown,
    }


# ---------- decision ----------

def _evaluate_decision(agg):
    m_M = agg["mask3d"]["case_rates"]["M"]
    if m_M >= M_STRONG:
        m_verdict = "STRONG"
    elif m_M >= M_MEDIUM:
        m_verdict = "MEDIUM"
    else:
        m_verdict = "WEAK"

    gain = agg["hybrid"]["gain_over_best_single"]
    if gain >= HYBRID_GAIN_STRONG:
        gain_verdict = "STRONG"
    elif gain >= HYBRID_GAIN_MEDIUM:
        gain_verdict = "MEDIUM"
    else:
        gain_verdict = "WEAK"

    comp = agg["hybrid"]["complementarity"]
    if comp >= COMPLEMENT_STRONG:
        comp_verdict = "STRONG"
    elif comp >= COMPLEMENT_WEAK:
        comp_verdict = "MEDIUM"
    else:
        comp_verdict = "WEAK"

    if m_M >= M_STRONG:
        branch = "Mask3D 단독 복귀 검토 — Decision A 첫 결정 재고."
    elif gain >= HYBRID_GAIN_STRONG and comp >= COMPLEMENT_STRONG:
        branch = ("Hybrid (Mask3D + HDBSCAN) 강하게 권고. "
                  "W2부터 dual-source proposal로 진행.")
    elif m_M < M_MEDIUM and gain < HYBRID_GAIN_MEDIUM:
        branch = "Mask3D도 HDBSCAN도 부족. CenterPoint 또는 다른 source 검토."
    else:
        branch = "Mixed result. 본인 검토 후 결정."

    return {
        "cond_mask3d_M_rate": {
            "value": m_M, "verdict": m_verdict,
            "thresholds": {"STRONG": M_STRONG, "MEDIUM": M_MEDIUM},
        },
        "cond_hybrid_gain": {
            "value": gain, "verdict": gain_verdict,
            "thresholds": {"STRONG": HYBRID_GAIN_STRONG, "MEDIUM": HYBRID_GAIN_MEDIUM},
        },
        "cond_complementarity": {
            "value": comp, "verdict": comp_verdict,
            "thresholds": {"STRONG": COMPLEMENT_STRONG, "WEAK": COMPLEMENT_WEAK},
        },
        "branch": branch,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_matching_comparison_bar(agg, out):
    sources = ["Mask3D", "HDBSCAN", "Hybrid (union)"]
    cases = ["M", "L", "D", "miss"]
    colors = {"M": "#117733", "L": "#DDCC77", "D": "#882255", "miss": "#999999"}
    rates_per_source = [
        agg["mask3d"]["case_rates"],
        agg["hdbscan"]["case_rates"],
        agg["hybrid"]["union_case_rates"],
    ]
    x = np.arange(len(sources))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(cases):
        ax.bar(x + (i - 1.5) * width,
               [r[c] * 100 for r in rates_per_source],
               width, color=colors[c], label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylabel("rate (%)")
    ax.set_title("Mask3D vs HDBSCAN-best vs Hybrid (union) — GT match cases")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_coverage_overlap_venn(agg, out):
    counts = agg["hybrid"]["counts"]
    rates = agg["hybrid"]["rates"]
    keys = ["mask3d_only", "both", "hdbscan_only", "neither"]
    colors = ["#88CCEE", "#117733", "#DDCC77", "#999999"]
    labels = [f"Mask3D only\n{rates['mask3d_only']*100:.1f}%\n(n={counts['mask3d_only']})",
              f"Both\n{rates['both']*100:.1f}%\n(n={counts['both']})",
              f"HDBSCAN only\n{rates['hdbscan_only']*100:.1f}%\n(n={counts['hdbscan_only']})",
              f"Neither\n{rates['neither']*100:.1f}%\n(n={counts['neither']})"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(range(4), [counts[k] for k in keys], color=colors)
    for bar, lab in zip(bars, labels):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, lab,
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Mask3D only", "Both", "HDBSCAN only", "Neither"])
    ax.set_ylabel("# GT boxes")
    ax.set_title(f"GT coverage decomposition (total {agg['n_gt_total']} GTs)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_distance_stratified_comparison(agg, out):
    bins = DISTANCE_BIN_LABELS
    bb = agg["by_distance_bin"]
    m_M = []
    h_M = []
    hyb_cov = []
    n_GT = []
    for b in bins:
        n = bb[b]["n_GT"]
        n_GT.append(n)
        m_M.append((bb[b]["mask3d"]["M"] / n) * 100 if n else 0.0)
        h_M.append((bb[b]["hdbscan"]["M"] / n) * 100 if n else 0.0)
        hyb_cov.append((bb[b]["hybrid_covered_either"] / n) * 100 if n else 0.0)
    x = np.arange(len(bins))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - width, m_M, width, color="#88CCEE", label="Mask3D M")
    ax.bar(x,         h_M, width, color="#CC6677", label="HDBSCAN M")
    ax.bar(x + width, hyb_cov, width, color="#117733", label="Hybrid covered")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, n_GT)])
    ax.set_ylabel("rate (%)")
    ax.set_title("Distance-stratified comparison — M_rate (Mask3D, HDBSCAN) and Hybrid coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_n_proposals_comparison(samples_m, samples_h, samples_y, out):
    m = [s["n_instances"] for s in samples_m]
    h = [s["n_instances"] for s in samples_h]
    y = [s["n_proposals_total"] for s in samples_y]
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = [1, 2, 3]
    bp = ax.boxplot([m, h, y], positions=pos, widths=0.55, showfliers=True,
                    patch_artist=True)
    colors = ["#88CCEE", "#CC6677", "#117733"]
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
    ax.set_xticks(pos)
    ax.set_xticklabels(["Mask3D", "HDBSCAN-best", "Hybrid (union)"])
    ax.set_ylabel("# proposals per sample")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_title(f"Proposal count distribution (n={len(m)} samples)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_comparison(samples_m, samples_h, out):
    m = [s["timing"]["total_s"] for s in samples_m]
    h = [s["timing"]["total"] for s in samples_h]
    fig, ax = plt.subplots(figsize=(8, 4))
    pos = [1, 2]
    bp = ax.boxplot([m, h], positions=pos, widths=0.55, showfliers=True,
                    patch_artist=True)
    colors = ["#88CCEE", "#CC6677"]
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
    ax.set_xticks(pos)
    ax.set_xticklabels(["Mask3D", "HDBSCAN-best"])
    ax.set_ylabel("walltime per sample (s)")
    ax.set_title(f"Per-sample inference timing (n={len(m)} / {len(h)})")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_hybrid_marginal_gain(agg, out):
    """Marginal coverage gain from adding HDBSCAN on top of Mask3D, per distance bin."""
    bins = DISTANCE_BIN_LABELS
    bb = agg["by_distance_bin"]
    base = []
    extra = []
    for b in bins:
        n = bb[b]["n_GT"]
        if not n:
            base.append(0); extra.append(0); continue
        # Mask3D-covered (M+L+D) is the base; HDBSCAN-only is the marginal pick-up.
        m_cov = (bb[b]["mask3d"]["M"] + bb[b]["mask3d"]["L"] + bb[b]["mask3d"]["D"]) / n * 100
        either_cov = bb[b]["hybrid_covered_either"] / n * 100
        base.append(m_cov)
        extra.append(max(0.0, either_cov - m_cov))
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, base, color="#88CCEE", label="Mask3D-covered (%)")
    ax.bar(x, extra, bottom=base, color="#117733", label="HDBSCAN marginal +")
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("GT covered (%)")
    ax.set_title("Hybrid marginal gain by distance bin")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


# ---------- top-3 + report ----------

def _top3(agg, decision):
    obs = []
    cr_m = agg["mask3d"]["case_rates"]
    cr_h = agg["hdbscan"]["case_rates"]
    obs.append(
        f"Mask3D vs HDBSCAN-best M_rate: **{cr_m['M']*100:.1f}%** vs "
        f"**{cr_h['M']*100:.1f}%** (Δ={cr_m['M']-cr_h['M']:+.3f}). "
        f"miss_rate: {cr_m['miss']*100:.1f}% vs {cr_h['miss']*100:.1f}%. "
        f"Mask3D produces {agg['mask3d']['mean_n_instances']:.1f} proposals/frame "
        f"vs HDBSCAN's {agg['hdbscan']['mean_n_instances']:.1f}."
    )
    g = agg["hybrid"]["gain_over_best_single"]
    cov_m = agg["mask3d"]["covered_rate"]
    cov_h = agg["hdbscan"]["covered_rate"]
    cov_y = agg["hybrid"]["covered_rate"]
    obs.append(
        f"Hybrid covered = {cov_y*100:.1f}% (Mask3D {cov_m*100:.1f}%, "
        f"HDBSCAN {cov_h*100:.1f}%). Hybrid gain over best single source = "
        f"**{g*100:+.1f} pts**. Complementarity (mask3d_only + hdbscan_only) = "
        f"{agg['hybrid']['complementarity']*100:.1f}%."
    )
    bb = agg["by_distance_bin"]
    near_n = bb["0-10m"]["n_GT"] + bb["10-20m"]["n_GT"]
    far_n = bb["30-50m"]["n_GT"] + bb["50m+"]["n_GT"]
    near_m_cov = ((bb["0-10m"]["mask3d"]["M"] + bb["0-10m"]["mask3d"]["L"] + bb["0-10m"]["mask3d"]["D"]) +
                  (bb["10-20m"]["mask3d"]["M"] + bb["10-20m"]["mask3d"]["L"] + bb["10-20m"]["mask3d"]["D"]))
    far_m_cov = ((bb["30-50m"]["mask3d"]["M"] + bb["30-50m"]["mask3d"]["L"] + bb["30-50m"]["mask3d"]["D"]) +
                 (bb["50m+"]["mask3d"]["M"] + bb["50m+"]["mask3d"]["L"] + bb["50m+"]["mask3d"]["D"]))
    near_h_cov = ((bb["0-10m"]["hdbscan"]["M"] + bb["0-10m"]["hdbscan"]["L"] + bb["0-10m"]["hdbscan"]["D"]) +
                  (bb["10-20m"]["hdbscan"]["M"] + bb["10-20m"]["hdbscan"]["L"] + bb["10-20m"]["hdbscan"]["D"]))
    far_h_cov = ((bb["30-50m"]["hdbscan"]["M"] + bb["30-50m"]["hdbscan"]["L"] + bb["30-50m"]["hdbscan"]["D"]) +
                 (bb["50m+"]["hdbscan"]["M"] + bb["50m+"]["hdbscan"]["L"] + bb["50m+"]["hdbscan"]["D"]))
    obs.append(
        f"Distance split — near (0–20m, n={near_n}): Mask3D covered "
        f"{(near_m_cov/max(1,near_n))*100:.1f}% vs HDBSCAN "
        f"{(near_h_cov/max(1,near_n))*100:.1f}%. "
        f"Far (30m+, n={far_n}): Mask3D {(far_m_cov/max(1,far_n))*100:.1f}% vs "
        f"HDBSCAN {(far_h_cov/max(1,far_n))*100:.1f}%. "
        + ("Sources have different distance preferences." if
           (near_m_cov/max(1,near_n) > near_h_cov/max(1,near_n)) !=
           (far_m_cov/max(1,far_n) > far_h_cov/max(1,far_n))
           else "Sources prefer the same distance regime — Hybrid effectively redundant.")
    )
    return obs


def _write_report(samples_m, samples_h, samples_y, agg, decision,
                  failed_m, failed_h, observations, figures_dir, out_path):
    L = []
    L.append("# Step 1 — Mask3D vs HDBSCAN-best vs Hybrid")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}**")
    L.append(f"- Total GT boxes scored: {agg['n_gt_total']}")
    L.append(f"- Mask3D failures: {len(failed_m)}; HDBSCAN failures: {len(failed_h)}")
    L.append("")

    # Comparison table
    L.append("## Mask3D vs HDBSCAN-best vs Hybrid — headline table")
    L.append("")
    L.append("| metric | Mask3D | HDBSCAN-best | Hybrid (union) |")
    L.append("|---|---|---|---|")
    L.append(f"| mean # proposals / frame | {agg['mask3d']['mean_n_instances']:.2f} | "
             f"{agg['hdbscan']['mean_n_instances']:.2f} | "
             f"{agg['hybrid']['mean_n_total_proposals']:.2f} |")
    L.append(f"| M rate | {agg['mask3d']['case_rates']['M']*100:.1f}% | "
             f"{agg['hdbscan']['case_rates']['M']*100:.1f}% | "
             f"{agg['hybrid']['union_case_rates']['M']*100:.1f}% |")
    L.append(f"| L rate | {agg['mask3d']['case_rates']['L']*100:.1f}% | "
             f"{agg['hdbscan']['case_rates']['L']*100:.1f}% | "
             f"{agg['hybrid']['union_case_rates']['L']*100:.1f}% |")
    L.append(f"| D rate | {agg['mask3d']['case_rates']['D']*100:.1f}% | "
             f"{agg['hdbscan']['case_rates']['D']*100:.1f}% | "
             f"{agg['hybrid']['union_case_rates']['D']*100:.1f}% |")
    L.append(f"| miss rate | {agg['mask3d']['case_rates']['miss']*100:.1f}% | "
             f"{agg['hdbscan']['case_rates']['miss']*100:.1f}% | "
             f"{agg['hybrid']['union_case_rates']['miss']*100:.1f}% |")
    L.append(f"| GT covered (M+L+D or any) | {agg['mask3d']['covered_rate']*100:.1f}% | "
             f"{agg['hdbscan']['covered_rate']*100:.1f}% | "
             f"{agg['hybrid']['covered_rate']*100:.1f}% |")
    L.append(f"| median timing (s) | {agg['mask3d']['median_timing_s']:.3f} | "
             f"{agg['hdbscan']['median_timing_s']:.3f} | — |")
    L.append("")

    # Hybrid Venn
    L.append("## Hybrid coverage decomposition")
    L.append("")
    cov = agg["hybrid"]
    L.append(f"- Mask3D-only covered: {cov['rates']['mask3d_only']*100:.1f}% "
             f"({cov['counts']['mask3d_only']} GTs)")
    L.append(f"- HDBSCAN-only covered: {cov['rates']['hdbscan_only']*100:.1f}% "
             f"({cov['counts']['hdbscan_only']} GTs)")
    L.append(f"- Both: {cov['rates']['both']*100:.1f}% ({cov['counts']['both']} GTs)")
    L.append(f"- Neither: {cov['rates']['neither']*100:.1f}% ({cov['counts']['neither']} GTs)")
    L.append(f"- Covered by either: **{cov['rates']['covered_by_either']*100:.1f}%**")
    L.append("")
    L.append(f"- Hybrid gain over best single source: **{cov['gain_over_best_single']*100:+.1f} pts**")
    L.append(f"- Complementarity (mask3d_only + hdbscan_only): **{cov['complementarity']*100:.1f}%**")
    L.append("")

    # Distance-stratified
    L.append("## Distance-stratified breakdown")
    L.append("")
    L.append("| bin | n_GT | Mask3D M% | Mask3D miss% | HDBSCAN M% | HDBSCAN miss% | Hybrid covered% |")
    L.append("|---|---|---|---|---|---|---|")
    for b in DISTANCE_BIN_LABELS:
        rec = agg["by_distance_bin"][b]
        n = rec["n_GT"]
        if not n:
            continue
        m_M = rec["mask3d"]["M"] / n * 100
        m_miss = rec["mask3d"]["miss"] / n * 100
        h_M = rec["hdbscan"]["M"] / n * 100
        h_miss = rec["hdbscan"]["miss"] / n * 100
        y_cov = rec["hybrid_covered_either"] / n * 100
        L.append(f"| {b} | {n} | {m_M:.1f} | {m_miss:.1f} | {h_M:.1f} | {h_miss:.1f} | {y_cov:.1f} |")
    L.append("")

    # Figures
    L.append("## Figures")
    L.append("")
    figs = [
        ("matching_comparison_bar.png",
         "Side-by-side M/L/D/miss rates for Mask3D, HDBSCAN-best, and the Hybrid (union) source."),
        ("coverage_overlap_venn.png",
         "Per-GT coverage decomposition. Both = covered by both sources; Mask3D only / HDBSCAN only = the gain side of Hybrid; Neither = lower bound on irreducible miss."),
        ("distance_stratified_comparison.png",
         "M_rate per source plus Hybrid covered rate, broken down by distance bin. Distance preference asymmetry is the key Hybrid signal."),
        ("n_proposals_comparison.png",
         "Per-sample proposal count distribution (symlog). Mask3D is sparse; HDBSCAN-best is dense; Hybrid is their sum."),
        ("timing_comparison.png",
         "Per-sample walltime distribution for Mask3D and HDBSCAN inference."),
        ("hybrid_marginal_gain.png",
         "Stacked bar — Mask3D-covered fraction (base) plus the marginal lift from adding HDBSCAN, per distance bin."),
    ]
    for fname, desc in figs:
        L.append(f"### {fname}")
        L.append("")
        L.append(f"![{fname}](figures/{fname})")
        L.append("")
        L.append(desc)
        L.append("")

    # Decision
    L.append("## Decision")
    L.append("")
    a = decision["cond_mask3d_M_rate"]
    b = decision["cond_hybrid_gain"]
    c = decision["cond_complementarity"]
    L.append(f"- **cond_mask3d_M_rate** = {a['value']*100:.2f}%  → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"MEDIUM ≥ {a['thresholds']['MEDIUM']*100:.0f}%)")
    L.append(f"- **cond_hybrid_gain** = {b['value']*100:+.2f} pts  → **{b['verdict']}** "
             f"(STRONG ≥ {b['thresholds']['STRONG']*100:.0f} pts, "
             f"MEDIUM ≥ {b['thresholds']['MEDIUM']*100:.0f} pts)")
    L.append(f"- **cond_complementarity** = {c['value']*100:.2f}%  → **{c['verdict']}** "
             f"(STRONG ≥ {c['thresholds']['STRONG']*100:.0f}%, "
             f"WEAK < {c['thresholds']['WEAK']*100:.0f}%)")
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_mask3d_M_rate    = {a['value']:.4f}  → {a['verdict']}")
    L.append(f"cond_hybrid_gain      = {b['value']:+.4f}  → {b['verdict']}")
    L.append(f"cond_complementarity  = {c['value']:.4f}  → {c['verdict']}")
    L.append(f"→ 분기: {decision['branch']}")
    L.append("```")
    L.append("")

    # Top-3 observations
    L.append("## Top-3 observations")
    L.append("")
    for i, o in enumerate(observations, 1):
        L.append(f"{i}. {o}")
        L.append("")

    # Failed
    L.append("## Failed samples")
    L.append("")
    if not failed_m and not failed_h:
        L.append("None.")
    else:
        L.append(f"Mask3D: {len(failed_m)}, HDBSCAN: {len(failed_h)}")
        for fr in failed_m:
            L.append(f"- mask3d `{fr.get('sample_token','?')}` — {fr.get('status')} ({fr.get('stage','?')})")
        for fr in failed_h:
            L.append(f"- hdbscan `{fr.get('sample_token','?')}` — {fr.get('status')} ({fr.get('stage','?')})")
    L.append("")

    # Next-step candidates
    L.append("## Next-step candidates (manual decision required)")
    L.append("")
    L.append("- **mask3d_M_rate STRONG**: revisit Decision A — Mask3D may suffice on its own.")
    L.append("- **hybrid_gain + complementarity STRONG**: lock Hybrid as proposal stage, advance to W2.")
    L.append("- **both WEAK**: prepare CenterPoint / alternate proposal source; Decision A's premise stays valid but HDBSCAN was the wrong replacement.")
    L.append("- **mixed**: re-read the per-distance breakdown before committing.")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_step1(samples_m, samples_h, samples_y, agg, failed_m, failed_h,
                    figures_dir, report_path):
    fig_matching_comparison_bar(agg, osp.join(figures_dir, "matching_comparison_bar.png"))
    fig_coverage_overlap_venn(agg, osp.join(figures_dir, "coverage_overlap_venn.png"))
    fig_distance_stratified_comparison(agg, osp.join(figures_dir, "distance_stratified_comparison.png"))
    fig_n_proposals_comparison(samples_m, samples_h, samples_y,
                               osp.join(figures_dir, "n_proposals_comparison.png"))
    fig_timing_comparison(samples_m, samples_h,
                          osp.join(figures_dir, "timing_comparison.png"))
    fig_hybrid_marginal_gain(agg, osp.join(figures_dir, "hybrid_marginal_gain.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg, decision)
    _write_report(samples_m, samples_h, samples_y, agg, decision,
                  failed_m, failed_h, observations, figures_dir, report_path)
