"""γ aggregator + report writer."""

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
M_STRONG = 0.55
M_PARTIAL = 0.45
M_MARGINAL_LO = 0.36
M_MARGINAL_HI = 0.45
GAP_ACCEPTABLE = 0.20
GAP_PROBLEMATIC = 0.30
GAP_SEVERE = 0.50
N_PASS = 30
N_WARN = 50
T_PASS = 3.0
T_WARN = 5.0


# Baselines
BETA1_M = 0.3628          # post-install
BETA1_L = 0.0426
BETA1_D = 0.2306
BETA1_MISS = 0.3640
BETA1_N_CL = 182.16
BETA1_T = 0.998
OPTION5_M = 0.151
OPTION5_MISS = 0.459
OPTION5_N = 229.4
OPTION5_T = 0.329
W1_5_M = 0.285


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _per_sample_iter(out_dirs, combo_id):
    pat = osp.join(out_dirs["per_sample_per_config"], combo_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            yield json.load(f)


def aggregate_gamma(summaries, best, sweep_record, regression, cache, out_dirs):
    best_id = best["combo_id"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0,
                   "n_GT_A": 0, "n_GT_B": 0, "M_A": 0, "M_B": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0,
                          "n_GT_A": 0, "n_GT_B": 0, "M_A": 0, "M_B": 0}

    # paired comparison vs β1: for each sample, also need β1's M_rate
    # Step A best (pillar0.5x0.5_zthr0.3) saved per_sample = β1 best per_sample
    paired_M_beta1 = {}
    for tok in cache:
        try:
            with open(f"results/diagnosis_step_a/per_sample_per_config/pillar0.5x0.5_zthr0.3/{tok}.json") as f:
                paired_M_beta1[tok] = json.load(f)["M_rate"]
        except FileNotFoundError:
            paired_M_beta1[tok] = None

    paired = []
    for rec in _per_sample_iter(out_dirs, best_id):
        b1_M = paired_M_beta1.get(rec["sample_token"])
        paired.append({
            "tok": rec["sample_token"],
            "gamma_M": rec["M_rate_all"],
            "beta1_M": b1_M,
        })
        for g in rec["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in by_bin:
                b = "unknown"
            by_bin[b]["n_GT"] += 1
            by_bin[b][g["case"]] += 1
            grp = g.get("group", "B")
            if grp == "A":
                by_bin[b]["n_GT_A"] += 1
                if g["case"] == "M":
                    by_bin[b]["M_A"] += 1
            else:
                by_bin[b]["n_GT_B"] += 1
                if g["case"] == "M":
                    by_bin[b]["M_B"] += 1

    bin_rates = {b: {
        "n_GT": rec["n_GT"],
        "M_rate": rec["M"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "miss_rate": rec["miss"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "n_GT_A": rec["n_GT_A"], "n_GT_B": rec["n_GT_B"],
        "M_rate_A": rec["M_A"] / rec["n_GT_A"] if rec["n_GT_A"] else 0.0,
        "M_rate_B": rec["M_B"] / rec["n_GT_B"] if rec["n_GT_B"] else 0.0,
    } for b, rec in by_bin.items()}

    # paired analysis
    n_both = n_gamma_only = n_beta1_only = n_neither = 0
    for p in paired:
        if p["beta1_M"] is None:
            continue
        # crude: positive M_rate counts as "covers some GT". for paired comparison
        # at sample level — does γ beat β1 on this sample, lose, or equal?
        if p["gamma_M"] > p["beta1_M"] + 0.05:
            n_gamma_only += 1
        elif p["beta1_M"] > p["gamma_M"] + 0.05:
            n_beta1_only += 1
        else:
            n_both += 1

    return {
        "n_samples": len(cache),
        "n_combos": len(summaries),
        "selection_verdict": sweep_record["selection_verdict"],
        "regression": regression,
        "best": best,
        "best_per_sample_distance": {"by_bin": by_bin, "by_bin_rates": bin_rates},
        "paired_vs_beta1": {
            "n_paired": len([p for p in paired if p["beta1_M"] is not None]),
            "n_gamma_better": n_gamma_only,
            "n_beta1_better": n_beta1_only,
            "n_similar": n_both,
            "per_sample": paired,
        },
        "open_vocab_gap": {
            "M_rate_A": best["mean_M_rate_A"],
            "M_rate_B": best["mean_M_rate_B"],
            "gap": best["mean_M_rate_A"] - best["mean_M_rate_B"],
            "n_gt_A": best["n_gt_A_total"],
            "n_gt_B": best["n_gt_B_total"],
        },
        "comparison_to_baselines": {
            "M_rate":      {"w1_5": W1_5_M,    "beta1": BETA1_M,   "option5": OPTION5_M, "gamma": best["mean_M_rate_all"]},
            "L_rate":      {"beta1": BETA1_L,    "gamma": best["mean_L_rate_all"]},
            "D_rate":      {"beta1": BETA1_D,    "gamma": best["mean_D_rate_all"]},
            "miss_rate":   {"beta1": BETA1_MISS, "option5": OPTION5_MISS, "gamma": best["mean_miss_rate_all"]},
            "n_proposals": {"beta1": BETA1_N_CL, "option5": OPTION5_N, "gamma": best["mean_n_proposals"]},
            "timing":      {"beta1": BETA1_T,    "option5": OPTION5_T, "gamma": best["median_timing_total_s"]},
        },
    }


def _evaluate_decision(agg):
    best = agg["best"]
    M = best["mean_M_rate_all"]
    if M >= M_STRONG:
        m_v = "STRONG"
    elif M >= M_PARTIAL:
        m_v = "PARTIAL"
    elif M_MARGINAL_LO <= M < M_MARGINAL_HI:
        m_v = "MARGINAL"
    else:
        m_v = "FAIL"

    gap = agg["open_vocab_gap"]["gap"]
    if gap <= GAP_ACCEPTABLE:
        gap_v = "ACCEPTABLE"
    elif gap > GAP_SEVERE:
        gap_v = "SEVERE"
    elif gap > GAP_PROBLEMATIC:
        gap_v = "PROBLEMATIC"
    else:
        gap_v = "MODERATE"

    n = best["mean_n_proposals"]
    if n <= N_PASS:
        n_v = "PASS"
    elif n <= N_WARN:
        n_v = "WARN"
    else:
        n_v = "FAIL"

    t = best["median_timing_total_s"]
    if t < T_PASS:
        t_v = "PASS"
    elif t < T_WARN:
        t_v = "WARN"
    else:
        t_v = "FAIL"

    if m_v == "STRONG" and gap_v == "ACCEPTABLE" and n_v == "PASS":
        branch = ("γ STRONG. CenterPoint proposal + open-vocab labeling 정당화. "
                  "narrative 보존 가능 (with class-agnostic framing). W2 진행.")
    elif m_v == "STRONG" and gap_v in ("PROBLEMATIC", "SEVERE"):
        branch = ("γ M_rate strong but closed-set 한계 명확. "
                  "narrative 부분 후퇴 또는 hybrid (β1 ∪ γ) 검토.")
    elif m_v == "PARTIAL":
        branch = ("γ PARTIAL. β1 대비 개선 있지만 큰 폭 아님. "
                  "hybrid 검토 또는 다른 옵션.")
    elif m_v in ("MARGINAL", "FAIL"):
        branch = ("γ FAIL/MARGINAL. CenterPoint도 한계. "
                  "β1 + caveat W2 또는 학습된 unsupervised (ψ) 검토.")
    else:
        branch = "unknown verdict combination — review manually"

    return {
        "cond_M_rate": {"value": M, "verdict": m_v,
                         "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL,
                                         "MARGINAL_lo": M_MARGINAL_LO,
                                         "MARGINAL_hi": M_MARGINAL_HI}},
        "cond_open_vocab_gap": {"value": gap, "verdict": gap_v,
                                  "thresholds": {"ACCEPTABLE": GAP_ACCEPTABLE,
                                                  "PROBLEMATIC": GAP_PROBLEMATIC,
                                                  "SEVERE": GAP_SEVERE}},
        "cond_n_proposals": {"value": n, "verdict": n_v,
                               "thresholds": {"PASS": N_PASS, "WARN": N_WARN}},
        "cond_timing": {"value": t, "verdict": t_v,
                         "thresholds": {"PASS": T_PASS, "WARN": T_WARN}},
        "branch": branch,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_M_centerpoint_vs_beta1(summaries, best, agg, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    Ms = [s["mean_M_rate_all"] * 100 for s in summaries]
    ns = [s["mean_n_proposals"] for s in summaries]
    ax.scatter(ns, Ms, s=24, alpha=0.6, color="#666666", label="γ combos")
    ax.scatter([best["mean_n_proposals"]], [best["mean_M_rate_all"] * 100],
               s=160, edgecolor="red", facecolor="none", linewidth=2, label="γ best")
    ax.scatter([BETA1_N_CL], [BETA1_M * 100], s=140, marker="x", color="orange",
               label=f"β1 baseline ({BETA1_M*100:.1f}% / {BETA1_N_CL:.0f})")
    ax.scatter([OPTION5_N], [OPTION5_M * 100], s=140, marker="x", color="purple",
               label=f"Option 5 baseline ({OPTION5_M*100:.1f}% / {OPTION5_N:.0f})")
    ax.axvline(N_PASS, color="green", ls="--", lw=1, label=f"PASS≤{N_PASS}")
    ax.axhline(M_STRONG * 100, color="green", ls=":", lw=1, label=f"STRONG≥{M_STRONG*100:.0f}%")
    ax.axhline(M_PARTIAL * 100, color="orange", ls=":", lw=1, label=f"PARTIAL≥{M_PARTIAL*100:.0f}%")
    ax.set_xlabel("mean # proposals")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_title(f"γ sweep ({len(summaries)} combos) vs β1 / Option 5")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_learned_vs_unseen(agg, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ov = agg["open_vocab_gap"]
    cats = ["Group A\n(trained 10-class)", "Group B\n(unseen)"]
    M_rates = [ov["M_rate_A"] * 100, ov["M_rate_B"] * 100]
    counts = [ov["n_gt_A"], ov["n_gt_B"]]
    bars = ax.bar(cats, M_rates, color=["#117733", "#882255"])
    for bar, M, n in zip(bars, M_rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, M + 1,
                f"{M:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("M_rate (%)")
    ax.set_title(f"Closed-set vs open-vocab — gap = {ov['gap']*100:+.1f} pts")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_distance_stratified_comparison(agg, out):
    bins = DISTANCE_BIN_LABELS
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    M_all = [rates[b]["M_rate"] * 100 for b in bins]
    M_A = [rates[b]["M_rate_A"] * 100 for b in bins]
    M_B = [rates[b]["M_rate_B"] * 100 for b in bins]
    n_GT = [rates[b]["n_GT"] for b in bins]
    x = np.arange(len(bins))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, M_all, width, color="#4477AA", label="γ all-class M")
    ax.bar(x,         M_A,   width, color="#117733", label="γ Group A M")
    ax.bar(x + width, M_B,   width, color="#882255", label="γ Group B M")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(bins, n_GT)])
    ax.set_ylabel("M_rate (%)")
    ax.set_title("γ best — distance-stratified by Group")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_score_threshold_sweep(summaries, out):
    score_thrs = sorted({s["combo"]["score_threshold"] for s in summaries})
    nms_thrs = sorted({s["combo"]["nms_iou_threshold"] for s in summaries})
    table = {(s["combo"]["score_threshold"], s["combo"]["nms_iou_threshold"]):
             s["mean_M_rate_all"] for s in summaries}
    fig, ax = plt.subplots(figsize=(7, 4))
    for nms in nms_thrs:
        ys = [table.get((sc, nms), 0) * 100 for sc in score_thrs]
        ax.plot(score_thrs, ys, "o-", label=f"NMS={nms:g}")
    ax.set_xlabel("score_threshold")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title("γ — M_rate vs score threshold")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_n_proposals_distribution(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ns = [s["mean_n_proposals"] for s in summaries]
    Ms = [s["mean_M_rate_all"] * 100 for s in summaries]
    ax.scatter(ns, Ms, s=30, alpha=0.7, color="#4477AA", label="γ combos")
    ax.scatter([best["mean_n_proposals"]], [best["mean_M_rate_all"] * 100],
               s=140, edgecolor="red", facecolor="none", linewidth=2, label="best")
    ax.set_xlabel("mean # proposals")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title("γ proposal count vs M_rate trade-off")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_proposal_overlap(agg, out):
    p = agg["paired_vs_beta1"]
    cats = ["γ better", "β1 better", "similar (±5pt)"]
    vals = [p["n_gamma_better"], p["n_beta1_better"], p["n_similar"]]
    colors = ["#117733", "#882255", "#999999"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("# samples")
    ax.set_title(f"Per-sample paired comparison (β1 vs γ, n={p['n_paired']})")
    _save(fig, out)


def fig_timing_breakdown(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [s["median_timing_total_s"] for s in summaries]
    ax.hist(ts, bins=8, color="#CC6677", edgecolor="white")
    ax.axvline(best["median_timing_total_s"], color="red", ls="--",
               label=f"best={best['median_timing_total_s']:.2f}s")
    ax.axvline(T_PASS, color="green", ls=":", label=f"PASS<{T_PASS}s")
    ax.axvline(T_WARN, color="orange", ls=":", label=f"WARN<{T_WARN}s")
    ax.set_xlabel("median timing (s)")
    ax.set_ylabel("# combos")
    ax.set_title(f"γ sweep timing ({len(ts)} combos)")
    ax.legend(fontsize=8)
    _save(fig, out)


# ---------- top-3 ----------

def _top3(agg):
    cmp = agg["comparison_to_baselines"]
    obs = []
    obs.append(
        f"M progression: W1.5 {cmp['M_rate']['w1_5']*100:.1f}% → "
        f"β1 {cmp['M_rate']['beta1']*100:.1f}% → "
        f"Option 5 {cmp['M_rate']['option5']*100:.1f}% → "
        f"**γ {cmp['M_rate']['gamma']*100:.1f}%** "
        f"(Δ vs β1 = {(cmp['M_rate']['gamma'] - cmp['M_rate']['beta1'])*100:+.1f} pts). "
        f"n_proposals {cmp['n_proposals']['beta1']:.0f} → {cmp['n_proposals']['gamma']:.1f}."
    )
    ov = agg["open_vocab_gap"]
    obs.append(
        f"Closed-set vs open-vocab: Group A (trained 10-class) M_rate = "
        f"**{ov['M_rate_A']*100:.1f}%** (n={ov['n_gt_A']}) vs Group B (unseen) "
        f"M_rate = **{ov['M_rate_B']*100:.1f}%** (n={ov['n_gt_B']}). "
        f"Gap = **{ov['gap']*100:+.1f} pts** "
        f"({'class-agnostic framing 정당' if ov['gap'] <= GAP_ACCEPTABLE else 'closed-set bias 명확'}).")
    p = agg["paired_vs_beta1"]
    obs.append(
        f"Per-sample paired (β1 vs γ): γ better on {p['n_gamma_better']} samples, "
        f"β1 better on {p['n_beta1_better']}, similar (±5pt) on {p['n_similar']} "
        f"(of {p['n_paired']} paired). "
        + ("γ dominates β1 across most samples → hybrid不要." if p['n_gamma_better'] > p['n_beta1_better'] + p['n_similar']
           else "Mixed dominance — hybrid (β1 ∪ γ) 검토 가치 있음."
                if p['n_gamma_better'] > p['n_beta1_better']
                else "β1 still competitive on many samples → hybrid는 추가 가치 제한적.")
    )
    return obs


# ---------- report ----------

def _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, out_path):
    L = []
    L.append("# γ — CenterPoint proposal + open-vocab labeling")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}** (W1.5 set)")
    L.append(f"- Sweep: **{agg['n_combos']}** combos (4 score × 2 NMS)")
    L.append(f"- Selection verdict: **{agg['selection_verdict']}**")
    if regression.get("passes"):
        L.append(f"- Post-install β1 regression: M={regression['mean_M_rate']:.4f} "
                 f"(expected {regression['expected_M_rate']:.4f}, "
                 f"Δ={regression['delta_M']:.6f}) → **PASS**")
    else:
        L.append(f"- Post-install β1 regression: **FAIL** "
                 f"(M={regression['mean_M_rate']:.4f}, Δ={regression['delta_M']:.6f})")
    L.append("")

    L.append("## Sweep — 8 combos")
    L.append("")
    L.append("![M_vs_beta1](figures/M_rate_centerpoint_vs_β1.png)")
    L.append("")
    L.append("![score_sweep](figures/score_threshold_sweep.png)")
    L.append("")

    L.append("### Sweep table (sorted by M_rate)")
    L.append("")
    L.append("| score | NMS | M% all | M% A | M% B | gap pts | miss% | n_prop | timing s | n_ok |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: -x["mean_M_rate_all"]):
        c = s["combo"]
        gap_pts = (s["mean_M_rate_A"] - s["mean_M_rate_B"]) * 100
        L.append(f"| {c['score_threshold']:g} | {c['nms_iou_threshold']:g} | "
                 f"{s['mean_M_rate_all']*100:.1f} | "
                 f"{s['mean_M_rate_A']*100:.1f} | "
                 f"{s['mean_M_rate_B']*100:.1f} | "
                 f"{gap_pts:+.1f} | "
                 f"{s['mean_miss_rate_all']*100:.1f} | "
                 f"{s['mean_n_proposals']:.1f} | "
                 f"{s['median_timing_total_s']:.2f} | "
                 f"{s['n_samples_succeeded']}/{agg['n_samples']} |")
    L.append("")

    bc = best["combo"]
    L.append("## Best combo")
    L.append("")
    L.append(f"- score_threshold: **{bc['score_threshold']:g}**")
    L.append(f"- nms_iou_threshold: **{bc['nms_iou_threshold']:g}**")
    L.append("")

    # Pipeline comparison
    cmp = agg["comparison_to_baselines"]
    L.append("## Pipeline comparison: β1 vs Option 5 vs γ")
    L.append("")
    L.append("| metric | β1 | Option 5 | γ | Δγ vs β1 |")
    L.append("|---|---|---|---|---|")
    L.append(f"| M_rate | {cmp['M_rate']['beta1']*100:.1f}% | "
             f"{cmp['M_rate']['option5']*100:.1f}% | "
             f"{cmp['M_rate']['gamma']*100:.1f}% | "
             f"{(cmp['M_rate']['gamma'] - cmp['M_rate']['beta1'])*100:+.1f} |")
    L.append(f"| miss_rate | {cmp['miss_rate']['beta1']*100:.1f}% | "
             f"{cmp['miss_rate']['option5']*100:.1f}% | "
             f"{cmp['miss_rate']['gamma']*100:.1f}% | "
             f"{(cmp['miss_rate']['gamma'] - cmp['miss_rate']['beta1'])*100:+.1f} |")
    L.append(f"| L_rate | {cmp['L_rate']['beta1']*100:.1f}% | — | "
             f"{cmp['L_rate']['gamma']*100:.1f}% | "
             f"{(cmp['L_rate']['gamma'] - cmp['L_rate']['beta1'])*100:+.1f} |")
    L.append(f"| D_rate | {cmp['D_rate']['beta1']*100:.1f}% | — | "
             f"{cmp['D_rate']['gamma']*100:.1f}% | "
             f"{(cmp['D_rate']['gamma'] - cmp['D_rate']['beta1'])*100:+.1f} |")
    L.append(f"| n_proposals | {cmp['n_proposals']['beta1']:.1f} | "
             f"{cmp['n_proposals']['option5']:.1f} | "
             f"{cmp['n_proposals']['gamma']:.1f} | "
             f"{cmp['n_proposals']['gamma'] - cmp['n_proposals']['beta1']:+.1f} |")
    L.append(f"| timing (s) | {cmp['timing']['beta1']:.3f} | "
             f"{cmp['timing']['option5']:.3f} | "
             f"{cmp['timing']['gamma']:.3f} | "
             f"{cmp['timing']['gamma'] - cmp['timing']['beta1']:+.3f} |")
    L.append("")

    # Closed-set vs open-vocab
    L.append("## Closed-set vs Open-vocab analysis")
    L.append("")
    L.append("![learned_vs_unseen](figures/learned_vs_unseen_categories.png)")
    L.append("")
    ov = agg["open_vocab_gap"]
    L.append(f"- **Group A** (CenterPoint trained 10-class, n={ov['n_gt_A']}) — "
             f"M_rate = **{ov['M_rate_A']*100:.1f}%**")
    L.append(f"- **Group B** (unseen categories, n={ov['n_gt_B']}) — "
             f"M_rate = **{ov['M_rate_B']*100:.1f}%**")
    L.append(f"- **Gap = {ov['gap']*100:+.1f} pts**")
    L.append("")

    L.append("## Distance-stratified")
    L.append("")
    L.append("![distance](figures/distance_stratified_comparison.png)")
    L.append("")
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    L.append("| bin | n_GT | M% all | M% A (n) | M% B (n) | miss% |")
    L.append("|---|---|---|---|---|---|")
    for b in DISTANCE_BIN_LABELS:
        r = rates[b]
        if r["n_GT"] == 0:
            continue
        L.append(f"| {b} | {r['n_GT']} | {r['M_rate']*100:.1f} | "
                 f"{r['M_rate_A']*100:.1f} ({r['n_GT_A']}) | "
                 f"{r['M_rate_B']*100:.1f} ({r['n_GT_B']}) | "
                 f"{r['miss_rate']*100:.1f} |")
    L.append("")

    # Paired comparison
    L.append("## Per-sample paired (β1 vs γ)")
    L.append("")
    L.append("![paired](figures/proposal_overlap_analysis.png)")
    L.append("")
    p = agg["paired_vs_beta1"]
    L.append(f"- γ better (+5pt): {p['n_gamma_better']}")
    L.append(f"- β1 better (+5pt): {p['n_beta1_better']}")
    L.append(f"- Similar (±5pt): {p['n_similar']}")
    L.append(f"- (out of {p['n_paired']} paired samples)")
    L.append("")

    L.append("## Other figures")
    L.append("")
    L.append("![n_proposals](figures/n_proposals_distribution.png)")
    L.append("")
    L.append("![timing](figures/timing_breakdown.png)")
    L.append("")

    # Decision
    L.append("## Decision")
    L.append("")
    a = decision["cond_M_rate"]
    b = decision["cond_open_vocab_gap"]
    c = decision["cond_n_proposals"]
    d = decision["cond_timing"]
    L.append(f"- **cond_M_rate** = {a['value']*100:.2f}% → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"PARTIAL ≥ {a['thresholds']['PARTIAL']*100:.0f}%, "
             f"MARGINAL [{a['thresholds']['MARGINAL_lo']*100:.0f}, "
             f"{a['thresholds']['MARGINAL_hi']*100:.0f}]%)")
    L.append(f"- **cond_open_vocab_gap** = {b['value']*100:+.2f} pts → **{b['verdict']}** "
             f"(ACCEPTABLE ≤ {b['thresholds']['ACCEPTABLE']*100:.0f}, "
             f"PROBLEMATIC > {b['thresholds']['PROBLEMATIC']*100:.0f}, "
             f"SEVERE > {b['thresholds']['SEVERE']*100:.0f})")
    L.append(f"- **cond_n_proposals** = {c['value']:.2f} → **{c['verdict']}** "
             f"(PASS ≤ {c['thresholds']['PASS']}, WARN ≤ {c['thresholds']['WARN']})")
    L.append(f"- **cond_timing** = {d['value']:.3f}s → **{d['verdict']}** "
             f"(PASS < {d['thresholds']['PASS']}s, WARN < {d['thresholds']['WARN']}s)")
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_M_rate         = {a['value']:.4f}   → {a['verdict']}")
    L.append(f"cond_open_vocab_gap = {b['value']:+.4f}  → {b['verdict']}")
    L.append(f"cond_n_proposals    = {c['value']:.4f}   → {c['verdict']}")
    L.append(f"cond_timing         = {d['value']:.4f}s  → {d['verdict']}")
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
    L.append("## Next-step candidates (manual decision)")
    L.append("")
    L.append("- **STRONG + ACCEPTABLE**: lock γ config, advance to W2 with class-agnostic framing.")
    L.append("- **STRONG + PROBLEMATIC/SEVERE**: narrative compromise needed — closed-set bias acknowledged.")
    L.append("- **PARTIAL**: hybrid (β1 ∪ γ) detailed measurement.")
    L.append("- **MARGINAL/FAIL**: β1 + caveat W2 or learned unsupervised (ψ).")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_gamma(summaries, best, sweep_record, regression, agg,
                      figures_dir, report_path):
    fig_M_centerpoint_vs_beta1(summaries, best, agg,
                                 osp.join(figures_dir, "M_rate_centerpoint_vs_β1.png"))
    fig_learned_vs_unseen(agg, osp.join(figures_dir, "learned_vs_unseen_categories.png"))
    fig_distance_stratified_comparison(agg,
                                         osp.join(figures_dir, "distance_stratified_comparison.png"))
    fig_score_threshold_sweep(summaries, osp.join(figures_dir, "score_threshold_sweep.png"))
    fig_n_proposals_distribution(summaries, best,
                                   osp.join(figures_dir, "n_proposals_distribution.png"))
    fig_proposal_overlap(agg, osp.join(figures_dir, "proposal_overlap_analysis.png"))
    fig_timing_breakdown(summaries, best, osp.join(figures_dir, "timing_breakdown.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg)
    _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, report_path)
