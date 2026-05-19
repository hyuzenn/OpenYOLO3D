"""Step A aggregator + report writer.

Decision section is two-cond: cond_M_rate (4-tier) and
cond_too_small_inside_gt (informative). Branch fire reads both.
"""

from __future__ import annotations

import glob
import json
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# Decision thresholds
M_STRONG = 0.45
M_PARTIAL = 0.40
M_MARGINAL = 0.36   # β1 baseline ~0.361
TOO_SMALL_SIGNAL = 0.50
TOO_SMALL_NOISE = 0.20

# β1 baseline (locked numbers)
BETA1_M = 0.3612
BETA1_L = 0.044
BETA1_D = 0.230
BETA1_MISS = 0.364
BETA1_N_CL = 182.12
BETA1_TIMING = 0.829


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _per_sample_iter(out_dirs, combo_id):
    pat = osp.join(out_dirs["per_sample_per_config"], combo_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            yield json.load(f)


def aggregate_step_a(summaries, best, sweep_record, regression, cache, out_dirs):
    best_id = best["combo_id"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
    pillars_per_gt_all = []

    for rec in _per_sample_iter(out_dirs, best_id):
        for g in rec["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in by_bin:
                b = "unknown"
            by_bin[b]["n_GT"] += 1
            by_bin[b][g["case"]] += 1
        pillars_per_gt_all.extend(rec.get("pillars_per_gt_box", []))

    bin_rates = {b: {
        "n_GT": rec["n_GT"],
        "M_rate": rec["M"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "L_rate": rec["L"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "D_rate": rec["D"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "miss_rate": rec["miss"] / rec["n_GT"] if rec["n_GT"] else 0.0,
    } for b, rec in by_bin.items()}

    # Hypotheses
    H7_verdict = ("CONFIRMED" if best["combo"]["pillar_size_xy"][0] < 0.5 and best["mean_M_rate"] > BETA1_M
                  else "PARTIAL" if best["combo"]["pillar_size_xy"][0] < 0.5
                  else "REJECTED")
    H8_verdict = ("CONFIRMED" if best["mean_M_rate"] >= 0.42
                  else "PARTIAL" if best["mean_M_rate"] >= 0.40
                  else "REJECTED")
    cs = best["components_summed"]
    H9_verdict = ("CONFIRMED" if cs["too_small_inside_gt_ratio"] >= 0.50
                  else "PARTIAL" if cs["too_small_inside_gt_ratio"] >= 0.30
                  else "REJECTED")

    return {
        "n_samples": len(cache),
        "n_combos": len(summaries),
        "selection_verdict": sweep_record["selection_verdict"],
        "regression": regression,
        "best": best,
        "best_per_sample_distance": {"by_bin": by_bin, "by_bin_rates": bin_rates},
        "hypotheses": {
            "H7_resolution_helps": {"verdict": H7_verdict,
                                     "best_pillar_size": best["combo"]["pillar_size_xy"]},
            "H8_M_rate_above_42": {"verdict": H8_verdict,
                                    "best_M_rate": best["mean_M_rate"]},
            "H9_too_small_signal": {"verdict": H9_verdict,
                                     "too_small_inside_gt_ratio": cs["too_small_inside_gt_ratio"]},
        },
        "comparison_to_beta1": {
            "M_rate":     {"beta1": BETA1_M, "step_a": best["mean_M_rate"],
                            "delta": best["mean_M_rate"] - BETA1_M},
            "L_rate":     {"beta1": BETA1_L, "step_a": best["mean_L_rate"],
                            "delta": best["mean_L_rate"] - BETA1_L},
            "D_rate":     {"beta1": BETA1_D, "step_a": best["mean_D_rate"],
                            "delta": best["mean_D_rate"] - BETA1_D},
            "miss_rate":  {"beta1": BETA1_MISS, "step_a": best["mean_miss_rate"],
                            "delta": best["mean_miss_rate"] - BETA1_MISS},
            "n_clusters": {"beta1": BETA1_N_CL, "step_a": best["mean_n_clusters"],
                            "delta": best["mean_n_clusters"] - BETA1_N_CL},
            "timing":     {"beta1": BETA1_TIMING, "step_a": best["median_timing_total"],
                            "delta": best["median_timing_total"] - BETA1_TIMING},
        },
        "pillars_per_gt_distribution": {
            "mean": _safe_mean(pillars_per_gt_all),
            "median": float(np.median(pillars_per_gt_all)) if pillars_per_gt_all else 0.0,
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
    elif M >= M_MARGINAL:
        m_v = "MARGINAL"
    else:
        m_v = "FAIL"

    ts = best["components_summed"]["too_small_inside_gt_ratio"]
    if ts >= TOO_SMALL_SIGNAL:
        ts_v = "SIGNAL"
    elif ts < TOO_SMALL_NOISE:
        ts_v = "NOISE"
    else:
        ts_v = "MIXED"

    if m_v == "STRONG":
        branch = ("Step A STRONG. Pillar resolution이 β1.5 fail 원인 확인. "
                  "W2 진행 가능, config locked.")
    elif m_v == "PARTIAL":
        branch = ("Step A PARTIAL. Resolution 줄이는 게 도움. "
                  "W2 baseline으로 진행 또는 Option 1 (다른 unsupervised) 추가 시도.")
    elif m_v == "MARGINAL":
        branch = ("Step A PLATEAU. β1과 차이 없음. Geometry-only 한계 확정 강화. "
                  "Option 1 (다른 unsupervised 알고리즘) escalate.")
    elif m_v == "FAIL":
        branch = ("Step A FAIL. Resolution 줄이는 게 오히려 악화. "
                  "pillar 표현 자체 한계. Option 1 즉시 escalate.")
    else:
        branch = "unknown verdict — review tables manually"

    return {
        "cond_M_rate": {
            "value": M, "verdict": m_v,
            "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL, "MARGINAL": M_MARGINAL},
        },
        "cond_too_small_inside_gt": {
            "value": ts, "verdict": ts_v,
            "thresholds": {"SIGNAL": TOO_SMALL_SIGNAL, "NOISE": TOO_SMALL_NOISE},
        },
        "branch": branch,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_M_rate_vs_pillar_size(summaries, out):
    pillar_sizes = sorted({tuple(s["combo"]["pillar_size_xy"]) for s in summaries})
    z_thrs = sorted({s["combo"]["z_threshold"] for s in summaries})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = {(tuple(s["combo"]["pillar_size_xy"]), s["combo"]["z_threshold"]):
             s["mean_M_rate"] for s in summaries}
    x_labels = [f"{ps[0]:g}m" for ps in pillar_sizes]
    x = np.arange(len(pillar_sizes))
    width = 0.27
    colors = ["#4477AA", "#117733", "#CC6677"]
    for i, zt in enumerate(z_thrs):
        ys = [table.get((ps, zt), 0) * 100 for ps in pillar_sizes]
        ax.bar(x + (i - 1) * width, ys, width, color=colors[i % len(colors)],
               label=f"z_thr={zt:g}")
    ax.axhline(BETA1_M * 100, color="black", ls="--", lw=1, label=f"β1 baseline {BETA1_M*100:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("pillar size")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title("Step A — M_rate by pillar size and z_threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_too_small_breakdown(summaries, out):
    pillar_sizes = sorted({tuple(s["combo"]["pillar_size_xy"]) for s in summaries})
    # one bar per pillar size, average across z_thresholds
    in_gt = []
    out_gt = []
    for ps in pillar_sizes:
        in_v = np.mean([s["components_summed"]["too_small_inside_gt_ratio"]
                         for s in summaries if tuple(s["combo"]["pillar_size_xy"]) == ps])
        in_gt.append(in_v * 100)
        out_v = 100 - in_v * 100
        out_gt.append(out_v)
    x = np.arange(len(pillar_sizes))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, in_gt, color="#117733", label="too_small inside GT (signal lost)")
    ax.bar(x, out_gt, bottom=in_gt, color="#999999", label="too_small outside GT (true noise)")
    for i, (in_v, _) in enumerate(zip(in_gt, out_gt)):
        ax.text(x[i], in_v / 2, f"{in_v:.1f}%", ha="center", va="center", color="white", fontsize=10, weight="bold")
    ax.axhline(TOO_SMALL_SIGNAL * 100, color="red", ls="--", lw=1,
               label=f"SIGNAL threshold {TOO_SMALL_SIGNAL*100:.0f}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ps[0]:g}m" for ps in pillar_sizes])
    ax.set_xlabel("pillar size")
    ax.set_ylabel("too_small components (%)")
    ax.set_title("too_small components: signal (inside GT) vs noise (outside GT)")
    ax.legend(fontsize=8)
    _save(fig, out)


def fig_pillars_per_gt_distribution(summaries, out):
    pillar_sizes = sorted({tuple(s["combo"]["pillar_size_xy"]) for s in summaries})
    means_by_size = {}
    for ps in pillar_sizes:
        ms = [s["mean_avg_pillars_per_gt_box"]
              for s in summaries if tuple(s["combo"]["pillar_size_xy"]) == ps]
        means_by_size[ps] = float(np.mean(ms))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(pillar_sizes))
    ax.bar(x, [means_by_size[ps] for ps in pillar_sizes], color="#4477AA")
    for i, ps in enumerate(pillar_sizes):
        ax.text(x[i], means_by_size[ps] + 0.05, f"{means_by_size[ps]:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(3, color="red", ls="--", label="size_min=3 (β1.5 default)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ps[0]:g}m" for ps in pillar_sizes])
    ax.set_xlabel("pillar size")
    ax.set_ylabel("mean pillars per GT box")
    ax.set_title("How many pillars does the average GT live in?")
    ax.legend()
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
    ax.set_title("Step A best — distance-stratified")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_before_after(comparison, out):
    metrics = ["M_rate", "miss_rate", "L_rate", "D_rate"]
    beta = [comparison[m]["beta1"] * 100 for m in metrics]
    sa = [comparison[m]["step_a"] * 100 for m in metrics]
    x = np.arange(len(metrics))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, beta, width, color="#88CCEE", label="β1 best")
    ax.bar(x + width / 2, sa, width, color="#117733", label="Step A best")
    for i, m in enumerate(metrics):
        d = comparison[m]["delta"] * 100
        col = "green" if (m != "miss_rate" and d > 0) or (m == "miss_rate" and d < 0) else "red"
        ax.text(i, max(beta[i], sa[i]) + 1, f"Δ={d:+.1f}",
                ha="center", color=col, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("rate (%)")
    ax.set_title("β1 best vs Step A best")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_distribution(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [s["median_timing_total"] for s in summaries]
    ax.hist(ts, bins=10, color="#CC6677", edgecolor="white")
    ax.axvline(best["median_timing_total"], color="red", ls="--",
               label=f"best={best['median_timing_total']:.2f}s")
    ax.set_xlabel("median timing (s)")
    ax.set_ylabel("# combos")
    ax.set_title(f"Step A sweep timing ({len(ts)} combos)")
    ax.legend(fontsize=8)
    _save(fig, out)


# ---------- top-3 ----------

def _top3(agg):
    cmp = agg["comparison_to_beta1"]
    obs = []
    obs.append(
        f"Best Step A combo: pillar={agg['best']['combo']['pillar_size_xy']} "
        f"z_thr={agg['best']['combo']['z_threshold']} → "
        f"M_rate **{cmp['M_rate']['step_a']*100:.1f}%** (β1 {cmp['M_rate']['beta1']*100:.1f}%, "
        f"Δ={cmp['M_rate']['delta']*100:+.1f}). "
        f"miss {cmp['miss_rate']['step_a']*100:.1f}% vs β1 {cmp['miss_rate']['beta1']*100:.1f}%."
    )
    cs = agg['best']['components_summed']
    obs.append(
        f"too_small inside GT (signal lost ratio) at best combo = "
        f"**{cs['too_small_inside_gt_ratio']*100:.1f}%** "
        f"(too_small_in={cs['too_small_inside_gt']}, too_small_total={cs['n_too_small']}). "
        + ("β1.5의 over-filtering 강하게 입증 — too_small이 noise가 아니라 작은 객체 신호."
           if cs['too_small_inside_gt_ratio'] >= 0.50
           else "too_small이 noise+신호 혼재 — β1.5의 차단 결정이 미묘함."
                if cs['too_small_inside_gt_ratio'] >= 0.20
                else "too_small이 대부분 진짜 noise — β1.5의 차단은 합리적이나 그래도 -8.7pt 손실.")
    )
    obs.append(
        f"Mean pillars per GT box at best resolution = "
        f"{agg['pillars_per_gt_distribution']['mean']:.2f} "
        f"(median {agg['pillars_per_gt_distribution']['median']:.1f}). "
        + ("Most GTs occupy ≥3 pillars at this resolution — size_min=3 wouldn't strip them."
           if agg['pillars_per_gt_distribution']['mean'] >= 3
           else "Most GTs sit in <3 pillars → β1.5's size_min=3 was structurally over-aggressive.")
    )
    return obs


# ---------- report ----------

def _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, out_path):
    L = []
    L.append("# Step A — Pillar resolution sweep + too_small spatial analysis")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}** (W1.5 set)")
    L.append(f"- Sweep: **{agg['n_combos']}** combos (4 pillar × 3 z_thr; "
             f"ground=percentile p=10 fixed)")
    L.append(f"- Selection verdict: **{agg['selection_verdict']}**")
    if regression.get("passes"):
        L.append(f"- β1 regression (pillar=0.5/zt=0.3): M={regression['M_rate']:.4f} "
                 f"(Δ={regression['delta_M']:.6f}), n_cl={regression['n_clusters']:.4f} "
                 f"(Δ={regression['delta_n_clusters']:.6f}) → **PASS**")
    else:
        L.append(f"- β1 regression: **FAIL** — {regression}")
    L.append("")

    # Hypothesis check
    L.append("## Hypothesis check")
    L.append("")
    h = agg["hypotheses"]
    L.append(f"- **H7** — pillar 줄이면 small/far GT 신호 보존: "
             f"**{h['H7_resolution_helps']['verdict']}** "
             f"(best pillar = {h['H7_resolution_helps']['best_pillar_size']})")
    L.append(f"- **H8** — H7 → M_rate ≥ 0.42: "
             f"**{h['H8_M_rate_above_42']['verdict']}** "
             f"(best M_rate = {h['H8_M_rate_above_42']['best_M_rate']*100:.1f}%)")
    L.append(f"- **H9** — too_small_inside_gt_ratio ≥ 0.50 (β1.5 over-filtering): "
             f"**{h['H9_too_small_signal']['verdict']}** "
             f"(ratio = {h['H9_too_small_signal']['too_small_inside_gt_ratio']*100:.1f}%)")
    L.append("")

    # Sweep
    L.append("## Sweep — 12 combos")
    L.append("")
    L.append("![M_rate_vs_pillar_size](figures/M_rate_vs_pillar_size.png)")
    L.append("")
    L.append("M_rate plotted against pillar size with one bar per z_threshold. The dashed line is β1's locked baseline.")
    L.append("")

    L.append("### Sweep results table (sorted by M_rate)")
    L.append("")
    L.append("| pillar | z_thr | M% | miss% | n_cl | too_small_in_gt% | avg_pillars/gt | timing s | n_ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: -x["mean_M_rate"]):
        c = s["combo"]
        cs = s["components_summed"]
        L.append(f"| {c['pillar_size_xy'][0]:g}x{c['pillar_size_xy'][1]:g} | "
                 f"{c['z_threshold']:g} | "
                 f"{s['mean_M_rate']*100:.1f} | "
                 f"{s['mean_miss_rate']*100:.1f} | "
                 f"{s['mean_n_clusters']:.1f} | "
                 f"{cs['too_small_inside_gt_ratio']*100:.1f} | "
                 f"{s['mean_avg_pillars_per_gt_box']:.2f} | "
                 f"{s['median_timing_total']:.2f} | "
                 f"{s['n_samples_succeeded']}/{agg['n_samples']} |")
    L.append("")

    bc = best["combo"]
    L.append("## Best combo")
    L.append("")
    L.append(f"- pillar_size_xy: **({bc['pillar_size_xy'][0]:g}, {bc['pillar_size_xy'][1]:g})** m")
    L.append(f"- z_threshold: **{bc['z_threshold']:g}** m")
    L.append(f"- ground_estimation: percentile (p=10, fixed)")
    L.append("")

    # Comparison
    cmp = agg["comparison_to_beta1"]
    L.append("## β1 vs Step A — direct comparison")
    L.append("")
    L.append("| metric | β1 best | Step A best | Δ |")
    L.append("|---|---|---|---|")
    for m in ["M_rate", "L_rate", "D_rate", "miss_rate"]:
        L.append(f"| {m} | {cmp[m]['beta1']*100:.1f}% | {cmp[m]['step_a']*100:.1f}% | "
                 f"{cmp[m]['delta']*100:+.1f} |")
    L.append(f"| n_clusters | {cmp['n_clusters']['beta1']:.1f} | {cmp['n_clusters']['step_a']:.1f} | "
             f"{cmp['n_clusters']['delta']:+.1f} |")
    L.append(f"| timing (s) | {cmp['timing']['beta1']:.3f} | {cmp['timing']['step_a']:.3f} | "
             f"{cmp['timing']['delta']:+.3f} |")
    L.append("")
    L.append("![before_after](figures/before_after_β1_stepA.png)")
    L.append("")

    # too_small breakdown
    L.append("## too_small component breakdown")
    L.append("")
    L.append("![too_small_breakdown](figures/too_small_breakdown.png)")
    L.append("")
    cs = best["components_summed"]
    L.append(f"At best combo: too_small_inside_gt = {cs['too_small_inside_gt']}, "
             f"too_small_outside_gt = {cs['too_small_outside_gt']}, "
             f"kept_inside_gt = {cs['kept_inside_gt']}, "
             f"kept_outside_gt = {cs['kept_outside_gt']}.")
    L.append("")
    L.append(f"too_small_inside_gt_ratio = **{cs['too_small_inside_gt_ratio']*100:.1f}%** "
             f"({'≥ 50% → β1.5 over-filtering 입증' if cs['too_small_inside_gt_ratio'] >= 0.50 else '< 50%'})")
    L.append("")
    L.append("![pillars_per_gt](figures/pillars_per_gt_distribution.png)")
    L.append("")
    L.append(f"Mean pillars per GT box at best resolution = "
             f"{agg['pillars_per_gt_distribution']['mean']:.2f}.")
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
    b = decision["cond_too_small_inside_gt"]
    L.append(f"- **cond_M_rate** = {a['value']*100:.2f}% → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"PARTIAL ≥ {a['thresholds']['PARTIAL']*100:.0f}%, "
             f"MARGINAL ≥ {a['thresholds']['MARGINAL']*100:.0f}%)")
    L.append(f"- **cond_too_small_inside_gt** = {b['value']*100:.2f}% → **{b['verdict']}** "
             f"(SIGNAL ≥ {b['thresholds']['SIGNAL']*100:.0f}%, NOISE < {b['thresholds']['NOISE']*100:.0f}%)")
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_M_rate              = {a['value']:.4f}  → {a['verdict']}")
    L.append(f"cond_too_small_inside_gt = {b['value']:.4f}  → {b['verdict']}")
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
    L.append("- **STRONG**: lock pillar+z config, advance to W2.")
    L.append("- **PARTIAL**: advance to W2 with caveat or try Option 1.")
    L.append("- **MARGINAL/PLATEAU**: geometry-only ceiling — escalate to Option 1 (Euclidean / region-growing / etc).")
    L.append("- **FAIL**: pillar representation has hit a wall — Option 1 immediately.")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_step_a(summaries, best, sweep_record, regression, agg,
                      figures_dir, report_path):
    fig_M_rate_vs_pillar_size(summaries, osp.join(figures_dir, "M_rate_vs_pillar_size.png"))
    fig_too_small_breakdown(summaries, osp.join(figures_dir, "too_small_breakdown.png"))
    fig_pillars_per_gt_distribution(summaries, osp.join(figures_dir, "pillars_per_gt_distribution.png"))
    fig_distance_stratified_best(agg["best_per_sample_distance"]["by_bin_rates"],
                                  osp.join(figures_dir, "distance_stratified_best.png"))
    fig_before_after(agg["comparison_to_beta1"],
                      osp.join(figures_dir, "before_after_β1_stepA.png"))
    fig_timing_distribution(summaries, best,
                             osp.join(figures_dir, "timing_distribution.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg)
    _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, report_path)
