"""W1 aggregate, figures, report.

Decision section fires three checks dynamically against measured values:
  cond_n_clusters_mean: 5..15 PASS, <5 FAIL_LOW, >15 FAIL_HIGH
  cond_timing:          <1.0s PASS, 1..3s WARN, ≥3s FAIL
  cond_match_M_rate:    info-only; flagged when <0.30
Branch routing is derived from those verdicts.
"""

from __future__ import annotations

import json
import os.path as osp
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------- thresholds ----------
N_CLUSTERS_LO = 5
N_CLUSTERS_HI = 15
N_CLUSTERS_TARGET = 10
TIMING_PASS_S = 1.0
TIMING_WARN_S = 3.0
M_RATE_INFO_FLAG = 0.30


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _safe_median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _safe_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(xs)) if xs else None


# ---------- aggregate ----------

def aggregate_w1(samples, sweep_record):
    n_clusters_per = [s["n_clusters"] for s in samples]
    timings = [s["timing"]["total"] for s in samples]
    noise_ratios = [s["noise_ratio"] for s in samples]
    ground_ratios = [s["ground_filtered_ratio"] for s in samples]

    case_totals = {"M": 0, "L": 0, "D": 0, "miss": 0}
    n_gt_total = 0
    for s in samples:
        for k in case_totals:
            case_totals[k] += s["case_counts"].get(k, 0)
        n_gt_total += s["n_gt_total"]
    case_rates = {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in case_totals.items()}

    all_sizes = []
    all_extents_xy = []
    for s in samples:
        all_sizes.extend(s["cluster_sizes"])
        all_extents_xy.extend(s["cluster_extent_xy"])

    # mini vs trainval split
    mini_n = [s["n_clusters"] for s in samples if s["source"] == "mini"]
    tv_n = [s["n_clusters"] for s in samples if s["source"] == "trainval"]

    return {
        "n_samples": len(samples),
        "best_config": sweep_record["best"],
        "selection_rule": sweep_record["selection_rule"],
        "n_clusters": {
            "mean": _safe_mean(n_clusters_per),
            "median": _safe_median(n_clusters_per),
            "std": _safe_std(n_clusters_per),
            "min": int(min(n_clusters_per)) if n_clusters_per else None,
            "max": int(max(n_clusters_per)) if n_clusters_per else None,
            "by_source": {
                "mini_mean": _safe_mean(mini_n),
                "mini_median": _safe_median(mini_n),
                "mini_n_samples": len(mini_n),
                "trainval_mean": _safe_mean(tv_n),
                "trainval_median": _safe_median(tv_n),
                "trainval_n_samples": len(tv_n),
            },
        },
        "cluster_size": {
            "mean": _safe_mean(all_sizes),
            "median": _safe_median(all_sizes),
            "min": int(min(all_sizes)) if all_sizes else None,
            "max": int(max(all_sizes)) if all_sizes else None,
        },
        "cluster_extent_xy": {
            "mean": _safe_mean(all_extents_xy),
            "median": _safe_median(all_extents_xy),
        },
        "noise_ratio": {
            "mean": _safe_mean(noise_ratios),
            "median": _safe_median(noise_ratios),
        },
        "ground_filtered_ratio": {
            "mean": _safe_mean(ground_ratios),
        },
        "timing_total_s": {
            "mean": _safe_mean(timings),
            "median": _safe_median(timings),
            "p95": float(np.percentile(timings, 95)) if timings else None,
            "max": float(max(timings)) if timings else None,
        },
        "gt_cluster_matching": {
            "n_gt_total": n_gt_total,
            "case_counts": case_totals,
            "case_rates": case_rates,
        },
    }


# ---------- decision ----------

def _evaluate_decision(agg):
    n_mean = agg["n_clusters"]["mean"]
    if n_mean is None:
        n_verdict = "UNDEFINED"
    elif n_mean < N_CLUSTERS_LO:
        n_verdict = "FAIL_LOW"
    elif n_mean > N_CLUSTERS_HI:
        n_verdict = "FAIL_HIGH"
    else:
        n_verdict = "PASS"

    t_med = agg["timing_total_s"]["median"]
    if t_med is None:
        t_verdict = "UNDEFINED"
    elif t_med < TIMING_PASS_S:
        t_verdict = "PASS"
    elif t_med < TIMING_WARN_S:
        t_verdict = "WARN"
    else:
        t_verdict = "FAIL"

    m_rate = agg["gt_cluster_matching"]["case_rates"]["M"]

    # Branch routing
    if n_verdict == "PASS" and t_verdict == "PASS":
        branch = "HDBSCAN 채택"
        rationale = ("cond_n_clusters_mean is inside [5, 15] and timing is sub-second. "
                     "Lock these parameters as the method config; proceed to W2 (Coverage reliability).")
    elif n_verdict == "FAIL_LOW":
        branch = "CenterPoint fallback 권고"
        rationale = ("HDBSCAN under-segments at this parameter band — additional tuning is "
                     "unlikely to resolve a structural under-segmentation. Reconsider closed-set "
                     "CenterPoint as proposal source (open-vocab justification needed for committee).")
    elif n_verdict == "FAIL_HIGH":
        branch = "추가 sweep 또는 CenterPoint"
        rationale = ("HDBSCAN over-segments — a denser parameter sweep or a post-filter merge step "
                     "is the cheapest remediation; only escalate to CenterPoint if those fail.")
    elif t_verdict == "FAIL":
        branch = "Real-time 위협 — GPU HDBSCAN 또는 pre-cluster 점 수 감축"
        rationale = ("Median walltime exceeded 3s; the proposal stage cannot meet a real-time budget. "
                     "Replace CPU HDBSCAN with cuML or downsample input.")
    elif t_verdict == "WARN":
        branch = "HDBSCAN 채택 (timing 모니터링)"
        rationale = ("cond_n_clusters_mean PASSes but timing landed in the 1–3s warn band. "
                     "Acceptable to proceed if downstream W2/W3 cost remains small; otherwise plan "
                     "a GPU port.")
    else:
        branch = "조사 필요"
        rationale = "Unexpected verdict combination — inspect aggregate.json directly."

    return {
        "cond_n_clusters_mean": {"value": n_mean, "verdict": n_verdict,
                                 "thresholds": {"PASS_lo": N_CLUSTERS_LO, "PASS_hi": N_CLUSTERS_HI,
                                                "target": N_CLUSTERS_TARGET}},
        "cond_timing_median_s": {"value": t_med, "verdict": t_verdict,
                                 "thresholds": {"PASS_max": TIMING_PASS_S, "WARN_max": TIMING_WARN_S}},
        "cond_match_M_rate": {"value": m_rate, "info_flag_threshold": M_RATE_INFO_FLAG,
                              "flagged_low": m_rate < M_RATE_INFO_FLAG},
        "branch": branch,
        "rationale": rationale,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_cluster_count_distribution(samples, agg, out):
    ns = [s["n_clusters"] for s in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    if ns:
        bins = np.arange(0, max(ns) + 2) - 0.5
        ax.hist(ns, bins=bins, color="#4477AA", edgecolor="white")
        m = agg["n_clusters"]["mean"]
        if m is not None:
            ax.axvline(m, color="red", ls="--", label=f"mean={m:.2f}")
        ax.axvspan(N_CLUSTERS_LO, N_CLUSTERS_HI, color="green", alpha=0.10,
                   label=f"PASS band [{N_CLUSTERS_LO},{N_CLUSTERS_HI}]")
        ax.legend()
    ax.set_xlabel("# clusters per frame")
    ax.set_ylabel("# samples")
    ax.set_title(f"Cluster count distribution (n={len(ns)})")
    _save(fig, out)


def fig_cluster_size_distribution(samples, out):
    sizes = []
    for s in samples:
        sizes.extend(s["cluster_sizes"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if sizes:
        upper = int(np.percentile(sizes, 99))
        ax.hist(np.clip(sizes, 0, upper), bins=40, color="#117733", edgecolor="white")
    ax.set_xlabel("# points per cluster (clipped at p99)")
    ax.set_ylabel("# clusters")
    ax.set_title(f"Cluster size distribution (n_clusters={len(sizes)})")
    ax.set_yscale("log")
    _save(fig, out)


def fig_gt_cluster_matching(agg, out):
    rates = agg["gt_cluster_matching"]["case_rates"]
    counts = agg["gt_cluster_matching"]["case_counts"]
    cases = ["M", "L", "D", "miss"]
    colors = ["#117733", "#DDCC77", "#882255", "#999999"]
    descriptions = {
        "M": "1↔1 match (ideal)",
        "L": "1 GT ↔ many clusters\n(over-seg)",
        "D": "many GTs ↔ 1 cluster\n(under-seg)",
        "miss": "GT ↔ no cluster",
    }
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cases))
    bars = ax.bar(x, [rates[c] for c in cases], color=colors)
    for bar, c in zip(bars, cases):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{rates[c]*100:.1f}%\n(n={counts[c]})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{descriptions[c]}" for c in cases], fontsize=9)
    ax.set_ylim(0, max(0.05, max(rates.values()) * 1.25))
    ax.set_ylabel("rate of LiDAR-supported GT boxes")
    ax.set_title(f"GT-cluster matching cases (total GT={agg['gt_cluster_matching']['n_gt_total']})")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_distribution(samples, agg, out):
    timings = [s["timing"]["total"] for s in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    if timings:
        ax.hist(timings, bins=20, color="#CC6677", edgecolor="white")
        m = agg["timing_total_s"]["median"]
        if m is not None:
            ax.axvline(m, color="black", ls="--", label=f"median={m:.3f}s")
        ax.axvline(TIMING_PASS_S, color="green", ls=":", label=f"PASS<{TIMING_PASS_S}s")
        ax.axvline(TIMING_WARN_S, color="red", ls=":", label=f"FAIL≥{TIMING_WARN_S}s")
        ax.legend()
    ax.set_xlabel("HDBSCAN total walltime (s)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Per-sample HDBSCAN timing (n={len(timings)})")
    _save(fig, out)


def fig_noise_ratio_distribution(samples, out):
    rs = [s["noise_ratio"] for s in samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    if rs:
        ax.hist(rs, bins=20, color="#882255", edgecolor="white", range=(0, 1))
    ax.set_xlabel("noise ratio (HDBSCAN -1 fraction over filtered points)")
    ax.set_ylabel("# samples")
    ax.set_title(f"Noise ratio distribution (n={len(rs)})")
    _save(fig, out)


def plot_parameter_sweep(results, grid, out):
    """3-panel heatmap: rows=min_cluster_size, cols=min_samples, one panel per epsilon."""
    eps_vals = grid["cluster_selection_epsilon"]
    mcs_vals = grid["min_cluster_size"]
    ms_vals = grid["min_samples"]
    fig, axes = plt.subplots(1, len(eps_vals), figsize=(4.0 * len(eps_vals), 4.0), squeeze=False)
    # Build a lookup
    table = {(r["min_cluster_size"], r["min_samples"], r["cluster_selection_epsilon"]):
             r["mean_n_clusters"] for r in results}
    vmin = min(r["mean_n_clusters"] for r in results)
    vmax = max(r["mean_n_clusters"] for r in results)
    for i, eps in enumerate(eps_vals):
        ax = axes[0][i]
        Z = np.zeros((len(mcs_vals), len(ms_vals)))
        for r, mcs in enumerate(mcs_vals):
            for c, ms in enumerate(ms_vals):
                Z[r, c] = table.get((mcs, ms, eps), np.nan)
        im = ax.imshow(Z, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ms_vals)))
        ax.set_xticklabels(ms_vals)
        ax.set_yticks(range(len(mcs_vals)))
        ax.set_yticklabels(mcs_vals)
        ax.set_xlabel("min_samples")
        if i == 0:
            ax.set_ylabel("min_cluster_size")
        ax.set_title(f"eps={eps}")
        for r in range(len(mcs_vals)):
            for c in range(len(ms_vals)):
                v = Z[r, c]
                colour = "white" if v > (vmin + vmax) / 2 else "black"
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", color=colour, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Parameter sweep — mean n_clusters across {sum(1 for _ in results)} combos\n"
                 f"green band [5,15] is the gate", fontsize=11)
    _save(fig, out)


# ---------- top-3 observations ----------

def _top3_observations(agg, samples):
    """Return three observations in the order the W1 spec calls for:
       1. cluster-count distribution characteristic
       2. dominant GT-cluster matching case (M/L/D/miss)
       3. mini vs trainval delta (when both splits present)
    """
    obs = []

    # 1) cluster count distribution shape
    n = agg["n_clusters"]
    if n["mean"] is not None:
        obs.append({
            "observation": (
                f"n_clusters/frame: mean={n['mean']:.2f}, median={n['median']}, "
                f"std={n['std']:.2f}, range=[{n['min']}, {n['max']}]. "
                f"Distance from target ({N_CLUSTERS_TARGET}) = "
                f"{abs(n['mean'] - N_CLUSTERS_TARGET):.2f}; the [5, 15] PASS band is "
                + ("entered." if 5.0 <= n['mean'] <= 15.0 else "not entered — over-seg side." if n['mean'] > 15.0 else "not entered — under-seg side.")
            ),
        })

    # 2) dominant GT-cluster matching case
    cr = agg["gt_cluster_matching"]["case_rates"]
    cc = agg["gt_cluster_matching"]["case_counts"]
    n_gt = agg["gt_cluster_matching"]["n_gt_total"]
    dominant = max(cr.items(), key=lambda kv: kv[1])
    obs.append({
        "observation": (
            f"Dominant GT-cluster case = **{dominant[0]}** at {dominant[1]*100:.1f}% "
            f"({cc[dominant[0]]} of {n_gt} GTs). "
            f"Full mix: M={cr['M']*100:.1f}% (n={cc['M']}), "
            f"L={cr['L']*100:.1f}% (n={cc['L']}), "
            f"D={cr['D']*100:.1f}% (n={cc['D']}), "
            f"miss={cr['miss']*100:.1f}% (n={cc['miss']}). "
            f"This is the principal input for W3 fusion design — Case M rate is the bound on "
            f"clean per-object lifting."
        ),
    })

    # 3) mini vs trainval split
    by = n["by_source"]
    if by["mini_n_samples"] and by["trainval_n_samples"]:
        delta = by["mini_mean"] - by["trainval_mean"]
        obs.append({
            "observation": (
                f"mini (n={by['mini_n_samples']}) mean n_clusters = {by['mini_mean']:.2f} "
                f"vs trainval (n={by['trainval_n_samples']}) = {by['trainval_mean']:.2f} "
                f"(Δ={delta:+.2f}). "
                + ("Distributions are within ~1 cluster — no scene-mix bias detected." if abs(delta) < 1.5
                   else "Splits diverge by >1.5 clusters — trainval scenes are systematically "
                        + ("denser" if delta < 0 else "sparser") + " in foreground objects than mini.")
            ),
        })
    return obs


# ---------- report ----------

def _write_report(samples, agg, decision, sweep_record, failed, observations,
                  figures_dir, out_path):
    L = []
    L.append("# W1 — HDBSCAN LiDAR proposal gate")
    L.append("")
    n_total = len(samples) + len(failed)
    L.append(f"- Samples succeeded: **{len(samples)}** / {n_total}")
    L.append(f"- Samples failed: {len(failed)}")
    L.append(f"- GT boxes scored: {agg['gt_cluster_matching']['n_gt_total']}")
    L.append("")

    # Locked config
    bc = agg["best_config"]
    L.append("## Best HDBSCAN config (locked from parameter sweep)")
    L.append("")
    L.append(f"- min_cluster_size: **{bc['min_cluster_size']}**")
    L.append(f"- min_samples: **{bc['min_samples']}**")
    L.append(f"- cluster_selection_epsilon: **{bc['cluster_selection_epsilon']}**")
    L.append(f"- selection: {bc['selection']}; mean_n_clusters during sweep = "
             f"{bc['mean_n_clusters']:.2f}")
    L.append(f"- selection rule: {sweep_record['selection_rule']}")
    L.append("")

    # Summary table
    L.append("## Summary table")
    L.append("")
    L.append("| Metric | Value |")
    L.append("|---|---|")
    L.append(f"| # samples (mini + trainval) | {agg['n_clusters']['by_source']['mini_n_samples']} + "
             f"{agg['n_clusters']['by_source']['trainval_n_samples']} |")
    L.append(f"| Mean n_clusters / frame | {agg['n_clusters']['mean']:.2f} |")
    L.append(f"| Median n_clusters / frame | {agg['n_clusters']['median']} |")
    L.append(f"| Range n_clusters | [{agg['n_clusters']['min']}, {agg['n_clusters']['max']}] |")
    L.append(f"| Median cluster size (pts) | {agg['cluster_size']['median']} |")
    L.append(f"| Median cluster xy-extent (m) | {agg['cluster_extent_xy']['median']:.2f} |")
    L.append(f"| Median noise ratio | {agg['noise_ratio']['median']*100:.1f}% |")
    L.append(f"| Mean ground-filtered ratio | {agg['ground_filtered_ratio']['mean']*100:.1f}% |")
    L.append(f"| Median HDBSCAN walltime | {agg['timing_total_s']['median']:.3f} s |")
    L.append(f"| p95 HDBSCAN walltime | {agg['timing_total_s']['p95']:.3f} s |")
    cr = agg["gt_cluster_matching"]["case_rates"]
    L.append(f"| GT match M / L / D / miss | "
             f"{cr['M']*100:.1f}% / {cr['L']*100:.1f}% / "
             f"{cr['D']*100:.1f}% / {cr['miss']*100:.1f}% |")
    L.append("")

    # mini vs trainval breakdown
    by = agg["n_clusters"]["by_source"]
    if by["mini_n_samples"] and by["trainval_n_samples"]:
        L.append("### mini vs trainval mean n_clusters")
        L.append("")
        L.append("| split | n samples | mean | median |")
        L.append("|---|---|---|---|")
        L.append(f"| mini | {by['mini_n_samples']} | {by['mini_mean']:.2f} | {by['mini_median']} |")
        L.append(f"| trainval | {by['trainval_n_samples']} | {by['trainval_mean']:.2f} | {by['trainval_median']} |")
        L.append("")

    # ---- figures ----
    L.append("## Parameter sweep")
    L.append("")
    L.append("![parameter_sweep_heatmap](figures/parameter_sweep_heatmap.png)")
    L.append("")
    L.append(f"Mean n_clusters across the {len(sweep_record['results'])} grid combos. "
             f"The locked config sits at min_cluster_size={bc['min_cluster_size']}, "
             f"min_samples={bc['min_samples']}, eps={bc['cluster_selection_epsilon']}.")
    L.append("")

    L.append("## Cluster statistics")
    L.append("")
    L.append("![cluster_count_distribution](figures/cluster_count_distribution.png)")
    L.append("")
    L.append("Per-frame cluster count. The shaded band is the [5, 15] PASS gate; the dashed line "
             "marks the measured mean.")
    L.append("")
    L.append("![cluster_size_distribution](figures/cluster_size_distribution.png)")
    L.append("")
    L.append("Per-cluster point count (log-y). Long tail toward larger clusters is expected — "
             "background structures (large building facades) survive ground-filter.")
    L.append("")

    L.append("## GT–cluster matching (3-case + miss)")
    L.append("")
    L.append("![gt_cluster_matching](figures/gt_cluster_matching.png)")
    L.append("")
    L.append("M = 1↔1 (ideal), L = 1 GT spans many clusters (over-seg), D = many GTs share 1 cluster (under-seg), "
             "miss = GT has no clustered points. The M rate is the dominant input for W3 fusion design.")
    L.append("")

    L.append("## Timing")
    L.append("")
    L.append("![timing_distribution](figures/timing_distribution.png)")
    L.append("")
    L.append("HDBSCAN per-sample walltime. Real-time priority is sub-second median; warn band 1–3 s; "
             "fail at ≥3 s.")
    L.append("")

    L.append("![noise_ratio_distribution](figures/noise_ratio_distribution.png)")
    L.append("")
    L.append("Fraction of filtered (post-ground) points that HDBSCAN assigns to noise.")
    L.append("")

    # ---- decision ----
    L.append("## Decision")
    L.append("")
    a = decision["cond_n_clusters_mean"]
    b = decision["cond_timing_median_s"]
    c = decision["cond_match_M_rate"]
    L.append(f"- **cond_n_clusters_mean** = {a['value']:.2f}  → **{a['verdict']}** "
             f"(PASS band [{a['thresholds']['PASS_lo']}, {a['thresholds']['PASS_hi']}], "
             f"target {a['thresholds']['target']})")
    L.append(f"- **cond_timing_median_s** = {b['value']:.3f} s  → **{b['verdict']}** "
             f"(PASS < {b['thresholds']['PASS_max']}, WARN [{b['thresholds']['PASS_max']}, "
             f"{b['thresholds']['WARN_max']}), FAIL ≥ {b['thresholds']['WARN_max']})")
    L.append(f"- **cond_match_M_rate** = {c['value']:.3f}  *(info-only; flagged when < {c['info_flag_threshold']})*"
             + ("  ⚠ FLAGGED LOW" if c["flagged_low"] else ""))
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append(decision["rationale"])
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_n_clusters_mean = {a['value']:.2f}  → {a['verdict']}")
    L.append(f"cond_timing          = {b['value']:.3f}s → {b['verdict']}")
    L.append(f"cond_match_M_rate    = {c['value']:.3f}  (info only)")
    L.append(f"→ 분기: {decision['branch']}")
    L.append("```")
    L.append("")

    # ---- top-3 observations ----
    L.append("## Top-3 observations")
    L.append("")
    if not observations:
        L.append("No observations passed the deviation threshold.")
    else:
        for i, o in enumerate(observations, 1):
            L.append(f"{i}. {o['observation']}")
            L.append("")

    # ---- failed ----
    L.append("## Failed samples")
    L.append("")
    if not failed:
        L.append("None.")
    else:
        for fr in failed:
            L.append(f"- `{fr['sample_token']}` — {fr.get('reason', 'unknown')}")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_w1(samples, agg, sweep_record, failed, figures_dir, report_path):
    fig_cluster_count_distribution(samples, agg, osp.join(figures_dir, "cluster_count_distribution.png"))
    fig_cluster_size_distribution(samples, osp.join(figures_dir, "cluster_size_distribution.png"))
    fig_gt_cluster_matching(agg, osp.join(figures_dir, "gt_cluster_matching.png"))
    fig_timing_distribution(samples, agg, osp.join(figures_dir, "timing_distribution.png"))
    fig_noise_ratio_distribution(samples, osp.join(figures_dir, "noise_ratio_distribution.png"))
    # parameter_sweep_heatmap.png is written during the sweep step itself,
    # so don't re-draw it here.
    decision = _evaluate_decision(agg)
    observations = _top3_observations(agg, samples)
    _write_report(samples, agg, decision, sweep_record, failed, observations,
                  figures_dir, report_path)
