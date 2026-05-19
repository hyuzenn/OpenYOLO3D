"""Option 5 aggregator + report writer."""

from __future__ import annotations

import glob
import json
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# --- Decision thresholds ---
M_STRONG = 0.50
M_PARTIAL = 0.42
M_PLATEAU_LO = 0.33   # β1 baseline 0.36 ± 0.03
M_PLATEAU_HI = 0.39
N_PROP_PASS = 30
N_PROP_WARN = 50
TIMING_PASS = 3.0
TIMING_WARN = 4.0
RECALL_HIGH = 0.55     # 2D detection recall — informational

# Pipeline baselines
W1_5_M = 0.285;     W1_5_MISS = 0.305;  W1_5_N_CL = 189.7;  W1_5_T = 1.480
BETA1_M = 0.3612;   BETA1_MISS = 0.364; BETA1_N_CL = 182.12; BETA1_T = 0.829
BETA1_L = 0.044; BETA1_D = 0.230


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _per_sample_iter(out_dirs, combo_id):
    pat = osp.join(out_dirs["per_sample_per_config"], combo_id, "*.json")
    for fp in sorted(glob.glob(pat)):
        with open(fp) as f:
            yield json.load(f)


def aggregate_option5(summaries, best, sweep_record, regression, cache, out_dirs):
    best_id = best["combo_id"]
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0,
                   "detected_in_any": 0} for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0,
                          "detected_in_any": 0}

    M_in_det = M_in_undet = miss_in_det = miss_in_undet = 0
    n_det = n_undet = 0
    n_proposals_total = 0
    timings_yolo_per_sample = []
    timings_pipeline_per_sample = []

    for rec in _per_sample_iter(out_dirs, best_id):
        n_proposals_total += rec["n_proposals_total"]
        timings_pipeline_per_sample.append(rec["timing_total_s"])
        # rec["timing_breakdown"] doesn't include YOLO (which was cached);
        # so timing here represents only the per-sample frustum+pillar+HDBSCAN+matching cost.
        for g in rec["per_gt"]:
            b = g.get("distance_bin") or "unknown"
            if b not in by_bin:
                b = "unknown"
            by_bin[b]["n_GT"] += 1
            by_bin[b][g["case"]] += 1
            if g.get("detected_in_any_cam"):
                by_bin[b]["detected_in_any"] += 1
                if g["case"] == "M":
                    M_in_det += 1
                elif g["case"] == "miss":
                    miss_in_det += 1
                n_det += 1
            else:
                if g["case"] == "M":
                    M_in_undet += 1
                elif g["case"] == "miss":
                    miss_in_undet += 1
                n_undet += 1

    bin_rates = {b: {
        "n_GT": rec["n_GT"],
        "M_rate": rec["M"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "L_rate": rec["L"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "D_rate": rec["D"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "miss_rate": rec["miss"] / rec["n_GT"] if rec["n_GT"] else 0.0,
        "two_d_recall": rec["detected_in_any"] / rec["n_GT"] if rec["n_GT"] else 0.0,
    } for b, rec in by_bin.items()}

    # Hypotheses
    H10_verdict = ("CONFIRMED" if 7 <= best["mean_n_proposals"] <= 30
                    else "PARTIAL" if best["mean_n_proposals"] <= 50
                    else "REJECTED")
    H11_verdict = ("CONFIRMED" if best["mean_M_rate"] >= 0.50
                    else "PARTIAL" if best["mean_M_rate"] >= 0.42
                    else "REJECTED")
    far_M_30plus = ((bin_rates["30-50m"]["M_rate"] * by_bin["30-50m"]["n_GT"]
                      + bin_rates["50m+"]["M_rate"] * by_bin["50m+"]["n_GT"])
                     / max(1, by_bin["30-50m"]["n_GT"] + by_bin["50m+"]["n_GT"]))
    H12_verdict = ("CONFIRMED" if far_M_30plus >= 0.40
                    else "PARTIAL" if far_M_30plus >= 0.30
                    else "REJECTED")
    H13_recall = best["mean_2d_recall"]
    H13_verdict = (f"INFORMATIONAL — 2D recall = {H13_recall*100:.1f}% "
                    f"({'high (≥55%)' if H13_recall >= RECALL_HIGH else 'moderate' if H13_recall >= 0.30 else 'low'})")

    return {
        "n_samples": len(cache),
        "n_combos": len(summaries),
        "selection_verdict": sweep_record["selection_verdict"],
        "regression": regression,
        "best": best,
        "best_per_sample_distance": {"by_bin": by_bin, "by_bin_rates": bin_rates},
        "M_within_split": {
            "n_det": n_det, "n_undet": n_undet,
            "M_in_det": M_in_det, "M_in_undet": M_in_undet,
            "miss_in_det": miss_in_det, "miss_in_undet": miss_in_undet,
            "M_rate_within_detected": (M_in_det / n_det) if n_det else 0.0,
            "M_rate_within_undetected": (M_in_undet / n_undet) if n_undet else 0.0,
        },
        "far_M_rate_30plus": far_M_30plus,
        "hypotheses": {
            "H10_n_proposals_in_band": {"verdict": H10_verdict,
                                          "best_n_proposals": best["mean_n_proposals"]},
            "H11_M_rate_above_50": {"verdict": H11_verdict,
                                      "best_M_rate": best["mean_M_rate"]},
            "H12_far_M_above_40": {"verdict": H12_verdict,
                                     "far_M_rate_30plus": far_M_30plus},
            "H13_2d_recall_upper_bound": {"verdict": H13_verdict,
                                            "best_2d_recall": H13_recall},
        },
        "comparison_to_baselines": {
            "M_rate":     {"w1_5": W1_5_M, "beta1": BETA1_M, "option5": best["mean_M_rate"]},
            "L_rate":     {"w1_5": 0.054, "beta1": BETA1_L, "option5": best["mean_L_rate"]},
            "D_rate":     {"w1_5": 0.356, "beta1": BETA1_D, "option5": best["mean_D_rate"]},
            "miss_rate":  {"w1_5": W1_5_MISS, "beta1": BETA1_MISS, "option5": best["mean_miss_rate"]},
            "n_proposals":{"w1_5": W1_5_N_CL, "beta1": BETA1_N_CL, "option5": best["mean_n_proposals"]},
            "timing":     {"w1_5": W1_5_T, "beta1": BETA1_T, "option5": best["median_timing_total_s"]},
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
    elif M_PLATEAU_LO <= M <= M_PLATEAU_HI:
        m_v = "PLATEAU"
    else:
        m_v = "FAIL"

    n = best["mean_n_proposals"]
    if n <= N_PROP_PASS:
        n_v = "PASS"
    elif n <= N_PROP_WARN:
        n_v = "WARN"
    else:
        n_v = "FAIL"

    t = best["median_timing_total_s"]
    if t < TIMING_PASS:
        t_v = "PASS"
    elif t < TIMING_WARN:
        t_v = "WARN"
    else:
        t_v = "FAIL"

    recall = best["mean_2d_recall"]
    r_v = "HIGH" if recall >= RECALL_HIGH else ("MODERATE" if recall >= 0.30 else "LOW")

    if m_v == "STRONG" and n_v == "PASS" and t_v == "PASS":
        branch = ("Option 5 STRONG. Detection-guided 가설 confirmed. "
                  "config locked. W2 (Coverage reliability) 진행 가능.")
    elif m_v == "PARTIAL":
        branch = ("Option 5 PARTIAL. β1 ceiling 깨졌지만 STRONG 미달. "
                  "W2 baseline 또는 hyperparameter 추가 튜닝.")
    elif m_v == "PLATEAU":
        branch = ("Option 5 PLATEAU. β1과 차이 없음 — 2D detection이 그래도 "
                  "ceiling을 못 깸. CenterPoint γ 검토 또는 학습된 모델 escalate.")
    elif m_v == "FAIL":
        branch = ("Option 5 FAIL. 2D detection 도입이 오히려 악화. "
                  "CenterPoint γ 즉시 escalate.")
    else:
        branch = "unknown verdict — review tables manually"

    return {
        "cond_M_rate": {"value": M, "verdict": m_v,
                         "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL,
                                         "PLATEAU_lo": M_PLATEAU_LO,
                                         "PLATEAU_hi": M_PLATEAU_HI}},
        "cond_n_proposals": {"value": n, "verdict": n_v,
                               "thresholds": {"PASS": N_PROP_PASS, "WARN": N_PROP_WARN}},
        "cond_timing": {"value": t, "verdict": t_v,
                         "thresholds": {"PASS": TIMING_PASS, "WARN": TIMING_WARN}},
        "cond_2d_recall": {"value": recall, "verdict": r_v,
                             "thresholds": {"HIGH": RECALL_HIGH}},
        "branch": branch,
    }


# ---------- figures ----------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_M_heatmap_expand_minpoints(summaries, out):
    """3 panels (one per min_depth), each = expand_ratio × min_points heatmap of M."""
    expands = sorted({s["combo"]["expand_ratio"] for s in summaries})
    min_pts = sorted({s["combo"]["min_points_per_frustum"] for s in summaries})
    min_depths = sorted({s["combo"]["min_depth"] for s in summaries})
    table = {(s["combo"]["expand_ratio"], s["combo"]["min_points_per_frustum"],
              s["combo"]["min_depth"]): s["mean_M_rate"] for s in summaries}
    fig, axes = plt.subplots(1, len(min_depths), figsize=(4.5 * len(min_depths), 4),
                              squeeze=False)
    vmin = min(s["mean_M_rate"] for s in summaries) * 100
    vmax = max(s["mean_M_rate"] for s in summaries) * 100
    for i, md in enumerate(min_depths):
        ax = axes[0][i]
        Z = np.full((len(expands), len(min_pts)), np.nan)
        for r, er in enumerate(expands):
            for c, mp in enumerate(min_pts):
                if (er, mp, md) in table:
                    Z[r, c] = table[(er, mp, md)] * 100
        im = ax.imshow(Z, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(min_pts)))
        ax.set_xticklabels(min_pts)
        ax.set_yticks(range(len(expands)))
        ax.set_yticklabels([f"{e:g}" for e in expands])
        ax.set_xlabel("min_points_per_frustum")
        if i == 0:
            ax.set_ylabel("expand_ratio")
        ax.set_title(f"min_depth={md:g}m")
        for r in range(len(expands)):
            for c in range(len(min_pts)):
                v = Z[r, c]
                if not np.isnan(v):
                    colour = "white" if v > (vmin + vmax) / 2 else "black"
                    ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                            color=colour, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Option 5 — mean M_rate (%)", fontsize=11)
    _save(fig, out)


def fig_n_proposals_distribution(summaries, best, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ns = [s["mean_n_proposals"] for s in summaries]
    Ms = [s["mean_M_rate"] * 100 for s in summaries]
    ax.scatter(ns, Ms, s=24, alpha=0.6, color="#666666", label="combos")
    ax.scatter([best["mean_n_proposals"]], [best["mean_M_rate"] * 100],
               s=160, edgecolor="red", facecolor="none", linewidth=2, label="best")
    ax.scatter([BETA1_N_CL], [BETA1_M * 100], s=140, marker="x", color="orange",
               label=f"β1 baseline (n={BETA1_N_CL:.0f})")
    ax.scatter([W1_5_N_CL], [W1_5_M * 100], s=140, marker="x", color="blue",
               label=f"W1.5 baseline (n={W1_5_N_CL:.0f})")
    ax.axvline(N_PROP_PASS, color="green", ls="--", lw=1, label=f"PASS≤{N_PROP_PASS}")
    ax.axvline(N_PROP_WARN, color="orange", ls="--", lw=1, label=f"WARN≤{N_PROP_WARN}")
    ax.axhline(M_STRONG * 100, color="green", ls=":", lw=1, label=f"M STRONG≥{M_STRONG*100:.0f}%")
    ax.axhline(M_PARTIAL * 100, color="orange", ls=":", lw=1, label=f"M PARTIAL≥{M_PARTIAL*100:.0f}%")
    ax.set_xlabel("mean # proposals per frame")
    ax.set_ylabel("mean M_rate (%)")
    ax.set_title("Option 5 sweep vs baselines")
    ax.set_xscale("symlog", linthresh=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_M_within_vs_outside(agg, out):
    s = agg["M_within_split"]
    cats = ["detected\nGTs", "undetected\nGTs"]
    M_rates = [s["M_rate_within_detected"] * 100, s["M_rate_within_undetected"] * 100]
    counts = [s["n_det"], s["n_undet"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cats, M_rates, color=["#117733", "#999999"])
    for bar, m, n in zip(bars, M_rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.5,
                f"{m:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("M_rate (%)")
    ax.set_title("M_rate split by 2D detection coverage of GT")
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
    ax.set_title("Option 5 best — distance-stratified")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_2d_recall_by_distance(by_bin_rates, out):
    bins = DISTANCE_BIN_LABELS
    rec = [by_bin_rates[b]["two_d_recall"] * 100 for b in bins]
    n_GT = [by_bin_rates[b]["n_GT"] for b in bins]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, rec, color="#4477AA")
    for i, r in enumerate(rec):
        ax.text(x[i], r + 1, f"{r:.1f}%\nn={n_GT[i]}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("2D detection recall (%)")
    ax.set_title("YOLO-World 2D recall by distance — fundamental upper bound on Option 5")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_pipeline_progression(comparison, out):
    metrics = ["M_rate", "miss_rate", "L_rate", "D_rate"]
    stages = ["W1.5", "β1", "Option 5"]
    keys = ["w1_5", "beta1", "option5"]
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
    ax.set_title("Pipeline progression: W1.5 → β1 → Option 5")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_timing_breakdown(summaries, best, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [s["median_timing_total_s"] for s in summaries]
    ax.hist(ts, bins=12, color="#CC6677", edgecolor="white")
    ax.axvline(best["median_timing_total_s"], color="red", ls="--",
               label=f"best={best['median_timing_total_s']:.2f}s")
    ax.axvline(TIMING_PASS, color="green", ls=":", label=f"PASS<{TIMING_PASS}s")
    ax.axvline(TIMING_WARN, color="orange", ls=":", label=f"WARN<{TIMING_WARN}s")
    ax.set_xlabel("median per-sample pipeline timing (s)")
    ax.set_ylabel("# combos")
    ax.set_title(f"Option 5 sweep timing ({len(ts)} combos)")
    ax.legend(fontsize=8)
    _save(fig, out)


# ---------- top-3 ----------

def _top3(agg):
    cmp = agg["comparison_to_baselines"]
    obs = []
    obs.append(
        f"M_rate progression: W1.5 {cmp['M_rate']['w1_5']*100:.1f}% → "
        f"β1 {cmp['M_rate']['beta1']*100:.1f}% → "
        f"**Option 5 {cmp['M_rate']['option5']*100:.1f}%** "
        f"(Δ vs β1 = {(cmp['M_rate']['option5'] - cmp['M_rate']['beta1'])*100:+.1f} pts). "
        f"n_proposals collapsed from {cmp['n_proposals']['beta1']:.0f} (β1) to "
        f"{cmp['n_proposals']['option5']:.1f} (Option 5)."
    )
    s = agg["M_within_split"]
    obs.append(
        f"M_rate split by 2D detection: "
        f"**{s['M_rate_within_detected']*100:.1f}%** within 2D-detected GTs "
        f"(n={s['n_det']}) vs "
        f"**{s['M_rate_within_undetected']*100:.1f}%** within undetected GTs "
        f"(n={s['n_undet']}). "
        f"Method's M_rate is bounded above by 2D recall — "
        f"undetected GTs are unrecoverable by construction."
    )
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    near = (rates["0-10m"]["M_rate"] + rates["10-20m"]["M_rate"]) / 2 * 100
    far = agg["far_M_rate_30plus"] * 100
    obs.append(
        f"Distance: 0–20m mean M = {near:.1f}%, 30m+ mean M = **{far:.1f}%** "
        f"(β1 was 19.3%). 2D recall by bin: 0–10m "
        f"{rates['0-10m']['two_d_recall']*100:.0f}% / "
        f"10–20m {rates['10-20m']['two_d_recall']*100:.0f}% / "
        f"30–50m {rates['30-50m']['two_d_recall']*100:.0f}% / "
        f"50m+ {rates['50m+']['two_d_recall']*100:.0f}%. "
        f"Far range improvement is bottlenecked by 2D detection coverage."
    )
    return obs


# ---------- report ----------

def _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, out_path):
    L = []
    L.append("# Option 5 — 2D detection-guided clustering")
    L.append("")
    L.append(f"- Samples: **{agg['n_samples']}** (W1.5 set)")
    L.append(f"- Sweep: **{agg['n_combos']}** combos "
             f"(3 expand_ratio × 3 min_points × 3 min_depth)")
    L.append(f"- Selection verdict: **{agg['selection_verdict']}**")
    if regression.get("passes"):
        L.append(f"- β1 regression spot-check ({regression['checked_samples']} samples, "
                 f"no frustum): mean_M={regression['mean_M_rate']:.4f}, "
                 f"mean_n_cl={regression['mean_n_clusters']:.4f} → **PASS**")
    else:
        L.append(f"- β1 regression: **FAIL**")
    L.append(f"- Locked configs: pillar `{sweep_record['fixed_pillar_config']}`, "
             f"HDBSCAN `{sweep_record['fixed_hdbscan_config']}`")
    L.append("")

    # Hypothesis
    L.append("## Hypothesis check")
    L.append("")
    h = agg["hypotheses"]
    L.append(f"- **H10** — n_proposals in [7, 30]: "
             f"**{h['H10_n_proposals_in_band']['verdict']}** "
             f"(best n_prop = {h['H10_n_proposals_in_band']['best_n_proposals']:.1f})")
    L.append(f"- **H11** — H10 → M_rate ≥ 0.50: "
             f"**{h['H11_M_rate_above_50']['verdict']}** "
             f"(best M_rate = {h['H11_M_rate_above_50']['best_M_rate']*100:.1f}%)")
    L.append(f"- **H12** — 30m+ M_rate ≥ 0.40: "
             f"**{h['H12_far_M_above_40']['verdict']}** "
             f"(far_M = {h['H12_far_M_above_40']['far_M_rate_30plus']*100:.1f}%)")
    L.append(f"- **H13** — {h['H13_2d_recall_upper_bound']['verdict']}")
    L.append("")

    # Sweep
    L.append("## Sweep — 27 combos")
    L.append("")
    L.append("![M_heatmap](figures/M_rate_heatmap_expand_minpoints.png)")
    L.append("")
    L.append("Three side-by-side heatmaps (one per min_depth), expand_ratio × min_points axes.")
    L.append("")
    L.append("![n_proposals](figures/n_proposals_distribution.png)")
    L.append("")
    L.append("Each combo plotted in (n_proposals, M_rate) space alongside W1.5 raw / β1 baselines. The best combo is circled red.")
    L.append("")

    L.append("### Sweep table (sorted by M_rate)")
    L.append("")
    L.append("| expand | min_pts | min_depth | M% | miss% | n_prop | 2D_recall% | timing s | n_ok |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(summaries, key=lambda x: -x["mean_M_rate"]):
        c = s["combo"]
        L.append(f"| {c['expand_ratio']:g} | {c['min_points_per_frustum']} | "
                 f"{c['min_depth']:g} | "
                 f"{s['mean_M_rate']*100:.1f} | {s['mean_miss_rate']*100:.1f} | "
                 f"{s['mean_n_proposals']:.1f} | "
                 f"{s['mean_2d_recall']*100:.1f} | "
                 f"{s['median_timing_total_s']:.2f} | "
                 f"{s['n_samples_succeeded']}/{agg['n_samples']} |")
    L.append("")

    # Best
    bc = best["combo"]
    L.append("## Best combo")
    L.append("")
    L.append(f"- expand_ratio: **{bc['expand_ratio']:g}**")
    L.append(f"- min_points_per_frustum: **{bc['min_points_per_frustum']}**")
    L.append(f"- min_depth: **{bc['min_depth']:g}** m")
    L.append("")

    # Pipeline comparison
    L.append("## Pipeline comparison: W1.5 raw → β1 best → Option 5 best")
    L.append("")
    cmp = agg["comparison_to_baselines"]
    L.append("| metric | W1.5 raw | β1 best | Option 5 best | Δ vs β1 |")
    L.append("|---|---|---|---|---|")
    for m in ["M_rate", "L_rate", "D_rate", "miss_rate"]:
        L.append(f"| {m} | {cmp[m]['w1_5']*100:.1f}% | {cmp[m]['beta1']*100:.1f}% | "
                 f"{cmp[m]['option5']*100:.1f}% | "
                 f"{(cmp[m]['option5'] - cmp[m]['beta1'])*100:+.1f} |")
    L.append(f"| n_proposals | {cmp['n_proposals']['w1_5']:.1f} | "
             f"{cmp['n_proposals']['beta1']:.1f} | "
             f"{cmp['n_proposals']['option5']:.1f} | "
             f"{cmp['n_proposals']['option5'] - cmp['n_proposals']['beta1']:+.1f} |")
    L.append(f"| timing (s) | {cmp['timing']['w1_5']:.3f} | "
             f"{cmp['timing']['beta1']:.3f} | "
             f"{cmp['timing']['option5']:.3f} | "
             f"{cmp['timing']['option5'] - cmp['timing']['beta1']:+.3f} |")
    L.append("")
    L.append("![pipeline_progression](figures/before_after_pipeline_comparison.png)")
    L.append("")
    L.append("(Note: Option 5 timing is per-sample frustum+pillar+HDBSCAN+matching; YOLO-World inference is amortised across the sweep via cache and not included in the per-sample number.)")
    L.append("")

    # M within vs outside
    L.append("## M_rate split by 2D detection coverage")
    L.append("")
    L.append("![M_within](figures/M_within_vs_outside_detected_GT.png)")
    L.append("")
    s = agg["M_within_split"]
    L.append(f"- Within 2D-detected GTs (n={s['n_det']}): M_rate = **{s['M_rate_within_detected']*100:.1f}%**")
    L.append(f"- Within undetected GTs (n={s['n_undet']}): M_rate = **{s['M_rate_within_undetected']*100:.1f}%**")
    L.append("")
    L.append("Detection-guided clustering structurally cannot recover undetected GTs; the within-detected number is the *real* method ceiling.")
    L.append("")

    # Distance
    L.append("## Distance-stratified (best combo)")
    L.append("")
    L.append("![distance](figures/distance_stratified_best.png)")
    L.append("")
    rates = agg["best_per_sample_distance"]["by_bin_rates"]
    L.append("| bin | n_GT | M% | L% | D% | miss% | 2D_recall% |")
    L.append("|---|---|---|---|---|---|---|")
    for b in DISTANCE_BIN_LABELS:
        r = rates[b]
        if r["n_GT"] == 0:
            continue
        L.append(f"| {b} | {r['n_GT']} | {r['M_rate']*100:.1f} | "
                 f"{r['L_rate']*100:.1f} | {r['D_rate']*100:.1f} | "
                 f"{r['miss_rate']*100:.1f} | "
                 f"{r['two_d_recall']*100:.1f} |")
    L.append("")
    L.append("![2d_recall_distance](figures/2D_detection_recall_by_distance.png)")
    L.append("")
    L.append("YOLO-World 2D recall by distance bin. Method's distance-stratified M is bounded above by this curve.")
    L.append("")
    L.append("![timing](figures/timing_breakdown.png)")
    L.append("")

    # Decision
    L.append("## Decision")
    L.append("")
    a = decision["cond_M_rate"]
    b = decision["cond_n_proposals"]
    c = decision["cond_timing"]
    d = decision["cond_2d_recall"]
    L.append(f"- **cond_M_rate** = {a['value']*100:.2f}% → **{a['verdict']}** "
             f"(STRONG ≥ {a['thresholds']['STRONG']*100:.0f}%, "
             f"PARTIAL ≥ {a['thresholds']['PARTIAL']*100:.0f}%, "
             f"PLATEAU in [{a['thresholds']['PLATEAU_lo']*100:.0f}, "
             f"{a['thresholds']['PLATEAU_hi']*100:.0f}]%)")
    L.append(f"- **cond_n_proposals** = {b['value']:.2f} → **{b['verdict']}** "
             f"(PASS ≤ {b['thresholds']['PASS']}, WARN ≤ {b['thresholds']['WARN']})")
    L.append(f"- **cond_timing** = {c['value']:.3f}s → **{c['verdict']}** "
             f"(PASS < {c['thresholds']['PASS']}s, WARN < {c['thresholds']['WARN']}s)")
    L.append(f"- **cond_2d_recall** = {d['value']*100:.2f}% → **{d['verdict']}** "
             f"(HIGH ≥ {d['thresholds']['HIGH']*100:.0f}%) — informational ceiling")
    L.append("")
    L.append(f"### Branch fire — {decision['branch']}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"cond_M_rate      = {a['value']:.4f}  → {a['verdict']}")
    L.append(f"cond_n_proposals = {b['value']:.4f}  → {b['verdict']}")
    L.append(f"cond_timing      = {c['value']:.4f}s → {c['verdict']}")
    L.append(f"cond_2d_recall   = {d['value']:.4f}  → {d['verdict']}")
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
    L.append("- **STRONG**: lock Option 5 config, advance to W2.")
    L.append("- **PARTIAL**: W2 baseline 또는 추가 hyperparameter 튜닝.")
    L.append("- **PLATEAU**: 2D detection alone insufficient — escalate to CenterPoint γ or learned proposal stage.")
    L.append("- **FAIL**: detection-guided 자체가 wrong axis — escalate to γ immediately.")
    L.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(L))


def render_all_option5(summaries, best, sweep_record, regression, agg,
                       figures_dir, report_path):
    fig_M_heatmap_expand_minpoints(summaries, osp.join(figures_dir, "M_rate_heatmap_expand_minpoints.png"))
    fig_n_proposals_distribution(summaries, best, osp.join(figures_dir, "n_proposals_distribution.png"))
    fig_M_within_vs_outside(agg, osp.join(figures_dir, "M_within_vs_outside_detected_GT.png"))
    fig_distance_stratified_best(agg["best_per_sample_distance"]["by_bin_rates"],
                                  osp.join(figures_dir, "distance_stratified_best.png"))
    fig_2d_recall_by_distance(agg["best_per_sample_distance"]["by_bin_rates"],
                               osp.join(figures_dir, "2D_detection_recall_by_distance.png"))
    fig_pipeline_progression(agg["comparison_to_baselines"],
                              osp.join(figures_dir, "before_after_pipeline_comparison.png"))
    fig_timing_breakdown(summaries, best, osp.join(figures_dir, "timing_breakdown.png"))
    decision = _evaluate_decision(agg)
    observations = _top3(agg)
    _write_report(summaries, best, sweep_record, regression, agg, decision,
                  observations, figures_dir, report_path)
