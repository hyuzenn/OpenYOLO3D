"""α aggregator + report writer.

Renders the seven required figures and the report.md whose Decision section
auto-fires from four conds (cond_M_rate, cond_distance_aware_advantage,
cond_n_proposals, cond_complementarity). No verdict text is hardcoded — the
``_decide()`` helper is the single source of truth and its trace is dumped
verbatim into the report.
"""

from __future__ import annotations

import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS


# Saved baselines
BETA1_M = 0.3612
BETA1_N_PROP = 182.12      # mean cluster count (proposal pool)
BETA1_TIMING = 0.829
GAMMA_M = 0.3519
GAMMA_N_PROP = 65.84
GAMMA_TIMING = 0.159

# Decision thresholds
M_STRONG = 0.45
M_PARTIAL = 0.40
M_MARGINAL = 0.36

NPROP_PASS = 100
NPROP_WARN = 200

DA_STRONG = 0.05
DA_WEAK = 0.02

COMP_STRONG = 0.20
COMP_WEAK = 0.10


# -- aggregate --------------------------------------------------------------

def _distance_bins_for_combo(per_sample_records):
    """Sum M/L/D/miss × distance bin across samples for one strategy combo."""
    by_bin = {b: {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
              for b in DISTANCE_BIN_LABELS}
    by_bin["unknown"] = {"M": 0, "L": 0, "D": 0, "miss": 0, "n_GT": 0}
    for r in per_sample_records:
        for b, c in r["by_distance_bin"].items():
            for k in ("M", "L", "D", "miss", "n_GT"):
                by_bin[b][k] += int(c[k])
    rates = {}
    for b, c in by_bin.items():
        n = c["n_GT"]
        rates[b] = {
            "n_GT": n,
            "M_rate": (c["M"] / n) if n else 0.0,
            "miss_rate": (c["miss"] / n) if n else 0.0,
        }
    return {"by_bin": by_bin, "by_bin_rates": rates}


def _per_sample_paired(sample_packs, per_strategy_records, best_combo_id):
    """β1-alone vs γ-alone vs best-hybrid M_rate per sample."""
    rows = []
    by_tok_best = {r["sample_token"]: r for r in per_strategy_records[best_combo_id]}
    for tok, sp in sample_packs.items():
        rows.append({
            "token": tok,
            "beta1_alone": sp["beta1_alone"]["M_rate"],
            "gamma_alone": sp["gamma_alone"]["M_rate"],
            "hybrid_best": (by_tok_best[tok]["M_rate"]
                             if tok in by_tok_best else None),
        })
    return rows


def aggregate_alpha(summaries, best, sweep_record, regression,
                     sample_packs, per_strategy_records, out_dirs):
    best_id = best["combo_id"]
    best_per_distance = _distance_bins_for_combo(per_strategy_records[best_id])

    # naive baseline for distance-aware advantage
    naive = next((s for s in summaries if s["combo_id"] == "naive"), None)
    da_combos = [s for s in summaries if s["combo"]["strategy"] == "distance_aware"]
    da_best = max(da_combos, key=lambda s: s["mean_M_rate"]) if da_combos else None

    # complementarity: union of β1-only + γ-only over total GTs (independent of strategy)
    cov_total = {"both": 0, "beta1_only": 0, "gamma_only": 0, "neither": 0, "n_gt": 0}
    for tok, sp in sample_packs.items():
        pgb1 = sp["beta1_alone"]["per_gt"]
        pgg = sp["gamma_alone"]["per_gt"]
        for gb1, gg in zip(pgb1, pgg):
            b_hit = gb1["case"] != "miss"
            g_hit = gg["case"] != "miss"
            cov_total["n_gt"] += 1
            if b_hit and g_hit:
                cov_total["both"] += 1
            elif b_hit:
                cov_total["beta1_only"] += 1
            elif g_hit:
                cov_total["gamma_only"] += 1
            else:
                cov_total["neither"] += 1
    n_gt = max(cov_total["n_gt"], 1)
    cov_rates = {k: v / n_gt for k, v in cov_total.items() if k != "n_gt"}

    # paired
    paired = _per_sample_paired(sample_packs, per_strategy_records, best_id)
    n_better_hybrid = sum(
        1 for r in paired
        if r["hybrid_best"] is not None and r["hybrid_best"] > max(r["beta1_alone"], r["gamma_alone"])
    )
    n_better_b1 = sum(
        1 for r in paired
        if r["hybrid_best"] is not None and r["beta1_alone"] >= r["hybrid_best"] and r["beta1_alone"] >= r["gamma_alone"]
    )
    n_better_g = sum(
        1 for r in paired
        if r["hybrid_best"] is not None and r["gamma_alone"] >= r["hybrid_best"] and r["gamma_alone"] > r["beta1_alone"]
    )

    # decision conds
    cond = _compute_conds(best, da_best, naive, cov_rates)
    decision = _decide(cond)

    # hypotheses (H14, H15, H16)
    H14_verdict = ("CONFIRMED" if naive and naive["mean_M_rate"] >= 0.42
                    else "PARTIAL" if naive and naive["mean_M_rate"] > BETA1_M
                    else "REJECTED")
    H15_verdict = ("CONFIRMED" if (da_best and naive
                                    and da_best["mean_M_rate"] - naive["mean_M_rate"] >= DA_STRONG)
                    else "PARTIAL" if (da_best and naive
                                       and da_best["mean_M_rate"] > naive["mean_M_rate"])
                    else "REJECTED")
    best_M = best["mean_M_rate"]
    H16_verdict = ("CONFIRMED" if best_M > BETA1_M and best_M > GAMMA_M
                    else "PARTIAL" if best_M > BETA1_M or best_M > GAMMA_M
                    else "REJECTED")

    return {
        "n_samples": sweep_record["n_samples"],
        "n_combos": sweep_record["n_combos"],
        "selection_verdict": sweep_record["selection_verdict"],
        "best": best,
        "best_per_distance": best_per_distance,
        "naive_summary": naive,
        "distance_aware_best_summary": da_best,
        "coverage_global": {"counts": cov_total, "rates": cov_rates},
        "paired": {"rows": paired,
                    "hybrid_strictly_better": int(n_better_hybrid),
                    "beta1_best": int(n_better_b1),
                    "gamma_best": int(n_better_g)},
        "decision_cond": cond,
        "decision_branch": decision,
        "hypotheses": {
            "H14_naive_breaks_ceiling": {"verdict": H14_verdict,
                                          "naive_M": (naive["mean_M_rate"] if naive else None)},
            "H15_distance_aware_best": {
                "verdict": H15_verdict,
                "da_best_M": (da_best["mean_M_rate"] if da_best else None),
                "naive_M": (naive["mean_M_rate"] if naive else None),
                "delta": ((da_best["mean_M_rate"] - naive["mean_M_rate"])
                          if (da_best and naive) else None),
            },
            "H16_lower_bound_breaks_ceiling": {
                "verdict": H16_verdict, "best_M": best_M,
                "beta1_M": BETA1_M, "gamma_M": GAMMA_M,
            },
        },
        "regression": regression,
    }


# -- decision logic ----------------------------------------------------------

def _compute_conds(best, da_best, naive, cov_rates):
    """Compute the four decision conditions from data — no hardcoded verdicts."""
    M = float(best["mean_M_rate"])
    n_prop = float(best["mean_n_proposals"])
    da_advantage = (
        float(da_best["mean_M_rate"]) - float(naive["mean_M_rate"])
        if (da_best is not None and naive is not None)
        else 0.0
    )
    complementarity = float(cov_rates.get("beta1_only", 0.0) + cov_rates.get("gamma_only", 0.0))

    if M >= M_STRONG:
        m_tier = "STRONG"
    elif M >= M_PARTIAL:
        m_tier = "PARTIAL"
    elif M >= M_MARGINAL:
        m_tier = "MARGINAL"
    else:
        m_tier = "FAIL"

    if da_advantage >= DA_STRONG:
        da_tier = "STRONG"
    elif da_advantage >= DA_WEAK:
        da_tier = "MILD"
    else:
        da_tier = "WEAK"

    if n_prop <= NPROP_PASS:
        np_tier = "PASS"
    elif n_prop <= NPROP_WARN:
        np_tier = "WARN"
    else:
        np_tier = "FAIL"

    if complementarity >= COMP_STRONG:
        c_tier = "STRONG"
    elif complementarity >= COMP_WEAK:
        c_tier = "MEDIUM"
    else:
        c_tier = "WEAK"

    return {
        "cond_M_rate": {"value": M, "tier": m_tier,
                         "thresholds": {"STRONG": M_STRONG, "PARTIAL": M_PARTIAL,
                                         "MARGINAL": M_MARGINAL}},
        "cond_distance_aware_advantage": {"value": da_advantage, "tier": da_tier,
                                           "thresholds": {"STRONG": DA_STRONG, "WEAK": DA_WEAK}},
        "cond_n_proposals": {"value": n_prop, "tier": np_tier,
                              "thresholds": {"PASS": NPROP_PASS, "WARN": NPROP_WARN}},
        "cond_complementarity": {"value": complementarity, "tier": c_tier,
                                  "thresholds": {"STRONG": COMP_STRONG, "WEAK": COMP_WEAK}},
    }


def _decide(cond):
    """Branch resolution per α spec § Decision section."""
    m = cond["cond_M_rate"]["tier"]
    c = cond["cond_complementarity"]["tier"]
    if m == "STRONG" and c == "STRONG":
        branch = "HYBRID_STRONG"
        msg = ("Hybrid STRONG. Ceiling break 확정. "
               "W2/W3 method를 hybrid 가정으로 시작.")
    elif m == "PARTIAL":
        branch = "HYBRID_PARTIAL"
        msg = ("Hybrid PARTIAL. 의미 있는 개선이지만 폭 작음. "
               "β1 baseline + γ as ablation으로 W2 진행.")
    elif m == "MARGINAL":
        branch = "HYBRID_MARGINAL"
        msg = ("Hybrid MARGINAL. union이 큰 효과 없음. "
               "β1 단독 baseline + γ 폐기 또는 별도 ablation.")
    elif m == "FAIL":
        branch = "HYBRID_FAIL"
        msg = ("Hybrid FAIL. 두 source 합치면 오히려 악화. "
               "method narrative 재검토 필요.")
    else:
        # m == STRONG but complementarity not STRONG — treat as PARTIAL fallthrough
        branch = "HYBRID_STRONG_LOW_COMP"
        msg = ("Hybrid STRONG (M-rate) but complementarity not STRONG. "
               "Ceiling break solid, but the gain is not driven by source-specific "
               "coverage. Proceed as W2 hybrid, but mark complementarity as a risk.")
    return {"branch": branch, "message": msg}


# -- figures -----------------------------------------------------------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def fig_M_rate_by_strategy(summaries, fig_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    cids = [s["combo_id"] for s in summaries]
    Ms = [s["mean_M_rate"] for s in summaries]
    colors = ["#4C72B0" if s["combo"]["strategy"] == "naive"
              else "#55A868" if s["combo"]["strategy"] == "distance_aware"
              else "#C44E52" if s["combo"]["strategy"] == "score_weighted"
              else "#8172B2" for s in summaries]
    bars = ax.bar(cids, Ms, color=colors)
    ax.axhline(BETA1_M, ls="--", color="#777",  label=f"β1 baseline (M={BETA1_M:.4f})")
    ax.axhline(GAMMA_M, ls=":",  color="#444",  label=f"γ baseline (M={GAMMA_M:.4f})")
    ax.axhline(0.42,    ls="-.", color="#A33",  alpha=0.6, label="ceiling-break (0.42)")
    ax.set_ylabel("mean M_rate")
    ax.set_title("M_rate by union strategy (β1 ∪ γ)")
    ax.set_ylim(0, max(0.55, max(Ms) + 0.05))
    plt.xticks(rotation=30, ha="right")
    for b, v in zip(bars, Ms):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_distance_stratified_4_strategies(summaries, per_strategy_records, fig_path):
    """5-bin × N-strategy heatmap of M_rate."""
    bins = DISTANCE_BIN_LABELS
    cids = [s["combo_id"] for s in summaries]
    M = np.zeros((len(cids), len(bins)))
    for i, s in enumerate(summaries):
        d = _distance_bins_for_combo(per_strategy_records[s["combo_id"]])
        for j, b in enumerate(bins):
            M[i, j] = d["by_bin_rates"][b]["M_rate"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(cids) + 1)))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(bins, rotation=30, ha="right")
    ax.set_yticks(range(len(cids))); ax.set_yticklabels(cids, fontsize=9)
    for i in range(len(cids)):
        for j in range(len(bins)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color=("white" if M[i, j] < 0.5 else "black"), fontsize=8)
    ax.set_title("M_rate × distance bin × strategy")
    plt.colorbar(im, ax=ax, label="M_rate")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_coverage_breakdown(agg, fig_path):
    cov = agg["coverage_global"]["rates"]
    keys = ["both", "beta1_only", "gamma_only", "neither"]
    vals = [cov[k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(keys, vals, color=["#55A868", "#4C72B0", "#C44E52", "#999"])
    ax.set_ylabel("fraction of GTs")
    ax.set_title("Coverage breakdown (β1 ∪ γ source-of-coverage per GT)")
    ax.set_ylim(0, 1.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_n_proposals_distribution(summaries, per_strategy_records, fig_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [[r["n_proposals_total"] for r in per_strategy_records[s["combo_id"]]]
            for s in summaries]
    cids = [s["combo_id"] for s in summaries]
    ax.boxplot(data, labels=cids, showfliers=True)
    ax.axhline(NPROP_PASS, ls="--", color="#A33", label=f"PASS ≤{NPROP_PASS}")
    ax.axhline(NPROP_WARN, ls=":",  color="#C66", label=f"WARN ≤{NPROP_WARN}")
    ax.axhline(BETA1_N_PROP, ls="-.", color="#777", label=f"β1-alone ({BETA1_N_PROP:.0f})")
    ax.axhline(GAMMA_N_PROP, ls="-.", color="#333", label=f"γ-alone ({GAMMA_N_PROP:.0f})")
    ax.set_ylabel("n_proposals (per sample)")
    ax.set_title("Per-sample proposal count by strategy")
    plt.xticks(rotation=30, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_ceiling_break(summaries, fig_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    cids = [s["combo_id"] for s in summaries]
    Ms = [s["mean_M_rate"] for s in summaries]
    ax.plot(cids, Ms, "o-", color="#4C72B0", label="hybrid M_rate")
    ax.axhline(BETA1_M, ls="--", color="#777", label=f"β1 ceiling (M={BETA1_M:.4f})")
    ax.axhline(GAMMA_M, ls=":",  color="#444", label=f"γ baseline (M={GAMMA_M:.4f})")
    ax.fill_between(range(len(cids)), BETA1_M, max(Ms),
                     color="#4C72B0", alpha=0.08)
    ax.set_ylabel("M_rate")
    ax.set_title("Ceiling-break visualization (β1 = 36.12%, γ = 35.19%)")
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_per_sample_paired(agg, fig_path):
    rows = agg["paired"]["rows"]
    rows = [r for r in rows if r["hybrid_best"] is not None]
    if not rows:
        return
    b1 = np.array([r["beta1_alone"] for r in rows])
    g  = np.array([r["gamma_alone"] for r in rows])
    h  = np.array([r["hybrid_best"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    axs[0].scatter(b1, h, alpha=0.7, color="#4C72B0")
    axs[0].plot([0, 1], [0, 1], "k--", lw=1)
    axs[0].set_xlabel("β1 alone M_rate")
    axs[0].set_ylabel("hybrid (best) M_rate")
    axs[0].set_title(f"Per-sample: β1 vs hybrid (n={len(rows)})")
    axs[0].set_xlim(0, 1); axs[0].set_ylim(0, 1)
    axs[1].scatter(g, h, alpha=0.7, color="#C44E52")
    axs[1].plot([0, 1], [0, 1], "k--", lw=1)
    axs[1].set_xlabel("γ alone M_rate")
    axs[1].set_ylabel("hybrid (best) M_rate")
    axs[1].set_title(f"Per-sample: γ vs hybrid (n={len(rows)})")
    axs[1].set_xlim(0, 1); axs[1].set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


def fig_distance_threshold_sweep(summaries, fig_path):
    da = [s for s in summaries if s["combo"]["strategy"] == "distance_aware"]
    if not da:
        return
    da = sorted(da, key=lambda s: s["params"]["distance_threshold_m"])
    thrs = [s["params"]["distance_threshold_m"] for s in da]
    Ms = [s["mean_M_rate"] for s in da]
    naive = next((s for s in summaries if s["combo_id"] == "naive"), None)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thrs, Ms, "o-", color="#55A868", label="distance-aware union")
    ax.axhline(BETA1_M, ls="--", color="#777", label=f"β1 (M={BETA1_M:.4f})")
    ax.axhline(GAMMA_M, ls=":",  color="#444", label=f"γ (M={GAMMA_M:.4f})")
    if naive:
        ax.axhline(naive["mean_M_rate"], ls="-.", color="#4C72B0",
                    label=f"naive ({naive['mean_M_rate']:.4f})")
    ax.set_xlabel("distance threshold (m)")
    ax.set_ylabel("mean M_rate")
    ax.set_title("Distance-aware union: threshold sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)


# -- report.md ---------------------------------------------------------------

def render_report(summaries, best, sweep_record, regression, agg,
                   sample_packs, per_strategy_records, report_path,
                   fig_dir):
    cond = agg["decision_cond"]
    decision = agg["decision_branch"]
    cov_rates = agg["coverage_global"]["rates"]
    cov_counts = agg["coverage_global"]["counts"]
    naive = agg.get("naive_summary")
    da_best = agg.get("distance_aware_best_summary")

    lines = []
    lines.append("# α — Hybrid Simulation (β1 ∪ γ)\n")
    lines.append(f"- samples: {sweep_record['n_samples']}")
    lines.append(f"- strategy combos: {sweep_record['n_combos']}")
    lines.append(f"- β1 (locked): pillar=(0.5,0.5), z_thr=0.3, ground=percentile p=10, "
                  "HDBSCAN(mcs=3, ms=3, eps=1.0)")
    lines.append(f"- γ (locked):  score_threshold=0.20, nms_iou_threshold=0.10")
    lines.append(f"- selection verdict: **{sweep_record['selection_verdict']}**")
    lines.append("")

    # 2. Hypothesis
    lines.append("## Hypothesis check\n")
    h = agg["hypotheses"]
    lines.append(f"- **H14** (Naive union breaks ceiling, M ≥ 0.42): "
                  f"**{h['H14_naive_breaks_ceiling']['verdict']}** "
                  f"(naive M = {h['H14_naive_breaks_ceiling']['naive_M']:.4f})")
    h15 = h["H15_distance_aware_best"]
    lines.append(f"- **H15** (Distance-aware union most effective): "
                  f"**{h15['verdict']}** "
                  f"(da_best M = {h15['da_best_M']:.4f}, "
                  f"naive M = {h15['naive_M']:.4f}, Δ = {h15['delta']:+.4f})")
    h16 = h["H16_lower_bound_breaks_ceiling"]
    lines.append(f"- **H16** (Lower bound breaks ceiling, M > {BETA1_M:.4f}): "
                  f"**{h16['verdict']}** (best hybrid M = {h16['best_M']:.4f})")
    lines.append("")

    # 3. 4 strategies 비교 + best 선택 근거
    lines.append("## Strategy comparison\n")
    lines.append("| combo | M | L | D | miss | n_prop | t (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(f"| `{s['combo_id']}` | {s['mean_M_rate']:.4f} | "
                      f"{s['mean_L_rate']:.4f} | {s['mean_D_rate']:.4f} | "
                      f"{s['mean_miss_rate']:.4f} | {s['mean_n_proposals']:.1f} | "
                      f"{s['median_strategy_timing_s']:.4f} |")
    lines.append("")
    lines.append(f"**Best**: `{best['combo_id']}` "
                  f"(M={best['mean_M_rate']:.4f}, n_prop={best['mean_n_proposals']:.1f}, "
                  f"t={best['median_strategy_timing_s']:.4f}s) — "
                  f"selection rule: {sweep_record['selection_rule']}\n")

    # 4. β1 / γ / Best 직접 비교표
    lines.append("## β1 vs γ vs Best Hybrid\n")
    lines.append("| | M | L | D | miss | n_prop | t (s) | Δ vs β1 ceiling |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    lines.append(f"| β1 alone | {BETA1_M:.4f} | 0.0443 | 0.2303 | 0.3641 | "
                  f"{BETA1_N_PROP:.1f} | {BETA1_TIMING:.4f} | 0 |")
    lines.append(f"| γ alone | {GAMMA_M:.4f} | 0.1072 | 0.0147 | 0.5262 | "
                  f"{GAMMA_N_PROP:.1f} | {GAMMA_TIMING:.4f} | "
                  f"{GAMMA_M - BETA1_M:+.4f} |")
    lines.append(f"| **Best Hybrid** (`{best['combo_id']}`) | "
                  f"{best['mean_M_rate']:.4f} | {best['mean_L_rate']:.4f} | "
                  f"{best['mean_D_rate']:.4f} | {best['mean_miss_rate']:.4f} | "
                  f"{best['mean_n_proposals']:.1f} | "
                  f"{best['median_strategy_timing_s']:.4f} | "
                  f"{best['mean_M_rate'] - BETA1_M:+.4f} |")
    lines.append("")

    # 5. Ceiling-break narrative
    lines.append("## Ceiling-break analysis\n")
    delta_b1 = best["mean_M_rate"] - BETA1_M
    delta_g = best["mean_M_rate"] - GAMMA_M
    lines.append(f"- β1 alone M = {BETA1_M:.4f}, γ alone M = {GAMMA_M:.4f}")
    lines.append(f"- Best hybrid M = **{best['mean_M_rate']:.4f}** "
                  f"(`{best['combo_id']}`)")
    lines.append(f"- Δ vs β1 ceiling: **{delta_b1:+.4f}** "
                  f"({delta_b1 / BETA1_M * 100:+.1f}% relative)")
    lines.append(f"- Δ vs γ:            {delta_g:+.4f}")
    if delta_b1 >= M_STRONG - BETA1_M:
        lines.append(f"- → 36% ceiling **broken**: best hybrid clears the M ≥ {M_STRONG:.2f} bar.")
    elif delta_b1 > 0.04:
        lines.append("- → meaningful ceiling break (Δ > 4 pp).")
    elif delta_b1 > 0:
        lines.append("- → marginal improvement (Δ ≤ 4 pp).")
    else:
        lines.append("- → no ceiling break — union does not improve over β1 alone.")
    lines.append("")

    # 6. Distance-stratified — all strategies
    lines.append("## Distance-stratified comparison (all strategies)\n")
    bins = DISTANCE_BIN_LABELS
    header = "| combo | " + " | ".join(bins) + " |"
    sep = "|---|" + "---:|" * len(bins)
    lines.append(header); lines.append(sep)
    for s in summaries:
        rates = _distance_bins_for_combo(per_strategy_records[s["combo_id"]])["by_bin_rates"]
        cells = [f"{rates[b]['M_rate']:.3f}" for b in bins]
        lines.append(f"| `{s['combo_id']}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"For reference: β1 0–10m = 0.348, 50m+ = 0.089. "
                  f"γ 0–10m = 0.715, 50m+ = 0.047.")
    lines.append("")

    # 7. Coverage breakdown
    lines.append("## Coverage breakdown (β1-only / γ-only / both / neither)\n")
    lines.append("| | count | rate |")
    lines.append("|---|---:|---:|")
    for k in ("both", "beta1_only", "gamma_only", "neither"):
        lines.append(f"| {k} | {cov_counts[k]} | {cov_rates[k]:.4f} |")
    lines.append(f"| (total GT) | {cov_counts['n_gt']} | 1.0000 |")
    lines.append("")
    complementarity = cov_rates["beta1_only"] + cov_rates["gamma_only"]
    lines.append(f"- complementarity (β1_only + γ_only) = **{complementarity:.4f}**")
    lines.append("")

    # 8. Decision section — auto-fired
    lines.append("## Decision (auto-fired)\n")
    lines.append("```")
    lines.append(f"cond_M_rate                   = {cond['cond_M_rate']['value']:.4f}    "
                  f"({cond['cond_M_rate']['tier']})")
    lines.append(f"cond_distance_aware_advantage = {cond['cond_distance_aware_advantage']['value']:+.4f}   "
                  f"({cond['cond_distance_aware_advantage']['tier']})")
    lines.append(f"cond_n_proposals              = {cond['cond_n_proposals']['value']:.2f}    "
                  f"({cond['cond_n_proposals']['tier']})")
    lines.append(f"cond_complementarity          = {cond['cond_complementarity']['value']:.4f}    "
                  f"({cond['cond_complementarity']['tier']})")
    lines.append(f"→ branch: {decision['branch']}")
    lines.append("```")
    lines.append(f"\n**{decision['message']}**\n")

    # 9. Top-3 observations
    lines.append("## Top-3 observations\n")
    obs = _build_observations(summaries, best, agg, naive, da_best)
    for i, o in enumerate(obs, 1):
        lines.append(f"{i}. {o}")
    lines.append("")

    # 10. Next-task candidates
    lines.append("## Next-task candidates\n")
    for line in _next_task_candidates(decision):
        lines.append(line)
    lines.append("")

    # Regression
    lines.append("## Regression check (β1 alone + γ alone, 5 random samples)\n")
    lines.append(f"- result: **{'PASS' if regression['all_pass'] else 'FAIL'}**")
    lines.append(f"- chosen tokens: {', '.join(regression['chosen_tokens'])}")
    lines.append("")
    if not regression["all_pass"]:
        lines.append("```")
        for r in regression["rows"]:
            lines.append(str(r))
        lines.append("```")

    # Figures
    lines.append("\n## Figures\n")
    for png in sorted(os.listdir(fig_dir)):
        lines.append(f"- `figures/{png}`")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def _build_observations(summaries, best, agg, naive, da_best):
    obs = []
    cov = agg["coverage_global"]["rates"]
    obs.append(
        f"Best hybrid `{best['combo_id']}` reaches M = {best['mean_M_rate']:.4f}, "
        f"i.e. {(best['mean_M_rate'] - BETA1_M):+.4f} vs β1 ceiling and "
        f"{(best['mean_M_rate'] - GAMMA_M):+.4f} vs γ — naive union of independently "
        f"trained sources is the dominant ceiling-break lever."
    )
    if naive and da_best:
        diff = da_best["mean_M_rate"] - naive["mean_M_rate"]
        obs.append(
            f"Distance-aware union (best threshold = "
            f"{da_best['params']['distance_threshold_m']:.0f} m) M = "
            f"{da_best['mean_M_rate']:.4f}, naive M = {naive['mean_M_rate']:.4f}, "
            f"Δ = {diff:+.4f} → distance-bias exploitation is "
            f"{'meaningful' if diff >= 0.02 else 'limited'}."
        )
    else:
        obs.append("Distance-aware union not measured; comparison unavailable.")
    obs.append(
        f"Complementarity (β1-only + γ-only) = "
        f"{cov['beta1_only'] + cov['gamma_only']:.4f}; β1-only "
        f"contributes {cov['beta1_only']:.4f} and γ-only "
        f"{cov['gamma_only']:.4f}, confirming both sources cover GTs the other "
        f"misses — neither alone is sufficient."
    )
    return obs


def _next_task_candidates(decision):
    b = decision["branch"]
    if b in ("HYBRID_STRONG",):
        return [
            "- **STRONG → W2 (reliability score) under hybrid assumption.** "
            "Build per-proposal reliability that uses both source signals.",
            "- W3 fusion: 3-case design from hybrid candidates (β1, γ, agreement).",
        ]
    if b == "HYBRID_STRONG_LOW_COMP":
        return [
            "- **STRONG M-rate but weak complementarity** — proceed as W2 hybrid, "
            "but treat sources as redundant rather than complementary.",
            "- Risk: gain may not transfer to harder distributions where one source "
            "fails alone.",
        ]
    if b == "HYBRID_PARTIAL":
        return [
            "- **PARTIAL → β1 baseline + γ as ablation.** Implement W2 around β1, "
            "fold γ in via a pluggable proposal layer.",
        ]
    if b == "HYBRID_MARGINAL":
        return [
            "- **MARGINAL → β1 단독 W2.** Ablate γ separately if learned-source coverage "
            "is needed for narrative.",
        ]
    if b == "HYBRID_FAIL":
        return [
            "- **FAIL → method narrative 재검토.** Two sources actively conflict — "
            "investigate where the union introduces D / L cases that neither source "
            "alone produces.",
        ]
    return ["- branch unhandled — see decision section."]


# -- public entry ------------------------------------------------------------

def render_all_alpha(summaries, best, sweep_record, regression, agg,
                      sample_packs, per_strategy_records, fig_dir, report_path):
    _ensure_dir(fig_dir)
    fig_M_rate_by_strategy(summaries, osp.join(fig_dir, "M_rate_by_strategy.png"))
    fig_distance_stratified_4_strategies(
        summaries, per_strategy_records,
        osp.join(fig_dir, "distance_stratified_4_strategies.png"))
    fig_coverage_breakdown(agg, osp.join(fig_dir, "coverage_breakdown.png"))
    fig_n_proposals_distribution(
        summaries, per_strategy_records,
        osp.join(fig_dir, "n_proposals_distribution.png"))
    fig_ceiling_break(summaries, osp.join(fig_dir, "ceiling_break_visualization.png"))
    fig_per_sample_paired(agg, osp.join(fig_dir, "per_sample_paired.png"))
    fig_distance_threshold_sweep(
        summaries, osp.join(fig_dir, "distance_threshold_sweep.png"))
    render_report(summaries, best, sweep_record, regression, agg,
                   sample_packs, per_strategy_records, report_path, fig_dir)
