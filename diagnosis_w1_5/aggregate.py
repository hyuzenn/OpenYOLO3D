"""W1.5 unified report.

Combines Phase A/B/C verdicts into a single report.md plus a unified branch
decision. The decision text is fired purely from the (B verdict, C verdict)
pair — no hardcoded recommendations.
"""

from __future__ import annotations

import json
import os.path as osp


def _unified_branch(b_verdict: str, c_verdict: str) -> tuple[str, str]:
    pass_b = b_verdict in ("B1_PASS", "B2_SOFT_PASS")
    if pass_b and c_verdict == "C1_UNIFORM_OK":
        return ("PASS", "HDBSCAN config locked — proceed to W2 (Coverage reliability).")
    if pass_b and c_verdict == "C2_DISTANCE_GAP":
        return ("CONDITIONAL_PASS",
                "Near-range OK, far-range fails. Run W1.6 distance-aware clustering "
                "before W2 if far-range performance matters.")
    if b_verdict == "B3_FAIL" and c_verdict == "C2_DISTANCE_GAP":
        return ("DISTANCE_GAP_DOMINANT",
                "Distance gap is the dominant failure mode. Distance-aware clustering "
                "is the candidate root fix.")
    if b_verdict == "B3_FAIL" and c_verdict == "C3_UNIFORM_FAIL":
        return ("HDBSCAN_LIMIT_CONFIRMED",
                "HDBSCAN does not work as proposal generator outdoors at any distance. "
                "Escalate to CenterPoint fallback after advisor sign-off.")
    if pass_b and c_verdict == "C3_UNIFORM_FAIL":
        # contradictory — selection found a config but stratification fails everywhere.
        return ("MIXED",
                "Phase B selected a config but Phase C shows uniform per-bin fail. "
                "Inspect overall vs per-bin discrepancy before deciding.")
    return ("MIXED", f"({b_verdict}, {c_verdict}) — review the per-phase tables manually.")


def write_unified_report(out_dir: str,
                         provenance: dict,
                         phase_a: dict,
                         phase_b: dict,
                         phase_c: dict,
                         walltime_s: float) -> None:
    L = []
    L.append("# W1.5 — HDBSCAN diagnostic sweep extended")
    L.append("")
    L.append(f"- Source: mini={provenance['n_mini']} + trainval={provenance['n_trainval']} samples, seed={provenance['seed']}")
    L.append(f"- Walltime: {walltime_s/60:.1f} min")
    L.append("")
    L.append(f"- Phase A verdict: **{phase_a['verdict']}**")
    L.append(f"- Phase B verdict: **{phase_b['verdict']}**")
    L.append(f"- Phase C verdict: **{phase_c['verdict']}**")
    L.append("")

    # --------- Phase A ---------
    L.append("## Phase A — Ground filter sanity")
    L.append("")
    L.append(phase_a["verdict_text"])
    L.append("")
    L.append("![ground_filter_ratio_vs_threshold](phase_a/figures/ground_filter_ratio_vs_threshold.png)")
    L.append("")
    L.append("Ground-filter removal ratio per config. Low values mean the filter is essentially a no-op (root cause candidate); very high values may cut into real foreground.")
    L.append("")
    L.append("![miss_rate_by_ground_mode](phase_a/figures/miss_rate_by_ground_mode.png)")
    L.append("")
    L.append("miss_rate (bars) and n_clusters (line) per ground-filter config. Green band on the right axis is the target [5, 30] for n_clusters.")
    L.append("")
    L.append("### Per-config table")
    L.append("")
    L.append("| label | mean n_clusters | mean miss% | mean M% | ground-drop% | median timing s |")
    L.append("|---|---|---|---|---|---|")
    for c in phase_a["config_summaries"]:
        L.append(f"| {c['label']} | {c['mean_n_clusters']:.2f} | "
                 f"{c['mean_miss_rate']*100:.1f} | {c['mean_M_rate']*100:.1f} | "
                 f"{c['mean_ground_filtered_ratio']*100:.1f} | {c['median_timing_s']:.3f} |")
    L.append("")
    if phase_a.get("best_config"):
        b = phase_a["best_config"]
        L.append(f"**Phase A best config**: `{b['label']}` "
                 f"(n_cl={b['mean_n_clusters']:.2f}, miss={b['mean_miss_rate']*100:.1f}%, "
                 f"M={b['mean_M_rate']*100:.1f}%, timing={b['median_timing_s']:.3f}s).")
        L.append("")

    # --------- Phase B ---------
    L.append("## Phase B — Foreground-aware extended sweep")
    L.append("")
    L.append(phase_b["verdict_text"])
    L.append("")
    L.append(f"Ground filter applied: `{phase_b['ground_filter_kwargs']}`")
    L.append("")
    L.append("![n_clusters_heatmap](phase_b/figures/n_clusters_heatmap.png)")
    L.append("")
    L.append("Mean n_clusters per (min_cluster_size, min_samples) panel — one panel per epsilon. Closer to the [5,15] band is better.")
    L.append("")
    L.append("![miss_rate_heatmap](phase_b/figures/miss_rate_heatmap.png)")
    L.append("")
    L.append("Mean miss_rate (lower better). Colour scale is per-figure; both heatmaps share the same axes.")
    L.append("")
    L.append("![M_rate_heatmap](phase_b/figures/M_rate_heatmap.png)")
    L.append("")
    L.append("Mean 1↔1 match rate (higher better). The strict selection rule requires M ≥ 0.30.")
    L.append("")
    L.append("![pareto_front](phase_b/figures/pareto_front.png)")
    L.append("")
    L.append("Each grid combo plotted in n_clusters × miss_rate space. The red ring marks the selected best config.")
    L.append("")
    bb = phase_b["best_config"]
    L.append("### Phase B best config")
    L.append("")
    L.append("| param | value |")
    L.append("|---|---|")
    L.append(f"| min_cluster_size | **{bb['min_cluster_size']}** |")
    L.append(f"| min_samples | **{bb['min_samples']}** |")
    L.append(f"| cluster_selection_epsilon | **{bb['cluster_selection_epsilon']}** |")
    L.append(f"| mean n_clusters | {bb['mean_n_clusters']:.2f} |")
    L.append(f"| mean miss_rate | {bb['mean_miss_rate']*100:.1f}% |")
    L.append(f"| mean M_rate | {bb['mean_M_rate']*100:.1f}% |")
    L.append(f"| mean L_rate | {bb['mean_L_rate']*100:.1f}% |")
    L.append(f"| mean D_rate | {bb['mean_D_rate']*100:.1f}% |")
    L.append(f"| median timing | {bb['median_timing_s']:.3f} s |")
    L.append("")

    # --------- Phase C ---------
    L.append("## Phase C — Distance-stratified analysis")
    L.append("")
    L.append(phase_c["verdict_text"])
    L.append("")
    L.append(f"Near-range (0–20m) M_rate = {phase_c['near_M_rate']*100:.1f}%, "
             f"far-range (30m+) M_rate = {phase_c['far_M_rate']*100:.1f}%, "
             f"overall M_rate = {phase_c['overall_M_rate']*100:.1f}%.")
    L.append("")
    L.append("| bin | n_GT | M% | L% | D% | miss% |")
    L.append("|---|---|---|---|---|---|")
    for b, rec in phase_c["by_bin"].items():
        if rec["n_GT"] == 0:
            continue
        L.append(f"| {b} | {rec['n_GT']} | {rec['M_rate']*100:.1f} | "
                 f"{rec['L_rate']*100:.1f} | {rec['D_rate']*100:.1f} | "
                 f"{rec['miss_rate']*100:.1f} |")
    L.append("")
    L.append("![M_rate_by_distance](phase_c/figures/M_rate_by_distance.png)")
    L.append("")
    L.append("Per-bin M_rate against the overall mean (red dashed). Far bins lagging the overall is the C2 signature.")
    L.append("")
    L.append("![miss_rate_by_distance](phase_c/figures/miss_rate_by_distance.png)")
    L.append("")
    L.append("Per-bin miss_rate against the overall mean.")
    L.append("")
    L.append("![case_distribution_by_distance](phase_c/figures/case_distribution_by_distance.png)")
    L.append("")
    L.append("Stacked composition (M/L/D/miss) per bin. Shifts in the M slice across bins reveal whether the proposal generator's failure mode is range-dependent.")
    L.append("")

    # --------- unified branch ---------
    branch, branch_text = _unified_branch(phase_b["verdict"], phase_c["verdict"])
    L.append("## Unified branch decision")
    L.append("")
    L.append(f"**→ {branch}**: {branch_text}")
    L.append("")
    L.append("Trace:")
    L.append("")
    L.append("```")
    L.append(f"Phase A verdict = {phase_a['verdict']}")
    L.append(f"Phase B verdict = {phase_b['verdict']}")
    L.append(f"Phase C verdict = {phase_c['verdict']}")
    L.append(f"→ {branch}: {branch_text}")
    L.append("```")
    L.append("")

    # --------- top-3 observations (W1 vs W1.5 explicit) ---------
    L.append("## Top-3 observations")
    L.append("")
    # 1) cluster count change vs W1
    bb_n = bb["mean_n_clusters"]
    L.append(f"1. n_clusters/frame at Phase B best = **{bb_n:.2f}** "
             f"(W1 baseline 31.05; Δ={bb_n - 31.05:+.2f}). "
             + ("In-band [5,15]." if 5 <= bb_n <= 15
                else "Soft-band [5,30]." if 5 <= bb_n <= 30
                else "Out of operational band."))
    L.append("")
    # 2) dominant case + W1 comparison
    cr = {k: bb[f"mean_{k}_rate"] for k in ["M", "L", "D"]}
    cr["miss"] = bb["mean_miss_rate"]
    dominant = max(cr.items(), key=lambda kv: kv[1])
    L.append(f"2. Dominant GT-cluster case at Phase B best = **{dominant[0]}** at "
             f"{dominant[1]*100:.1f}%. W1 baseline dominant was **miss** at 53.3%. "
             + ("Dominant-case shift is the bigger-than-expected change." if dominant[0] != "miss"
                else "Dominant case unchanged from W1."))
    L.append("")
    # 3) distance gap magnitude
    near = phase_c["near_M_rate"] * 100
    far = phase_c["far_M_rate"] * 100
    gap = near - far
    L.append(f"3. Distance gap: near-M={near:.1f}%, far-M={far:.1f}%, "
             f"gap={gap:+.1f} pts. "
             + ("Significant gap — distance-aware variant warranted." if gap >= 15
                else "No significant gap — uniformity is structural."))
    L.append("")

    L.append("## Next-step candidates (manual decision required)")
    L.append("")
    L.append("- **PASS**: lock Phase B config, advance to W2 (Coverage reliability).")
    L.append("- **CONDITIONAL_PASS / DISTANCE_GAP_DOMINANT**: design distance-aware clustering (W1.6) or proceed to W2 with caveat.")
    L.append("- **HDBSCAN_LIMIT_CONFIRMED**: prepare CenterPoint adapter; advisor sign-off needed for closed-set fallback.")
    L.append("- **MIXED**: re-read per-phase tables before deciding.")
    L.append("")

    with open(osp.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(L))
