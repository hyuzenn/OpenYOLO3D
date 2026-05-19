"""Stage 3 — build report.md + figures from aggregate.json + nuscenes_eval/.

Auto-fires the four conditional decision branches (β-A/B/C/D) from the
overall mAP, plus pedestrian mAP / far-range mAP / real-time tags. Trace
of which conditional triggered each branch is printed inline.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]
DISTANCE_BIN_LABELS = ["0-10m", "10-20m", "20-30m", "30-50m", "50m+"]


# ---------- decision branches ---------------------------------------------


def decide_overall(map_pct: float):
    if map_pct >= 30.0:
        return "β-A", ("STRONG ≥ 30 — OpenYOLO3D unexpectedly works on nuScenes; "
                       "method narrative needs reconsideration.")
    if map_pct >= 15.0:
        return "β-B", ("MODERATE 15–30 — working but limited. Treat as the "
                       "baseline our method must exceed.")
    if map_pct >= 5.0:
        return "β-C", ("WEAK 5–15 — indoor pipeline does not transfer to outdoor. "
                       "Method narrative is well justified; nuScenes work proceeds.")
    return "β-D", ("FAIL < 5 — catastrophic; OpenYOLO3D is not a viable nuScenes "
                   "baseline. Narrative is very strong; ALSO consider adding a "
                   "second baseline so the comparison is meaningful.")


def decide_class(name: str, ap: float, label: str):
    if ap >= 0.30:
        return f"{label}_{name}_STRONG"
    if ap >= 0.10:
        return f"{label}_{name}_MODERATE"
    if ap >= 0.02:
        return f"{label}_{name}_WEAK"
    return f"{label}_{name}_FAIL"


def decide_real_time(median_s: float):
    if median_s is None:
        return "RT_NA"
    if median_s <= 0.5:
        return "RT_OK"   # ≥ 2 Hz on a 0.1 s budget is unrealistic; flag fast vs slow
    if median_s <= 5.0:
        return "RT_SLOW"
    return "RT_TOO_SLOW"


# ---------- figures --------------------------------------------------------


def fig_per_class_ap(per_class: dict, out_path: str):
    cls = list(per_class.keys())
    ap_mean = [per_class[c]["AP_mean"] for c in cls]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(cls, ap_mean, color="steelblue")
    ax.set_ylabel("AP (mean over 0.5/1.0/2.0/4.0 m)")
    ax.set_ylim(0, max(0.05, max(ap_mean) * 1.15))
    ax.set_title("Per-class detection AP")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fig_per_distance_M(distance_strata: dict, out_path: str):
    bins = [b for b in DISTANCE_BIN_LABELS if b in distance_strata]
    M = [distance_strata[b]["M_rate"] for b in bins]
    miss = [distance_strata[b]["miss_rate"] for b in bins]
    x = np.arange(len(bins))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, M, width, label="M_rate", color="seagreen")
    ax.bar(x + width / 2, miss, width, label="miss_rate", color="firebrick")
    ax.set_xticks(x); ax.set_xticklabels(bins)
    ax.set_ylim(0, 1)
    ax.set_title("M / miss rate by ego-distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fig_case_breakdown(instance_level: dict, out_path: str):
    cases = instance_level["case_counts"]
    labels = ["M", "L", "D", "miss"]
    vals = [cases.get(c, 0) for c in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals,
           color=["seagreen", "goldenrod", "darkorange", "firebrick"])
    ax.set_ylabel("# GT")
    ax.set_title(f"Instance-level case breakdown (n_GT={sum(vals)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def fig_timing(timing_s: dict, out_path: str):
    rows = ["adapter_s", "predict_s", "format_s", "metrics_s", "total_s"]
    medians = [timing_s[k]["median"] if timing_s[k]["median"] is not None else 0.0
               for k in rows]
    p95 = [timing_s[k]["p95"] if timing_s[k]["p95"] is not None else 0.0 for k in rows]
    x = np.arange(len(rows))
    width = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, medians, width, label="median", color="steelblue")
    ax.bar(x + width / 2, p95, width, label="p95", color="lightsteelblue")
    ax.set_xticks(x); ax.set_xticklabels(rows, rotation=20)
    ax.set_ylabel("seconds")
    ax.set_title("Per-sample timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------- markdown writer -----------------------------------------------


def _table_per_class(per_class: dict, summary: dict, out_lines):
    out_lines.append("\n### Per-class breakdown (10 detection classes)\n")
    out_lines.append(
        "| class | AP_mean | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 |"
        " trans_err | scale_err | orient_err | vel_err | attr_err |\n"
    )
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for c in CLASS_NAMES:
        d = per_class[c]
        out_lines.append(
            f"| {c} | {d['AP_mean']:.4f} | {d['AP@0.5']:.4f} | {d['AP@1.0']:.4f} | "
            f"{d['AP@2.0']:.4f} | {d['AP@4.0']:.4f} | {d['trans_err']:.4f} | "
            f"{d['scale_err']:.4f} | {d['orient_err']:.4f} | {d['vel_err']:.4f} | "
            f"{d['attr_err']:.4f} |\n"
        )


def _table_distance(distance_strata: dict, out_lines):
    out_lines.append("\n### Distance-stratified instance metrics\n")
    out_lines.append("| bin | n_GT | M_rate | L_rate | D_rate | miss_rate |\n")
    out_lines.append("|---|---:|---:|---:|---:|---:|\n")
    for b in DISTANCE_BIN_LABELS:
        if b not in distance_strata:
            continue
        d = distance_strata[b]
        out_lines.append(
            f"| {b} | {d['n_gt']} | {d['M_rate']:.3f} | {d['L_rate']:.3f} | "
            f"{d['D_rate']:.3f} | {d['miss_rate']:.3f} |\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/diagnosis_beta_baseline")
    args = parser.parse_args()

    out_dir = args.out_dir
    agg = json.load(open(osp.join(out_dir, "aggregate.json")))
    summary = agg["nuscenes_eval"]
    per_class = json.load(open(osp.join(out_dir, "nuscenes_eval", "per_class.json")))

    # figures
    fig_dir = osp.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_per_class_ap(per_class, osp.join(fig_dir, "mAP_per_class.png"))
    fig_per_distance_M(agg["instance_level"]["distance_strata"],
                       osp.join(fig_dir, "mAP_per_distance.png"))
    fig_case_breakdown(agg["instance_level"], osp.join(fig_dir, "M_rate_breakdown.png"))
    fig_timing(agg["timing_s"], osp.join(fig_dir, "timing.png"))

    # decision
    map_pct = summary["mean_ap"] * 100.0
    branch, branch_msg = decide_overall(map_pct)
    ped_ap = per_class["pedestrian"]["AP_mean"]
    ped_branch = decide_class("pedestrian", ped_ap, "PED")
    far_bin = agg["instance_level"]["distance_strata"].get("50m+", {})
    far_M = far_bin.get("M_rate", 0.0)
    far_branch = "FAR_OK" if far_M >= 0.10 else "FAR_FAIL"
    rt_branch = decide_real_time(agg["timing_s"]["total_s"]["median"])

    lines = []
    lines.append("# β baseline — OpenYOLO3D × nuScenes detection (50 samples)\n\n")
    lines.append(f"- samples: {agg['n_samples_evaluated']}/{agg['n_samples_total']}"
                 f" evaluated (skipped {agg['n_samples_skipped']})\n")
    lines.append(f"- samples_used: `{agg.get('samples_used_path')}`"
                 f" (seed={agg.get('samples_used_seed')})\n")
    lines.append(f"- text prompts: `{agg['text_prompts']}`\n")
    lines.append(f"- OpenYOLO3D init: {agg['openyolo3d_init_s']:.1f}s\n\n")

    lines.append("## Standard nuScenes detection metrics\n\n")
    lines.append(f"- **mAP**: {summary['mean_ap']*100:.2f}%  ({summary['mean_ap']:.4f})\n")
    lines.append(f"- **NDS**: {summary['nd_score']:.4f}\n")
    for k, v in summary["tp_errors"].items():
        lines.append(f"- {k}: {v:.4f}\n")
    counts = summary.get("counts", {})
    lines.append(f"- n_pred_boxes (after class_range filter): {counts.get('n_pred_boxes')}\n")
    lines.append(f"- n_gt_boxes (after class_range filter):   {counts.get('n_gt_boxes')}\n")

    _table_per_class(per_class, summary, lines)

    lines.append("\n## Instance-level (W1-style)\n\n")
    il = agg["instance_level"]
    lines.append(f"- n_GT total: {il['n_gt_total']}\n")
    lines.append(f"- M_rate:    {il['M_rate']:.3f}\n")
    lines.append(f"- L_rate:    {il['L_rate']:.3f}\n")
    lines.append(f"- D_rate:    {il['D_rate']:.3f}\n")
    lines.append(f"- miss_rate: {il['miss_rate']:.3f}\n")

    _table_distance(il["distance_strata"], lines)

    lines.append("\n## Timing (s/sample)\n\n")
    lines.append("| stage | median | p95 | mean | n |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for k in ["adapter_s", "predict_s", "format_s", "metrics_s", "total_s"]:
        s = agg["timing_s"][k]
        med = s["median"] if s["median"] is not None else float("nan")
        p95 = s["p95"]    if s["p95"]    is not None else float("nan")
        mn  = s["mean"]   if s["mean"]   is not None else float("nan")
        lines.append(f"| {k} | {med:.2f} | {p95:.2f} | {mn:.2f} | {s['n']} |\n")

    lines.append("\n## Decision\n\n")
    lines.append(f"- overall mAP = {map_pct:.2f}% → **{branch}**\n")
    lines.append(f"  - {branch_msg}\n")
    lines.append(f"- pedestrian AP_mean = {ped_ap:.4f} → **{ped_branch}**\n")
    lines.append(f"- far-range (50m+) M_rate = {far_M:.3f} → **{far_branch}**\n")
    lines.append(f"- median per-sample latency = "
                 f"{agg['timing_s']['total_s']['median']:.2f}s → **{rt_branch}**\n")
    if branch == "β-A":
        lines.append(
            "\n**Trace**: mAP ≥ 30 → β-A. Surprising. Re-check whether the "
            "indoor-pretrained Mask3D + YOLO-World aggregation actually fits "
            "outdoor LiDAR semantics, or whether the metric setup leaks. "
            "Method narrative ('OpenYOLO3D fails outdoor') needs revision.\n")
    elif branch == "β-B":
        lines.append(
            "\n**Trace**: mAP in [15,30) → β-B. Working baseline; method must "
            "improve on it materially. Pick the metric our method most clearly "
            "wins on for the headline.\n")
    elif branch == "β-C":
        lines.append(
            "\n**Trace**: mAP in [5,15) → β-C. Indoor pipeline does not transfer. "
            "Narrative for our method is justified. Stop — wait for next-step "
            "decision before continuing.\n")
    else:
        lines.append(
            "\n**Trace**: mAP < 5 → β-D. Baseline is meaningless on its own; we "
            "still report it, but additionally need a second nuScenes baseline "
            "(e.g. 3D detector trained on outdoor data) so the comparison "
            "section is not vacuous.\n")

    lines.append("\n## Figures\n\n")
    for name in ("mAP_per_class.png", "mAP_per_distance.png",
                 "M_rate_breakdown.png", "timing.png"):
        lines.append(f"- ![{name}](figures/{name})\n")

    with open(osp.join(out_dir, "report.md"), "w") as f:
        f.writelines(lines)
    print(f"wrote {osp.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
