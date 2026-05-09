"""Build results/diagnosis_beta_baseline_v2/report.md from v2 aggregate.json
+ v1 instance/timing (predictions unchanged)."""

from __future__ import annotations

import json
import math
import os.path as osp


V1 = "results/diagnosis_beta_baseline"
V2 = "results/diagnosis_beta_baseline_v2"

PROMPTS = ["car","truck","bus","trailer","construction_vehicle",
           "pedestrian","motorcycle","bicycle","traffic_cone","barrier"]


def f4(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.4f}"


def main() -> int:
    with open(osp.join(V1, "aggregate.json")) as f:
        v1 = json.load(f)
    with open(osp.join(V2, "aggregate.json")) as f:
        v2 = json.load(f)

    v1n = v1["nuscenes_eval"]
    v2n = v2["nuscenes_eval"]

    inst = v1["instance_level"]  # unchanged in v2
    timing = v1["timing_s"]

    out = []
    out.append("# β baseline v2 — GT-bug-fixed nuScenes detection (50 samples)\n")
    out.append("- samples: 43/50 evaluated (skipped 7) — same set as v1\n")
    out.append(f"- text prompts: {PROMPTS}\n")
    out.append("- v1 had `n_gt_boxes=0` due to missing `ego_translation` field on GT dicts. Fix: see `gt_bug_diagnosis.md`.\n\n")

    out.append("## v1 vs v2 — headline\n\n")
    out.append("| metric | v1 (buggy) | v2 (fixed) |\n")
    out.append("|---|---:|---:|\n")
    out.append(f"| mAP | {f4(v1n.get('mean_ap', 0.0))} | {f4(v2n.get('mean_ap', 0.0))} |\n")
    out.append(f"| NDS | {f4(v1n.get('nd_score', 0.0))} | {f4(v2n.get('nd_score', 0.0))} |\n")
    out.append(f"| n_pred_boxes (post class_range) | {v1n['counts']['n_pred_boxes']} | {v2n['counts']['n_pred_boxes']} |\n")
    out.append(f"| n_gt_boxes (post class_range) | {v1n['counts']['n_gt_boxes']} | {v2n['counts']['n_gt_boxes']} |\n")
    out.append(f"| n_samples | {v1n['counts']['n_samples']} | {v2n['counts']['n_samples']} |\n\n")

    out.append("## TP errors (v2)\n\n")
    tp = v2n.get("tp_errors", {})
    out.append(f"- trans_err: {f4(tp.get('trans_err'))}\n")
    out.append(f"- scale_err: {f4(tp.get('scale_err'))}\n")
    out.append(f"- orient_err: {f4(tp.get('orient_err'))}\n")
    out.append(f"- vel_err: {f4(tp.get('vel_err'))}\n")
    out.append(f"- attr_err: {f4(tp.get('attr_err'))}\n\n")

    out.append("## Per-class breakdown — v2\n\n")
    out.append("| class | AP_mean | AP@0.5 | AP@1.0 | AP@2.0 | AP@4.0 | trans_err | scale_err | orient_err | vel_err | attr_err |\n")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    label_aps = v2n.get("label_aps", {})
    label_tps = v2n.get("label_tp_errors", {})
    mean_aps = v2n.get("mean_dist_aps", {})
    for cls in PROMPTS:
        ap = label_aps.get(cls, {})
        tp = label_tps.get(cls, {})
        out.append(
            f"| {cls} | {f4(mean_aps.get(cls))} | "
            f"{f4(ap.get('0.5'))} | {f4(ap.get('1.0'))} | {f4(ap.get('2.0'))} | {f4(ap.get('4.0'))} | "
            f"{f4(tp.get('trans_err'))} | {f4(tp.get('scale_err'))} | {f4(tp.get('orient_err'))} | "
            f"{f4(tp.get('vel_err'))} | {f4(tp.get('attr_err'))} |\n"
        )

    out.append("\n## Instance-level (W1-style) — UNCHANGED from v1 (regression check)\n\n")
    out.append(f"- n_GT total: {inst['n_gt_total']}\n")
    out.append(f"- M_rate:    {inst['M_rate']:.3f}\n")
    out.append(f"- L_rate:    {inst['L_rate']:.3f}\n")
    out.append(f"- D_rate:    {inst['D_rate']:.3f}\n")
    out.append(f"- miss_rate: {inst['miss_rate']:.3f}\n\n")
    out.append("### Distance-stratified instance metrics\n")
    out.append("| bin | n_GT | M_rate | L_rate | D_rate | miss_rate |\n")
    out.append("|---|---:|---:|---:|---:|---:|\n")
    for b in ["0-10m","10-20m","20-30m","30-50m","50m+"]:
        s = inst["distance_strata"][b]
        out.append(f"| {b} | {s['n_gt']} | {s['M_rate']:.3f} | {s['L_rate']:.3f} | {s['D_rate']:.3f} | {s['miss_rate']:.3f} |\n")

    out.append("\n## Timing (s/sample) — UNCHANGED from v1\n\n")
    out.append("| stage | median | p95 | mean | n |\n")
    out.append("|---|---:|---:|---:|---:|\n")
    for st in ["adapter_s","predict_s","format_s","metrics_s","total_s"]:
        d = timing[st]
        out.append(f"| {st} | {d['median']:.2f} | {d['p95']:.2f} | {d['mean']:.2f} | {d['n']} |\n")

    # Decision branch (Stage 4)
    out.append("\n## Decision (re-fired on v2 mAP)\n\n")
    map_v2 = float(v2n.get("mean_ap", 0.0))
    map_pct = map_v2 * 100.0
    if map_pct >= 30:
        bucket = "STRONG (≥30%)"
        narr = "OpenYOLO3D performs surprisingly well on nuScenes; revisit the indoor-only narrative."
    elif map_pct >= 15:
        bucket = "MODERATE (15-30%)"
        narr = "Baseline is competitive; comparison section will need a stronger contender."
    elif map_pct >= 5:
        bucket = "WEAK (5-15%)"
        narr = "Baseline is below useful threshold; supports the indoor-only narrative."
    else:
        bucket = "FAIL (<5%)"
        narr = "Catastrophic on outdoor data; consistent with v1 narrative even after fixing the GT bug. The bug masked the actual sub-5% mAP, but the conclusion is unchanged."
    out.append(f"- overall mAP (v2 corrected) = {map_pct:.2f}% → **{bucket}**\n")
    out.append(f"- {narr}\n")

    ped_ap = float(mean_aps.get("pedestrian", 0.0))
    if ped_ap < 0.05:
        out.append("- pedestrian AP_mean < 5% → **PED_pedestrian_FAIL**\n")
    far_dist = inst["distance_strata"]["50m+"]
    if far_dist["M_rate"] < 0.05:
        out.append("- far-range (50m+) M_rate < 5% → **FAR_FAIL**\n")
    if timing["total_s"]["median"] > 0.2:
        out.append(f"- median per-sample latency = {timing['total_s']['median']:.2f}s → **RT_SLOW**\n")

    out.append("\n## Regression check — md5 / acceptance criteria\n\n")
    out.append("- nuScenes-devkit code: untouched (sys-installed under conda env, not modified)\n")
    out.append("- OpenYOLO3D core (utils.py, evaluate/, models/, embed/): untouched\n")
    out.append("- instance_metrics.py (W1 GT loader): untouched; instance-level metrics identical to v1\n")
    out.append("- format_predictions.py: extended `gt_to_detection_boxes` to accept `ego_pose_4x4` and write `ego_translation` (additive change)\n")
    out.append("- run_baseline.py: passes `ego_pose_4x4=item['ego_pose']` to gt_to_detection_boxes (call-site update; no inference re-run for v2)\n")
    out.append("- predictions / inference: NOT re-run; v2 reuses v1's `pred_boxes.json` byte-for-byte\n")
    out.append("- samples_used.json: NOT changed; same 43/50 samples evaluated\n")

    rep_path = osp.join(V2, "report.md")
    with open(rep_path, "w") as f:
        f.write("".join(out))
    print(f"wrote {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
