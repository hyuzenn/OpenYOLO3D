"""~200-sample parity smoke: corrected pipeline vs the validated 10-sweep anchor.

Consumes the tracks.json the native evaluator just wrote (which carries BOTH
this run's emitted predictions and its GT), restricts the official anchor JSON
to the same sample tokens, and scores both through the SAME corrected evaluator
against the SAME GT. Then compares box for box.

No inference here. No thresholds. Nothing is tuned.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/rintern16/OpenYOLO3D")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "audit/table1_regen_2026-08-28/parity_smoke"
TRACKS = OUT / "run/outputs/axis_baseline/tracks.json"
ANCHOR = ROOT / "audit/official_centerpoint_ref/nusc_submission/pred_instances_3d/results_nusc.json"

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
# nuScenes detection_cvpr_2019 class_range, used only to report counts on the
# same footing as the anchor (which mmdet3d range-filters before serialising).
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50,
               "construction_vehicle": 50, "pedestrian": 40, "motorcycle": 40,
               "bicycle": 40, "traffic_cone": 30, "barrier": 30}
# Gates from the audit's proposed acceptance criteria.
GATE_COUNT_REL = 0.10
GATE_VEL_MEDIAN = 0.30


def yaw_of(q_wxyz) -> float:
    from pyquaternion import Quaternion
    return float(Quaternion(*q_wxyz).yaw_pitch_roll[0])


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def greedy_match(ours: list[dict], theirs: list[dict], radius: float = 2.0):
    """Per-class greedy centre-distance matching, highest anchor score first."""
    pairs = []
    by_cls_o = defaultdict(list)
    for b in ours:
        by_cls_o[b["detection_name"]].append(b)
    for cls, tl in defaultdict(list, {c: [b for b in theirs if b["detection_name"] == c]
                                      for c in CLASSES}).items():
        ol = by_cls_o.get(cls, [])
        if not ol or not tl:
            continue
        O = np.array([b["translation"][:2] for b in ol])
        used = np.zeros(len(ol), dtype=bool)
        for t in sorted(tl, key=lambda b: -b["detection_score"]):
            d = np.linalg.norm(O - np.array(t["translation"][:2]), axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= radius:
                used[j] = True
                pairs.append((ol[j], t, float(d[j])))
    return pairs


def main() -> int:
    tracks = json.loads(TRACKS.read_text())
    ours_by_tok = tracks["pred"]
    gt_by_tok = tracks["gt"]
    tokens = sorted(gt_by_tok)
    print(f"tokens in smoke: {len(tokens)}", flush=True)

    anchor_all = json.load(open(ANCHOR))["results"]
    missing = [t for t in tokens if t not in anchor_all]
    if missing:
        print(f"ABORT: {len(missing)} smoke tokens absent from the anchor", flush=True)
        return 2
    anchor_by_tok = {t: anchor_all[t] for t in tokens}
    del anchor_all

    # ---------------- metrics: both sides, same GT, same evaluator ----------
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate
    from dataloaders.nuscenes_loader import NuScenesLoader

    loader = NuScenesLoader(config_path="configs/nuscenes_trainval_multisweep.yaml")

    gt_eb = EvalBoxes()
    for t in tokens:
        gt_eb.add_boxes(t, [DetectionBox.deserialize(d) for d in gt_by_tok[t]])

    def eb(pred_map, anchor_style: bool) -> EvalBoxes:
        e = EvalBoxes()
        for t in tokens:
            boxes = []
            for b in pred_map.get(t, []):
                d = dict(b)
                if anchor_style:
                    d = {"sample_token": t, "translation": b["translation"],
                         "size": b["size"], "rotation": b["rotation"],
                         "velocity": b["velocity"], "ego_translation": [0.0, 0.0, 0.0],
                         "num_pts": -1, "detection_name": b["detection_name"],
                         "detection_score": b["detection_score"],
                         "attribute_name": b["attribute_name"]}
                boxes.append(DetectionBox.deserialize(d))
            e.add_boxes(t, boxes)
        return e

    summary = {}
    for name, pred_map, anchor_style in (("ours_corrected", ours_by_tok, False),
                                         ("anchor", anchor_by_tok, True)):
        s = nu_evaluate(pred_boxes=eb(pred_map, anchor_style), gt_boxes=gt_eb,
                        output_dir=str(OUT / f"eval_{name}"),
                        config_name="detection_cvpr_2019", nusc=loader.nusc)
        summary[name] = {"mAP": s["mean_ap"], "NDS": s["nd_score"],
                         "tp_errors": s["tp_errors"], "counts": s["counts"]}
        print(f"{name}: mAP={s['mean_ap']:.6f} NDS={s['nd_score']:.6f} "
              f"{s['tp_errors']}", flush=True)

    # ---------------- box-level comparison ---------------------------------
    n = len(tokens)
    o_raw = sum(len(ours_by_tok.get(t, [])) for t in tokens)
    a_tot = sum(len(anchor_by_tok[t]) for t in tokens)
    o_ranged, o_cls_r, a_cls = 0, defaultdict(int), defaultdict(int)
    o_scores, a_scores = [], []
    for t in tokens:
        for b in ours_by_tok.get(t, []):
            o_scores.append(b["detection_score"])
            # The harness stores ego_translation as the ego's ABSOLUTE global
            # position (nuscenes_native_evaluator: ego_pose[:3, 3]); the devkit's
            # add_center_dist replaces it with the relative vector only later,
            # inside the evaluator. So the ego distance is the difference.
            ego = (np.asarray(b["translation"][:2], dtype=float)
                   - np.asarray(b["ego_translation"][:2], dtype=float))
            if np.linalg.norm(ego) <= CLASS_RANGE.get(b["detection_name"], 0.0):
                o_ranged += 1
                o_cls_r[b["detection_name"]] += 1
        for b in anchor_by_tok[t]:
            a_scores.append(b["detection_score"])
            a_cls[b["detection_name"]] += 1
    o_scores, a_scores = np.asarray(o_scores), np.asarray(a_scores)

    cd, vr, yr, sr = [], [], [], []
    per_cls_res = defaultdict(lambda: {"n": 0, "cd": [], "vel": [], "yaw": []})
    for t in tokens:
        for o, a, d in greedy_match(ours_by_tok.get(t, []), anchor_by_tok[t]):
            cd.append(d)
            v = float(np.linalg.norm(np.asarray(o["velocity"]) - np.asarray(a["velocity"])))
            y = float(wrap(np.array([yaw_of(o["rotation"]) - yaw_of(a["rotation"])]))[0])
            s = float(np.linalg.norm(np.asarray(o["size"]) - np.asarray(a["size"])))
            vr.append(v); yr.append(y); sr.append(s)
            c = per_cls_res[o["detection_name"]]
            c["n"] += 1; c["cd"].append(d); c["vel"].append(v); c["yaw"].append(abs(y))

    def st(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0:
            return None
        return {"n": int(a.size), "median": float(np.median(a)),
                "mean": float(a.mean()), "p90": float(np.percentile(a, 90))}

    yr_a = np.abs(np.asarray(yr))
    comparison = {
        "n_samples": n,
        "counts": {
            "ours_raw_total": o_raw, "ours_raw_per_sample": o_raw / n,
            "ours_after_class_range": o_ranged,
            "ours_after_class_range_per_sample": o_ranged / n,
            "anchor_total": a_tot, "anchor_per_sample": a_tot / n,
            "count_rel_err_vs_anchor": (o_ranged - a_tot) / a_tot,
        },
        "score": {"ours": st(o_scores), "anchor": st(a_scores)},
        "matched_residuals_2m": {
            "n_matched": len(cd),
            "match_rate_vs_anchor": len(cd) / a_tot if a_tot else None,
            "center_distance_m": st(cd),
            "velocity_residual_ms": st(vr),
            "abs_yaw_residual_rad": st(yr_a),
            "frac_yaw_gt_pi_over_2": float((yr_a > np.pi / 2).mean()) if len(yr_a) else None,
            "size_residual_m": st(sr),
        },
        "per_class": {
            c: {"ours_ranged": o_cls_r.get(c, 0), "anchor": a_cls.get(c, 0),
                "rel_err": ((o_cls_r.get(c, 0) - a_cls.get(c, 0)) / a_cls[c]
                            if a_cls.get(c) else None),
                "matched": per_cls_res[c]["n"],
                "median_center_m": st(per_cls_res[c]["cd"])["median"] if per_cls_res[c]["n"] else None,
                "median_vel_ms": st(per_cls_res[c]["vel"])["median"] if per_cls_res[c]["n"] else None,
                "median_abs_yaw_rad": st(per_cls_res[c]["yaw"])["median"] if per_cls_res[c]["n"] else None}
            for c in CLASSES
        },
    }

    gates = {
        "count_within_10pct": abs(comparison["counts"]["count_rel_err_vs_anchor"]) <= GATE_COUNT_REL,
        "median_velocity_residual_lt_0.3": (
            comparison["matched_residuals_2m"]["velocity_residual_ms"]["median"] < GATE_VEL_MEDIAN
            if cd else False),
    }
    out = {"metrics": summary, "comparison": comparison, "gates": gates,
           "gate_all_pass": all(gates.values())}
    (OUT / "parity_smoke_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"gates": gates, "counts": comparison["counts"],
                      "residuals": comparison["matched_residuals_2m"]}, indent=2), flush=True)
    return 0 if all(gates.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
