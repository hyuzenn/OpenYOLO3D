"""Oracle-score decomposition of the CenterPoint recall->AP gap (read-only).

Question: a gravity-corrected CenterPoint proposal set has an in-range recall
ceiling ~0.79 but native mAP is only ~0.34. How much of that gap is *score
ranking* vs *localization*?

Method (no model / proposal code touched):
  1. Reproduce the native-baseline emission exactly: every cache proposal ->
     one DetectionBox in the global frame with its native CenterPoint score
     (all temporal methods OFF == axes=baseline). Evaluated this == native
     anchor 0.3407, a free regression check.
  2. Oracle score: score=1.0 if the proposal matches a same-class GT within the
     loosest nuScenes distance threshold (4.0 m, BEV center-distance, the exact
     dist_fcn the devkit uses), else 0.0. Same geometry, only score swapped.
  3. Evaluate native + oracle with the official nuScenes devkit (cvpr_2019),
     reading per-class per-threshold AP.

Decomposition (per distance threshold):
     native AP   <=   oracle AP   <=   recall ceiling
     |____________|    |________________|
      score ranking     residual: a single global-binary score cannot rank
      (FP contamination) per-threshold, so loosely-matched (2-4 m) boxes ranked
                         first become FPs at the 0.5/1.0 m thresholds ==
                         localization-limited ranking.

Plus PR diagnostics: matched-vs-unmatched score distributions, precision across
score bins (FP density), and proposal multiplicity per GT.

Output: results/<date>_oracle_score_v01/
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

PROJECT_ROOT = "/home/rintern16/OpenYOLO3D"
sys.path.insert(0, PROJECT_ROOT)

from nuscenes.utils.geometry_utils import transform_matrix  # noqa: E402

from method_scannet.streaming.nuscenes_native_evaluator import (  # noqa: E402
    NativeTemporalNuScenesEvaluator, NAME_TO_IDX, CLASS_NAMES, _list_val_scenes,
)
from method_scannet.streaming.nuscenes_evaluator import _detection_box_dict  # noqa: E402
from dataloaders.nuscenes_loader import NuScenesLoader  # noqa: E402
from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate  # noqa: E402

CACHE_DIR = osp.join(
    PROJECT_ROOT,
    "results/outdoor_native_temporal_cpcache_thr000_single_gravity")
NUSC_CONFIG = osp.join(PROJECT_ROOT, "configs/nuscenes_trainval.yaml")

ORACLE_MATCH_M = 4.0          # loosest nuScenes dist threshold
DIST_THS = [0.5, 1.0, 2.0, 4.0]


def bev_dist(a_xy, b_xy):
    return float(np.hypot(a_xy[0] - b_xy[0], a_xy[1] - b_xy[1]))


def main():
    t_start = time.time()
    out_dir = Path(PROJECT_ROOT) / "results" / f"{date.today()}_oracle_score_v01"
    (out_dir / "native").mkdir(parents=True, exist_ok=True)
    (out_dir / "oracle").mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes (trainval) ...", flush=True)
    loader = NuScenesLoader(config_path=NUSC_CONFIG)
    loader.multi_sweep = False
    loader.num_sweeps = 1

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None,
        cp_cache_dir=CACHE_DIR, proposal_source="gamma",
        proposal_score_threshold=0.0)

    scenes = _list_val_scenes(loader)
    limit = int(os.environ.get("ORACLE_SCENE_LIMIT", "0"))
    if limit > 0:
        scenes = scenes[:limit]
    print(f"  val scenes={len(scenes)}  cache={CACHE_DIR}", flush=True)

    per_sample_native: dict[str, list[dict]] = {}
    per_sample_oracle: dict[str, list[dict]] = {}
    per_sample_dedup: dict[str, list[dict]] = {}    # one nearest prop/GT -> 1.0
    per_sample_gt: dict[str, list[dict]] = {}

    # PR diagnostics accumulators
    matched_scores: list[float] = []      # native scores of TP-matched proposals (@4m)
    unmatched_scores: list[float] = []     # native scores of unmatched proposals
    # per-class multiplicity: proposals within 2.0 m of each GT (same class)
    mult_per_gt: list[int] = []
    mult_per_gt_byclass: dict[str, list[int]] = defaultdict(list)
    n_props_total = 0

    for si, sc in enumerate(scenes):
        if (si + 1) % 25 == 0 or si == 0:
            print(f"  [{si+1}/{len(scenes)}] {sc[:8]} "
                  f"({time.time()-t_start:.0f}s)", flush=True)
        ev._scene_cache = {}
        for tok in ev._scene_sample_tokens(sc):
            props = ev._get_proposals(tok)
            ego_pose, T_lidar_to_ego, gts = ev._load_meta(tok)
            ego_translation = ego_pose[:3, 3]
            ego_quat = Quaternion(matrix=ego_pose[:3, :3])
            l2e_quat = Quaternion(matrix=T_lidar_to_ego[:3, :3])

            # GT global xy per class for matching
            gt_xy_byclass: dict[str, list[np.ndarray]] = defaultdict(list)
            for g in gts:
                gt_xy_byclass[g["detection_name"]].append(
                    np.asarray(g["translation"][:2], dtype=np.float64))

            # Pass 1: materialise each proposal's global geometry + match info.
            entries = []                       # per-proposal dicts
            gt_hit_count: dict[str, list[int]] = {
                c: [0] * len(v) for c, v in gt_xy_byclass.items()}
            # nearest proposal per GT for the dedup oracle: (entry_idx, dist)
            gt_best_prop: dict[tuple, tuple] = {}

            for p in props:
                cls_name = p["cls_name"]
                if cls_name not in NAME_TO_IDX:
                    continue
                n_props_total += 1
                bbox_lidar = p["bbox_lidar"]
                centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
                yaw_lidar = float(bbox_lidar[6]) if len(bbox_lidar) >= 7 else 0.0
                box_q_ego = l2e_quat * Quaternion(axis=(0., 0., 1.), angle=yaw_lidar)
                centroid_global = (ego_pose[:3, :3] @ centroid_ego[:3]) + ego_translation
                global_q = ego_quat * box_q_ego
                rot = [float(global_q.w), float(global_q.x),
                       float(global_q.y), float(global_q.z)]
                score = float(p.get("score", 0.0))
                gxy = centroid_global[:2]

                cand = gt_xy_byclass.get(cls_name, [])
                matched = False
                best4m = None            # (gi, d) nearest GT within 4 m
                best2m = None
                for gi, g_xy in enumerate(cand):
                    d = bev_dist(gxy, g_xy)
                    if d <= ORACLE_MATCH_M:
                        matched = True
                        if best4m is None or d < best4m[1]:
                            best4m = (gi, d)
                    if d <= 2.0 and (best2m is None or d < best2m[1]):
                        best2m = (gi, d)
                if best2m is not None:
                    gt_hit_count[cls_name][best2m[0]] += 1
                (matched_scores if matched else unmatched_scores).append(score)

                ei = len(entries)
                entries.append({
                    "bbox_lidar": bbox_lidar, "centroid_global": centroid_global,
                    "rot": rot, "cls_name": cls_name, "score": score,
                    "matched": matched})
                # claim nearest-prop slot for this GT (dedup oracle)
                if best4m is not None:
                    key = (cls_name, best4m[0])
                    if key not in gt_best_prop or best4m[1] < gt_best_prop[key][1]:
                        gt_best_prop[key] = (ei, best4m[1])

            dedup_winner = {ei for ei, _d in gt_best_prop.values()}

            # Pass 2: emit three score variants (same geometry).
            native_preds, oracle_preds, dedup_preds = [], [], []
            for ei, e in enumerate(entries):
                base = dict(global_id=0, sample_token=tok,
                            bbox_lidar=e["bbox_lidar"],
                            centroid_global=e["centroid_global"],
                            ego_translation=ego_translation,
                            rotation_global_wxyz=e["rot"],
                            detection_name=e["cls_name"])
                native_preds.append(_detection_box_dict(score=e["score"], **base))
                oracle_preds.append(_detection_box_dict(
                    score=(1.0 if e["matched"] else 0.0), **base))
                dedup_preds.append(_detection_box_dict(
                    score=(1.0 if ei in dedup_winner else 0.0), **base))

            for c, counts in gt_hit_count.items():
                for k in counts:
                    mult_per_gt.append(k)
                    mult_per_gt_byclass[c].append(k)

            per_sample_native[tok] = native_preds
            per_sample_oracle[tok] = oracle_preds
            per_sample_dedup[tok] = dedup_preds
            per_sample_gt[tok] = gts

    print(f"emission done: {n_props_total} props, "
          f"{len(per_sample_gt)} samples ({time.time()-t_start:.0f}s)", flush=True)

    # --- devkit eval: native (anchor) + oracle -------------------------------
    def to_eb(d):
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox
        eb = EvalBoxes()
        for tok, dicts in d.items():
            eb.add_boxes(tok, [DetectionBox.deserialize(x) for x in dicts])
        return eb

    gt_eb = to_eb(per_sample_gt)

    print("eval native ...", flush=True)
    s_native = nu_evaluate(to_eb(per_sample_native), gt_eb,
                           str(out_dir / "native"), "detection_cvpr_2019")
    print(f"  native mAP={s_native['mean_ap']:.4f}", flush=True)

    print("eval oracle ...", flush=True)
    s_oracle = nu_evaluate(to_eb(per_sample_oracle), gt_eb,
                           str(out_dir / "oracle"), "detection_cvpr_2019")
    print(f"  oracle mAP={s_oracle['mean_ap']:.4f}", flush=True)

    (out_dir / "dedup").mkdir(parents=True, exist_ok=True)
    print("eval dedup-oracle ...", flush=True)
    s_dedup = nu_evaluate(to_eb(per_sample_dedup), gt_eb,
                          str(out_dir / "dedup"), "detection_cvpr_2019")
    print(f"  dedup mAP={s_dedup['mean_ap']:.4f}", flush=True)

    pc_native = json.loads((out_dir / "native" / "per_class.json").read_text())
    pc_oracle = json.loads((out_dir / "oracle" / "per_class.json").read_text())
    pc_dedup = json.loads((out_dir / "dedup" / "per_class.json").read_text())

    # --- PR diagnostics ------------------------------------------------------
    bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
    ms = np.asarray(matched_scores)
    us = np.asarray(unmatched_scores)
    pr_table = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = int(((ms >= lo) & (ms < hi)).sum())
        u = int(((us >= lo) & (us < hi)).sum())
        tot = m + u
        pr_table.append({
            "score_bin": f"[{lo:.2f},{hi:.2f})",
            "tp_matched": m, "fp_unmatched": u, "total": tot,
            "precision": (m / tot if tot else None),
        })

    mult = np.asarray(mult_per_gt)
    mult_hist = {str(k): int((mult == k).sum()) for k in range(0, 6)}
    mult_hist["6+"] = int((mult >= 6).sum())
    mult_byclass = {
        c: {"n_gt": len(v), "mean_mult": float(np.mean(v)),
            "frac_zero": float(np.mean(np.asarray(v) == 0)),
            "frac_dup_2plus": float(np.mean(np.asarray(v) >= 2))}
        for c, v in sorted(mult_per_gt_byclass.items())}

    # --- recall-ceiling reference (from the prior probe) ---------------------
    probe_fp = osp.join(
        PROJECT_ROOT,
        "results/2026-06-10_proposal_recall_gravity_v01/recall_probe.json")
    recall_ref = None
    if osp.exists(probe_fp):
        recall_ref = osp.relpath(probe_fp, PROJECT_ROOT)

    # --- assemble report -----------------------------------------------------
    per_class_cmp = {}
    for c in CLASS_NAMES:
        n = pc_native.get(c, {})
        o = pc_oracle.get(c, {})
        dd = pc_dedup.get(c, {})
        per_class_cmp[c] = {
            "native_AP_mean": n.get("AP_mean"),
            "oracle_AP_mean": o.get("AP_mean"),
            "dedup_AP_mean": dd.get("AP_mean"),
            "gain_oracle": (None if (n.get("AP_mean") is None or o.get("AP_mean") is None)
                            else o["AP_mean"] - n["AP_mean"]),
            "gain_dedup": (None if (n.get("AP_mean") is None or dd.get("AP_mean") is None)
                           else dd["AP_mean"] - n["AP_mean"]),
            "native_per_th": {k: n.get(k) for k in
                              ("AP@0.5", "AP@1.0", "AP@2.0", "AP@4.0")},
            "oracle_per_th": {k: o.get(k) for k in
                              ("AP@0.5", "AP@1.0", "AP@2.0", "AP@4.0")},
            "dedup_per_th": {k: dd.get(k) for k in
                             ("AP@0.5", "AP@1.0", "AP@2.0", "AP@4.0")},
        }

    # devkit averages per-threshold AP across classes
    def mean_th(pc, key):
        vals = [pc[c][key] for c in CLASS_NAMES if c in pc]
        return float(np.mean(vals)) if vals else None

    summary = {
        "cache_dir": osp.relpath(CACHE_DIR, PROJECT_ROOT),
        "n_val_scenes": len(scenes),
        "n_samples": len(per_sample_gt),
        "n_proposals": n_props_total,
        "n_gt_in_eval": int(sum(len(v) for v in per_sample_gt.values())),
        "oracle_match_dist_m": ORACLE_MATCH_M,
        "recall_ceiling_reference": recall_ref,
        "mAP": {
            "native": s_native["mean_ap"],
            "oracle_matchany_4m": s_oracle["mean_ap"],
            "dedup_oracle": s_dedup["mean_ap"],
            "gain_oracle": s_oracle["mean_ap"] - s_native["mean_ap"],
            "gain_dedup": s_dedup["mean_ap"] - s_native["mean_ap"],
        },
        "mAP_per_threshold": {
            f"@{th}": {"native": mean_th(pc_native, f"AP@{th}"),
                       "oracle": mean_th(pc_oracle, f"AP@{th}"),
                       "dedup": mean_th(pc_dedup, f"AP@{th}")}
            for th in DIST_THS},
        "per_class": per_class_cmp,
        "pr_diag": {
            "n_matched_4m": int(ms.size),
            "n_unmatched_4m": int(us.size),
            "matched_score": {"mean": float(ms.mean()) if ms.size else None,
                              "median": float(np.median(ms)) if ms.size else None,
                              "p10": float(np.percentile(ms, 10)) if ms.size else None},
            "unmatched_score": {"mean": float(us.mean()) if us.size else None,
                                "median": float(np.median(us)) if us.size else None,
                                "p90": float(np.percentile(us, 90)) if us.size else None},
            "precision_by_score_bin": pr_table,
        },
        "multiplicity_per_gt_within_2m": {
            "overall_hist": mult_hist,
            "mean": float(mult.mean()) if mult.size else None,
            "frac_gt_zero_prop": float((mult == 0).mean()) if mult.size else None,
            "frac_gt_duplicated_2plus": float((mult >= 2).mean()) if mult.size else None,
            "by_class": mult_byclass,
        },
        "walltime_s": time.time() - t_start,
    }
    (out_dir / "oracle_score_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'oracle_score_report.json'}", flush=True)
    print(json.dumps({
        "native_mAP": round(s_native["mean_ap"], 4),
        "oracle_matchany_mAP": round(s_oracle["mean_ap"], 4),
        "dedup_oracle_mAP": round(s_dedup["mean_ap"], 4),
        "per_th": summary["mAP_per_threshold"],
        "mult_mean": summary["multiplicity_per_gt_within_2m"]["mean"],
        "frac_gt_dup_2plus": summary["multiplicity_per_gt_within_2m"]["frac_gt_duplicated_2plus"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
