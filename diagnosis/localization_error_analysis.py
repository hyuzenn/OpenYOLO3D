"""Localization-error decomposition of the CenterPoint proposal set (read-only).

Follow-up to the oracle-score experiment, which showed score ranking is NOT the
bottleneck (perfect real/fake oracle = +0.034 mAP; native conf already
calibrated) and that the geometric ceiling (dedup oracle) is 0.548. Everything
above 0.548 is localization + coverage. This script asks: *where exactly does
the localization error come from?*

Method (no model / proposal / eval code modified):
  For every in-frame GT (NUSC_10 classes), find the nearest SAME-CLASS proposal
  in the gravity-corrected cache by BEV (horizontal) center distance -- the exact
  geometry nuScenes AP matches on. Record the signed/abs errors of that nearest
  proposal:
      ex, ey, ez        (global-frame component errors)
      horiz = hypot(ex, ey)   <-- the only error that affects nuScenes mAP
      d3d   = sqrt(ex^2+ey^2+ez^2)
  Break down by class and by ego-range bin {0-15, 15-30, 30-50, 50-80} m.
  Percentiles p50/p75/p90/p95; fractions within {0.5,1.0,2.0,4.0} m (horizontal).

  GTs with NO same-class proposal anywhere in the frame are counted separately
  (a pure coverage miss, not a localization error) and excluded from the error
  percentiles.

Perfect-localization ceiling (point 7): for each GT that HAS a same-class
proposal within COVERAGE_M (4 m BEV = loosest nuScenes threshold), emit one
detection at the EXACT GT pose with score 1.0; all else dropped. Evaluate with
the official devkit. This is the max mAP reachable if localization were perfect
while coverage stays exactly as the current proposal set provides it. The gap
  dedup-oracle (0.548, real geometry)  ->  perfect-loc  ->  1.0
isolates localization (dedup->perfloc) from coverage (perfloc->1.0).

Output: results/<date>_localization_error_v01/
"""

from __future__ import annotations

import json
import os
import os.path as osp
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

PROJECT_ROOT = "/home/rintern16/OpenYOLO3D"
sys.path.insert(0, PROJECT_ROOT)

from method_scannet.streaming.nuscenes_native_evaluator import (  # noqa: E402
    NativeTemporalNuScenesEvaluator, NAME_TO_IDX, CLASS_NAMES, _list_val_scenes,
)
from dataloaders.nuscenes_loader import NuScenesLoader  # noqa: E402
from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate  # noqa: E402

CACHE_DIR = osp.join(
    PROJECT_ROOT,
    "results/outdoor_native_temporal_cpcache_thr000_single_gravity")
NUSC_CONFIG = osp.join(PROJECT_ROOT, "configs/nuscenes_trainval.yaml")

COVERAGE_M = 4.0                               # loosest nuScenes dist threshold
DIST_THS = [0.5, 1.0, 2.0, 4.0]
RANGE_EDGES = [0.0, 15.0, 30.0, 50.0, 80.0]    # ego-range bins (m, horizontal)
RANGE_LABELS = ["0-15", "15-30", "30-50", "50-80"]
PCTS = [50, 75, 90, 95]


def range_bin(r):
    for i in range(len(RANGE_EDGES) - 1):
        if RANGE_EDGES[i] <= r < RANGE_EDGES[i + 1]:
            return RANGE_LABELS[i]
    return None        # >=80 m, dropped


def pct_block(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean())}
    for p in PCTS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def frac_within(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {f"<= {t}m": None for t in DIST_THS}
    return {f"<= {t}m": float((a <= t).mean()) for t in DIST_THS}


def main():
    t0 = time.time()
    out_dir = Path(PROJECT_ROOT) / "results" / f"{date.today()}_localization_error_v01"
    (out_dir / "perfloc").mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes (trainval) ...", flush=True)
    loader = NuScenesLoader(config_path=NUSC_CONFIG)
    loader.multi_sweep = False
    loader.num_sweeps = 1

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None,
        cp_cache_dir=CACHE_DIR, proposal_source="gamma",
        proposal_score_threshold=0.0)

    scenes = _list_val_scenes(loader)
    limit = int(os.environ.get("LOC_SCENE_LIMIT", "0"))
    if limit > 0:
        scenes = scenes[:limit]
    print(f"  val scenes={len(scenes)}  cache={CACHE_DIR}", flush=True)

    # ---- accumulators -------------------------------------------------------
    # raw per-GT records (only those with a same-class candidate)
    rec_horiz, rec_d3d = [], []
    rec_ax, rec_ay, rec_az = [], [], []      # abs component errors
    rec_class, rec_rbin = [], []
    rec_range = []

    n_gt_total = 0          # all in-frame NUSC_10 GT
    n_gt_no_cand = 0        # no same-class proposal in frame at all
    n_gt_no_cand_byclass = defaultdict(int)
    n_gt_byclass = defaultdict(int)

    # perfect-localization eval boxes
    per_sample_gt: dict[str, list[dict]] = {}
    per_sample_perfloc: dict[str, list[dict]] = {}

    for si, sc in enumerate(scenes):
        if (si + 1) % 25 == 0 or si == 0:
            print(f"  [{si+1}/{len(scenes)}] {sc[:8]} ({time.time()-t0:.0f}s)",
                  flush=True)
        ev._scene_cache = {}
        for tok in ev._scene_sample_tokens(sc):
            props = ev._get_proposals(tok)
            ego_pose, T_lidar_to_ego, gts = ev._load_meta(tok)
            ego_translation = ego_pose[:3, 3]

            # proposal global xyz grouped by class
            prop_xyz_byclass: dict[str, list[np.ndarray]] = defaultdict(list)
            for p in props:
                cls_name = p["cls_name"]
                if cls_name not in NAME_TO_IDX:
                    continue
                centroid_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
                centroid_global = (ego_pose[:3, :3] @ centroid_ego[:3]) + ego_translation
                prop_xyz_byclass[cls_name].append(centroid_global)

            prop_xyz_np = {c: np.vstack(v) for c, v in prop_xyz_byclass.items()}

            perfloc_preds = []
            for g in gts:
                cls = g["detection_name"]
                if cls not in NAME_TO_IDX:
                    continue
                n_gt_total += 1
                n_gt_byclass[cls] += 1
                gt_xyz = np.asarray(g["translation"], dtype=np.float64)
                rng = float(np.hypot(gt_xyz[0] - ego_translation[0],
                                     gt_xyz[1] - ego_translation[1]))
                rb = range_bin(rng)

                cand = prop_xyz_np.get(cls)
                if cand is None:
                    n_gt_no_cand += 1
                    n_gt_no_cand_byclass[cls] += 1
                    continue

                # nearest same-class proposal by BEV (horizontal) distance
                dxy = cand[:, :2] - gt_xyz[:2]
                horiz_all = np.hypot(dxy[:, 0], dxy[:, 1])
                j = int(np.argmin(horiz_all))
                ex = float(cand[j, 0] - gt_xyz[0])
                ey = float(cand[j, 1] - gt_xyz[1])
                ez = float(cand[j, 2] - gt_xyz[2])
                horiz = float(horiz_all[j])
                d3d = float(np.sqrt(ex * ex + ey * ey + ez * ez))

                rec_horiz.append(horiz); rec_d3d.append(d3d)
                rec_ax.append(abs(ex)); rec_ay.append(abs(ey)); rec_az.append(abs(ez))
                rec_class.append(cls); rec_rbin.append(rb); rec_range.append(rng)

                # perfect-localization ceiling: GT covered within COVERAGE_M ?
                if horiz <= COVERAGE_M:
                    pf = dict(g)
                    pf["detection_score"] = 1.0
                    perfloc_preds.append(pf)

            per_sample_gt[tok] = gts
            per_sample_perfloc[tok] = perfloc_preds

    print(f"emission done: {n_gt_total} GT, {len(per_sample_gt)} samples "
          f"({time.time()-t0:.0f}s)", flush=True)

    horiz = np.asarray(rec_horiz); d3d = np.asarray(rec_d3d)
    ax = np.asarray(rec_ax); ay = np.asarray(rec_ay); az = np.asarray(rec_az)
    rclass = np.asarray(rec_class); rbin = np.asarray(rec_rbin)

    # ---- overall ------------------------------------------------------------
    overall = {
        "n_gt_with_candidate": int(horiz.size),
        "horizontal_err": pct_block(horiz),
        "d3d_err": pct_block(d3d),
        "abs_ex": pct_block(ax), "abs_ey": pct_block(ay), "abs_ez": pct_block(az),
        "frac_within_horizontal": frac_within(horiz),
        "frac_within_3d": frac_within(d3d),
    }

    # ---- by class -----------------------------------------------------------
    by_class = {}
    for c in CLASS_NAMES:
        m = rclass == c
        h = horiz[m]
        by_class[c] = {
            "n_gt": int(n_gt_byclass.get(c, 0)),
            "n_with_cand": int(m.sum()),
            "n_no_cand": int(n_gt_no_cand_byclass.get(c, 0)),
            "frac_no_cand": (n_gt_no_cand_byclass.get(c, 0) / n_gt_byclass[c]
                             if n_gt_byclass.get(c) else None),
            "horizontal_err": pct_block(h),
            "abs_ez": pct_block(az[m]),
            "frac_within_horizontal": frac_within(h),
        }

    # ---- by range bin -------------------------------------------------------
    by_range = {}
    for lab in RANGE_LABELS:
        m = rbin == lab
        h = horiz[m]
        by_range[lab] = {
            "n_with_cand": int(m.sum()),
            "horizontal_err": pct_block(h),
            "abs_ez": pct_block(az[m]),
            "frac_within_horizontal": frac_within(h),
        }

    # ---- class x range grid of median horizontal error ----------------------
    grid = {}
    for c in CLASS_NAMES:
        grid[c] = {}
        for lab in RANGE_LABELS:
            m = (rclass == c) & (rbin == lab)
            h = horiz[m]
            grid[c][lab] = {
                "n": int(m.sum()),
                "median_horiz": (float(np.median(h)) if h.size else None),
                "frac_le_0.5m": (float((h <= 0.5).mean()) if h.size else None),
            }

    # ---- factor-dominance analysis -----------------------------------------
    # how much does median horizontal error move across each factor?
    range_medians = [np.median(horiz[rbin == lab]) for lab in RANGE_LABELS
                     if (rbin == lab).any()]
    class_medians = [np.median(horiz[rclass == c]) for c in CLASS_NAMES
                     if (rclass == c).any()]
    # z vs xy: nuScenes AP is BEV-only, so ez is irrelevant to mAP by construction
    factor = {
        "median_horizontal_err_overall": float(np.median(horiz)),
        "median_abs_ez_overall": float(np.median(az)),
        "median_abs_exy_overall": float(np.median(np.hypot(ax, ay))),
        "note_z_irrelevant_to_map":
            "nuScenes mAP matches on BEV center distance only; ez does NOT "
            "affect mAP (it only affects IoU3D / ATE-style metrics).",
        "spread_median_horiz_across_range": {
            "min": float(min(range_medians)), "max": float(max(range_medians)),
            "ratio_max_min": float(max(range_medians) / max(min(range_medians), 1e-9)),
        },
        "spread_median_horiz_across_class": {
            "min": float(min(class_medians)), "max": float(max(class_medians)),
            "ratio_max_min": float(max(class_medians) / max(min(class_medians), 1e-9)),
        },
    }

    # ---- perfect-localization ceiling (devkit) ------------------------------
    def to_eb(d):
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox
        eb = EvalBoxes()
        for tok, dicts in d.items():
            eb.add_boxes(tok, [DetectionBox.deserialize(x) for x in dicts])
        return eb

    gt_eb = to_eb(per_sample_gt)
    print("eval perfect-localization ceiling ...", flush=True)
    s_perf = nu_evaluate(to_eb(per_sample_perfloc), gt_eb,
                         str(out_dir / "perfloc"), "detection_cvpr_2019")
    print(f"  perfect-loc mAP={s_perf['mean_ap']:.4f}", flush=True)
    pc_perf = json.loads((out_dir / "perfloc" / "per_class.json").read_text())

    perfloc = {
        "coverage_radius_m": COVERAGE_M,
        "mAP": s_perf["mean_ap"],
        "per_class_AP_mean": {c: pc_perf.get(c, {}).get("AP_mean")
                              for c in CLASS_NAMES},
        "interpretation": {
            "native_mAP": 0.3408,
            "dedup_oracle_mAP": 0.5477,
            "perfect_loc_mAP": s_perf["mean_ap"],
            "localization_headroom_dedup_to_perfloc":
                s_perf["mean_ap"] - 0.5477,
            "coverage_headroom_perfloc_to_1": 1.0 - s_perf["mean_ap"],
        },
    }

    summary = {
        "cache_dir": osp.relpath(CACHE_DIR, PROJECT_ROOT),
        "n_val_scenes": len(scenes),
        "n_samples": len(per_sample_gt),
        "n_gt_total_inframe": n_gt_total,
        "n_gt_no_candidate": n_gt_no_cand,
        "frac_gt_no_candidate": (n_gt_no_cand / n_gt_total if n_gt_total else None),
        "nearest_rule": "same-class proposal, min BEV(horizontal) center distance",
        "overall": overall,
        "by_class": by_class,
        "by_range": by_range,
        "class_x_range_grid_median_horiz": grid,
        "factor_dominance": factor,
        "perfect_localization_ceiling": perfloc,
        "walltime_s": time.time() - t0,
    }
    (out_dir / "localization_error_report.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir / 'localization_error_report.json'}", flush=True)
    print(json.dumps({
        "n_gt": n_gt_total,
        "frac_no_candidate": round(summary["frac_gt_no_candidate"], 4),
        "median_horiz_err": round(overall["horizontal_err"]["mean"], 3),
        "p50_horiz": round(overall["horizontal_err"]["p50"], 3),
        "p90_horiz": round(overall["horizontal_err"]["p90"], 3),
        "frac_within_horiz": {k: round(v, 4) for k, v in
                              overall["frac_within_horizontal"].items()},
        "median_abs_ez": round(overall["abs_ez"]["p50"], 3),
        "perfect_loc_mAP": round(s_perf["mean_ap"], 4),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
