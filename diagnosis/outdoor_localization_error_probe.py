"""Outdoor localization-error probe (Task: localization bottleneck diagnosis).

EVIDENCE-ONLY. No method changes, no evaluator changes. This script imports the
geometry / GT / proposal-loading helpers from
``diagnosis.outdoor_proposal_recall_probe`` (unchanged) and the existing
nuScenes-devkit eval from ``diagnosis_beta_baseline.evaluate_nuscenes`` (called,
never edited). The only side effect is the JSON written under ``results/``.

Given (from prior diagnosis, the calibration verdict):
    native mAP   = 0.3408
    oracle score = 0.3749   (re-score, same boxes)   -> calibration ~ small
    dedup oracle = 0.5477   (re-label + dedup)        -> class/dup headroom
This probe isolates the *localization* axis underneath all of that.

What it measures, per GT, using the gravity-corrected γ (CenterPoint) cache:
  1. nearest SAME-CLASS proposal (by BEV / horizontal centre distance).
  2. signed errors ex, ey, ez; |ex| |ey| |ez|; horizontal dist; full-3D dist.
  3. percentiles p50/p75/p90/p95 of each, overall + per-class + per-distance-bin.
  4. fraction of GT whose nearest same-class proposal is within 0.5/1/2/4 m
     (== the same-class, per-mAP-threshold recall).
  5. dominant-factor read-out: range vs class vs z vs x/y.
  6. localization-only mAP ceiling: re-run the devkit eval on a counterfactual
     where every native proposal that has a same-class GT within 4 m is snapped
     onto that GT's BEV centre (native class + native score + native proposal
     set all preserved). The gap to 0.3408 is the pure-localization headroom.

nuScenes mAP matches on BEV centre distance, so "nearest" is taken in BEV. ez is
reported but is mAP-irrelevant by construction (kept to settle the z-error
question quantitatively).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

# Reuse, never modify, the recall-probe machinery.
from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10,
    NUSC_10_SET,
    DIST_BINS,
    _gt_records,
    _proposals_to_global,
    _load_cached_proposals,
    _list_val_scenes,
    _scene_sample_tokens,
    _distance_bin,
)

BOX_DIST_THRESHOLDS_M = (0.5, 1.0, 2.0, 4.0)
# Radius beyond which a same-class proposal can never be the GT's match under
# the loosest official nuScenes bucket (4 m). Errors are profiled over the
# "matchable" set (nearest same-class proposal within this radius); coverage is
# reported with and without the cap.
MATCH_RADIUS_M = 4.0
PCTLS = (50, 75, 90, 95)


def _pctls(vals) -> dict:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    out = {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std())}
    for p in PCTLS:
        out[f"p{p}"] = float(np.percentile(a, p))
    return out


def _nearest_same_class(gt: dict, props_by_cls: dict[str, list[dict]]):
    """Return (horiz_dist, ex, ey, ez, dist3d, proposal) for the nearest
    same-class proposal in BEV, or None when no same-class proposal exists."""
    cands = props_by_cls.get(gt["cls_name"])
    if not cands:
        return None
    gc = gt["centroid_global"]
    gxy = gc[:2]
    best = None
    best_d = float("inf")
    for p in cands:
        pc = p["centroid_global"]
        d = float(np.linalg.norm(pc[:2] - gxy))
        if d < best_d:
            best_d = d
            best = p
    pc = best["centroid_global"]
    ex = float(pc[0] - gc[0])
    ey = float(pc[1] - gc[1])
    ez = float(pc[2] - gc[2])
    dist3d = float(np.linalg.norm(pc[:3] - gc[:3]))
    return best_d, ex, ey, ez, dist3d, best


class LocAccumulator:
    """Per-GT localization-error records, sliced by class and distance bin."""

    def __init__(self):
        # Raw per-GT records (matchable set only: nearest same-class <= 4 m).
        self.rec: dict[str, list] = {k: [] for k in (
            "cls", "dbin", "horiz", "dist3d", "ex", "ey", "ez",
            "abs_ex", "abs_ey", "abs_ez",
        )}
        # Coverage bookkeeping over ALL GT.
        self.n_gt = 0
        self.gt_by_cls_dbin: dict[tuple, int] = {}
        self.n_same_class_any = 0          # >=1 same-class proposal anywhere
        self.n_same_class_within4 = 0      # nearest same-class <= 4 m
        # within-threshold counts (same-class), per (cls, dbin).
        self.within: dict[float, dict[tuple, int]] = {
            t: {} for t in BOX_DIST_THRESHOLDS_M
        }
        self.n_props = 0

    def add_gt(self, gt: dict, props_by_cls: dict[str, list[dict]]):
        dbin = _distance_bin(gt["ego_distance_m"])
        if dbin is None:
            return
        cls = gt["cls_name"]
        key = (cls, dbin)
        self.n_gt += 1
        self.gt_by_cls_dbin[key] = self.gt_by_cls_dbin.get(key, 0) + 1

        m = _nearest_same_class(gt, props_by_cls)
        if m is None:
            return
        horiz, ex, ey, ez, dist3d, _p = m
        self.n_same_class_any += 1
        for t in BOX_DIST_THRESHOLDS_M:
            if horiz <= t:
                self.within[t][key] = self.within[t].get(key, 0) + 1
        if horiz > MATCH_RADIUS_M:
            return
        self.n_same_class_within4 += 1
        r = self.rec
        r["cls"].append(cls)
        r["dbin"].append(dbin)
        r["horiz"].append(horiz)
        r["dist3d"].append(dist3d)
        r["ex"].append(ex)
        r["ey"].append(ey)
        r["ez"].append(ez)
        r["abs_ex"].append(abs(ex))
        r["abs_ey"].append(abs(ey))
        r["abs_ez"].append(abs(ez))

    # -- aggregation -------------------------------------------------------
    def _err_block(self, mask=None) -> dict:
        r = self.rec
        def sel(key):
            a = np.asarray(r[key], dtype=np.float64)
            return a if mask is None else a[mask]
        return {
            "horiz_dist": _pctls(sel("horiz")),
            "dist3d": _pctls(sel("dist3d")),
            "abs_ex": _pctls(sel("abs_ex")),
            "abs_ey": _pctls(sel("abs_ey")),
            "abs_ez": _pctls(sel("abs_ez")),
            "ex_signed": _pctls(sel("ex")),
            "ey_signed": _pctls(sel("ey")),
            "ez_signed": _pctls(sel("ez")),
        }

    def summary(self) -> dict:
        cls_arr = np.asarray(self.rec["cls"])
        dbin_arr = np.asarray(self.rec["dbin"])

        per_class = {}
        for c in sorted(NUSC_10_SET):
            mask = cls_arr == c
            if mask.any():
                per_class[c] = self._err_block(mask)

        all_dbins = [f"{int(lo)}-{int(hi)}m" for lo, hi in DIST_BINS]
        per_dbin = {}
        for db in all_dbins:
            mask = dbin_arr == db
            if mask.any():
                per_dbin[db] = self._err_block(mask)

        # within-threshold (same-class) recall, overall + per-class + per-dbin.
        def within_rate(t, key_filter=None) -> dict:
            num = den = 0
            for key, n_gt in self.gt_by_cls_dbin.items():
                if key_filter is not None and not key_filter(key):
                    continue
                den += n_gt
                num += self.within[t].get(key, 0)
            return {"recall": (num / den) if den else None,
                    "n_hit": num, "n_gt": den}

        within_overall = {f"{t}m": within_rate(t) for t in BOX_DIST_THRESHOLDS_M}
        within_class = {
            c: {f"{t}m": within_rate(t, lambda k, c=c: k[0] == c)
                for t in BOX_DIST_THRESHOLDS_M}
            for c in sorted(NUSC_10_SET)
        }
        within_dbin = {
            db: {f"{t}m": within_rate(t, lambda k, db=db: k[1] == db)
                 for t in BOX_DIST_THRESHOLDS_M}
            for db in all_dbins
        }

        return {
            "n_gt_total": self.n_gt,
            "n_props_total": self.n_props,
            "coverage": {
                "n_gt": self.n_gt,
                "frac_same_class_any": (self.n_same_class_any / self.n_gt) if self.n_gt else None,
                "frac_same_class_within_4m": (self.n_same_class_within4 / self.n_gt) if self.n_gt else None,
                "n_matchable_for_error_stats": len(self.rec["horiz"]),
            },
            "error_stats": {
                "note": "matchable set = nearest same-class proposal within 4 m. "
                        "nuScenes mAP matches on BEV centre dist, so ez is "
                        "mAP-irrelevant (reported for completeness).",
                "overall": self._err_block(None),
                "per_class": per_class,
                "per_dbin": per_dbin,
            },
            "within_threshold_same_class_recall": {
                "overall": within_overall,
                "per_class": within_class,
                "per_dbin": within_dbin,
            },
        }


# ---------------------------------------------------------------------------
# Localization-only mAP ceiling via the existing devkit eval.
# ---------------------------------------------------------------------------

def _detbox(translation, size, yaw, score, name, ego_pose) -> dict:
    return {
        "sample_token": "",  # filled by caller via add_boxes
        "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
        "size": [float(size[0]), float(size[1]), float(size[2])],
        "rotation": list(Quaternion(axis=(0.0, 0.0, 1.0), angle=float(yaw)).q),
        "velocity": [0.0, 0.0],
        "ego_translation": [float(ego_pose[0, 3]), float(ego_pose[1, 3]), float(ego_pose[2, 3])],
        "num_pts": 1,
        "detection_name": name,
        "detection_score": float(score),
        "attribute_name": "",
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" / "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    ap.add_argument("--skip-map", action="store_true",
                    help="Skip the devkit localization-only mAP counterfactuals.")
    args = ap.parse_args()

    gamma_cache = Path(args.gamma_cache).resolve()
    if not gamma_cache.exists():
        raise SystemExit(f"cache directory missing: {gamma_cache}")

    date = time.strftime("%Y-%m-%d")
    if args.output_name:
        out_dir = project_root / "results" / args.output_name
    else:
        existing = list(project_root.glob(f"results/{date}_outdoor_localization_error_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_localization_error_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[loc] γ cache    : {gamma_cache}", flush=True)
    print(f"[loc] output dir : {out_dir}", flush=True)

    print("[loc] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[loc] val scenes : {len(val_scenes)}", flush=True)

    acc = LocAccumulator()

    # Counterfactual prediction stores for devkit (token -> [detbox]).
    pred_native: dict[str, list] = {}
    pred_locperfect: dict[str, list] = {}
    gt_store: dict[str, list] = {}

    t0 = time.time()
    n_missing = 0
    for si, sc_tok in enumerate(val_scenes):
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            sample = nusc.get("sample", sa_tok)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))
            T_lidar_to_ego = transform_matrix(lidar_cs["translation"], Quaternion(lidar_cs["rotation"]))
            ego_quat = Quaternion(matrix=ego_pose[:3, :3])
            lidar_to_ego_q = Quaternion(matrix=T_lidar_to_ego[:3, :3])

            gts = _gt_records(nusc, sa_tok, ego_pose)
            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                continue
            props = _proposals_to_global(raw, ego_pose, lidar_to_ego_q, ego_quat)
            acc.n_props += len(props)

            props_by_cls: dict[str, list] = {}
            for p in props:
                props_by_cls.setdefault(p["cls_name"], []).append(p)
            for gt in gts:
                acc.add_gt(gt, props_by_cls)

            if args.skip_map:
                continue

            # --- build devkit GT + counterfactual predictions for this sample
            gt_store[sa_tok] = [
                _detbox(g["translation"], g["size"],
                        g["yaw_global"], -1.0, g["cls_name"], ego_pose)
                for g in gts
            ]
            # same-class GT centres for snapping.
            gt_xy_by_cls: dict[str, list] = {}
            for g in gts:
                gt_xy_by_cls.setdefault(g["cls_name"], []).append(g)

            nat_list, lp_list = [], []
            for p in props:
                tcx, tcy, tcz = (float(p["centroid_global"][0]),
                                 float(p["centroid_global"][1]),
                                 float(p["centroid_global"][2]))
                size = [float(p["size_xy"][0]), float(p["size_xy"][1]), float(p["size_z"])]
                nat_list.append(_detbox([tcx, tcy, tcz], size, p["yaw_global"],
                                        p["score"], p["cls_name"], ego_pose))
                # snap to nearest same-class GT within 4 m (BEV).
                cand = gt_xy_by_cls.get(p["cls_name"])
                sx, sy = tcx, tcy
                if cand:
                    best_g, best_d = None, MATCH_RADIUS_M + 1e-9
                    for g in cand:
                        d = float(np.hypot(g["centroid_global"][0] - tcx,
                                           g["centroid_global"][1] - tcy))
                        if d < best_d:
                            best_d, best_g = d, g
                    if best_g is not None:
                        sx = float(best_g["centroid_global"][0])
                        sy = float(best_g["centroid_global"][1])
                lp_list.append(_detbox([sx, sy, tcz], size, p["yaw_global"],
                                       p["score"], p["cls_name"], ego_pose))
            pred_native[sa_tok] = nat_list
            pred_locperfect[sa_tok] = lp_list

        if (si + 1) % 10 == 0:
            print(f"[loc] scene {si+1}/{len(val_scenes)} done — "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "scene_limit": args.scene_limit,
            "match_radius_m": MATCH_RADIUS_M,
            "box_distance_thresholds_m": list(BOX_DIST_THRESHOLDS_M),
            "distance_bins": [[lo, hi] for lo, hi in DIST_BINS],
            "percentiles": list(PCTLS),
            "matching": "nearest SAME-CLASS proposal by BEV centre distance",
            "n_samples_missing_cache": n_missing,
            "native_mAP_reference": 0.3408,
        },
        "localization": acc.summary(),
    }

    # --- localization-only mAP ceiling -----------------------------------
    if not args.skip_map:
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox
        from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate

        def run_eval(pred_store, label):
            pred_eb, gt_eb = EvalBoxes(), EvalBoxes()
            for tok, gboxes in gt_store.items():
                gt_eb.add_boxes(tok, [DetectionBox.deserialize({**d, "sample_token": tok})
                                      for d in gboxes])
                pboxes = pred_store.get(tok, [])
                pred_eb.add_boxes(tok, [DetectionBox.deserialize({**d, "sample_token": tok})
                                        for d in pboxes])
            summ = nu_evaluate(pred_boxes=pred_eb, gt_boxes=gt_eb,
                               output_dir=str(out_dir / f"map_{label}"),
                               config_name="detection_cvpr_2019")
            return {"mAP": summ.get("mean_ap"), "NDS": summ.get("nd_score"),
                    "per_class_AP": summ.get("label_aps")}

        print("[loc] devkit eval: native (sanity ~0.3408) ...", flush=True)
        payload["map_native"] = run_eval(pred_native, "native")
        print(f"        native mAP = {payload['map_native']['mAP']}", flush=True)
        print("[loc] devkit eval: localization-perfect (snap<=4m, native cls+score) ...", flush=True)
        payload["map_locperfect"] = run_eval(pred_locperfect, "locperfect")
        print(f"        locperfect mAP = {payload['map_locperfect']['mAP']}", flush=True)
        mn = payload["map_native"]["mAP"] or 0.0
        mp = payload["map_locperfect"]["mAP"] or 0.0
        payload["localization_headroom"] = {
            "native_mAP": mn,
            "loc_perfect_mAP": mp,
            "abs_gain": mp - mn,
            "note": "loc_perfect keeps native proposal set + class + score, snaps "
                    "every proposal with a same-class GT within 4 m onto that GT "
                    "centre. Gain over native is the pure-localization headroom.",
        }

    out_file = out_dir / "loc_error_probe.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[loc] wrote {out_file}", flush=True)

    # console head-line
    loc = payload["localization"]
    ov = loc["error_stats"]["overall"]
    cov = loc["coverage"]
    print("\n=== Localization head-line ===")
    print(f"  coverage: same-class any={cov['frac_same_class_any']:.4f} "
          f"within4m={cov['frac_same_class_within_4m']:.4f}")
    print(f"  horiz dist  p50={ov['horiz_dist']['p50']:.3f} p90={ov['horiz_dist']['p90']:.3f}")
    print(f"  |ex|        p50={ov['abs_ex']['p50']:.3f} p90={ov['abs_ex']['p90']:.3f}")
    print(f"  |ey|        p50={ov['abs_ey']['p50']:.3f} p90={ov['abs_ey']['p90']:.3f}")
    print(f"  |ez|        p50={ov['abs_ez']['p50']:.3f} p90={ov['abs_ez']['p90']:.3f}")
    wr = loc["within_threshold_same_class_recall"]["overall"]
    print(f"  same-class recall @0.5/1/2/4m = "
          f"{wr['0.5m']['recall']:.3f}/{wr['1.0m']['recall']:.3f}/"
          f"{wr['2.0m']['recall']:.3f}/{wr['4.0m']['recall']:.3f}")
    if "localization_headroom" in payload:
        h = payload["localization_headroom"]
        print(f"  mAP: native={h['native_mAP']:.4f} loc_perfect={h['loc_perfect_mAP']:.4f} "
              f"gain={h['abs_gain']:+.4f}")


if __name__ == "__main__":
    main()
