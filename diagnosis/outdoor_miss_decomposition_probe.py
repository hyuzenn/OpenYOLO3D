"""Outdoor miss decomposition (Task: split the 28.5% same-class miss).

EVIDENCE-ONLY. Reuses the geometry / GT / proposal loaders from
``diagnosis.outdoor_proposal_recall_probe`` (unchanged) and the devkit eval from
``diagnosis_beta_baseline.evaluate_nuscenes`` (called, never edited).

Background (gravity cache, full val): same-class recall@4m = 0.715, so 28.5% of
GT have no same-class proposal within 4 m. This probe categorises EVERY GT by
its nearest proposal (within the 4 m max-match radius):

  C  same-class proposal within 4 m            (the 71.5% hit)
  B  proposal within 4 m but ALL wrong class   (semantic / classification miss)
  A  no proposal within 4 m at all             (proposal-generation miss)

Then:
  - overall + per-class A/B/C percentages,
  - confusion matrix for B: gt_class -> class of the nearest within-4m proposal,
  - split of the 28.5% miss into A (generation) vs B (classification),
  - devkit mAP for a "relabel-B-correctly, geometry unchanged" counterfactual
    (for each B GT, relabel its nearest within-4m proposal to the GT class;
    everything else native). Gain over native 0.3408 = classification headroom.

"nearest" = BEV/horizontal centre distance (nuScenes mAP matching metric).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10,
    NUSC_10_SET,
    _gt_records,
    _proposals_to_global,
    _load_cached_proposals,
    _list_val_scenes,
    _scene_sample_tokens,
    _distance_bin,
)

MATCH_RADIUS_M = 4.0


def _detbox(translation, size, yaw, score, name, ego_pose, tok) -> dict:
    return {
        "sample_token": tok,
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
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()

    gamma_cache = Path(args.gamma_cache).resolve()
    if not gamma_cache.exists():
        raise SystemExit(f"cache directory missing: {gamma_cache}")

    date = time.strftime("%Y-%m-%d")
    if args.output_name:
        out_dir = project_root / "results" / args.output_name
    else:
        existing = list(project_root.glob(f"results/{date}_outdoor_miss_decomp_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_miss_decomp_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[miss] γ cache    : {gamma_cache}", flush=True)
    print(f"[miss] output dir : {out_dir}", flush=True)

    print("[miss] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[miss] val scenes : {len(val_scenes)}", flush=True)

    # counters[cls] = {"A":n,"B":n,"C":n}
    cat = defaultdict(lambda: {"A": 0, "B": 0, "C": 0})
    # confusion[gt_cls][pred_cls] = n   (category B only)
    confusion = defaultdict(lambda: defaultdict(int))

    gt_store: dict[str, list] = {}
    pred_native: dict[str, list] = {}
    pred_relabelB: dict[str, list] = {}

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

            # index proposals (need stable ids for relabel counterfactual)
            P = []
            for idx, p in enumerate(props):
                P.append({
                    "id": idx,
                    "cls": p["cls_name"],
                    "xy": np.asarray(p["centroid_global"][:2], dtype=np.float64),
                    "z": float(p["centroid_global"][2]),
                    "size": [float(p["size_xy"][0]), float(p["size_xy"][1]), float(p["size_z"])],
                    "yaw": float(p["yaw_global"]),
                    "score": float(p.get("score", 0.0)),
                })

            relabel = {}  # proposal id -> new class (and the GT dist that claimed it)
            relabel_dist = {}

            for g in gts:
                dbin = _distance_bin(g["ego_distance_m"])
                if dbin is None:
                    continue
                gcls = g["cls_name"]
                gxy = np.asarray(g["centroid_global"][:2], dtype=np.float64)
                # nearest same-class and nearest any-class within radius
                best_same = MATCH_RADIUS_M + 1e-9
                best_any = MATCH_RADIUS_M + 1e-9
                best_any_p = None
                for p in P:
                    d = float(np.linalg.norm(p["xy"] - gxy))
                    if d < best_any:
                        best_any = d
                        best_any_p = p
                    if p["cls"] == gcls and d < best_same:
                        best_same = d
                if best_same <= MATCH_RADIUS_M:
                    cat[gcls]["C"] += 1
                elif best_any_p is not None and best_any <= MATCH_RADIUS_M:
                    cat[gcls]["B"] += 1
                    confusion[gcls][best_any_p["cls"]] += 1
                    # relabel that proposal to gt class (nearest GT wins on conflict)
                    pid = best_any_p["id"]
                    if pid not in relabel or best_any < relabel_dist[pid]:
                        relabel[pid] = gcls
                        relabel_dist[pid] = best_any
                else:
                    cat[gcls]["A"] += 1

            if args.skip_map:
                continue

            gt_store[sa_tok] = [
                _detbox(g["translation"], g["size"], g["yaw_global"], -1.0,
                        g["cls_name"], ego_pose, sa_tok)
                for g in gts if _distance_bin(g["ego_distance_m"]) is not None
            ]
            nat, rel = [], []
            for p in P:
                box = _detbox([p["xy"][0], p["xy"][1], p["z"]], p["size"],
                              p["yaw"], p["score"], p["cls"], ego_pose, sa_tok)
                nat.append(box)
                relcls = relabel.get(p["id"], p["cls"])
                relbox = dict(box)
                relbox["detection_name"] = relcls
                rel.append(relbox)
            pred_native[sa_tok] = nat
            pred_relabelB[sa_tok] = rel

        if (si + 1) % 10 == 0:
            print(f"[miss] scene {si+1}/{len(val_scenes)} done — elapsed {time.time()-t0:.0f}s",
                  flush=True)

    # ---- aggregate ----
    tot = {"A": 0, "B": 0, "C": 0}
    per_class = {}
    for c in sorted(NUSC_10_SET):
        a, b, cc = cat[c]["A"], cat[c]["B"], cat[c]["C"]
        n = a + b + cc
        tot["A"] += a; tot["B"] += b; tot["C"] += cc
        if n:
            per_class[c] = {"n": n, "A": a, "B": b, "C": cc,
                            "pA": a / n, "pB": b / n, "pC": cc / n}
    N = tot["A"] + tot["B"] + tot["C"]
    miss = tot["A"] + tot["B"]
    payload = {
        "config": {"gamma_cache": str(gamma_cache), "n_val_scenes": len(val_scenes),
                   "match_radius_m": MATCH_RADIUS_M, "n_samples_missing_cache": n_missing,
                   "categories": {"A": "no proposal within 4m (generation miss)",
                                  "B": "proposal within 4m but wrong class (classification miss)",
                                  "C": "same-class within 4m (hit)"}},
        "overall": {"n_gt": N, "A": tot["A"], "B": tot["B"], "C": tot["C"],
                    "pA": tot["A"] / N, "pB": tot["B"] / N, "pC": tot["C"] / N,
                    "miss_total": miss, "miss_frac": miss / N,
                    "miss_from_generation_A": tot["A"] / miss if miss else None,
                    "miss_from_classification_B": tot["B"] / miss if miss else None},
        "per_class": per_class,
        "confusion_B": {g: dict(sorted(p.items(), key=lambda kv: -kv[1]))
                        for g, p in confusion.items()},
    }

    if not args.skip_map:
        from nuscenes.eval.common.data_classes import EvalBoxes
        from nuscenes.eval.detection.data_classes import DetectionBox
        from diagnosis_beta_baseline.evaluate_nuscenes import evaluate as nu_evaluate

        def run_eval(store, label):
            pred_eb, gt_eb = EvalBoxes(), EvalBoxes()
            for tok, gb in gt_store.items():
                gt_eb.add_boxes(tok, [DetectionBox.deserialize(x) for x in gb])
                pred_eb.add_boxes(tok, [DetectionBox.deserialize(x) for x in store.get(tok, [])])
            s = nu_evaluate(pred_boxes=pred_eb, gt_boxes=gt_eb,
                            output_dir=str(out_dir / f"map_{label}"),
                            config_name="detection_cvpr_2019")
            return {"mAP": s.get("mean_ap"), "NDS": s.get("nd_score"),
                    "per_class_AP": s.get("label_aps")}

        print("[miss] devkit eval: native ...", flush=True)
        payload["map_native"] = run_eval(pred_native, "native")
        print(f"        native mAP={payload['map_native']['mAP']}", flush=True)
        print("[miss] devkit eval: relabel-B (geometry unchanged) ...", flush=True)
        payload["map_relabelB"] = run_eval(pred_relabelB, "relabelB")
        print(f"        relabelB mAP={payload['map_relabelB']['mAP']}", flush=True)
        mn = payload["map_native"]["mAP"] or 0.0
        mr = payload["map_relabelB"]["mAP"] or 0.0
        payload["classification_headroom"] = {"native_mAP": mn, "relabelB_mAP": mr,
                                               "abs_gain": mr - mn}

    out_file = out_dir / "miss_decomp.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[miss] wrote {out_file}", flush=True)

    o = payload["overall"]
    print("\n=== Miss decomposition head-line ===")
    print(f"  n_gt={o['n_gt']}  C(hit)={o['pC']:.4f}  B(wrong-class)={o['pB']:.4f}  A(no-prop)={o['pA']:.4f}")
    print(f"  of the {o['miss_frac']:.4f} miss:  generation(A)={o['miss_from_generation_A']:.3f}  "
          f"classification(B)={o['miss_from_classification_B']:.3f}")
    if "classification_headroom" in payload:
        h = payload["classification_headroom"]
        print(f"  mAP: native={h['native_mAP']:.4f}  relabelB={h['relabelB_mAP']:.4f}  gain={h['abs_gain']:+.4f}")


if __name__ == "__main__":
    main()
