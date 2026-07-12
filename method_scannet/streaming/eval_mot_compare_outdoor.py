"""E1 — MOT-metric comparison for OV-TCS validation (nuScenes outdoor).

Compares OV-TCS_C against established GT-track MOT metrics (HOTA / DetA /
AssA, IDF1 / IDP / IDR, MOTA / MOTP; AMOTA / AMOTP via the official devkit)
on the SAME tracker outputs, ego- vs global-frame association arms.
EVALUATION ONLY — pure cache replay over the gravity-corrected CenterPoint
cache; no new detector inference, no code in the existing pipeline modified.

Design notes:
- Rides NativeTemporalNuScenesEvaluator (baseline axis, class-agnostic
  association) exactly like eval_ovtcs_track_outdoor, so the OV-TCS side of
  the comparison is the frozen pipeline bit-for-bit.
- The only hook is a runtime wrapper around _detection_box_dict that ADDS a
  "tracking_id" key to the emitted per-sample box dicts (the existing code
  drops the track id at emission). Additive key, existing behavior untouched.
- Per-scene MOT metrics are class-agnostic with a 2.0 m BEV center-distance
  gate (the nuScenes tracking TP threshold), matching the class-agnostic
  association under test and the 10-class OV-TCS population.
- HOTA/DetA/AssA follow the TrackEval reference algorithm (Luiten et al.,
  IJCV 2021) with similarity s = max(0, 1 - d/2.0 m); nuScenes has no
  official HOTA. Unit-tested in tests/test_mot_compare.py.
- AMOTA/AMOTP/MOTA/MOTP (dataset-level, 7 tracking classes) come from the
  UNMODIFIED official devkit TrackingEval on a standard submission JSON
  (--devkit-eval; requires the full 150-scene val split).

Run (PBS container; CPU-only cache replay):
  python -u -m method_scannet.streaming.eval_mot_compare_outdoor \
    --cp-cache-dir results/outdoor_native_temporal_cpcache_thr000_single_gravity \
    --output results/<date>_outdoor_mot_compare_v01/outputs \
    --arms ego global [--scene-limit 5] [--devkit-eval]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

DIST_TP_M = 2.0                      # nuScenes tracking TP center-distance gate
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)   # 19 thresholds, TrackEval default

TRACKING_NAMES = {"bicycle", "bus", "car", "motorcycle", "pedestrian",
                  "trailer", "truck"}     # devkit 7-class tracking vocabulary


# ---------------------------------------------------------------------------
# HOTA (TrackEval reference algorithm, center-distance similarity).
# ---------------------------------------------------------------------------
def hota_from_frames(frames: list) -> dict:
    """frames: list of (gt_ids, pred_ids, sim) per timestep, sim in [0,1],
    shape (len(gt_ids), len(pred_ids)). Returns HOTA/DetA/AssA (mean over
    alpha) plus the per-alpha arrays."""
    gt_ids_all = sorted({g for gt, _, _ in frames for g in gt})
    pr_ids_all = sorted({p for _, pr, _ in frames for p in pr})
    gmap = {g: i for i, g in enumerate(gt_ids_all)}
    pmap = {p: i for i, p in enumerate(pr_ids_all)}
    nG, nP = len(gt_ids_all), len(pr_ids_all)
    if nG == 0 or nP == 0:
        z = np.zeros(len(HOTA_ALPHAS))
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0,
                "HOTA_alpha": z.tolist(), "DetA_alpha": z.tolist(),
                "AssA_alpha": z.tolist(),
                "n_gt_dets": sum(len(f[0]) for f in frames),
                "n_pred_dets": sum(len(f[1]) for f in frames)}

    # Pass 1: global alignment scores (potential matches).
    potential = np.zeros((nG, nP))
    gt_cnt = np.zeros(nG)
    pr_cnt = np.zeros(nP)
    for gt, pr, sim in frames:
        gi = [gmap[g] for g in gt]
        pi = [pmap[p] for p in pr]
        for g in gi:
            gt_cnt[g] += 1
        for p in pi:
            pr_cnt[p] += 1
        if len(gi) and len(pi):
            s = np.asarray(sim, dtype=np.float64)
            denom = s.sum(0)[np.newaxis, :] + s.sum(1)[:, np.newaxis] - s
            with np.errstate(divide="ignore", invalid="ignore"):
                sim_iou = np.where(denom > 1e-12, s / denom, 0.0)
            potential[np.ix_(gi, pi)] += sim_iou
    align = potential / np.maximum(
        1e-12, gt_cnt[:, np.newaxis] + pr_cnt[np.newaxis, :] - potential)

    # Pass 2: per-alpha matching.
    nA = len(HOTA_ALPHAS)
    tp = np.zeros(nA)
    fn = np.zeros(nA)
    fp = np.zeros(nA)
    match_cnt = [np.zeros((nG, nP)) for _ in range(nA)]
    for gt, pr, sim in frames:
        gi = np.asarray([gmap[g] for g in gt], dtype=int)
        pi = np.asarray([pmap[p] for p in pr], dtype=int)
        if len(gi) == 0 or len(pi) == 0:
            fn += len(gi)
            fp += len(pi)
            continue
        s = np.asarray(sim, dtype=np.float64)
        score = align[np.ix_(gi, pi)] * s
        r, c = linear_sum_assignment(-score)
        for a, alpha in enumerate(HOTA_ALPHAS):
            ok = s[r, c] >= alpha - 1e-12
            m = int(ok.sum())
            tp[a] += m
            fn[a] += len(gi) - m
            fp[a] += len(pi) - m
            mc = match_cnt[a]
            for rr, cc in zip(r[ok], c[ok]):
                mc[gi[rr], pi[cc]] += 1
    det_a = tp / np.maximum(1e-12, tp + fn + fp)
    ass_a = np.zeros(nA)
    for a in range(nA):
        mc = match_cnt[a]
        if tp[a] == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ass = mc / np.maximum(
                1e-12, gt_cnt[:, np.newaxis] + pr_cnt[np.newaxis, :] - mc)
        ass_a[a] = float((ass * mc).sum() / tp[a])
    hota = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota.mean()), "DetA": float(det_a.mean()),
            "AssA": float(ass_a.mean()),
            "HOTA_alpha": hota.tolist(), "DetA_alpha": det_a.tolist(),
            "AssA_alpha": ass_a.tolist(),
            # per-alpha counts for exact cross-scene combination (track ids
            # never cross scenes, so the global problem is block-diagonal).
            "tp_alpha": tp.tolist(), "fn_alpha": fn.tolist(),
            "fp_alpha": fp.tolist(),
            "ass_sum_alpha": (ass_a * tp).tolist(),
            "n_gt_dets": int(gt_cnt.sum()), "n_pred_dets": int(pr_cnt.sum())}


def combine_hota(scene_results: list) -> dict:
    """Exact dataset-level HOTA from per-scene counts (valid because id sets
    are disjoint across scenes -> block-diagonal alignment/association)."""
    nA = len(HOTA_ALPHAS)
    tp = np.zeros(nA); fn = np.zeros(nA); fp = np.zeros(nA)
    ass_sum = np.zeros(nA)
    for r in scene_results:
        tp += np.asarray(r["tp_alpha"]); fn += np.asarray(r["fn_alpha"])
        fp += np.asarray(r["fp_alpha"])
        ass_sum += np.asarray(r["ass_sum_alpha"])
    det_a = tp / np.maximum(1e-12, tp + fn + fp)
    ass_a = np.where(tp > 0, ass_sum / np.maximum(1e-12, tp), 0.0)
    hota = np.sqrt(det_a * ass_a)
    return {"HOTA": float(hota.mean()), "DetA": float(det_a.mean()),
            "AssA": float(ass_a.mean()),
            "HOTA_alpha": hota.tolist(), "DetA_alpha": det_a.tolist(),
            "AssA_alpha": ass_a.tolist()}


def combine_clear_id(rows: list) -> dict:
    """Exact dataset-level CLEAR/ID metrics from per-scene counts (id sets
    disjoint across scenes -> the global ID bipartite matching decomposes)."""
    idtp = sum(r["IDTP"] for r in rows)
    idfp = sum(r["IDFP"] for r in rows)
    idfn = sum(r["IDFN"] for r in rows)
    fp = sum(r["FP"] for r in rows); fn = sum(r["FN"] for r in rows)
    ids = sum(r["IDS"] for r in rows); n_gt = sum(r["n_gt"] for r in rows)
    n_m = sum(r["n_matches"] for r in rows)
    motp = (sum(r["MOTP_m"] * r["n_matches"] for r in rows) / n_m
            if n_m else None)
    return {"MOTA": 1.0 - (fn + fp + ids) / max(1, n_gt), "MOTP_m": motp,
            "IDF1": 2 * idtp / max(1e-12, 2 * idtp + idfp + idfn),
            "IDP": idtp / max(1e-12, idtp + idfp),
            "IDR": idtp / max(1e-12, idtp + idfn),
            "IDS": ids, "FRAG": sum(r["FRAG"] for r in rows),
            "FP": fp, "FN": fn, "n_gt": n_gt, "n_matches": n_m}


# ---------------------------------------------------------------------------
# Per-scene CLEAR/identity metrics via motmetrics (2 m gate, class-agnostic).
# ---------------------------------------------------------------------------
def clear_id_from_frames(frames: list) -> dict:
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    # motmetrics 1.4 + pandas 2.x rejects string ids -> map to ints locally.
    gmap: dict = {}
    pmap: dict = {}
    for gt_raw, pr_raw, sim in frames:
        gt = [gmap.setdefault(g, len(gmap)) for g in gt_raw]
        pr = [pmap.setdefault(p, -1 - len(pmap)) for p in pr_raw]
        s = np.asarray(sim, dtype=np.float64)
        # motmetrics wants a cost; gate at sim<=0 (i.e. d>=2 m).
        d = np.where(s > 0.0, DIST_TP_M * (1.0 - s), np.nan) \
            if s.size else np.empty((len(gt), len(pr)))
        acc.update(gt, pr, d)
    mh = mm.metrics.create()
    res = mh.compute(acc, metrics=[
        "mota", "motp", "idf1", "idp", "idr", "idtp", "idfp", "idfn",
        "num_switches", "num_fragmentations", "num_false_positives",
        "num_misses", "num_matches", "num_objects"], name="scene")
    row = res.iloc[0]
    n_matches = int(row["num_matches"])
    return {"MOTA": float(row["mota"]),
            "MOTP_m": float(row["motp"]) if n_matches else 0.0,
            "IDF1": float(row["idf1"]), "IDP": float(row["idp"]),
            "IDR": float(row["idr"]), "IDTP": int(row["idtp"]),
            "IDFP": int(row["idfp"]), "IDFN": int(row["idfn"]),
            "IDS": int(row["num_switches"]),
            "FRAG": int(row["num_fragmentations"]),
            "FP": int(row["num_false_positives"]),
            "FN": int(row["num_misses"]), "n_matches": n_matches,
            "n_gt": int(row["num_objects"])}


def build_frames(sample_tokens, gt_by_tok, pred_by_tok):
    """Per-timestep (gt_ids, pred_ids, sim) with BEV center-distance
    similarity s = max(0, 1 - d/2m)."""
    frames = []
    for tok in sample_tokens:
        gts = gt_by_tok.get(tok, [])
        prs = pred_by_tok.get(tok, [])
        gt_ids = [g["instance_token"] for g in gts]
        pr_ids = [p["tracking_id"] for p in prs]
        if gts and prs:
            g_xy = np.asarray([g["translation"][:2] for g in gts])
            p_xy = np.asarray([p["translation"][:2] for p in prs])
            d = np.linalg.norm(g_xy[:, None, :] - p_xy[None, :, :], axis=2)
            sim = np.maximum(0.0, 1.0 - d / DIST_TP_M)
        else:
            sim = np.zeros((len(gts), len(prs)))
        frames.append((gt_ids, pr_ids, sim))
    return frames


# ---------------------------------------------------------------------------
# Per-scene OV-TCS_C (identical formula/aggregation to compute_variant_metrics).
# ---------------------------------------------------------------------------
def ovtcs_c_per_scene(track_seq: dict, stride: int) -> dict:
    per_scene = defaultdict(list)
    for gid, seq in track_seq.items():
        L = len(seq)
        if L == 0:
            continue
        sw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        csr = (sw / (L - 1)) if L >= 2 else 0.0
        c = (1.0 - 1.0 / L) * (1.0 - csr)
        per_scene[int(gid) // stride].append(c)
    return {s: {"ovtcs_C_mean": float(np.mean(v)), "n_tracks": len(v)}
            for s, v in per_scene.items()}


# ---------------------------------------------------------------------------
# Tracking-submission JSON (official devkit format, 7 tracking classes).
# ---------------------------------------------------------------------------
def build_submission(pred_by_tok: dict) -> dict:
    results = {}
    for tok, preds in pred_by_tok.items():
        boxes = []
        for p in preds:
            if p["detection_name"] not in TRACKING_NAMES:
                continue
            boxes.append({
                "sample_token": tok,
                "translation": p["translation"],
                "size": p["size"],
                "rotation": p["rotation"],
                "velocity": p["velocity"],
                "tracking_id": p["tracking_id"],
                "tracking_name": p["detection_name"],
                "tracking_score": p["detection_score"],
            })
        results[tok] = boxes
    return {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                     "use_map": False, "use_external": False},
            "results": results}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--proposal-source", default="gamma",
                    choices=["gamma", "detguided", "hybrid"])
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--arms", nargs="+", default=["ego", "global"],
                    choices=["ego", "global"])
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--devkit-eval", action="store_true",
                    help="run the official nuScenes TrackingEval (AMOTA/AMOTP/"
                         "MOTA/MOTP, 7 classes) — needs the FULL val split")
    args = ap.parse_args()

    import method_scannet.streaming.nuscenes_native_evaluator as nne
    from dataloaders.nuscenes_loader import NuScenesLoader
    from method_scannet.streaming.nuscenes_evaluator import SCENE_ID_STRIDE

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # Runtime hook: retain the track id on emitted boxes. Additive key only;
    # the wrapped function and every existing consumer are unchanged.
    _orig_box_dict = nne._detection_box_dict

    def _box_dict_with_tid(global_id, **kw):
        d = _orig_box_dict(global_id=global_id, **kw)
        d["tracking_id"] = str(int(global_id))
        return d
    nne._detection_box_dict = _box_dict_with_tid

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = nne._list_val_scenes(loader)
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    print(f"  val scenes={len(scenes)} source={args.proposal_source} "
          f"arms={args.arms}", flush=True)

    summary = {"config": vars(args), "arms": {}}
    for arm in args.arms:
        print(f"\n=== arm: {arm} association (class-agnostic, baseline axis) ===",
              flush=True)
        ev = nne.NativeTemporalNuScenesEvaluator(
            loader=loader, cp_proposals=None, cp_cache_dir=args.cp_cache_dir,
            proposal_source=args.proposal_source,
            class_agnostic_association=True,
            association_frame=arm,
            collect_track_metrics=True)
        ev.install_axis("baseline")
        ev.begin_axis()
        t0 = time.time()
        scene_tokens_map = {}
        for i, sc in enumerate(scenes):
            toks = ev._scene_sample_tokens(sc)
            scene_tokens_map[i] = toks
            ev.run_scene(sc, scene_idx=i)
            if (i + 1) % 25 == 0 or (i + 1) == len(scenes):
                print(f"    {i+1}/{len(scenes)} tracks={len(ev._track_seq)} "
                      f"{time.time()-t0:.0f}s", flush=True)

        # Frozen-pipeline OV-TCS aggregate (gate anchor) + per-scene values.
        vm = ev.compute_variant_metrics()
        print(f"  OV-TCS_C mean = {vm['ov_tcs']['C_mean']:.4f} "
              f"n_tracks={vm['n_tracks']}", flush=True)
        scene_ovtcs = ovtcs_c_per_scene(ev._track_seq, SCENE_ID_STRIDE)

        # Per-scene MOT metrics on the identical outputs; dataset-level values
        # are combined exactly from per-scene counts (ids are scene-disjoint).
        per_scene = {}
        hota_raw, clear_rows = [], []
        scalar = ("HOTA", "DetA", "AssA", "n_gt_dets", "n_pred_dets")
        for i, sc in enumerate(scenes):
            frames = build_frames(scene_tokens_map[i],
                                  ev.per_sample_gt_boxes,
                                  ev.per_sample_pred_boxes)
            h = hota_from_frames(frames)
            c = clear_id_from_frames(frames)
            hota_raw.append(h)
            clear_rows.append(c)
            per_scene[i] = {"scene_token": sc,
                            **{k: h[k] for k in scalar}, **c,
                            **scene_ovtcs.get(i, {"ovtcs_C_mean": None,
                                                  "n_tracks": 0})}
        overall_hota = combine_hota(hota_raw)
        overall_clear = combine_clear_id(clear_rows)

        sub = build_submission(ev.per_sample_pred_boxes)
        sub_path = out / f"tracking_submission_{arm}.json"
        sub_path.write_text(json.dumps(sub))
        # Format check so a smoke run catches submission problems the devkit
        # would only surface at full scale.
        from nuscenes.eval.tracking.data_classes import TrackingBox
        for bx in list(sub["results"].values())[:3]:
            for b in bx[:5]:
                TrackingBox.deserialize(b)
        (out / f"per_scene_{arm}.json").write_text(json.dumps(per_scene, indent=2))
        (out / f"variant_metrics_{arm}.json").write_text(json.dumps(vm, indent=2))
        summary["arms"][arm] = {
            "ovtcs": vm, "overall_hota": overall_hota,
            "overall_clear_id": overall_clear,
            "n_scenes": len(scenes),
        }
        print(f"  overall: HOTA={overall_hota['HOTA']:.4f} "
              f"DetA={overall_hota['DetA']:.4f} AssA={overall_hota['AssA']:.4f} "
              f"IDF1={overall_clear['IDF1']:.4f} MOTA={overall_clear['MOTA']:.4f} "
              f"IDS={overall_clear['IDS']}", flush=True)

        if args.devkit_eval:
            print("  official devkit TrackingEval ...", flush=True)
            from nuscenes.eval.common.config import config_factory
            from nuscenes.eval.tracking.evaluate import TrackingEval
            dk_out = out / f"devkit_eval_{arm}"
            dk_out.mkdir(exist_ok=True)
            te = TrackingEval(config=config_factory("tracking_nips_2019"),
                              result_path=str(sub_path), eval_set="val",
                              output_dir=str(dk_out),
                              nusc_version=loader.version,
                              nusc_dataroot=loader.dataroot, verbose=True)
            dk = te.main(render_curves=False)
            summary["arms"][arm]["devkit"] = {
                k: dk[k] for k in ("amota", "amotp", "mota", "motp", "ids",
                                   "frag", "recall", "tid", "lgd")
                if k in dk}

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}/summary.json per_scene_*.json "
          f"tracking_submission_*.json", flush=True)


if __name__ == "__main__":
    main()
