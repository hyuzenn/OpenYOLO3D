"""E1: GT-based MOT metrics + GT-free predictors from one cell's tracks.json.

Per pre-registration results/2026-07-18_e1_prereg_v01/PREREGISTRATION.md §1:
- identical (emitted) track set on both sides, no per-side filtering;
- HOTA/AssA/DetA/IDF1: class-agnostic, similarity = clip(1 - d_xy/2.0, 0, 1)
  (center-distance <= 2.0 m), per-scene sequences via TrackEval;
- AMOTA/AMOTP: official nuscenes-devkit tracking eval (class-aware, 7 classes).

Outputs into the cell's axis dir: e1_metrics.json (summary) and
e1_perscene.pkl (per-scene TrackEval res dicts + per-scene GT-free arrays,
for the preregistered scene-bootstrap in e1_stats.py).

Usage: python scripts/e1_gt_metrics.py --cell results/.../cells/<name> [--no-amota]
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MATCH_DIST_M = 2.0
NUM_CLASSES = 10  # nuScenes-10 (entropy normalization for formulation A)
TRACKING_7 = {"bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"}


def find_tracks_json(cell: Path) -> Path:
    hits = sorted(cell.glob("axis_*/tracks.json"))
    if len(hits) != 1:
        raise SystemExit(f"{cell}: expected exactly 1 axis_*/tracks.json, got {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# GT-free predictors from the emitted per-frame stream.
# ---------------------------------------------------------------------------
def gt_free_per_scene(pred: dict, scene_of_tok: dict) -> dict:
    """Per-track emitted-label sequences -> per-scene arrays of
    L_norm, (1-CSR), OVTCS A/B/C. Track's scene = scene of its first frame
    (tracks never span scenes: gids are scene-offset)."""
    seq: dict[int, list[str]] = defaultdict(list)
    first_scene: dict[int, int] = {}
    for tok, boxes in pred.items():  # json preserves file (chronological) order
        s = scene_of_tok[tok]
        for b in boxes:
            g = int(b["tracking_id"])
            seq[g].append(b["detection_name"])
            first_scene.setdefault(g, s)

    per_scene: dict[int, dict[str, list]] = defaultdict(
        lambda: {"L_norm": [], "one_minus_csr": [], "A": [], "B": [], "C": [], "len": []})
    for g, labels in seq.items():
        L = len(labels)
        cnt = Counter(labels)
        H = -sum((c / L) * math.log2(c / L) for c in cnt.values())
        Hn = H / math.log2(NUM_CLASSES)
        DR = max(cnt.values()) / L
        sw = sum(1 for a, b in zip(labels[:-1], labels[1:]) if a != b)
        csr = (sw / (L - 1)) if L >= 2 else 0.0
        Ln = 1.0 - 1.0 / L
        d = per_scene[first_scene[g]]
        d["L_norm"].append(Ln)
        d["one_minus_csr"].append(1.0 - csr)
        d["A"].append(Ln * (1.0 - Hn))
        d["B"].append(Ln * DR)
        d["C"].append(Ln * (1.0 - csr))
        d["len"].append(L)
    return {s: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
            for s, d in per_scene.items()}


# ---------------------------------------------------------------------------
# TrackEval per-scene HOTA / Identity.
# ---------------------------------------------------------------------------
def trackeval_per_scene(pred: dict, gt: dict, scene_of_tok: dict) -> dict:
    from trackeval.metrics import HOTA, Identity

    toks_by_scene: dict[int, list[str]] = defaultdict(list)
    for tok in pred:  # chronological within scene
        toks_by_scene[scene_of_tok[tok]].append(tok)

    hota, ident = HOTA(), Identity({"THRESHOLD": 1e-9, "PRINT_CONFIG": False})
    res = {}
    for s, toks in toks_by_scene.items():
        gt_ids_raw, tr_ids_raw, sims = [], [], []
        for tok in toks:
            g_boxes = gt.get(tok, [])
            p_boxes = pred.get(tok, [])
            gt_ids_raw.append([b["instance_token"] for b in g_boxes])
            tr_ids_raw.append([int(b["tracking_id"]) for b in p_boxes])
            g_xy = np.asarray([b["translation"][:2] for b in g_boxes], dtype=np.float64)
            p_xy = np.asarray([b["translation"][:2] for b in p_boxes], dtype=np.float64)
            if len(g_boxes) and len(p_boxes):
                d = np.linalg.norm(g_xy[:, None, :] - p_xy[None, :, :], axis=2)
                sims.append(np.clip(1.0 - d / MATCH_DIST_M, 0.0, 1.0))
            else:
                sims.append(np.zeros((len(g_boxes), len(p_boxes))))
        gmap = {t: i for i, t in enumerate(sorted({t for f in gt_ids_raw for t in f}))}
        tmap = {t: i for i, t in enumerate(sorted({t for f in tr_ids_raw for t in f}))}
        data = {
            "num_timesteps": len(toks),
            "num_gt_ids": len(gmap), "num_tracker_ids": len(tmap),
            "num_gt_dets": sum(len(f) for f in gt_ids_raw),
            "num_tracker_dets": sum(len(f) for f in tr_ids_raw),
            "gt_ids": [np.asarray([gmap[t] for t in f], dtype=int) for f in gt_ids_raw],
            "tracker_ids": [np.asarray([tmap[t] for t in f], dtype=int) for f in tr_ids_raw],
            "similarity_scores": sims,
        }
        res[s] = {"HOTA": hota.eval_sequence(data), "Identity": ident.eval_sequence(data)}
    return res


def combine(res_per_scene: dict) -> dict:
    from trackeval.metrics import HOTA, Identity
    hota, ident = HOTA(), Identity({"THRESHOLD": 1e-9, "PRINT_CONFIG": False})
    h = hota.combine_sequences({s: r["HOTA"] for s, r in res_per_scene.items()})
    i = ident.combine_sequences({s: r["Identity"] for s, r in res_per_scene.items()})
    return {
        "HOTA": float(np.mean(h["HOTA"])), "AssA": float(np.mean(h["AssA"])),
        "DetA": float(np.mean(h["DetA"])), "LocA": float(np.mean(h["LocA"])),
        "IDF1": float(i["IDF1"]),
    }


# ---------------------------------------------------------------------------
# Official nuScenes tracking eval (AMOTA/AMOTP).
# ---------------------------------------------------------------------------
def run_amota(pred: dict, out_dir: Path, dataroot: str, version: str) -> dict:
    from nuscenes import NuScenes
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    val_scenes = set(create_splits_scenes()["val"])
    results = {}
    for sc in nusc.scene:
        if sc["name"] not in val_scenes:
            continue
        tok = sc["first_sample_token"]
        while tok:
            results[tok] = []
            tok = nusc.get("sample", tok)["next"]
    n_missing = sum(1 for t in results if t not in pred)
    for tok, boxes in pred.items():
        results[tok] = [{
            "sample_token": tok,
            "translation": b["translation"], "size": b["size"],
            "rotation": b["rotation"], "velocity": b["velocity"],
            "tracking_id": str(b["tracking_id"]),
            "tracking_name": b["detection_name"],
            "tracking_score": float(b["detection_score"]),
        } for b in boxes if b["detection_name"] in TRACKING_7]

    sub = {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False,
                    "use_map": False, "use_external": False},
           "results": results}
    res_path = out_dir / "amota_submission.json"
    res_path.write_text(json.dumps(sub))
    te = TrackingEval(config=config_factory("tracking_nips_2019"),
                      result_path=str(res_path), eval_set="val",
                      output_dir=str(out_dir / "amota_eval"),
                      nusc_version=version, nusc_dataroot=dataroot, verbose=False)
    summ = te.main(render_curves=False)
    res_path.unlink()  # 6019-sample submission json is large; eval dir keeps summary
    return {"amota": float(summ["amota"]), "amotp": float(summ["amotp"]),
            "n_val_samples_missing_pred": n_missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--no-amota", action="store_true")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    args = ap.parse_args()

    cell = Path(args.cell)
    tj = find_tracks_json(cell)
    axis_dir = tj.parent
    print(f"[e1] loading {tj}", flush=True)
    d = json.loads(tj.read_text())
    pred, gt = d["pred"], d["gt"]
    scene_of_tok = {t: int(s) for t, s in d["sample_scene_idx"].items()}

    gf = gt_free_per_scene(pred, scene_of_tok)
    te = trackeval_per_scene(pred, gt, scene_of_tok)
    comb = combine(te)

    def pooled(key):
        a = np.concatenate([v[key] for v in gf.values()]) if gf else np.array([])
        return float(a.mean()) if a.size else None

    summary = {
        "n_scenes": len(te),
        "n_tracks": int(sum(v["len"].size for v in gf.values())),
        "gt_free": {k: pooled(k) for k in ("L_norm", "one_minus_csr", "A", "B", "C")},
        "gt_based": comb,
    }
    if not args.no_amota:
        print("[e1] running official AMOTA eval ...", flush=True)
        summary["gt_based"].update(run_amota(pred, axis_dir, args.dataroot, args.version))

    (axis_dir / "e1_metrics.json").write_text(json.dumps(summary, indent=2))
    with open(axis_dir / "e1_perscene.pkl", "wb") as f:
        pickle.dump({"trackeval": te, "gt_free": gf}, f)
    print(f"[e1] {cell.name}: {json.dumps(summary['gt_based'])} "
          f"C={summary['gt_free']['C']}", flush=True)


if __name__ == "__main__":
    main()
