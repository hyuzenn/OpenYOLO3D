"""Characterize category A ("no proposal within 4m") via nearest-proposal distance.

EVIDENCE-ONLY. Reuses geometry / GT / proposal loaders from
``diagnosis.outdoor_proposal_recall_probe`` (unchanged). No devkit eval, no
method edits — pure geometry.

For every GT (valid distance bin):
  - categorize exactly as the miss-decomposition probe:
      C  same-class proposal within 4 m
      B  proposal within 4 m but all wrong class
      A  no proposal within 4 m at all
  - record the UNCAPPED nearest-proposal distance (regardless of class), so we
    can see *how far* the nearest box actually is — especially for category A,
    where it can be 4-6 / 6-10 / >10 m or effectively absent.

Reports per category (A/B/C) and overall:
  - p50 / p75 / p90 / p95 of nearest-proposal distance,
  - histogram: <0.5 / 0.5-1 / 1-2 / 2-4 / 4-6 / 6-10 / 10-20 / >20 m.
For category A specifically: fraction 4-6 / 6-10 / >10 m, to decide whether A is
true proposal absence vs near-miss localization vs far-range detection failure.

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
    NUSC_10_SET,
    _gt_records,
    _proposals_to_global,
    _load_cached_proposals,
    _list_val_scenes,
    _scene_sample_tokens,
    _distance_bin,
)

MATCH_RADIUS_M = 4.0
PCTLS = (50, 75, 90, 95)
# (low_inclusive, high_exclusive) — last bin high=inf
HIST_EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0),
              (4.0, 6.0), (6.0, 10.0), (10.0, 20.0), (20.0, float("inf"))]
HIST_LABELS = ["<0.5", "0.5-1", "1-2", "2-4", "4-6", "6-10", "10-20", ">20"]
NO_PROPOSAL = float("inf")  # sample had zero proposals


def _hist(dists: np.ndarray) -> dict:
    out = {}
    for (lo, hi), lab in zip(HIST_EDGES, HIST_LABELS):
        out[lab] = int(np.count_nonzero((dists >= lo) & (dists < hi)))
    return out


def _pctls(dists: np.ndarray) -> dict:
    finite = dists[np.isfinite(dists)]
    if finite.size == 0:
        return {f"p{p}": None for p in PCTLS}
    return {f"p{p}": float(np.percentile(finite, p)) for p in PCTLS}


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" / "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    gamma_cache = Path(args.gamma_cache).resolve()
    if not gamma_cache.exists():
        raise SystemExit(f"cache directory missing: {gamma_cache}")

    date = time.strftime("%Y-%m-%d")
    if args.output_name:
        out_dir = project_root / "results" / args.output_name
    else:
        existing = list(project_root.glob(f"results/{date}_outdoor_catA_dist_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_catA_dist_v{len(existing)+1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[catA] γ cache    : {gamma_cache}", flush=True)
    print(f"[catA] output dir : {out_dir}", flush=True)

    print("[catA] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[catA] val scenes : {len(val_scenes)}", flush=True)

    # per-category lists of nearest-any-class distances
    dist_by_cat: dict[str, list] = {"A": [], "B": [], "C": []}
    # per (category, class) distances for the A breakdown by class
    dist_A_by_class: dict[str, list] = defaultdict(list)

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

            if props:
                P_xy = np.asarray([p["centroid_global"][:2] for p in props], dtype=np.float64)
                P_cls = [p["cls_name"] for p in props]
            else:
                P_xy = np.empty((0, 2), dtype=np.float64)
                P_cls = []

            for g in gts:
                if _distance_bin(g["ego_distance_m"]) is None:
                    continue
                gcls = g["cls_name"]
                gxy = np.asarray(g["centroid_global"][:2], dtype=np.float64)

                if P_xy.shape[0] == 0:
                    nearest_any = NO_PROPOSAL
                    same_within = False
                    any_within = False
                else:
                    d = np.linalg.norm(P_xy - gxy, axis=1)
                    nearest_any = float(d.min())
                    any_within = nearest_any <= MATCH_RADIUS_M
                    same_mask = np.array([c == gcls for c in P_cls])
                    if same_mask.any():
                        same_within = float(d[same_mask].min()) <= MATCH_RADIUS_M
                    else:
                        same_within = False

                if same_within:
                    cat = "C"
                elif any_within:
                    cat = "B"
                else:
                    cat = "A"
                dist_by_cat[cat].append(nearest_any)
                if cat == "A":
                    dist_A_by_class[gcls].append(nearest_any)

        if (si + 1) % 20 == 0:
            print(f"[catA] scene {si+1}/{len(val_scenes)} — elapsed {time.time()-t0:.0f}s",
                  flush=True)

    # ---- aggregate ----
    def cat_block(dists_list):
        d = np.asarray(dists_list, dtype=np.float64)
        n = int(d.size)
        n_inf = int(np.count_nonzero(~np.isfinite(d)))
        block = {"n": n, "n_no_proposal_in_sample": n_inf,
                 "pctls": _pctls(d), "hist": _hist(d)}
        if n:
            block["hist_frac"] = {k: v / n for k, v in block["hist"].items()}
        return block

    all_d = dist_by_cat["A"] + dist_by_cat["B"] + dist_by_cat["C"]
    payload = {
        "config": {"gamma_cache": str(gamma_cache), "n_val_scenes": len(val_scenes),
                   "match_radius_m": MATCH_RADIUS_M, "n_samples_missing_cache": n_missing,
                   "metric": "BEV/horizontal nearest-proposal distance (uncapped), regardless of class",
                   "categories": {"A": "no proposal within 4m",
                                  "B": "proposal within 4m, all wrong class",
                                  "C": "same-class proposal within 4m"}},
        "overall": cat_block(all_d),
        "by_category": {c: cat_block(dist_by_cat[c]) for c in ("A", "B", "C")},
    }

    # Category-A specific question (Q4)
    A = np.asarray(dist_by_cat["A"], dtype=np.float64)
    nA = int(A.size)
    if nA:
        f_4_6 = int(np.count_nonzero((A >= 4.0) & (A < 6.0))) / nA
        f_6_10 = int(np.count_nonzero((A >= 6.0) & (A < 10.0))) / nA
        f_gt10 = int(np.count_nonzero(A >= 10.0)) / nA  # includes inf (no proposal)
        f_gt10_finite = int(np.count_nonzero(np.isfinite(A) & (A >= 10.0))) / nA
        f_inf = int(np.count_nonzero(~np.isfinite(A))) / nA
        payload["category_A_focus"] = {
            "n": nA,
            "frac_4_6m": f_4_6,
            "frac_6_10m": f_6_10,
            "frac_gt10m": f_gt10,
            "frac_gt10m_finite": f_gt10_finite,
            "frac_no_proposal_in_sample": f_inf,
            "interpretation_keys": {
                "near_miss_localization": "4-6m",
                "far_range_detection_failure": "6-10m and finite >10m",
                "true_proposal_absence": ">20m / no-proposal-in-sample",
            },
        }
        # per-class A nearest-distance percentiles
        payload["category_A_by_class"] = {}
        for c in sorted(NUSC_10_SET):
            if dist_A_by_class.get(c):
                payload["category_A_by_class"][c] = cat_block(dist_A_by_class[c])

    out_file = out_dir / "catA_distance.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[catA] wrote {out_file}", flush=True)

    # ---- console summary ----
    print("\n=== nearest-proposal distance (uncapped, BEV) by category ===")
    for c in ("A", "B", "C", "overall"):
        blk = payload["overall"] if c == "overall" else payload["by_category"][c]
        p = blk["pctls"]
        print(f"  {c:<8} n={blk['n']:>7}  p50={p['p50']}  p75={p['p75']}  "
              f"p90={p['p90']}  p95={p['p95']}")
    print("\n  histograms (fraction):")
    print("    " + " ".join(f"{l:>7}" for l in HIST_LABELS))
    for c in ("A", "B", "C"):
        hf = payload["by_category"][c].get("hist_frac", {})
        print(f"  {c}: " + " ".join(f"{hf.get(l, 0.0):7.3f}" for l in HIST_LABELS))
    if "category_A_focus" in payload:
        f = payload["category_A_focus"]
        print(f"\n  category A (n={f['n']}): 4-6m={f['frac_4_6m']:.3f}  "
              f"6-10m={f['frac_6_10m']:.3f}  >10m={f['frac_gt10m']:.3f} "
              f"(no-proposal-in-sample={f['frac_no_proposal_in_sample']:.3f})")


if __name__ == "__main__":
    main()
