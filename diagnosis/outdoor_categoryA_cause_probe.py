"""Determine the CAUSE of category A ("no proposal within 4m").

EVIDENCE-ONLY. Reuses geometry / GT / proposal loaders from
``diagnosis.outdoor_proposal_recall_probe`` (unchanged). No devkit eval, no
method edits — pure geometry.

For every GT (valid distance bin) we categorise exactly as before:
  C  same-class proposal within 4 m
  B  proposal within 4 m but all wrong class
  A  no proposal within 4 m at all
and for each A GT we additionally record:
  - gt class, gt ego-distance + bin,
  - nearest proposal distance (any class, uncapped) + its class,
  - nearest SAME-class proposal distance (uncapped).

Outputs:
  Q2  A-rate by distance bin, by class, by (class x distance).
  Q4  fraction of A that would leave A if the radius relaxed 4->6 / 4->10 m,
      under both "any-class proposal appears" and "same-class proposal appears".
  Q5  decomposition A = localization-near-miss + class-confusion + true void,
      using a 6 m near-band:
        near-miss      : a SAME-class proposal exists within 6 m,
        class-confusion: no same-class <=6 m but SOME proposal <=6 m,
        true void      : nothing within 6 m.

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
    DIST_BINS,
    _gt_records,
    _proposals_to_global,
    _load_cached_proposals,
    _list_val_scenes,
    _scene_sample_tokens,
    _distance_bin,
)

MATCH_RADIUS_M = 4.0
NEAR_BAND_M = 6.0          # near-miss / class-confusion band for Q5
RELAX_RADII = (6.0, 10.0)  # Q4
INF = float("inf")

# _distance_bin() already returns canonical string labels ("0-15m", ...).
BIN_LABELS = [f"{int(lo)}-{int(hi)}m" for lo, hi in DIST_BINS]


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
        existing = list(project_root.glob(f"results/{date}_outdoor_catA_cause_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_catA_cause_v{len(existing)+1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[catA-cause] γ cache    : {gamma_cache}", flush=True)
    print(f"[catA-cause] output dir : {out_dir}", flush=True)

    print("[catA-cause] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[catA-cause] val scenes : {len(val_scenes)}", flush=True)

    bin_labels = BIN_LABELS

    # denominators (all valid GT) and A-counts
    tot_by_bin = defaultdict(int)
    A_by_bin = defaultdict(int)
    tot_by_cls = defaultdict(int)
    A_by_cls = defaultdict(int)
    tot_by_cd = defaultdict(int)   # (cls,binlabel)
    A_by_cd = defaultdict(int)

    # per-A records (lightweight: dist + nearest-any + nearest-same)
    A_nearest_any = []     # uncapped nearest-any distance
    A_nearest_same = []    # uncapped nearest same-class distance (inf if none)
    A_near_cls = []        # class of nearest-any proposal ("" if none)
    A_gt_cls = []
    # Q5 decomposition counters (overall + per class)
    decomp = {"near_miss": 0, "class_confusion": 0, "true_void": 0}
    decomp_by_cls = defaultdict(lambda: {"near_miss": 0, "class_confusion": 0, "true_void": 0})

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
                P_cls = np.array([p["cls_name"] for p in props])
            else:
                P_xy = np.empty((0, 2), dtype=np.float64)
                P_cls = np.array([], dtype=object)

            for g in gts:
                blab = _distance_bin(g["ego_distance_m"])
                if blab is None:
                    continue
                gcls = g["cls_name"]
                gxy = np.asarray(g["centroid_global"][:2], dtype=np.float64)

                tot_by_bin[blab] += 1
                tot_by_cls[gcls] += 1
                tot_by_cd[(gcls, blab)] += 1

                if P_xy.shape[0] == 0:
                    near_any, near_same, near_cls = INF, INF, ""
                else:
                    d = np.linalg.norm(P_xy - gxy, axis=1)
                    j = int(np.argmin(d))
                    near_any = float(d[j]); near_cls = str(P_cls[j])
                    same_mask = (P_cls == gcls)
                    near_same = float(d[same_mask].min()) if same_mask.any() else INF

                same_within4 = near_same <= MATCH_RADIUS_M
                any_within4 = near_any <= MATCH_RADIUS_M
                if same_within4:
                    continue  # C
                if any_within4:
                    continue  # B
                # --- category A ---
                A_by_bin[blab] += 1
                A_by_cls[gcls] += 1
                A_by_cd[(gcls, blab)] += 1
                A_nearest_any.append(near_any)
                A_nearest_same.append(near_same)
                A_near_cls.append(near_cls)
                A_gt_cls.append(gcls)
                # Q5 decomposition (6 m near-band)
                if near_same <= NEAR_BAND_M:
                    decomp["near_miss"] += 1
                    decomp_by_cls[gcls]["near_miss"] += 1
                elif near_any <= NEAR_BAND_M:
                    decomp["class_confusion"] += 1
                    decomp_by_cls[gcls]["class_confusion"] += 1
                else:
                    decomp["true_void"] += 1
                    decomp_by_cls[gcls]["true_void"] += 1

        if (si + 1) % 20 == 0:
            print(f"[catA-cause] scene {si+1}/{len(val_scenes)} — elapsed {time.time()-t0:.0f}s",
                  flush=True)

    A_any = np.asarray(A_nearest_any, dtype=np.float64)
    A_same = np.asarray(A_nearest_same, dtype=np.float64)
    nA = int(A_any.size)
    N = sum(tot_by_bin.values())

    # ---- Q2 rates ----
    rate_by_bin = {b: {"n": tot_by_bin[b], "A": A_by_bin[b],
                       "A_rate": (A_by_bin[b] / tot_by_bin[b]) if tot_by_bin[b] else None}
                   for b in bin_labels}
    rate_by_cls = {c: {"n": tot_by_cls[c], "A": A_by_cls[c],
                       "A_rate": (A_by_cls[c] / tot_by_cls[c]) if tot_by_cls[c] else None}
                   for c in sorted(NUSC_10_SET) if tot_by_cls[c]}
    rate_by_cd = {}
    for c in sorted(NUSC_10_SET):
        row = {}
        for b in bin_labels:
            n = tot_by_cd[(c, b)]
            if n:
                row[b] = {"n": n, "A": A_by_cd[(c, b)], "A_rate": A_by_cd[(c, b)] / n}
        if row:
            rate_by_cd[c] = row

    # ---- Q4 radius relaxation (fraction of A that leaves) ----
    relax = {}
    for R in RELAX_RADII:
        leave_any = int(np.count_nonzero(A_any <= R)) / nA if nA else None
        leave_same = int(np.count_nonzero(A_same <= R)) / nA if nA else None
        relax[f"{R:g}m"] = {
            "frac_A_leaves_any_proposal": leave_any,
            "frac_A_becomes_recoverable_same_class": leave_same,
            "n_leaves_any": int(np.count_nonzero(A_any <= R)),
            "n_same_class": int(np.count_nonzero(A_same <= R)),
        }

    # ---- Q5 decomposition ----
    decomp_pct = {k: (v / nA if nA else None) for k, v in decomp.items()}

    payload = {
        "config": {"gamma_cache": str(gamma_cache), "n_val_scenes": len(val_scenes),
                   "match_radius_m": MATCH_RADIUS_M, "near_band_m": NEAR_BAND_M,
                   "n_samples_missing_cache": n_missing,
                   "metric": "BEV/horizontal nearest-proposal distance (uncapped)"},
        "overall": {"n_gt": N, "n_A": nA, "A_rate": nA / N if N else None},
        "Q2_A_rate_by_bin": rate_by_bin,
        "Q2_A_rate_by_class": rate_by_cls,
        "Q2_A_rate_by_class_x_bin": rate_by_cd,
        "Q4_radius_relaxation": relax,
        "Q5_decomposition": {
            "near_band_m": NEAR_BAND_M,
            "counts": decomp,
            "fractions": decomp_pct,
            "definitions": {
                "near_miss": "same-class proposal exists within 6m (localization/threshold near-miss)",
                "class_confusion": "no same-class <=6m but some proposal <=6m (box present, wrong class)",
                "true_void": "no proposal at all within 6m (genuine detection void)",
            },
            "by_class": {c: dict(v) for c, v in sorted(decomp_by_cls.items())},
        },
    }

    out_file = out_dir / "catA_cause.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[catA-cause] wrote {out_file}", flush=True)

    # ---- console ----
    print(f"\n=== overall: n_gt={N}  n_A={nA}  A_rate={nA/N:.4f} ===")
    print("\nQ2  A-rate by distance bin:")
    for b in bin_labels:
        r = rate_by_bin[b]
        print(f"  {b:<8} n={r['n']:>7}  A={r['A']:>6}  A_rate={r['A_rate']:.4f}")
    print("\nQ2  A-rate by class:")
    for c, r in sorted(rate_by_cls.items(), key=lambda kv: -kv[1]["A_rate"]):
        print(f"  {c:<22} n={r['n']:>6}  A_rate={r['A_rate']:.4f}")
    print("\nQ4  radius relaxation (fraction of A leaving):")
    for k, v in relax.items():
        print(f"  -> {k}: any-proposal={v['frac_A_leaves_any_proposal']:.4f}  "
              f"same-class={v['frac_A_becomes_recoverable_same_class']:.4f}")
    print(f"\nQ5  decomposition of A (n={nA}, 6m band):")
    for k in ("near_miss", "class_confusion", "true_void"):
        print(f"  {k:<16} {decomp[k]:>6}  ({decomp_pct[k]:.4f})")


if __name__ == "__main__":
    main()
