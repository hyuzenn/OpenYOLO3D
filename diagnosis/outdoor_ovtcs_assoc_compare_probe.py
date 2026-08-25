"""OV-TCS head-to-head: baseline (ego-frame) vs global-frame associator.

EVIDENCE-ONLY. No detector output modified; no devkit eval. Builds object tracks
under TWO associators in ONE data pass over the γ cache and scores the SAME
OV-TCS formulations (A/B/C, reused verbatim from `outdoor_ov_tcs_probe.py`) on
each, so the only variable is the matching frame.

  * baseline : production `ClassAgnosticAssociator` — ego-frame centroids, greedy,
               static, gate 2.0 m, max_age 5 (the current pipeline).
  * global   : self-contained `Associator` from the associator ablation, identical
               knobs (greedy / static / gate 2.0 / max_age 5) but matching in the
               GLOBAL frame (ego-motion compensated). Single-variable change.

OV-TCS is a property of the predicted-label sequence of a track. The ablation
showed the global frame cuts fragmentation 10.7 → 4.5 (longer tracks); the open
question this probe answers is whether those longer tracks are also more
temporally CONSISTENT in label, i.e. whether OV-TCS actually rises and whether
stitching introduces cross-class label mixing.

Reported per associator
  - track population: n_tracks, singleton fraction, mean/p50/p95 length
  - OV-TCS_A/B/C over ALL tracks (singletons score 0) — the honest pipeline view
  - OV-TCS_A/B/C detection-weighted: Σ(Lᵢ·scoreᵢ)/Σ Lᵢ — score a random detection
    experiences (length-fair; does not reward fragmentation into singletons)
  - per-frame consistency on multi-frame tracks (L≥2): mean entropy, dominant
    ratio, class-switch rate, unique-label count — does stitching hurt purity?
And the global−baseline delta for every line.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10_SET,
    _load_cached_proposals,
    _ego_to_global_centroid,
    _list_val_scenes,
    _scene_sample_tokens,
)
from diagnosis.outdoor_ov_tcs_probe import (
    _track_metrics,
    _ov_tcs,
    _l_norm,
    _dist,
)
from diagnosis.outdoor_associator_ablation_probe import Associator, GATE_M
from method_scannet.streaming.nuscenes_native_evaluator import (
    ClassAgnosticAssociator,
    DEFAULT_ASSOC_DIST_M,
)

ASSOC_MAX_AGE = 5
DEFAULT_DT = 0.5


def _agg(seqs, scores_by_track, log2K):
    """Compute the full OV-TCS report for one associator's track set.

    seqs:            list of label sequences (one per track)
    scores_by_track: parallel list of (A,B,C) is computed here; we also need the
                     mean proposal score per track for detection-weighting — but
                     OV-TCS does not depend on score, so detection-weighting uses
                     track length L as the weight (each frame = one observation).
    """
    L_l, U_l, H_l, DR_l, CSR_l = [], [], [], [], []
    A_l, B_l, C_l = [], [], []
    for seq in seqs:
        L, U, H, DR, CSR = _track_metrics(seq)
        A, B, C = _ov_tcs(L, H, DR, CSR, log2K)
        L_l.append(L); U_l.append(U); H_l.append(H); DR_l.append(DR)
        CSR_l.append(CSR if CSR is not None else np.nan)
        A_l.append(A); B_l.append(B); C_l.append(C)
    L = np.asarray(L_l, float); U = np.asarray(U_l, float)
    H = np.asarray(H_l, float); DR = np.asarray(DR_l, float)
    CSR = np.asarray(CSR_l, float)
    A = np.asarray(A_l, float); B = np.asarray(B_l, float); C = np.asarray(C_l, float)
    multi = L >= 2
    n = L.size

    def wmean(x):
        # length-weighted (per-observation / per-detection) mean
        return float((x * L).sum() / L.sum()) if L.sum() else 0.0

    return {
        "n_tracks": int(n),
        "n_singleton": int((L < 2).sum()),
        "singleton_frac": float((L < 2).mean()) if n else 0.0,
        "track_length": _dist(L),
        "total_detections": int(L.sum()),
        # OV-TCS over ALL tracks (singletons -> 0)
        "ovtcs_all": {
            "A": _dist(A), "B": _dist(B), "C": _dist(C),
            "mean_A": float(A.mean()), "mean_B": float(B.mean()), "mean_C": float(C.mean()),
        },
        # length-weighted (a random detection's experienced consistency)
        "ovtcs_detection_weighted": {
            "A": wmean(A), "B": wmean(B), "C": wmean(C),
        },
        # per-frame label consistency on multi-frame tracks only
        "consistency_multiframe_L>=2": {
            "n_tracks": int(multi.sum()),
            "mean_unique_labels": float(U[multi].mean()) if multi.any() else None,
            "mean_entropy_bits": float(H[multi].mean()) if multi.any() else None,
            "mean_dominant_ratio": float(DR[multi].mean()) if multi.any() else None,
            "mean_class_switch_rate": float(np.nanmean(CSR[multi])) if multi.any() else None,
            "frac_single_class": float((U[multi] == 1).mean()) if multi.any() else None,
        },
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" /
        "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--assoc-dist-m", type=float, default=DEFAULT_ASSOC_DIST_M)
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
        existing = list(project_root.glob(f"results/{date}_outdoor_ovtcs_assoc_compare_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_ovtcs_assoc_compare_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[cmp] γ cache    : {gamma_cache}", flush=True)
    print(f"[cmp] output dir : {out_dir}", flush=True)

    K = len(NUSC_10_SET)
    log2K = math.log2(K)

    print("[cmp] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[cmp] val scenes : {len(val_scenes)}", flush=True)

    tracks_base: dict = defaultdict(list)   # (si, gid) -> [cls,...]  ego-frame production
    tracks_glob: dict = defaultdict(list)   # (si, gid) -> [cls,...]  global-frame
    n_missing = n_props_total = 0
    t0 = time.time()

    for si, sc_tok in enumerate(val_scenes):
        base = ClassAgnosticAssociator(threshold_m=args.assoc_dist_m, max_age=ASSOC_MAX_AGE)
        base.reset()
        glob = Associator(GATE_M, max_age=ASSOC_MAX_AGE, hungarian=False, motion=False)
        glob.reset()
        prev_t = None
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            sample = nusc.get("sample", sa_tok)
            t_sec = sample["timestamp"] * 1e-6
            dt = (t_sec - prev_t) if prev_t is not None else DEFAULT_DT
            prev_t = t_sec

            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                base.step([])
                glob.step(np.empty((0, 2)), [], dt)
                continue

            props = []          # for production associator (dicts, ego centroids)
            classes = []
            scores = []
            P_glob = []
            for p in raw:
                cls = p.get("cls_name")
                if cls not in NUSC_10_SET:
                    continue
                n_props_total += 1
                c_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
                sc = float(p.get("score", 0.0))
                props.append({"cls_name": cls, "score": sc, "centroid_ego": c_ego})
                classes.append(cls)
                scores.append(sc)
                P_glob.append(_ego_to_global_centroid(c_ego, ego_pose)[:2])

            if not props:
                base.step([])
                glob.step(np.empty((0, 2)), [], dt)
                continue

            gids_b = base.step(props)
            gids_g = glob.step(np.asarray(P_glob, dtype=np.float64),
                               np.asarray(scores, dtype=np.float64), dt)
            for cls, gb, gg in zip(classes, gids_b, gids_g):
                tracks_base[(si, gb)].append(cls)
                tracks_glob[(si, gg)].append(cls)

        if (si + 1) % 25 == 0:
            print(f"[cmp] scene {si+1}/{len(val_scenes)} — "
                  f"tracks base {len(tracks_base)} glob {len(tracks_glob)} "
                  f"— {time.time()-t0:.0f}s", flush=True)

    rep_base = _agg(list(tracks_base.values()), None, log2K)
    rep_glob = _agg(list(tracks_glob.values()), None, log2K)

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "n_proposals_total": n_props_total,
            "n_samples_missing_cache": n_missing,
            "K_classes": K,
            "baseline": "ClassAgnosticAssociator ego-frame (production)",
            "global": "Associator global-frame greedy/static (ablation)",
        },
        "baseline": rep_base,
        "global": rep_glob,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[cmp] wrote {out_dir/'metrics.json'}", flush=True)
    _write_notes(out_dir, payload)
    _console(payload)


def _write_notes(out_dir, p):
    c = p["config"]; b = p["baseline"]; g = p["global"]
    L = []
    L.append("# OV-TCS — baseline (ego) vs global associator, head-to-head\n")
    L.append("Probe: `diagnosis/outdoor_ovtcs_assoc_compare_probe.py` (evidence-only). "
             f"γ gravity cache, full val ({c['n_val_scenes']} scenes), "
             f"{c['n_proposals_total']:,} nuScenes-10 proposals. Both associators run in "
             f"ONE pass; gate {c['assoc_threshold_m']} m, max_age {c['assoc_max_age']}, "
             "greedy/static. Only the matching FRAME differs (ego vs global / ego-motion "
             "compensated). OV-TCS formulations identical to `outdoor_ov_tcs_probe.py`.\n")

    def delta(x, y):
        return f"{y - x:+.3f}"

    L.append("## Track population")
    L.append("| | baseline (ego) | global | Δ |")
    L.append("|---|---|---|---|")
    L.append(f"| n tracks | {b['n_tracks']:,} | {g['n_tracks']:,} | {g['n_tracks']-b['n_tracks']:+,} |")
    L.append(f"| singleton frac | {b['singleton_frac']:.3f} | {g['singleton_frac']:.3f} | "
             f"{delta(b['singleton_frac'], g['singleton_frac'])} |")
    L.append(f"| mean track length | {b['track_length']['mean']:.3f} | {g['track_length']['mean']:.3f} | "
             f"{delta(b['track_length']['mean'], g['track_length']['mean'])} |")
    L.append(f"| p50 length | {b['track_length']['p50']:.0f} | {g['track_length']['p50']:.0f} | — |")
    L.append(f"| p95 length | {b['track_length']['p95']:.0f} | {g['track_length']['p95']:.0f} | — |")
    L.append("")

    L.append("## OV-TCS over ALL tracks (singletons score 0 — pipeline view)")
    L.append("| formulation | baseline | global | Δ | rel |")
    L.append("|---|---|---|---|---|")
    for k in ("A", "B", "C"):
        bv = b["ovtcs_all"][f"mean_{k}"]; gv = g["ovtcs_all"][f"mean_{k}"]
        rel = f"{100*(gv-bv)/bv:+.1f}%" if bv else "—"
        L.append(f"| OV-TCS_{k} | {bv:.3f} | {gv:.3f} | {delta(bv, gv)} | {rel} |")
    L.append("")

    L.append("## OV-TCS detection-weighted (Σ L·score / Σ L — length-fair)")
    L.append("| formulation | baseline | global | Δ | rel |")
    L.append("|---|---|---|---|---|")
    for k in ("A", "B", "C"):
        bv = b["ovtcs_detection_weighted"][k]; gv = g["ovtcs_detection_weighted"][k]
        rel = f"{100*(gv-bv)/bv:+.1f}%" if bv else "—"
        L.append(f"| OV-TCS_{k} | {bv:.3f} | {gv:.3f} | {delta(bv, gv)} | {rel} |")
    L.append("")

    L.append("## Per-frame label consistency on multi-frame tracks (L≥2)")
    L.append("_isolates label purity from length — does stitching mix classes?_")
    bc = b["consistency_multiframe_L>=2"]; gc = g["consistency_multiframe_L>=2"]
    L.append("| | baseline | global | Δ |")
    L.append("|---|---|---|---|")
    L.append(f"| n multi-frame tracks | {bc['n_tracks']:,} | {gc['n_tracks']:,} | "
             f"{gc['n_tracks']-bc['n_tracks']:+,} |")
    L.append(f"| mean unique labels | {bc['mean_unique_labels']:.3f} | {gc['mean_unique_labels']:.3f} | "
             f"{delta(bc['mean_unique_labels'], gc['mean_unique_labels'])} |")
    L.append(f"| mean entropy (bits) | {bc['mean_entropy_bits']:.3f} | {gc['mean_entropy_bits']:.3f} | "
             f"{delta(bc['mean_entropy_bits'], gc['mean_entropy_bits'])} |")
    L.append(f"| mean dominant ratio | {bc['mean_dominant_ratio']:.3f} | {gc['mean_dominant_ratio']:.3f} | "
             f"{delta(bc['mean_dominant_ratio'], gc['mean_dominant_ratio'])} |")
    L.append(f"| mean class-switch rate | {bc['mean_class_switch_rate']:.3f} | "
             f"{gc['mean_class_switch_rate']:.3f} | "
             f"{delta(bc['mean_class_switch_rate'], gc['mean_class_switch_rate'])} |")
    L.append(f"| frac single-class tracks | {bc['frac_single_class']:.3f} | "
             f"{gc['frac_single_class']:.3f} | "
             f"{delta(bc['frac_single_class'], gc['frac_single_class'])} |")
    L.append("")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[cmp] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    b = p["baseline"]; g = p["global"]
    print("\n=== OV-TCS: baseline (ego) vs global associator ===")
    print(f"  tracks       base {b['n_tracks']:,}  glob {g['n_tracks']:,}")
    print(f"  singleton    base {b['singleton_frac']:.3f}  glob {g['singleton_frac']:.3f}")
    print(f"  mean length  base {b['track_length']['mean']:.2f}  glob {g['track_length']['mean']:.2f}")
    print("  --- OV-TCS over all tracks ---")
    for k in ("A", "B", "C"):
        bv = b["ovtcs_all"][f"mean_{k}"]; gv = g["ovtcs_all"][f"mean_{k}"]
        print(f"   {k}  base {bv:.3f}  glob {gv:.3f}  Δ {gv-bv:+.3f}")
    print("  --- OV-TCS detection-weighted ---")
    for k in ("A", "B", "C"):
        bv = b["ovtcs_detection_weighted"][k]; gv = g["ovtcs_detection_weighted"][k]
        print(f"   {k}  base {bv:.3f}  glob {gv:.3f}  Δ {gv-bv:+.3f}")
    bc = b["consistency_multiframe_L>=2"]; gc = g["consistency_multiframe_L>=2"]
    print("  --- per-frame consistency (L>=2) ---")
    print(f"   dominant_ratio  base {bc['mean_dominant_ratio']:.3f}  glob {gc['mean_dominant_ratio']:.3f}")
    print(f"   switch_rate     base {bc['mean_class_switch_rate']:.3f}  glob {gc['mean_class_switch_rate']:.3f}")
    print(f"   single_class    base {bc['frac_single_class']:.3f}  glob {gc['frac_single_class']:.3f}")


if __name__ == "__main__":
    main()
