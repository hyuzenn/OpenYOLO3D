"""Semantic temporal consistency of native CenterPoint proposals.

EVIDENCE-ONLY. No detector output is modified; no devkit eval. We reuse the
existing CenterPoint γ cache and the existing streaming track associator
(``ClassAgnosticAssociator``) to build object tracks across consecutive frames
and measure how stable each track's *predicted class* is over time.

Why the class-AGNOSTIC associator:
  The production default ``CentroidAssociator`` only matches a proposal to an
  active id of the SAME class, so every track is single-class by construction
  (class-switch rate is structurally 0 — see
  results/2026-05-21_task_3_1_lsc_diagnosis_v01). To *observe* label flicker we
  must associate by geometry alone, then read off the class sequence. This is
  the only honest way to measure semantic temporal consistency. The geometry
  (2.0 m gate, max_age 5) is the production setting (DEFAULT_ASSOC_DIST_M).

For every track we collect:
  - predicted class sequence, score sequence, length (#frames observed).
Per-track metrics:
  - class-switch rate = #(consecutive class changes) / (length - 1)
  - #unique classes
  - Shannon entropy (bits) of the within-track class distribution.
Reported overall and per dominant-class; stable vs unstable classes ranked.

Theoretical gain of a temporal consistency layer (no detector edit):
  Three aggregators relabel a track without changing geometry/recall:
    majority_vote     : count-mode of the whole track, applied to every frame.
    ema_voting        : causal EMA of one-hot class votes (per-frame argmax).
    track_aggregation : score-weighted argmax of the whole track.
  Gain is reported two ways:
    (1) GT-free stabilization — fraction of per-frame labels that CHANGE under
        the aggregator (= flicker removed; entropy -> 0 for offline modes).
    (2) GT-anchored accuracy — match each frame's proposal to the nearest GT
        (<= match radius, global BEV) and read its class; report per-frame
        class accuracy BEFORE vs AFTER each aggregator over matched frames.

"nearest" / association distances are BEV/horizontal (the nuScenes metric).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10_SET,
    _gt_records,
    _load_cached_proposals,
    _ego_to_global_centroid,
    _list_val_scenes,
    _scene_sample_tokens,
)
from method_scannet.streaming.nuscenes_native_evaluator import (
    ClassAgnosticAssociator,
    DEFAULT_ASSOC_DIST_M,
)

ASSOC_MAX_AGE = 5
GT_MATCH_RADIUS_M = 2.0       # proposal->GT class read-off radius (global BEV)
EMA_ALPHAS = (0.3, 0.5, 0.7)  # causal EMA voting sweep
EMA_DEFAULT = 0.5


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _entropy_bits(seq) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return float(h)


def _count_mode(seq, scores=None):
    """Most frequent class; tie-break by total score then name."""
    if not seq:
        return None
    cnt = Counter(seq)
    score_sum: dict = defaultdict(float)
    if scores is not None:
        for c, s in zip(seq, scores):
            score_sum[c] += float(s)
    best = sorted(cnt.keys(), key=lambda c: (-cnt[c], -score_sum.get(c, 0.0), c))
    return best[0]


def _score_weighted_mode(seq, scores):
    if not seq:
        return None
    ssum: dict = defaultdict(float)
    cnt = Counter(seq)
    for c, s in zip(seq, scores):
        ssum[c] += float(s)
    best = sorted(ssum.keys(), key=lambda c: (-ssum[c], -cnt[c], c))
    return best[0]


def _ema_sequence(seq, classes_index, alpha):
    """Causal per-frame argmax of EMA one-hot votes. Returns list parallel to seq."""
    K = len(classes_index)
    ema = np.zeros(K, dtype=np.float64)
    out = []
    for c in seq:
        v = np.zeros(K, dtype=np.float64)
        v[classes_index[c]] = 1.0
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(max(classes_index, key=lambda cl: ema[classes_index[cl]]))
    return out


# --------------------------------------------------------------------------- #
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
        existing = list(project_root.glob(f"results/{date}_outdoor_temporal_consistency_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_temporal_consistency_v{len(existing)+1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[tc] γ cache    : {gamma_cache}", flush=True)
    print(f"[tc] output dir : {out_dir}", flush=True)
    print(f"[tc] assoc      : class-AGNOSTIC, gate={args.assoc_dist_m}m, max_age={ASSOC_MAX_AGE}", flush=True)

    print("[tc] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[tc] val scenes : {len(val_scenes)}", flush=True)

    # tracks[(scene_idx, gid)] = {"cls": [...], "score": [...], "gt_cls": [...]}
    tracks: dict = defaultdict(lambda: {"cls": [], "score": [], "gt_cls": []})

    t0 = time.time()
    n_missing = 0
    n_props_total = 0
    n_props_non10 = 0
    for si, sc_tok in enumerate(val_scenes):
        assoc = ClassAgnosticAssociator(threshold_m=args.assoc_dist_m, max_age=ASSOC_MAX_AGE)
        assoc.reset()
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            sample = nusc.get("sample", sa_tok)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                continue

            # Keep proposals evaluable under nuScenes-10 (cache gates on emit,
            # but be explicit). Need cls_name, score, centroid_ego.
            props = []
            for p in raw:
                n_props_total += 1
                cls = p.get("cls_name")
                if cls not in NUSC_10_SET:
                    n_props_non10 += 1
                    continue
                props.append({
                    "cls_name": cls,
                    "score": float(p.get("score", 0.0)),
                    "centroid_ego": np.asarray(p["centroid_ego"], dtype=np.float64),
                })

            gids = assoc.step(props)

            # per-frame nearest-GT class read-off (global BEV)
            gts = _gt_records(nusc, sa_tok, ego_pose)
            if gts:
                G_xy = np.asarray([g["centroid_global"][:2] for g in gts], dtype=np.float64)
                G_cls = [g["cls_name"] for g in gts]
            else:
                G_xy = np.empty((0, 2), dtype=np.float64)
                G_cls = []

            for p, gid in zip(props, gids):
                key = (si, gid)
                tr = tracks[key]
                tr["cls"].append(p["cls_name"])
                tr["score"].append(p["score"])
                if G_xy.shape[0]:
                    pg = _ego_to_global_centroid(p["centroid_ego"], ego_pose)[:2]
                    d = np.linalg.norm(G_xy - pg, axis=1)
                    j = int(d.argmin())
                    tr["gt_cls"].append(G_cls[j] if d[j] <= GT_MATCH_RADIUS_M else None)
                else:
                    tr["gt_cls"].append(None)

        if (si + 1) % 20 == 0:
            print(f"[tc] scene {si+1}/{len(val_scenes)} — tracks so far "
                  f"{len(tracks)} — elapsed {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    # per-track metrics
    # ----------------------------------------------------------------------- #
    classes_sorted = sorted(NUSC_10_SET)
    cls_index = {c: i for i, c in enumerate(classes_sorted)}

    # accumulators (overall and per dominant class)
    def _fresh():
        return {
            "n_tracks": 0, "n_tracks_multi": 0,
            "len_sum": 0, "obs_sum": 0,
            "transitions": 0, "switches": 0,
            "uniq_sum": 0, "entropy_sum": 0.0,
            # GT-anchored frame counters
            "gt_frames": 0, "acc_raw": 0,
            "acc_majority": 0, "acc_track_agg": 0,
            "acc_ema": {a: 0 for a in EMA_ALPHAS},
            # GT-free stabilization (frames changed by aggregator)
            "changed_majority": 0, "changed_track_agg": 0,
            "changed_ema": {a: 0 for a in EMA_ALPHAS},
        }

    overall = _fresh()
    per_cls = defaultdict(_fresh)
    # length>=2 only (single-obs tracks have no transitions); but entropy/uniq
    # computed over all tracks. We keep a separate multi-obs accumulator for
    # switch-rate, and count singletons.
    n_singleton = 0

    for key, tr in tracks.items():
        seq = tr["cls"]
        scores = tr["score"]
        gt = tr["gt_cls"]
        L = len(seq)
        dom = _count_mode(seq, scores)

        for acc in (overall, per_cls[dom]):
            acc["n_tracks"] += 1
            acc["len_sum"] += L
            acc["obs_sum"] += L
            uniq = len(set(seq))
            acc["uniq_sum"] += uniq
            acc["entropy_sum"] += _entropy_bits(seq)
            if uniq > 1:
                acc["n_tracks_multi"] += 1

        if L < 2:
            n_singleton += 1

        # switches over consecutive frames
        sw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        for acc in (overall, per_cls[dom]):
            acc["transitions"] += max(L - 1, 0)
            acc["switches"] += sw

        # aggregator predictions
        maj = _count_mode(seq, scores)               # offline count-mode
        tagg = _score_weighted_mode(seq, scores)      # offline score-weighted
        ema_seqs = {a: _ema_sequence(seq, cls_index, a) for a in EMA_ALPHAS}
        ema_def = ema_seqs[EMA_DEFAULT]

        # GT-free stabilization: frames whose label changes under aggregator
        for acc in (overall, per_cls[dom]):
            acc["changed_majority"] += sum(1 for c in seq if c != maj)
            acc["changed_track_agg"] += sum(1 for c in seq if c != tagg)
            for a in EMA_ALPHAS:
                acc["changed_ema"][a] += sum(1 for c, e in zip(seq, ema_seqs[a]) if c != e)

        # GT-anchored accuracy over matched frames
        for t in range(L):
            g = gt[t]
            if g is None:
                continue
            for acc in (overall, per_cls[dom]):
                acc["gt_frames"] += 1
                acc["acc_raw"] += int(seq[t] == g)
                acc["acc_majority"] += int(maj == g)
                acc["acc_track_agg"] += int(tagg == g)
                for a in EMA_ALPHAS:
                    acc["acc_ema"][a] += int(ema_seqs[a][t] == g)

    # ----------------------------------------------------------------------- #
    # finalize
    # ----------------------------------------------------------------------- #
    def _summ(acc):
        nt = acc["n_tracks"]
        tr_ = acc["transitions"]
        gf = acc["gt_frames"]
        out = {
            "n_tracks": nt,
            "n_tracks_multiclass": acc["n_tracks_multi"],
            "frac_tracks_multiclass": (acc["n_tracks_multi"] / nt) if nt else None,
            "mean_track_length": (acc["len_sum"] / nt) if nt else None,
            "mean_unique_classes": (acc["uniq_sum"] / nt) if nt else None,
            "mean_entropy_bits": (acc["entropy_sum"] / nt) if nt else None,
            "n_transitions": tr_,
            "n_switches": acc["switches"],
            "class_switch_rate": (acc["switches"] / tr_) if tr_ else None,
            "stabilization_gtfree": {
                "frac_frames_changed_majority": (acc["changed_majority"] / acc["obs_sum"]) if acc["obs_sum"] else None,
                "frac_frames_changed_track_agg": (acc["changed_track_agg"] / acc["obs_sum"]) if acc["obs_sum"] else None,
                **{f"frac_frames_changed_ema_a{a}": (acc["changed_ema"][a] / acc["obs_sum"]) if acc["obs_sum"] else None
                   for a in EMA_ALPHAS},
            },
            "gt_anchored": {
                "n_matched_frames": gf,
                "acc_raw": (acc["acc_raw"] / gf) if gf else None,
                "acc_majority_vote": (acc["acc_majority"] / gf) if gf else None,
                "acc_track_aggregation": (acc["acc_track_agg"] / gf) if gf else None,
                **{f"acc_ema_a{a}": (acc["acc_ema"][a] / gf) if gf else None
                   for a in EMA_ALPHAS},
            },
        }
        if gf:
            out["gt_anchored"]["delta_majority_vote"] = out["gt_anchored"]["acc_majority_vote"] - out["gt_anchored"]["acc_raw"]
            out["gt_anchored"]["delta_track_aggregation"] = out["gt_anchored"]["acc_track_aggregation"] - out["gt_anchored"]["acc_raw"]
            for a in EMA_ALPHAS:
                out["gt_anchored"][f"delta_ema_a{a}"] = out["gt_anchored"][f"acc_ema_a{a}"] - out["gt_anchored"]["acc_raw"]
        return out

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "associator": "ClassAgnosticAssociator (geometry-only)",
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "gt_match_radius_m": GT_MATCH_RADIUS_M,
            "ema_alphas": list(EMA_ALPHAS),
            "ema_default_alpha": EMA_DEFAULT,
            "n_samples_missing_cache": n_missing,
            "n_proposals_total": n_props_total,
            "n_proposals_dropped_non_nusc10": n_props_non10,
            "n_tracks_total": len(tracks),
            "n_singleton_tracks": n_singleton,
            "metric": "BEV/horizontal; class-switch over consecutive frames within a geometry-only track",
        },
        "overall": _summ(overall),
        "per_dominant_class": {c: _summ(per_cls[c]) for c in classes_sorted if per_cls[c]["n_tracks"]},
    }

    # stable / unstable ranking by class-switch rate (tie -> multiclass frac)
    ranked = [(c, payload["per_dominant_class"][c]["class_switch_rate"],
               payload["per_dominant_class"][c]["frac_tracks_multiclass"],
               payload["per_dominant_class"][c]["n_tracks"])
              for c in payload["per_dominant_class"]
              if payload["per_dominant_class"][c]["class_switch_rate"] is not None]
    ranked.sort(key=lambda x: (x[1], x[2]))
    payload["stability_ranking"] = [
        {"class": c, "class_switch_rate": csr, "frac_multiclass": fm, "n_tracks": n}
        for c, csr, fm, n in ranked]
    if ranked:
        payload["stable_classes"] = [r["class"] for r in payload["stability_ranking"][:3]]
        payload["unstable_classes"] = [r["class"] for r in payload["stability_ranking"][-3:]][::-1]

    out_file = out_dir / "temporal_consistency.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[tc] wrote {out_file}", flush=True)

    # ----------------------------- console ---------------------------------- #
    o = payload["overall"]
    print("\n=== OVERALL (class-agnostic tracks) ===")
    print(f"  tracks={o['n_tracks']}  singleton={n_singleton}  "
          f"mean_len={o['mean_track_length']:.2f}")
    print(f"  class_switch_rate={o['class_switch_rate']}  "
          f"mean_unique_cls={o['mean_unique_classes']:.3f}  "
          f"mean_entropy_bits={o['mean_entropy_bits']:.3f}  "
          f"frac_multiclass={o['frac_tracks_multiclass']:.3f}")
    ga = o["gt_anchored"]
    print(f"  GT-anchored (n={ga['n_matched_frames']}): raw={ga['acc_raw']:.4f}  "
          f"majority={ga['acc_majority_vote']:.4f}  track_agg={ga['acc_track_aggregation']:.4f}  "
          f"ema@{EMA_DEFAULT}={ga[f'acc_ema_a{EMA_DEFAULT}']:.4f}")
    if ga.get("delta_majority_vote") is not None:
        print(f"  Δacc: majority={ga['delta_majority_vote']:+.4f}  "
              f"track_agg={ga['delta_track_aggregation']:+.4f}  "
              f"ema@{EMA_DEFAULT}={ga[f'delta_ema_a{EMA_DEFAULT}']:+.4f}")
    sg = o["stabilization_gtfree"]
    print(f"  GT-free stabilization: majority changes {sg['frac_frames_changed_majority']:.4f} of frames")

    print("\n=== per dominant class (switch rate / multiclass / mean entropy) ===")
    for c in classes_sorted:
        b = payload["per_dominant_class"].get(c)
        if not b:
            continue
        csr = b["class_switch_rate"]
        print(f"  {c:<22} n={b['n_tracks']:>6}  csr={csr if csr is None else round(csr,4)}  "
              f"multi={b['frac_tracks_multiclass']:.3f}  H={b['mean_entropy_bits']:.3f}")
    if ranked:
        print(f"\n  stable   : {payload['stable_classes']}")
        print(f"  unstable : {payload['unstable_classes']}")


if __name__ == "__main__":
    main()
