"""Where does the 0.618 class-switch rate come from?

EVIDENCE-ONLY. No detector output modified, no pipeline code touched, no devkit
eval. We reuse the existing CenterPoint gamma cache and the existing streaming
``ClassAgnosticAssociator`` (geometry-only, production 2.0 m gate, max_age 5) to
rebuild the same tracks as
``diagnosis/outdoor_temporal_consistency_probe.py`` and then ATTRIBUTE the
observed label instability to one of four candidate sources:

  (1) detector confidence uncertainty  -> switches happen at low score
  (2) proposal identity fragmentation  -> a geometry track spans >1 real object
  (3) true feature ambiguity           -> label flips persist at high score
  (4) short track duration             -> csr is a small-sample / fragmentation
                                          artifact, falls with track length

Two complementary views:

A. PER-TRACK metrics + correlations (length>=2 tracks):
     track length, switch count, switch rate, mean score, score variance,
     class entropy, mean object (ego) distance.
   Pearson + Spearman of per-track switch_rate against
     {track length, detector confidence (mean score), score variance,
      object distance}.
   Single-feature OLS R^2 and standardized multivariate coefficients, to say
   which factor explains the most variance in switch rate.

B. PER-TRANSITION attribution of the pooled switch budget. For every
   consecutive (t, t+1) where the predicted class changes (a "switch"), look at
   the nearest-GT class (<=2 m global BEV) at both frames:
     - both matched, GT class SAME  -> detector instability on one real object
         further split by min(score_t, score_t1):
           low score  -> confidence uncertainty (1)
           high score -> true feature ambiguity (3)
     - both matched, GT class DIFFERS -> identity fragmentation (2)
     - either unmatched               -> unattributable
   This apportions the 0.618 directly into sources (1)/(2)/(3).

Short-duration (4) is read from the csr-vs-length relationship in view A and the
length distribution of switching tracks.

All distances are BEV/horizontal (the nuScenes metric).
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
GT_MATCH_RADIUS_M = 2.0
# score threshold separating "confidence uncertainty" from "true ambiguity"
# at a detector-instability switch. CenterPoint scores are emitted with a low
# floor; 0.3 is the conventional nuScenes operating confidence.
HICONF_TH = 0.30


# --------------------------------------------------------------------------- #
def _entropy_bits(seq) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    h = 0.0
    for c in Counter(seq).values():
        p = c / n
        h -= p * math.log2(p)
    return float(h)


def _count_mode(seq, scores=None):
    if not seq:
        return None
    cnt = Counter(seq)
    ssum: dict = defaultdict(float)
    if scores is not None:
        for c, s in zip(seq, scores):
            ssum[c] += float(s)
    return sorted(cnt.keys(), key=lambda c: (-cnt[c], -ssum.get(c, 0.0), c))[0]


def _pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return None
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return _pearson(rx, ry)


def _binned(x, y, edges):
    """Mean y and count per x-bin defined by edges (list of right-edges)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    out = []
    lo = -np.inf
    for hi in edges:
        m = (x > lo) & (x <= hi)
        out.append((f"({lo:g},{hi:g}]", int(m.sum()),
                    float(y[m].mean()) if m.any() else None))
        lo = hi
    m = x > lo
    out.append((f"({lo:g},inf)", int(m.sum()),
                float(y[m].mean()) if m.any() else None))
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
        existing = list(project_root.glob(f"results/{date}_outdoor_instability_source_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_instability_source_v{len(existing)+1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[is] gamma cache : {gamma_cache}", flush=True)
    print(f"[is] output dir  : {out_dir}", flush=True)
    print(f"[is] assoc       : class-AGNOSTIC, gate={args.assoc_dist_m}m, max_age={ASSOC_MAX_AGE}", flush=True)

    print("[is] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[is] val scenes  : {len(val_scenes)}", flush=True)

    # tracks[(scene_idx, gid)] = parallel per-frame lists
    tracks: dict = defaultdict(lambda: {"cls": [], "score": [], "dist": [], "gt_cls": []})

    t0 = time.time()
    n_missing = 0
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

            props = []
            for p in raw:
                cls = p.get("cls_name")
                if cls not in NUSC_10_SET:
                    continue
                ce = np.asarray(p["centroid_ego"], dtype=np.float64)
                props.append({
                    "cls_name": cls,
                    "score": float(p.get("score", 0.0)),
                    "centroid_ego": ce,
                    "dist": float(np.linalg.norm(ce[:2])),  # BEV ego distance
                })

            gids = assoc.step(props)

            gts = _gt_records(nusc, sa_tok, ego_pose)
            if gts:
                G_xy = np.asarray([g["centroid_global"][:2] for g in gts], dtype=np.float64)
                G_cls = [g["cls_name"] for g in gts]
            else:
                G_xy = np.empty((0, 2), dtype=np.float64)
                G_cls = []

            for p, gid in zip(props, gids):
                tr = tracks[(si, gid)]
                tr["cls"].append(p["cls_name"])
                tr["score"].append(p["score"])
                tr["dist"].append(p["dist"])
                if G_xy.shape[0]:
                    pg = _ego_to_global_centroid(p["centroid_ego"], ego_pose)[:2]
                    d = np.linalg.norm(G_xy - pg, axis=1)
                    j = int(d.argmin())
                    tr["gt_cls"].append(G_cls[j] if d[j] <= GT_MATCH_RADIUS_M else None)
                else:
                    tr["gt_cls"].append(None)

        if (si + 1) % 20 == 0:
            print(f"[is] scene {si+1}/{len(val_scenes)} — tracks {len(tracks)} "
                  f"— elapsed {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    # A. per-track records (length>=2 only have a defined switch rate)
    # ----------------------------------------------------------------------- #
    rec_len, rec_switch, rec_csr = [], [], []
    rec_mscore, rec_vscore, rec_entropy, rec_mdist = [], [], [], []
    rec_dom = []
    n_singleton = 0
    pooled_switches = 0
    pooled_transitions = 0

    # B. per-transition attribution accumulators
    att = {
        "detector_instability": 0,   # GT same -> real object, label flipped
        "fragmentation": 0,          # GT differs -> track spans >1 object
        "unattributable": 0,         # one/both frames unmatched to GT
    }
    # within detector_instability, split by confidence at the switch
    det_lowconf = 0   # min(score) < HICONF_TH  -> confidence uncertainty
    det_hiconf = 0    # min(score) >= HICONF_TH -> true ambiguity
    # switch rate by score / distance bin (transition-level)
    sw_score_vals, sw_score_is_switch = [], []
    sw_dist_vals, sw_dist_is_switch = [], []

    for tr in tracks.values():
        seq, sc, dist, gt = tr["cls"], tr["score"], tr["dist"], tr["gt_cls"]
        L = len(seq)
        if L < 2:
            n_singleton += 1
            continue
        sw = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
        pooled_switches += sw
        pooled_transitions += (L - 1)

        rec_len.append(L)
        rec_switch.append(sw)
        rec_csr.append(sw / (L - 1))
        rec_mscore.append(float(np.mean(sc)))
        rec_vscore.append(float(np.var(sc)))
        rec_entropy.append(_entropy_bits(seq))
        rec_mdist.append(float(np.mean(dist)))
        rec_dom.append(_count_mode(seq, sc))

        # transition-level
        for t in range(L - 1):
            is_sw = int(seq[t] != seq[t + 1])
            mn_score = min(sc[t], sc[t + 1])
            mean_dist = 0.5 * (dist[t] + dist[t + 1])
            sw_score_vals.append(mn_score); sw_score_is_switch.append(is_sw)
            sw_dist_vals.append(mean_dist); sw_dist_is_switch.append(is_sw)
            if not is_sw:
                continue
            g0, g1 = gt[t], gt[t + 1]
            if g0 is None or g1 is None:
                att["unattributable"] += 1
            elif g0 == g1:
                att["detector_instability"] += 1
                if mn_score < HICONF_TH:
                    det_lowconf += 1
                else:
                    det_hiconf += 1
            else:
                att["fragmentation"] += 1

    csr = np.asarray(rec_csr, float)
    feats = {
        "track_length": np.asarray(rec_len, float),
        "detector_confidence_mean_score": np.asarray(rec_mscore, float),
        "score_variance": np.asarray(rec_vscore, float),
        "object_distance_m": np.asarray(rec_mdist, float),
    }

    correlations = {}
    single_r2 = {}
    for name, x in feats.items():
        pr = _pearson(x, csr)
        sr = _spearman(x, csr)
        correlations[name] = {"pearson_r": pr, "spearman_rho": sr,
                              "pearson_r2": (pr * pr) if pr is not None else None}
        single_r2[name] = (pr * pr) if pr is not None else 0.0

    # standardized multivariate OLS coefficients (relative weight)
    multi = {}
    try:
        X = np.column_stack([(x - x.mean()) / (x.std() + 1e-12) for x in feats.values()])
        yv = (csr - csr.mean()) / (csr.std() + 1e-12)
        Xd = np.column_stack([np.ones(len(yv)), X])
        beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        yhat = Xd @ beta
        ss_res = float(((yv - yhat) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        multi = {
            "standardized_coeffs": {n: float(b) for n, b in zip(feats.keys(), beta[1:])},
            "model_r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None,
        }
    except Exception as e:  # pragma: no cover
        multi = {"error": str(e)}

    # transition-level switch-rate by bin
    score_edges = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
    dist_edges = [15.0, 30.0, 50.0, 80.0]
    sw_by_score = _binned(sw_score_vals, sw_score_is_switch, score_edges)
    sw_by_dist = _binned(sw_dist_vals, sw_dist_is_switch, dist_edges)

    # csr by length bin (short-duration test)
    csr_by_len = _binned(rec_len, rec_csr, [2, 3, 4, 5, 7, 10])

    tot_sw = sum(att.values())
    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "associator": "ClassAgnosticAssociator (geometry-only)",
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "gt_match_radius_m": GT_MATCH_RADIUS_M,
            "hiconf_threshold": HICONF_TH,
            "n_tracks_total": len(tracks),
            "n_singleton_tracks": n_singleton,
            "n_multiobs_tracks": len(rec_csr),
            "n_samples_missing_cache": n_missing,
        },
        "pooled": {
            "switches": pooled_switches,
            "transitions": pooled_transitions,
            "pooled_class_switch_rate": (pooled_switches / pooled_transitions)
                                        if pooled_transitions else None,
            "mean_per_track_csr": float(csr.mean()) if csr.size else None,
        },
        "A_correlations_csr_vs_factor": correlations,
        "A_single_feature_r2_ranked": dict(sorted(single_r2.items(),
                                                   key=lambda kv: -kv[1])),
        "A_multivariate_ols_standardized": multi,
        "A_csr_by_track_length": [
            {"bin": b, "n": n, "mean_csr": m} for b, n, m in csr_by_len],
        "B_transition_attribution_of_switches": {
            "n_switches_attributed": tot_sw,
            "detector_instability": att["detector_instability"],
            "fragmentation": att["fragmentation"],
            "unattributable": att["unattributable"],
            "frac_detector_instability": (att["detector_instability"] / tot_sw) if tot_sw else None,
            "frac_fragmentation": (att["fragmentation"] / tot_sw) if tot_sw else None,
            "frac_unattributable": (att["unattributable"] / tot_sw) if tot_sw else None,
            "detector_instability_split": {
                "low_conf_confidence_uncertainty": det_lowconf,
                "high_conf_true_ambiguity": det_hiconf,
                "frac_low_conf": (det_lowconf / (det_lowconf + det_hiconf))
                                 if (det_lowconf + det_hiconf) else None,
                "frac_high_conf": (det_hiconf / (det_lowconf + det_hiconf))
                                  if (det_lowconf + det_hiconf) else None,
            },
        },
        "B_switch_rate_by_min_score": [
            {"bin": b, "n": n, "switch_rate": m} for b, n, m in sw_by_score],
        "B_switch_rate_by_object_distance": [
            {"bin": b, "n": n, "switch_rate": m} for b, n, m in sw_by_dist],
    }

    out_file = out_dir / "instability_source.json"
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[is] wrote {out_file}", flush=True)

    # ------------------------------ console -------------------------------- #
    p = payload["pooled"]
    print(f"\n=== POOLED ===")
    print(f"  multiobs_tracks={len(rec_csr)}  singleton={n_singleton}")
    print(f"  pooled_csr={p['pooled_class_switch_rate']:.4f}  "
          f"mean_per_track_csr={p['mean_per_track_csr']:.4f}")

    print("\n=== A. correlations: per-track switch_rate vs factor ===")
    for n, c in correlations.items():
        print(f"  {n:<32} pearson r={c['pearson_r']:+.4f} (r2={c['pearson_r2']:.4f})  "
              f"spearman={c['spearman_rho']:+.4f}")
    print("  single-feature R2 ranked:",
          {k: round(v, 4) for k, v in payload["A_single_feature_r2_ranked"].items()})
    if "standardized_coeffs" in multi:
        print("  OLS std coeffs:",
              {k: round(v, 4) for k, v in multi["standardized_coeffs"].items()},
              f"model_R2={multi['model_r2']:.4f}")

    print("\n  csr by track length:")
    for b, n, m in csr_by_len:
        print(f"    len {b:<12} n={n:>7}  mean_csr={m if m is None else round(m,4)}")

    print("\n=== B. attribution of the switch budget ===")
    bb = payload["B_transition_attribution_of_switches"]
    print(f"  switches={bb['n_switches_attributed']}")
    print(f"    detector_instability  {bb['detector_instability']:>7}  "
          f"({bb['frac_detector_instability']:.3f})")
    print(f"    fragmentation         {bb['fragmentation']:>7}  "
          f"({bb['frac_fragmentation']:.3f})")
    print(f"    unattributable        {bb['unattributable']:>7}  "
          f"({bb['frac_unattributable']:.3f})")
    sp = bb["detector_instability_split"]
    print(f"    within det-instability: low_conf(confidence)={sp['low_conf_confidence_uncertainty']} "
          f"({sp['frac_low_conf']:.3f})  high_conf(ambiguity)={sp['high_conf_true_ambiguity']} "
          f"({sp['frac_high_conf']:.3f})")

    print("\n  switch rate by min-score:")
    for b, n, m in sw_by_score:
        print(f"    score {b:<12} n={n:>8}  sw_rate={m if m is None else round(m,4)}")
    print("  switch rate by object distance:")
    for b, n, m in sw_by_dist:
        print(f"    dist {b:<12} n={n:>8}  sw_rate={m if m is None else round(m,4)}")


if __name__ == "__main__":
    main()
