"""GT-anchored track fragmentation diagnostic.

EVIDENCE-ONLY. No detector output is modified; no devkit eval. Reuses the γ
CenterPoint cache and the existing geometry-only streaming associator
(``ClassAgnosticAssociator``, 2.0 m gate, max_age 5) to build predicted tracks,
then anchors them to nuScenes GT *instances* (persistent ``instance_token``).

Matching is GT-centric (CLEAR-MOT style): in every keyframe, each GT box of a
nuScenes-10 class is matched to its nearest predicted proposal within
``GT_MATCH_RADIUS_M`` (global BEV). The matched proposal's streaming track id
(gid) is the id covering that GT in that frame.

Per GT object (instance_token within a scene) we collect
  - lifetime         = # keyframes the instance is present (regardless of detection)
  - covered_frames   = # keyframes a predicted track covers it
  - gid sequence over covered frames
Fragmentation count = # DISTINCT predicted track ids covering the instance over
its life. 0 = never detected, 1 = single clean track, >=2 = fragmented.

Reports
  1. GT lifetime distribution
  2. predicted track lifetime distribution (matched tracks + all)
  3. average fragments per GT object
  4. % of GT objects split into multiple short tracks (frag >= 2)
  5. theoretical track length if fragmentation were removed
     (= union of covered frames per GT vs current per-fragment length)
plus per-class breakdown, fragmentation histogram, detection coverage.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.detection.utils import category_to_detection_name

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10_SET,
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


def _gt_instances(nusc, sample_token):
    """[(instance_token, cls_name, xy_global)] for nuScenes-10 GT in this frame."""
    sample = nusc.get("sample", sample_token)
    out = []
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        det = category_to_detection_name(ann["category_name"])
        if det is None or det not in NUSC_10_SET:
            continue
        out.append((ann["instance_token"], det,
                    np.asarray(ann["translation"][:2], dtype=np.float64)))
    return out


def _dist(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()),
        "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)), "max": float(a.max()),
    }


def _hist(arr, lo, hi, nbins):
    counts, edges = np.histogram(np.asarray(arr, dtype=np.float64), bins=nbins, range=(lo, hi))
    return {"bin_edges": [float(x) for x in edges], "counts": [int(c) for c in counts]}


def _int_hist(arr, cap):
    """Counts for integer values 0..cap and a >cap bucket."""
    c = Counter(int(min(x, cap + 1)) for x in arr)
    rows = [(str(i), c.get(i, 0)) for i in range(0, cap + 1)]
    rows.append((f">{cap}", c.get(cap + 1, 0)))
    return rows


def _ascii(rows, width=46):
    mx = max((v for _, v in rows), default=1) or 1
    return "\n".join(f"  {lbl:>5} {v:>9} |{'#' * int(round(width * v / mx))}" for lbl, v in rows)


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" /
        "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--assoc-dist-m", type=float, default=DEFAULT_ASSOC_DIST_M)
    ap.add_argument("--match-radius-m", type=float, default=GT_MATCH_RADIUS_M)
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
        existing = list(project_root.glob(f"results/{date}_outdoor_gt_fragmentation_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_gt_fragmentation_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[frag] γ cache    : {gamma_cache}", flush=True)
    print(f"[frag] output dir : {out_dir}", flush=True)
    print(f"[frag] match radius: {args.match_radius_m} m (GT-centric, global BEV)", flush=True)

    print("[frag] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[frag] val scenes : {len(val_scenes)}", flush=True)

    # per-GT-instance accumulators (instance_token unique across the dataset)
    gt_life = defaultdict(int)                       # instance -> #keyframes present
    gt_cls = {}                                      # instance -> cls
    gt_cov_frames = defaultdict(set)                 # instance -> set(frame_idx covered)
    gt_gid_frames = defaultdict(lambda: defaultdict(set))  # instance -> gid -> {frame_idx}
    pred_track_len = defaultdict(int)                # (scene,gid) -> #frames
    pred_track_matched = defaultdict(bool)           # (scene,gid) -> ever matched a GT

    t0 = time.time()
    n_missing = 0
    for si, sc_tok in enumerate(val_scenes):
        assoc = ClassAgnosticAssociator(threshold_m=args.assoc_dist_m, max_age=ASSOC_MAX_AGE)
        assoc.reset()
        for fi, sa_tok in enumerate(_scene_sample_tokens(nusc, sc_tok)):
            sample = nusc.get("sample", sa_tok)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            # GT instances present this frame -> lifetime
            ginst = _gt_instances(nusc, sa_tok)
            for inst, cls, _xy in ginst:
                gt_life[inst] += 1
                gt_cls[inst] = cls

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                continue

            props = []
            for p in raw:
                if p.get("cls_name") not in NUSC_10_SET:
                    continue
                props.append({
                    "score": float(p.get("score", 0.0)),
                    "cls_name": p["cls_name"],
                    "centroid_ego": np.asarray(p["centroid_ego"], dtype=np.float64),
                })
            gids = assoc.step(props)
            for gid in gids:
                pred_track_len[(si, gid)] += 1

            if not props or not ginst:
                continue
            # proposal global xy
            P_xy = np.asarray([_ego_to_global_centroid(p["centroid_ego"], ego_pose)[:2]
                               for p in props], dtype=np.float64)
            # GT-centric nearest-proposal match within radius
            for inst, cls, gxy in ginst:
                d = np.linalg.norm(P_xy - gxy, axis=1)
                j = int(d.argmin())
                if d[j] <= args.match_radius_m:
                    gid = gids[j]
                    gt_cov_frames[inst].add((si, fi))
                    gt_gid_frames[inst][gid].add(fi)
                    pred_track_matched[(si, gid)] = True

        if (si + 1) % 25 == 0:
            print(f"[frag] scene {si+1}/{len(val_scenes)} — GT inst {len(gt_life)} "
                  f"— {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    # per-GT metrics
    # ----------------------------------------------------------------------- #
    all_insts = list(gt_life.keys())
    life_all = np.asarray([gt_life[i] for i in all_insts], dtype=np.float64)
    frag = np.asarray([len(gt_gid_frames[i]) for i in all_insts], dtype=np.float64)  # 0 if never matched
    covered = np.asarray([len(gt_cov_frames[i]) for i in all_insts], dtype=np.float64)
    detected = frag >= 1
    multi = frag >= 2

    # theoretical lengths
    # current per-fragment length (mean over (inst,gid) segments)
    frag_seg_lengths = []
    merged_lengths = []        # union of covered frames per detected GT (= covered)
    for i in all_insts:
        if not gt_gid_frames[i]:
            continue
        for gid, frames in gt_gid_frames[i].items():
            frag_seg_lengths.append(len(frames))
        merged_lengths.append(len(gt_cov_frames[i]))
    frag_seg_lengths = np.asarray(frag_seg_lengths, dtype=np.float64)
    merged_lengths = np.asarray(merged_lengths, dtype=np.float64)

    # predicted track lifetimes
    ptl_all = np.asarray(list(pred_track_len.values()), dtype=np.float64)
    ptl_matched = np.asarray([L for k, L in pred_track_len.items() if pred_track_matched.get(k)],
                             dtype=np.float64)

    # per-class
    per_class = {}
    cls_of = {i: gt_cls[i] for i in all_insts}
    for c in sorted(NUSC_10_SET):
        idx = [k for k, i in enumerate(all_insts) if cls_of[i] == c]
        if not idx:
            continue
        idx = np.asarray(idx)
        det_c = detected[idx]
        f_det = frag[idx][det_c]
        per_class[c] = {
            "n_gt_instances": int(idx.size),
            "n_detected": int(det_c.sum()),
            "mean_lifetime": float(life_all[idx].mean()),
            "mean_fragments_detected": float(f_det.mean()) if f_det.size else None,
            "pct_multi_track_of_detected": float(100.0 * (frag[idx][det_c] >= 2).mean()) if det_c.any() else None,
            "mean_covered_frames_detected": float(covered[idx][det_c].mean()) if det_c.any() else None,
        }

    # ----------------------------------------------------------------------- #
    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "associator": "ClassAgnosticAssociator (geometry-only)",
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "gt_match_radius_m": args.match_radius_m,
            "match_policy": "GT-centric nearest proposal within radius (global BEV)",
            "n_samples_missing_cache": n_missing,
            "n_gt_instances": len(all_insts),
            "n_gt_detected": int(detected.sum()),
            "n_predicted_tracks": len(pred_track_len),
        },
        "gt_lifetime_distribution": _dist(life_all),
        "predicted_track_lifetime": {
            "all_tracks": _dist(ptl_all),
            "matched_tracks": _dist(ptl_matched),
        },
        "fragmentation": {
            "mean_fragments_per_gt_all": float(frag.mean()),
            "mean_fragments_per_detected_gt": float(frag[detected].mean()) if detected.any() else None,
            "n_detected": int(detected.sum()),
            "n_never_detected": int((~detected).sum()),
            "pct_detected": float(100.0 * detected.mean()),
            "pct_gt_multi_track_of_all": float(100.0 * multi.mean()),
            "pct_gt_multi_track_of_detected": float(100.0 * multi[detected].mean()) if detected.any() else None,
            "fragments_distribution_detected": _dist(frag[detected]),
        },
        "theoretical_defragmented_length": {
            "mean_fragment_segment_length": float(frag_seg_lengths.mean()) if frag_seg_lengths.size else None,
            "mean_merged_length_covered_frames": float(merged_lengths.mean()) if merged_lengths.size else None,
            "uplift_factor": (float(merged_lengths.mean() / frag_seg_lengths.mean())
                              if frag_seg_lengths.size and frag_seg_lengths.mean() > 0 else None),
            "mean_gt_lifetime_detected": float(life_all[detected].mean()) if detected.any() else None,
            "mean_coverage_ratio_detected": float((covered[detected] / life_all[detected]).mean()) if detected.any() else None,
            "note": ("merged length = union of covered frames per GT (fragmentation removed). "
                     "GT lifetime is the ceiling if detection gaps were also filled."),
        },
        "per_class": per_class,
        "histograms": {
            "gt_lifetime": _hist(life_all, 0, float(max(20.0, life_all.max())), 20),
            "fragments_detected": _int_hist(frag[detected], 8),
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[frag] wrote {out_dir/'metrics.json'}", flush=True)

    _render_pngs(out_dir / "outputs", life_all, frag[detected], ptl_matched)
    np.savez_compressed(out_dir / "outputs" / "per_gt.npz",
                        lifetime=life_all, fragments=frag, covered=covered,
                        cls=np.asarray([cls_of[i] for i in all_insts]))
    _write_notes(out_dir, payload)
    _console(payload)


def _render_pngs(outputs, life, frag_det, ptl_matched):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[frag] matplotlib unavailable: {e}", flush=True)
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].hist(life, bins=30, range=(0, max(20, life.max())), color="#3b6", edgecolor="k", lw=0.3)
    ax[0].set_title("GT lifetime (frames)"); ax[0].set_ylabel("GT instances")
    ax[1].hist(ptl_matched, bins=30, range=(0, max(20, ptl_matched.max() if ptl_matched.size else 20)),
               color="#36b", edgecolor="k", lw=0.3)
    ax[1].set_title("matched predicted-track lifetime")
    ax[2].hist(frag_det, bins=range(1, 12), color="#b63", edgecolor="k", lw=0.3, align="left")
    ax[2].set_title("fragments per detected GT")
    fig.tight_layout(); fig.savefig(outputs / "fragmentation.png", dpi=110); plt.close(fig)
    print(f"[frag] wrote PNG to {outputs}", flush=True)


def _write_notes(out_dir, p):
    c = p["config"]; fr = p["fragmentation"]; th = p["theoretical_defragmented_length"]
    gl = p["gt_lifetime_distribution"]; pt = p["predicted_track_lifetime"]
    L = []
    L.append("# GT-anchored track fragmentation\n")
    L.append("Probe: `diagnosis/outdoor_gt_fragmentation_probe.py` (evidence-only; no")
    L.append("detector output modified, no devkit eval). γ cache, full val "
             f"({c['n_val_scenes']} scenes). Predicted tracks = geometry-only "
             f"`ClassAgnosticAssociator` ({c['assoc_threshold_m']} m, max_age "
             f"{c['assoc_max_age']}). GT = nuScenes-10 instances; GT-centric nearest-"
             f"proposal match within {c['gt_match_radius_m']} m (global BEV).\n")
    L.append(f"Corpus: **{c['n_gt_instances']:,} GT instances** "
             f"({c['n_gt_detected']:,} ever detected = {fr['pct_detected']:.1f}%), "
             f"{c['n_predicted_tracks']:,} predicted tracks.\n")

    L.append("## 1. GT lifetime distribution (keyframes per instance)")
    L.append("| mean | std | p5 | p25 | p50 | p75 | p95 | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| {gl['mean']:.2f} | {gl['std']:.2f} | {gl['p5']:.0f} | {gl['p25']:.0f} | "
             f"{gl['p50']:.0f} | {gl['p75']:.0f} | {gl['p95']:.0f} | {gl['max']:.0f} |\n")

    L.append("## 2. Predicted track lifetime distribution (frames)")
    L.append("| set | mean | std | p50 | p95 | max |")
    L.append("|---|---|---|---|---|---|")
    for nm, k in (("all tracks", "all_tracks"), ("matched tracks", "matched_tracks")):
        s = pt[k]
        L.append(f"| {nm} | {s['mean']:.2f} | {s['std']:.2f} | {s['p50']:.0f} | {s['p95']:.0f} | {s['max']:.0f} |")
    L.append("")

    L.append("## 3. Average fragments per GT object")
    L.append(f"- mean fragments / **detected** GT: **{fr['mean_fragments_per_detected_gt']:.3f}**")
    L.append(f"- mean fragments / GT incl. misses: {fr['mean_fragments_per_gt_all']:.3f} "
             f"({fr['n_never_detected']:,} never detected)")
    fd = fr["fragments_distribution_detected"]
    L.append(f"- detected-GT fragment count: p50={fd['p50']:.0f}, p95={fd['p95']:.0f}, max={fd['max']:.0f}\n")

    L.append("## 4. % GT objects split into multiple (short) tracks")
    L.append(f"- frag ≥ 2 among **detected** GT: **{fr['pct_gt_multi_track_of_detected']:.1f}%**")
    L.append(f"- frag ≥ 2 among **all** GT: {fr['pct_gt_multi_track_of_all']:.1f}%\n")

    L.append("## 5. Theoretical track length if fragmentation removed")
    L.append("| quantity | frames |")
    L.append("|---|---|")
    L.append(f"| mean current fragment-segment length | {th['mean_fragment_segment_length']:.2f} |")
    L.append(f"| mean merged length (union of covered frames) | {th['mean_merged_length_covered_frames']:.2f} |")
    L.append(f"| **uplift factor** | **×{th['uplift_factor']:.2f}** |")
    L.append(f"| mean GT lifetime (detected) — ceiling | {th['mean_gt_lifetime_detected']:.2f} |")
    L.append(f"| mean coverage ratio (covered/lifetime) | {th['mean_coverage_ratio_detected']:.3f} |")
    L.append(f"\n_{th['note']}_\n")

    L.append("## Per-class")
    L.append("| class | n_gt | det% | mean_life | mean_frag(det) | %multi(det) | cov_frames(det) |")
    L.append("|---|---|---|---|---|---|---|")
    for cls, b in sorted(p["per_class"].items(), key=lambda kv: -kv[1]["n_gt_instances"]):
        detpct = 100.0 * b["n_detected"] / b["n_gt_instances"] if b["n_gt_instances"] else 0
        mf = b["mean_fragments_detected"]; pm = b["pct_multi_track_of_detected"]; cv = b["mean_covered_frames_detected"]
        L.append(f"| {cls} | {b['n_gt_instances']:,} | {detpct:.0f}% | {b['mean_lifetime']:.2f} | "
                 f"{'—' if mf is None else f'{mf:.2f}'} | {'—' if pm is None else f'{pm:.0f}%'} | "
                 f"{'—' if cv is None else f'{cv:.2f}'} |")
    L.append("")
    L.append("## Histogram — fragments per detected GT")
    L.append("```")
    L.append(_ascii(p["histograms"]["fragments_detected"]))
    L.append("```")
    L.append("(PNG: `outputs/fragmentation.png`)")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[frag] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    c = p["config"]; fr = p["fragmentation"]; th = p["theoretical_defragmented_length"]
    gl = p["gt_lifetime_distribution"]; ptm = p["predicted_track_lifetime"]["matched_tracks"]
    print("\n=== GT fragmentation ===")
    print(f"  GT instances={c['n_gt_instances']:,}  detected={c['n_gt_detected']:,} "
          f"({fr['pct_detected']:.1f}%)")
    print(f"  GT lifetime: mean={gl['mean']:.2f} p50={gl['p50']:.0f} p95={gl['p95']:.0f} max={gl['max']:.0f}")
    print(f"  matched pred track len: mean={ptm['mean']:.2f} p50={ptm['p50']:.0f} max={ptm['max']:.0f}")
    print(f"  fragments/detected GT: mean={fr['mean_fragments_per_detected_gt']:.3f}")
    print(f"  %multi-track (detected): {fr['pct_gt_multi_track_of_detected']:.1f}%  "
          f"(all GT: {fr['pct_gt_multi_track_of_all']:.1f}%)")
    print(f"  defrag: fragment_len={th['mean_fragment_segment_length']:.2f} -> "
          f"merged={th['mean_merged_length_covered_frames']:.2f} (×{th['uplift_factor']:.2f}); "
          f"lifetime ceiling={th['mean_gt_lifetime_detected']:.2f}, "
          f"coverage={th['mean_coverage_ratio_detected']:.3f}")


if __name__ == "__main__":
    main()
