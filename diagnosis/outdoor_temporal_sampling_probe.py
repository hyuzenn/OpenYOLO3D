"""Temporal sampling-density diagnostic for the γ-cache pipeline.

EVIDENCE-ONLY. No detector output modified; no devkit eval. Question: is the
observed ~10.6 fragments/GT (see outdoor_gt_fragmentation_probe) caused by
sparse keyframe-only sampling (object motion exceeds the 2.0 m associator gate
between frames) or by the associator failing *within* the gate?

For every processed frame we record the keyframe timestamp; for every GT
instance we record its global-BEV position per frame plus whether/which
predicted track (gid) covers it (GT-centric nearest-proposal match, identical
policy to outdoor_gt_fragmentation_probe).

Reports
  1. sampling mode (keyframe vs keyframe+sweep) — structural, asserted via the
     consecutive timestamp gap (keyframe-only ⇒ ≈0.5 s; +sweeps ⇒ ≈0.05 s)
  2. timestamp-gap distribution between consecutive processed frames
  3. GT inter-frame displacement (global): median, p90, fraction > gate
  4. fragmentation decomposed by motion:
       motion_floor(GT)   = 1 + #present-transitions with disp > gate
       observed_frag(GT)  = #distinct predicted ids covering the GT
       excess(GT)         = observed_frag - motion_floor   (associator-attributable)
     plus a static-class control (barrier/traffic_cone: ~0 motion ⇒ any
     fragmentation is associator-attributable by construction), and a
     break-rate-vs-displacement table over consecutive *covered* frames
     (does the id break even when the GT barely moves?).
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
STATIC_CLASSES = {"barrier", "traffic_cone"}  # immobile control group


def _gt_instances(nusc, sample_token):
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
        "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
        "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)), "max": float(a.max()),
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
    ap.add_argument("--match-radius-m", type=float, default=GT_MATCH_RADIUS_M)
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    gamma_cache = Path(args.gamma_cache).resolve()
    if not gamma_cache.exists():
        raise SystemExit(f"cache directory missing: {gamma_cache}")
    gate = float(args.assoc_dist_m)

    date = time.strftime("%Y-%m-%d")
    if args.output_name:
        out_dir = project_root / "results" / args.output_name
    else:
        existing = list(project_root.glob(f"results/{date}_outdoor_temporal_sampling_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_temporal_sampling_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[samp] γ cache    : {gamma_cache}", flush=True)
    print(f"[samp] output dir : {out_dir}", flush=True)
    print(f"[samp] assoc gate : {gate} m   match radius: {args.match_radius_m} m", flush=True)

    print("[samp] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[samp] val scenes : {len(val_scenes)}", flush=True)

    dt_gaps = []                 # consecutive keyframe timestamp gaps (seconds)
    # per (scene,instance): ordered list of (frame_idx, t_sec, xy_global, gid_or_None)
    inst_track = defaultdict(list)
    inst_cls = {}
    t0 = time.time()
    n_missing = 0
    for si, sc_tok in enumerate(val_scenes):
        assoc = ClassAgnosticAssociator(threshold_m=gate, max_age=ASSOC_MAX_AGE)
        assoc.reset()
        prev_t = None
        for fi, sa_tok in enumerate(_scene_sample_tokens(nusc, sc_tok)):
            sample = nusc.get("sample", sa_tok)
            t_sec = sample["timestamp"] * 1e-6  # nuScenes timestamps are microseconds
            if prev_t is not None:
                dt_gaps.append(t_sec - prev_t)
            prev_t = t_sec

            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            ginst = _gt_instances(nusc, sa_tok)

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            props = []
            if raw is not None:
                for p in raw:
                    if p.get("cls_name") not in NUSC_10_SET:
                        continue
                    props.append({
                        "score": float(p.get("score", 0.0)),
                        "cls_name": p["cls_name"],
                        "centroid_ego": np.asarray(p["centroid_ego"], dtype=np.float64),
                    })
            else:
                n_missing += 1
            gids = assoc.step(props)

            P_xy = None
            if props:
                P_xy = np.asarray([_ego_to_global_centroid(p["centroid_ego"], ego_pose)[:2]
                                   for p in props], dtype=np.float64)
            for inst, cls, gxy in ginst:
                inst_cls[inst] = cls
                gid = None
                if P_xy is not None:
                    d = np.linalg.norm(P_xy - gxy, axis=1)
                    j = int(d.argmin())
                    if d[j] <= args.match_radius_m:
                        gid = int(gids[j])
                inst_track[(si, inst)].append((fi, t_sec, gxy, gid))

        if (si + 1) % 25 == 0:
            print(f"[samp] scene {si+1}/{len(val_scenes)} — inst {len(inst_track)} "
                  f"— {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    # 3. GT inter-frame displacement (consecutive PRESENT keyframes)
    # ----------------------------------------------------------------------- #
    disp_all, disp_by_cls = [], defaultdict(list)
    speed_all = []
    for key, seq in inst_track.items():
        cls = inst_cls[key[1]]
        for a, b in zip(seq, seq[1:]):
            if b[0] - a[0] != 1:    # only truly consecutive keyframes
                continue
            d = float(np.linalg.norm(b[2] - a[2]))
            dt = b[1] - a[1]
            disp_all.append(d)
            disp_by_cls[cls].append(d)
            if dt > 0:
                speed_all.append(d / dt)
    disp_all = np.asarray(disp_all, dtype=np.float64)
    speed_all = np.asarray(speed_all, dtype=np.float64)
    frac_disp_over_gate = float((disp_all > gate).mean()) if disp_all.size else 0.0

    # ----------------------------------------------------------------------- #
    # 4. fragmentation decomposed by motion
    #    motion_floor = 1 + #present-transitions with disp > gate
    #    observed_frag = #distinct gids
    #    excess = observed - motion_floor
    # ----------------------------------------------------------------------- #
    rows_obs, rows_floor, rows_excess = [], [], []
    rows_cls = []
    # break-rate vs displacement over consecutive COVERED frames (both detected)
    # bins by displacement
    disp_bin_edges = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9]
    brk_tot = [0] * (len(disp_bin_edges) - 1)
    brk_break = [0] * (len(disp_bin_edges) - 1)

    for key, seq in inst_track.items():
        cls = inst_cls[key[1]]
        gids_present = [g for (_, _, _, g) in seq if g is not None]
        if not gids_present:
            continue  # never detected
        observed = len(set(gids_present))
        # motion floor: count present-frame transitions whose disp exceeds gate
        n_exceed = 0
        for a, b in zip(seq, seq[1:]):
            if b[0] - a[0] != 1:
                continue
            if float(np.linalg.norm(b[2] - a[2])) > gate:
                n_exceed += 1
        floor = 1 + n_exceed
        rows_obs.append(observed)
        rows_floor.append(floor)
        rows_excess.append(observed - floor)
        rows_cls.append(cls)
        # break-rate vs disp over consecutive covered frames (both detected, adjacent)
        for a, b in zip(seq, seq[1:]):
            if b[0] - a[0] != 1:
                continue
            if a[3] is None or b[3] is None:
                continue
            d = float(np.linalg.norm(b[2] - a[2]))
            bi = np.searchsorted(disp_bin_edges, d, side="right") - 1
            bi = max(0, min(bi, len(brk_tot) - 1))
            brk_tot[bi] += 1
            if a[3] != b[3]:
                brk_break[bi] += 1

    rows_obs = np.asarray(rows_obs, dtype=np.float64)
    rows_floor = np.asarray(rows_floor, dtype=np.float64)
    rows_excess = np.asarray(rows_excess, dtype=np.float64)
    rows_cls = np.asarray(rows_cls)

    # static-class control
    static_mask = np.asarray([c in STATIC_CLASSES for c in rows_cls])
    static_obs = rows_obs[static_mask]
    static_floor = rows_floor[static_mask]

    brk_rate = [(brk_break[i] / brk_tot[i]) if brk_tot[i] else None
                for i in range(len(brk_tot))]

    # how much of observed fragmentation is "explained" by motion?
    total_obs_breaks = float((rows_obs - 1).sum())          # observed id-breaks
    total_motion_breaks = float((rows_floor - 1).sum())     # motion-forced breaks
    explained_frac = (total_motion_breaks / total_obs_breaks) if total_obs_breaks > 0 else None

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "assoc_gate_m": gate,
            "assoc_max_age": ASSOC_MAX_AGE,
            "gt_match_radius_m": args.match_radius_m,
            "n_samples_missing_cache": n_missing,
            "static_classes": sorted(STATIC_CLASSES),
        },
        "q1_sampling_mode": {
            "frames_visited": "sample['next'] chain — nuScenes keyframes only (2 Hz)",
            "sweeps_visited": False,
            "median_gap_s": float(np.median(dt_gaps)) if dt_gaps else None,
            "verdict": ("keyframe-only (~0.5 s gap)"
                        if dt_gaps and np.median(dt_gaps) > 0.2
                        else "keyframe+sweeps (~0.05 s gap)"),
        },
        "q2_timestamp_gap_s": _dist(dt_gaps),
        "q3_gt_displacement_m": {
            "all": _dist(disp_all),
            "median_m": float(np.median(disp_all)) if disp_all.size else None,
            "p90_m": float(np.percentile(disp_all, 90)) if disp_all.size else None,
            "frac_over_gate": frac_disp_over_gate,
            "gate_m": gate,
            "speed_mps": _dist(speed_all),
            "per_class_median_p90_fracgate": {
                c: {"n": len(v),
                    "median": float(np.median(v)),
                    "p90": float(np.percentile(v, 90)),
                    "frac_over_gate": float((np.asarray(v) > gate).mean())}
                for c, v in sorted(disp_by_cls.items())
            },
        },
        "q4_fragmentation_vs_motion": {
            "n_detected_gt": int(rows_obs.size),
            "observed_frag": _dist(rows_obs),
            "motion_floor_frag": _dist(rows_floor),
            "excess_frag": _dist(rows_excess),
            "total_observed_id_breaks": total_obs_breaks,
            "total_motion_forced_breaks": total_motion_breaks,
            "frac_breaks_explained_by_motion": explained_frac,
            "static_control": {
                "classes": sorted(STATIC_CLASSES),
                "n": int(static_obs.size),
                "observed_frag": _dist(static_obs),
                "motion_floor_frag": _dist(static_floor),
                "mean_excess": float((static_obs - static_floor).mean()) if static_obs.size else None,
            },
            "break_rate_vs_displacement": {
                "bin_edges_m": disp_bin_edges,
                "n_transitions": brk_tot,
                "n_breaks": brk_break,
                "break_rate": brk_rate,
                "note": ("consecutive covered frames (GT detected in both adjacent "
                         "keyframes); break = predicted id differs. High rate at "
                         "small displacement ⇒ associator-attributable."),
            },
        },
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[samp] wrote {out_dir/'metrics.json'}", flush=True)

    _render_png(out_dir / "outputs", disp_all, gate, disp_bin_edges, brk_rate, brk_tot)
    np.savez_compressed(out_dir / "outputs" / "per_gt_motion.npz",
                        observed=rows_obs, floor=rows_floor, excess=rows_excess,
                        cls=rows_cls, disp=disp_all)
    _write_notes(out_dir, payload)
    _console(payload)


def _render_png(outputs, disp, gate, edges, brk_rate, brk_tot):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[samp] matplotlib unavailable: {e}", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(disp, bins=60, range=(0, max(6.0, np.percentile(disp, 99))),
               color="#3b6", edgecolor="k", lw=0.3)
    ax[0].axvline(gate, color="r", lw=1.2, ls="--", label=f"gate {gate} m")
    ax[0].set_title("GT inter-frame displacement (m)"); ax[0].legend()
    centers = [0.5 * (edges[i] + min(edges[i + 1], edges[i] + 1)) for i in range(len(brk_rate))]
    rates = [r if r is not None else 0 for r in brk_rate]
    ax[1].bar(range(len(rates)), rates, color="#b63", edgecolor="k", lw=0.3)
    ax[1].set_xticks(range(len(rates)))
    ax[1].set_xticklabels([f"{edges[i]:.2g}-{edges[i+1]:.2g}" for i in range(len(rates))],
                          rotation=45, ha="right", fontsize=7)
    ax[1].axvline(np.searchsorted(edges, gate) - 1.5, color="r", lw=1.0, ls="--")
    ax[1].set_title("id-break rate vs displacement (covered frames)")
    ax[1].set_ylabel("P(id changes)")
    fig.tight_layout(); fig.savefig(outputs / "temporal_sampling.png", dpi=110); plt.close(fig)
    print(f"[samp] wrote PNG to {outputs}", flush=True)


def _write_notes(out_dir, p):
    c = p["config"]; q1 = p["q1_sampling_mode"]; q2 = p["q2_timestamp_gap_s"]
    q3 = p["q3_gt_displacement_m"]; q4 = p["q4_fragmentation_vs_motion"]
    L = []
    L.append("# Temporal sampling density vs fragmentation\n")
    L.append("Probe: `diagnosis/outdoor_temporal_sampling_probe.py` (evidence-only). "
             f"γ cache, full val ({c['n_val_scenes']} scenes). Associator gate "
             f"{c['assoc_gate_m']} m, max_age {c['assoc_max_age']}; GT match radius "
             f"{c['gt_match_radius_m']} m (global BEV).\n")

    L.append("## 1. Sampling mode")
    L.append(f"- frames visited: {q1['frames_visited']}")
    L.append(f"- sweeps visited: **{q1['sweeps_visited']}**")
    L.append(f"- median consecutive-frame gap: **{q1['median_gap_s']:.3f} s** → {q1['verdict']}\n")

    L.append("## 2. Timestamp gap between consecutive processed frames (s)")
    L.append("| mean | std | p5 | p50 | p95 | max |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| {q2['mean']:.3f} | {q2['std']:.3f} | {q2['p5']:.3f} | {q2['p50']:.3f} | "
             f"{q2['p95']:.3f} | {q2['max']:.3f} |\n")

    L.append("## 3. GT inter-frame displacement (global BEV, consecutive present keyframes)")
    a = q3["all"]
    L.append(f"- **median = {q3['median_m']:.3f} m**, p90 = {q3['p90_m']:.3f} m, "
             f"p95 = {a['p95']:.3f} m, max = {a['max']:.2f} m")
    L.append(f"- **fraction moving > gate ({q3['gate_m']} m) = {q3['frac_over_gate']*100:.2f}%**")
    sp = q3["speed_mps"]
    L.append(f"- implied speed: median {sp['p50']:.2f} m/s, p90 {sp['p90']:.2f} m/s, max {sp['max']:.1f} m/s\n")
    L.append("| class | n | median_m | p90_m | %>gate |")
    L.append("|---|---|---|---|---|")
    for cls, b in sorted(q3["per_class_median_p90_fracgate"].items(),
                         key=lambda kv: -kv[1]["frac_over_gate"]):
        L.append(f"| {cls} | {b['n']:,} | {b['median']:.3f} | {b['p90']:.3f} | {b['frac_over_gate']*100:.2f}% |")
    L.append("")

    L.append("## 4. Fragmentation decomposed by motion")
    ob = q4["observed_frag"]; fl = q4["motion_floor_frag"]; ex = q4["excess_frag"]
    L.append(f"- observed fragments / detected GT: mean **{ob['mean']:.2f}** (p50 {ob['p50']:.0f})")
    L.append(f"- motion floor (1 + #transitions with disp>gate): mean **{fl['mean']:.2f}** (p50 {fl['p50']:.0f})")
    L.append(f"- **excess (associator-attributable): mean {ex['mean']:.2f}** (p50 {ex['p50']:.0f})")
    L.append(f"- of {q4['total_observed_id_breaks']:.0f} observed id-breaks, "
             f"{q4['total_motion_forced_breaks']:.0f} are motion-forced "
             f"→ **{q4['frac_breaks_explained_by_motion']*100:.1f}% explained by sparse sampling**, "
             f"{(1-q4['frac_breaks_explained_by_motion'])*100:.1f}% associator.\n")
    sc = q4["static_control"]
    L.append(f"### Static control ({', '.join(sc['classes'])}; ~0 motion)")
    L.append(f"- n = {sc['n']:,}; observed frag mean **{sc['observed_frag']['mean']:.2f}**, "
             f"motion floor mean {sc['motion_floor_frag']['mean']:.2f}, "
             f"excess mean **{sc['mean_excess']:.2f}**")
    L.append("- immobile objects cannot exceed the gate by motion, so their "
             "fragmentation is associator-attributable by construction.\n")
    br = q4["break_rate_vs_displacement"]
    L.append("### id-break rate vs displacement (consecutive covered frames)")
    L.append("| disp bin (m) | n | breaks | break rate |")
    L.append("|---|---|---|---|")
    for i in range(len(br["break_rate"])):
        lo, hi = br["bin_edges_m"][i], br["bin_edges_m"][i + 1]
        hi_s = "∞" if hi > 1e8 else f"{hi:.2g}"
        r = br["break_rate"][i]
        L.append(f"| {lo:.2g}–{hi_s} | {br['n_transitions'][i]:,} | {br['n_breaks'][i]:,} | "
                 f"{'—' if r is None else f'{r*100:.1f}%'} |")
    L.append(f"\n_{br['note']}_")
    L.append("(PNG: `outputs/temporal_sampling.png`)")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[samp] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    q1 = p["q1_sampling_mode"]; q3 = p["q3_gt_displacement_m"]; q4 = p["q4_fragmentation_vs_motion"]
    print("\n=== temporal sampling ===")
    print(f"  sampling: {q1['verdict']}  (median gap {q1['median_gap_s']:.3f}s, sweeps={q1['sweeps_visited']})")
    print(f"  GT disp: median={q3['median_m']:.3f}m p90={q3['p90_m']:.3f}m  >gate={q3['frac_over_gate']*100:.2f}%")
    print(f"  frag: observed mean={q4['observed_frag']['mean']:.2f}  "
          f"motion_floor={q4['motion_floor_frag']['mean']:.2f}  excess={q4['excess_frag']['mean']:.2f}")
    print(f"  breaks explained by motion: {q4['frac_breaks_explained_by_motion']*100:.1f}%")
    sc = q4["static_control"]
    print(f"  static control ({'+'.join(sc['classes'])}): observed frag={sc['observed_frag']['mean']:.2f} "
          f"excess={sc['mean_excess']:.2f}")
    br = q4["break_rate_vs_displacement"]
    print("  break-rate vs disp:", [None if r is None else round(r, 3) for r in br["break_rate"]])


if __name__ == "__main__":
    main()
