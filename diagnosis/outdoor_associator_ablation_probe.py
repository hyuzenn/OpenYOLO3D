"""Associator-design ablation: what recovers GT lifetime 17.5 → track 4.7?

EVIDENCE-ONLY. No production code modified; no devkit eval. Builds several
*self-contained* associator variants and runs them all in ONE data pass over the
γ cache, re-measuring GT-anchored fragmentation (distinct predicted ids per GT
instance) for each. GT-centric nearest-proposal matching (global BEV, 2.0 m) is
identical to outdoor_gt_fragmentation_probe, so the per-variant fragment counts
are directly comparable to the 10.64 baseline.

Why these knobs (see outdoor_temporal_sampling_probe): only ~18% of id-breaks
are motion-forced; the production `ClassAgnosticAssociator` matches in the
**ego frame** with no ego-motion compensation, so stationary objects flow out of
the 2 m gate as the ego car moves. We therefore ablate, incrementally:
  * frame      : ego-frame  vs  global-frame (ego-motion compensation)
  * assignment : greedy(score order)  vs  Hungarian(optimal one-to-one)
  * motion     : last-position  vs  constant-velocity prediction
  * persistence: max_age coast length (drop ∈ {2,5,10,20})

Each variant is the same matcher with these flags; the BASELINE variant
(ego/greedy/static/age5) reproduces the production associator and should land at
~10.64 fragments/detected GT.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
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

GATE_M = 2.0
GT_MATCH_RADIUS_M = 2.0
DEFAULT_DT = 0.5  # nuScenes keyframe gap fallback


class Associator:
    """Self-contained centroid associator with switchable design knobs.

    frame is applied by the caller (it passes ego-frame or global-frame xy);
    this class only sees 2-D points. greedy vs Hungarian, constant-velocity
    motion prediction, and max_age coasting are flags.
    """

    def __init__(self, gate: float, max_age: int, hungarian: bool, motion: bool):
        self.gate = float(gate)
        self.max_age = int(max_age)
        self.hungarian = bool(hungarian)
        self.motion = bool(motion)
        self.reset()

    def reset(self):
        self._t = {}        # gid -> {"pos": (2,), "vel": (2,), "age": int}
        self._next = 0

    def step(self, P, scores, dt):
        """P: (n,2) positions in the caller's chosen frame. Returns list[int] gids."""
        # age + drop
        for gid in list(self._t.keys()):
            self._t[gid]["age"] += 1
            if self._t[gid]["age"] > self.max_age:
                self._t.pop(gid, None)
        n = len(P)
        if n == 0:
            return []
        P = np.asarray(P, dtype=np.float64).reshape(n, 2)
        gids: list = [None] * n
        active = list(self._t.keys())
        dt = float(dt) if dt and dt > 0 else DEFAULT_DT

        # predicted track positions
        if active:
            pred = np.asarray([
                self._t[g]["pos"] + (self._t[g]["vel"] * dt if self.motion else 0.0)
                for g in active], dtype=np.float64).reshape(len(active), 2)
        else:
            pred = np.empty((0, 2), dtype=np.float64)

        matched_pairs = []   # (proposal_idx, gid)
        if active:
            if self.hungarian:
                # cost = distance; gate by masking to a large finite cost
                C = np.linalg.norm(P[:, None, :] - pred[None, :, :], axis=2)  # (n, A)
                big = self.gate * 100.0
                Cc = np.where(C <= self.gate, C, big)
                rows, cols = linear_sum_assignment(Cc)
                for r, cdx in zip(rows, cols):
                    if C[r, cdx] <= self.gate:
                        matched_pairs.append((int(r), active[int(cdx)]))
            else:
                used = set()
                order = np.argsort(-np.asarray(scores, dtype=np.float64))
                for j in order:
                    diff = pred - P[j]
                    d = np.linalg.norm(diff, axis=1)
                    # mask used tracks
                    for ui, g in enumerate(active):
                        if g in used:
                            d[ui] = np.inf
                    k = int(np.argmin(d)) if d.size else -1
                    if k >= 0 and d[k] <= self.gate:
                        matched_pairs.append((int(j), active[k]))
                        used.add(active[k])

        matched_props = set()
        for j, gid in matched_pairs:
            old = self._t[gid]["pos"]
            new = P[j]
            self._t[gid]["vel"] = (new - old) / dt
            self._t[gid]["pos"] = new
            self._t[gid]["age"] = 0
            gids[j] = gid
            matched_props.add(j)
        for j in range(n):
            if j in matched_props:
                continue
            g = self._next
            self._next += 1
            self._t[g] = {"pos": P[j].copy(), "vel": np.zeros(2), "age": 0}
            gids[j] = g
        return [int(x) for x in gids]


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


# variant table: name -> (frame, hungarian, motion, max_age)
def _variants():
    V = {}
    V["BASELINE_ego_greedy_static_a5"] = ("ego", False, False, 5)
    # single-knob from baseline (ego frame)
    V["ego_hungarian_static_a5"] = ("ego", True, False, 5)
    V["ego_greedy_motion_a5"] = ("ego", False, True, 5)
    # frame convention (the suspected dominant knob)
    V["global_greedy_static_a5"] = ("global", False, False, 5)
    # on global frame, add each knob
    V["global_hungarian_static_a5"] = ("global", True, False, 5)
    V["global_greedy_motion_a5"] = ("global", False, True, 5)
    V["global_hungarian_motion_a5"] = ("global", True, True, 5)
    # persistence sweep on global/greedy/static
    V["global_greedy_static_a2"] = ("global", False, False, 2)
    V["global_greedy_static_a10"] = ("global", False, False, 10)
    V["global_greedy_static_a20"] = ("global", False, False, 20)
    # full stack
    V["global_hungarian_motion_a10"] = ("global", True, True, 10)
    return V


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" /
        "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
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
        existing = list(project_root.glob(f"results/{date}_outdoor_associator_ablation_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_associator_ablation_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[abl] γ cache    : {gamma_cache}", flush=True)
    print(f"[abl] output dir : {out_dir}", flush=True)

    print("[abl] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[abl] val scenes : {len(val_scenes)}", flush=True)

    vtable = _variants()
    assocs = {name: Associator(GATE_M, max_age, hung, motion)
              for name, (frame, hung, motion, max_age) in vtable.items()}

    gt_life = defaultdict(int)
    gt_cls = {}
    # per-variant: instance -> set(gid), and instance -> set(covered frame idx)
    gt_gid = {name: defaultdict(set) for name in vtable}
    gt_cov = {name: defaultdict(set) for name in vtable}

    t0 = time.time()
    n_missing = 0
    for si, sc_tok in enumerate(val_scenes):
        for a in assocs.values():
            a.reset()
        prev_t = None
        for fi, sa_tok in enumerate(_scene_sample_tokens(nusc, sc_tok)):
            sample = nusc.get("sample", sa_tok)
            t_sec = sample["timestamp"] * 1e-6
            dt = (t_sec - prev_t) if prev_t is not None else DEFAULT_DT
            prev_t = t_sec

            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            ginst = _gt_instances(nusc, sa_tok)
            for inst, cls, _ in ginst:
                gt_life[inst] += 1
                gt_cls[inst] = cls

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            props = []
            if raw is not None:
                for p in raw:
                    if p.get("cls_name") not in NUSC_10_SET:
                        continue
                    props.append((float(p.get("score", 0.0)),
                                  np.asarray(p["centroid_ego"], dtype=np.float64)))
            else:
                n_missing += 1
            if not props:
                # still must advance ages so max_age coasting is consistent
                for name in vtable:
                    assocs[name].step(np.empty((0, 2)), [], dt)
                continue

            scores = np.asarray([s for s, _ in props], dtype=np.float64)
            C_ego = np.asarray([c for _, c in props], dtype=np.float64)  # (n,3)
            P_ego = C_ego[:, :2]
            P_glob = np.asarray([_ego_to_global_centroid(c, ego_pose)[:2] for _, c in props],
                                dtype=np.float64)

            gids_v = {}
            for name, (frame, hung, motion, max_age) in vtable.items():
                P = P_glob if frame == "global" else P_ego
                gids_v[name] = assocs[name].step(P, scores, dt)

            if not ginst:
                continue
            for inst, cls, gxy in ginst:
                d = np.linalg.norm(P_glob - gxy, axis=1)
                j = int(d.argmin())
                if d[j] <= args.match_radius_m:
                    for name in vtable:
                        gt_gid[name][inst].add(gids_v[name][j])
                        gt_cov[name][inst].add((si, fi))

        if (si + 1) % 25 == 0:
            print(f"[abl] scene {si+1}/{len(val_scenes)} — inst {len(gt_life)} "
                  f"— {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    all_insts = list(gt_life.keys())
    life = np.asarray([gt_life[i] for i in all_insts], dtype=np.float64)
    results = {}
    baseline_name = "BASELINE_ego_greedy_static_a5"
    for name in vtable:
        frag = np.asarray([len(gt_gid[name][i]) for i in all_insts], dtype=np.float64)
        cov = np.asarray([len(gt_cov[name][i]) for i in all_insts], dtype=np.float64)
        det = frag >= 1
        # fragment segment length: covered frames / fragments per detected GT
        seg = []
        for i in all_insts:
            f = len(gt_gid[name][i])
            if f >= 1:
                seg.append(len(gt_cov[name][i]) / f)
        seg = np.asarray(seg, dtype=np.float64)
        results[name] = {
            "frame": vtable[name][0], "hungarian": vtable[name][1],
            "motion": vtable[name][2], "max_age": vtable[name][3],
            "n_detected": int(det.sum()),
            "mean_frag_detected": float(frag[det].mean()) if det.any() else None,
            "p50_frag_detected": float(np.percentile(frag[det], 50)) if det.any() else None,
            "p95_frag_detected": float(np.percentile(frag[det], 95)) if det.any() else None,
            "pct_multi_track_detected": float(100.0 * (frag[det] >= 2).mean()) if det.any() else None,
            "mean_covered_frames_detected": float(cov[det].mean()) if det.any() else None,
            "mean_fragment_segment_len": float(seg.mean()) if seg.size else None,
        }

    base = results[baseline_name]["mean_frag_detected"]
    for name in vtable:
        mf = results[name]["mean_frag_detected"]
        results[name]["frag_reduction_vs_baseline_pct"] = (
            float(100.0 * (base - mf) / base) if (base and mf is not None) else None)

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "gate_m": GATE_M, "gt_match_radius_m": args.match_radius_m,
            "n_samples_missing_cache": n_missing,
            "n_gt_instances": len(all_insts),
            "baseline_variant": baseline_name,
            "gt_lifetime_mean": float(life.mean()),
        },
        "variants": results,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[abl] wrote {out_dir/'metrics.json'}", flush=True)
    _write_notes(out_dir, payload)
    _console(payload)


def _row(name, r):
    fr = "glob" if r["frame"] == "global" else "ego"
    asg = "Hung" if r["hungarian"] else "greedy"
    mo = "mot" if r["motion"] else "stat"
    mf = r["mean_frag_detected"]
    red = r["frag_reduction_vs_baseline_pct"]
    return (f"| {name} | {fr} | {asg} | {mo} | {r['max_age']} | "
            f"{'—' if mf is None else f'{mf:.2f}'} | {r['p50_frag_detected']:.0f} | "
            f"{r['pct_multi_track_detected']:.1f}% | {r['mean_covered_frames_detected']:.2f} | "
            f"{'—' if red is None else f'{red:+.1f}%'} |")


def _write_notes(out_dir, p):
    c = p["config"]; R = p["variants"]
    L = []
    L.append("# Associator-design ablation — recovering GT lifetime → track length\n")
    L.append("Probe: `diagnosis/outdoor_associator_ablation_probe.py` (evidence-only). "
             f"γ cache, full val ({c['n_val_scenes']} scenes), {c['n_gt_instances']:,} GT "
             f"instances, GT lifetime mean **{c['gt_lifetime_mean']:.2f}** frames. All "
             "variants run in one data pass; fragments = distinct predicted ids per GT "
             f"instance (GT-centric nearest-proposal match, global BEV {c['gt_match_radius_m']} m). "
             f"Gate {c['gate_m']} m.\n")
    L.append("| variant | frame | assign | motion | max_age | mean_frag | p50 | %multi | cov_frames | Δ vs base |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    order = [
        "BASELINE_ego_greedy_static_a5",
        "ego_hungarian_static_a5", "ego_greedy_motion_a5",
        "global_greedy_static_a5",
        "global_hungarian_static_a5", "global_greedy_motion_a5", "global_hungarian_motion_a5",
        "global_greedy_static_a2", "global_greedy_static_a10", "global_greedy_static_a20",
        "global_hungarian_motion_a10",
    ]
    for name in order:
        if name in R:
            L.append(_row(name, R[name]))
    L.append("")
    base = R["BASELINE_ego_greedy_static_a5"]["mean_frag_detected"]

    def red(name):
        return R[name]["frag_reduction_vs_baseline_pct"]
    L.append("## Single-knob effect (each added alone vs BASELINE)")
    L.append(f"- **frame: ego → global** (ego-motion compensation): "
             f"{base:.2f} → {R['global_greedy_static_a5']['mean_frag_detected']:.2f} "
             f"(**{red('global_greedy_static_a5'):+.1f}%**)")
    L.append(f"- **assignment: greedy → Hungarian** (ego frame): "
             f"{base:.2f} → {R['ego_hungarian_static_a5']['mean_frag_detected']:.2f} "
             f"({red('ego_hungarian_static_a5'):+.1f}%)")
    L.append(f"- **motion: static → const-velocity** (ego frame): "
             f"{base:.2f} → {R['ego_greedy_motion_a5']['mean_frag_detected']:.2f} "
             f"({red('ego_greedy_motion_a5'):+.1f}%)")
    L.append("")
    L.append("## Knobs stacked on the global frame")
    L.append(f"- global + Hungarian: {R['global_hungarian_static_a5']['mean_frag_detected']:.2f} "
             f"({red('global_hungarian_static_a5'):+.1f}%)")
    L.append(f"- global + motion: {R['global_greedy_motion_a5']['mean_frag_detected']:.2f} "
             f"({red('global_greedy_motion_a5'):+.1f}%)")
    L.append(f"- global + Hungarian + motion: {R['global_hungarian_motion_a5']['mean_frag_detected']:.2f} "
             f"({red('global_hungarian_motion_a5'):+.1f}%)")
    L.append(f"- **full stack** (global+Hung+motion+age10): "
             f"{R['global_hungarian_motion_a10']['mean_frag_detected']:.2f} "
             f"({red('global_hungarian_motion_a10'):+.1f}%)")
    L.append("")
    L.append("## Persistence (max_age) sweep on global/greedy/static")
    for nm in ["global_greedy_static_a2", "global_greedy_static_a5",
               "global_greedy_static_a10", "global_greedy_static_a20"]:
        L.append(f"- max_age={R[nm]['max_age']:>2}: frag {R[nm]['mean_frag_detected']:.2f} "
                 f"({red(nm):+.1f}%), cov_frames {R[nm]['mean_covered_frames_detected']:.2f}")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[abl] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    R = p["variants"]
    print("\n=== associator ablation (mean frags / detected GT) ===")
    base = R["BASELINE_ego_greedy_static_a5"]["mean_frag_detected"]
    print(f"  GT lifetime mean = {p['config']['gt_lifetime_mean']:.2f}")
    for name in p["variants"]:
        r = R[name]
        print(f"  {name:34s} {r['mean_frag_detected']:6.2f}  "
              f"Δ={r['frag_reduction_vs_baseline_pct']:+6.1f}%  cov={r['mean_covered_frames_detected']:.2f}")


if __name__ == "__main__":
    main()
