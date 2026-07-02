"""Method variant: global (ego-motion-compensated) associator in the streaming pipeline.

Treats the global-frame tracker as a *production method variant*, not a probe:
the only change vs the baseline is the associator matching frame
(ego -> global, GlobalCentroidAssociator). For each (frame x axis) cell it runs
the real NativeTemporalNuScenesEvaluator over full-val and measures, in ONE
pass, all four quantities the ablation cares about:

  1. OV-TCS (A/B/C)        — per-track temporal-consistency score
  2. GT-anchored fragments — distinct track ids per GT instance
  3. track length          — detections per associator track
  4. mAP / NDS             — nuScenes devkit detection eval

Two axes isolate the mechanism:
  * baseline : raw associator, no temporal methods. The emitted detection boxes
    carry the native CenterPoint class + score and are computed per-proposal, so
    the track id never reaches the devkit -> mAP is associator-INVARIANT by
    construction. This is the clean test of "OV-TCS / fragmentation change while
    mAP does not."
  * phase1   : M11 frame-counting gate + M21 relabel + M31 merge. The M11 gate
    keeps/drops boxes by track age, so track structure DOES feed back into the
    emitted boxes -> mAP can move. This tests whether better continuity helps or
    hurts mAP once the temporal layer consumes the tracks.

EVIDENCE-ONLY w.r.t. the detector: no CenterPoint output is modified; only the
association frame and the (optional, default-off) temporal layer change.

Run (cache-only, no GPU):
  python -m method_scannet.streaming.eval_global_assoc_variant \
    --cp-cache-dir results/outdoor_native_temporal_cpcache_thr000_single_gravity \
    --output results/2026-06-12_ablation_global_associator_v01 \
    --axes baseline phase1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dataloaders.nuscenes_loader import NuScenesLoader
from method_scannet.streaming.nuscenes_native_evaluator import (
    NativeTemporalNuScenesEvaluator,
    _list_val_scenes,
)

FRAMES = ("ego", "global")


def _run_cell(ev: NativeTemporalNuScenesEvaluator, axis: str, scenes: list[str],
              out_dir: Path, baseline_map, *, m11_N, m12_threshold, m31_iou,
              m32_distance) -> dict:
    ev.install_axis(axis, m11_N=m11_N, m12_threshold=m12_threshold,
                    m31_iou=m31_iou, m32_distance=m32_distance)
    ev.begin_axis()
    t0 = time.time()
    for i, sc in enumerate(scenes):
        try:
            ev.run_scene(sc, scene_idx=i)
        except Exception as exc:
            print(f"    SCENE {sc[:8]} FAILED: {exc!r}", flush=True)
        if (i + 1) % 25 == 0 or (i + 1) == len(scenes):
            print(f"    [{ev.association_frame}/{axis}] {i+1}/{len(scenes)} "
                  f"tracks={len(ev._track_seq)} {time.time()-t0:.0f}s", flush=True)
    ev.last_axis_walltime_s = time.time() - t0
    return ev.aggregate_axis_metrics(out_dir, baseline_map)


def _fmt(x, nd=4):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def _table(cells: dict) -> str:
    """cells: {(frame, axis): summary}. Markdown ablation table."""
    rows = []
    hdr = ("| frame | axis | mAP | NDS | OV-TCS_A | OV-TCS_B | OV-TCS_C | "
           "track_len_mean | track_len_med | GT_frag_mean | lsc | n_tracks |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    rows += [hdr, sep]
    for frame in FRAMES:
        for axis in ("baseline", "phase1"):
            s = cells.get((frame, axis))
            if s is None:
                continue
            vm = s.get("variant_metrics", {})
            ov = vm.get("ov_tcs", {})
            tl = vm.get("track_length", {})
            gf = vm.get("gt_fragmentation", {})
            rows.append(
                f"| {frame} | {axis} | {_fmt(s['mAP'])} | {_fmt(s['NDS'])} | "
                f"{_fmt(ov.get('A_mean'))} | {_fmt(ov.get('B_mean'))} | "
                f"{_fmt(ov.get('C_mean'))} | {_fmt(tl.get('mean'),2)} | "
                f"{_fmt(tl.get('median'),2)} | {_fmt(gf.get('mean_fragments'),3)} | "
                f"{s['temporal']['label_switch_count_total']} | {vm.get('n_tracks')} |")
    return "\n".join(rows)


def _deltas(cells: dict) -> list[str]:
    """global - ego, per axis."""
    out = []
    for axis in ("baseline", "phase1"):
        e, g = cells.get(("ego", axis)), cells.get(("global", axis))
        if not e or not g:
            continue
        eve, gve = e.get("variant_metrics", {}), g.get("variant_metrics", {})

        def d(a, b):
            return None if (a is None or b is None) else b - a
        dmap = d(e["mAP"], g["mAP"])
        dnds = d(e["NDS"], g["NDS"])
        dA = d(eve["ov_tcs"]["A_mean"], gve["ov_tcs"]["A_mean"])
        dB = d(eve["ov_tcs"]["B_mean"], gve["ov_tcs"]["B_mean"])
        dC = d(eve["ov_tcs"]["C_mean"], gve["ov_tcs"]["C_mean"])
        dtl = d(eve["track_length"]["mean"], gve["track_length"]["mean"])
        dgf = d(eve["gt_fragmentation"]["mean_fragments"],
                gve["gt_fragmentation"]["mean_fragments"])
        out.append(
            f"[{axis}] Δ(global-ego): mAP={_fmt(dmap)} NDS={_fmt(dnds)} | "
            f"OV-TCS A={_fmt(dA)} B={_fmt(dB)} C={_fmt(dC)} | "
            f"track_len={_fmt(dtl,2)} GT_frag={_fmt(dgf,3)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--cp-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--axes", nargs="+", default=["baseline", "phase1"])
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--association-threshold-m", type=float, default=2.0)
    ap.add_argument("--association-max-age", type=int, default=5)
    ap.add_argument("--m11-N", type=int, default=3)
    ap.add_argument("--m12-threshold", type=float, default=0.85)
    ap.add_argument("--m31-iou", type=float, default=0.5)
    ap.add_argument("--m32-distance", type=float, default=0.5)
    args = ap.parse_args()

    out_root = Path(args.output)
    (out_root / "cells").mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = _list_val_scenes(loader)
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    print(f"  val scenes={len(scenes)} axes={args.axes}", flush=True)

    cells: dict = {}
    for frame in FRAMES:
        # Both frames run class-AGNOSTIC so the *only* knob that changes between
        # the ego baseline and the global variant is the matching frame. This
        # also reproduces outdoor_ovtcs_assoc_compare_probe (ego 0.301/0.260/
        # 0.136, global 0.263/0.241/0.168) as a correctness check. (The class
        # gate is irrelevant to the baseline-axis mAP anchor, which is
        # associator-invariant: native class + score reach the devkit per
        # proposal, never via the track id.)
        ev = NativeTemporalNuScenesEvaluator(
            loader=loader, cp_proposals=None,
            association_threshold_m=args.association_threshold_m,
            association_max_age=args.association_max_age,
            cp_cache_dir=args.cp_cache_dir,
            proposal_source="gamma",
            class_agnostic_association=True,
            association_frame=frame,
            collect_track_metrics=True)
        baseline_map = None
        for axis in args.axes:
            print(f"\n[{frame}/{axis}] running ...", flush=True)
            cell_dir = out_root / "cells" / f"{frame}_{axis}"
            summary = _run_cell(
                ev, axis, scenes, cell_dir, baseline_map,
                m11_N=args.m11_N, m12_threshold=args.m12_threshold,
                m31_iou=args.m31_iou, m32_distance=args.m32_distance)
            if axis == "baseline":
                baseline_map = summary.get("mAP")
            cells[(frame, axis)] = summary
            vm = summary.get("variant_metrics", {})
            print(f"[{frame}/{axis}] mAP={_fmt(summary['mAP'])} "
                  f"NDS={_fmt(summary['NDS'])} "
                  f"OV-TCS={_fmt(vm.get('ov_tcs',{}).get('A_mean'))}/"
                  f"{_fmt(vm.get('ov_tcs',{}).get('B_mean'))}/"
                  f"{_fmt(vm.get('ov_tcs',{}).get('C_mean'))} "
                  f"trk_len={_fmt(vm.get('track_length',{}).get('mean'),2)} "
                  f"GT_frag={_fmt(vm.get('gt_fragmentation',{}).get('mean_fragments'),3)} "
                  f"wall={summary['axis_walltime_s']:.0f}s", flush=True)

    table = _table(cells)
    deltas = _deltas(cells)
    combined = {f"{fr}_{ax}": s for (fr, ax), s in cells.items()}
    (out_root / "ablation.json").write_text(json.dumps(combined, indent=2))
    notes = ("# Global-associator method variant — ablation\n\n"
             "Single knob vs baseline: associator matching frame (ego -> global,\n"
             "ego-motion-compensated). OV-TCS / track-length / GT-fragmentation are\n"
             "measured on the pipeline's own tracks; mAP/NDS via nuScenes devkit.\n\n"
             "## Ablation table\n\n" + table + "\n\n## Δ (global - ego)\n\n"
             + "\n".join(deltas) + "\n")
    (out_root / "notes.md").write_text(notes)

    print("\n" + "=" * 78)
    print(table)
    print()
    for d in deltas:
        print(d)
    print(f"\nwrote {out_root / 'ablation.json'}\nwrote {out_root / 'notes.md'}",
          flush=True)


if __name__ == "__main__":
    main()
