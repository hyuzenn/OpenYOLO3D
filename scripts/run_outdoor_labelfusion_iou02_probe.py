#!/usr/bin/env python
"""Grid-edge probe: extend Study 2's tau_iou axis below 0.3 (grid edge in
run_outdoor_labelfusion_grid v01) on the best allowlist A3 {bi,mo,cone,ped}.

v01 found tau_iou monotone 0.3<0.5<0.7, so 0.3 was best-of-tested but on the
grid edge -> possible headroom below. This runs ONLY the new iou=0.2 row (sc in
{0.1,0.3,0.5}) plus the in-grid anchor iou=0.3/sc=0.3 for a same-process,
same-eval comparison. GPU-free read-time overlay, no cache rebuild.

STOP rule: if iou=0.2 best mAP <= the iou=0.3/sc=0.3 anchor, the axis has
plateaued -> 0.3 stays the operating point, done. Promote only on a clear win.
"""
import argparse, json, time
from pathlib import Path

from method_scannet.streaming.nuscenes_native_evaluator import (
    NativeTemporalNuScenesEvaluator, _list_val_scenes)
from dataloaders.nuscenes_loader import NuScenesLoader
from scripts.run_outdoor_labelfusion_grid import (
    run_one, override_stats, pc_capmean, SCORED)

A3 = frozenset({"bicycle", "motorcycle", "traffic_cone", "pedestrian"})
# (tau_iou, tau_score): new edge row + the v01 anchor for comparison.
CELLS = [(0.2, 0.1), (0.2, 0.3), (0.2, 0.5), (0.3, 0.3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--scene-limit", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=a.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = _list_val_scenes(loader)
    if a.scene_limit and a.scene_limit > 0:
        scenes = scenes[: a.scene_limit]
    print(f"  val scenes used: {len(scenes)}", flush=True)

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None, cp_cache_dir=a.src,
        proposal_source="hybrid")

    # native anchor (overlay off) for a self-consistent d_native in this process.
    s_nat = run_one(ev, scenes, out / "A0_native", frozenset(), 0.5, 0.4)
    nat = s_nat["mAP"]
    print(f"\n  native mAP={nat:.4f}", flush=True)

    grid = {}
    print("\n##### iou-edge probe (allowlist=A3 {bi,mo,cone,ped}) #####", flush=True)
    for ti, ts in CELLS:
        s = run_one(ev, scenes, out / f"S2_A3_iou{ti}_sc{ts}", A3, ti, ts)
        n_ov, prec, f, t = override_stats(s)
        pc = pc_capmean(s)
        grid[(ti, ts)] = {"mAP": s["mAP"], "bicycle": pc["bicycle"],
                          "motorcycle": pc["motorcycle"],
                          "traffic_cone": pc["traffic_cone"], "pedestrian": pc["pedestrian"],
                          "n_overrides": n_ov, "ov_precision": prec, "per_class": pc}
        print(f"  iou={ti} sc={ts}: mAP={s['mAP']:.4f} d_native={s['mAP']-nat:+.4f} "
              f"bike={pc['bicycle']:.4f} moto={pc['motorcycle']:.4f} "
              f"cone={pc['traffic_cone']:.4f} ovr={n_ov} prec={prec:.3f}", flush=True)

    anchor = grid[(0.3, 0.3)]["mAP"]
    edge = [(c, g["mAP"]) for c, g in grid.items() if c[0] == 0.2]
    best_edge_cell, best_edge = max(edge, key=lambda x: x[1])
    verdict = ("PROMOTE iou=0.2" if best_edge > anchor + 1e-4 else
               "STOP: plateau, keep iou=0.3")
    print(f"\n  anchor iou0.3/sc0.3 mAP={anchor:.4f} | "
          f"best edge {best_edge_cell} mAP={best_edge:.4f} "
          f"({best_edge-anchor:+.4f}) -> {verdict}", flush=True)

    (out / "probe_summary.json").write_text(json.dumps({
        "native_mAP": nat, "anchor_iou0.3_sc0.3": anchor,
        "cells": {f"iou{ti}_sc{ts}": v for (ti, ts), v in grid.items()},
        "verdict": verdict}, indent=2))
    print(f"\n=== DONE {time.strftime('%F %T')} ===", flush=True)


if __name__ == "__main__":
    main()
