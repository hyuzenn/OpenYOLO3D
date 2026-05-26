"""Before/after ablation for the M22 (margin gate + per-frame L2-norm) and
M32 (distance 2.0->0.5 m, semantic 0.3->0.95) fixes — 5 ScanNet200 scenes.

Per config: ScanNet200 AP / AP50 (instance seg) and lsc (label-switch count,
from the per-frame pred_history). M22/M22+M32 exercise the per-frame label path
(lsc-sensitive); M32-only is finalize-only (lsc == baseline by construction).

Usage:
  python -m diagnosis.m22_m32_fix_ablation --n-scenes 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "2026-05-13_mask3d_cache"
DEFAULT_CONFIG = PROJECT_ROOT / "pretrained" / "config_scannet200.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "m22_m32_fix"
GT_DIR = str(PROJECT_ROOT / "data" / "scannet200" / "ground_truth")

# (name, has_m22, m22_kwargs, has_m32, m32_kwargs)
CONFIGS = [
    ("baseline",        False, {}, False, {}),
    ("M22_before",      True,  {"normalize_per_frame": False, "margin": 0.0},   False, {}),
    ("M22_after",       True,  {"normalize_per_frame": True,  "margin": 0.006}, False, {}),
    ("M32_before",      False, {}, True,  {"distance_threshold": 2.0, "semantic_threshold": 0.3}),
    ("M32_after",       False, {}, True,  {"distance_threshold": 0.5, "semantic_threshold": 0.95}),
    ("M22+M32_before",  True,  {"normalize_per_frame": False, "margin": 0.0},   True,  {"distance_threshold": 2.0, "semantic_threshold": 0.3}),
    ("M22+M32_after",   True,  {"normalize_per_frame": True,  "margin": 0.006}, True,  {"distance_threshold": 0.5, "semantic_threshold": 0.95}),
]


def run_config(oy3d, cfg, cache_dir, scenes, has_m22, m22_kw, has_m32, m32_kw,
               idsw_iou=0.5):
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
    from method_scannet.streaming.hooks_streaming import (
        install_method_22, install_method_32, uninstall_all_streaming)
    from method_scannet.streaming.metrics import (
        label_switch_count, time_to_confirm, id_switch_count)
    from method_scannet.streaming import gt_matching as _gtm

    preds_full = {}
    lsc_total = 0
    ttc_values = []          # flat list of per-instance TTC across all scenes
    idsw_total = 0           # Σ ID switches over GT instances, all scenes
    n_gt_total = 0           # Σ GT instances, all scenes (ID Sw / Obj denom)
    for sc in scenes:
        scene_dir = PROJECT_ROOT / "data" / "scannet200" / sc
        ev = StreamingScanNetEvaluator(
            openyolo3d_instance=oy3d, scene_dir=str(scene_dir),
            depth_scale=cfg["openyolo3d"]["depth_scale"],
            depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
            num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
            topk=int(cfg["openyolo3d"].get("topk", 40)),
            topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
        )
        frequency = int(cfg["openyolo3d"].get("frequency", 10))
        ev.frame_indices = [f for f in ev.frame_indices if f % frequency == 0]
        ev.setup_scene(mask3d_cache_path=str(cache_dir / f"{sc}.pt"))
        uninstall_all_streaming(ev)
        if has_m22:
            install_method_22(ev, **m22_kw)
        if has_m32:
            install_method_32(ev, **m32_kw)
        for fi in ev.frame_indices:
            ev.step_frame(fi)
        preds = ev.compute_method_predictions()
        preds_full[sc] = {
            "pred_masks": preds["pred_masks"],
            "pred_classes": preds["pred_classes"],
            "pred_scores": np.ones_like(preds["pred_scores"]),
        }
        pred_history = list(ev.pred_history)
        lsc_total += int(label_switch_count(pred_history))
        # TTC (K=3) — same metric the main ablation runner records.
        ttc_values.extend(time_to_confirm(pred_history, K=3).values())
        # ID switches per GT instance (Task 3.2).
        gt_txt = PROJECT_ROOT / "data" / "scannet200" / "ground_truth" / f"{sc}.txt"
        try:
            gmatch, n_gt = _gtm.gt_matching_for_scene(
                pred_history, ev.instance_vertex_masks, str(gt_txt),
                iou_threshold=idsw_iou)
            idsw_total += int(id_switch_count(pred_history, gmatch))
            n_gt_total += int(n_gt)
        except Exception as exc:
            print(f"  [id_switch] {sc} failed: {exc!r}", flush=True)
    temporal = {
        "ttc_n_instances": len(ttc_values),
        "ttc_mean": float(np.mean(ttc_values)) if ttc_values else None,
        "ttc_median": float(np.median(ttc_values)) if ttc_values else None,
        "id_switch_total": idsw_total,
        "n_gt_instances": n_gt_total,
        "id_sw_per_obj": (idsw_total / n_gt_total) if n_gt_total else None,
        "idsw_iou_threshold": idsw_iou,
    }
    return preds_full, lsc_total, temporal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=5,
                    help="number of scenes; <=0 uses ALL available (312).")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--configs", nargs="+", default=None,
                    help="subset of config names to run (default: all).")
    ap.add_argument("--tag", default="m22_m32_fix_ablation",
                    help="output JSON basename (results/m22_m32_fix/<tag>.json).")
    ap.add_argument("--idsw-iou", type=float, default=0.5,
                    help="Full-scene IoU threshold for GT↔pred matching in "
                         "the id_switch_count metric (default 0.5).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    from evaluate import SCENE_NAMES_SCANNET200, evaluate_scannet200
    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    all_scenes = [s for s in SCENE_NAMES_SCANNET200
                  if (cache_dir / f"{s}.pt").exists()
                  and (PROJECT_ROOT / "data" / "scannet200" / s).is_dir()]
    scenes = all_scenes if args.n_scenes <= 0 else all_scenes[: args.n_scenes]
    if not scenes:
        raise SystemExit("no usable scenes")

    selected = set(args.configs) if args.configs else None
    configs = [c for c in CONFIGS if selected is None or c[0] in selected]
    if not configs:
        raise SystemExit(f"no configs match {args.configs}")

    print(f"=== M22/M32 fix before/after ablation, scenes={scenes} ===", flush=True)
    cfg = load_yaml(args.config)
    print("constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)

    print(f"  scenes={len(scenes)} configs={[c[0] for c in configs]}", flush=True)
    rows = []
    for name, has22, kw22, has32, kw32 in configs:
        print(f"\n[{name}] m22={kw22 if has22 else '-'} m32={kw32 if has32 else '-'}", flush=True)
        preds_full, lsc, temporal = run_config(
            oy3d, cfg, cache_dir, scenes, has22, kw22, has32, kw32,
            idsw_iou=args.idsw_iou)
        avgs, _ar, _rc, _pc = evaluate_scannet200(
            preds_full, GT_DIR, output_file="/tmp/_m22m32_eval.txt",
            dataset="scannet200", pretrained_on_scannet200=True)
        n_pred = int(sum(v["pred_classes"].shape[0] for v in preds_full.values()))
        row = {"config": name, "AP": float(avgs["all_ap"]),
               "AP_50": float(avgs["all_ap_50%"]), "lsc_total": lsc, "n_pred": n_pred,
               **temporal}
        rows.append(row)
        print(f"[{name}] AP={row['AP']:.5f} AP50={row['AP_50']:.5f} lsc={lsc} "
              f"n_pred={n_pred} ttc_mean={temporal['ttc_mean']} "
              f"idsw={temporal['id_switch_total']} "
              f"id_sw_per_obj={temporal['id_sw_per_obj']}", flush=True)

    report = {"n_scenes": len(scenes), "scenes": scenes, "rows": rows}
    out_path = OUT_DIR / f"{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2))

    by = {r["config"]: r for r in rows}
    print("\n==================== SUMMARY ====================", flush=True)
    print(f"{'config':16s}{'AP':>9s}{'AP50':>9s}{'lsc':>9s}{'n_pred':>9s}", flush=True)
    for r in rows:
        print(f"{r['config']:16s}{r['AP']:9.5f}{r['AP_50']:9.5f}{r['lsc_total']:9d}{r['n_pred']:9d}", flush=True)
    for a in ("M22", "M32", "M22+M32"):
        b, af = by.get(f"{a}_before"), by.get(f"{a}_after")
        if b and af:
            print(f"  {a}: ΔAP={af['AP']-b['AP']:+.5f}  Δlsc={af['lsc_total']-b['lsc_total']:+d}", flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
