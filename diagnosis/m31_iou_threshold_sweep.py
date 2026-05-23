"""M31 IoU-threshold sweep on the indoor (ScanNet200) streaming pipeline.

Keeps the architecturally-correct once-per-scene finalize merge (3D masks are
scene-constant), and measures how the IoUMerger threshold trades merge count
against instance-segmentation AP. For each scene the per-frame streaming loop
runs ONCE (YOLO per frame); the label-assigned pre-merge prediction is then
merged at each threshold (cheap, no re-inference) and scored.

Per threshold: total merges (Σ K_pre − K_post) and ScanNet200 AP / AP50 / AP25
over the evaluated scenes. A 'no_merge' row is included as the pre-merge anchor.

Usage:
  python -m diagnosis.m31_iou_threshold_sweep --n-scenes 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "2026-05-13_mask3d_cache"
DEFAULT_CONFIG = PROJECT_ROOT / "pretrained" / "config_scannet200.yaml"
OUT_DIR = PROJECT_ROOT / "results" / "indoor_module_invocation"
GT_DIR = str(PROJECT_ROOT / "data" / "scannet200" / "ground_truth")
THRESHOLDS = [0.5, 0.4, 0.3, 0.25]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenes", type=int, default=5)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    from evaluate import SCENE_NAMES_SCANNET200, evaluate_scannet200
    from method_scannet.method_31_iou_merging import IoUMerger
    from method_scannet.streaming import method_adapters as _ma
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    scenes = [s for s in SCENE_NAMES_SCANNET200
              if (cache_dir / f"{s}.pt").exists()
              and (PROJECT_ROOT / "data" / "scannet200" / s).is_dir()][: args.n_scenes]
    if not scenes:
        raise SystemExit("no usable scenes (cache + data) found")

    print(f"=== M31 IoU-threshold sweep ===", flush=True)
    print(f"n_scenes={len(scenes)} thresholds={args.thresholds} scenes={scenes}", flush=True)
    cfg = load_yaml(args.config)
    print("constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)

    labels = ["no_merge"] + [f"iou_{t}" for t in args.thresholds]
    preds_by_label = {lab: {} for lab in labels}
    merges_by_label = {lab: 0 for lab in labels}
    kin_by_label = {lab: 0 for lab in labels}
    kout_by_label = {lab: 0 for lab in labels}

    for i, sc in enumerate(scenes):
        print(f"[{i+1}/{len(scenes)}] {sc} ...", flush=True)
        scene_dir = PROJECT_ROOT / "data" / "scannet200" / sc
        ev = StreamingScanNetEvaluator(
            openyolo3d_instance=oy3d,
            scene_dir=str(scene_dir),
            depth_scale=cfg["openyolo3d"]["depth_scale"],
            depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
            num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
            topk=int(cfg["openyolo3d"].get("topk", 40)),
            topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
        )
        frequency = int(cfg["openyolo3d"].get("frequency", 10))
        ev.frame_indices = [f for f in ev.frame_indices if f % frequency == 0]
        ev.setup_scene(mask3d_cache_path=str(cache_dir / f"{sc}.pt"))
        for fi in ev.frame_indices:
            ev.step_frame(fi)

        # Pre-merge, label-assigned prediction (baseline labels; real scores
        # for NMS ranking). Merge uses real scores; AP uses ones (pipeline
        # convention in eval_streaming_ablation).
        pre = ev.compute_baseline_predictions()
        verts = ev.scene_vertices

        kpre = int(pre["pred_classes"].shape[0])
        preds_by_label["no_merge"][sc] = {
            "pred_masks": pre["pred_masks"],
            "pred_classes": pre["pred_classes"],
            "pred_scores": np.ones_like(pre["pred_scores"]),
        }
        kin_by_label["no_merge"] += kpre
        kout_by_label["no_merge"] += kpre

        for t in args.thresholds:
            lab = f"iou_{t}"
            post = _ma.apply_method31_merge(
                {"pred_masks": pre["pred_masks"],
                 "pred_classes": pre["pred_classes"],
                 "pred_scores": pre["pred_scores"]},
                IoUMerger(iou_threshold=float(t), use_kdtree=True),
                verts,
            )
            kpost = int(post["pred_classes"].shape[0])
            merges_by_label[lab] += (kpre - kpost)
            kin_by_label[lab] += kpre
            kout_by_label[lab] += kpost
            preds_by_label[lab][sc] = {
                "pred_masks": post["pred_masks"],
                "pred_classes": post["pred_classes"],
                "pred_scores": np.ones_like(post["pred_scores"]),
            }
        print(f"    K_pre={kpre} merges@{args.thresholds}="
              f"{[merges_by_label[f'iou_{t}'] for t in args.thresholds]}", flush=True)

    # ---- AP per label ------------------------------------------------------
    rows = []
    for lab in labels:
        avgs, ar_avgs, rc_avgs, pcdc_avgs = evaluate_scannet200(
            preds_by_label[lab], GT_DIR, output_file="/tmp/_m31_sweep_eval.txt",
            dataset="scannet200", pretrained_on_scannet200=True,
        )
        row = {
            "label": lab,
            "merges_total": int(merges_by_label[lab]),
            "input_proposals_total": int(kin_by_label[lab]),
            "output_proposals_total": int(kout_by_label[lab]),
            "AP": float(avgs["all_ap"]),
            "AP_50": float(avgs["all_ap_50%"]),
            "AP_25": float(avgs["all_ap_25%"]),
        }
        rows.append(row)
        print(f"  [{lab}] merges={row['merges_total']} "
              f"(in {row['input_proposals_total']} -> out {row['output_proposals_total']}) "
              f"AP={row['AP']:.5f} AP50={row['AP_50']:.5f} AP25={row['AP_25']:.5f}", flush=True)

    report = {"n_scenes": len(scenes), "scenes": scenes,
              "thresholds": args.thresholds, "rows": rows}
    out_path = OUT_DIR / "m31_iou_sweep.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
