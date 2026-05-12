"""Single-scene streaming sanity check (Task 1.2b).

Runs the streaming pipeline + offline OpenYOLO3D on the same scene, sends
both through the ScanNet200 evaluator, and reports whether their APs
agree within the sanity threshold (default ±0.005).

Usage:
    python -m method_scannet.streaming.run_streaming_scene \
        --scene scene0011_00 \
        --output results/2026-05-12_streaming_scene0011_00_v01/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from method_scannet.streaming.metrics import evaluate_scene_scannet200
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from utils import OpenYolo3D
from utils.utils_2d import load_yaml


def _to_serializable(obj):
    """Recursive JSON sanitizer for numpy / torch scalars."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_to_serializable(obj), indent=2))


def run(
    scene_name: str,
    output_dir: str,
    config_path: str = "pretrained/config_scannet200.yaml",
    sanity_threshold: float = 0.005,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("a")

    def log_print(*args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    cfg = load_yaml(config_path)
    depth_scale = cfg["openyolo3d"]["depth_scale"]
    num_classes = len(cfg["network2d"]["text_prompts"]) + 1
    topk = int(cfg["openyolo3d"].get("topk", 40))
    topk_per_image = int(cfg["openyolo3d"].get("topk_per_image", 600))
    depth_threshold = float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05))

    scene_dir = Path("data/scannet200") / scene_name
    gt_dir = "data/scannet200/ground_truth"

    log_print(f"=== Task 1.2b streaming sanity for {scene_name} ===")
    log_print(f"config       : {config_path}")
    log_print(f"scene_dir    : {scene_dir}")
    log_print(f"output_dir   : {out}")
    log_print(f"num_classes  : {num_classes}")
    log_print(f"topk         : {topk}")
    log_print(f"topk_per_img : {topk_per_image}")
    log_print(f"depth_th     : {depth_threshold} m")

    # ------------------------------------------------------------------
    # Construct OpenYOLO3D (loads YOLO-World + Mask3D weights; ~1 min)
    # ------------------------------------------------------------------
    log_print("\n[1/4] Constructing OpenYolo3D ...")
    t0 = time.time()
    oy3d = OpenYolo3D(config_path)
    log_print(f"  OpenYolo3D ready in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # OFFLINE reference (single-scene Mask3D + YOLO + MVPDist)
    # ------------------------------------------------------------------
    log_print("\n[2/4] Running OFFLINE OpenYolo3D.predict on scene ...")
    t0 = time.time()
    offline_pred = oy3d.predict(
        path_2_scene_data=str(scene_dir),
        depth_scale=depth_scale,
        datatype="mesh",
        processed_scene=str(scene_dir / f"{scene_name.replace('scene', '')}.npy"),
    )
    offline_pred = offline_pred[scene_name]
    offline_for_eval = {
        "pred_masks": offline_pred[0].cpu().numpy(),
        "pred_classes": offline_pred[1].cpu().numpy(),
        "pred_scores": torch.ones_like(offline_pred[2]).cpu().numpy(),
    }
    offline_time = time.time() - t0
    log_print(f"  offline done in {offline_time:.1f}s")
    log_print(
        f"  offline preds: {offline_for_eval['pred_masks'].shape[1]} (V, K_pred)"
    )

    # ------------------------------------------------------------------
    # STREAMING run on the same scene
    # ------------------------------------------------------------------
    log_print("\n[3/4] Running STREAMING evaluator on scene ...")
    t0 = time.time()
    evaluator = StreamingScanNetEvaluator(
        openyolo3d_instance=oy3d,
        scene_dir=str(scene_dir),
        depth_scale=depth_scale,
        depth_threshold=depth_threshold,
        num_classes=num_classes,
        topk=topk,
        topk_per_image=topk_per_image,
    )

    # Subsample frames by the same ``frequency`` knob as offline. The
    # streaming wrapper's frame_indices come from disk listing — restrict
    # to multiples of ``frequency`` so we cover the exact same frame set.
    frequency = int(cfg["openyolo3d"].get("frequency", 10))
    all_frames = evaluator.frame_indices
    streaming_frames = [f for f in all_frames if f % frequency == 0]
    evaluator.frame_indices = streaming_frames
    log_print(
        f"  scene has {len(all_frames)} raw frames; streaming uses "
        f"{len(streaming_frames)} (frequency={frequency})"
    )

    evaluator.setup_scene()
    log_print(
        f"  Mask3D produced {evaluator.instance_vertex_masks.shape[0]} instances"
    )

    visible_counts = []
    for t_idx, frame_id in enumerate(streaming_frames):
        snap = evaluator.step_frame(frame_id)
        visible_counts.append(int(len(snap["visible_instances"])))
        if (t_idx + 1) % 20 == 0 or t_idx == len(streaming_frames) - 1:
            log_print(
                f"  [streaming] frame {t_idx + 1}/{len(streaming_frames)} "
                f"(file={frame_id}.jpg, visible_instances={visible_counts[-1]})"
            )

    streaming_predictions = evaluator.compute_baseline_predictions()
    streaming_time = time.time() - t0
    log_print(f"  streaming done in {streaming_time:.1f}s")
    log_print(
        f"  streaming preds: {streaming_predictions['pred_masks'].shape[1]} (V, K_pred)"
    )

    # ------------------------------------------------------------------
    # ScanNet200 evaluation for both
    # ------------------------------------------------------------------
    log_print("\n[4/4] Evaluating both predictions with ScanNet200 evaluator ...")
    streaming_for_eval = {
        "pred_masks": streaming_predictions["pred_masks"],
        "pred_classes": streaming_predictions["pred_classes"],
        "pred_scores": np.ones_like(streaming_predictions["pred_scores"]),
    }

    t0 = time.time()
    streaming_metrics = evaluate_scene_scannet200(
        streaming_for_eval, gt_dir, scene_name
    )
    log_print(f"  streaming eval in {time.time() - t0:.1f}s")

    t0 = time.time()
    offline_metrics = evaluate_scene_scannet200(offline_for_eval, gt_dir, scene_name)
    log_print(f"  offline   eval in {time.time() - t0:.1f}s")

    delta_ap = streaming_metrics["AP"] - offline_metrics["AP"]
    sanity_pass = abs(delta_ap) <= sanity_threshold

    log_print("\n=== Result ===")
    log_print(f"offline    AP / AP_50 / AP_25 = "
              f"{offline_metrics['AP']:.4f} / {offline_metrics['AP_50']:.4f} "
              f"/ {offline_metrics['AP_25']:.4f}")
    log_print(f"streaming  AP / AP_50 / AP_25 = "
              f"{streaming_metrics['AP']:.4f} / {streaming_metrics['AP_50']:.4f} "
              f"/ {streaming_metrics['AP_25']:.4f}")
    log_print(f"Δ AP       = {delta_ap:+.4f}")
    log_print(f"sanity (±{sanity_threshold}): {'PASS' if sanity_pass else 'FAIL'}")
    log_print(f"D3 visible instances per frame: "
              f"min={min(visible_counts) if visible_counts else 0} "
              f"median={int(np.median(visible_counts)) if visible_counts else 0} "
              f"mean={float(np.mean(visible_counts)) if visible_counts else 0:.2f} "
              f"max={max(visible_counts) if visible_counts else 0}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    final_record = {
        "scene_name": scene_name,
        "config_path": config_path,
        "n_streaming_frames": len(streaming_frames),
        "n_offline_predictions": int(offline_for_eval["pred_masks"].shape[1]),
        "n_streaming_predictions": int(streaming_predictions["pred_masks"].shape[1]),
        "offline_metrics": offline_metrics,
        "streaming_metrics": streaming_metrics,
        "delta_AP": delta_ap,
        "sanity_threshold": sanity_threshold,
        "sanity_pass": sanity_pass,
        "offline_inference_seconds": offline_time,
        "streaming_inference_seconds": streaming_time,
        "d3_visible_per_frame": {
            "min": int(min(visible_counts)) if visible_counts else 0,
            "median": float(np.median(visible_counts)) if visible_counts else 0.0,
            "mean": float(np.mean(visible_counts)) if visible_counts else 0.0,
            "max": int(max(visible_counts)) if visible_counts else 0,
            "samples": visible_counts,
        },
    }
    _save_json(out / "final_metrics.json", final_record)
    log_print(f"\nfinal_metrics.json -> {out / 'final_metrics.json'}")

    log.close()
    return final_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--config", default="pretrained/config_scannet200.yaml", type=str
    )
    parser.add_argument("--sanity-threshold", default=0.005, type=float)
    args = parser.parse_args()
    run(
        scene_name=args.scene,
        output_dir=args.output,
        config_path=args.config,
        sanity_threshold=args.sanity_threshold,
    )


if __name__ == "__main__":
    main()
