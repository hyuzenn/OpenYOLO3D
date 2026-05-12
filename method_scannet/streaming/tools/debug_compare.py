"""Task 1.2c Option E — run offline + streaming on the SAME scene and dump
per-instance internals from both pipelines for direct comparison.

Outputs:
  results/<run_dir>/<scene>/offline_dump.json
  results/<run_dir>/<scene>/streaming_dump.json
  results/<run_dir>/<scene>/comparison.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from utils import OpenYolo3D
from utils.utils_2d import load_yaml


# ---------------------------------------------------------------------------
# Offline monkey-patch — capture per-instance state from
# OpenYolo3D.label_3d_masks_from_label_maps without modifying core code.
# ---------------------------------------------------------------------------


def install_offline_dump(target_dump: dict) -> None:
    """Wrap ``OpenYolo3D.label_3d_masks_from_label_maps`` to fill
    ``target_dump`` with per-instance label-distribution diagnostics.

    The original method is unchanged on disk; we replace the class
    attribute with a wrapper for the duration of the run.
    """
    original = OpenYolo3D.label_3d_masks_from_label_maps

    def wrapper(self, prediction_3d_masks, predictions_2d_bboxes,
                projections_mesh_to_frame, keep_visible_points, is_gt):
        from utils import get_visibility_mat

        # Re-run the visibility step to capture which frames are reps.
        topk_call = 25 if is_gt else self.openyolo3d_config["openyolo3d"]["topk"]
        visibility_matrix = get_visibility_mat(
            prediction_3d_masks.cuda().permute(1, 0),
            keep_visible_points.cuda(),
            topk=topk_call,
        ).cpu().numpy()  # (K, F)
        valid_frames = visibility_matrix.sum(axis=0) >= 1

        target_dump["topk"] = topk_call
        target_dump["n_instances"] = int(prediction_3d_masks.shape[1])
        target_dump["n_frames"] = int(prediction_3d_masks.shape[1] if False else visibility_matrix.shape[1])
        target_dump["n_valid_frames"] = int(valid_frames.sum())
        target_dump["visibility_matrix_repframes"] = [
            np.where(row)[0].tolist() for row in visibility_matrix
        ]
        target_dump["per_instance_total_visible"] = (
            (prediction_3d_masks.cpu().numpy().astype(bool).T
             & keep_visible_points.cpu().numpy().astype(bool).any(axis=0)[None, :]
             ).sum(axis=1).tolist()
        )

        # Run the original to get final preds.
        out = original(
            self, prediction_3d_masks, predictions_2d_bboxes,
            projections_mesh_to_frame, keep_visible_points, is_gt
        )
        masks_out, classes_out, scores_out = out
        target_dump["final_pred_classes"] = classes_out.cpu().tolist()
        target_dump["final_pred_scores"] = scores_out.cpu().tolist()
        target_dump["n_final_preds"] = int(masks_out.shape[1])
        return out

    OpenYolo3D.label_3d_masks_from_label_maps = wrapper
    return original


def uninstall_offline_dump(original) -> None:
    OpenYolo3D.label_3d_masks_from_label_maps = original


# ---------------------------------------------------------------------------
# Streaming dump — wrap BaselineLabelAccumulator.compute_predictions.
# ---------------------------------------------------------------------------


def install_streaming_dump(target_dump: dict):
    from method_scannet.streaming import baseline

    original = baseline.BaselineLabelAccumulator.compute_predictions

    def wrapper(self):
        from utils import get_visibility_mat

        topk_call = 25 if self.is_gt else self.topk
        projections, visible_masks, _label_maps = self._stack()
        pred_masks_KV = self.prediction_3d_masks.permute(1, 0).to(self.device)
        visible_masks_dev = visible_masks.to(self.device)
        visibility_matrix = get_visibility_mat(
            pred_masks_KV, visible_masks_dev, topk=topk_call
        ).cpu().numpy()
        valid_frames = visibility_matrix.sum(axis=0) >= 1

        target_dump["topk"] = topk_call
        target_dump["n_instances"] = int(self.n_instances)
        target_dump["n_frames"] = int(self.n_frames)
        target_dump["n_valid_frames"] = int(valid_frames.sum())
        target_dump["visibility_matrix_repframes"] = [
            np.where(row)[0].tolist() for row in visibility_matrix
        ]
        target_dump["per_instance_total_visible"] = (
            (self.prediction_3d_masks.cpu().numpy().astype(bool).T
             & visible_masks.cpu().numpy().astype(bool).any(axis=0)[None, :]
             ).sum(axis=1).tolist()
        )

        out = original(self)
        masks_out, classes_out, scores_out = out
        target_dump["final_pred_classes"] = classes_out.cpu().tolist()
        target_dump["final_pred_scores"] = scores_out.cpu().tolist()
        target_dump["n_final_preds"] = int(masks_out.shape[1])
        return out

    baseline.BaselineLabelAccumulator.compute_predictions = wrapper
    return original


def uninstall_streaming_dump(original):
    from method_scannet.streaming import baseline

    baseline.BaselineLabelAccumulator.compute_predictions = original


# ---------------------------------------------------------------------------
# Per-scene driver
# ---------------------------------------------------------------------------


def run_one_scene(scene_name: str, output_dir: Path, oy3d: OpenYolo3D, cfg: dict) -> dict:
    print(f"\n=== {scene_name} ===", flush=True)
    scene_dir = Path("data/scannet200") / scene_name
    scene_id = scene_name.replace("scene", "")
    npy_path = scene_dir / f"{scene_id}.npy"

    # ---- OFFLINE -------------------------------------------------------
    offline_dump: dict = {}
    orig_offline = install_offline_dump(offline_dump)
    try:
        t0 = time.time()
        oy3d.predict(
            path_2_scene_data=str(scene_dir),
            depth_scale=cfg["openyolo3d"]["depth_scale"],
            datatype="mesh",
            processed_scene=str(npy_path) if npy_path.exists() else None,
        )
        offline_dump["walltime_seconds"] = time.time() - t0
    finally:
        uninstall_offline_dump(orig_offline)
    print(f"  offline    n_inst={offline_dump['n_instances']}  "
          f"n_valid_frames={offline_dump['n_valid_frames']}  "
          f"n_final={offline_dump['n_final_preds']}  "
          f"wt={offline_dump['walltime_seconds']:.1f}s", flush=True)

    # ---- STREAMING -----------------------------------------------------
    streaming_dump: dict = {}
    orig_streaming = install_streaming_dump(streaming_dump)
    try:
        t0 = time.time()
        evaluator = StreamingScanNetEvaluator(
            openyolo3d_instance=oy3d,
            scene_dir=str(scene_dir),
            depth_scale=cfg["openyolo3d"]["depth_scale"],
            depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
            num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
            topk=int(cfg["openyolo3d"].get("topk", 40)),
            topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
        )
        frequency = int(cfg["openyolo3d"].get("frequency", 10))
        evaluator.frame_indices = [f for f in evaluator.frame_indices if f % frequency == 0]
        evaluator.setup_scene(
            processed_scene_path=str(npy_path) if npy_path.exists() else None,
            apply_mask3d_filter=True,
        )
        for fi in evaluator.frame_indices:
            evaluator.step_frame(fi)
        evaluator.compute_baseline_predictions()
        streaming_dump["walltime_seconds"] = time.time() - t0
    finally:
        uninstall_streaming_dump(orig_streaming)
    print(f"  streaming  n_inst={streaming_dump['n_instances']}  "
          f"n_valid_frames={streaming_dump['n_valid_frames']}  "
          f"n_final={streaming_dump['n_final_preds']}  "
          f"wt={streaming_dump['walltime_seconds']:.1f}s", flush=True)

    # ---- COMPARISON ----------------------------------------------------
    cmp = compare_dumps(offline_dump, streaming_dump)

    # ---- Persist -------------------------------------------------------
    scene_out = output_dir / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)
    (scene_out / "offline_dump.json").write_text(json.dumps(offline_dump, indent=2))
    (scene_out / "streaming_dump.json").write_text(json.dumps(streaming_dump, indent=2))
    (scene_out / "comparison.json").write_text(json.dumps(cmp, indent=2))

    print(f"  -> {scene_out}", flush=True)
    return {"scene_name": scene_name, "comparison_summary": cmp.get("summary", {})}


def compare_dumps(offline: dict, streaming: dict) -> dict:
    """Compute structural differences between offline / streaming dumps."""
    summary: dict = {}
    summary["n_instances_offline"] = offline.get("n_instances")
    summary["n_instances_streaming"] = streaming.get("n_instances")
    summary["n_instances_match"] = (
        offline.get("n_instances") == streaming.get("n_instances")
    )
    summary["n_valid_frames_offline"] = offline.get("n_valid_frames")
    summary["n_valid_frames_streaming"] = streaming.get("n_valid_frames")
    summary["n_final_preds_offline"] = offline.get("n_final_preds")
    summary["n_final_preds_streaming"] = streaming.get("n_final_preds")

    # Per-instance representative-frame set agreement (Jaccard).
    off_reps = offline.get("visibility_matrix_repframes", [])
    str_reps = streaming.get("visibility_matrix_repframes", [])
    if summary["n_instances_match"] and len(off_reps) == len(str_reps):
        jaccards = []
        diffs = []
        for i, (a, b) in enumerate(zip(off_reps, str_reps)):
            sa, sb = set(a), set(b)
            inter = len(sa & sb)
            union = len(sa | sb)
            j = inter / union if union > 0 else 1.0
            jaccards.append(j)
            if j < 0.95:
                diffs.append({
                    "instance_idx": i,
                    "n_off": len(sa), "n_str": len(sb),
                    "intersection": inter, "union": union,
                    "jaccard": j,
                    "offline_only": sorted(sa - sb)[:10],
                    "streaming_only": sorted(sb - sa)[:10],
                })
        summary["rep_frames_jaccard_mean"] = float(np.mean(jaccards)) if jaccards else None
        summary["rep_frames_jaccard_median"] = float(np.median(jaccards)) if jaccards else None
        summary["rep_frames_jaccard_min"] = float(np.min(jaccards)) if jaccards else None
        summary["instances_with_low_jaccard_lt_0_95"] = len(diffs)

    # Per-instance total-visible-vertex agreement
    off_vis = offline.get("per_instance_total_visible", [])
    str_vis = streaming.get("per_instance_total_visible", [])
    if summary["n_instances_match"] and len(off_vis) == len(str_vis):
        diffs = [int(a - b) for a, b in zip(off_vis, str_vis)]
        summary["total_visible_diff_mean"] = float(np.mean(diffs)) if diffs else 0.0
        summary["total_visible_diff_max_abs"] = int(max(abs(x) for x in diffs)) if diffs else 0
        summary["instances_with_visible_mismatch"] = int(
            sum(1 for x in diffs if x != 0)
        )

    # Final class agreement (only meaningful when K_inst match)
    off_cls = offline.get("final_pred_classes", [])
    str_cls = streaming.get("final_pred_classes", [])
    if len(off_cls) == len(str_cls) and len(off_cls) > 0:
        matches = sum(1 for a, b in zip(off_cls, str_cls) if a == b)
        summary["final_class_match_count"] = int(matches)
        summary["final_class_match_ratio"] = float(matches / len(off_cls))
    else:
        summary["final_class_match_count"] = None
        summary["final_class_match_ratio"] = None

    return {"summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", required=True, type=str)
    parser.add_argument(
        "--output", default="results/2026-05-13_streaming_debug_E", type=str
    )
    parser.add_argument(
        "--config", default="pretrained/config_scannet200.yaml", type=str
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    print("Constructing OpenYolo3D ...", flush=True)
    t0 = time.time()
    oy3d = OpenYolo3D(args.config)
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    summaries = []
    for scene_name in args.scenes:
        summaries.append(run_one_scene(scene_name, out, oy3d, cfg))

    (out / "scenes_run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out / 'scenes_run_summary.json'}")


if __name__ == "__main__":
    main()
