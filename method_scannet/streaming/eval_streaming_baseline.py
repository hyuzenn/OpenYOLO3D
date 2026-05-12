"""Task 1.2c — 312-scene streaming baseline evaluation.

Runs the Task 1.2b streaming wrapper on every ScanNet200 validation scene,
collects predictions, and feeds them to the same ``evaluate_scannet200``
helper that the offline reference (results/2026-05-07_scannet_eval_v01)
uses. The 312-scene mean averages out the per-scene Mask3D noise that
made the single-scene v02 sanity check fail by 0.0093.

Output layout matches the offline runner so the aggregate metrics.json
can be diffed against results/2026-05-07_scannet_eval_v01/metrics.json
directly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate import SCENE_NAMES_SCANNET200, evaluate_scannet200
from method_scannet.streaming.metrics import (
    id_switch_count,
    label_switch_count,
    time_to_confirm,
)
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from utils import OpenYolo3D
from utils.utils_2d import load_yaml


def _to_py(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return _to_py(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize(obj), indent=2))


def _dump_aggregate_metrics(out_dir: Path, avgs: dict, ar_avgs: dict, rc_avgs: dict, pcdc_avgs: dict) -> None:
    """Mirror run_evaluation._maybe_dump_metrics so the aggregate JSON has
    exactly the same schema as the offline reference run."""
    out = {
        "dataset": "scannet200",
        "metrics": {
            "average": {
                "AP": _to_py(avgs["all_ap"]),
                "AP_50": _to_py(avgs["all_ap_50%"]),
                "AP_25": _to_py(avgs["all_ap_25%"]),
                "AR": _to_py(ar_avgs["all_ar"]),
                "RC_50": _to_py(rc_avgs["all_rc_50%"]),
                "RC_25": _to_py(rc_avgs["all_rc_25%"]),
                "APCDC": _to_py(pcdc_avgs["all_pcdc"]),
                "PCDC_50": _to_py(pcdc_avgs["all_pcdc_50%"]),
                "PCDC_25": _to_py(pcdc_avgs["all_pcdc_25%"]),
            },
        },
    }
    for cat in ("head", "common", "tail"):
        out["metrics"][cat] = {
            "AP": _to_py(avgs.get(f"{cat}_ap")),
            "AP_50": _to_py(avgs.get(f"{cat}_ap50%")),
            "AP_25": _to_py(avgs.get(f"{cat}_ap25%")),
            "AR": _to_py(ar_avgs.get(f"{cat}_ar")),
            "RC_50": _to_py(rc_avgs.get(f"{cat}_rc50%")),
            "RC_25": _to_py(rc_avgs.get(f"{cat}_rc25%")),
        }
    _save_json(out_dir / "metrics.json", out)


def run(
    output_dir: str,
    config_path: str = "pretrained/config_scannet200.yaml",
    scene_names: list[str] | None = None,
    log_every: int = 10,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_scene_dir = out / "per_scene"
    per_scene_dir.mkdir(exist_ok=True)
    log_path = out / "run.log"
    log = log_path.open("a")

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
    frequency = int(cfg["openyolo3d"].get("frequency", 10))

    scenes = scene_names if scene_names is not None else list(SCENE_NAMES_SCANNET200)

    log_print(f"=== Task 1.2c streaming baseline — 312 scenes ===")
    log_print(f"config         : {config_path}")
    log_print(f"output         : {out}")
    log_print(f"num scenes     : {len(scenes)}")
    log_print(f"num_classes    : {num_classes}")
    log_print(f"topk / topk_pi : {topk} / {topk_per_image}")
    log_print(f"depth_th       : {depth_threshold}")
    log_print(f"frequency      : {frequency}")

    log_print("\n[init] Constructing OpenYolo3D (~1 min) ...")
    t0 = time.time()
    oy3d = OpenYolo3D(config_path)
    log_print(f"[init] OpenYolo3D ready in {time.time() - t0:.1f}s")

    preds_full: dict[str, dict] = {}
    per_scene_stats: dict[str, dict] = {}
    visibility_samples_all: list[int] = []
    walltime_per_scene: list[float] = []

    t_total = time.time()
    for s_idx, scene_name in enumerate(scenes):
        scene_dir = Path("data/scannet200") / scene_name
        scene_id = scene_name.replace("scene", "")
        processed_npy = scene_dir / f"{scene_id}.npy"
        processed_scene_path = str(processed_npy) if processed_npy.exists() else None

        t_scene = time.time()
        try:
            evaluator = StreamingScanNetEvaluator(
                openyolo3d_instance=oy3d,
                scene_dir=str(scene_dir),
                depth_scale=depth_scale,
                depth_threshold=depth_threshold,
                num_classes=num_classes,
                topk=topk,
                topk_per_image=topk_per_image,
            )

            # Subsample to match offline ``frequency`` knob.
            all_frames = evaluator.frame_indices
            evaluator.frame_indices = [f for f in all_frames if f % frequency == 0]

            # Mask3D filter + .npy input mirror ``OpenYolo3D.predict`` exactly.
            evaluator.setup_scene(
                processed_scene_path=processed_scene_path,
                apply_mask3d_filter=True,
            )

            scene_visible_counts: list[int] = []
            for frame_id in evaluator.frame_indices:
                snap = evaluator.step_frame(frame_id)
                scene_visible_counts.append(int(len(snap["visible_instances"])))

            preds = evaluator.compute_baseline_predictions()
            walltime = time.time() - t_scene
            walltime_per_scene.append(walltime)

            preds_full[scene_name] = {
                "pred_masks": preds["pred_masks"],
                "pred_classes": preds["pred_classes"],
                "pred_scores": np.ones_like(preds["pred_scores"]),
            }

            per_scene_stats[scene_name] = {
                "n_frames_streamed": len(evaluator.frame_indices),
                "n_raw_frames_in_scene": len(all_frames),
                "n_mask3d_instances_after_filter": int(
                    evaluator.instance_vertex_masks.shape[0]
                ),
                "n_final_predictions": int(preds["pred_masks"].shape[1]),
                "d3_visibility": {
                    "min": int(min(scene_visible_counts) if scene_visible_counts else 0),
                    "median": float(
                        np.median(scene_visible_counts) if scene_visible_counts else 0
                    ),
                    "mean": float(
                        np.mean(scene_visible_counts) if scene_visible_counts else 0
                    ),
                    "max": int(max(scene_visible_counts) if scene_visible_counts else 0),
                },
                "label_switch_count": label_switch_count(evaluator.pred_history),
                "time_to_confirm": time_to_confirm(evaluator.pred_history, K=3),
                "walltime_seconds": walltime,
            }
            visibility_samples_all.extend(scene_visible_counts)

            # Persist a lightweight per-scene snapshot (no large arrays).
            _save_json(per_scene_dir / f"{scene_name}.json", per_scene_stats[scene_name])

            if (s_idx + 1) % log_every == 0 or s_idx == len(scenes) - 1:
                elapsed = time.time() - t_total
                rate = (s_idx + 1) / max(elapsed, 1e-6)
                eta = (len(scenes) - s_idx - 1) / max(rate, 1e-6)
                log_print(
                    f"[{s_idx + 1}/{len(scenes)}] {scene_name} "
                    f"frames={len(evaluator.frame_indices)} "
                    f"K_mask3d={per_scene_stats[scene_name]['n_mask3d_instances_after_filter']} "
                    f"d3_med={per_scene_stats[scene_name]['d3_visibility']['median']:.0f} "
                    f"wt={walltime:.1f}s  "
                    f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min"
                )
        except Exception as exc:
            log_print(f"[{s_idx + 1}/{len(scenes)}] {scene_name} FAILED: {exc!r}")
            per_scene_stats[scene_name] = {"error": str(exc)}
            continue

    log_print(f"\n[streaming] all scenes processed in {(time.time() - t_total)/60:.1f} min")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    log_print("\n[eval] Running evaluate_scannet200 on aggregated predictions ...")
    gt_dir = "data/scannet200/ground_truth"
    eval_t = time.time()
    avgs, ar_avgs, rc_avgs, pcdc_avgs = evaluate_scannet200(
        preds_full,
        gt_dir,
        output_file="/tmp/streaming_baseline_eval_unused.txt",
        dataset="scannet200",
        pretrained_on_scannet200=True,
    )
    log_print(f"[eval] done in {(time.time() - eval_t):.1f}s")

    _dump_aggregate_metrics(out, avgs, ar_avgs, rc_avgs, pcdc_avgs)
    log_print(f"[eval] aggregate metrics.json -> {out / 'metrics.json'}")

    # ------------------------------------------------------------------
    # D3 visibility + temporal metric aggregates
    # ------------------------------------------------------------------
    if visibility_samples_all:
        vis_stats = {
            "frame_visible_instance_count": {
                "n_samples": len(visibility_samples_all),
                "min": int(min(visibility_samples_all)),
                "median": float(np.median(visibility_samples_all)),
                "mean": float(np.mean(visibility_samples_all)),
                "max": int(max(visibility_samples_all)),
                "p90": float(np.percentile(visibility_samples_all, 90)),
                "p99": float(np.percentile(visibility_samples_all, 99)),
            }
        }
    else:
        vis_stats = {}
    _save_json(out / "d3_visibility_stats.json", vis_stats)
    log_print(f"[viz] d3_visibility_stats.json written")

    label_switches = [
        v["label_switch_count"]
        for v in per_scene_stats.values()
        if isinstance(v, dict) and "label_switch_count" in v
    ]
    ttc_values: list[int] = []
    for v in per_scene_stats.values():
        if isinstance(v, dict) and isinstance(v.get("time_to_confirm"), dict):
            ttc_values.extend(v["time_to_confirm"].values())
    walltimes = [
        v["walltime_seconds"]
        for v in per_scene_stats.values()
        if isinstance(v, dict) and "walltime_seconds" in v
    ]
    temporal = {
        "label_switch_count": {
            "n_scenes": len(label_switches),
            "total": int(sum(label_switches)),
            "mean_per_scene": float(np.mean(label_switches)) if label_switches else 0.0,
            "median_per_scene": float(np.median(label_switches)) if label_switches else 0.0,
            "p90_per_scene": float(np.percentile(label_switches, 90)) if label_switches else 0.0,
        },
        "time_to_confirm": {
            "n_instances": len(ttc_values),
            "mean": float(np.mean(ttc_values)) if ttc_values else None,
            "median": float(np.median(ttc_values)) if ttc_values else None,
            "p90": float(np.percentile(ttc_values, 90)) if ttc_values else None,
        },
        "walltime_seconds": {
            "n_scenes": len(walltimes),
            "total_seconds": float(sum(walltimes)),
            "mean_per_scene": float(np.mean(walltimes)) if walltimes else 0.0,
            "median_per_scene": float(np.median(walltimes)) if walltimes else 0.0,
            "max_per_scene": float(max(walltimes)) if walltimes else 0.0,
        },
    }
    _save_json(out / "temporal_metrics.json", temporal)
    log_print(f"[temporal] temporal_metrics.json written")

    # ------------------------------------------------------------------
    # Sanity vs offline reference
    # ------------------------------------------------------------------
    offline_metrics_path = Path(
        "results/2026-05-07_scannet_eval_v01/metrics.json"
    )
    sanity = {"sanity_threshold": 0.005}
    if offline_metrics_path.exists():
        offline = json.loads(offline_metrics_path.read_text())["metrics"]["average"]
        streaming_avg = json.loads((out / "metrics.json").read_text())["metrics"]["average"]
        for key in ("AP", "AP_50", "AP_25"):
            sanity[f"offline_{key}"] = offline.get(key)
            sanity[f"streaming_{key}"] = streaming_avg.get(key)
            try:
                sanity[f"delta_{key}"] = streaming_avg[key] - offline[key]
            except Exception:
                sanity[f"delta_{key}"] = None
        try:
            sanity["sanity_pass"] = abs(sanity["delta_AP"]) <= sanity["sanity_threshold"]
        except Exception:
            sanity["sanity_pass"] = None
    _save_json(out / "sanity_check.json", sanity)
    log_print(f"[sanity] {json.dumps(sanity, indent=2)}")

    log.close()
    return sanity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--config", default="pretrained/config_scannet200.yaml", type=str
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: evaluate only the first N scenes (for smoke).",
    )
    args = parser.parse_args()

    scenes = list(SCENE_NAMES_SCANNET200)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    run(output_dir=args.output, config_path=args.config, scene_names=scenes)


if __name__ == "__main__":
    main()
