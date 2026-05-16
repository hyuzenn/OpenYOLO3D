"""Task 1.4b — 12-way streaming ablation runner.

Loops over a configurable list of method ids (baseline, single-axis M*,
phase1 / phase2 compounds, mix combos), runs the streaming evaluator on
every ScanNet200 val scene using the *shared Mask3D cache* from Task
1.2c Option G, and aggregates AP + temporal metrics per axis.

This file is the entry point; the actual qsub schedule (single 24 h PBS
or split per-axis PBS) is decided after the Option G sanity has passed
and the user picks the schedule.
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
from method_scannet.streaming.hooks_streaming import (
    install_method_streaming,
    list_method_ids,
    uninstall_all_streaming,
)
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from utils import OpenYolo3D
from utils.utils_2d import load_yaml


# Default 12-way ablation matrix. Compounds can be added/removed at the CLI.
DEFAULT_AXES: tuple[tuple[str, dict], ...] = (
    ("baseline", {}),
    ("M11", {"N": 3}),
    ("M12", {"prior": 0.5, "detection_likelihood": 0.8, "threshold": 0.95}),
    ("M21", {}),
    ("M22", {}),
    ("M31", {}),
    ("M31_iou07", {}),  # variant on M31 — installed manually below
    ("M32", {}),
    ("phase1", {}),     # M11 + M21 + M31
    ("phase2", {}),     # M12 + M22 + M32
    ("M21+M31", {}),    # label + merge (no registration)
    ("M22+M32", {}),    # Phase 2 label + merge (no registration)
)


def _to_py(x):
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


def _dump_aggregate_metrics(out_dir: Path, avgs, ar_avgs, rc_avgs, pcdc_avgs) -> None:
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
        }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))


def run_one_axis(
    name: str,
    method_id: str,
    method_kwargs: dict,
    oy3d: OpenYolo3D,
    cfg: dict,
    cache_dir: Path,
    out_root: Path,
    scenes: list[str],
) -> dict:
    out_dir = out_root / f"axis_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    preds_full: dict[str, dict] = {}

    print(f"\n[axis {name}] method_id={method_id} kwargs={method_kwargs}", flush=True)
    t_axis = time.time()

    for s_idx, scene_name in enumerate(scenes):
        scene_dir = Path("data/scannet200") / scene_name
        cache_path = cache_dir / f"{scene_name}.pt"
        if not cache_path.exists():
            print(f"  skip {scene_name} (no cache)", flush=True)
            continue

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
        evaluator.setup_scene(mask3d_cache_path=str(cache_path))

        uninstall_all_streaming(evaluator)
        if method_id != "baseline":
            install_method_streaming(evaluator, method_id, **method_kwargs)

        for fi in evaluator.frame_indices:
            evaluator.step_frame(fi)
        preds = evaluator.compute_method_predictions()
        preds_full[scene_name] = {
            "pred_masks": preds["pred_masks"],
            "pred_classes": preds["pred_classes"],
            "pred_scores": np.ones_like(preds["pred_scores"]),
        }

        if (s_idx + 1) % 25 == 0 or s_idx == len(scenes) - 1:
            elapsed = time.time() - t_axis
            rate = (s_idx + 1) / max(elapsed, 1e-6)
            eta = (len(scenes) - s_idx - 1) / max(rate, 1e-6)
            print(f"  [{s_idx + 1}/{len(scenes)}] elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min",
                  flush=True)

    gt_dir = "data/scannet200/ground_truth"
    avgs, ar_avgs, rc_avgs, pcdc_avgs = evaluate_scannet200(
        preds_full, gt_dir, output_file="/tmp/_ablation_eval_unused.txt",
        dataset="scannet200", pretrained_on_scannet200=True,
    )
    _dump_aggregate_metrics(out_dir, avgs, ar_avgs, rc_avgs, pcdc_avgs)
    summary = {
        "axis": name,
        "method_id": method_id,
        "kwargs": method_kwargs,
        "AP": _to_py(avgs["all_ap"]),
        "AP_50": _to_py(avgs["all_ap_50%"]),
        "AP_25": _to_py(avgs["all_ap_25%"]),
        "head_AP": _to_py(avgs.get("head_ap")),
        "common_AP": _to_py(avgs.get("common_ap")),
        "tail_AP": _to_py(avgs.get("tail_ap")),
        "axis_walltime_seconds": time.time() - t_axis,
        "n_scenes": len(preds_full),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[axis {name}] AP={summary['AP']:.4f} AP_50={summary['AP_50']:.4f}",
          flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=str,
                        help="Task 1.2c Option G Mask3D cache directory")
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument(
        "--config", default="pretrained/config_scannet200.yaml", type=str
    )
    parser.add_argument(
        "--axes",
        nargs="*",
        default=None,
        help="Subset of axis names to run (default: all DEFAULT_AXES).",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only evaluate the first N scenes.")
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    cfg = load_yaml(args.config)

    axes = [(n, k) for n, k in DEFAULT_AXES if args.axes is None or n in args.axes]
    print(f"=== Streaming ablation (Task 1.4b) ===")
    print(f"axes  : {[n for n, _ in axes]}")
    print(f"cache : {cache_dir}")
    print(f"output: {out_root}")

    print("Constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)

    scenes = list(SCENE_NAMES_SCANNET200)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    summaries: list[dict] = []
    for name, kwargs in axes:
        method_id_map = {"M31_iou07": "M31"}
        kw = dict(kwargs)
        if name == "M31_iou07":
            kw.setdefault("iou_threshold", 0.7)
        method_id = method_id_map.get(name, name if name in list_method_ids() else name)
        try:
            summaries.append(run_one_axis(
                name, method_id, kw, oy3d, cfg, cache_dir, out_root, scenes
            ))
        except Exception as exc:
            print(f"[axis {name}] FAILED: {exc!r}", flush=True)
            summaries.append({"axis": name, "error": str(exc)})

    (out_root / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out_root / 'all_summaries.json'}")


if __name__ == "__main__":
    main()
