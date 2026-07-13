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
from method_scannet.streaming.metrics import (
    id_switch_count,
    label_switch_count,
    time_to_confirm,
)
from method_scannet.streaming import gt_matching as _gtm
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
    # OV-TCS-aware EMA ablation (2026-06-23): both arms are M22 with confidence
    # plumbed; baseline is plain confidence-weighted EMA, the scaling arm scales
    # the incoming weight by the causal online OV-TCS_C. record_feature_trace is
    # enabled on BOTH so the OVTCS<->drift / OVTCS<->stability correlation is
    # read off the baseline arm (where drift is independent of OVTCS).
    ("M22_base_weighted", {"conf_mode": "weighted", "record_feature_trace": True}),
    ("M22_ovtcs_scale", {"conf_mode": "ovtcs_scale", "record_feature_trace": True}),
    # Attribution control: matched-average step-shrink (w=α·c·k, k=mean OVTCS)
    # with no per-update OV-TCS variation. ovtcs_scale > const_scale ⇒ OV-TCS
    # carries real per-update signal; ≈ ⇒ the gain is just a smaller effective α.
    ("M22_const_scale", {"conf_mode": "const_scale", "const_k": 0.335,
                         "record_feature_trace": True}),
    # Pure EMA incoming-weight sweep (2026-06-23, H1: over-update hypothesis).
    # w = α·c·k with a FIXED k — NO OV-TCS. k=1.0 is byte-identical to the
    # w=α·c baseline (α,c∈[0,1] ⇒ clamp is a no-op). record_feature_trace
    # gives feature-drift + applied-update count via ovtcs_diagnostics(); the
    # online_ovtcs column is diagnostic only and never enters the EMA math.
    # const_scale never DROPS updates, so applied-count is k-invariant by
    # design — drift / effective-α (=α·k) is the real step-size signal.
    *(
        (f"M22_emak_{k:g}",
         {"conf_mode": "const_scale", "const_k": k, "record_feature_trace": True})
        for k in (1.0, 0.75, 0.50, 0.335, 0.25, 0.10)
    ),
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


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of a 1D array (ties averaged) — scipy-free Spearman helper."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average tied ranks
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _corr(x: list, y: list) -> dict:
    """Pearson + Spearman of two equal-length sequences (numpy only)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    n = int(xa.size)
    if n < 3 or np.std(xa) == 0 or np.std(ya) == 0:
        return {"n": n, "pearson": None, "spearman": None}
    pear = float(np.corrcoef(xa, ya)[0, 1])
    rx, ry = _rankdata(xa), _rankdata(ya)
    spear = float(np.corrcoef(rx, ry)[0, 1])
    return {"n": n, "pearson": pear, "spearman": spear}


def _aggregate_ovtcs(scenes_diag: list[dict], axis_name: str) -> dict:
    """Pool per-scene FeatureFusionEMA.ovtcs_diagnostics() into one axis summary
    (scalar means + pooled OVTCS<->drift / OVTCS<->cos_to_final correlations)."""
    valid = [d for d in scenes_diag if d.get("n_instances", 0) > 0]
    total_inst = sum(int(d["n_instances"]) for d in valid)
    total_applied = sum(int(d["n_updates_applied_total"]) for d in valid)
    total_obs = sum(int(d["n_observations_total"]) for d in valid)

    def _wmean(key: str) -> float | None:
        num = 0.0
        den = 0
        for d in valid:
            s = d.get(key) or {}
            if s.get("mean") is not None and s.get("n"):
                num += s["mean"] * s["n"]
                den += s["n"]
        return float(num / den) if den else None

    tr_o: list = []
    tr_d: list = []
    tr_c: list = []
    for d in valid:
        tr = d.get("trace")
        if tr:
            tr_o.extend(tr["ovtcs"])
            tr_d.extend(tr["drift"])
            tr_c.extend(tr["cos_to_final"])

    return {
        "axis": axis_name,
        "n_scenes": len(valid),
        "n_instances_total": total_inst,
        "updates_applied_total": total_applied,
        "observations_total": total_obs,
        "updates_applied_per_track": (total_applied / total_inst) if total_inst else None,
        "observations_per_track": (total_obs / total_inst) if total_inst else None,
        # applied/observed → relative update-rate reduction is computed vs the
        # baseline arm by the aggregator script.
        "applied_fraction": (total_applied / total_obs) if total_obs else None,
        "feature_drift_mean": _wmean("feature_drift"),
        "online_ovtcs_mean": _wmean("online_ovtcs"),
        "trace_n": len(tr_o),
        "corr_ovtcs_drift": _corr(tr_o, tr_d),
        "corr_ovtcs_cos_to_final": _corr(tr_o, tr_c),
    }


def run_one_axis(
    name: str,
    method_id: str,
    method_kwargs: dict,
    oy3d: OpenYolo3D,
    cfg: dict,
    cache_dir: Path,
    out_root: Path,
    scenes: list[str],
    idsw_iou: float = 0.5,
) -> dict:
    out_dir = out_root / f"axis_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    preds_full: dict[str, dict] = {}
    temporal_per_scene: dict[str, dict] = {}
    ovtcs_scenes: list[dict] = []  # per-scene FeatureFusionEMA.ovtcs_diagnostics()

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

        # OV-TCS diagnostics — harvest the scene's FeatureFusionEMA before the
        # evaluator (and its per-scene M22 state) is discarded next iteration.
        m22 = getattr(evaluator, "method_22", None)
        if m22 is not None and hasattr(m22, "ovtcs_diagnostics"):
            try:
                ovtcs_scenes.append({"scene": scene_name, **m22.ovtcs_diagnostics()})
            except Exception as exc:
                print(f"  [ovtcs] {scene_name} diag failed: {exc!r}", flush=True)

        # Task 1.3 — temporal metrics from the running labeler's history.
        pred_history = list(evaluator.pred_history)
        lsc = int(label_switch_count(pred_history))
        ttc = time_to_confirm(pred_history, K=3)

        # Task 3.2 — ID switches per GT instance. Build the per-frame GT↔pred
        # matching from this scene's history + fixed proposal/GT masks, then
        # run the (previously unused) metrics.id_switch_count over it.
        gt_txt = Path("data/scannet200/ground_truth") / f"{scene_name}.txt"
        try:
            gmatch, n_gt = _gtm.gt_matching_for_scene(
                pred_history,
                evaluator.instance_vertex_masks,
                str(gt_txt),
                iou_threshold=idsw_iou,
            )
            idsw = int(id_switch_count(pred_history, gmatch))
        except Exception as exc:  # never let id-switch sink the whole axis
            print(f"  [id_switch] {scene_name} failed: {exc!r}", flush=True)
            idsw, n_gt = 0, 0

        temporal_per_scene[scene_name] = {
            "n_frames": len(pred_history),
            "n_unique_instances": len(set().union(*[h.keys() for h in pred_history])) if pred_history else 0,
            "label_switch_count": lsc,
            "time_to_confirm": dict(ttc),  # {iid: frames_to_confirm}
            "id_switch_count": idsw,
            "n_gt_instances": n_gt,
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

    # ---- Task 1.3 temporal-metric aggregation -------------------------
    lsc_values = [v["label_switch_count"] for v in temporal_per_scene.values()]
    ttc_values: list[int] = []
    for v in temporal_per_scene.values():
        ttc_values.extend(v["time_to_confirm"].values())
    n_inst_total = int(sum(v["n_unique_instances"] for v in temporal_per_scene.values()))
    idsw_total = int(sum(v.get("id_switch_count", 0) for v in temporal_per_scene.values()))
    n_gt_total = int(sum(v.get("n_gt_instances", 0) for v in temporal_per_scene.values()))
    temporal_axis = {
        "axis": name,
        "n_scenes": len(temporal_per_scene),
        "n_unique_instances_total": n_inst_total,
        "id_switch": {
            "total": idsw_total,
            "n_gt_instances": n_gt_total,
            # primary reported metric: ID switches per GT object.
            "per_obj": (idsw_total / n_gt_total) if n_gt_total else None,
            "iou_threshold": idsw_iou,
        },
        "label_switch_count": {
            "total": int(sum(lsc_values)),
            "mean_per_scene": float(np.mean(lsc_values)) if lsc_values else 0.0,
            "median_per_scene": float(np.median(lsc_values)) if lsc_values else 0.0,
            "p90_per_scene": float(np.percentile(lsc_values, 90)) if lsc_values else 0.0,
            "max_per_scene": int(max(lsc_values)) if lsc_values else 0,
        },
        "time_to_confirm": {
            "n_instances": len(ttc_values),
            "mean": float(np.mean(ttc_values)) if ttc_values else None,
            "median": float(np.median(ttc_values)) if ttc_values else None,
            "p90": float(np.percentile(ttc_values, 90)) if ttc_values else None,
            "max": int(max(ttc_values)) if ttc_values else None,
        },
    }
    (out_dir / "temporal_metrics.json").write_text(json.dumps(temporal_axis, indent=2))
    (out_dir / "temporal_per_scene.json").write_text(
        json.dumps(
            {
                s: {
                    "n_frames": v["n_frames"],
                    "n_unique_instances": v["n_unique_instances"],
                    "label_switch_count": v["label_switch_count"],
                    "time_to_confirm_n_instances": len(v["time_to_confirm"]),
                    "time_to_confirm_mean": (
                        float(np.mean(list(v["time_to_confirm"].values())))
                        if v["time_to_confirm"] else None
                    ),
                    "id_switch_count": v.get("id_switch_count", 0),
                    "n_gt_instances": v.get("n_gt_instances", 0),
                }
                for s, v in temporal_per_scene.items()
            },
            indent=2,
        )
    )

    # ---- OV-TCS diagnostics aggregation (only when an M22-OVTCS arm ran) ----
    if any(d.get("n_instances", 0) > 0 for d in ovtcs_scenes):
        ovtcs_axis = _aggregate_ovtcs(ovtcs_scenes, name)
        (out_dir / "ovtcs_diagnostics.json").write_text(json.dumps(ovtcs_axis, indent=2))
        cdr = ovtcs_axis["corr_ovtcs_drift"]
        ccf = ovtcs_axis["corr_ovtcs_cos_to_final"]
        print(f"[axis {name}] ovtcs: upd/track={ovtcs_axis['updates_applied_per_track']} "
              f"drift_mean={ovtcs_axis['feature_drift_mean']} "
              f"ovtcs_mean={ovtcs_axis['online_ovtcs_mean']} "
              f"corr(ovtcs,drift)={cdr['pearson']}/{cdr['spearman']} "
              f"corr(ovtcs,cos2final)={ccf['pearson']}/{ccf['spearman']}", flush=True)

    print(f"[axis {name}] AP={summary['AP']:.4f} AP_50={summary['AP_50']:.4f}  "
          f"lsc_total={temporal_axis['label_switch_count']['total']} "
          f"ttc_n={temporal_axis['time_to_confirm']['n_instances']} "
          f"ttc_mean={temporal_axis['time_to_confirm']['mean']} "
          f"idsw_total={temporal_axis['id_switch']['total']} "
          f"idsw_per_obj={temporal_axis['id_switch']['per_obj']}",
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
    parser.add_argument("--idsw-iou", type=float, default=0.5,
                        help="Full-scene IoU threshold for GT↔pred matching "
                             "in the id_switch_count metric (default 0.5).")
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=None,
        help="Optional explicit scene names (overrides --limit). "
             "E.g. --scenes scene0011_00",
    )
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

    if args.scenes:
        scenes = list(args.scenes)
    else:
        scenes = list(SCENE_NAMES_SCANNET200)
        if args.limit is not None:
            scenes = scenes[: args.limit]

    summaries: list[dict] = []
    for name, kwargs in axes:
        method_id_map = {
            "M31_iou07": "M31",
            "M22_base_weighted": "M22",
            "M22_ovtcs_scale": "M22",
            "M22_const_scale": "M22",
        }
        kw = dict(kwargs)
        if name == "M31_iou07":
            kw.setdefault("iou_threshold", 0.7)
        # All M22_* variants (weighted/ovtcs/const/emak sweep) install plain M22
        # and differ only by kwargs; anything else falls back to its own name.
        method_id = method_id_map.get(name) or (
            "M22" if name.startswith("M22_") else name
        )
        try:
            summaries.append(run_one_axis(
                name, method_id, kw, oy3d, cfg, cache_dir, out_root, scenes,
                idsw_iou=args.idsw_iou,
            ))
        except Exception as exc:
            print(f"[axis {name}] FAILED: {exc!r}", flush=True)
            summaries.append({"axis": name, "error": str(exc)})

    (out_root / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out_root / 'all_summaries.json'}")


if __name__ == "__main__":
    main()
