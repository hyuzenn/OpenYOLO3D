"""Run nuScenes-devkit DetectionEval-style metrics over a custom EvalBoxes
pair without going through DetectionEval's CLI / NuScenes-object loading.

Standard `DetectionEval` requires the predicted EvalBoxes to cover every
sample in the eval split. We have predictions for 50 hand-picked samples
spanning v1.0-mini and v1.0-trainval — two NuScenes objects, two splits
— so we drive `accumulate` / `calc_ap` / `calc_tp` directly with the
exact same parameters DetectionEval uses (cvpr_2019 config), restricted
to those 50 sample tokens.

This is *not* a re-implementation of the metric formulas — it calls the
official primitives end-to-end. The only piece DetectionEval does that
we don't: filter predictions/GT to the class_range and to the image-FOV.
We replicate the class_range filter (drop GT boxes whose ego_distance
exceeds class_range[detection_name]); the FOV filter is irrelevant since
we operate on predictions, not on the LiDAR sweep itself.
"""

from __future__ import annotations

import json
import os
import os.path as osp
from typing import Dict, List

import numpy as np

from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import (
    DetectionBox,
    DetectionMetricData,
    DetectionMetrics,
)


def _filter_by_range(boxes: EvalBoxes, class_range: Dict[str, float]) -> EvalBoxes:
    """Drop boxes farther than class_range[detection_name] from ego.

    Mirrors DetectionEval.filter_eval_boxes range filter. ego_translation
    is the ego pose of that sample, so distance = ||box.translation - ego||.
    """
    filtered = EvalBoxes()
    for token in boxes.sample_tokens:
        kept = []
        for b in boxes[token]:
            cls = b.detection_name
            if cls not in class_range:
                continue
            r = class_range[cls]
            ego = np.asarray(b.ego_translation, dtype=np.float64)
            xyz = np.asarray(b.translation, dtype=np.float64)
            if np.linalg.norm(xyz[:2] - ego[:2]) <= r:
                kept.append(b)
        filtered.add_boxes(token, kept)
    return filtered


def build_eval_boxes(per_sample_boxes: Dict[str, List[dict]]) -> EvalBoxes:
    eb = EvalBoxes()
    for token, dicts in per_sample_boxes.items():
        eb.add_boxes(token, [DetectionBox.deserialize(d) for d in dicts])
    return eb


def evaluate(
    pred_boxes: EvalBoxes,
    gt_boxes: EvalBoxes,
    output_dir: str,
    config_name: str = "detection_cvpr_2019",
) -> dict:
    """Run mAP + TP metrics + NDS, mirroring DetectionEval's main loop.

    Writes:
      - {output_dir}/eval_summary.json   serialized DetectionMetrics
      - {output_dir}/per_class.json      per-class AP@0.5/1.0/2.0/4.0 + TP
    Returns the parsed summary dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = config_factory(config_name)

    pred_boxes = _filter_by_range(pred_boxes, cfg.class_range)
    gt_boxes = _filter_by_range(gt_boxes, cfg.class_range)

    metrics = DetectionMetrics(cfg)
    metric_data_list = {}

    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name,
                            cfg.dist_fcn_callable, dist_th)
            metric_data_list[(class_name, dist_th)] = md

        for dist_th in cfg.dist_ths:
            md = metric_data_list[(class_name, dist_th)]
            ap = calc_ap(md, cfg.min_recall, cfg.min_precision)
            metrics.add_label_ap(class_name, dist_th, ap)

        # TP metrics use the dist_th_tp threshold (default 2.0 m).
        for metric_name in TP_METRICS:
            md = metric_data_list[(class_name, cfg.dist_th_tp)]
            if class_name in ("traffic_cone",) and metric_name in ("attr_err", "vel_err", "orient_err"):
                tp = float("nan")
            elif class_name in ("barrier",) and metric_name in ("attr_err", "vel_err"):
                tp = float("nan")
            else:
                tp = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

    summary = metrics.serialize()
    summary["counts"] = {
        "n_pred_boxes": int(sum(len(pred_boxes[t]) for t in pred_boxes.sample_tokens)),
        "n_gt_boxes":   int(sum(len(gt_boxes[t])   for t in gt_boxes.sample_tokens)),
        "n_samples":    int(len(gt_boxes.sample_tokens)),
    }

    with open(osp.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    per_class = {}
    for cls in cfg.class_names:
        per_class[cls] = {
            "AP@0.5":     metrics._label_aps[cls][0.5],
            "AP@1.0":     metrics._label_aps[cls][1.0],
            "AP@2.0":     metrics._label_aps[cls][2.0],
            "AP@4.0":     metrics._label_aps[cls][4.0],
            "AP_mean":    float(np.mean([metrics._label_aps[cls][th] for th in cfg.dist_ths])),
        }
        for tp_name in TP_METRICS:
            per_class[cls][tp_name] = metrics._label_tp_errors[cls][tp_name]
    with open(osp.join(output_dir, "per_class.json"), "w") as f:
        json.dump(per_class, f, indent=2)

    return summary
