"""Streaming evaluation wrapper for OpenYOLO3D ScanNet pipeline.

Task 1.2a skeleton. Implements:
  - D3 frame visibility (≥1 vertex visible per instance)
  - StreamingScanNetEvaluator wrapper (Mask3D-per-scene + per-frame YOLO)
  - 4 temporal metric function signatures (incremental mAP, ID switches,
    label switches, time-to-confirm with C1/K=3)

Method integration (M11/12/21/22/31/32) is Task 1.4. Baseline streaming
sanity check on real scene is Task 1.2b. Real metric measurement is
Task 1.2b/1.3.

OpenYOLO3D core (utils/, run_evaluation.py) and 5월 method modules
(method_scannet/method_*.py, hooks.py) are NOT modified.
"""
from method_scannet.streaming.visibility import compute_frame_visibility
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from method_scannet.streaming.metrics import (
    incremental_map_primary,
    incremental_map_secondary,
    id_switch_count,
    label_switch_count,
    time_to_confirm,
    mask_iou_map,
    evaluate_scene_scannet200,
)
from method_scannet.streaming.baseline import (
    BaselineLabelAccumulator,
    construct_label_map_single_frame,
)

__all__ = [
    "compute_frame_visibility",
    "StreamingScanNetEvaluator",
    "incremental_map_primary",
    "incremental_map_secondary",
    "id_switch_count",
    "label_switch_count",
    "time_to_confirm",
    "mask_iou_map",
    "evaluate_scene_scannet200",
    "BaselineLabelAccumulator",
    "construct_label_map_single_frame",
]
