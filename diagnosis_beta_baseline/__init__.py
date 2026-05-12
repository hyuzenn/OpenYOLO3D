"""β baseline — OpenYOLO3D × nuScenes mAP/NDS measurement.

Reuses the existing nuScenes adapter (single-cam CAM_FRONT) and
OpenYOLO3D pipeline as-is. Produces:
  - per_sample/<token>.json    raw predictions/instance counts/timings
  - nuscenes_eval/             EvalBoxes + DetectionEval-style summary
  - figures/                   per-class / per-distance bars
  - report.md                  standard + instance-level + decision branch
"""

NUSCENES_10_CLASS = [
    "car", "truck", "bus", "trailer", "construction_vehicle",
    "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier",
]

DETECTION_CLASS_RANGE = {
    "car": 50.0, "truck": 50.0, "bus": 50.0, "trailer": 50.0,
    "construction_vehicle": 50.0,
    "pedestrian": 40.0, "motorcycle": 40.0, "bicycle": 40.0,
    "traffic_cone": 30.0, "barrier": 30.0,
}

DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
