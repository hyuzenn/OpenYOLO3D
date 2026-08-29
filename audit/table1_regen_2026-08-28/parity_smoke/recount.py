"""Recompute the count comparison with the correct ego distance.

compare_to_anchor.py used ||ego_translation|| as the ego distance. The harness
stores `ego_translation` as the ego's ABSOLUTE global position
(nuscenes_native_evaluator: ego_translation = ego_pose[:3, 3]); the devkit's
add_center_dist overwrites it with the relative vector only later, inside the
evaluator. The correct distance is ||translation - ego_translation||.

Read-only over the existing tracks.json + anchor JSON. No nuScenes load.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/rintern16/OpenYOLO3D")
OUT = ROOT / "audit/table1_regen_2026-08-28/parity_smoke"
TRACKS = OUT / "run/outputs/axis_baseline/tracks.json"
ANCHOR = ROOT / "audit/official_centerpoint_ref/nusc_submission/pred_instances_3d/results_nusc.json"

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
           "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50,
               "construction_vehicle": 50, "pedestrian": 40, "motorcycle": 40,
               "bicycle": 40, "traffic_cone": 30, "barrier": 30}

tracks = json.loads(TRACKS.read_text())
ours, tokens = tracks["pred"], sorted(tracks["gt"])
anchor_all = json.load(open(ANCHOR))["results"]
anchor = {t: anchor_all[t] for t in tokens}
del anchor_all

o_cls, a_cls = defaultdict(int), defaultdict(int)
o_raw = 0
for t in tokens:
    for b in ours.get(t, []):
        o_raw += 1
        d = np.linalg.norm(np.asarray(b["translation"][:2], float)
                           - np.asarray(b["ego_translation"][:2], float))
        if d <= CLASS_RANGE.get(b["detection_name"], 0.0):
            o_cls[b["detection_name"]] += 1
    for b in anchor[t]:
        a_cls[b["detection_name"]] += 1

n = len(tokens)
o_tot, a_tot = sum(o_cls.values()), sum(a_cls.values())
res = {
    "n_samples": n,
    "ours_raw_total": o_raw, "ours_raw_per_sample": o_raw / n,
    "ours_after_class_range": o_tot, "ours_after_class_range_per_sample": o_tot / n,
    "anchor_total": a_tot, "anchor_per_sample": a_tot / n,
    "count_rel_err_vs_anchor": (o_tot - a_tot) / a_tot,
    "per_class": {c: {"ours_ranged": o_cls.get(c, 0), "anchor": a_cls.get(c, 0),
                      "rel_err": ((o_cls.get(c, 0) - a_cls.get(c, 0)) / a_cls[c]
                                  if a_cls.get(c) else None)}
                  for c in CLASSES},
}
res["gate_count_within_10pct"] = abs(res["count_rel_err_vs_anchor"]) <= 0.10
(OUT / "recount.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
