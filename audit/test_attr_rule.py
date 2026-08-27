"""Unit tests for the official CenterPoint attribute rule (#8 fix).

Pure CPU, no nuScenes data needed. Sections follow the task spec:
A moving vehicles, B slow/static vehicles, C pedestrian, D cycles,
E 0.2 m/s boundary (strict >), F velocity-frame sanity + _detection_box_dict
passthrough (helper consumes the vx,vy already computed; rotation-invariant).
"""
import math
import sys
import os.path as osp

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from method_scannet.streaming.nuscenes_evaluator import (
    _official_attribute, _detection_box_dict)

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


# A. moving vehicles (speed 5 > 0.2)
for cls in ("car", "truck", "bus", "trailer", "construction_vehicle"):
    check(f"A {cls} moving", _official_attribute(cls, 3.0, 4.0),
          "vehicle.moving")

# B. slow/static vehicles (speed 0)
for cls, want in (("car", "vehicle.parked"), ("truck", "vehicle.parked"),
                  ("trailer", "vehicle.parked"),
                  ("construction_vehicle", "vehicle.parked"),
                  ("bus", "vehicle.stopped")):
    check(f"B {cls} static", _official_attribute(cls, 0.0, 0.0), want)

# C. pedestrian
check("C pedestrian moving", _official_attribute("pedestrian", 0.3, 0.0),
      "pedestrian.moving")
check("C pedestrian static", _official_attribute("pedestrian", 0.0, 0.0),
      "pedestrian.standing")

# D. cycles
for cls in ("motorcycle", "bicycle"):
    check(f"D {cls} moving", _official_attribute(cls, 0.0, 1.0),
          "cycle.with_rider")
    check(f"D {cls} static", _official_attribute(cls, 0.0, 0.0),
          "cycle.without_rider")

# barrier / traffic_cone stay "" either way
for cls in ("barrier", "traffic_cone"):
    check(f"D {cls} moving", _official_attribute(cls, 3.0, 4.0), "")
    check(f"D {cls} static", _official_attribute(cls, 0.0, 0.0), "")

# E. boundary: official gate is strict `> 0.2`
check("E car speed==0.2", _official_attribute("car", 0.2, 0.0),
      "vehicle.parked")
check("E car speed just below", _official_attribute("car", 0.2 - 1e-9, 0.0),
      "vehicle.parked")
check("E car speed just above", _official_attribute("car", 0.2 + 1e-9, 0.0),
      "vehicle.moving")
check("E ped speed==0.2", _official_attribute("pedestrian", 0.0, 0.2),
      "pedestrian.standing")

# F. rotation invariance: same magnitude, any direction -> same attribute
rng = np.random.default_rng(0)
for i in range(20):
    theta = rng.uniform(0, 2 * math.pi)
    speed = rng.uniform(0.0, 3.0)
    a0 = _official_attribute("car", speed, 0.0)
    a1 = _official_attribute("car", speed * math.cos(theta),
                             speed * math.sin(theta))
    check(f"F rotation-invariant #{i} (speed={speed:.3f})", a1, a0)

# F. _detection_box_dict passthrough: attribute computed from the SAME vx,vy
# that lands in the record's velocity field (velocity_global when given,
# LiDAR-frame fallback otherwise) — no second rotation anywhere.
common = dict(global_id=1, sample_token="t",
              centroid_global=np.zeros(3), ego_translation=np.zeros(3),
              rotation_global_wxyz=[1.0, 0.0, 0.0, 0.0], score=0.9)
d = _detection_box_dict(bbox_lidar=[0, 0, 0, 1, 1, 1, 0, 9.0, 9.0],
                        detection_name="car",
                        velocity_global=[3.0, 4.0], **common)
check("F box dict uses velocity_global (moving car)",
      (d["velocity"], d["attribute_name"]),
      ([3.0, 4.0], "vehicle.moving"))
d = _detection_box_dict(bbox_lidar=[0, 0, 0, 1, 1, 1, 0, 0.05, 0.0],
                        detection_name="car", **common)
check("F box dict lidar-frame fallback (parked car)",
      (d["velocity"], d["attribute_name"]),
      ([0.05, 0.0], "vehicle.parked"))
d = _detection_box_dict(bbox_lidar=[0, 0, 0, 1, 1, 1, 0],
                        detection_name="pedestrian", **common)
check("F box dict no velocity (pedestrian.standing)",
      (d["velocity"], d["attribute_name"]),
      ([0.0, 0.0], "pedestrian.standing"))

if FAILURES:
    print(f"\n=== {len(FAILURES)} FAILURES ===")
    sys.exit(1)
print("\n=== ALL PASS ===")
