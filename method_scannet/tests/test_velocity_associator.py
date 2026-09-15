"""Constant-velocity associator: does prediction bridge a gap the static gate misses?"""
import numpy as np
from method_scannet.streaming.nuscenes_native_evaluator import (
    CentroidAssociator, VelocityCentroidAssociator)


def _prop(x, y, vx=0.0, vy=0.0, cls="car", score=0.9):
    return {"cls_name": cls, "score": score, "centroid_ego": [x, y, 0.0],
            "bbox_lidar": [x, y, 0.0, 2.0, 4.0, 1.5, 0.0, vx, vy]}


def test_velocity_keeps_fast_track_static_loses_it():
    # 5 m/s over 0.5 s = 2.5 m per frame, outside the 2.0 m gate.
    frames = [[_prop(0.0 + 2.5 * k, 0.0, vx=5.0)] for k in range(4)]

    static = CentroidAssociator(threshold_m=2.0, max_age=5)
    ids_static = [static.step(f)[0] for f in frames]
    assert len(set(ids_static)) == 4, "static gate should fragment a 2.5 m/frame track"

    vel = VelocityCentroidAssociator(threshold_m=2.0, max_age=5)
    vel.set_frame_context(np.eye(3), 0.5)
    ids_vel = [vel.step(f)[0] for f in frames]
    assert len(set(ids_vel)) == 1, f"velocity prediction should hold one id, got {ids_vel}"


def test_zero_velocity_is_byte_identical_to_static():
    frames = [[_prop(0.0, 0.0), _prop(10.0, 0.0)],
              [_prop(0.3, 0.0), _prop(10.4, 0.0)],
              [_prop(0.6, 0.0)]]
    a = CentroidAssociator(threshold_m=2.0, max_age=5)
    b = VelocityCentroidAssociator(threshold_m=2.0, max_age=5)
    b.set_frame_context(np.eye(3), 0.5)
    assert [a.step(f) for f in frames] == [b.step(f) for f in frames]


def test_rotation_is_applied_to_velocity():
    # Motion along +y, but the reported LiDAR velocity is along +x: only a
    # correct lidar->working rotation turns one into the other.
    frames = [[_prop(0.0, 2.5 * k, vx=5.0, vy=0.0)] for k in range(4)]
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +x -> +y
    vel = VelocityCentroidAssociator(threshold_m=2.0, max_age=5)
    vel.set_frame_context(R, 0.5)
    assert len(set(vel.step(f)[0] for f in frames)) == 1


if __name__ == "__main__":
    test_velocity_keeps_fast_track_static_loses_it()
    test_zero_velocity_is_byte_identical_to_static()
    test_rotation_is_applied_to_velocity()
    print("all 3 pass")
