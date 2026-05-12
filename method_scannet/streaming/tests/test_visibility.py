"""Unit tests for streaming/visibility.py (D3 frame visibility). GPU-free."""
from __future__ import annotations

import numpy as np

from method_scannet.streaming.visibility import compute_frame_visibility


def _intrinsic(fx: float = 500.0, fy: float = 500.0, cx: float = 320.0, cy: float = 240.0):
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _depth_map(value: float = 1.0, h: int = 480, w: int = 640):
    return np.full((h, w), value, dtype=np.float32)


def test_visibility_simple_in_view():
    """Instance with 2 vertices on the optical axis at z=1m, depth_map=1m → visible."""
    vertices = np.array([[0.0, 0.0, 1.0], [0.001, 0.001, 1.0]], dtype=np.float64)
    instance_masks = np.array([[True, True]])
    intrinsic = _intrinsic()
    extrinsic = np.eye(4, dtype=np.float64)
    depth_map = _depth_map(value=1.0)

    visible = compute_frame_visibility(
        instance_masks, vertices, intrinsic, extrinsic, depth_map
    )

    assert visible.shape == (1,)
    assert bool(visible[0]) is True


def test_visibility_all_behind_camera():
    """All vertices behind the camera (z < 0) → not visible."""
    vertices = np.array([[0.0, 0.0, -1.0], [0.1, 0.1, -1.0]], dtype=np.float64)
    instance_masks = np.array([[True, True]])
    intrinsic = _intrinsic()
    extrinsic = np.eye(4, dtype=np.float64)
    depth_map = _depth_map()

    visible = compute_frame_visibility(
        instance_masks, vertices, intrinsic, extrinsic, depth_map
    )

    assert bool(visible[0]) is False


def test_visibility_outside_frustum():
    """Vertex in front of camera but projects far outside image bounds → not visible."""
    # x = 100 at z=1, fx=500 → u = 50000 ≫ width=640
    vertices = np.array([[100.0, 0.0, 1.0]], dtype=np.float64)
    instance_masks = np.array([[True]])
    intrinsic = _intrinsic()
    extrinsic = np.eye(4, dtype=np.float64)
    depth_map = _depth_map()

    visible = compute_frame_visibility(
        instance_masks, vertices, intrinsic, extrinsic, depth_map
    )

    assert bool(visible[0]) is False


def test_visibility_depth_mismatch():
    """Vertex inside frustum but at depth 5m while depth_map says 1m → occluded, not visible."""
    vertices = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    instance_masks = np.array([[True]])
    intrinsic = _intrinsic()
    extrinsic = np.eye(4, dtype=np.float64)
    depth_map = _depth_map(value=1.0)

    visible = compute_frame_visibility(
        instance_masks, vertices, intrinsic, extrinsic, depth_map,
        depth_threshold=0.05,
    )

    assert bool(visible[0]) is False


def test_visibility_d3_lenient():
    """D3 instance gate: 10 vertices, 1 visible, 9 behind camera → instance still visible."""
    vertices = np.array(
        [[0.0, 0.0, 1.0]] + [[0.0, 0.0, -float(i)] for i in range(1, 10)],
        dtype=np.float64,
    )
    instance_masks = np.array([[True] * 10])
    intrinsic = _intrinsic()
    extrinsic = np.eye(4, dtype=np.float64)
    depth_map = _depth_map(value=1.0)

    visible = compute_frame_visibility(
        instance_masks, vertices, intrinsic, extrinsic, depth_map
    )

    assert bool(visible[0]) is True
