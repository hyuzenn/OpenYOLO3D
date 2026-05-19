"""Regression tests for the Task 1.2b intrinsic-rescale fix.

The streaming wrapper needs to rescale the color-resolution intrinsic
(from ``intrinsics.txt``) to depth resolution before vertex projection.
Without this, every projected (u, v) lands outside the depth-map bounds
and ``inside_mask`` collapses to mostly False → MVPDist accumulator
under-counts → ScanNet AP regresses by ~50% (scene0011_00 v01 FAIL).

These tests freeze the contract so a future refactor does not regress.
"""
from __future__ import annotations

import numpy as np

from method_scannet.streaming.visibility import compute_vertex_projection


def _adjust_via_world2cam(intrinsic, original_resolution, new_resolution):
    """Call ``WORLD_2_CAM.adjust_intrinsic`` exactly the way ``setup_scene`` does."""
    from utils import WORLD_2_CAM

    return WORLD_2_CAM.adjust_intrinsic(
        None, intrinsic, original_resolution, new_resolution
    )


def test_adjust_intrinsic_identity_on_matching_resolutions():
    """If color and depth share resolution, adjust returns the intrinsic unchanged."""
    raw = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    adjusted = _adjust_via_world2cam(raw, (480, 640), (480, 640))
    np.testing.assert_array_equal(adjusted, raw)


def test_adjust_intrinsic_scales_scannet_color_to_depth():
    """ScanNet (968, 1296) color → (480, 640) depth: pinhole moves to depth scale.

    The numeric expectations come straight from
    ``WORLD_2_CAM.adjust_intrinsic`` (utils/__init__.py:377-389), so the
    test is a guard against the streaming wrapper drifting away from the
    offline formula.
    """
    import math

    raw = np.array(
        [[1170.0, 0.0, 646.0], [0.0, 1170.0, 484.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    adjusted = _adjust_via_world2cam(raw, (968, 1296), (480, 640))

    # resize_width = floor(640 * 968 / 1296) = 478 (note: 1296·477 = 618 192
    # < 619 520 = 640·968 < 619 488 = 1296·478 → floor lands on 478).
    resize_width = int(math.floor(640 * 968 / 1296))
    assert resize_width == 478

    expected_fx = 1170.0 * resize_width / 968.0
    expected_fy = 1170.0 * 640.0 / 1296.0
    expected_cx = 646.0 * 479.0 / 967.0
    expected_cy = 484.0 * 639.0 / 1295.0

    assert abs(adjusted[0, 0] - expected_fx) < 1e-6
    assert abs(adjusted[1, 1] - expected_fy) < 1e-6
    assert abs(adjusted[0, 2] - expected_cx) < 1e-6
    assert abs(adjusted[1, 2] - expected_cy) < 1e-6

    # The rescaled intrinsic puts the principal point near the depth-image
    # centre (~320, ~240). Validates the high-level intent: project into a
    # 640×480 frame, not a 1296×968 one.
    assert 315 < adjusted[0, 2] < 325
    assert 235 < adjusted[1, 2] < 245


def test_visibility_recovers_with_adjusted_intrinsic():
    """Projecting through a depth-rescaled intrinsic puts the vertex inside the depth map.

    Drives the Task 1.2b sanity-failure scenario in miniature:
    - Color-resolution intrinsic (fx≈1170, cx≈646) used on a depth-resolution
      image (480×640) → projection lands outside frustum (u≈646 ≥ 640) →
      vertex marked invisible.
    - After rescaling to depth resolution → projection lands at the depth
      image centre → vertex marked visible.
    """
    raw_color_intrinsic = np.array(
        [[1170.0, 0.0, 646.0], [0.0, 1170.0, 484.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    extrinsic = np.eye(4, dtype=np.float64)
    # Vertex on the optical axis at 1 m → after K @ X / z, u = cx, v = cy.
    vertices = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    depth_map = np.ones((480, 640), dtype=np.float32)

    # ---- WITHOUT rescale: u=646 ≥ 640 width → not visible ----
    _proj_buggy, inside_buggy = compute_vertex_projection(
        vertices, raw_color_intrinsic, extrinsic, depth_map
    )
    assert not inside_buggy[0], (
        "color-resolution intrinsic should project the vertex outside the "
        "depth-image frustum (this is the v01 FAIL pattern)"
    )

    # ---- WITH rescale: u≈319, v≈239 inside the frustum → visible ----
    adjusted = _adjust_via_world2cam(raw_color_intrinsic, (968, 1296), (480, 640))
    _proj_fixed, inside_fixed = compute_vertex_projection(
        vertices, adjusted, extrinsic, depth_map
    )
    assert inside_fixed[0], (
        "depth-resolution intrinsic must put the optical-axis vertex inside "
        "the frustum and pass the depth check"
    )
