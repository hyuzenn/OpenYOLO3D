"""Unit tests for METHOD_22 hook (install_method_22_only).

GPU is NOT exercised — these tests only check install/uninstall mechanics,
state population, and conflict semantics. The actual prediction pipeline is
exercised in a separate smoke test (smoke_method_22_one_scene.py) that
needs CUDA.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure no patches leak between tests."""
    from method_scannet.hooks import (
        uninstall_method_21_only,
        uninstall_method_22_only,
        uninstall_method_31_only,
    )
    from utils import OpenYolo3D

    # Pre-clean
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        try:
            uninstall_method_31_only()
        except Exception:
            pass
        try:
            uninstall_method_22_only()
        except Exception:
            pass
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_label_maps"):
        try:
            uninstall_method_21_only()
        except Exception:
            pass
    yield
    # Post-clean
    if hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes"):
        try:
            uninstall_method_31_only()
        except Exception:
            pass
        try:
            uninstall_method_22_only()
        except Exception:
            pass


def test_install_uninstall_round_trip():
    from method_scannet.hooks import (
        _method22_state,
        install_method_22_only,
        uninstall_method_22_only,
    )
    from utils import OpenYolo3D

    assert not hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes")
    install_method_22_only()
    assert hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes")
    assert _method22_state["fusion"] is not None
    assert _method22_state["image_encoder"] is not None
    assert _method22_state["n_inference_classes"] == 198
    uninstall_method_22_only()
    assert not hasattr(OpenYolo3D, "_original_label_3d_masks_from_2d_bboxes")
    assert _method22_state["fusion"] is None


def test_inference_subset_class_count():
    """use_inference_subset=True drops wall+floor → 198 classes."""
    from method_scannet.hooks import (
        _method22_state,
        install_method_22_only,
        uninstall_method_22_only,
    )

    install_method_22_only(use_inference_subset=True)
    assert _method22_state["n_inference_classes"] == 198
    uninstall_method_22_only()


def test_full_class_count():
    """use_inference_subset=False uses all 200 classes."""
    from method_scannet.hooks import (
        _method22_state,
        install_method_22_only,
        uninstall_method_22_only,
    )

    install_method_22_only(use_inference_subset=False)
    assert _method22_state["n_inference_classes"] == 200
    uninstall_method_22_only()


def test_refuse_install_when_method_31_active():
    """install_method_22_only should refuse if METHOD_31 hook is already
    active to avoid silently clobbering the patch chain.
    """
    from method_scannet.hooks import (
        install_method_22_only,
        install_method_31_only,
        uninstall_method_31_only,
    )

    install_method_31_only()
    with pytest.raises(RuntimeError, match="already patched"):
        install_method_22_only()
    uninstall_method_31_only()


def test_param_overrides_persist():
    """Custom min_match_iou and topk_per_inst land in module state."""
    from method_scannet.hooks import (
        _method22_state,
        install_method_22_only,
        uninstall_method_22_only,
    )

    install_method_22_only(min_match_iou=0.15, topk_per_inst=8, ema_alpha=0.5)
    assert _method22_state["min_match_iou"] == 0.15
    assert _method22_state["topk_per_inst"] == 8
    assert _method22_state["fusion"].ema_alpha == 0.5
    uninstall_method_22_only()
