"""Unit tests for phase2_integration_draft.py.

Verifies (a) all install_* entries raise NotImplementedError so they can't
be activated by accident before the hooks.py merge, and (b) the
per-frame visual-embedding helper works on mock data. CPU-forced.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from method_scannet.clip_image_encoder import CLIPImageEncoder
from method_scannet.phase2_integration_draft import (
    extract_per_frame_visual_embeddings,
    install_method_22_only,
    install_method_32_only,
    install_mix_a,
    install_mix_b,
    install_phase2,
    uninstall_method_22_only,
    uninstall_method_32_only,
    uninstall_mix_a,
    uninstall_mix_b,
    uninstall_phase2,
)


@pytest.fixture(scope="module")
def encoder():
    return CLIPImageEncoder()


def test_install_functions_raise_not_implemented():
    """All install_* (and uninstall_*) raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        install_method_22_only()
    with pytest.raises(NotImplementedError):
        install_method_32_only()
    with pytest.raises(NotImplementedError):
        install_phase2()
    with pytest.raises(NotImplementedError):
        install_mix_a()
    with pytest.raises(NotImplementedError):
        install_mix_b()

    with pytest.raises(NotImplementedError):
        uninstall_method_22_only()
    with pytest.raises(NotImplementedError):
        uninstall_method_32_only()
    with pytest.raises(NotImplementedError):
        uninstall_phase2()
    with pytest.raises(NotImplementedError):
        uninstall_mix_a()
    with pytest.raises(NotImplementedError):
        uninstall_mix_b()


def test_install_method_22_only_validates_prep_path():
    """Even though install raises, the prep work runs first — confirm it
    raises NotImplementedError (not e.g. FileNotFoundError or an
    embedding-shape error). This exercises the .pt load + FeatureFusionEMA
    + CLIPImageEncoder construction chain."""
    with pytest.raises(NotImplementedError) as excinfo:
        install_method_22_only()
    msg = str(excinfo.value)
    assert "phase2_integration_draft" in msg
    assert "hooks.py" in msg


def test_extract_per_frame_basic(encoder):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    instance_ids = np.array([0, 1])
    result = extract_per_frame_visual_embeddings(image, bboxes, instance_ids, encoder)
    assert 0 in result and 1 in result
    assert result[0].shape == (512,)
    assert result[1].shape == (512,)
    assert torch.allclose(result[0].norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(result[1].norm(), torch.tensor(1.0), atol=1e-5)


def test_extract_per_frame_same_instance_averaged(encoder):
    """Two bboxes with the same instance id → fused, L2-normalized."""
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    instance_ids = np.array([0, 0])
    result = extract_per_frame_visual_embeddings(image, bboxes, instance_ids, encoder)
    assert len(result) == 1
    assert result[0].shape == (512,)
    assert torch.allclose(result[0].norm(), torch.tensor(1.0), atol=1e-5)


def test_extract_per_frame_empty(encoder):
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([]).reshape(0, 4)
    instance_ids = np.array([])
    result = extract_per_frame_visual_embeddings(image, bboxes, instance_ids, encoder)
    assert result == {}


def test_extract_per_frame_id_count_mismatch_raises(encoder):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    instance_ids = np.array([0])  # mismatched length
    with pytest.raises(ValueError):
        extract_per_frame_visual_embeddings(image, bboxes, instance_ids, encoder)
