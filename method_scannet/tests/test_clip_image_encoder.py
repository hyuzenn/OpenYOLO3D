"""Unit tests for method_scannet/clip_image_encoder.py — CPU-forced."""
from __future__ import annotations

import os

# Hard-pin CPU before any torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import pytest
import torch

from method_scannet.clip_image_encoder import CLIPImageEncoder


# Loading the model once per module is expensive (~1.5s). Share via a
# module-level fixture.
@pytest.fixture(scope="module")
def encoder():
    enc = CLIPImageEncoder()
    return enc


def test_init_loads_correct_variant(encoder):
    assert encoder.variant == "openai/clip-vit-base-patch32"
    assert encoder.embed_dim == 512
    # Hard CPU verification — every parameter should live on CPU.
    for p in encoder.model.parameters():
        assert p.device.type == "cpu"
        break  # one is enough; the .to(device) call applies uniformly


def test_encode_returns_correct_shape(encoder):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    embeddings = encoder.encode_cropped_bboxes(image, bboxes)
    assert embeddings.shape == (2, 512)
    assert embeddings.device.type == "cpu"
    assert embeddings.dtype == torch.float32


def test_l2_normalized(encoder):
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([[100, 100, 200, 200]])
    embeddings = encoder.encode_cropped_bboxes(image, bboxes)
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(1), atol=1e-5)


def test_empty_bboxes(encoder):
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = np.array([]).reshape(0, 4)
    embeddings = encoder.encode_cropped_bboxes(image, bboxes)
    assert embeddings.shape == (0, 512)


def test_consistency_with_text_encoder(encoder):
    """Image encoder variant must match the text encoder used to build
    pretrained/scannet200_prompt_embeddings.pt — otherwise cosine sim
    between image and text embeddings is meaningless."""
    text_data = torch.load("pretrained/scannet200_prompt_embeddings.pt")
    assert text_data["embed_dim"] == 512
    assert text_data["clip_variant"] == "openai/clip-vit-base-patch32"
    assert encoder.embed_dim == text_data["embed_dim"]
    assert encoder.variant == text_data["clip_variant"]
