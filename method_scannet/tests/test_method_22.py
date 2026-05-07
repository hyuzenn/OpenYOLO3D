"""Unit tests for METHOD_22 — FeatureFusionEMA. CPU-only mock data."""
from __future__ import annotations

import torch

from method_scannet.method_22_feature_fusion import FeatureFusionEMA


def _make_fusion(num_classes: int = 200, dim: int = 512, alpha: float = 0.7, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    prompt_emb = torch.randn(num_classes, dim, generator=g)
    names = [f"cls_{i:03d}" for i in range(num_classes)]
    return FeatureFusionEMA(ema_alpha=alpha, prompt_embeddings=prompt_emb, prompt_class_names=names), prompt_emb


def test_first_update_seeds_running_feature():
    fusion, _ = _make_fusion()
    v = torch.randn(512)
    fusion.update_instance_feature(7, v)
    stored = fusion.get_feature(7)
    assert stored is not None
    assert torch.allclose(stored, v)


def test_ema_recurrence_matches_formula():
    fusion, _ = _make_fusion(alpha=0.7)
    g = torch.Generator().manual_seed(42)
    f0 = torch.randn(512, generator=g)
    f1 = torch.randn(512, generator=g)
    f2 = torch.randn(512, generator=g)
    fusion.update_instance_feature(1, f0)
    fusion.update_instance_feature(1, f1)
    fusion.update_instance_feature(1, f2)
    expected = 0.7 * (0.7 * f0 + 0.3 * f1) + 0.3 * f2
    assert torch.allclose(fusion.get_feature(1), expected, atol=1e-6)


def test_predict_label_returns_valid_index_and_confidence_in_range():
    fusion, _ = _make_fusion(num_classes=200)
    g = torch.Generator().manual_seed(1)
    for iid in range(5):
        for _ in range(10):
            fusion.update_instance_feature(iid, torch.randn(512, generator=g))
    preds = fusion.predict_all()
    assert len(preds) == 5
    for iid, (name, conf) in preds.items():
        assert isinstance(name, str) and name.startswith("cls_")
        idx = int(name.split("_")[1])
        assert 0 <= idx < 200
        assert -1.0 - 1e-6 <= conf <= 1.0 + 1e-6


def test_argmax_recovers_planted_class():
    """When all of an instance's frames lie near prompt class C, predicted
    label should be class C."""
    fusion, prompts = _make_fusion(num_classes=200)
    target_idx = 137
    target = prompts[target_idx]
    g = torch.Generator().manual_seed(7)
    for _ in range(8):
        noisy = target + 0.05 * torch.randn(target.shape, generator=g)
        fusion.update_instance_feature(99, noisy)
    name, conf = fusion.predict_label(99)
    assert name == "cls_137"
    assert conf > 0.5


def test_predict_without_prompts_raises():
    fusion = FeatureFusionEMA(ema_alpha=0.5)
    fusion.update_instance_feature(0, torch.randn(8))
    try:
        fusion.predict_all()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when prompt embeddings are not set")
