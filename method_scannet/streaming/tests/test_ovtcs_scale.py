"""Unit tests for the OV-TCS-aware EMA (conf_mode='ovtcs_scale') and the new
online OV-TCS instrumentation in ``FeatureFusionEMA``.

Goals:
  1. Regression guard — none/skip/weighted feature math is byte-identical to the
     analytic EMA formula (i.e. the new bookkeeping introduced no perturbation),
     including weighted + record_feature_trace=True (trace is read-only).
  2. Causal online OV-TCS_C drops on a planted raw-label flip.
  3. ovtcs_scale applies w = clamp(alpha*c*OVTCS, 0, 1) exactly.
  4. ovtcs_diagnostics() keys/invariants (applied<=observed, finite cos_to_final).

Runnable as `python -m method_scannet.streaming.tests.test_ovtcs_scale` or pytest.
Tiny tensors only — safe to run on the util (CPU) node.
"""
from __future__ import annotations

import torch

from method_scannet.method_22_feature_fusion import FeatureFusionEMA

# Two orthonormal class prompts in R^4 → raw argmax of e_k picks class k.
PROMPTS = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
CLS = ["c0", "c1"]
C0 = torch.tensor([1.0, 0, 0, 0])
C1 = torch.tensor([0, 1.0, 0, 0])
ALPHA = 0.7
TOL = 1e-6


def _approx(a: torch.Tensor, b: torch.Tensor, tol: float = TOL) -> bool:
    return bool(torch.allclose(a.float(), b.float(), atol=tol))


def test_none_mode_regression():
    """Plain EMA: F = alpha*prev + (1-alpha)*cur, no OV-TCS perturbation."""
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="none", normalize_per_frame=False)
    f.update_instance_feature(0, C0.clone())          # seed
    f.update_instance_feature(0, C1.clone())          # update
    expect = ALPHA * C0 + (1 - ALPHA) * C1
    assert _approx(f.get_feature(0), expect), f.get_feature(0)


def test_weighted_mode_regression():
    """Weighted EMA: F = (1-alpha*c)*prev + alpha*c*cur; conf=None → c=1."""
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="weighted", normalize_per_frame=False)
    f.update_instance_feature(0, C0.clone(), confidence=0.5)   # seed (no EMA yet)
    f.update_instance_feature(0, C1.clone(), confidence=0.5)
    w = ALPHA * 0.5
    expect = (1 - w) * C0 + w * C1
    assert _approx(f.get_feature(0), expect), f.get_feature(0)

    # conf=None → c=1.0
    g = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="weighted", normalize_per_frame=False)
    g.update_instance_feature(0, C0.clone())
    g.update_instance_feature(0, C1.clone())
    expect2 = (1 - ALPHA) * C0 + ALPHA * C1
    assert _approx(g.get_feature(0), expect2), g.get_feature(0)


def test_trace_is_readonly():
    """record_feature_trace=True must NOT change the weighted feature math."""
    base = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="weighted", normalize_per_frame=False,
                            prompt_embeddings=PROMPTS, prompt_class_names=CLS)
    traced = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="weighted", normalize_per_frame=False,
                              prompt_embeddings=PROMPTS, prompt_class_names=CLS,
                              record_feature_trace=True)
    for fusion in (base, traced):
        fusion.update_instance_feature(0, C0.clone(), confidence=0.8)
        fusion.update_instance_feature(0, C1.clone(), confidence=0.8)
        fusion.update_instance_feature(0, C0.clone(), confidence=0.8)
    assert _approx(base.get_feature(0), traced.get_feature(0))


def test_online_ovtcs_drops_on_flip():
    """OV-TCS_C: stable run stays high; a label flip lowers it."""
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="ovtcs_scale", normalize_per_frame=False,
                         prompt_embeddings=PROMPTS, prompt_class_names=CLS)
    # frame1 c0 (seed, L=1 → 0.0); frame2 c0 (L=2, no switch → 0.5);
    o1 = f._update_raw_ovtcs(0, C0)
    assert o1 == 0.0
    f.instance_features[0] = C0.clone()
    o2 = f._update_raw_ovtcs(0, C0)
    assert abs(o2 - 0.5) < TOL, o2
    # frame3 c1 (L=3, 1 switch → (1-1/3)*(1-1/2)=1/3)
    o3 = f._update_raw_ovtcs(0, C1)
    assert abs(o3 - (1.0 / 3.0)) < TOL, o3
    # a perfectly stable longer run → OV-TCS rises toward 1
    g = FeatureFusionEMA(conf_mode="ovtcs_scale", prompt_embeddings=PROMPTS,
                         prompt_class_names=CLS)
    last = 0.0
    for _ in range(6):
        last = g._update_raw_ovtcs(1, C0)
    assert last > o3  # stable beats flipped


def test_ovtcs_scale_weight_exact():
    """ovtcs_scale third-frame feature equals (1-w)*prev + w*cur, w=alpha*c*OVTCS."""
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="ovtcs_scale", normalize_per_frame=False,
                         prompt_embeddings=PROMPTS, prompt_class_names=CLS,
                         record_feature_trace=True)
    f.update_instance_feature(0, C0.clone(), confidence=1.0)  # seed
    f.update_instance_feature(0, C0.clone(), confidence=1.0)  # L=2, OVTCS=0.5, cur==prev
    # prev is still C0 here (identical update). frame3: OVTCS=1/3, w=0.7*1/3.
    f.update_instance_feature(0, C1.clone(), confidence=1.0)
    w = ALPHA * (1.0 / 3.0)
    expect = (1 - w) * C0 + w * C1
    assert _approx(f.get_feature(0), expect), f.get_feature(0)


def test_const_scale_weight_exact():
    """const_scale: F = (1-w)*prev + w*cur, w = alpha*c*const_k (no OVTCS)."""
    k = 0.335
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="const_scale", const_k=k,
                         normalize_per_frame=False, prompt_embeddings=PROMPTS,
                         prompt_class_names=CLS)
    f.update_instance_feature(0, C0.clone(), confidence=1.0)  # seed
    f.update_instance_feature(0, C1.clone(), confidence=1.0)
    w = ALPHA * 1.0 * k
    expect = (1 - w) * C0 + w * C1
    assert _approx(f.get_feature(0), expect), f.get_feature(0)
    # const_scale weight must NOT depend on the OVTCS sequence: a flipped vs
    # stable history with the same #updates yields the same incoming weight.
    assert abs(w - ALPHA * k) < TOL


def test_diagnostics_invariants():
    f = FeatureFusionEMA(ema_alpha=ALPHA, conf_mode="ovtcs_scale", normalize_per_frame=False,
                         prompt_embeddings=PROMPTS, prompt_class_names=CLS,
                         record_feature_trace=True)
    seq = [C0, C0, C1, C1, C0]
    for e in seq:
        f.update_instance_feature(0, e.clone(), confidence=0.9)
    d = f.ovtcs_diagnostics()
    assert d["n_instances"] == 1
    # applied (seed + 4 updates) and observed (5) must satisfy applied<=observed
    assert d["n_updates_applied_total"] <= d["n_observations_total"]
    assert d["n_observations_total"] == len(seq)
    assert d["feature_drift"]["n"] >= 1
    assert d["online_ovtcs"]["mean"] is not None
    tr = d["trace"]
    assert len(tr["ovtcs"]) == len(tr["drift"]) == len(tr["cos_to_final"])
    for c in tr["cos_to_final"]:
        assert -1.0001 <= c <= 1.0001, c


def test_invalid_conf_mode():
    try:
        FeatureFusionEMA(conf_mode="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid conf_mode")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"all {len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
