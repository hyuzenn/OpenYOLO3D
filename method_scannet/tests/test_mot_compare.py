"""Unit tests for the E1 MOT-comparison evaluator (HOTA + motmetrics harness).

Hand-checkable cases:
- perfect tracking -> HOTA = DetA = AssA = IDF1 = MOTA = 1
- classic split-track (one GT, pred id switches mid-track, perfect boxes):
  DetA = 1, AssA = 0.5, HOTA = sqrt(0.5), IDF1 = 0.5 (Luiten et al. IJCV'21)
- empty prediction -> HOTA = 0, all GT are misses
"""
import math

import numpy as np
import pytest

from method_scannet.streaming.eval_mot_compare_outdoor import (
    clear_id_from_frames, combine_clear_id, combine_hota, hota_from_frames,
    ovtcs_c_per_scene)


def _sim(n_gt, n_pr, pairs):
    s = np.zeros((n_gt, n_pr))
    for i, j, v in pairs:
        s[i, j] = v
    return s


def test_perfect_tracking():
    frames = [(["g1", "g2"], ["p1", "p2"],
               _sim(2, 2, [(0, 0, 1.0), (1, 1, 1.0)])) for _ in range(10)]
    h = hota_from_frames(frames)
    assert h["HOTA"] == pytest.approx(1.0)
    assert h["DetA"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(1.0)
    c = clear_id_from_frames(frames)
    assert c["IDF1"] == pytest.approx(1.0)
    assert c["MOTA"] == pytest.approx(1.0)
    assert c["MOTP_m"] == pytest.approx(0.0)
    assert c["IDS"] == 0


def test_split_track():
    frames = [(["g"], ["pA" if t < 5 else "pB"], _sim(1, 1, [(0, 0, 1.0)]))
              for t in range(10)]
    h = hota_from_frames(frames)
    assert h["DetA"] == pytest.approx(1.0)
    assert h["AssA"] == pytest.approx(0.5)
    assert h["HOTA"] == pytest.approx(math.sqrt(0.5))
    c = clear_id_from_frames(frames)
    assert c["IDF1"] == pytest.approx(0.5)
    assert c["IDS"] == 1


def test_empty_prediction():
    frames = [(["g"], [], np.zeros((1, 0))) for _ in range(5)]
    h = hota_from_frames(frames)
    assert h["HOTA"] == 0.0
    c = clear_id_from_frames(frames)
    assert c["FN"] == 5 and c["n_matches"] == 0


def test_combine_equals_pooled():
    # Two "scenes" with disjoint id sets: combined per-scene counts must equal
    # the pooled computation exactly (block-diagonal argument).
    sc1 = [(["a"], ["x" if t < 3 else "y"], _sim(1, 1, [(0, 0, 0.8)]))
           for t in range(6)]
    sc2 = [(["b", "c"], ["z"], _sim(2, 1, [(0, 0, 0.6)])) for t in range(4)]
    pooled = hota_from_frames(sc1 + sc2)
    comb = combine_hota([hota_from_frames(sc1), hota_from_frames(sc2)])
    for k in ("HOTA", "DetA", "AssA"):
        assert comb[k] == pytest.approx(pooled[k])
    pooled_c = clear_id_from_frames(sc1 + sc2)
    comb_c = combine_clear_id([clear_id_from_frames(sc1),
                               clear_id_from_frames(sc2)])
    for k in ("MOTA", "IDF1", "IDP", "IDR", "IDS", "FP", "FN"):
        assert comb_c[k] == pytest.approx(pooled_c[k])


def test_ovtcs_per_scene_grouping():
    # gid // stride buckets tracks by scene; formula = (1-1/L)(1-CSR).
    stride = 1_000_000
    seqs = {0: [1, 1, 1, 1], 1: [1, 2, 1, 2],           # scene 0
            stride + 5: [3, 3]}                          # scene 1
    per = ovtcs_c_per_scene(seqs, stride)
    assert per[0]["n_tracks"] == 2
    assert per[0]["ovtcs_C_mean"] == pytest.approx((0.75 * 1.0 + 0.75 * 0.0) / 2)
    assert per[1]["ovtcs_C_mean"] == pytest.approx(0.5)
