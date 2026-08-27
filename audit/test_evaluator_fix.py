"""Targeted unit/isolation tests for the 2026-08-28 evaluator correction
(plan section F). Proves WHY the numbers change, not just that they change.

F1  GT parity: the REAL corrected _load_meta (+ official filters) must produce
    box-for-box identical GT to the devkit's own load_gt + add_center_dist +
    filter_eval_boxes on random val samples.
F2  Synthetic TP-error semantics: known velocity/attribute/yaw/scale offsets
    produce the hand-computed errors; the all-NaN attr_err -> 1.0 mechanism
    (the old bug's NDS damage) is demonstrated explicitly.
F3  num_pts==0 GT filtering (lidar+radar sum).
F4  bike-rack filtering: a bicycle GT planted at a real bicycle_rack's center
    is dropped by the corrected path, kept by the legacy range-only path.
F5  Prediction velocity frame: _detection_box_dict honors velocity_global;
    the step_sample rotation formula maps a known LiDAR-frame velocity to the
    correct global-frame vector under a 90-degree ego yaw.

Run (PBS, openyolo3d-dev):  python -u audit/test_evaluator_fix.py
Exits nonzero on any failure.
"""
from __future__ import annotations

import os.path as osp
import random
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (add_center_dist, filter_eval_boxes,
                                          load_gt)
from nuscenes.eval.common.utils import (attr_acc, cummean, scale_iou,
                                        velocity_l2, yaw_diff)
from nuscenes.eval.detection.data_classes import DetectionBox

from audit._common import DATAROOT, VERSION
from audit.reeval_official_devkit import val_sample_tokens

N_PARITY_TOKENS = 20
SEED = 0

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAILURES.append(name)


def build_gt_via_load_meta(nusc, tokens):
    """GT through the REAL corrected _load_meta (unbound call, stub self)."""
    from method_scannet.streaming.nuscenes_native_evaluator import (
        NativeTemporalNuScenesEvaluator)
    stub = SimpleNamespace(loader=SimpleNamespace(nusc=nusc))
    eb = EvalBoxes()
    for tok in tokens:
        _ego, _t, gts = NativeTemporalNuScenesEvaluator._load_meta(stub, tok)
        eb.add_boxes(tok, [DetectionBox.deserialize(d) for d in gts])
    return eb


def box_key(b):
    return (b.detection_name, tuple(np.round(b.translation, 6)))


def test_f1_gt_parity(nusc):
    print("=== F1: GT parity vs official load_gt ===", flush=True)
    from nuscenes.eval.detection.config import config_factory
    cfg = config_factory("detection_cvpr_2019")
    tokens = val_sample_tokens(nusc)
    random.seed(SEED)
    sample = random.sample(tokens, N_PARITY_TOKENS)

    ours = build_gt_via_load_meta(nusc, sample)
    ours = filter_eval_boxes(nusc, add_center_dist(nusc, ours), cfg.class_range)

    official_full = load_gt(nusc, "val", DetectionBox, verbose=False)
    official = EvalBoxes()
    for tok in sample:
        official.add_boxes(tok, official_full[tok])
    official = filter_eval_boxes(nusc, add_center_dist(nusc, official),
                                 cfg.class_range)

    n_ours = sum(len(ours[t]) for t in sample)
    n_off = sum(len(official[t]) for t in sample)
    check("F1 box count", n_ours == n_off, f"ours={n_ours} official={n_off}")

    mismatch = 0
    for tok in sample:
        a = sorted(ours[tok], key=box_key)
        b = sorted(official[tok], key=box_key)
        if len(a) != len(b):
            mismatch += abs(len(a) - len(b))
            continue
        for x, y in zip(a, b):
            same = (x.detection_name == y.detection_name
                    and np.allclose(x.translation, y.translation)
                    and np.allclose(x.size, y.size)
                    and np.allclose(x.rotation, y.rotation)
                    and np.allclose(x.velocity, y.velocity, equal_nan=True)
                    and x.num_pts == y.num_pts
                    and x.attribute_name == y.attribute_name
                    and np.allclose(x.ego_translation, y.ego_translation))
            if not same:
                mismatch += 1
                if mismatch <= 3:
                    print(f"  mismatch @{tok}: ours={x.serialize()}\n"
                          f"             official={y.serialize()}", flush=True)
    check("F1 field-by-field parity", mismatch == 0, f"mismatches={mismatch}")


def test_f2_tp_error_semantics():
    print("=== F2: synthetic TP-error semantics ===", flush=True)
    mk = lambda **kw: DetectionBox(sample_token="x", detection_name="car",
                                   size=(2.0, 4.0, 1.5), rotation=(1, 0, 0, 0),
                                   **kw)
    gt_v = mk(velocity=(3.0, 0.0), detection_score=-1.0)
    pr_same = mk(velocity=(3.0, 0.0), detection_score=0.9)
    pr_zero = mk(velocity=(0.0, 0.0), detection_score=0.9)
    check("F2 vel_err exact match = 0", velocity_l2(gt_v, pr_same) == 0.0)
    check("F2 vel_err zeroed pred = speed", velocity_l2(gt_v, pr_zero) == 3.0)

    gt_parked = mk(velocity=(0, 0), attribute_name="vehicle.parked")
    pr_parked = mk(velocity=(0, 0), attribute_name="vehicle.parked",
                   detection_score=0.9)
    pr_moving = mk(velocity=(0, 0), attribute_name="vehicle.moving",
                   detection_score=0.9)
    pr_empty = mk(velocity=(0, 0), attribute_name="", detection_score=0.9)
    gt_noattr = mk(velocity=(0, 0), attribute_name="")
    check("F2 attr correct -> acc 1", attr_acc(gt_parked, pr_parked) == 1.0)
    check("F2 attr wrong -> acc 0", attr_acc(gt_parked, pr_moving) == 0.0)
    check("F2 attr empty pred -> acc 0", attr_acc(gt_parked, pr_empty) == 0.0)
    check("F2 GT attr '' -> NaN", np.isnan(attr_acc(gt_noattr, pr_parked)))
    # The old bug's NDS mechanism: every GT attr '' -> all-NaN attr_err ->
    # cummean returns ones -> class attr_err pinned at 1.0.
    check("F2 all-NaN cummean -> ones",
          np.array_equal(cummean(np.array([np.nan, np.nan])), np.ones(2)))

    q45 = (np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8))  # yaw = 45 deg
    gt_r = mk(velocity=(0, 0))
    pr_r = mk(velocity=(0, 0), detection_score=0.9)
    pr_r.rotation = q45
    check("F2 yaw_diff 45deg", np.isclose(yaw_diff(gt_r, pr_r), np.pi / 4))

    pr_s = mk(velocity=(0, 0), detection_score=0.9)
    pr_s.size = (1.0, 4.0, 1.5)  # half the width -> IoU 0.5
    check("F2 scale_iou half width", np.isclose(scale_iou(gt_r, pr_s), 0.5))


def _synth_gt(nusc, tok, det="car", offset=(5.0, 0.0), num_pts=1,
              translation=None, attribute_name="vehicle.parked"):
    sample = nusc.get("sample", tok)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])["translation"]
    t = (translation if translation is not None
         else (ego[0] + offset[0], ego[1] + offset[1], ego[2]))
    # ego_translation = absolute ego pose: the legacy _filter_by_range
    # convention (it computes the relative diff itself); the corrected path
    # overwrites this via add_center_dist, so both paths see correct range.
    return DetectionBox(sample_token=tok, translation=tuple(map(float, t)),
                        size=(2.0, 4.0, 1.5), rotation=(1, 0, 0, 0),
                        velocity=(0.0, 0.0), num_pts=num_pts,
                        ego_translation=tuple(map(float, ego)),
                        detection_name=det, detection_score=-1.0,
                        attribute_name=attribute_name)


def test_f3_num_pts_filter(nusc, tok):
    print("=== F3: num_pts==0 filter ===", flush=True)
    from nuscenes.eval.detection.config import config_factory
    cfg = config_factory("detection_cvpr_2019")
    eb = EvalBoxes()
    eb.add_boxes(tok, [_synth_gt(nusc, tok, num_pts=0),
                       _synth_gt(nusc, tok, num_pts=1, offset=(8.0, 0.0))])
    out = filter_eval_boxes(nusc, add_center_dist(nusc, eb), cfg.class_range)
    kept = out[tok]
    check("F3 num_pts==0 dropped, num_pts==1 kept",
          len(kept) == 1 and kept[0].num_pts == 1, f"kept={len(kept)}")


def find_bikerack_sample(nusc, max_ego_dist=40.0):
    """A val sample with a bicycle_rack within bicycle class_range of ego, so
    the range filter cannot mask the rack filter in either path."""
    tokens = val_sample_tokens(nusc)
    for tok in tokens:
        sample = nusc.get("sample", tok)
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", sd["ego_pose_token"])["translation"]
        for ann_tok in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            if ann["category_name"] == "static_object.bicycle_rack":
                d = np.linalg.norm(np.asarray(ann["translation"][:2])
                                   - np.asarray(ego[:2]))
                if d < max_ego_dist:
                    return tok, ann
    return None, None


def test_f4_bikerack(nusc):
    print("=== F4: bike-rack filter ===", flush=True)
    from nuscenes.eval.detection.config import config_factory
    from diagnosis_beta_baseline.evaluate_nuscenes import _filter_by_range
    cfg = config_factory("detection_cvpr_2019")
    tok, rack = find_bikerack_sample(nusc)
    if tok is None:
        check("F4 bike-rack sample found", False, "no bicycle_rack in val")
        return
    bike = _synth_gt(nusc, tok, det="bicycle", num_pts=5,
                     translation=rack["translation"], attribute_name="")
    eb = EvalBoxes()
    eb.add_boxes(tok, [bike])
    corrected = filter_eval_boxes(nusc, add_center_dist(nusc, eb),
                                  cfg.class_range)
    check("F4 in-rack bicycle dropped (corrected path)",
          len(corrected[tok]) == 0, f"kept={len(corrected[tok])}")

    eb2 = EvalBoxes()
    bike2 = _synth_gt(nusc, tok, det="bicycle", num_pts=5,
                      translation=rack["translation"], attribute_name="")
    eb2.add_boxes(tok, [bike2])
    legacy = _filter_by_range(eb2, cfg.class_range)
    check("F4 legacy path kept it (documents old defect)",
          len(legacy[tok]) == 1, f"kept={len(legacy[tok])}")


def test_f5_velocity_frame():
    print("=== F5: prediction velocity frame ===", flush=True)
    from method_scannet.streaming.nuscenes_evaluator import _detection_box_dict
    d = _detection_box_dict(
        global_id=1, sample_token="x",
        bbox_lidar=[0, 0, 0, 2, 4, 1.5, 0.0, 1.0, 0.0],
        centroid_global=np.zeros(3), ego_translation=np.zeros(3),
        rotation_global_wxyz=[1, 0, 0, 0], detection_name="car", score=0.5,
        velocity_global=[0.0, 1.0])
    check("F5 velocity_global honored", d["velocity"] == [0.0, 1.0])
    d_legacy = _detection_box_dict(
        global_id=1, sample_token="x",
        bbox_lidar=[0, 0, 0, 2, 4, 1.5, 0.0, 1.0, 0.0],
        centroid_global=np.zeros(3), ego_translation=np.zeros(3),
        rotation_global_wxyz=[1, 0, 0, 0], detection_name="car", score=0.5)
    check("F5 legacy fallback = lidar-frame", d_legacy["velocity"] == [1.0, 0.0])

    # step_sample's rotation formula: ego yaw 90deg, lidar2ego identity,
    # v_lidar=(1,0) -> v_global=(0,1).
    yaw = np.pi / 2
    ego_R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    v_g = ego_R @ (np.eye(3) @ np.array([1.0, 0.0, 0.0]))
    check("F5 rotation formula 90deg", np.allclose(v_g[:2], [0.0, 1.0]))


def main():
    print(f"=== loading NuScenes {VERSION} from {DATAROOT} ===", flush=True)
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
    test_f2_tp_error_semantics()
    test_f5_velocity_frame()
    tok0 = val_sample_tokens(nusc)[0]
    test_f3_num_pts_filter(nusc, tok0)
    test_f4_bikerack(nusc)
    test_f1_gt_parity(nusc)

    print(f"\n=== {('ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES))} ===")
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
