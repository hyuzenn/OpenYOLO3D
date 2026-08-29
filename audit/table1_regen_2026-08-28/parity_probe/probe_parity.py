"""CPU-only, model-free parity probe for the CenterPoint adapter audit.

READ-ONLY. Runs NO inference, loads NO checkpoint, touches NO existing file.
It answers three questions with measurement instead of source-reading alone:

  P1  What does mmdet3d's own test pipeline do to a .bin handed to
      `inference_detector`?  (point count, timestamp channel)
  P2  Do the cached velocities look like a model that never saw a timestamp?
  P3  Is the 184.7 vs 82.7 boxes/sample gap explained by the per-class range
      filter that mmdet3d applies in `lidar_nusc_box_to_global`, or is there a
      genuine excess of detections?

Writes audit/table1_regen_2026-08-28/parity_probe/probe_results.json
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/rintern16/OpenYOLO3D")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "audit/table1_regen_2026-08-28/parity_probe"
CACHE = ROOT / "results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity"
ANCHOR = ROOT / "audit/official_centerpoint_ref/nusc_submission/pred_instances_3d/results_nusc.json"
CFG = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
       "centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")

# nuScenes detection_cvpr_2019 class_range (devkit config, quoted for the probe
# only -- the evaluator itself reads it from the devkit).
CLASS_RANGE = {"car": 50, "truck": 50, "bus": 50, "trailer": 50,
               "construction_vehicle": 50, "pedestrian": 40, "motorcycle": 40,
               "bicycle": 40, "traffic_cone": 30, "barrier": 30}

res = {}

# ---------------------------------------------------------------- P1
# Synthetic 5-channel cloud with a KNOWN, non-zero timestamp channel, pushed
# through the *unmodified* cfg.test_dataloader.dataset.pipeline exactly as
# mmdet3d.apis.inference_detector composes it. No model is built.
def p1():
    from copy import deepcopy
    from mmengine.config import Config
    from mmengine.dataset import Compose
    from mmengine.registry import init_default_scope
    from mmdet3d.structures import get_box_type

    init_default_scope("mmdet3d")
    cfg = Config.fromfile(CFG)
    pipeline = Compose(deepcopy(cfg.test_dataloader.dataset.pipeline))
    box_type_3d, box_mode_3d = get_box_type(cfg.test_dataloader.dataset.box_type_3d)

    rng = np.random.default_rng(0)
    n_in = 20000
    pts = np.zeros((n_in, 5), dtype=np.float32)
    pts[:, :3] = rng.uniform(-40, 40, size=(n_in, 3)).astype(np.float32)
    pts[:, 2] = rng.uniform(-3, 1, size=n_in).astype(np.float32)
    pts[:, 3] = rng.uniform(0, 255, size=n_in).astype(np.float32)
    # the Delta-t channel the adapter fills in: 10 distinct sweep lags
    pts[:, 4] = (rng.integers(0, 10, size=n_in) * 0.05).astype(np.float32)
    dt_in = pts[:, 4].copy()

    tmp = OUT / "synthetic_probe.bin"
    pts.tofile(tmp)

    data = pipeline(dict(lidar_points=dict(lidar_path=str(tmp)), timestamp=1,
                         axis_align_matrix=np.eye(4),
                         box_type_3d=box_type_3d, box_mode_3d=box_mode_3d))
    out = data["inputs"]["points"].numpy()
    tmp.unlink()

    return {
        "n_points_written": int(n_in),
        "n_points_after_pipeline": int(out.shape[0]),
        "duplication_factor": float(out.shape[0] / n_in),
        "dt_channel_in": {"min": float(dt_in.min()), "max": float(dt_in.max()),
                          "n_distinct": int(len(np.unique(np.round(dt_in, 4))))},
        "dt_channel_out": {"min": float(out[:, 4].min()), "max": float(out[:, 4].max()),
                           "n_distinct": int(len(np.unique(np.round(out[:, 4], 4))))},
        "pipeline": [t["type"] for t in cfg.test_dataloader.dataset.pipeline],
    }


# ---------------------------------------------------------------- P2 / P3
def p23():
    pkls = sorted(CACHE.glob("*.pkl"))
    tokens = [p.stem for p in pkls]
    assert len(tokens) == 6019, len(tokens)

    # --- anchor side -------------------------------------------------
    anchor = json.load(open(ANCHOR))["results"]
    a_count = {t: len(v) for t, v in anchor.items()}
    a_speed, a_score, a_cls = [], [], defaultdict(int)
    for t, boxes in anchor.items():
        for b in boxes:
            v = b["velocity"]
            a_speed.append(float(np.hypot(v[0], v[1])))
            a_score.append(float(b["detection_score"]))
            a_cls[b["detection_name"]] += 1
    a_speed = np.asarray(a_speed)
    a_score = np.asarray(a_score)

    # --- cache side --------------------------------------------------
    c_raw, c_ranged = {}, {}
    c_speed, c_score, c_cls, c_cls_ranged = [], [], defaultdict(int), defaultdict(int)
    for p in pkls:
        with open(p, "rb") as f:
            props = pickle.load(f)
        c_raw[p.stem] = len(props)
        kept = 0
        for pr in props:
            b = pr["bbox_lidar"]
            name = pr["cls_name"]
            c_cls[name] += 1
            c_score.append(float(pr["score"]))
            if len(b) >= 9:
                c_speed.append(float(np.hypot(b[7], b[8])))
            # per-class range filter, ego frame, exactly as
            # lidar_nusc_box_to_global does (radius on the ego-frame centre)
            ce = pr["centroid_ego"]
            if np.hypot(ce[0], ce[1]) <= CLASS_RANGE.get(name, 0.0):
                kept += 1
                c_cls_ranged[name] += 1
        c_ranged[p.stem] = kept
    c_speed = np.asarray(c_speed)
    c_score = np.asarray(c_score)

    n = len(tokens)
    common = [t for t in tokens if t in a_count]

    def q(a):
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max()),
                "min": float(a.min()), "frac_below_0.2ms": float((a < 0.2).mean())}

    return {
        "n_tokens_cache": n,
        "n_tokens_anchor": len(a_count),
        "n_tokens_common": len(common),
        "counts": {
            "cache_raw_total": int(sum(c_raw.values())),
            "cache_raw_per_sample": float(sum(c_raw.values()) / n),
            "cache_after_class_range_total": int(sum(c_ranged.values())),
            "cache_after_class_range_per_sample": float(sum(c_ranged.values()) / n),
            "anchor_total": int(sum(a_count.values())),
            "anchor_per_sample": float(sum(a_count.values()) / len(a_count)),
            "excess_ratio_after_range_filter":
                float(sum(c_ranged.values()) / sum(a_count.values())),
        },
        "score": {"cache": q(c_score), "anchor": q(a_score),
                  "cache_min_is_head_threshold": float(c_score.min())},
        "speed_mps": {"cache": q(c_speed), "anchor": q(a_speed)},
        "per_class_counts": {
            k: {"cache_raw": c_cls.get(k, 0),
                "cache_ranged": c_cls_ranged.get(k, 0),
                "anchor": a_cls.get(k, 0)}
            for k in sorted(set(list(c_cls) + list(a_cls)))
        },
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res["P1_test_pipeline_on_a_bin"] = p1()
    print("P1 done", flush=True)
    res["P23_cache_vs_anchor"] = p23()
    print("P23 done", flush=True)
    (OUT / "probe_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
