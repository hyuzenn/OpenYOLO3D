"""C1/C2 input-pipeline validation. Runs BEFORE any parity smoke.

Asserts, on the real runtime objects (not on comments):

  1. the already-aggregated cloud no longer passes through LoadPointsFromMultiSweeps
  2. point count is preserved
  3. the 5-channel structure is preserved
  4. the 10 distinct Delta-t values are preserved
  5. no timestamp channel is zeroed
  6. exactly one velocity rotation exists in the emission path
  7. the official CenterPoint inference/decode/NMS path is still in use
  8. C2 emits (w, l, h)

Exits nonzero on the first failure. Writes validate_pipeline.json.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/rintern16/OpenYOLO3D")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "audit/table1_regen_2026-08-28/parity_smoke"

CKPT = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
        "centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_"
        "20220810_011659-04cb3a3b.pth")
CFG = ("/home/rintern16/pretrained/centerpoint_nuscenes/"
       "centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py")

res: dict = {}
fails: list[str] = []


def check(name: str, ok: bool, detail) -> None:
    res[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    if not ok:
        fails.append(name)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    from adapters.centerpoint_proposals import CenterPointProposalGenerator
    from method_scannet.streaming.nuscenes_evaluator import _detection_box_dict

    gen = CenterPointProposalGenerator(config_path=CFG, checkpoint_path=CKPT,
                                       score_threshold=0.0, device="cuda:0")

    # --- 1. the offending transform is gone from the runtime pipeline -----
    check("1_multisweeps_removed",
          "LoadPointsFromMultiSweeps" not in gen.pipeline_steps
          and gen.pipeline_dropped == ["LoadPointsFromMultiSweeps"],
          {"steps": gen.pipeline_steps, "dropped": gen.pipeline_dropped})

    # --- 2/3/4/5. push a known cloud through the REAL runtime pipeline ----
    # All points kept inside point_cloud_range so PointsRangeFilter cannot drop
    # any: a count change then means duplication, not filtering.
    rng = np.random.default_rng(0)
    n_in = 20000
    pts = np.zeros((n_in, 5), dtype=np.float32)
    pts[:, 0] = rng.uniform(-50, 50, n_in)
    pts[:, 1] = rng.uniform(-50, 50, n_in)
    pts[:, 2] = rng.uniform(-4, 2, n_in)
    pts[:, 3] = rng.uniform(0, 255, n_in)
    pts[:, 4] = (rng.integers(0, 10, n_in) * 0.05).astype(np.float32)
    dt_in = np.unique(np.round(pts[:, 4], 4))

    tmp = OUT / "validate_probe.bin"
    pts.tofile(tmp)
    data = gen._pipeline(dict(lidar_points=dict(lidar_path=str(tmp)), timestamp=1,
                              axis_align_matrix=np.eye(4),
                              box_type_3d=gen._box_type_3d,
                              box_mode_3d=gen._box_mode_3d))
    out = data["inputs"]["points"].numpy()
    tmp.unlink()
    dt_out = np.unique(np.round(out[:, 4], 4))

    check("2_point_count_preserved", out.shape[0] == n_in,
          {"in": n_in, "out": int(out.shape[0]),
           "ratio": float(out.shape[0] / n_in)})
    check("3_five_channels", out.shape[1] == 5, {"channels": int(out.shape[1])})
    check("4_dt_values_preserved",
          len(dt_out) == 10 and np.allclose(dt_out, dt_in),
          {"in_distinct": int(len(dt_in)), "out_distinct": int(len(dt_out)),
           "out_values": [float(x) for x in dt_out]})
    check("5_dt_not_zeroed", float(out[:, 4].max()) > 0.0,
          {"max": float(out[:, 4].max()), "mean": float(out[:, 4].mean())})

    # --- 6. exactly one velocity rotation in the emission path ------------
    src = (ROOT / "method_scannet/streaming/nuscenes_native_evaluator.py").read_text()
    n_rot = len(re.findall(r"T_lidar_to_ego\[:3, :3\] @ v_l", src))
    ev_src = (ROOT / "method_scannet/streaming/nuscenes_evaluator.py").read_text()
    # _detection_box_dict must pass velocity_global straight through
    passthrough = "vx, vy = float(velocity_global[0]), float(velocity_global[1])" in ev_src
    check("6_single_velocity_rotation", n_rot == 1 and passthrough,
          {"rotations_in_native_evaluator": n_rot,
           "detection_box_dict_passthrough": passthrough})

    # --- 7. official model path still in use ------------------------------
    from mmdet3d.models.detectors.centerpoint import CenterPoint
    from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
    tc = gen.model.pts_bbox_head.test_cfg
    check("7_official_model_path",
          isinstance(gen.model, CenterPoint)
          and isinstance(gen.model.pts_bbox_head, CenterHead)
          and gen.model.pts_bbox_head.bbox_coder.score_threshold == 0.1
          and tc["nms_type"] == "circle" and tc["post_max_size"] == 83
          and tc["max_per_img"] == 500,
          {"detector": type(gen.model).__name__,
           "head": type(gen.model.pts_bbox_head).__name__,
           "coder_score_threshold": float(gen.model.pts_bbox_head.bbox_coder.score_threshold),
           "nms_type": tc["nms_type"], "post_max_size": tc["post_max_size"],
           "max_per_img": tc["max_per_img"], "min_radius": list(tc["min_radius"])})

    # --- 8. C2 emits (w, l, h) --------------------------------------------
    # bbox_lidar dims are (dx, dy, dz) = (length, width, height) = (4.6, 1.9, 1.7)
    d = _detection_box_dict(global_id=0, sample_token="t",
                            bbox_lidar=[0, 0, 0, 4.6, 1.9, 1.7, 0.0, 0.0, 0.0],
                            centroid_global=np.zeros(3), ego_translation=np.zeros(3),
                            rotation_global_wxyz=[1, 0, 0, 0], detection_name="car",
                            score=0.5, velocity_global=[0.0, 0.0])
    check("8_size_is_wlh", d["size"] == [1.9, 4.6, 1.7], {"size": d["size"]})

    (OUT / "validate_pipeline.json").write_text(json.dumps(res, indent=2))
    if fails:
        print(f"\nVALIDATION FAILED: {fails}", flush=True)
        return 1
    print("\nVALIDATION PASSED (8/8)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
