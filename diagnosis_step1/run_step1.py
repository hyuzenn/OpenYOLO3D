"""Step 1 orchestrator — Mask3D vs HDBSCAN-best vs Hybrid (union) on the
W1.5 50-sample set.

Single PBS job. Loaders + Mask3D + HDBSCAN are paid once, samples are
processed sequentially with per-sample timeouts. No method code is changed;
this is measurement only.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import signal
import sys
import tempfile
import time
import traceback

import numpy as np
import yaml

from diagnosis_w1.run_clustering_check import _build_loader
from adapters.nuscenes_to_openyolo3d import CAMERA
from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_step1.matching import (
    match_gt_to_instances, cluster_ids_to_masks,
)
from diagnosis_step1.mask3d_runner import Mask3DProposalRunner
from diagnosis_step1.hybrid_simulator import simulate_hybrid
from diagnosis_step1.aggregate import aggregate_step1, render_all_step1


MASK3D_TIMEOUT_S = 60
HDBSCAN_TIMEOUT_S = 30


class StepTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise StepTimeout()


def _set_alarm(seconds: int):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)


def _clear_alarm():
    signal.alarm(0)


def _cache_samples_with_image(loader, tokens, source_label):
    """Like diagnosis_w1._cache_samples but keeps the CAM_FRONT image + intrinsic +
    extrinsic so Mask3D can colourise its PLY identically to the Tier-1 / smoke-test
    pipeline. ScanNet-trained Mask3D is sensitive to RGB normalisation; uniform-gray
    PLYs were observed to produce zero post-filter proposals.
    """
    tok_to_idx = {tok: i for i, tok in enumerate(loader.sample_tokens)}
    cache = {}
    for tok in tokens:
        if tok not in tok_to_idx:
            continue
        item = loader[tok_to_idx[tok]]
        cache[tok] = {
            "source": source_label,
            "pc_ego": item["point_cloud"],
            "gt_boxes": item["gt_boxes"],
            "ego_pose": item["ego_pose"],
            "sample_token": tok,
            "cam_front_image": item["images"][CAMERA],
            "cam_front_K": item["intrinsics"][CAMERA],
            "cam_front_T_to_ego": item["cam_to_ego"][CAMERA],
        }
    return cache


# HDBSCAN-best config from W1.5 Phase B
HDBSCAN_BEST = {
    "min_cluster_size": 3,
    "min_samples": 3,
    "cluster_selection_epsilon": 1.0,
    "ground_filter": "z_threshold",
    "ground_z_max": -1.4,
}


def _process_sample(tok, rec, mask3d_runner, hdbscan_gen, work_root, out_dirs):
    pc = rec["pc_ego"]
    pc_xyz = pc[:, :3]
    sample_payload = {"sample_token": tok, "source": rec["source"]}

    # ---- Mask3D ----
    _set_alarm(MASK3D_TIMEOUT_S)
    try:
        m3d = mask3d_runner.run(
            pc_xyz, tok,
            image=rec.get("cam_front_image"),
            K=rec.get("cam_front_K"),
            T_cam_to_ego=rec.get("cam_front_T_to_ego"),
        )
        _clear_alarm()
        m3d_masks = m3d["masks"]
        per_gt_m, cases_m = match_gt_to_instances(
            rec["gt_boxes"], rec["ego_pose"], pc_xyz, m3d_masks,
        )
        n_gt = len(rec["gt_boxes"])
        m3d_payload = {
            **sample_payload,
            "n_instances": m3d["n_instances"],
            "scores": m3d["scores"].tolist(),
            "timing": m3d["timing"],
            "n_gt_total": n_gt,
            "case_counts": cases_m,
            "M_rate": (cases_m["M"] / n_gt) if n_gt else 0.0,
            "L_rate": (cases_m["L"] / n_gt) if n_gt else 0.0,
            "D_rate": (cases_m["D"] / n_gt) if n_gt else 0.0,
            "miss_rate": (cases_m["miss"] / n_gt) if n_gt else 0.0,
            "per_gt": per_gt_m,
            "status": "ok",
        }
    except StepTimeout:
        _clear_alarm()
        m3d_payload = {**sample_payload, "status": "timeout", "stage": "mask3d"}
        m3d_masks = np.zeros((pc_xyz.shape[0], 0), dtype=bool)
    finally:
        if "ply_path" in (m3d if isinstance(m3d, dict) else {}):
            mask3d_runner.cleanup_ply(m3d["ply_path"])

    with open(osp.join(out_dirs["mask3d"], f"{tok}.json"), "w") as f:
        json.dump(m3d_payload, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    # ---- HDBSCAN-best ----
    _set_alarm(HDBSCAN_TIMEOUT_S)
    try:
        h_out = hdbscan_gen.generate(pc)
        _clear_alarm()
        h_masks = cluster_ids_to_masks(h_out["cluster_ids"], h_out["n_clusters"])
        per_gt_h, cases_h = match_gt_to_instances(
            rec["gt_boxes"], rec["ego_pose"], pc_xyz, h_masks,
        )
        n_gt = len(rec["gt_boxes"])
        h_payload = {
            **sample_payload,
            "config": hdbscan_gen.config_dict,
            "n_instances": int(h_out["n_clusters"]),
            "noise_ratio": float(h_out["noise_ratio"]),
            "ground_filtered_ratio": float(h_out["ground_filtered_ratio"]),
            "timing": dict(h_out["timing"]),
            "n_gt_total": n_gt,
            "case_counts": cases_h,
            "M_rate": (cases_h["M"] / n_gt) if n_gt else 0.0,
            "L_rate": (cases_h["L"] / n_gt) if n_gt else 0.0,
            "D_rate": (cases_h["D"] / n_gt) if n_gt else 0.0,
            "miss_rate": (cases_h["miss"] / n_gt) if n_gt else 0.0,
            "per_gt": per_gt_h,
            "status": "ok",
        }
    except StepTimeout:
        _clear_alarm()
        h_payload = {**sample_payload, "status": "timeout", "stage": "hdbscan"}
        h_masks = np.zeros((pc_xyz.shape[0], 0), dtype=bool)

    with open(osp.join(out_dirs["hdbscan"], f"{tok}.json"), "w") as f:
        json.dump(h_payload, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    # ---- Hybrid simulation (always — calculation only, no model invocation) ----
    hyb = simulate_hybrid(rec["gt_boxes"], rec["ego_pose"], pc_xyz, m3d_masks, h_masks)
    n_gt = len(rec["gt_boxes"])
    hyb_payload = {
        **sample_payload,
        "n_proposals_mask3d": hyb["n_proposals_mask3d"],
        "n_proposals_hdbscan": hyb["n_proposals_hdbscan"],
        "n_proposals_total": hyb["n_proposals_total"],
        "n_gt_total": n_gt,
        "counts": hyb["counts"],
        "rates": hyb["rates"],
        "union_case_counts": hyb["union_case_counts"],
        "union_M_rate": (hyb["union_case_counts"]["M"] / n_gt) if n_gt else 0.0,
        "union_miss_rate": (hyb["union_case_counts"]["miss"] / n_gt) if n_gt else 0.0,
        "per_gt": hyb["per_gt"],
        "status": "ok",
    }
    with open(osp.join(out_dirs["hybrid"], f"{tok}.json"), "w") as f:
        json.dump(hyb_payload, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))

    return {"mask3d": m3d_payload, "hdbscan": h_payload, "hybrid": hyb_payload}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-w1-5-samples",
                        default="results/w1_5_diagnostic_sweep/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--openyolo-config", default="configs/openyolo3d_nuscenes.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_step1")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "mask3d": osp.join(out_dir, "per_sample", "mask3d"),
        "hdbscan": osp.join(out_dir, "per_sample", "hdbscan"),
        "hybrid": osp.join(out_dir, "per_sample", "hybrid"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # ---- copy W1.5 samples_used.json (provenance) ----
    with open(args.use_w1_5_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_w1_5_samples, osp.join(out_dir, "samples_used.json"))
    print("=" * 60)
    print(f"Step 1 — Mask3D vs HDBSCAN-best vs Hybrid")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens)+len(trainval_tokens)} samples (from W1.5)")
    print("=" * 60)

    # ---- load samples ----
    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini_step1")
    mini_cache = _cache_samples_with_image(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval_step1")
    trainval_cache = _cache_samples_with_image(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")

    # ---- init Mask3D + HDBSCAN ----
    print("\nInitializing Mask3D ...")
    with open(args.openyolo_config) as f:
        oy3d_cfg = yaml.safe_load(f)
    work_root = tempfile.mkdtemp(prefix="step1_ply_")
    t0 = time.time()
    mask3d_runner = Mask3DProposalRunner(oy3d_cfg, work_root)
    print(f"  Mask3D ready in {time.time() - t0:.1f}s (work_root={work_root})")

    hdbscan_gen = LiDARProposalGenerator(**HDBSCAN_BEST)

    # ---- per-sample loop ----
    succeeded_m, succeeded_h, succeeded_y = [], [], []
    failed_m, failed_h = [], []
    for i, tok in enumerate(sorted(cache.keys())):
        rec = cache[tok]
        print(f"\n[{i+1}/{len(cache)}] {tok} ({rec['source']})")
        try:
            t_s = time.time()
            payloads = _process_sample(tok, rec, mask3d_runner, hdbscan_gen,
                                       work_root, out_dirs)
            elapsed = time.time() - t_s
            m = payloads["mask3d"]
            h = payloads["hdbscan"]
            y = payloads["hybrid"]
            if m.get("status") == "ok":
                succeeded_m.append(m)
            else:
                failed_m.append(m)
            if h.get("status") == "ok":
                succeeded_h.append(h)
            else:
                failed_h.append(h)
            succeeded_y.append(y)
            m_n = m.get("n_instances", 0)
            h_n = h.get("n_instances", 0)
            print(f"  ✓ {elapsed:.1f}s — Mask3D {m_n} ({m.get('M_rate',0)*100:.1f}% M), "
                  f"HDBSCAN {h_n} ({h.get('M_rate',0)*100:.1f}% M), "
                  f"Hybrid covered={y['rates']['covered_by_either']*100:.1f}%")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ✗ {e}\n{tb}")

    shutil.rmtree(work_root, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"Mask3D OK: {len(succeeded_m)}/{len(cache)},  HDBSCAN OK: {len(succeeded_h)}/{len(cache)}")
    print("=" * 60)

    # ---- aggregate + report ----
    if min(len(succeeded_m), len(succeeded_h)) >= 45:
        agg = aggregate_step1(succeeded_m, succeeded_h, succeeded_y)
        with open(osp.join(out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        render_all_step1(succeeded_m, succeeded_h, succeeded_y, agg,
                         failed_m, failed_h, out_dirs["figures"],
                         osp.join(out_dir, "report.md"))
        print(f"  → {out_dir}/aggregate.json")
        print(f"  → {out_dir}/report.md")
    else:
        msg = (f"Below ≥45/50 acceptance bar — Mask3D OK={len(succeeded_m)}, "
               f"HDBSCAN OK={len(succeeded_h)}.")
        print("  " + msg)
        with open(osp.join(out_dir, "report.md"), "w") as f:
            f.write(f"# Step 1 — INCOMPLETE\n\n{msg}\n")

    with open(osp.join(out_dir, "_failed.json"), "w") as f:
        json.dump({"mask3d": failed_m, "hdbscan": failed_h}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
