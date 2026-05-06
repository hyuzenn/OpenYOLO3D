"""γ Stage C orchestrator — CenterPoint proposal sweep.

Loads the W1.5 50 samples (with per-sample lidar→ego transform), instantiates
CenterPointProposalGenerator once, and re-uses that single model across the
8 (score × NMS) combos. nuScenes loader's ``T_lidar_to_ego`` per sample is
recomputed from ``calibrated_sensor`` records.

Acceptance #9 spot-check: at sweep end we re-run β1 best on the same 50
samples to confirm the post-install baseline (M_rate=0.3628, n_cl=182.16)
is still produced — this guards against any state CenterPoint might leave
behind in the env.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import time

import numpy as np
import yaml
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from dataloaders.nuscenes_loader import NuScenesLoader
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from adapters.centerpoint_proposals import CenterPointProposalGenerator
from diagnosis_w1.measurements import match_gt_to_clusters
from diagnosis_gamma.measurements import (
    SampleTimeout, measure_with_timeout, PER_SAMPLE_TIMEOUT_S,
)
from diagnosis_gamma.aggregate import aggregate_gamma, render_all_gamma


GRID = {
    "score_threshold": [0.05, 0.10, 0.20, 0.30],
    "nms_iou_threshold": [0.10, 0.20],
}


CKPT = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"
CONFIG = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"

# β1 post-install baseline
BETA1_POST = {
    "M_rate": 0.3628,
    "L_rate": 0.0426,
    "D_rate": 0.2306,
    "miss_rate": 0.3640,
    "n_clusters": 182.16,
}


def _combo_id(combo: dict) -> str:
    return f"score{combo['score_threshold']:g}_nms{combo['nms_iou_threshold']:g}"


def _build_loader(version, out_dir, label):
    cfg = yaml.safe_load(open("configs/nuscenes_baseline.yaml"))
    cfg["nuscenes"]["version"] = version
    cfg["nuscenes"]["cameras"] = ["CAM_FRONT"]
    tmp = osp.join(out_dir, f"_data_config_{label}.yaml")
    yaml.safe_dump(cfg, open(tmp, "w"))
    print(f"  loading {version} ...")
    t0 = time.time()
    L = NuScenesLoader(config_path=tmp)
    print(f"    ready in {time.time() - t0:.1f}s ({len(L)} samples)")
    return L


def _cache_with_lidar_extrinsic(loader, tokens, source_label):
    tok_idx = {t: i for i, t in enumerate(loader.sample_tokens)}
    cache = {}
    for tok in tokens:
        if tok not in tok_idx:
            continue
        item = loader[tok_idx[tok]]
        sample = loader.nusc.get("sample", tok)
        lidar_sd = loader.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        lidar_cs = loader.nusc.get("calibrated_sensor",
                                     lidar_sd["calibrated_sensor_token"])
        T_lidar_to_ego = transform_matrix(
            translation=lidar_cs["translation"],
            rotation=Quaternion(lidar_cs["rotation"]),
        )
        cache[tok] = {
            "source": source_label,
            "sample_token": tok,
            "pc_ego": item["point_cloud"],
            "gt_boxes": item["gt_boxes"],
            "ego_pose": item["ego_pose"],
            "T_lidar_to_ego": T_lidar_to_ego,
        }
    return cache


def _run_combo(combo, cache, cp_gen, work_root, out_dirs):
    cid = _combo_id(combo)
    combo_dir = osp.join(out_dirs["per_sample_per_config"], cid)
    os.makedirs(combo_dir, exist_ok=True)

    cp_gen.update_thresholds(combo["score_threshold"], combo["nms_iou_threshold"])

    per_sample = []
    timeouts = errors = 0
    for tok in sorted(cache.keys()):
        try:
            rec = measure_with_timeout(cp_gen, cache[tok], work_root)
            with open(osp.join(combo_dir, f"{tok}.json"), "w") as f:
                json.dump(rec, f, indent=2,
                          default=lambda o: float(o) if hasattr(o, "item") else str(o))
            per_sample.append(rec)
        except SampleTimeout:
            timeouts += 1
        except Exception as e:
            errors += 1
            print(f"  [combo {cid} | {tok}] error: {e}")

    if not per_sample:
        return {"combo": combo, "combo_id": cid,
                "n_samples_succeeded": 0, "n_timeouts": timeouts, "n_errors": errors}

    # aggregate
    M_all = [r["M_rate_all"] for r in per_sample]
    L_all = [r["L_rate_all"] for r in per_sample]
    D_all = [r["D_rate_all"] for r in per_sample]
    miss_all = [r["miss_rate_all"] for r in per_sample]
    M_A = [r["M_rate_A"] for r in per_sample if r["n_gt_A"] > 0]
    M_B = [r["M_rate_B"] for r in per_sample if r["n_gt_B"] > 0]
    n_props = [r["n_proposals"] for r in per_sample]
    timings = [r["timing_total_s"] for r in per_sample]

    return {
        "combo": combo,
        "combo_id": cid,
        "n_samples_succeeded": len(per_sample),
        "n_timeouts": timeouts,
        "n_errors": errors,
        "mean_M_rate_all": float(np.mean(M_all)),
        "mean_L_rate_all": float(np.mean(L_all)),
        "mean_D_rate_all": float(np.mean(D_all)),
        "mean_miss_rate_all": float(np.mean(miss_all)),
        "mean_M_rate_A": float(np.mean(M_A)) if M_A else 0.0,
        "mean_M_rate_B": float(np.mean(M_B)) if M_B else 0.0,
        "mean_n_proposals": float(np.mean(n_props)),
        "median_n_proposals": float(np.median(n_props)),
        "median_timing_total_s": float(np.median(timings)),
        "p95_timing_total_s": float(np.percentile(timings, 95)),
        "n_gt_A_total": sum(r["n_gt_A"] for r in per_sample),
        "n_gt_B_total": sum(r["n_gt_B"] for r in per_sample),
    }


def _select_best(summaries):
    valid = [c for c in summaries if c["n_samples_succeeded"] >= 45]
    if not valid:
        return None, "NO_VALID_COMBOS"
    f1 = [c for c in valid if c["mean_M_rate_all"] >= 0.55
          and c["mean_n_proposals"] <= 30
          and c["median_timing_total_s"] < 3.0]
    if f1:
        return max(f1, key=lambda c: c["mean_M_rate_all"]), "TIER1_STRONG"
    f2 = [c for c in valid if c["mean_M_rate_all"] >= 0.45
          and c["median_timing_total_s"] < 5.0]
    if f2:
        return max(f2, key=lambda c: c["mean_M_rate_all"]), "TIER2_PARTIAL"
    return max(valid, key=lambda c: c["mean_M_rate_all"]), "TIER3_FAIL"


def _verify_post_install_beta1(cache) -> dict:
    """Acceptance #9 — re-run β1 50-sample after sweep to confirm post-install
    baseline reproduces (M ≈ 0.3628 ± 0.001).
    """
    ext = PillarForegroundExtractor(pillar_size_xy=(0.5, 0.5),
                                      z_threshold=0.3,
                                      ground_estimation="percentile",
                                      percentile_p=10.0)
    gen = LiDARProposalGenerator(min_cluster_size=3, min_samples=3,
                                  cluster_selection_epsilon=1.0)
    M_rates = []
    n_cls = []
    for tok in sorted(cache.keys()):
        rec = cache[tok]
        fg = ext.extract(rec["pc_ego"])
        if fg["foreground_pcd"].shape[0] == 0:
            M_rates.append(0.0); n_cls.append(0); continue
        h = gen.generate(fg["foreground_pcd"])
        per_gt, cases = match_gt_to_clusters(
            rec["gt_boxes"], rec["ego_pose"],
            fg["foreground_pcd"][:, :3], h["cluster_ids"],
        )
        n_gt = len(rec["gt_boxes"])
        M_rates.append(cases["M"] / n_gt if n_gt else 0.0)
        n_cls.append(int(h["n_clusters"]))
    mean_M = float(np.mean(M_rates))
    mean_n = float(np.mean(n_cls))
    return {
        "mean_M_rate": mean_M,
        "expected_M_rate": BETA1_POST["M_rate"],
        "delta_M": abs(mean_M - BETA1_POST["M_rate"]),
        "mean_n_clusters": mean_n,
        "expected_n_clusters": BETA1_POST["n_clusters"],
        "delta_n_clusters": abs(mean_n - BETA1_POST["n_clusters"]),
        "passes": (abs(mean_M - BETA1_POST["M_rate"]) < 0.001
                    and abs(mean_n - BETA1_POST["n_clusters"]) < 0.5),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-step-a-samples",
                        default="results/diagnosis_step_a/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_gamma")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "per_sample_per_config": osp.join(out_dir, "per_sample_per_config"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(args.use_step_a_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_step_a_samples, osp.join(out_dir, "samples_used.json"))

    n_combos = len(GRID["score_threshold"]) * len(GRID["nms_iou_threshold"])
    print("=" * 60)
    print(f"γ Stage C — CenterPoint proposal sweep")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens) + len(trainval_tokens)} samples")
    print(f"  sweep: {n_combos} combos (4 score × 2 NMS)")
    print("=" * 60)

    mini_loader = _build_loader("v1.0-mini", out_dir, "mini_g")
    mini_cache = _cache_with_lidar_extrinsic(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", out_dir, "trainval_g")
    trainval_cache = _cache_with_lidar_extrinsic(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")
    gc.collect()

    print("\nInitializing CenterPoint ...")
    t0 = time.time()
    cp_gen = CenterPointProposalGenerator(
        config_path=CONFIG, checkpoint_path=CKPT, device="cuda:0",
    )
    print(f"  CenterPoint ready in {time.time() - t0:.1f}s")

    work_root = tempfile.mkdtemp(prefix="gamma_bin_")
    try:
        combos = []
        for sc, ni in itertools.product(GRID["score_threshold"], GRID["nms_iou_threshold"]):
            combos.append({"score_threshold": sc, "nms_iou_threshold": ni})

        print(f"\n=== sweep: {len(combos)} combos × {len(cache)} samples ===")
        t_sweep_start = time.time()
        summaries = []
        for i, combo in enumerate(combos):
            cid = _combo_id(combo)
            t_c = time.time()
            s = _run_combo(combo, cache, cp_gen, work_root, out_dirs)
            elapsed = time.time() - t_c
            summaries.append(s)
            if s["n_samples_succeeded"] == 0:
                print(f"  [{i+1}/{len(combos)}] {cid}: FAILED")
            else:
                print(f"  [{i+1}/{len(combos)}] {cid}: "
                      f"M={s['mean_M_rate_all']*100:.1f}%, "
                      f"M_A={s['mean_M_rate_A']*100:.1f}%, M_B={s['mean_M_rate_B']*100:.1f}%, "
                      f"n_prop={s['mean_n_proposals']:.1f}, "
                      f"miss={s['mean_miss_rate_all']*100:.1f}%, "
                      f"t={s['median_timing_total_s']:.2f}s "
                      f"({s['n_samples_succeeded']}/{len(cache)} ok, {elapsed:.1f}s)")
        print(f"  sweep finished in {time.time() - t_sweep_start:.1f}s")

        best, verdict = _select_best(summaries)
        sweep_record = {
            "n_samples": len(cache),
            "grid": GRID,
            "n_combos": len(combos),
            "results": summaries,
            "selection_verdict": verdict,
            "best": best,
            "selection_rule": ("tier1: M≥0.55, n_prop≤30, t<3.0s; "
                               "tier2: M≥0.45, t<5.0s; tier3 (FAIL): argmax(M)"),
        }
        with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
            json.dump(sweep_record, f, indent=2)

        if best is None:
            print("  No valid combo. Skipping aggregate.")
        else:
            print(f"\n  selection verdict: {verdict}")
            print(f"  best: {best['combo_id']} "
                  f"(M={best['mean_M_rate_all']*100:.1f}%, "
                  f"M_A={best['mean_M_rate_A']*100:.1f}%, "
                  f"M_B={best['mean_M_rate_B']*100:.1f}%, "
                  f"n_prop={best['mean_n_proposals']:.1f}, "
                  f"t={best['median_timing_total_s']:.2f}s)")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        del cp_gen
        gc.collect()

    # post-install β1 regression (acceptance #9)
    print("\nPost-sweep β1 regression check ...")
    reg = _verify_post_install_beta1(cache)
    print(f"  M={reg['mean_M_rate']:.4f} (expected {reg['expected_M_rate']:.4f}, "
          f"Δ={reg['delta_M']:.6f}), n_cl={reg['mean_n_clusters']:.4f} "
          f"(expected {reg['expected_n_clusters']:.4f}, Δ={reg['delta_n_clusters']:.6f}) "
          f"→ {'PASS' if reg['passes'] else 'FAIL'}")
    with open(osp.join(out_dir, "beta1_post_install_regression.json"), "w") as f:
        json.dump(reg, f, indent=2)

    if best is not None:
        agg = aggregate_gamma(summaries, best, sweep_record, reg, cache, out_dirs)
        with open(osp.join(out_dir, "aggregate.json"), "w") as f:
            json.dump(agg, f, indent=2)
        render_all_gamma(summaries, best, sweep_record, reg, agg,
                          out_dirs["figures"], osp.join(out_dir, "report.md"))
        print(f"  → {out_dir}/aggregate.json")
        print(f"  → {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
