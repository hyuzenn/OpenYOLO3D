"""α — Hybrid Simulation orchestrator.

Per α spec: 50 Step-A samples, β1 (best config) + γ (best config), 4 union
strategy families (10 sweep combos total). Inference is paid once per
sample; each sample's β1/γ artifacts are then re-used across all 10 strategies.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import os.path as osp
import shutil
import sys
import tempfile
import time
import random

import numpy as np
import yaml
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix

from dataloaders.nuscenes_loader import NuScenesLoader
from preprocessing.pillar_foreground import PillarForegroundExtractor
from adapters.lidar_proposals import LiDARProposalGenerator
from adapters.centerpoint_proposals import CenterPointProposalGenerator
from diagnosis_alpha.measurements import (
    BETA1_PILLAR, BETA1_HDBSCAN, GAMMA_THRESHOLDS,
    PER_SAMPLE_TIMEOUT_S, SampleTimeout,
    _set_alarm, _clear_alarm,
    run_sources_once, build_per_sample_strategy_record,
)
from diagnosis_alpha.union_strategies import (
    STRATEGY_GRID, combo_id, apply_strategy,
)
from diagnosis_alpha.aggregate import (
    aggregate_alpha, render_all_alpha,
)


# γ pretrained CenterPoint weights/config — same as diagnosis_gamma
CKPT = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth"
CONFIG = "/home/rintern16/pretrained/centerpoint_nuscenes/centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"

# Saved baselines (from diagnosis_beta1/aggregate.json + diagnosis_gamma/aggregate.json)
BETA1_BASELINE = {"M_rate": 0.3611923834280709,    # 0.3612 to 4 dp
                   "n_clusters_mean": 182.12}
GAMMA_BASELINE = {"M_rate_all": 0.351888136708548,  # 0.3519 to 4 dp
                   "n_proposals_mean": 65.84}


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


def _per_sample_dir(out_dirs, combo_id_str):
    d = osp.join(out_dirs["per_sample_per_strategy"], combo_id_str)
    os.makedirs(d, exist_ok=True)
    return d


def _summary_for_strategy(combo, records, n_samples):
    """Aggregate per-sample records for one strategy combo."""
    if not records:
        return {
            "combo": combo, "combo_id": combo["combo_id"],
            "strategy": combo["strategy"], "params": combo["params"],
            "n_samples_succeeded": 0,
        }
    M = [r["M_rate"] for r in records]
    L = [r["L_rate"] for r in records]
    D = [r["D_rate"] for r in records]
    miss = [r["miss_rate"] for r in records]
    n_prop = [r["n_proposals_total"] for r in records]
    n_b = [r["n_proposals_beta1"] for r in records]
    n_g = [r["n_proposals_gamma"] for r in records]
    timing = [r["strategy_timing_s"] for r in records]

    # coverage rollup (sum over GT counts)
    cov_sum = {"both": 0, "beta1_only": 0, "gamma_only": 0, "neither": 0}
    n_gt_total = 0
    for r in records:
        for k in cov_sum:
            cov_sum[k] += r["coverage"][k]
        n_gt_total += r["n_gt_total"]
    cov_rates = {k: (v / n_gt_total) if n_gt_total else 0.0 for k, v in cov_sum.items()}

    return {
        "combo": combo, "combo_id": combo["combo_id"],
        "strategy": combo["strategy"], "params": combo["params"],
        "n_samples_succeeded": len(records),
        "n_samples_total": n_samples,
        "mean_M_rate": float(np.mean(M)),
        "mean_L_rate": float(np.mean(L)),
        "mean_D_rate": float(np.mean(D)),
        "mean_miss_rate": float(np.mean(miss)),
        "mean_n_proposals": float(np.mean(n_prop)),
        "median_n_proposals": float(np.median(n_prop)),
        "mean_n_proposals_beta1": float(np.mean(n_b)),
        "mean_n_proposals_gamma": float(np.mean(n_g)),
        "median_strategy_timing_s": float(np.median(timing)),
        "p95_strategy_timing_s": float(np.percentile(timing, 95)) if len(timing) > 1 else float(timing[0]),
        "coverage_counts": cov_sum,
        "coverage_rates": cov_rates,
        "n_gt_total": int(n_gt_total),
    }


def _select_best(summaries):
    """4-tier selection — STRONG / PARTIAL / MARGINAL / FAIL.

    Tier rules (per α spec § Best 선택):
      STRONG    M ≥ 0.45 AND n_prop ≤ 100 AND median_strategy_timing < 3.0 s
      PARTIAL   M ≥ 0.40 AND n_prop ≤ 200
      MARGINAL  M ∈ [0.36, 0.40)
      FAIL      M < 0.36
    """
    valid = [c for c in summaries if c["n_samples_succeeded"] >= 45]
    if not valid:
        return None, "NO_VALID_COMBOS"
    t1 = [c for c in valid if c["mean_M_rate"] >= 0.45
          and c["mean_n_proposals"] <= 100
          and c["median_strategy_timing_s"] < 3.0]
    if t1:
        return max(t1, key=lambda c: c["mean_M_rate"]), "TIER1_STRONG"
    t2 = [c for c in valid if c["mean_M_rate"] >= 0.40
          and c["mean_n_proposals"] <= 200]
    if t2:
        return max(t2, key=lambda c: c["mean_M_rate"]), "TIER2_PARTIAL"
    best_overall = max(valid, key=lambda c: c["mean_M_rate"])
    if best_overall["mean_M_rate"] >= 0.36:
        return best_overall, "TIER3_MARGINAL"
    return best_overall, "TIER4_FAIL"


def _regression_check(sample_packs, baseline_files):
    """5-sample spot check: β1 alone + γ alone vs saved per-sample JSONs.

    For each randomly chosen sample, compare M_rate, L_rate, D_rate, miss_rate
    to 4 decimals against the saved β1 (pillar0.5x0.5_zthr0.3_percentile) and
    γ (score0.2_nms0.1) per-sample records.
    """
    rng = random.Random(0xA1B2C3)
    tokens = sorted(sample_packs.keys())
    rng.shuffle(tokens)
    chosen = tokens[:5]

    rows = []
    all_pass = True
    for tok in chosen:
        sp = sample_packs[tok]
        b_alone = sp["beta1_alone"]
        g_alone = sp["gamma_alone"]

        b_path = osp.join(baseline_files["beta1_dir"], f"{tok}.json")
        g_path = osp.join(baseline_files["gamma_dir"], f"{tok}.json")
        b_saved = json.load(open(b_path)) if osp.exists(b_path) else None
        g_saved = json.load(open(g_path)) if osp.exists(g_path) else None

        row = {"sample_token": tok}
        if b_saved is not None:
            for k in ("M_rate", "L_rate", "D_rate", "miss_rate"):
                cur = round(b_alone[k], 4)
                exp = round(b_saved[k], 4)
                row[f"beta1_{k}"] = cur
                row[f"beta1_{k}_saved"] = exp
                row[f"beta1_{k}_match"] = (cur == exp)
                if not row[f"beta1_{k}_match"]:
                    all_pass = False
        else:
            row["beta1_missing_saved"] = True
            all_pass = False
        if g_saved is not None:
            for k in ("M_rate_all", "L_rate_all", "D_rate_all", "miss_rate_all"):
                cur = round(g_alone[k.replace("_all", "")], 4)
                exp = round(g_saved[k], 4)
                row[f"gamma_{k}"] = cur
                row[f"gamma_{k}_saved"] = exp
                row[f"gamma_{k}_match"] = (cur == exp)
                if not row[f"gamma_{k}_match"]:
                    all_pass = False
        else:
            row["gamma_missing_saved"] = True
            all_pass = False
        rows.append(row)

    return {"chosen_tokens": chosen, "rows": rows, "all_pass": bool(all_pass)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-step-a-samples",
                        default="results/diagnosis_step_a/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_alpha")
    parser.add_argument("--beta1-baseline-dir",
                        default="results/diagnosis_beta1/per_sample_per_config/pillar0.5x0.5_zthr0.3_percentile")
    parser.add_argument("--gamma-baseline-dir",
                        default="results/diagnosis_gamma/per_sample_per_config/score0.2_nms0.1")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "per_sample_per_strategy": osp.join(out_dir, "per_sample_per_strategy"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # ---- samples ----
    with open(args.use_step_a_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_step_a_samples, osp.join(out_dir, "samples_used.json"))

    print("=" * 60)
    print("α — Hybrid Simulation (β1 ∪ γ)")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens) + len(trainval_tokens)} samples (Step A set)")
    n_strategies = len(STRATEGY_GRID)
    print(f"  strategies: {n_strategies} combos "
          f"({sum(1 for c in STRATEGY_GRID if c['strategy']=='naive')} naive, "
          f"{sum(1 for c in STRATEGY_GRID if c['strategy']=='distance_aware')} distance-aware, "
          f"{sum(1 for c in STRATEGY_GRID if c['strategy']=='score_weighted')} score-weighted, "
          f"{sum(1 for c in STRATEGY_GRID if c['strategy']=='spatial_nms')} spatial-NMS)")
    print(f"  β1 config (locked): {BETA1_PILLAR}, HDBSCAN={BETA1_HDBSCAN}")
    print(f"  γ config (locked):  {GAMMA_THRESHOLDS}")
    print("=" * 60)

    # ---- caches ----
    mini_loader = _build_loader("v1.0-mini", out_dir, "mini_a")
    mini_cache = _cache_with_lidar_extrinsic(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", out_dir, "trainval_a")
    trainval_cache = _cache_with_lidar_extrinsic(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")
    gc.collect()

    # ---- generators ----
    extractor = PillarForegroundExtractor(**BETA1_PILLAR)
    hdbscan_gen = LiDARProposalGenerator(**BETA1_HDBSCAN)

    print("\nInitializing CenterPoint ...")
    t0 = time.time()
    cp_gen = CenterPointProposalGenerator(
        config_path=CONFIG, checkpoint_path=CKPT,
        score_threshold=GAMMA_THRESHOLDS["score_threshold"],
        nms_iou_threshold=GAMMA_THRESHOLDS["nms_iou_threshold"],
        device="cuda:0",
    )
    print(f"  CenterPoint ready in {time.time() - t0:.1f}s")

    work_root = tempfile.mkdtemp(prefix="alpha_bin_")
    try:
        # ---- pass 1: per-sample β1 + γ inference (paid once) ----
        sample_packs = {}
        timeouts = errors = 0
        print(f"\n=== pass 1: per-sample β1+γ inference ({len(cache)} samples) ===")
        t_p1 = time.time()
        for tok in sorted(cache.keys()):
            try:
                _set_alarm(PER_SAMPLE_TIMEOUT_S)
                pack = run_sources_once(extractor, hdbscan_gen, cp_gen,
                                         cache[tok], work_root)
                _clear_alarm()
                sample_packs[tok] = pack
            except SampleTimeout:
                _clear_alarm()
                timeouts += 1
                print(f"  [{tok}] TIMEOUT")
            except Exception as e:
                _clear_alarm()
                errors += 1
                print(f"  [{tok}] ERROR: {e}")
        print(f"  pass 1 done in {time.time() - t_p1:.1f}s "
              f"(ok={len(sample_packs)}, timeouts={timeouts}, errors={errors})")

        # ---- pass 2: apply each strategy per sample ----
        per_strategy_records = {combo_id(c): [] for c in STRATEGY_GRID}
        sweep_combos = []
        print(f"\n=== pass 2: {len(STRATEGY_GRID)} strategies × {len(sample_packs)} samples ===")
        t_p2 = time.time()
        for c in STRATEGY_GRID:
            cid = combo_id(c)
            this_combo = {**c, "combo_id": cid}
            sweep_combos.append(this_combo)
            d = _per_sample_dir(out_dirs, cid)
            n_ok = 0
            ts0 = time.time()
            for tok in sorted(sample_packs.keys()):
                pack = sample_packs[tok]
                try:
                    t_sp0 = time.perf_counter()
                    out = apply_strategy(this_combo, pack["artifacts"])
                    t_sp1 = time.perf_counter()
                    rec = build_per_sample_strategy_record(
                        this_combo, out, pack, t_sp1 - t_sp0,
                    )
                    with open(osp.join(d, f"{tok}.json"), "w") as f:
                        json.dump(rec, f, indent=2,
                                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
                    per_strategy_records[cid].append(rec)
                    n_ok += 1
                except Exception as e:
                    print(f"  [{cid} | {tok}] strategy error: {e}")
            print(f"  [{cid}] {n_ok}/{len(sample_packs)} ok in {time.time() - ts0:.1f}s")
        print(f"  pass 2 done in {time.time() - t_p2:.1f}s")

        # ---- summaries ----
        summaries = []
        for c in sweep_combos:
            cid = c["combo_id"]
            summaries.append(_summary_for_strategy(c, per_strategy_records[cid], len(sample_packs)))

        best, verdict = _select_best(summaries)
        sweep_record = {
            "n_samples": len(sample_packs),
            "n_combos": len(STRATEGY_GRID),
            "strategy_grid": STRATEGY_GRID,
            "selection_verdict": verdict,
            "best_combo_id": best["combo_id"] if best else None,
            "selection_rule": ("STRONG ≥0.45 AND n_prop≤100 AND t<3.0s; "
                                "PARTIAL ≥0.40 AND n_prop≤200; "
                                "MARGINAL ≥0.36; FAIL <0.36"),
            "results": summaries,
            "beta1_locked_config": {"pillar": BETA1_PILLAR, "hdbscan": BETA1_HDBSCAN},
            "gamma_locked_config": GAMMA_THRESHOLDS,
        }
        with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
            json.dump(sweep_record, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, "item") else str(o))
        if best is None:
            print("  No valid combo. Skipping aggregate.")
            return 1
        print(f"\n  selection verdict: {verdict}")
        print(f"  best: {best['combo_id']} "
              f"(M={best['mean_M_rate']*100:.1f}%, "
              f"n_prop={best['mean_n_proposals']:.1f}, "
              f"t={best['median_strategy_timing_s']:.3f}s)")

        # ---- regression (β1 alone + γ alone, 5 random samples) ----
        print("\nRegression check (β1=0.3612, γ=0.3519) on 5 random samples ...")
        reg = _regression_check(sample_packs, {
            "beta1_dir": args.beta1_baseline_dir,
            "gamma_dir": args.gamma_baseline_dir,
        })
        print(f"  → {'PASS' if reg['all_pass'] else 'FAIL'} ({len(reg['chosen_tokens'])} samples)")
        with open(osp.join(out_dir, "regression.json"), "w") as f:
            json.dump(reg, f, indent=2)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        del cp_gen
        gc.collect()

    # ---- aggregate + report ----
    agg = aggregate_alpha(summaries, best, sweep_record, reg, sample_packs,
                            per_strategy_records, out_dirs)
    with open(osp.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2,
                  default=lambda o: float(o) if hasattr(o, "item") else str(o))
    render_all_alpha(summaries, best, sweep_record, reg, agg, sample_packs,
                      per_strategy_records, out_dirs["figures"],
                      osp.join(out_dir, "report.md"))
    print(f"  → {out_dir}/aggregate.json")
    print(f"  → {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
