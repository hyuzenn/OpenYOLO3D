"""β1.5 orchestrator.

Single PBS job. β1's pillar foreground extraction runs ONCE per sample
(config locked); the resulting foreground point cloud is cached and reused
across all 27 verticality combos. HDBSCAN config is also locked.

Sweep grid (size_min, size_max, aspect_max) = 3×3×3 = 27 combos.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import os.path as osp
import shutil
import sys
import time

import numpy as np

from diagnosis_w1.run_clustering_check import _build_loader, _cache_samples
from preprocessing.pillar_foreground import PillarForegroundExtractor
from preprocessing.verticality_filter import VerticalityFilter
from adapters.lidar_proposals import LiDARProposalGenerator
from diagnosis_beta1_5.measurements import (
    BETA1_BEST_PILLAR, HDBSCAN_BEST,
    SampleTimeout, measure_with_timeout, PER_SAMPLE_TIMEOUT_S,
)
from diagnosis_beta1_5.aggregate import aggregate_beta1_5, render_all_beta1_5


GRID = {
    "size_min": [3, 5, 8],
    "size_max": [50, 100, 200],
    "aspect_max": [3.0, 5.0, 10.0],
}


def _combo_id(combo: dict) -> str:
    return (f"smin{combo['size_min']}_smax{combo['size_max']}_"
            f"asp{combo['aspect_max']:g}")


def _verify_beta1_regression(cache, foreground_cache) -> dict:
    """Acceptance #7 — running β1 best WITHOUT verticality should reproduce
    β1's saved best M_rate (0.361). We use the pre-cached foreground PCs and
    run HDBSCAN directly (skipping verticality entirely).
    """
    from diagnosis_w1.measurements import match_gt_to_clusters
    gen = LiDARProposalGenerator(**HDBSCAN_BEST)
    M_rates = []
    n_cl = []
    for tok, rec in cache.items():
        fg = foreground_cache[tok]
        if fg.shape[0] == 0:
            n_gt = len(rec["gt_boxes"])
            M_rates.append(0.0)
            n_cl.append(0)
            continue
        h = gen.generate(fg)
        per_gt, cases = match_gt_to_clusters(
            rec["gt_boxes"], rec["ego_pose"], fg[:, :3], h["cluster_ids"],
        )
        n_gt = len(rec["gt_boxes"])
        M_rates.append(cases["M"] / n_gt if n_gt else 0.0)
        n_cl.append(int(h["n_clusters"]))
    mean_M = float(np.mean(M_rates))
    mean_n = float(np.mean(n_cl))
    return {
        "checked_samples": len(M_rates),
        "mean_M_rate": mean_M,
        "expected_M_rate": 0.3612,
        "delta_M": abs(mean_M - 0.3612),
        "mean_n_clusters": mean_n,
        "expected_n_clusters": 182.12,
        "delta_n_clusters": abs(mean_n - 182.12),
        "passes": (abs(mean_M - 0.3612) < 1e-3) and (abs(mean_n - 182.12) < 0.5),
    }


def _run_combo(combo: dict, cache: dict, foreground_cache: dict,
               hdbscan_gen, out_dirs: dict) -> dict:
    cid = _combo_id(combo)
    combo_dir = osp.join(out_dirs["per_sample_per_config"], cid)
    os.makedirs(combo_dir, exist_ok=True)

    vert = VerticalityFilter(**combo)

    per_sample = []
    timeouts = errors = 0
    for tok in sorted(cache.keys()):
        try:
            rec = measure_with_timeout(vert, hdbscan_gen, cache[tok], foreground_cache[tok])
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

    n_cl = [r["n_clusters"] for r in per_sample]
    M = [r["M_rate"] for r in per_sample]
    L = [r["L_rate"] for r in per_sample]
    D = [r["D_rate"] for r in per_sample]
    miss = [r["miss_rate"] for r in per_sample]
    timing = [r["timing_total"] for r in per_sample]
    rem = [r["verticality"].get("removed_point_ratio", 0.0) for r in per_sample]
    return {
        "combo": combo,
        "combo_id": cid,
        "n_samples_succeeded": len(per_sample),
        "n_timeouts": timeouts,
        "n_errors": errors,
        "mean_n_clusters": float(np.mean(n_cl)),
        "median_n_clusters": float(np.median(n_cl)),
        "mean_M_rate": float(np.mean(M)),
        "mean_L_rate": float(np.mean(L)),
        "mean_D_rate": float(np.mean(D)),
        "mean_miss_rate": float(np.mean(miss)),
        "mean_removed_point_ratio": float(np.mean(rem)),
        "median_timing_total": float(np.median(timing)),
        "p95_timing_total": float(np.percentile(timing, 95)),
    }


def _select_best(summaries):
    valid = [c for c in summaries if c["n_samples_succeeded"] >= 45]
    if not valid:
        return None, "NO_VALID_COMBOS"
    f1 = [c for c in valid if c["mean_M_rate"] >= 0.50
          and 10 <= c["mean_n_clusters"] <= 50
          and c["median_timing_total"] < 3.0]
    if f1:
        return max(f1, key=lambda c: c["mean_M_rate"]), "TIER1_PASS"
    f2 = [c for c in valid if c["mean_M_rate"] >= 0.40
          and c["mean_n_clusters"] <= 80
          and c["median_timing_total"] < 3.5]
    if f2:
        return max(f2, key=lambda c: c["mean_M_rate"]), "TIER2_SOFT_PASS"
    return max(valid, key=lambda c: c["mean_M_rate"]), "TIER3_FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-beta1-samples",
                        default="results/diagnosis_beta1/samples_used.json")
    parser.add_argument("--data-config", default="configs/nuscenes_baseline.yaml")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        default="results/diagnosis_beta1_5")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dirs = {
        "root": out_dir,
        "per_sample_per_config": osp.join(out_dir, "per_sample_per_config"),
        "figures": osp.join(out_dir, "figures"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(args.use_beta1_samples) as f:
        prov = json.load(f)
    mini_tokens = prov["tokens_by_source"]["mini"]
    trainval_tokens = prov["tokens_by_source"]["trainval"]
    shutil.copy(args.use_beta1_samples, osp.join(out_dir, "samples_used.json"))

    n_combos = (len(GRID["size_min"]) * len(GRID["size_max"])
                * len(GRID["aspect_max"]))
    print("=" * 60)
    print(f"β1.5 — verticality filter sweep")
    print(f"  {len(mini_tokens)} mini + {len(trainval_tokens)} trainval = "
          f"{len(mini_tokens) + len(trainval_tokens)} samples (from β1)")
    print(f"  sweep: {n_combos} combos")
    print("=" * 60)

    # ---- load samples ----
    mini_loader = _build_loader("v1.0-mini", args.data_config, out_dir, "mini_b15")
    mini_cache, _ = _cache_samples(mini_loader, mini_tokens, "mini")
    del mini_loader

    trainval_loader = _build_loader("v1.0-trainval", args.data_config, out_dir, "trainval_b15")
    trainval_cache, _ = _cache_samples(trainval_loader, trainval_tokens, "trainval")
    del trainval_loader

    cache = {**mini_cache, **trainval_cache}
    print(f"  cached {len(cache)} samples")

    # ---- pre-compute β1 foreground (once, shared across 27 combos) ----
    print("\nPre-computing β1 best foreground for all samples ...")
    t0 = time.time()
    extractor = PillarForegroundExtractor(**BETA1_BEST_PILLAR)
    foreground_cache = {}
    for tok, rec in cache.items():
        out = extractor.extract(rec["pc_ego"])
        foreground_cache[tok] = out["foreground_pcd"]
    print(f"  pre-computed {len(foreground_cache)} foreground PCs in {time.time() - t0:.1f}s")

    # ---- regression check (acceptance #7) ----
    print("\nβ1 best regression check (foreground + HDBSCAN, no verticality) ...")
    reg = _verify_beta1_regression(cache, foreground_cache)
    print(f"  mean_M_rate = {reg['mean_M_rate']:.4f} (expected 0.3612, Δ={reg['delta_M']:.6f})")
    print(f"  mean_n_clusters = {reg['mean_n_clusters']:.4f} "
          f"(expected 182.12, Δ={reg['delta_n_clusters']:.6f})")
    print(f"  → {'PASS' if reg['passes'] else 'FAIL'}")
    with open(osp.join(out_dir, "beta1_regression.json"), "w") as f:
        json.dump(reg, f, indent=2)
    if not reg["passes"]:
        print("  ABORT — β1 baseline drift detected.")
        sys.exit(2)

    # ---- sweep ----
    hdbscan_gen = LiDARProposalGenerator(**HDBSCAN_BEST)

    combos = []
    for sm, sx, am in itertools.product(GRID["size_min"], GRID["size_max"], GRID["aspect_max"]):
        combos.append({"size_min": sm, "size_max": sx, "aspect_max": am})

    print(f"\n=== sweep: {len(combos)} combos × {len(cache)} samples ===")
    t_sweep_start = time.time()
    summaries = []
    for i, combo in enumerate(combos):
        cid = _combo_id(combo)
        t_c = time.time()
        s = _run_combo(combo, cache, foreground_cache, hdbscan_gen, out_dirs)
        elapsed = time.time() - t_c
        summaries.append(s)
        if s["n_samples_succeeded"] == 0:
            print(f"  [{i+1}/{len(combos)}] {cid}: FAILED")
        else:
            print(f"  [{i+1}/{len(combos)}] {cid}: "
                  f"M={s['mean_M_rate']*100:.1f}%, "
                  f"miss={s['mean_miss_rate']*100:.1f}%, "
                  f"n_cl={s['mean_n_clusters']:.1f}, "
                  f"removed={s['mean_removed_point_ratio']*100:.1f}%, "
                  f"t={s['median_timing_total']:.2f}s "
                  f"({s['n_samples_succeeded']}/{len(cache)} ok, {elapsed:.1f}s)")
    print(f"  sweep finished in {time.time() - t_sweep_start:.1f}s")

    # ---- best ----
    best, verdict = _select_best(summaries)
    sweep_record = {
        "n_samples": len(cache),
        "grid": GRID,
        "n_combos": len(combos),
        "results": summaries,
        "selection_verdict": verdict,
        "best": best,
        "selection_rule": ("tier1: M≥0.50 and 10≤n_cl≤50 and t<3.0s; "
                           "tier2: M≥0.40 and n_cl≤80 and t<3.5s; "
                           "tier3 (FAIL): argmax(M)"),
        "fixed_pillar_config": BETA1_BEST_PILLAR,
        "fixed_hdbscan_config": HDBSCAN_BEST,
    }
    with open(osp.join(out_dir, "parameter_sweep.json"), "w") as f:
        json.dump(sweep_record, f, indent=2)
    if best is None:
        print("  No valid combo. Skipping aggregate.")
        return 0
    print(f"\n  selection verdict: {verdict}")
    print(f"  best combo: {best['combo_id']} "
          f"(M={best['mean_M_rate']*100:.1f}%, n_cl={best['mean_n_clusters']:.1f}, "
          f"miss={best['mean_miss_rate']*100:.1f}%, t={best['median_timing_total']:.2f}s)")

    # ---- aggregate + report ----
    agg = aggregate_beta1_5(summaries, best, sweep_record, reg, cache, out_dirs)
    with open(osp.join(out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    render_all_beta1_5(summaries, best, sweep_record, reg, agg,
                       out_dirs["figures"], osp.join(out_dir, "report.md"))
    print(f"  → {out_dir}/aggregate.json")
    print(f"  → {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
