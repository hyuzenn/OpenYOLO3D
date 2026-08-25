"""Indoor matched control — gate (M11 N=3) vs. score-matched static selection.

Pre-registration: experiments/preregistration_indoor_matched_2026-08-01.md
(committed 273ff13 BEFORE this file was written).

Per scene, K_s = |M11 end-of-scene confirmed set|. The control keeps the
top-K_s Mask3D proposals by cached detection score via a StaticSetGate in the
same `method_11` slot (per-frame gate() + finalize _confirmed filter), so the
arms differ only in WHICH identities are kept and WHEN suppression applies.
Primary endpoint: per-scene lsc. Secondary null: AP.

Usage (inside PBS, GPU needed for the YOLO-World 2D pass):
  python scripts/indoor_matched_control.py --cache-dir results/2026-05-13_mask3d_cache \
      --output results/2026-08-01_indoor_matched_control_v01 [--limit 3]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate import SCENE_NAMES_SCANNET200, evaluate_scannet200
from method_scannet.streaming.hooks_streaming import (
    install_method_streaming,
    uninstall_all_streaming,
)
from method_scannet.streaming.metrics import label_switch_count
from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
from utils import OpenYolo3D
from utils.utils_2d import load_yaml

SEED, N_BOOT = 20260718, 10000
# frozen 312-scene anchors (results/2026-05-15_streaming_ablation_core_temporal)
ANCHOR = {"AP": 0.19540, "lsc": 17023}


class StaticSetGate:
    """Non-temporal stand-in for FrameCountingGate: a fixed allowed set.

    Duck-types the two touchpoints the wrapper uses (wrapper.py:371, :642):
    per-frame ``gate(visible)`` and finalize ``._confirmed``.
    """

    def __init__(self, allowed: set[int]) -> None:
        self._confirmed = set(int(i) for i in allowed)

    def gate(self, visible_instances) -> list[int]:
        return sorted({int(i) for i in visible_instances} & self._confirmed)

    def reset(self) -> None:  # per-scene evaluators; kept for interface parity
        pass


def run_arm(name, oy3d, cfg, cache_dir, out_root, scenes, gate_factory):
    """Copy of run_one_axis' loop with a per-scene gate factory.

    gate_factory(scene_name, evaluator) -> None installs the arm's gate (or
    nothing for baseline-like arms) and may read the cache. Returns summary and
    writes per-scene lsc + kept-set sizes.
    """
    out_dir = out_root / f"axis_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_full, per_scene = {}, {}
    t0 = time.time()
    for s_idx, scene_name in enumerate(scenes):
        cache_path = cache_dir / f"{scene_name}.pt"
        if not cache_path.exists():
            print(f"  skip {scene_name} (no cache)", flush=True)
            continue
        evaluator = StreamingScanNetEvaluator(
            openyolo3d_instance=oy3d,
            scene_dir=str(Path("data/scannet200") / scene_name),
            depth_scale=cfg["openyolo3d"]["depth_scale"],
            depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
            num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
            topk=int(cfg["openyolo3d"].get("topk", 40)),
            topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
        )
        frequency = int(cfg["openyolo3d"].get("frequency", 10))
        evaluator.frame_indices = [f for f in evaluator.frame_indices if f % frequency == 0]
        evaluator.setup_scene(mask3d_cache_path=str(cache_path))
        uninstall_all_streaming(evaluator)
        gate_factory(scene_name, evaluator)

        for fi in evaluator.frame_indices:
            evaluator.step_frame(fi)
        preds = evaluator.compute_method_predictions()
        preds_full[scene_name] = {
            "pred_masks": preds["pred_masks"],
            "pred_classes": preds["pred_classes"],
            "pred_scores": np.ones_like(preds["pred_scores"]),
        }
        hist = list(evaluator.pred_history)
        gate = getattr(evaluator, "method_11", None)
        per_scene[scene_name] = {
            "lsc": int(label_switch_count(hist)),
            "n_unique": len(set().union(*[h.keys() for h in hist])) if hist else 0,
            "n_confirmed": len(gate._confirmed) if gate is not None else None,
        }
        if (s_idx + 1) % 25 == 0 or s_idx == len(scenes) - 1:
            el = time.time() - t0
            print(f"  [{name}] {s_idx+1}/{len(scenes)} elapsed={el/60:.1f}min "
                  f"eta={(len(scenes)-s_idx-1)*el/max(s_idx+1,1)/60:.1f}min", flush=True)

    avgs, *_ = evaluate_scannet200(
        preds_full, "data/scannet200/ground_truth",
        output_file="/tmp/_indoor_ctrl_eval_unused.txt",
        dataset="scannet200", pretrained_on_scannet200=True)
    summary = {
        "axis": name,
        "AP": float(avgs["all_ap"]), "AP_50": float(avgs["all_ap_50%"]),
        "lsc_total": int(sum(v["lsc"] for v in per_scene.values())),
        "n_unique_total": int(sum(v["n_unique"] for v in per_scene.values())),
        "n_scenes": len(per_scene),
        "walltime_s": time.time() - t0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "per_scene.json").write_text(json.dumps(per_scene, indent=2))
    print(f"[{name}] AP={summary['AP']:.5f} lsc={summary['lsc_total']} "
          f"uniq={summary['n_unique_total']}", flush=True)
    return summary, per_scene


def topk_by_score(cache_path: Path, K: int) -> set[int]:
    _, scores = torch.load(cache_path, map_location="cpu")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return set(int(i) for i in order[:K])


def paired_stats(ctrl: dict, gate: dict) -> dict:
    from scipy import stats as st
    common = sorted(set(ctrl) & set(gate))
    d = np.array([ctrl[s]["lsc"] - gate[s]["lsc"] for s in common], dtype=np.float64)
    nz = d[d != 0]
    w = st.wilcoxon(nz) if len(nz) else None
    n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
    rb = (n_pos - n_neg) / max(n_pos + n_neg, 1)
    rng = np.random.default_rng(SEED)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return {
        "n_scenes": len(common),
        "mean_delta_ctrl_minus_gate": float(d.mean()),
        "median_delta": float(np.median(d)),
        "wilcoxon_p": float(w.pvalue) if w else None,
        "rank_biserial": float(rb),
        "scenes_gate_better": n_pos,
        "ci95_mean_delta": [float(np.percentile(boots, 2.5)),
                            float(np.percentile(boots, 97.5))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--config", default="pretrained/config_scannet200.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--random-seeds", type=int, default=3)
    args = ap.parse_args()
    out_root = args.output
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    print("Constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)
    scenes = list(SCENE_NAMES_SCANNET200)
    if args.limit:
        scenes = scenes[: args.limit]
    print(f"scenes={len(scenes)}", flush=True)

    # ---- Arm G: gate (M11 N=3), records K_s ---------------------------
    def g_factory(_scene, ev):
        install_method_streaming(ev, "M11", N=3)
    g_sum, g_ps = run_arm("gate_m11", oy3d, cfg, args.cache_dir, out_root, scenes, g_factory)

    # anchor reproduction check (prereg §6) — full runs only
    anchor_ok = None
    if args.limit is None:
        anchor_ok = (abs(g_sum["AP"] - ANCHOR["AP"]) <= 0.002
                     and abs(g_sum["lsc_total"] - ANCHOR["lsc"]) / ANCHOR["lsc"] <= 0.02)
        print(f"[anchor] AP {g_sum['AP']:.5f} vs {ANCHOR['AP']}  "
              f"lsc {g_sum['lsc_total']} vs {ANCHOR['lsc']}  ok={anchor_ok}", flush=True)
        if not anchor_ok:
            (out_root / "report.json").write_text(json.dumps(
                {"status": "ABORT: gate arm failed frozen-anchor reproduction",
                 "gate": g_sum}, indent=2))
            raise SystemExit(2)

    K = {s: v["n_confirmed"] for s, v in g_ps.items()}

    # ---- Arm C: top-K_s by score --------------------------------------
    def c_factory(scene, ev):
        allowed = topk_by_score(args.cache_dir / f"{scene}.pt", K[scene])
        assert len(allowed) == K[scene], f"{scene}: K_s exceeds proposal count"
        ev.method_11 = StaticSetGate(allowed)
    c_sum, c_ps = run_arm("ctrl_topk", oy3d, cfg, args.cache_dir, out_root, scenes, c_factory)

    # ---- random sanity arms -------------------------------------------
    rand_lsc = []
    for i in range(args.random_seeds):
        rng = np.random.default_rng(SEED + i)
        def r_factory(scene, ev, rng=rng):
            _, scores = torch.load(args.cache_dir / f"{scene}.pt", map_location="cpu")
            ev.method_11 = StaticSetGate(
                set(int(x) for x in rng.choice(len(scores), K[scene], replace=False)))
        r_sum, _ = run_arm(f"ctrl_rand_s{i}", oy3d, cfg, args.cache_dir, out_root,
                           scenes, r_factory)
        rand_lsc.append(r_sum["lsc_total"])

    # ---- decision -----------------------------------------------------
    stats = paired_stats(c_ps, g_ps)
    lo, hi = stats["ci95_mean_delta"]
    gate_wins = stats["mean_delta_ctrl_minus_gate"] > 0 and lo > 0
    report = {
        "prereg": "experiments/preregistration_indoor_matched_2026-08-01.md",
        "anchor_reproduced": anchor_ok,
        "arms": {"gate": g_sum, "ctrl_topk": c_sum, "random_lsc_totals": rand_lsc},
        "match": {
            "K_total": int(sum(K.values())),
            "ctrl_n_unique_total": c_sum["n_unique_total"],
            "gate_n_unique_total": g_sum["n_unique_total"],
        },
        "primary_lsc_stats": stats,
        "secondary_AP": {"gate": g_sum["AP"], "ctrl_topk": c_sum["AP"],
                         "baseline_frozen": 0.19560},
        "decision": ("GATE WINS — indoor hygiene gain is temporal-selection-specific"
                     if gate_wins else
                     "CONFOUND CONFIRMED — indoor gain is generic pruning; demote indoor leg"),
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("match", "primary_lsc_stats", "decision")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
