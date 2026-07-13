"""Does OV-TCS predict downstream temporal performance? (surrogate-metric test)

EVALUATION ONLY — no new method is proposed. We deliberately manufacture a
spread of *temporal-quality levels* using knobs that are already in the
pipeline, then ask whether OV-TCS (a per-track, GT-free temporal-consistency
score) tracks the downstream temporal-aware detector metric (phase1 mAP, where
the M11 track-age gate makes mAP depend on track structure).

Temporal-quality levels (10), all on the same gamma cache / full val 150 scenes,
phase1 axis (M11 gate + M21 vote + M31 merge):

  better continuity ----------------------------------------> worse continuity
  global a10/a5/a2 | ego a10/a5/a2 | ego+frag p=.1/.2/.3/.5

  * frame ego|global        : matching frame (ego-motion compensation)
  * association_max_age      : how long a lost track survives (2 / 5 / 10)
  * frag_inject_p            : controlled fragmentation injection — break each
    continuing track id with prob p (degrades emitted track structure WITHOUT
    touching the matcher). A pure, monotone temporal-quality dial.

For each level we measure, in one pass:
  OV-TCS A, OV-TCS B, OV-TCS C, track length, GT fragmentation, phase1 mAP,
  phase1 NDS.

Then (Pearson r, Spearman rho):
  OV-TCS A/B/C  vs phase1 mAP    <- is OV-TCS the strongest predictor?
  GT fragmentation vs phase1 mAP <- does OV-TCS beat raw fragmentation?
  track length    vs phase1 mAP  <- does OV-TCS beat track length?
and the OV-TCS<->track-length collinearity (does OV-TCS add info beyond length?).

Run (cache-only, inside the PBS GPU container):
  python -m method_scannet.streaming.eval_ovtcs_surrogate \
    --cp-cache-dir results/outdoor_native_temporal_cpcache_thr000_single_gravity \
    --output results/2026-06-12_ablation_ovtcs_surrogate_v01
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from dataloaders.nuscenes_loader import NuScenesLoader
from method_scannet.streaming.nuscenes_native_evaluator import (
    NativeTemporalNuScenesEvaluator,
    _list_val_scenes,
)

# (name, association_frame, association_max_age, frag_inject_p)
VARIANTS = [
    ("global_a10",  "global", 10, 0.0),
    ("global_a5",   "global",  5, 0.0),
    ("global_a2",   "global",  2, 0.0),
    ("ego_a10",     "ego",    10, 0.0),
    ("ego_a5",      "ego",     5, 0.0),
    ("ego_a2",      "ego",     2, 0.0),
    ("ego_frag0.1", "ego",     5, 0.1),
    ("ego_frag0.2", "ego",     5, 0.2),
    ("ego_frag0.3", "ego",     5, 0.3),
    ("ego_frag0.5", "ego",     5, 0.5),
]

AXIS = "phase1"   # the temporal-aware downstream metric (track structure -> mAP)


# --------------------------------------------------------------------------- #
# correlation helpers (dependency-free; scipy used only for exact p if present)
# --------------------------------------------------------------------------- #
def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0          # 1-based average rank for ties
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


def _tstat(r, n):
    if abs(r) >= 1.0:
        return float("inf")
    return r * math.sqrt((n - 2) / (1.0 - r * r))


def _corr(x, y):
    """Pearson r, Spearman rho, t-stat, df, and (if scipy) two-sided p-values."""
    n = len(x)
    r = _pearson(x, y)
    rho = _spearman(x, y)
    out = {"n": n, "pearson_r": r, "spearman_rho": rho,
           "t_stat": _tstat(r, n), "df": n - 2,
           "pearson_p": None, "spearman_p": None}
    try:
        from scipy import stats
        out["pearson_p"] = float(stats.pearsonr(x, y)[1])
        out["spearman_p"] = float(stats.spearmanr(x, y)[1])
    except Exception:
        pass
    return out


def _fmt(x, nd=4):
    return "  n/a" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
def _run_variant(loader, scenes, name, frame, max_age, frag_p, cache_dir,
                 cell_dir, m11_N, m12_threshold, m31_iou, m32_distance) -> dict:
    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None,
        association_threshold_m=2.0, association_max_age=max_age,
        cp_cache_dir=cache_dir, proposal_source="gamma",
        class_agnostic_association=True, association_frame=frame,
        collect_track_metrics=True, frag_inject_p=frag_p)
    ev.install_axis(AXIS, m11_N=m11_N, m12_threshold=m12_threshold,
                    m31_iou=m31_iou, m32_distance=m32_distance)
    ev.begin_axis()
    t0 = time.time()
    for i, sc in enumerate(scenes):
        try:
            ev.run_scene(sc, scene_idx=i)
        except Exception as exc:
            print(f"    SCENE {sc[:8]} FAILED: {exc!r}", flush=True)
        if (i + 1) % 50 == 0 or (i + 1) == len(scenes):
            print(f"    [{name}] {i+1}/{len(scenes)} "
                  f"tracks={len(ev._track_seq)} {time.time()-t0:.0f}s", flush=True)
    ev.last_axis_walltime_s = time.time() - t0
    summary = ev.aggregate_axis_metrics(cell_dir, None)

    vm = summary.get("variant_metrics", {})
    ov = vm.get("ov_tcs", {})
    tl = vm.get("track_length", {})
    gf = vm.get("gt_fragmentation", {})
    row = {
        "name": name, "frame": frame, "max_age": max_age, "frag_p": frag_p,
        "mAP": summary.get("mAP"), "NDS": summary.get("NDS"),
        "ovtcs_A": ov.get("A_mean"), "ovtcs_B": ov.get("B_mean"),
        "ovtcs_C": ov.get("C_mean"),
        "track_len_mean": tl.get("mean"), "track_len_median": tl.get("median"),
        "gt_frag_mean": gf.get("mean_fragments"),
        "n_tracks": vm.get("n_tracks"),
        "walltime_s": summary.get("axis_walltime_s"),
    }
    print(f"[{name}] mAP={_fmt(row['mAP'])} NDS={_fmt(row['NDS'])} "
          f"OV-TCS={_fmt(row['ovtcs_A'])}/{_fmt(row['ovtcs_B'])}/{_fmt(row['ovtcs_C'])} "
          f"trk_len={_fmt(row['track_len_mean'],2)} "
          f"GT_frag={_fmt(row['gt_frag_mean'],3)} "
          f"wall={row['walltime_s']:.0f}s", flush=True)
    return row


def _table(rows) -> str:
    hdr = ("| variant | frame | max_age | frag_p | OV-TCS_A | OV-TCS_B | OV-TCS_C "
           "| Frag | TrackLen | phase1_mAP | phase1_NDS | n_tracks |")
    sep = "|" + "|".join(["---"] * 12) + "|"
    out = [hdr, sep]
    for r in rows:
        out.append(
            f"| {r['name']} | {r['frame']} | {r['max_age']} | {r['frag_p']} | "
            f"{_fmt(r['ovtcs_A'])} | {_fmt(r['ovtcs_B'])} | {_fmt(r['ovtcs_C'])} | "
            f"{_fmt(r['gt_frag_mean'],3)} | {_fmt(r['track_len_mean'],2)} | "
            f"{_fmt(r['mAP'])} | {_fmt(r['NDS'])} | {r['n_tracks']} |")
    return "\n".join(out)


PREDICTORS = [
    ("OV-TCS A", "ovtcs_A"),
    ("OV-TCS B", "ovtcs_B"),
    ("OV-TCS C", "ovtcs_C"),
    ("GT fragmentation", "gt_frag_mean"),
    ("track length", "track_len_mean"),
]


def _correlations(rows) -> dict:
    """Every predictor vs phase1 mAP (primary) and NDS (secondary), plus the
    OV-TCS<->track-length collinearity check."""
    map_y = [r["mAP"] for r in rows]
    nds_y = [r["NDS"] for r in rows]
    tlen = [r["track_len_mean"] for r in rows]
    res = {"vs_phase1_mAP": {}, "vs_phase1_NDS": {},
           "ovtcs_vs_track_length": {}}
    for label, key in PREDICTORS:
        xs = [r[key] for r in rows]
        res["vs_phase1_mAP"][label] = _corr(xs, map_y)
        res["vs_phase1_NDS"][label] = _corr(xs, nds_y)
    for label, key in PREDICTORS[:3]:                 # A/B/C vs track length
        res["ovtcs_vs_track_length"][label] = _corr([r[key] for r in rows], tlen)
    return res


def _rank_predictors(res) -> list[tuple[str, float, float]]:
    """(label, |pearson r|, |spearman rho|) vs phase1 mAP, strongest first."""
    items = []
    for label, c in res["vs_phase1_mAP"].items():
        items.append((label, abs(c["pearson_r"]), abs(c["spearman_rho"])))
    items.sort(key=lambda t: t[1], reverse=True)
    return items


def _scatter(rows, res, out_png) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  [plot] matplotlib unavailable, skipping scatter: {exc!r}", flush=True)
        return False
    map_y = [r["mAP"] for r in rows]
    names = [r["name"] for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, (label, key) in zip(axes, PREDICTORS):
        xs = [r[key] for r in rows]
        ax.scatter(xs, map_y, s=60, c="tab:blue", zorder=3)
        for x, y, nm in zip(xs, map_y, names):
            ax.annotate(nm, (x, y), fontsize=7, xytext=(3, 3),
                        textcoords="offset points")
        c = res["vs_phase1_mAP"][label]
        ax.set_title(f"{label} vs phase1 mAP\n"
                     f"Pearson r={c['pearson_r']:.3f}  "
                     f"Spearman rho={c['spearman_rho']:.3f}  (n={c['n']})",
                     fontsize=10)
        ax.set_xlabel(label)
        ax.set_ylabel("phase1 mAP")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    ranking = _rank_predictors(res)
    txt = "Predictors of phase1 mAP\n(|Pearson r|, strongest first)\n\n"
    txt += "\n".join(f"{i+1}. {lab}: |r|={pr:.3f}  |rho|={sp:.3f}"
                     for i, (lab, pr, sp) in enumerate(ranking))
    axes[-1].text(0.02, 0.98, txt, va="top", ha="left", fontsize=11,
                  family="monospace")
    fig.suptitle("Is OV-TCS a surrogate for downstream temporal performance? "
                 "(full val 150 scenes, phase1 axis)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)
    return True


def _verdict(rows, res) -> str:
    ranking = _rank_predictors(res)
    best_label, best_r, best_rho = ranking[0]
    ovtcs_items = [(lab, pr, sp) for (lab, pr, sp) in ranking if lab.startswith("OV-TCS")]
    best_ovtcs = max(ovtcs_items, key=lambda t: t[1])
    frag = next(c for lab, c in res["vs_phase1_mAP"].items() if lab == "GT fragmentation")
    tlen = next(c for lab, c in res["vs_phase1_mAP"].items() if lab == "track length")
    coll = res["ovtcs_vs_track_length"]

    lines = ["## Verdict — is OV-TCS a surrogate for downstream temporal performance?", ""]
    lines.append(f"- Strongest predictor of phase1 mAP: **{best_label}** "
                 f"(|Pearson r|={best_r:.3f}, |Spearman rho|={best_rho:.3f}, n={len(rows)}).")
    lines.append(f"- Best OV-TCS formulation: **{best_ovtcs[0]}** "
                 f"(|r|={best_ovtcs[1]:.3f}, |rho|={best_ovtcs[2]:.3f}).")
    lines.append(f"- OV-TCS vs raw fragmentation: best OV-TCS |r|={best_ovtcs[1]:.3f} "
                 f"vs GT-frag |r|={abs(frag['pearson_r']):.3f} "
                 f"-> OV-TCS is {'STRONGER' if best_ovtcs[1] > abs(frag['pearson_r']) else 'NOT stronger'}.")
    lines.append(f"- OV-TCS vs track length: best OV-TCS |r|={best_ovtcs[1]:.3f} "
                 f"vs track-len |r|={abs(tlen['pearson_r']):.3f} "
                 f"-> OV-TCS is {'STRONGER' if best_ovtcs[1] > abs(tlen['pearson_r']) else 'NOT stronger'}.")
    lines.append("- Collinearity (does OV-TCS add info beyond length?): " + ", ".join(
        f"{lab} vs track-len r={c['pearson_r']:.3f}" for lab, c in coll.items()) + ".")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--cp-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--m11-N", type=int, default=3)
    ap.add_argument("--m12-threshold", type=float, default=0.85)
    ap.add_argument("--m31-iou", type=float, default=0.5)
    ap.add_argument("--m32-distance", type=float, default=0.5)
    args = ap.parse_args()

    out_root = Path(args.output)
    (out_root / "cells").mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = _list_val_scenes(loader)
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    print(f"  val scenes={len(scenes)} variants={len(VARIANTS)} axis={AXIS}", flush=True)

    rows = []
    for name, frame, max_age, frag_p in VARIANTS:
        print(f"\n[{name}] frame={frame} max_age={max_age} frag_p={frag_p} ...", flush=True)
        cell_dir = out_root / "cells" / name
        rows.append(_run_variant(
            loader, scenes, name, frame, max_age, frag_p, args.cp_cache_dir,
            cell_dir, args.m11_N, args.m12_threshold, args.m31_iou, args.m32_distance))

    res = _correlations(rows)
    table = _table(rows)
    verdict = _verdict(rows, res)
    ranking = _rank_predictors(res)

    (out_root / "variants.json").write_text(json.dumps(rows, indent=2))
    (out_root / "correlations.json").write_text(json.dumps(res, indent=2))
    _scatter(rows, res, out_root / "ovtcs_vs_phase1map.png")

    corr_lines = ["## Correlation with phase1 mAP (primary downstream metric)", "",
                  "| predictor | Pearson r | p | Spearman rho | p |",
                  "|---|---|---|---|---|"]
    for label, _ in PREDICTORS:
        c = res["vs_phase1_mAP"][label]
        corr_lines.append(
            f"| {label} | {c['pearson_r']:.3f} | {_fmt(c['pearson_p'],4)} | "
            f"{c['spearman_rho']:.3f} | {_fmt(c['spearman_p'],4)} |")
    rank_lines = ["", "## Predictor ranking (|Pearson r| vs phase1 mAP)", ""]
    rank_lines += [f"{i+1}. {lab}: |r|={pr:.3f} |rho|={sp:.3f}"
                   for i, (lab, pr, sp) in enumerate(ranking)]

    notes = ("# OV-TCS as a surrogate for downstream temporal performance\n\n"
             "10 temporal-quality levels, gamma cache, full val 150 scenes, phase1 "
             "axis (M11 age gate + M21 vote + M31 merge). OV-TCS / track-length / "
             "GT-fragmentation are GT-free track properties; phase1 mAP/NDS are the "
             "downstream temporal-aware detector metric. EVALUATION ONLY.\n\n"
             "## Temporal-quality ablation table\n\n" + table + "\n\n"
             + "\n".join(corr_lines) + "\n" + "\n".join(rank_lines) + "\n\n"
             + verdict + "\n")
    (out_root / "notes.md").write_text(notes)

    print("\n" + "=" * 78)
    print(table)
    print("\n".join(rank_lines))
    print()
    print(verdict)
    print(f"\nwrote {out_root/'variants.json'}\nwrote {out_root/'correlations.json'}\n"
          f"wrote {out_root/'notes.md'}", flush=True)


if __name__ == "__main__":
    main()
