"""E2 report: Baseline vs threshold-matched Control vs Temporal Layer (M11 gate).

Reads the frozen E1 grid cells (baseline + gated arms) and the E2 control cells,
then emits: result table (md), Delta(Control->Gate) with per-scene Wilcoxon
signed-rank + rank-biserial effect size + 95% scene-bootstrap CI, a figure, and
the paper LaTeX table.

Bootstrap follows scripts/e1_stats.py: resample scenes, recombine HOTA/Identity
exactly as TrackEval combine_sequences. AMOTA/mAP are dataset-level devkit
numbers (point estimates only, per the E1 pre-registration amendment b).

Usage: python scripts/e2_thrmatch_report.py --run-dir results/<e2 dir> \
           --grid results/2026-07-18_e1_grid_v01/cells
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats as st

SEED, N_BOOT = 20260718, 10000
METRICS = ("HOTA", "AssA", "DetA", "IDF1", "DetRe")
ARMS = [("ego", "gamma_ego", "gamma_ego_p1", "ctrl_ego"),
        ("global", "gamma_global", "gamma_global_p1", "ctrl_global")]


def load_cell(cell: Path) -> dict:
    ax = sorted(cell.glob("axis_*/e1_metrics.json"))
    assert len(ax) == 1, f"{cell}: {ax}"
    d = {"e1": json.loads(ax[0].read_text()),
         "det": json.loads((ax[0].parent / "metrics.json").read_text())}
    with open(ax[0].parent / "e1_perscene.pkl", "rb") as f:
        d["perscene"] = pickle.load(f)
    return d


# -- per-scene arrays / TrackEval recombination (mirrors e1_stats.py) --------
def scene_arrays(perscene: dict, scenes: list[int]) -> dict:
    te = perscene["trackeval"]
    out = {f: np.stack([np.asarray(te[s]["HOTA"][f], dtype=np.float64) for s in scenes])
           for f in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA")}
    for f in ("IDTP", "IDFN", "IDFP"):
        out[f] = np.asarray([float(np.sum(te[s]["Identity"][f])) for s in scenes])
    return out


def combine(arr: dict, idx: np.ndarray) -> dict:
    tp, fn, fp = (arr[k][idx].sum(0) for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP"))
    assa = (arr["AssA"][idx] * arr["HOTA_TP"][idx]).sum(0) / np.maximum(tp, 1e-10)
    deta = tp / np.maximum(tp + fn + fp, 1e-10)
    detre = tp / np.maximum(tp + fn, 1e-10)
    idtp, idfn, idfp = (arr[k][idx].sum() for k in ("IDTP", "IDFN", "IDFP"))
    return {"AssA": float(assa.mean()), "DetA": float(deta.mean()),
            "HOTA": float(np.sqrt(deta * assa).mean()), "DetRe": float(detre.mean()),
            "IDF1": float(2 * idtp / max(2 * idtp + idfn + idfp, 1e-10))}


def per_scene_metrics(perscene: dict, scenes: list[int]) -> dict:
    """One scalar per scene per metric (alpha-averaged), for the paired test."""
    te = perscene["trackeval"]
    out = {m: [] for m in METRICS}
    for s in scenes:
        h, i = te[s]["HOTA"], te[s]["Identity"]
        tp, fn, fp = (np.asarray(h[k], dtype=np.float64)
                      for k in ("HOTA_TP", "HOTA_FN", "HOTA_FP"))
        assa = np.asarray(h["AssA"], dtype=np.float64)
        deta = tp / np.maximum(tp + fn + fp, 1e-10)
        idtp, idfn, idfp = (float(np.sum(i[k])) for k in ("IDTP", "IDFN", "IDFP"))
        out["AssA"].append(assa.mean())
        out["DetA"].append(deta.mean())
        out["HOTA"].append(np.sqrt(deta * assa).mean())
        out["DetRe"].append((tp / np.maximum(tp + fn, 1e-10)).mean())
        out["IDF1"].append(2 * idtp / max(2 * idtp + idfn + idfp, 1e-10))
    return {k: np.asarray(v) for k, v in out.items()}


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """b - a per scene: Wilcoxon signed-rank + rank-biserial + bootstrap CI."""
    d = b - a
    nz = d[d != 0]
    if nz.size:
        w = st.wilcoxon(nz, alternative="two-sided", zero_method="wilcox")
        r = np.argsort(np.argsort(np.abs(nz))) + 1
        rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum())  # rank-biserial
        p, stat = float(w.pvalue), float(w.statistic)
    else:
        p, stat, rb = 1.0, 0.0, 0.0
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, d.size, size=(N_BOOT, d.size))
    boot = d[idx].mean(1)
    return {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
            "wilcoxon_W": stat, "p": p, "rank_biserial": rb,
            "n_scenes": int(d.size), "n_scenes_gate_better": int((d > 0).sum()),
            "ci95_mean_delta": [float(np.percentile(boot, 2.5)),
                                float(np.percentile(boot, 97.5))]}


def boot_ci_combined(arr_a: dict, arr_b: dict, n_scenes: int) -> dict:
    """95% CI of the combined (dataset-level) Delta, scene bootstrap."""
    rng = np.random.default_rng(SEED)
    out = {m: [] for m in METRICS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_scenes, size=n_scenes)
        ca, cb = combine(arr_a, idx), combine(arr_b, idx)
        for m in METRICS:
            out[m].append(cb[m] - ca[m])
    return {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for m, v in out.items()}


def row(name: str, cell: dict) -> dict:
    e1, det = cell["e1"]["gt_based"], cell["det"]
    return {"method": name, "boxes": det["n_pred_boxes_total"], "mAP": det["mAP"],
            "NDS": det["NDS"], "HOTA": e1["HOTA"], "AssA": e1["AssA"],
            "DetA": e1["DetA"], "IDF1": e1["IDF1"], "AMOTA": e1.get("amota"),
            "frag": det["variant_metrics"]["gt_fragmentation"]["mean_fragments"],
            "tracklen": det["variant_metrics"]["track_length"]["mean"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--grid", required=True)
    args = ap.parse_args()
    run, grid = Path(args.run_dir), Path(args.grid)

    report: dict = {"arms": {}, "seed": SEED, "n_boot": N_BOOT}
    md, tex = [], []
    fig_data = {}

    for frame, base_c, gate_c, ctrl_c in ARMS:
        cells = {"Baseline": grid / base_c, "Control": run / "cells" / ctrl_c,
                 "Gate": grid / gate_c}
        if not (cells["Control"] / "axis_baseline" / "e1_metrics.json").exists():
            print(f"[e2] skip {frame}: control cell missing")
            continue
        cd = {k: load_cell(v) for k, v in cells.items()}
        rows = {k: row(k, v) for k, v in cd.items()}
        scenes = sorted(set(cd["Gate"]["perscene"]["trackeval"]) &
                        set(cd["Control"]["perscene"]["trackeval"]))

        ps_c = per_scene_metrics(cd["Control"]["perscene"], scenes)
        ps_g = per_scene_metrics(cd["Gate"]["perscene"], scenes)
        stats = {m: paired_stats(ps_c[m], ps_g[m]) for m in METRICS}
        ci = boot_ci_combined(scene_arrays(cd["Control"]["perscene"], scenes),
                              scene_arrays(cd["Gate"]["perscene"], scenes), len(scenes))
        recall = {k: combine(scene_arrays(cd[k]["perscene"], scenes),
                             np.arange(len(scenes)))["DetRe"] for k in cd}
        thr = json.loads((run / f"threshold_{frame}.json").read_text())

        report["arms"][frame] = {"rows": rows, "threshold": thr, "n_scenes": len(scenes),
                                 "recall_DetRe": recall, "paired_stats": stats,
                                 "bootstrap_ci_combined_delta": ci}
        fig_data[frame] = {"rows": rows, "per_scene_delta":
                           {m: (ps_g[m] - ps_c[m]).tolist() for m in METRICS}}

        md += [f"\n### Association frame: {frame} (gate = M11 N=3, phase1)",
               f"\nMatched threshold t = {thr['threshold']:.6f} "
               f"(count {rows['Control']['boxes']} vs gate {rows['Gate']['boxes']}, "
               f"rel. err {(rows['Control']['boxes']-rows['Gate']['boxes'])/rows['Gate']['boxes']:+.3%})\n",
               "| Method | #Boxes | mAP | HOTA | AssA | DetA | IDF1 | AMOTA | Recall | Frag |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        label = {"Baseline": "Baseline (N=1)",
                 "Control": "Threshold-matched Control",
                 "Gate": "Temporal Layer (N=3)"}
        for k in ("Baseline", "Control", "Gate"):
            r = rows[k]
            md.append(f"| {label[k]} | {r['boxes']:,} | {r['mAP']:.4f} | {r['HOTA']:.4f} | "
                      f"{r['AssA']:.4f} | {r['DetA']:.4f} | {r['IDF1']:.4f} | "
                      f"{r['AMOTA']:.4f} | {recall[k]:.4f} | {r['frag']:.2f} |")
        md += ["", "**Delta (Control -> Gate)** — paired over "
               f"{len(scenes)} scenes; CI = 10k scene bootstrap of the combined delta.", "",
               "| Metric | Control | Gate | Delta | 95% CI | Wilcoxon p | rank-biserial | scenes gate>ctrl |",
               "|---|---|---|---|---|---|---|---|"]
        for m in METRICS:
            c = combine(scene_arrays(cd["Control"]["perscene"], scenes), np.arange(len(scenes)))[m]
            g = combine(scene_arrays(cd["Gate"]["perscene"], scenes), np.arange(len(scenes)))[m]
            s = stats[m]
            md.append(f"| {m} | {c:.4f} | {g:.4f} | {g-c:+.4f} | "
                      f"[{ci[m][0]:+.4f}, {ci[m][1]:+.4f}] | {s['p']:.3g} | "
                      f"{s['rank_biserial']:+.3f} | {s['n_scenes_gate_better']}/{s['n_scenes']} |")
        md.append(f"| mAP | {rows['Control']['mAP']:.4f} | {rows['Gate']['mAP']:.4f} | "
                  f"{rows['Gate']['mAP']-rows['Control']['mAP']:+.4f} | — (dataset-level) | — | — | — |")
        md.append(f"| AMOTA | {rows['Control']['AMOTA']:.4f} | {rows['Gate']['AMOTA']:.4f} | "
                  f"{rows['Gate']['AMOTA']-rows['Control']['AMOTA']:+.4f} | — (dataset-level) | — | — | — |")

        tex += [f"% --- {frame} association ---",
                r"\begin{tabular}{lrrrrrrr}", r"\toprule",
                r"Method & \#Boxes & mAP & HOTA & AssA & DetA & IDF1 & AMOTA \\", r"\midrule"]
        for k in ("Baseline", "Control", "Gate"):
            r = rows[k]
            tex.append(f"{label[k]} & {r['boxes']:,} & {r['mAP']:.3f} & {r['HOTA']:.3f} & "
                       f"{r['AssA']:.3f} & {r['DetA']:.3f} & {r['IDF1']:.3f} & {r['AMOTA']:.3f} \\\\")
        d = {m: rows['Gate'][m] - rows['Control'][m] for m in ("mAP", "HOTA", "AssA", "DetA", "IDF1", "AMOTA")}
        tex += [r"\midrule",
                r"$\Delta$ (Control$\rightarrow$Gate) & --- & "
                + " & ".join(f"{d[m]:+.3f}" for m in ("mAP", "HOTA", "AssA", "DetA", "IDF1", "AMOTA")) + r" \\",
                r"\bottomrule", r"\end{tabular}", ""]

    (run / "e2_report.json").write_text(json.dumps(report, indent=2))
    (run / "e2_table.md").write_text(
        "# E2 — Score-threshold-matched control for the temporal layer\n" + "\n".join(md) + "\n")
    (run / "e2_table.tex").write_text("\n".join(tex) + "\n")
    make_figure(fig_data, run / "fig_e2_thrmatch")
    print("[e2] wrote e2_table.md / e2_table.tex / e2_report.json / fig_e2_thrmatch.{png,pdf}")


def make_figure(fig_data: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = list(fig_data)
    if not frames:
        return
    fig, axes = plt.subplots(2, len(frames), figsize=(6.2 * len(frames), 7.4), squeeze=False)
    bars = ("HOTA", "AssA", "DetA", "IDF1", "AMOTA")
    colors = {"Baseline": "#9aa0a6", "Control": "#e07a5f", "Gate": "#3d5a80"}
    label = {"Baseline": "Baseline (N=1)", "Control": "Thr-matched Control", "Gate": "Temporal Layer (N=3)"}

    for j, f in enumerate(frames):
        rows = fig_data[f]["rows"]
        ax = axes[0][j]
        x = np.arange(len(bars))
        for i, k in enumerate(("Baseline", "Control", "Gate")):
            ax.bar(x + (i - 1) * 0.27, [rows[k][m] for m in bars], 0.26,
                   label=f"{label[k]} ({rows[k]['boxes']:,} boxes)", color=colors[k])
        ax.set_xticks(x, bars)
        ax.set_title(f"{f} association — count-matched comparison")
        ax.set_ylabel("metric value")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1][j]
        deltas = fig_data[f]["per_scene_delta"]
        keys = [m for m in ("HOTA", "AssA", "DetA", "IDF1") if m in deltas]
        ax.axhline(0, color="k", lw=0.8)
        ax.boxplot([deltas[m] for m in keys], tick_labels=keys, showfliers=False)
        for i, m in enumerate(keys):
            v = np.asarray(deltas[m])
            ax.scatter(np.full(v.size, i + 1) + np.random.default_rng(0).normal(0, 0.05, v.size),
                       v, s=4, alpha=0.25, color=colors["Gate"])
        ax.set_title(f"{f}: per-scene $\\Delta$ (Gate $-$ Control)")
        ax.set_ylabel("per-scene delta")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
