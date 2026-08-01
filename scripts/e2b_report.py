"""E2b report: retro-gate vs track-count-matched control.

Pre-registration: experiments/preregistration_E2b_2026-07-31.md. Stats path
imported unchanged from e2_thrmatch_report (Wilcoxon + rank-biserial + 10k
scene-bootstrap of the combined delta, seed 20260718).

Usage: python scripts/e2b_report.py --run-dir results/<e2b dir> \
           --e2c results/2026-07-30_e2c_retro_thrmatch_v01
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_thrmatch_report import (METRICS, boot_ci_combined, combine,  # noqa: E402
                                paired_stats, per_scene_metrics, scene_arrays)

FRAMES = ("ego", "global")


def load(cell: Path) -> dict:
    ax = sorted(cell.glob("axis_*/e1_metrics.json"))
    assert len(ax) == 1, f"{cell}: {ax}"
    d = {"e1": json.loads(ax[0].read_text()), "dir": ax[0].parent}
    with open(ax[0].parent / "e1_perscene.pkl", "rb") as f:
        d["perscene"] = pickle.load(f)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--e2c", required=True)
    args = ap.parse_args()
    run, e2c = Path(args.run_dir), Path(args.e2c)

    report: dict = {"arms": {}, "prereg": "experiments/preregistration_E2b_2026-07-31.md"}
    md, fig_data = [], {}
    assa_win = {}

    for frame in FRAMES:
        gate_cell, ctrl_cell = e2c / "cells" / f"retro_{frame}", run / "cells" / f"trackctrl_{frame}"
        if not (ctrl_cell / "axis_baseline" / "e1_metrics.json").exists():
            print(f"[e2b] skip {frame}: control cell missing")
            continue
        cd = {"Control": load(ctrl_cell), "Gate": load(gate_cell)}
        scenes = sorted(set(cd["Gate"]["perscene"]["trackeval"]) &
                        set(cd["Control"]["perscene"]["trackeval"]))
        ps = {k: per_scene_metrics(v["perscene"], scenes) for k, v in cd.items()}
        stats = {m: paired_stats(ps["Control"][m], ps["Gate"][m]) for m in METRICS}
        ci = boot_ci_combined(scene_arrays(cd["Control"]["perscene"], scenes),
                              scene_arrays(cd["Gate"]["perscene"], scenes), len(scenes))
        comb = {k: combine(scene_arrays(v["perscene"], scenes), np.arange(len(scenes)))
                for k, v in cd.items()}
        amota = {k: cd[k]["e1"]["gt_based"].get("amota") for k in cd}
        match = json.loads((ctrl_cell / "axis_baseline" / "match_stats.json").read_text())
        gate_boxes = json.loads((next(iter(sorted(gate_cell.glob("axis_*/metrics.json"))))
                                 .read_text()))["n_pred_boxes_total"]

        # secondary sanity: random-K arms (AssA pooled only)
        rand = sorted(glob.glob(str(run / "cells" / f"randctrl_{frame}_s*")))
        rand_assa = []
        for rc in rand:
            hits = sorted(Path(rc).glob("axis_*/e1_metrics.json"))
            if hits:
                rand_assa.append(json.loads(hits[0].read_text())["gt_based"]["AssA"])

        assa_win[frame] = (ci["AssA"][0] > 0 and comb["Gate"]["AssA"] > comb["Control"]["AssA"])
        report["arms"][frame] = {
            "n_scenes": len(scenes), "track_match": match, "gate_boxes": gate_boxes,
            "combined": comb, "amota": amota, "paired_stats": stats,
            "bootstrap_ci_combined_delta": ci, "random_K_AssA": rand_assa,
            "assa_win_ci_excludes_zero": assa_win[frame]}
        fig_data[frame] = {"comb": comb, "amota": amota,
                           "d_assa": (ps["Gate"]["AssA"] - ps["Control"]["AssA"]).tolist()}

        md += [f"\n### Association frame: {frame}",
               f"\nTrack-count match: {match['n_tracks_total']} = {match['n_tracks_target']} "
               f"tracks (exact={match['track_match_exact']}); box budgets NOT matched by design "
               f"(control {match['n_pred_boxes_total']:,} vs gate {gate_boxes:,}).\n",
               "| Metric | Control (top-K) | Retro-gate | Delta | 95% CI | Wilcoxon p | scenes gate>ctrl |",
               "|---|---|---|---|---|---|---|"]
        for m in METRICS:
            s = stats[m]
            md.append(f"| {m} | {comb['Control'][m]:.4f} | {comb['Gate'][m]:.4f} | "
                      f"{comb['Gate'][m]-comb['Control'][m]:+.4f} | "
                      f"[{ci[m][0]:+.4f}, {ci[m][1]:+.4f}] | {s['p']:.3g} | "
                      f"{s['n_scenes_gate_better']}/{s['n_scenes']} |")
        if amota["Control"] is not None and amota["Gate"] is not None:
            md.append(f"| AMOTA | {amota['Control']:.4f} | {amota['Gate']:.4f} | "
                      f"{amota['Gate']-amota['Control']:+.4f} | — (dataset-level) | — | — |")
        if rand_assa:
            md.append(f"\nRandom-K sanity (n={len(rand_assa)} seeds): pooled AssA "
                      f"{min(rand_assa):.4f}–{max(rand_assa):.4f} "
                      f"(top-K {cd['Control']['e1']['gt_based']['AssA']:.4f}, "
                      f"gate {cd['Gate']['e1']['gt_based']['AssA']:.4f})")

    # pre-registered decision (prereg §3): gate must win AssA in BOTH frames
    if len(assa_win) == len(FRAMES):
        verdict = ("CONFOUND REJECTED — identity-hygiene claim stands (unconditional)"
                   if all(assa_win.values())
                   else "CONFOUND CONFIRMED — identity-hygiene claim falls; protocol-only paper")
    else:
        verdict = "INCOMPLETE — not all frames evaluated"
    report["decision"] = {"assa_win_both_frames": assa_win, "verdict": verdict}
    md += [f"\n## Pre-registered decision\n\nAssA CI-excluding-zero win per frame: {assa_win}",
           f"\n**{verdict}**"]

    (run / "e2b_report.json").write_text(json.dumps(report, indent=2))
    (run / "e2b_table.md").write_text(
        "# E2b — Track-count-matched control (retro-gate vs top-K baseline)\n" + "\n".join(md) + "\n")
    make_figure(fig_data, run / "fig_e2b_trackmatch")
    print(f"[e2b] wrote e2b_table.md / e2b_report.json / fig_e2b_trackmatch.png — {verdict}")


def make_figure(fig_data: dict, stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = list(fig_data)
    if not frames:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = ("AssA", "HOTA", "DetA", "IDF1")
    w, x = 0.2, np.arange(len(bars) + 1)
    for i, f in enumerate(frames):
        d = fig_data[f]
        vals_c = [d["comb"][ "Control"][m] for m in bars] + [d["amota"]["Control"] or 0]
        vals_g = [d["comb"]["Gate"][m] for m in bars] + [d["amota"]["Gate"] or 0]
        axes[0].bar(x + (2 * i - 1.5) * w, vals_c, w, label=f"{f} control (top-K)", alpha=0.75)
        axes[0].bar(x + (2 * i - 0.5) * w, vals_g, w, label=f"{f} retro-gate", alpha=0.75)
    axes[0].set_xticks(x, list(bars) + ["AMOTA"])
    axes[0].set_title("Track-count-matched comparison")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].boxplot([fig_data[f]["d_assa"] for f in frames], tick_labels=frames, showfliers=False)
    axes[1].set_title("Per-scene $\\Delta$AssA (Gate $-$ Control)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=200)
    fig.savefig(f"{stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
