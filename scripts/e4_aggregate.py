#!/usr/bin/env python
"""E4 associator-sensitivity aggregation.

Reads results/e4_associator_sensitivity/runs/<config>_d<D>_a<A>_p<P>/axis_baseline/metrics.json
and writes metrics.csv, summary.md, paper_table.md, plots/ into the results dir.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results" / "e4_associator_sensitivity"
RUNS = ROOT / "runs"
DEFAULT = (2.0, 5, 0.0)
CONFIGS = ["ego", "global", "fusion"]
METRICS = ["mAP", "NDS", "ovtcs_C", "track_len_mean", "frag_mean", "csr_mean"]
NAME_RE = re.compile(r"^(ego|global|fusion)_d([\d.]+)_a(\d+)_p([\d.]+)$")


def load_rows() -> list[dict]:
    rows = []
    for d in sorted(RUNS.iterdir()):
        m = NAME_RE.match(d.name)
        f = d / "axis_baseline" / "metrics.json"
        if not m or not f.exists():
            continue
        s = json.loads(f.read_text())
        vm = s.get("variant_metrics", {})
        rows.append({
            "config": m.group(1),
            "dist": float(m.group(2)), "max_age": int(m.group(3)),
            "score_thr": float(m.group(4)),
            "mAP": s.get("mAP"), "NDS": s.get("NDS"),
            "ovtcs_A": vm.get("ov_tcs", {}).get("A_mean"),
            "ovtcs_B": vm.get("ov_tcs", {}).get("B_mean"),
            "ovtcs_C": vm.get("ov_tcs", {}).get("C_mean"),
            "track_len_mean": vm.get("track_length", {}).get("mean"),
            "frag_mean": vm.get("gt_fragmentation", {}).get("mean_fragments"),
            "csr_mean": vm.get("csr_mean"),
            "n_tracks": vm.get("n_tracks"),
        })
    return rows


def by_setting(rows):
    """{(d, a, p): {config: row}}"""
    out: dict = {}
    for r in rows:
        out.setdefault((r["dist"], r["max_age"], r["score_thr"]), {})[r["config"]] = r
    return out


def ranking(setting_rows, metric):
    vals = {c: setting_rows[c][metric] for c in CONFIGS
            if c in setting_rows and setting_rows[c][metric] is not None}
    if len(vals) < len(CONFIGS):
        return None
    return tuple(sorted(vals, key=vals.get, reverse=True))


def sensitivity(rows, config, metric):
    """(max-min)/|default value| over the OAT grid for one config."""
    vals = [r[metric] for r in rows if r["config"] == config and r[metric] is not None]
    ref = next((r[metric] for r in rows
                if r["config"] == config
                and (r["dist"], r["max_age"], r["score_thr"]) == DEFAULT), None)
    if not vals or ref in (None, 0):
        return None, None
    return (max(vals) - min(vals)) / abs(ref), float(np.std(vals) / abs(np.mean(vals)))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def make_plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    params = [("dist", "assoc distance (m)"), ("max_age", "max age"),
              ("score_thr", "proposal score thr")]
    for metric in ("mAP", "ovtcs_C"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, (p, label) in zip(axes, params):
            others = [q for q, _ in params if q != p]
            for c in CONFIGS:
                pts = sorted(
                    (r[p], r[metric]) for r in rows
                    if r["config"] == c and r[metric] is not None
                    and all(r[q] == dict(zip(("dist", "max_age", "score_thr"), DEFAULT))[q]
                            for q in others))
                if pts:
                    ax.plot(*zip(*pts), marker="o", label=c)
            ax.set_xlabel(label)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(metric)
        axes[-1].legend()
        fig.suptitle(f"{metric} vs association parameters (OAT around default)")
        fig.tight_layout()
        fig.savefig(plots / f"oat_{metric}.png", dpi=150)
        plt.close(fig)

    # sensitivity bars
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    for i, c in enumerate(CONFIGS):
        sens = [sensitivity(rows, c, m)[0] or 0 for m in ("mAP", "NDS", "ovtcs_C")]
        ax.bar(np.arange(3) + i * width, sens, width, label=c)
    ax.set_xticks(np.arange(3) + width)
    ax.set_xticklabels(["mAP", "NDS", "OV-TCS_C"])
    ax.set_ylabel("(max−min)/default")
    ax.set_title("Normalized sensitivity over the association grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "sensitivity_bars.png", dpi=150)
    plt.close(fig)


def main():
    rows = load_rows()
    if not rows:
        sys.exit(f"no completed runs under {RUNS}")
    ROOT.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    settings = by_setting(rows)
    complete = {k: v for k, v in settings.items() if len(v) == len(CONFIGS)}

    # Q1 ranking stability
    rank_lines, flips = [], {"mAP": 0, "ovtcs_C": 0}
    base_rank = {}
    for metric in ("mAP", "ovtcs_C"):
        base_rank[metric] = ranking(complete.get(DEFAULT, {}), metric)
    for k in sorted(complete):
        line = f"| d={k[0]} a={k[1]} p={k[2]} |"
        for metric in ("mAP", "ovtcs_C"):
            rk = ranking(complete[k], metric)
            flip = rk is not None and base_rank[metric] is not None and rk != base_rank[metric]
            flips[metric] += int(flip)
            line += f" {' > '.join(rk) if rk else 'n/a'}{' **(flip)**' if flip else ''} |"
        rank_lines.append(line)

    # Q2 sensitivity table
    sens_lines = []
    for c in CONFIGS:
        for m in ("mAP", "NDS", "ovtcs_C", "track_len_mean", "frag_mean", "csr_mean"):
            rng, cv = sensitivity(rows, c, m)
            if rng is not None:
                sens_lines.append(f"| {c} | {m} | {rng:.3f} | {cv:.3f} |")

    # grid-level mAP vs OV-TCS_C rank correlation
    pairs = [(r["mAP"], r["ovtcs_C"]) for r in rows
             if r["mAP"] is not None and r["ovtcs_C"] is not None]
    rho = spearman(*map(np.array, zip(*pairs))) if len(pairs) >= 5 else float("nan")

    summary = ["# E4 associator sensitivity — summary", "",
               f"{len(rows)} runs, {len(complete)}/{len(settings)} settings complete "
               f"across all {len(CONFIGS)} configs.", "",
               "## Q1 — ranking stability (higher-is-better order per setting)", "",
               "| setting | mAP ranking | OV-TCS_C ranking |", "|---|---|---|",
               *rank_lines, "",
               f"Flips vs default setting: mAP {flips['mAP']}, OV-TCS_C {flips['ovtcs_C']} "
               f"out of {max(len(complete) - 1, 0)} non-default settings.", "",
               "## Q2 — parameter sensitivity", "",
               "| config | metric | (max−min)/default | CV |", "|---|---|---|---|",
               *sens_lines, "",
               f"Grid-level Spearman(mAP, OV-TCS_C) across all runs: **{rho:.3f}** "
               f"(n={len(pairs)}).", "",
               "## Q3 / Q4", "",
               "See paper_table.md and discussion; flagged flips above are the ",
               "candidate conclusion-changing regions."]
    (ROOT / "summary.md").write_text("\n".join(summary) + "\n")

    make_plots(rows)
    print(f"wrote {ROOT}/metrics.csv, summary.md, plots/  ({len(rows)} runs)")


if __name__ == "__main__":
    main()
