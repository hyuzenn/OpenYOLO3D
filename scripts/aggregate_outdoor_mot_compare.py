"""Aggregate the E1 MOT-metric comparison run (outdoor_mot_compare).

Reads <run_dir>/outputs/{summary.json, per_scene_{ego,global}.json} and writes
publication artifacts into <run_dir>:
  mot_compare_table.md   system-level table + per-scene correlation table
  per_scene_metrics.csv  one row per (scene, arm)
  correlations.json      machine-readable correlation battery
  fig_ovtcs_vs_mot.png/.pdf   scatter panels (per-scene, both arms) + deltas

Usage: python scripts/aggregate_outdoor_mot_compare.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

MOT_COLS = ["HOTA", "DetA", "AssA", "IDF1", "IDP", "IDR", "MOTA", "MOTP_m",
            "IDS", "FRAG"]
# Okabe-Ito blue / vermillion — CVD-safe pair, fixed arm order.
ARM_COLOR = {"ego": "#0072B2", "global": "#D55E00"}


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"n": int(ok.sum())}
    pr, pp = stats.pearsonr(x[ok], y[ok])
    sr, sp = stats.spearmanr(x[ok], y[ok])
    return {"n": int(ok.sum()), "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_r": float(sr), "spearman_p": float(sp)}


def main():
    run_dir = Path(sys.argv[1])
    out = run_dir / "outputs"
    summary = json.loads((out / "summary.json").read_text())
    arms = list(summary["arms"].keys())
    per_scene = {a: json.loads((out / f"per_scene_{a}.json").read_text())
                 for a in arms}

    # ---- per-scene csv --------------------------------------------------
    rows = []
    for a in arms:
        for sid, r in sorted(per_scene[a].items(), key=lambda kv: int(kv[0])):
            rows.append({"scene_idx": int(sid), "arm": a, **{
                k: r.get(k) for k in ["scene_token", "ovtcs_C_mean", "n_tracks",
                                      *MOT_COLS, "n_gt", "n_matches", "FP",
                                      "FN"]}})
    with open(run_dir / "per_scene_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- correlation battery -------------------------------------------
    corrs: dict = {}
    for a in arms:
        sc = [r for r in per_scene[a].values() if r.get("ovtcs_C_mean") is not None]
        ov = [r["ovtcs_C_mean"] for r in sc]
        corrs[a] = {m: _corr(ov, [r[m] for r in sc]) for m in MOT_COLS}
    # pooled over arms
    sc_all = [r for a in arms for r in per_scene[a].values()
              if r.get("ovtcs_C_mean") is not None]
    corrs["pooled"] = {m: _corr([r["ovtcs_C_mean"] for r in sc_all],
                                [r[m] for r in sc_all]) for m in MOT_COLS}
    # per-scene paired deltas (global - ego): does ΔOV-TCS track ΔMOT?
    deltas = {}
    if set(arms) >= {"ego", "global"}:
        common = sorted(set(per_scene["ego"]) & set(per_scene["global"]),
                        key=int)
        common = [s for s in common
                  if per_scene["ego"][s].get("ovtcs_C_mean") is not None
                  and per_scene["global"][s].get("ovtcs_C_mean") is not None]
        d_ov = [per_scene["global"][s]["ovtcs_C_mean"]
                - per_scene["ego"][s]["ovtcs_C_mean"] for s in common]
        for m in MOT_COLS:
            d_m = [per_scene["global"][s][m] - per_scene["ego"][s][m]
                   for s in common]
            deltas[m] = _corr(d_ov, d_m)
        # sign agreement of the flagship direction, per scene
        for m in ("HOTA", "AssA", "IDF1"):
            d_m = np.asarray([per_scene["global"][s][m] - per_scene["ego"][s][m]
                              for s in common])
            d_o = np.asarray(d_ov)
            nz = (d_m != 0) & (d_o != 0)
            deltas[m]["sign_agreement"] = (
                float(np.mean(np.sign(d_m[nz]) == np.sign(d_o[nz])))
                if nz.any() else None)
    corrs["delta_global_minus_ego"] = deltas
    (run_dir / "correlations.json").write_text(json.dumps(corrs, indent=2))

    # ---- disagreement scenes (rank divergence OV-TCS vs HOTA, per arm) --
    disagree = {}
    for a in arms:
        sc = [(int(k), r) for k, r in per_scene[a].items()
              if r.get("ovtcs_C_mean") is not None]
        ov_rank = stats.rankdata([r["ovtcs_C_mean"] for _, r in sc])
        ho_rank = stats.rankdata([r["HOTA"] for _, r in sc])
        dv = np.abs(ov_rank - ho_rank)
        order = np.argsort(-dv)[:10]
        disagree[a] = [{"scene_idx": sc[i][0],
                        "scene_token": sc[i][1]["scene_token"],
                        "rank_gap": float(dv[i]),
                        "ovtcs_C": sc[i][1]["ovtcs_C_mean"],
                        "HOTA": sc[i][1]["HOTA"], "AssA": sc[i][1]["AssA"],
                        "DetA": sc[i][1]["DetA"], "IDF1": sc[i][1]["IDF1"],
                        "FRAG": sc[i][1]["FRAG"], "IDS": sc[i][1]["IDS"]}
                       for i in order]
    (run_dir / "disagreement_scenes.json").write_text(
        json.dumps(disagree, indent=2))

    # ---- system-level markdown table ------------------------------------
    lines = ["# E1 — OV-TCS vs MOT metrics (nuScenes val, cache replay)", ""]
    lines += ["## System level", "",
              "| Arm | OV-TCS_C | HOTA | DetA | AssA | IDF1 | IDP | IDR | "
              "MOTA | MOTP(m) | IDS | FRAG | AMOTA | AMOTP |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = summary["arms"][a]
        h, c = s["overall_hota"], s["overall_clear_id"]
        dk = s.get("devkit", {})
        fmt = lambda v: ("—" if v is None else
                         f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append(
            f"| {a} | {s['ovtcs']['ov_tcs']['C_mean']:.4f} | {h['HOTA']:.4f} "
            f"| {h['DetA']:.4f} | {h['AssA']:.4f} | {c['IDF1']:.4f} "
            f"| {c['IDP']:.4f} | {c['IDR']:.4f} | {c['MOTA']:.4f} "
            f"| {fmt(c['MOTP_m'])} | {c['IDS']} | {c['FRAG']} "
            f"| {fmt(dk.get('amota'))} | {fmt(dk.get('amotp'))} |")
    lines += ["", "Class-agnostic, 2.0 m BEV center-distance gate, 10-class GT; "
              "AMOTA/AMOTP from the official devkit (7 tracking classes).", ""]

    lines += ["## Per-scene correlation with OV-TCS_C", "",
              "| Metric | " + " | ".join(
                  f"{a} r (Spearman)" for a in arms + ["pooled"])
              + " | Δ(global−ego) r |", "|---|" + "---|" * (len(arms) + 2)]
    for m in MOT_COLS:
        cells = []
        for a in arms + ["pooled"]:
            c = corrs[a][m]
            cells.append(f"{c.get('pearson_r', float('nan')):.3f} "
                         f"({c.get('spearman_r', float('nan')):.3f})"
                         if "pearson_r" in c else "—")
        d = deltas.get(m, {})
        cells.append(f"{d.get('pearson_r', float('nan')):.3f}"
                     if "pearson_r" in d else "—")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    (run_dir / "mot_compare_table.md").write_text("\n".join(lines))

    # ---- figure ----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = ["HOTA", "AssA", "DetA", "IDF1"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for ax, m in zip(axes.flat[:4], panels):
        for a in arms:
            sc = [r for r in per_scene[a].values()
                  if r.get("ovtcs_C_mean") is not None]
            ax.scatter([r["ovtcs_C_mean"] for r in sc], [r[m] for r in sc],
                       s=14, alpha=0.55, lw=0, color=ARM_COLOR.get(a, "#666"),
                       label=f"{a} (ρ={corrs[a][m].get('spearman_r', float('nan')):.2f})")
        ax.set_xlabel("per-scene OV-TCS$_C$")
        ax.set_ylabel(m)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=8)
    # delta panels: ΔOV-TCS vs ΔHOTA / ΔAssA
    if deltas:
        common = sorted(set(per_scene["ego"]) & set(per_scene["global"]), key=int)
        common = [s for s in common
                  if per_scene["ego"][s].get("ovtcs_C_mean") is not None
                  and per_scene["global"][s].get("ovtcs_C_mean") is not None]
        d_ov = [per_scene["global"][s]["ovtcs_C_mean"]
                - per_scene["ego"][s]["ovtcs_C_mean"] for s in common]
        for ax, m in zip(axes.flat[4:], ["HOTA", "AssA"]):
            d_m = [per_scene["global"][s][m] - per_scene["ego"][s][m]
                   for s in common]
            ax.scatter(d_ov, d_m, s=14, alpha=0.55, lw=0, color="#333333")
            ax.axhline(0, color="#999", lw=0.6)
            ax.axvline(0, color="#999", lw=0.6)
            ax.set_xlabel("Δ OV-TCS$_C$ (global − ego)")
            ax.set_ylabel(f"Δ {m} (global − ego)")
            d = deltas[m]
            ax.set_title(f"per-scene deltas, r={d.get('pearson_r', float('nan')):.2f}",
                         fontsize=9)
            ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle("OV-TCS vs GT-track MOT metrics — nuScenes val, per scene",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(run_dir / "fig_ovtcs_vs_mot.png", dpi=200)
    fig.savefig(run_dir / "fig_ovtcs_vs_mot.pdf")
    print(f"wrote {run_dir}/mot_compare_table.md per_scene_metrics.csv "
          f"correlations.json disagreement_scenes.json fig_ovtcs_vs_mot.png")


if __name__ == "__main__":
    main()
