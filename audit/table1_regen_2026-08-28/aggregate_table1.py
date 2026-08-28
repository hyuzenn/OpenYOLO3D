"""Table-1 regeneration, phase 4: collect the eight arms, compare against the
published values, run the sanity checks, emit the audit artifacts.

Writes
  table1_regenerated_results.json   machine-readable, one record per row
  TABLE1_REGENERATION_REPORT.md     human-readable report

Reads the arms produced by phases 2 and 3. Cell metrics are extracted with
audit/cbmot/aggregate.py::read_axis, the same reader that built the published
CBMOT table, so the regenerated numbers are not read by a second, divergent
code path.
"""
from __future__ import annotations

import json
import os
import os.path as osp
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/rintern16/OpenYOLO3D")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "audit/cbmot"))

AUDIT = ROOT / "audit/table1_regen_2026-08-28"
PHASE1 = AUDIT / "phase1_cpcache"
PHASE2 = AUDIT / "phase2_arms/cells"
PHASE3 = AUDIT / "phase3_cbmot/cells"
CACHE_10 = ROOT / "results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity"
CACHE_SINGLE = ROOT / "results/outdoor_native_temporal_cpcache_thr000_single_gravity"

# 10-sweep CenterPoint reference anchor (audit/official_centerpoint_ref,
# devkit 0.5580/0.6458 vs corrected custom 0.558050/0.645809).
ANCHOR = {"mAP": 0.558050, "NDS": 0.645809}

# Published Table 1 (paper_iccv_draft/sec/3_finalcopy.tex, tab:detmatch).
# Frozen here so old-vs-new is a comparison against what the paper actually says.
PUBLISHED = {
    ("Sensor", "Baseline (unfiltered)"):  dict(boxes=1029380, mAP=0.3408, NDS=0.3150, HOTA=0.1372, AssA=0.1896, DetA=0.1007, DetRe=0.5904, IDF1=0.0759, AMOTA=0.0501),
    ("Sensor", "Control (threshold)"):    dict(boxes=360309,  mAP=0.3324, NDS=0.3110, HOTA=0.2098, AssA=0.2044, DetA=0.2184, DetRe=0.5182, IDF1=0.1517, AMOTA=0.0547),
    ("Sensor", "Control (accumulation)"): dict(boxes=360309,  mAP=None,   NDS=None,   HOTA=0.2036, AssA=0.2217, DetA=0.1890, DetRe=0.4601, IDF1=0.1596, AMOTA=0.0701),
    ("Sensor", "Confirmation (N=3)"):     dict(boxes=360309,  mAP=0.2023, NDS=0.2518, HOTA=0.1960, AssA=0.2535, DetA=0.1537, DetRe=0.3859, IDF1=0.1531, AMOTA=0.0887),
    ("World",  "Baseline (unfiltered)"):  dict(boxes=1029380, mAP=0.3408, NDS=0.3150, HOTA=0.2011, AssA=0.4089, DetA=0.0995, DetRe=0.5838, IDF1=0.1764, AMOTA=0.1580),
    ("World",  "Control (threshold)"):    dict(boxes=754500,  mAP=0.3396, NDS=0.3143, HOTA=0.2307, AssA=0.4175, DetA=0.1282, DetRe=0.5668, IDF1=0.2190, AMOTA=0.1655),
    ("World",  "Control (accumulation)"): dict(boxes=754500,  mAP=None,   NDS=None,   HOTA=0.2307, AssA=0.4190, DetA=0.1278, DetRe=0.5650, IDF1=0.2212, AMOTA=0.1612),
    ("World",  "Confirmation (N=3)"):     dict(boxes=754500,  mAP=0.2900, NDS=0.3096, HOTA=0.2263, AssA=0.4314, DetA=0.1196, DetRe=0.5326, IDF1=0.2121, AMOTA=0.2033),
}

# row order -> (frame label, arm label, cell dir)
ROWS = [
    ("Sensor", "Baseline (unfiltered)",  PHASE2 / "gamma_ego"),
    ("Sensor", "Control (threshold)",    PHASE2 / "ctrl_ego"),
    ("Sensor", "Control (accumulation)", PHASE3 / "cbmot_retro_ego_N3_parallel_addition_noise0.05"),
    ("Sensor", "Confirmation (N=3)",     PHASE2 / "retro_ego"),
    ("World",  "Baseline (unfiltered)",  PHASE2 / "gamma_global"),
    ("World",  "Control (threshold)",    PHASE2 / "ctrl_global"),
    ("World",  "Control (accumulation)", PHASE3 / "cbmot_retro_global_N3_parallel_addition_noise0.05"),
    ("World",  "Confirmation (N=3)",     PHASE2 / "retro_global"),
]

TP = [("mATE", "trans_err"), ("mASE", "scale_err"), ("mAOE", "orient_err"),
      ("mAVE", "vel_err"), ("mAAE", "attr_err")]

problems: list[str] = []
notes: list[str] = []


def flag(msg):
    problems.append(msg)
    print(f"  [FLAG] {msg}", flush=True)


def collect() -> list[dict]:
    from aggregate import read_axis            # audit/cbmot/aggregate.py

    out = []
    for frame, arm, cell in ROWS:
        rec = {"frame": frame, "arm": arm, "cell": str(cell)}
        axes = sorted(cell.glob("axis_*")) if cell.exists() else []
        if not axes:
            rec["status"] = "MISSING"
            flag(f"{frame}/{arm}: no cell at {cell}")
            out.append(rec)
            continue
        axis = axes[0]
        rec["status"] = "OK"
        rec["axis"] = str(axis.relative_to(ROOT))
        rec.update(read_axis(axis))
        m = axis / "metrics.json"
        if m.exists():
            j = json.loads(m.read_text())
            rec["n_samples"] = j.get("n_samples")
            for label, key in TP:
                rec[label] = (j.get("tp_errors") or {}).get(key)
            t = j.get("temporal") or {}
            rec["label_switch_count_total"] = t.get("label_switch_count_total")
            ttc = t.get("time_to_confirm") or {}
            rec["TTC_mean"] = ttc.get("mean")
            rec["TTC_median"] = ttc.get("median")
            rec["TTC_p90"] = ttc.get("p90")
            rec["n_gt_boxes_total"] = j.get("n_gt_boxes_total")
        out.append(rec)
    return out


def sanity(rows: list[dict]) -> dict:
    print("\n=== SANITY CHECKS ===", flush=True)
    res = {}

    # A. metric ranges + the NDS < mAP red flag
    print("A. metric range / NDS>=mAP", flush=True)
    for r in rows:
        if r.get("status") != "OK":
            continue
        for k in ("mAP", "NDS"):
            v = r.get(k)
            if v is not None and not (0.0 <= v <= 1.0):
                flag(f"{r['frame']}/{r['arm']}: {k}={v} outside [0,1]")
        if r.get("mAP") is not None and r.get("NDS") is not None:
            if r["NDS"] < r["mAP"]:
                flag(f"{r['frame']}/{r['arm']}: NDS ({r['NDS']:.4f}) < mAP "
                     f"({r['mAP']:.4f}) — investigate before accepting")
    res["A_range_and_nds_ge_map"] = "see flags"

    # C. attribute sanity: mAAE == 1.0 means every attribute was wrong, the
    # signature of the pre-#8 heuristic.
    print("C. attribute rule active", flush=True)
    for r in rows:
        if r.get("mAAE") is not None and abs(r["mAAE"] - 1.0) < 1e-9:
            flag(f"{r['frame']}/{r['arm']}: mAAE==1.0 — old attribute heuristic?")
    res["C_attr"] = "checked"

    # E. sample count
    print("E. sample count == 6019", flush=True)
    for r in rows:
        n = r.get("n_samples")
        if n is not None and n != 6019:
            flag(f"{r['frame']}/{r['arm']}: n_samples={n}, expected 6019")
    res["E_sample_count"] = "checked"

    # F. no legacy path
    print("F. no legacy single-sweep / legacy evaluator path", flush=True)
    for r in rows:
        src = r.get("cell", "")
        if "single_gravity" in src or "2026-07-18_e1_grid" in src \
                or "2026-07-30_e2c" in src:
            flag(f"{r['frame']}/{r['arm']}: cell points at a single-sweep artifact")
    probe = PHASE1 / "sweep_runtime_probe.json"
    if probe.exists():
        d = json.loads(probe.read_text())
        ok = (d["multi_sweep"] and d["num_sweeps"] == 10
              and all(p["n_channels"] == 5 and p["n_distinct_dt_ms"] == 10
                      for p in d["probes"]))
        res["D_sweep_runtime"] = d
        print(f"D. sweep runtime probe: {'PASS' if ok else 'FAIL'} "
              f"(multi_sweep={d['multi_sweep']}, num_sweeps={d['num_sweeps']}, "
              f"{d['probes'][0]['n_points']} pts / {d['probes'][0]['n_channels']} ch "
              f"/ {d['probes'][0]['n_distinct_dt_ms']} dt)", flush=True)
        if not ok:
            flag("phase-1 sweep runtime probe did not show 10-sweep input")
    else:
        flag("phase-1 sweep runtime probe missing — cannot prove 10-sweep input")
    res["F_cache"] = str(CACHE_10)
    n_cache = len(list(CACHE_10.glob("*.pkl"))) if CACHE_10.exists() else 0
    res["F_cache_files"] = n_cache
    if n_cache != 6019:
        flag(f"10-sweep cache has {n_cache} files, expected 6019")

    # baseline anchor
    print("B/§6. CenterPoint baseline vs 10-sweep anchor", flush=True)
    base = next((r for r in rows if r["arm"].startswith("Baseline")
                 and r["frame"] == "Sensor"), None)
    if base and base.get("mAP") is not None:
        d_map = base["mAP"] - ANCHOR["mAP"]
        d_nds = (base.get("NDS") or 0) - ANCHOR["NDS"]
        res["anchor"] = {"anchor": ANCHOR, "baseline_mAP": base["mAP"],
                         "baseline_NDS": base.get("NDS"),
                         "delta_mAP": d_map, "delta_NDS": d_nds}
        print(f"   baseline mAP {base['mAP']:.6f} vs anchor {ANCHOR['mAP']:.6f} "
              f"(delta {d_map:+.6f})", flush=True)
        notes.append(
            "The unfiltered baseline is an unthresholded (score>=0) emission over "
            "tracked boxes, whereas the 0.5580 anchor is mmdet3d's own thresholded "
            "submission of the same checkpoint; they are expected to be close but "
            "not identical. A large negative gap would indicate the detector input "
            "is still wrong.")
        if abs(d_map) > 0.05:
            flag(f"baseline mAP differs from the 10-sweep anchor by {d_map:+.4f} "
                 f"— diagnose (checkpoint/config/split/sweeps/threshold/NMS/"
                 f"formatting/evaluator) before accepting")
    return res


def render(rows: list[dict], checks: dict) -> str:
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   text=True).strip()
    L = []
    A = L.append
    A("# Table 1 regeneration — results\n")
    A(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M %Z')} · git `{head}`\n")
    A("Corrected evaluator `ad94732` + official prediction-attribute rule "
      "`438f121` + **10-sweep** CenterPoint input.\n")
    A("Manuscript NOT modified. These are candidate numbers pending review.\n")

    A("\n## 1. Regenerated Table 1\n")
    A("| Frame | Arm | Boxes | mAP | NDS | mATE | mASE | mAOE | mAVE | mAAE |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(k, n=4):
            v = r.get(k)
            return "---" if v is None else f"{v:.{n}f}"
        b = r.get("n_boxes")
        A(f"| {r['frame']} | {r['arm']} | {b if b else '---'} | {f('mAP')} | "
          f"{f('NDS')} | {f('mATE')} | {f('mASE')} | {f('mAOE')} | {f('mAVE')} | "
          f"{f('mAAE')} |")

    A("\n## 2. Tracking metrics\n")
    A("| Frame | Arm | HOTA | AssA | DetA | DetRe | IDF1 | AMOTA | LSC | frag | tracks |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def f(k, n=4):
            v = r.get(k)
            return "---" if v is None else f"{v:.{n}f}"
        A(f"| {r['frame']} | {r['arm']} | {f('HOTA')} | {f('AssA')} | {f('DetA')} | "
          f"{f('DetRe')} | {f('IDF1')} | {f('amota')} | "
          f"{r.get('label_switch_count_total', '---')} | {f('frag', 2)} | "
          f"{r.get('n_tracks', '---')} |")

    A("\n## 3. Old (published) vs new (regenerated)\n")
    A("| Frame | Arm | Metric | Old | New | Abs Δ | Rel Δ |")
    A("|---|---|---|---:|---:|---:|---:|")
    for r in rows:
        pub = PUBLISHED.get((r["frame"], r["arm"]), {})
        for k, newk in (("mAP", "mAP"), ("NDS", "NDS"), ("HOTA", "HOTA"),
                        ("AssA", "AssA"), ("DetA", "DetA"), ("DetRe", "DetRe"),
                        ("IDF1", "IDF1"), ("AMOTA", "amota"), ("boxes", "n_boxes")):
            o, n = pub.get(k), r.get(newk)
            if o is None or n is None:
                continue
            d = n - o
            rel = f"{d / o:+.1%}" if o else "---"
            fmt = "{:.0f}" if k == "boxes" else "{:.4f}"
            A(f"| {r['frame']} | {r['arm']} | {k} | {fmt.format(o)} | "
              f"{fmt.format(n)} | {d:+.4f} | {rel} |")

    A("\n## 4. Sanity checks\n")
    a = checks.get("anchor")
    if a:
        A(f"- **Baseline vs 10-sweep anchor**: regenerated sensor baseline "
          f"mAP {a['baseline_mAP']:.6f} / NDS {a['baseline_NDS']:.6f} against the "
          f"anchor {a['anchor']['mAP']:.6f} / {a['anchor']['NDS']:.6f} "
          f"(Δ mAP {a['delta_mAP']:+.6f}, Δ NDS {a['delta_NDS']:+.6f}).")
    d = checks.get("D_sweep_runtime")
    if d:
        p = d["probes"][0]
        A(f"- **Sweep sanity (runtime, not config name)**: multi_sweep="
          f"{d['multi_sweep']}, num_sweeps={d['num_sweeps']}, "
          f"{p['n_points']} points / {p['n_channels']} channels / "
          f"{p['n_distinct_dt_ms']} distinct Δt.")
    A(f"- **Cache**: `{checks.get('F_cache')}` "
      f"({checks.get('F_cache_files')} files, expected 6019).")
    A("- **Evaluator**: corrected GT construction, official num_pts and bike-rack "
      "filters, official GT velocity/attributes, official prediction-attribute "
      "rule, devkit `accumulate`/`calc_ap`/`calc_tp` under "
      "`detection_cvpr_2019`.")
    for n in notes:
        A(f"- {n}")

    A("\n## 5. Flags\n")
    if problems:
        for p in problems:
            A(f"- ⚠️ {p}")
    else:
        A("- None.")

    A("\n## 6. Verdict\n")
    A("**NOT YET SAFE — REQUIRES INVESTIGATION**" if problems
      else "**SAFE TO UPDATE MANUSCRIPT** (subject to your review of §3).")
    A("\nThe manuscript was not modified by this task.\n")
    return "\n".join(L)


def main():
    print("=== Table-1 phase 4: aggregate ===", flush=True)
    rows = collect()
    checks = sanity(rows)
    payload = {
        "generated": datetime.now().isoformat(),
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "anchor": ANCHOR,
        "published_table1": {f"{k[0]}|{k[1]}": v for k, v in PUBLISHED.items()},
        "rows": rows,
        "checks": checks,
        "flags": problems,
        "verdict": ("NOT YET SAFE — REQUIRES INVESTIGATION" if problems
                    else "SAFE TO UPDATE MANUSCRIPT"),
    }
    (AUDIT / "table1_regenerated_results.json").write_text(
        json.dumps(payload, indent=2, default=str))
    (AUDIT / "TABLE1_REGENERATION_REPORT.md").write_text(render(rows, checks))
    print(f"\nwrote {AUDIT / 'table1_regenerated_results.json'}")
    print(f"wrote {AUDIT / 'TABLE1_REGENERATION_REPORT.md'}")
    print(f"\nVERDICT: {payload['verdict']}  ({len(problems)} flags)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
