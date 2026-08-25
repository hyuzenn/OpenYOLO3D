"""Aggregate the M22 OFF-vs-EMA decisive comparison (3 arms, 10-scene smoke).

Research question: does temporal feature aggregation help at all, and if so
what is the correct update strength? Arms:
  baseline       — M22 OFF (no EMA; default per-frame labeling)
  M22_emak_1     — full weighted EMA, w = α·c·1.0
  M22_emak_0.335 — reduced EMA, w = α·c·0.335 (sweep optimum)

Per arm reads metrics.json (AP/AP_50/AP_25/AR/RC_50), temporal_metrics.json
(LSC total, TTC mean), ovtcs_diagnostics.json (feature_drift_mean — absent for
OFF by construction). Writes notes.md: the arm table, the three requested deltas
(OFF→k1.0, OFF→k0.335, k1.0→k0.335), the best arm, and the decision gate.

Decision gate (two-sided — separates "EMA helps" from "tuning helps"):
  k=0.335 AP > OFF AP AND > k=1.0 AP → promote {OFF,k1.0,k0.335} to 312-scene val.
  otherwise                          → hold the full run (EMA and/or tuning unproven).
CPU-only — safe on the util node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# axis dir name -> human label, in report order (OFF first)
ARMS = (
    ("axis_baseline", "OFF (baseline)"),
    ("axis_M22_emak_1", "k=1.0 (full EMA)"),
    ("axis_M22_emak_0.335", "k=0.335 (reduced)"),
)


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _f(x, fmt="{:.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def _row(run_dir: Path, dname: str, label: str) -> dict:
    d = run_dir / dname
    m = _load(d / "metrics.json").get("metrics", {}).get("average", {})
    t = _load(d / "temporal_metrics.json")
    o = _load(d / "ovtcs_diagnostics.json")
    return {
        "label": label,
        "AP": m.get("AP"), "AP_50": m.get("AP_50"), "AP_25": m.get("AP_25"),
        "AR": m.get("AR"), "RC_50": m.get("RC_50"),
        "LSC": (t.get("label_switch_count") or {}).get("total"),
        "TTC": (t.get("time_to_confirm") or {}).get("mean"),
        "drift": o.get("feature_drift_mean"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    rows = {dname: _row(run_dir, dname, label) for dname, label in ARMS}
    off = rows["axis_baseline"]
    k1 = rows["axis_M22_emak_1"]
    k0 = rows["axis_M22_emak_0.335"]

    n_scenes = (_load(run_dir / "axis_baseline" / "summary.json").get("n_scenes")
                or "?")
    L = ["# M22 OFF vs EMA — does temporal feature aggregation help?\n",
         f"Run: `{run_dir.name}`  ·  {n_scenes}-scene  ·  3 arms (OFF / k=1.0 / k=0.335)\n"]

    L.append("| arm | AP | AP_50 | AP_25 | AR | RC_50 | LSC | TTC | drift |")
    L.append("|---|" + "---|" * 8)
    for _, label in ARMS:
        r = next(rows[d] for d, lab in ARMS if lab == label)
        L.append("| " + " | ".join([
            label, _f(r["AP"]), _f(r["AP_50"]), _f(r["AP_25"]), _f(r["AR"]),
            _f(r["RC_50"]), _f(r["LSC"], "{:.0f}"), _f(r["TTC"]), _f(r["drift"]),
        ]) + " |")
    L.append("")

    def delta(a: dict, b: dict, name: str) -> None:
        if isinstance(a["AP"], (int, float)) and isinstance(b["AP"], (int, float)):
            d = b["AP"] - a["AP"]
            pct = f" ({d / a['AP'] * 100:+.1f}%)" if a["AP"] else ""
            L.append(f"- **{name}**: ΔAP {d:+.4f}{pct}")
        else:
            L.append(f"- **{name}**: ΔAP — (missing AP)")

    L.append("## Deltas\n")
    delta(off, k1, "OFF → k=1.0")
    delta(off, k0, "OFF → k=0.335")
    delta(k1, k0, "k=1.0 → k=0.335")
    L.append("")

    have = [r for r in rows.values() if isinstance(r["AP"], (int, float))]
    L.append("## Verdict — does temporal aggregation help?\n")
    if not have:
        L.append("(no AP values — cannot decide)")
        (run_dir / "notes.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    best = max(have, key=lambda r: r["AP"])
    L.append(f"- **best arm: {best['label']}** (AP {best['AP']:.4f})")

    have_gate = all(isinstance(r["AP"], (int, float)) for r in (k0, off, k1))
    L.append("")
    if not have_gate:
        L.append("**→ GATE INDETERMINATE: missing AP for OFF, k=1.0, or k=0.335.**")
    elif k0["AP"] > off["AP"] and k0["AP"] > k1["AP"]:
        L.append("**→ GATE PASS: k=0.335 AP beats BOTH OFF and k=1.0. EMA helps AND "
                 "weight tuning helps. Promote {OFF, k=1.0, k=0.335} to 312-scene "
                 "full validation.**")
    else:
        why = []
        if k0["AP"] <= off["AP"]:
            why.append("k=0.335 <= OFF (EMA itself unproven)")
        if k0["AP"] <= k1["AP"]:
            why.append("k=0.335 <= k=1.0 (weight tuning unproven)")
        L.append(f"**→ GATE FAIL: {'; '.join(why)}. HOLD the full run.**")

    (run_dir / "notes.md").write_text("\n".join(L) + "\n")
    print(f"[aggregate_m22_off_vs_ema] wrote {run_dir / 'notes.md'}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
