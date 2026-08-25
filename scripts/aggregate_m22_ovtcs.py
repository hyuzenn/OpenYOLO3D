"""Aggregate the M22 OV-TCS-aware EMA smoke into a comparison + attribution.

Arms (axis dirs under --run-dir):
  M22_base_weighted  w = α·c            (confidence-only control)
  M22_ovtcs_scale    w = α·c·OVTCS_C    (OV-TCS-aware)
  M22_const_scale    w = α·c·k          (matched-average step-shrink control)

Reads per axis: metrics.json (AP/AP50/AP25/AR/RC_50), temporal_metrics.json
(LSC/TTC), ovtcs_diagnostics.json (updates/track, drift, online OV-TCS, corr).
Writes notes.md with the 3-arm table and the attribution reading:
  does ovtcs_scale beat a matched-average shrink (const_scale)?
CPU-only — safe on the util node. const_scale is optional (2-arm runs still work).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = [
    ("M22_base_weighted", "baseline"),
    ("M22_const_scale", "const_scale"),
    ("M22_ovtcs_scale", "ovtcs_scale"),
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _axis(run_dir: Path, name: str) -> dict:
    d = run_dir / f"axis_{name}"
    return {
        "metrics": _load(d / "metrics.json").get("metrics", {}).get("average", {}),
        "temporal": _load(d / "temporal_metrics.json"),
        "ovtcs": _load(d / "ovtcs_diagnostics.json"),
        "present": d.exists(),
    }


def _f(x, fmt="{:.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    arms = [(label, _axis(run_dir, name)) for name, label in ARMS]
    arms = [(label, a) for label, a in arms if a["present"]]
    labels = [label for label, _ in arms]

    def col(getter):
        return [getter(a) for _, a in arms]

    m = lambda a: a["metrics"]
    t = lambda a: a["temporal"]
    o = lambda a: a["ovtcs"]

    lines = ["# M22 OV-TCS-aware EMA — attribution (baseline / const_scale / ovtcs_scale)\n"]
    lines.append(f"Run: `{run_dir.name}`")
    lines.append("Arms: `baseline` w=α·c · `const_scale` w=α·c·k(0.335) · `ovtcs_scale` w=α·c·OVTCS_C\n")

    def table(title, rows):
        lines.append(f"## {title}\n")
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))
        for rlabel, getter, fmt in rows:
            vals = col(getter)
            lines.append(f"| {rlabel} | " + " | ".join(_f(v, fmt) for v in vals) + " |")
        lines.append("")

    table("Headline metrics", [
        ("AP", lambda a: m(a).get("AP"), "{:.4f}"),
        ("AP_50", lambda a: m(a).get("AP_50"), "{:.4f}"),
        ("AP_25", lambda a: m(a).get("AP_25"), "{:.4f}"),
        ("AR", lambda a: m(a).get("AR"), "{:.4f}"),
        ("RC_50", lambda a: m(a).get("RC_50"), "{:.4f}"),
        ("LSC (total)", lambda a: t(a).get("label_switch_count", {}).get("total"), "{:.0f}"),
        ("TTC (mean)", lambda a: t(a).get("time_to_confirm", {}).get("mean"), "{:.4f}"),
    ])
    table("Update dynamics & OV-TCS", [
        ("updates/track", lambda a: o(a).get("updates_applied_per_track"), "{:.3f}"),
        ("feature drift (mean)", lambda a: o(a).get("feature_drift_mean"), "{:.4f}"),
        ("online OV-TCS (mean)", lambda a: o(a).get("online_ovtcs_mean"), "{:.4f}"),
    ])

    # ---- attribution: ovtcs_scale vs const_scale -------------------------
    d = {label: a for label, a in arms}
    lines.append("## Attribution — does OV-TCS beat a matched-average shrink?\n")
    if "const_scale" in d and "ovtcs_scale" in d:
        base_ap = m(d["baseline"]).get("AP") if "baseline" in d else None
        for met, getter in [("AP", lambda a: m(a).get("AP")),
                            ("LSC", lambda a: t(a).get("label_switch_count", {}).get("total")),
                            ("feature drift", lambda a: o(a).get("feature_drift_mean"))]:
            cv, ov = getter(d["const_scale"]), getter(d["ovtcs_scale"])
            if isinstance(cv, (int, float)) and isinstance(ov, (int, float)):
                lines.append(f"- **{met}**: const_scale {cv:.4f} vs ovtcs_scale {ov:.4f} "
                             f"→ OV-TCS-specific Δ = {ov - cv:+.4f}")
        ap_b = m(d["baseline"]).get("AP") if "baseline" in d else None
        ap_c = m(d["const_scale"]).get("AP")
        ap_o = m(d["ovtcs_scale"]).get("AP")
        lines.append("")
        if all(isinstance(x, (int, float)) for x in (ap_b, ap_c, ap_o)):
            tot = ap_o - ap_b
            shrink = ap_c - ap_b
            specific = ap_o - ap_c
            lines.append(f"**AP decomposition vs baseline:** total {tot:+.4f} = "
                         f"generic-shrink {shrink:+.4f} (const) + OV-TCS-specific {specific:+.4f}.")
            if abs(specific) < 0.5 * abs(shrink) or specific <= 0:
                verdict = ("OV-TCS adds little beyond a matched-average shrink → the gain is "
                           "mostly a smaller effective α. Conclude generic EMA shrinkage; a "
                           "plain α/τ tweak is the simpler lever.")
            else:
                verdict = ("OV-TCS's per-update variation contributes a sizable share beyond the "
                           "matched shrink → real signal; piecewise/joint + 312-scene confirm justified.")
            lines.append(f"\n**Reading:** {verdict}")
    else:
        lines.append("(const_scale arm missing — run the 3-arm attribution to decide.)")
    lines.append("")

    (run_dir / "notes.md").write_text("\n".join(lines) + "\n")
    print(f"[aggregate_m22_ovtcs] wrote {run_dir / 'notes.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
