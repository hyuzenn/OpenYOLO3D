"""Aggregate the M22 pure-EMA incoming-weight sweep (w = α·c·k, NO OV-TCS).

Axes: every `axis_M22_emak_*` dir under --run-dir. k is read from
summary.json["kwargs"]["const_k"] (robust — no name parsing). Per axis reads:
  metrics.json          AP / AP_50 / AP_25 / AR / RC_50 / RC_25
  temporal_metrics.json LSC (total), TTC (mean)
  ovtcs_diagnostics.json feature_drift_mean, updates_applied_total/_per_track

Writes notes.md: the sweep table, the four requested curves (AP/drift/LSC/
update-count vs k), and the H1 verdict + the quantitative report (optimal k,
drift reduction, AP gain, update efficiency). CPU-only — safe on the util node.

H1 decision (over-update hypothesis):
  * AP rises monotonically as k decreases below 1.0  → supports H1.
  * interior peak near k≈0.335                       → promote {k=1.0, best-k}
                                                        to 312-scene full val.
NOTE: const_scale never DROPS updates, so applied-update count is k-invariant
by construction — effective-α (=α·k) and drift are the real step-size signals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EMA_ALPHA = 0.7  # install_method_22 default; effective incoming weight = α·c·k


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _f(x, fmt="{:.4f}"):
    return fmt.format(x) if isinstance(x, (int, float)) else "—"


def _collect(run_dir: Path) -> list[dict]:
    rows = []
    for d in sorted(run_dir.glob("axis_M22_emak_*")):
        summ = _load(d / "summary.json")
        k = (summ.get("kwargs") or {}).get("const_k")
        if k is None:
            continue
        m = _load(d / "metrics.json").get("metrics", {}).get("average", {})
        t = _load(d / "temporal_metrics.json")
        o = _load(d / "ovtcs_diagnostics.json")
        rows.append({
            "k": float(k),
            "AP": m.get("AP"), "AP_50": m.get("AP_50"), "AP_25": m.get("AP_25"),
            "AR": m.get("AR"), "RC_50": m.get("RC_50"), "RC_25": m.get("RC_25"),
            "LSC": (t.get("label_switch_count") or {}).get("total"),
            "TTC": (t.get("time_to_confirm") or {}).get("mean"),
            "drift": o.get("feature_drift_mean"),
            "upd_total": o.get("updates_applied_total"),
            "upd_per_track": o.get("updates_applied_per_track"),
        })
    rows.sort(key=lambda r: r["k"], reverse=True)  # 1.0 → 0.1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=str)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    rows = _collect(run_dir)

    L = ["# M22 pure-EMA incoming-weight sweep — w = α·c·k (NO OV-TCS)\n",
         f"Run: `{run_dir.name}`  ·  α={EMA_ALPHA}  ·  effective incoming weight = α·c·k\n"]

    if not rows:
        L.append("**No `axis_M22_emak_*` results found.**")
        (run_dir / "notes.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    ks = [r["k"] for r in rows]
    L.append("## Sweep table (k high→low)\n")
    L.append("| k | AP | AP_50 | AP_25 | AR | RC_50 | RC_25 | LSC | TTC | drift | upd/track |")
    L.append("|---|" + "---|" * 10)
    for r in rows:
        L.append("| " + " | ".join([
            f"{r['k']:g}", _f(r["AP"]), _f(r["AP_50"]), _f(r["AP_25"]), _f(r["AR"]),
            _f(r["RC_50"]), _f(r["RC_25"]), _f(r["LSC"], "{:.0f}"), _f(r["TTC"]),
            _f(r["drift"]), _f(r["upd_per_track"], "{:.3f}"),
        ]) + " |")
    L.append("")

    def curve(title, key, fmt="{:.4f}"):
        L.append(f"**{title} vs k:**  " +
                 "  ".join(f"k={r['k']:g}:{_f(r[key], fmt)}" for r in rows))

    L.append("## Curves\n")
    curve("AP", "AP")
    curve("feature drift", "drift")
    curve("LSC", "LSC", "{:.0f}")
    curve("applied update count", "upd_total", "{:.0f}")
    L.append("")

    # ---- verdict + quantitative report ----------------------------------
    have_ap = [r for r in rows if isinstance(r["AP"], (int, float))]
    L.append("## Verdict — H1 over-update hypothesis\n")
    if not have_ap:
        L.append("(no AP values — cannot decide)")
        (run_dir / "notes.md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return

    base = next((r for r in rows if abs(r["k"] - 1.0) < 1e-9), None)
    best = max(have_ap, key=lambda r: r["AP"])

    # Monotonic AP rise as k decreases (read on the descending-k order).
    aps = [r["AP"] for r in have_ap]  # already k high→low
    monotonic = all(aps[i + 1] >= aps[i] - 1e-9 for i in range(len(aps) - 1)) and aps[-1] > aps[0]
    interior_peak = best["k"] not in (max(ks), min(ks))
    near_335 = abs(best["k"] - 0.335) <= 0.165  # 0.25 ≤ best ≤ 0.50 bracket

    L.append(f"- **optimal k = {best['k']:g}** (AP {best['AP']:.4f})")
    if base and isinstance(base["AP"], (int, float)):
        d_ap = best["AP"] - base["AP"]
        L.append(f"- **AP gain vs k=1.0 (w=α·c baseline)**: {d_ap:+.4f} "
                 f"({d_ap / base['AP'] * 100:+.1f}%)" if base["AP"] else
                 f"- **AP gain vs k=1.0**: {d_ap:+.4f}")
        if isinstance(base["drift"], (int, float)) and isinstance(best["drift"], (int, float)) and base["drift"]:
            dd = base["drift"] - best["drift"]
            L.append(f"- **drift reduction vs k=1.0**: {dd:+.4f} "
                     f"({dd / base['drift'] * 100:+.1f}%); effective-α {EMA_ALPHA:g}→{EMA_ALPHA * best['k']:.3f}")
    # Update efficiency — const_scale applies every update; count should be flat.
    counts = {r["upd_total"] for r in rows if isinstance(r["upd_total"], (int, float))}
    if len(counts) <= 1:
        L.append("- **update efficiency**: applied-update count is k-INVARIANT "
                 "(const_scale down-weights, never drops). Efficiency = effective "
                 "step size (drift / α·k), not count.")
    else:
        L.append(f"- **applied-update count varies across k** {sorted(counts)} "
                 "(unexpected for const_scale — check tau_skip plumbing).")

    L.append("")
    if monotonic:
        L.append("**→ AP rises monotonically as k decreases: SUPPORTS the EMA "
                 "over-update hypothesis.** Smaller incoming weight helps.")
    if interior_peak and near_335:
        L.append(f"**→ interior AP peak at k={best['k']:g} (≈0.335 region): promote "
                 f"only {{k=1.0, k={best['k']:g}}} to 312-scene full validation.**")
    elif interior_peak:
        L.append(f"**→ interior AP peak at k={best['k']:g} (outside the ≈0.335 "
                 f"bracket): promote {{k=1.0, k={best['k']:g}}} to full validation.**")
    elif not monotonic:
        L.append(f"**→ peak at boundary k={best['k']:g}, non-monotonic interior: "
                 "no clean over-update signal — inspect the curve before promoting.**")

    (run_dir / "notes.md").write_text("\n".join(L) + "\n")
    print(f"[aggregate_m22_emak_sweep] wrote {run_dir / 'notes.md'}")
    print("\n".join(L))


if __name__ == "__main__":
    main()
