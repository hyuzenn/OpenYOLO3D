#!/usr/bin/env python3
"""Consolidate the Hybrid-Proposal v2 ablation into one table + attribution.

Consumes (all produced by run_hybrid_v2_eval.pbs + build_hybrid_cache.pbs):
  <run>/eval_gamma/axis_baseline/metrics.json     baseline (native CenterPoint class)
  <run>/eval_hybrid/axis_baseline/metrics.json    hybrid   (YOLO-World relabel)
  <run>/recall/recall_probe.json                  proposal recall (gamma, hybrid) + oracle
  <build_stats.json>                              label-transfer stats (optional)

Emits <run>/ablation.json + prints a markdown table. The geometry of the two
arms is identical by construction (hybrid reuses CenterPoint boxes), so:
  * proposal-stage recall is expected identical -> verifies attribution,
  * any mAP/NDS/per-class delta is purely the open-vocab relabel (classification),
  * native->oracle gap closed = (hybrid-gamma)/(oracle-gamma).
"""
from __future__ import annotations
import argparse, json, os

NUSC_10 = ("car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fnum(x, p=4):
    return "  --  " if x is None else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="eval run dir")
    ap.add_argument("--build-stats", default=None)
    args = ap.parse_args()
    R = args.run

    g = load(os.path.join(R, "eval_gamma/axis_baseline/metrics.json"))
    h = load(os.path.join(R, "eval_hybrid/axis_baseline/metrics.json"))
    rp = load(os.path.join(R, "recall/recall_probe.json"))
    bs = load(args.build_stats) if args.build_stats else None
    if g is None or h is None:
        raise SystemExit(f"missing eval metrics under {R}")

    out = {"run": R}
    lines = []
    P = lines.append

    # ---------------- Detection stage ----------------
    gmap, hmap = g.get("mAP"), h.get("mAP")
    gnds, hnds = g.get("NDS"), h.get("NDS")
    dmap = (hmap - gmap) if (gmap is not None and hmap is not None) else None
    P("## Detection stage (val-150, geometry held fixed)\n")
    P("| metric | baseline (gamma) | hybrid (YOLO relabel) | Δ |")
    P("|---|---|---|---|")
    P(f"| mAP | {fnum(gmap)} | {fnum(hmap)} | {fnum(dmap,4)} |")
    P(f"| NDS | {fnum(gnds)} | {fnum(hnds)} | "
      f"{fnum((hnds-gnds) if (gnds is not None and hnds is not None) else None)} |")
    a_g, a_h = g.get("fire_audit", {}), h.get("fire_audit", {})
    P(f"| emitted boxes | {a_g.get('n_emitted_total')} | {a_h.get('n_emitted_total')} | "
      f"{(a_h.get('n_emitted_total',0)-a_g.get('n_emitted_total',0))} |")
    P(f"| proposals total | {a_g.get('n_proposals_total')} | {a_h.get('n_proposals_total')} | |")
    out["detection"] = {"baseline": {"mAP": gmap, "NDS": gnds,
                                     "emit": a_g.get("n_emitted_total"),
                                     "prop": a_g.get("n_proposals_total")},
                        "hybrid": {"mAP": hmap, "NDS": hnds,
                                   "emit": a_h.get("n_emitted_total"),
                                   "prop": a_h.get("n_proposals_total")},
                        "delta_mAP": dmap}

    # ---------------- Per-class AP ----------------
    gap = g.get("per_class_AP") or {}
    hap = h.get("per_class_AP") or {}
    P("\n## Per-class AP\n")
    P("| class | gamma AP | hybrid AP | Δ |")
    P("|---|---|---|---|")
    perclass = {}
    for c in NUSC_10:
        ga, ha = gap.get(c), hap.get(c)
        d = (ha - ga) if (ga is not None and ha is not None) else None
        perclass[c] = {"gamma": ga, "hybrid": ha, "delta": d}
        P(f"| {c} | {fnum(ga,3)} | {fnum(ha,3)} | {fnum(d,3)} |")
    out["per_class_AP"] = perclass
    deltas = [(c, v["delta"]) for c, v in perclass.items() if v["delta"] is not None]
    if deltas:
        deltas.sort(key=lambda kv: kv[1])
        out["per_class_losers"] = deltas[:3]
        out["per_class_winners"] = deltas[-3:][::-1]

    # ---------------- Proposal recall (expected identical) ----------------
    if rp:
        ps = rp.get("per_source", {})
        P("\n## Proposal stage (class-agnostic, geometry-only)\n")
        P("| metric | gamma | hybrid | identical? |")
        P("|---|---|---|---|")
        rec = {}
        def micro(src, key, thr):
            return (ps.get(src, {}).get(key, {}).get(thr, {}).get("micro_overall"))
        rows = [("GT coverage @2m", "recall_box", "thr_2.0m"),
                ("GT coverage @4m", "recall_box", "thr_4.0m"),
                ("Recall@IoU0.25", "recall_iou3d", "thr_0.25"),
                ("Recall@IoU0.50", "recall_iou3d", "thr_0.50"),
                ("Recall@IoU0.70", "recall_iou3d", "thr_0.70")]
        for lab, key, thr in rows:
            mg, mh = micro("gamma", key, thr), micro("hybrid", key, thr)
            same = (mg is not None and mh is not None and abs(mg - mh) < 1e-9)
            rec[lab] = {"gamma": mg, "hybrid": mh, "identical": same}
            P(f"| {lab} | {fnum(mg)} | {fnum(mh)} | {'YES' if same else 'no'} |")
        out["proposal_recall"] = rec
        # oracle ceiling (per source)
        og = ps.get("gamma", {}).get("oracle_map", {})
        oh = ps.get("hybrid", {}).get("oracle_map", {})
        oracle_map = og.get("mAP")
        out["oracle"] = {"gamma_oracle_mAP": og.get("mAP"),
                         "hybrid_oracle_mAP": oh.get("mAP")}
        P(f"\noracle-mAP ceiling: gamma={fnum(og.get('mAP'))} hybrid={fnum(oh.get('mAP'))}")
        # native->oracle gap closed
        if gmap is not None and hmap is not None and oracle_map is not None:
            gap_total = oracle_map - gmap
            gap_closed = (hmap - gmap) / gap_total if abs(gap_total) > 1e-9 else None
            out["gap_closed_native_to_oracle"] = gap_closed
            P(f"native->oracle gap: total={fnum(gap_total)}, "
              f"hybrid closes {fnum(gap_closed,3) if gap_closed is not None else 'n/a'} "
              f"of it")

    # ---------------- Temporal stage ----------------
    def temporal_row(m, src):
        t = m.get("temporal", {})
        vm = m.get("variant_metrics", {}) or {}
        ov = vm.get("ov_tcs", {})
        tl = vm.get("track_length", {})
        fr = vm.get("gt_fragmentation", {})
        return {"src": src,
                "lsc": t.get("label_switch_count_total"),
                "ov_tcs_C": ov.get("C_mean"),
                "track_len_mean": tl.get("mean"),
                "gt_frag_mean": fr.get("mean_fragments")}
    tg, th_ = temporal_row(g, "gamma"), temporal_row(h, "hybrid")
    P("\n## Temporal stage\n")
    P("| metric | gamma | hybrid |")
    P("|---|---|---|")
    for k in ("lsc", "ov_tcs_C", "track_len_mean", "gt_frag_mean"):
        P(f"| {k} | {tg[k]} | {th_[k]} |")
    out["temporal"] = {"gamma": tg, "hybrid": th_}

    # ---------------- Runtime ----------------
    P("\n## Runtime\n")
    P(f"| stage | gamma walltime_s | hybrid walltime_s |")
    P("|---|---|---|")
    P(f"| eval (cache-only) | {fnum(g.get('axis_walltime_s'),1)} | "
      f"{fnum(h.get('axis_walltime_s'),1)} |")
    out["runtime"] = {"gamma_eval_s": g.get("axis_walltime_s"),
                      "hybrid_eval_s": h.get("axis_walltime_s")}
    if bs:
        rt = bs.get("runtime", {})
        ys = rt.get("yolo_per_sample_s", {})
        P(f"\nhybrid build: YOLO {ys.get('mean')}s/sample (6 cams), "
          f"total walltime {rt.get('total_walltime_s')}s for {bs['counts']['n_samples']} samples")
        out["build_runtime"] = rt
        out["transfer"] = {
            "matched_rate": bs["counts"].get("matched_rate"),
            "matched_rate_of_projected": bs["counts"].get("matched_rate_of_projected"),
            "cp_label_agreement": bs["label_distribution"].get("cp_label_agreement"),
            "label_dist": bs["label_distribution"].get("labels"),
            "by_distance": bs.get("by_distance"),
        }

    txt = "\n".join(lines)
    print(txt)
    json.dump(out, open(os.path.join(R, "ablation.json"), "w"), indent=2)
    open(os.path.join(R, "ablation_table.md"), "w").write(txt + "\n")
    print(f"\nwrote {os.path.join(R, 'ablation.json')} and ablation_table.md")


if __name__ == "__main__":
    main()
