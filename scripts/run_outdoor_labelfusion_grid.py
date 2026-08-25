#!/usr/bin/env python
"""Outdoor 2D->3D label-fusion OPERATING-REGION map (GPU-free, no proposal jobs).

Maps the operating region of the class-aware label-fusion overlay (arm G) by
re-running ONLY the CPU evaluator on the single existing hybrid source cache,
with the fusion applied as a read-time overlay (no cache duplication, no build):

  Study 1  Allowlist expansion (tau_iou=0.5, tau_score=0.4 fixed)
    A0 native | A1 {bi,mo} | A2 +cone | A3 +ped | A4 +barrier
  Study 2  Threshold sweep on the BEST allowlist from Study 1
    tau_iou in {0.3,0.5,0.7} x tau_score in {0.1,0.3,0.5}

nuScenes is loaded once; every config is a separate begin_axis/run/aggregate on
one shared evaluator. Per config we keep the full evaluator artifacts plus the
GT-matched override audit (override precision, FP->TP / TP->FP conversion).
"""
import argparse, json, time
from pathlib import Path

from method_scannet.streaming.nuscenes_native_evaluator import (
    NativeTemporalNuScenesEvaluator, _list_val_scenes)
from dataloaders.nuscenes_loader import NuScenesLoader

SCORED = ["car", "truck", "bus", "trailer", "construction_vehicle",
          "pedestrian", "motorcycle", "bicycle", "traffic_cone", "barrier"]

ALLOWLISTS = [
    ("A0_native",     frozenset()),
    ("A1_bike_moto",  frozenset({"bicycle", "motorcycle"})),
    ("A2_cone",       frozenset({"bicycle", "motorcycle", "traffic_cone"})),
    ("A3_ped",        frozenset({"bicycle", "motorcycle", "traffic_cone",
                                 "pedestrian"})),
    ("A4_barrier",    frozenset({"bicycle", "motorcycle", "traffic_cone",
                                 "pedestrian", "barrier"})),
]
TAU_IOU_GRID = [0.3, 0.5, 0.7]
TAU_SCORE_GRID = [0.1, 0.3, 0.5]


def capmean(d):
    return sum(d.values()) / len(d) if d else 0.0


def run_one(ev, scenes, out_dir, allow, ti, ts):
    ev.fuse_allow = allow
    ev.fuse_tau_iou = float(ti)
    ev.fuse_tau_score = float(ts)
    ev.install_axis("baseline")
    ev.begin_axis()
    t0 = time.time()
    for i, sc in enumerate(scenes):
        try:
            ev.run_scene(sc, scene_idx=i)
        except Exception as exc:
            print(f"    SCENE FAILED {sc[:8]}: {exc!r}", flush=True)
    ev.last_axis_walltime_s = time.time() - t0
    s = ev.aggregate_axis_metrics(Path(out_dir) / "axis_baseline", None)
    return s


def pc_capmean(summary):
    """per-class cap-mean AP dict (mean over dist-thresholds)."""
    pc = summary.get("per_class_AP") or {}
    return {c: capmean(pc.get(c, {})) for c in SCORED}


def override_stats(summary):
    """(n_overrides, precision, fp_to_tp, tp_to_fp) from the GT-matched audit."""
    oa = summary.get("override_audit") or {}
    by = oa.get("by_target") or {}
    f = sum(v.get("fp_to_tp", 0) for v in by.values())
    t = sum(v.get("tp_to_fp", 0) for v in by.values())
    nw = sum(v.get("neutral_wrong", 0) for v in by.values())
    matched = f + t + nw
    prec = (f / matched) if matched else float("nan")
    return oa.get("n_overrides_total", 0), prec, f, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="hybrid source cache dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--no-smoke", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=a.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = _list_val_scenes(loader)
    if a.scene_limit and a.scene_limit > 0:
        scenes = scenes[: a.scene_limit]
    print(f"  val scenes used: {len(scenes)}", flush=True)

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None, cp_cache_dir=a.src,
        proposal_source="hybrid")

    # ---- internal smoke (3 scenes, native + A1) before the full grid --------
    if not a.no_smoke and (not a.scene_limit or a.scene_limit > 3):
        print("\n##### SMOKE (3 scenes: native + A1) #####", flush=True)
        sm = scenes[:3]
        s_nat = run_one(ev, sm, out / "_smoke_native", frozenset(), 0.5, 0.4)
        s_a1 = run_one(ev, sm, out / "_smoke_A1",
                       frozenset({"bicycle", "motorcycle"}), 0.5, 0.4)
        n_ov, prec, f, t = override_stats(s_a1)
        print(f"  smoke native mAP={s_nat['mAP']}  A1 mAP={s_a1['mAP']} "
              f"overrides={n_ov} fp->tp={f} tp->fp={t}", flush=True)
        assert s_nat["mAP"] is not None and s_a1["mAP"] is not None, "smoke eval failed"
        assert n_ov > 0, "A1 produced no overrides on smoke -> overlay not firing"
        print("  smoke OK", flush=True)

    results = {}   # name -> {mAP, per_class, override}

    # ---- Study 1: allowlist expansion --------------------------------------
    print("\n##### STUDY 1: allowlist expansion (tau_iou=0.5 tau_score=0.4) #####",
          flush=True)
    for name, allow in ALLOWLISTS:
        s = run_one(ev, scenes, out / name, allow, 0.5, 0.4)
        n_ov, prec, f, t = override_stats(s)
        results[name] = {"mAP": s["mAP"], "NDS": s["NDS"],
                         "per_class": pc_capmean(s),
                         "n_overrides": n_ov, "ov_precision": prec,
                         "fp_to_tp": f, "tp_to_fp": t,
                         "override_audit": s.get("override_audit"),
                         "wall_s": s["axis_walltime_s"]}
        print(f"  {name:16s} mAP={s['mAP']:.4f}  overrides={n_ov} "
              f"prec={prec:.3f} fp->tp={f} tp->fp={t} "
              f"({s['axis_walltime_s']:.0f}s)", flush=True)

    # ---- pick best allowlist (max mAP over A1..A4; tie -> smaller set) ------
    nat = results["A0_native"]["mAP"]
    cand = [(n, l) for n, l in ALLOWLISTS if n != "A0_native"]
    best_name, best_allow = max(
        cand, key=lambda nl: (round(results[nl[0]]["mAP"], 4), -len(nl[1])))
    print(f"\n  native mAP={nat:.4f}", flush=True)
    print(f"  BEST allowlist for Study 2 = {best_name} "
          f"(mAP={results[best_name]['mAP']:.4f}, "
          f"d_native={results[best_name]['mAP']-nat:+.4f})", flush=True)

    # ---- Study 2: threshold sweep on best allowlist ------------------------
    print(f"\n##### STUDY 2: threshold sweep on {best_name} #####", flush=True)
    grid = {}
    for ti in TAU_IOU_GRID:
        for ts in TAU_SCORE_GRID:
            tag = f"S2_{best_name}_iou{ti}_sc{ts}"
            s = run_one(ev, scenes, out / tag, best_allow, ti, ts)
            n_ov, prec, f, t = override_stats(s)
            pc = pc_capmean(s)
            grid[(ti, ts)] = {"mAP": s["mAP"], "bicycle": pc["bicycle"],
                              "motorcycle": pc["motorcycle"],
                              "traffic_cone": pc["traffic_cone"],
                              "n_overrides": n_ov, "ov_precision": prec,
                              "per_class": pc}
            print(f"  iou={ti} sc={ts}: mAP={s['mAP']:.4f} "
                  f"bike={pc['bicycle']:.4f} moto={pc['motorcycle']:.4f} "
                  f"cone={pc['traffic_cone']:.4f} ovr={n_ov} prec={prec:.3f}",
                  flush=True)

    # ---- deliverable tables -------------------------------------------------
    print("\n\n================= STUDY 1: per-class AP vs native =================",
          flush=True)
    hdr = f"{'class':22s}" + "".join(f"{n.split('_')[0]:>10s}" for n, _ in ALLOWLISTS)
    print(hdr, flush=True)
    natpc = results["A0_native"]["per_class"]
    for c in SCORED:
        row = f"{c:22s}"
        for n, _ in ALLOWLISTS:
            row += f"{results[n]['per_class'][c]:10.4f}"
        print(row, flush=True)
    print(f"{'mAP':22s}" + "".join(f"{results[n]['mAP']:10.4f}" for n, _ in ALLOWLISTS),
          flush=True)
    print(f"{'dAP-native(mAP)':22s}" +
          "".join(f"{results[n]['mAP']-nat:+10.4f}" for n, _ in ALLOWLISTS), flush=True)

    print("\n----- Study 1 delta-vs-native (per class) + collateral flags -----",
          flush=True)
    print(f"{'class':22s}" + "".join(f"{n.split('_')[0]:>10s}" for n, _ in ALLOWLISTS[1:]),
          flush=True)
    for c in SCORED:
        row = f"{c:22s}"
        for n, _ in ALLOWLISTS[1:]:
            row += f"{results[n]['per_class'][c]-natpc[c]:+10.4f}"
        print(row, flush=True)

    corner = "iou \\ score".ljust(18)
    print("\n================= STUDY 2: mAP grid (allowlist=" + best_name +
          ") =================", flush=True)
    print(corner + "".join(f"{ts:>10.1f}" for ts in TAU_SCORE_GRID), flush=True)
    for ti in TAU_IOU_GRID:
        print(f"{ti:<18.1f}" + "".join(f"{grid[(ti,ts)]['mAP']:10.4f}"
                                       for ts in TAU_SCORE_GRID), flush=True)
    print("\n----- Study 2: bicycle AP grid -----", flush=True)
    print(corner + "".join(f"{ts:>10.1f}" for ts in TAU_SCORE_GRID), flush=True)
    for ti in TAU_IOU_GRID:
        print(f"{ti:<18.1f}" + "".join(f"{grid[(ti,ts)]['bicycle']:10.4f}"
                                       for ts in TAU_SCORE_GRID), flush=True)
    print("\n----- Study 2: n_overrides / precision grid -----", flush=True)
    for ti in TAU_IOU_GRID:
        for ts in TAU_SCORE_GRID:
            g = grid[(ti, ts)]
            print(f"  iou={ti} sc={ts}: overrides={g['n_overrides']:5d} "
                  f"precision={g['ov_precision']:.3f}", flush=True)

    # ---- persist machine-readable summary ----------------------------------
    payload = {
        "scene_count": len(scenes),
        "native_mAP": nat,
        "best_allowlist": best_name,
        "study1": results,
        "study2": {f"iou{ti}_sc{ts}": v for (ti, ts), v in grid.items()},
    }
    (out / "grid_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out/'grid_summary.json'}", flush=True)
    print(f"=== DONE {time.strftime('%F %T')} ===", flush=True)


if __name__ == "__main__":
    main()
