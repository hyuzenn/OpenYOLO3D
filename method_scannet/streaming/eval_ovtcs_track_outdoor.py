"""OV-TCS metric validation on nuScenes (outdoor) — per-track.

Mirror of eval_ovtcs_instance_scannet but for the outdoor associator tracks:
does a track's per-frame class stability (OV-TCS_C) predict downstream label
correctness beyond track length / fragmentation? EVALUATION ONLY.

Rides the #2 label-fusion evaluator (NativeTemporalNuScenesEvaluator) with
collect_track_metrics=True on the existing cached proposals (cache-replay, no
new GPU inference). OV-TCS is a pure associator property (native labels), so a
single native (no-overlay) pass over the 150 val scenes is enough; the
per-track table is read off build_track_records() and run through the same
partial-correlation battery as indoors.

Run (PBS container; proposals cached → CPU-only replay):
  python -u -m method_scannet.streaming.eval_ovtcs_track_outdoor \
    --cp-cache-dir results/outdoor_detguided_cpcache_thr000_full150 \
    --proposal-source detguided --association-class-agnostic \
    --output results/2026-06-25_outdoor_ovtcs_track_v01
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from method_scannet.streaming.eval_ovtcs_instance_scannet import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-cache-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--proposal-source", default="detguided",
                    choices=["gamma", "detguided", "hybrid"])
    ap.add_argument("--nuscenes-config", default="configs/nuscenes_trainval.yaml")
    ap.add_argument("--association-class-agnostic", action="store_true")
    ap.add_argument("--association-frame", default="ego", choices=["ego", "global"])
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--frag-inject-p", type=float, default=0.0,
                    help="Controlled fragmentation-injection rate in [0,1); per-track "
                         "fragmentation decomposition for §2c.")
    args = ap.parse_args()

    from dataloaders.nuscenes_loader import NuScenesLoader
    from method_scannet.streaming.nuscenes_native_evaluator import (
        NativeTemporalNuScenesEvaluator, _list_val_scenes)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading nuScenes ...", flush=True)
    loader = NuScenesLoader(config_path=args.nuscenes_config)
    loader.multi_sweep = False
    loader.num_sweeps = 1
    scenes = _list_val_scenes(loader)
    if args.scene_limit and args.scene_limit > 0:
        scenes = scenes[: args.scene_limit]
    print(f"  val scenes={len(scenes)} source={args.proposal_source}", flush=True)

    ev = NativeTemporalNuScenesEvaluator(
        loader=loader, cp_proposals=None, cp_cache_dir=args.cp_cache_dir,
        proposal_source=args.proposal_source,
        class_agnostic_association=args.association_class_agnostic,
        association_frame=args.association_frame,
        frag_inject_p=args.frag_inject_p,
        collect_track_metrics=True)
    ev.install_axis("baseline")
    ev.begin_axis()
    t0 = time.time()
    for i, sc in enumerate(scenes):
        try:
            ev.run_scene(sc, scene_idx=i)
        except Exception as exc:
            print(f"    SCENE {sc[:8]} FAILED: {exc!r}", flush=True)
        if (i + 1) % 50 == 0 or (i + 1) == len(scenes):
            print(f"    {i+1}/{len(scenes)} tracks={len(ev._track_seq)} "
                  f"{time.time()-t0:.0f}s", flush=True)

    rows = ev.build_track_records()
    (out / "tracks.json").write_text(json.dumps(rows, indent=2))

    # analysis on matched tracks with track_len>=2 (ovtcs_C defined)
    m = [r for r in rows if r["gt_matched"] and r["ovtcs_C"] is not None
         and r["correct"] is not None]
    have_frag = all(r["gt_frag"] is not None for r in m) and len(m) > 0
    res = analyze(
        [r["correct"] for r in m],
        [r["ovtcs_C"] for r in m],
        [r["track_len"] for r in m],
        frag=[r["gt_frag"] for r in m] if have_frag else None,
    )
    (out / "analysis_track.json").write_text(json.dumps(res, indent=2))

    if res.get("degenerate") or "partial_correlation" not in res:
        verdict = f"n={res.get('n')} degenerate/insufficient"
    else:
        pc = res["partial_correlation"]["ovtcsC_y_given_len_pearson"]
        reg = res["regression"]
        zo = res["zero_order"]
        ok = pc["p"] < 0.05 and reg["F_p"] < 0.05 and reg["delta_R2"] > 0
        frag = res.get("fragmentation", {})
        verdict = (
            f"n={res['n']}\n"
            f"- zero-order corr(OV-TCS_C, correct): r={zo['ovtcsC_y_pearson']:.3f} "
            f"(len vs correct r={zo['len_y_pearson']:.3f})\n"
            f"- partial corr(OV-TCS_C, correct | len): r={pc['r']:.3f} p={pc['p']:.2e}\n"
            f"- nested F (add OV-TCS to len): ΔR²={reg['delta_R2']:.3f} p={reg['F_p']:.2e}\n"
            f"- OV-TCS vs fragmentation: r={frag.get('ovtcsC_vs_frag_pearson')}\n"
            f"- GATE {'PASS' if ok else 'FAIL'}")
    notes = ("# OV-TCS metric validation — nuScenes outdoor (per-track)\n\n"
             f"source={args.proposal_source}, {len(scenes)} val scenes, "
             "native associator tracks. EVALUATION ONLY.\n\n" + verdict + "\n")
    (out / "notes.md").write_text(notes)
    print("\n" + notes)
    print(f"wrote {out}/tracks.json analysis_track.json notes.md")


if __name__ == "__main__":
    main()
