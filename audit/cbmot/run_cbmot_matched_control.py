"""Build the CBMOT-style matched-control arms and score them.

Read-only w.r.t. everything outside audit/cbmot/.  Every arm is a mask over
one frozen baseline tracks.json (see cbmot_policy.py); box contents are copied
by reference, never recomputed.

  python run_cbmot_matched_control.py --frame ego --arm retro --N 3
  python run_cbmot_matched_control.py --verify        # invariants only, no eval
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/home/rintern16/OpenYOLO3D")

import cbmot_policy as P
from method_scannet.streaming.metrics import label_switch_count

ROOT = Path("/home/rintern16/OpenYOLO3D")
# CBMOT_OUT / CBMOT_GRID / CBMOT_RUN_N3 let the Table-1 regeneration point this
# control at the 10-sweep cells without disturbing the frozen single-sweep audit:
# unset, every path below is exactly what the 2026-08-26 CBMOT run used.
OUT = Path(os.environ.get("CBMOT_OUT", Path(__file__).resolve().parent))
GRID = Path(os.environ.get(
    "CBMOT_GRID", ROOT / "results/2026-07-18_e1_grid_v01/cells"))

# N -> the frozen run whose retro arm defines that N's emission budget.
RUNS = {
    2: ROOT / "results/2026-08-04_nsweep_N2_v01",
    3: Path(os.environ.get(
        "CBMOT_RUN_N3", ROOT / "results/2026-07-30_e2c_retro_thrmatch_v01")),
    4: ROOT / "results/2026-08-04_nsweep_N4_v01",
    5: ROOT / "results/2026-08-04_nsweep_N5_v01",
}


def baseline_path(frame: str) -> Path:
    return GRID / f"gamma_{frame}" / "axis_baseline" / "tracks.json"


def frozen_cell(frame: str, N: int, kind: str) -> Path:
    """kind in {retro, ctrl}; the paper's C1 and C0 for this (N, frame)."""
    return RUNS[N] / "cells" / f"{kind}_{frame}"


def frozen_metrics(cell: Path) -> dict:
    m = sorted(cell.glob("axis_*/metrics.json"))
    return json.loads(m[0].read_text()) if m else {}


def budget(frame: str, N: int, causal: bool) -> tuple[int, str]:
    """The emission budget this arm must match, and where it came from."""
    if causal:
        # causal C1 = the streaming prefix-deleting gate, N=3 only (gamma_*_p1)
        c = GRID / f"gamma_{frame}_p1"
        return frozen_metrics(c)["n_pred_boxes_total"], str(c)
    c = frozen_cell(frame, N, "retro")
    return frozen_metrics(c)["n_pred_boxes_total"], str(c)


def build(frame: str, arm: str, N: int, update: str, noise: float, tag: str,
          d=None):
    base_p = baseline_path(frame)
    if d is None:
        d = json.loads(base_p.read_text())
    pred, ssi = d["pred"], d["sample_scene_idx"]
    n_base = sum(len(v) for v in pred.values())

    K, K_src = budget(frame, N, causal=(arm == "causal"))
    if arm == "frozenmask":
        mask, info = frozen_emission_mask(pred, frame, N)
        scores = None
    else:
        scores, peak = P.refined_scores(pred, ssi, update, noise=noise)
        if arm == "causal":
            mask, info = P.causal_mask(pred, scores, K)
        else:
            mask, info = P.retro_mask(pred, scores, peak, K)

    emitted = {tok: [b for b, keep in zip(boxes, mask[tok]) if keep]
               for tok, boxes in pred.items()}
    n_emit = sum(len(v) for v in emitted.values())

    # ---- invariants (fail loud; do not interpret metrics if these break) ----
    assert set(emitted) == set(pred), "sample token set changed"
    assert len(emitted) == 6019, f"expected 6019 samples, got {len(emitted)}"
    assert n_emit == info["emitted"] <= K, (n_emit, info["emitted"], K)
    for tok in pred:                       # identity, not equality: no copies
        kept = [b for b, k in zip(pred[tok], mask[tok]) if k]
        assert all(a is b for a, b in zip(emitted[tok], kept))

    if arm == "frozenmask":
        fz = json.loads(Path(ROOT / info["frozen_source"]).read_text())["pred"]
        full = lambda b: (b["tracking_id"], b["detection_name"],
                          round(b["translation"][0], 6), round(b["translation"][1], 6),
                          round(b["detection_score"], 6))
        identical = all({full(b) for b in emitted[t]} == {full(b) for b in fz.get(t, [])}
                        for t in emitted)
        info["identical_to_frozen"] = identical
        if identical:
            # M21 relabelled nothing here, so C1 already IS the emission-only arm
            # and its frozen metrics apply unchanged. Nothing to re-evaluate.
            (OUT / "out").mkdir(exist_ok=True)
            (OUT / "out" / f"frozenmask_identity_{frame}_N{N}.json").write_text(
                json.dumps({"frame": frame, "N": N, "identical_to_frozen": True,
                            "frozen_source": info["frozen_source"],
                            "n_boxes": n_emit}, indent=2))
            print(f"[frozenmask] {frame} N={N}: identical to {info['frozen_source']}"
                  f" ({n_emit} boxes) -- no re-evaluation needed", flush=True)
            return None

    cell = OUT / "cells" / tag
    axis = cell / "axis_cbmot"
    axis.mkdir(parents=True, exist_ok=True)
    (axis / "tracks.json").write_text(json.dumps(
        {"sample_scene_idx": ssi, "pred": emitted, "gt": d["gt"]}))

    flat = np.fromiter((k for tok in pred for k in mask[tok]), dtype=bool)  # noqa
    np.savez_compressed(axis / "emission_mask.npz", mask=flat)

    lsc = emitted_lsc(emitted)
    prov = {
        "tag": tag, "frame": frame, "arm": arm, "N": N,
        "update_fn": update, "noise": noise, "max_age": P.MAX_AGE,
        "baseline_tracks": str(base_p), "n_baseline_boxes": n_base,
        "budget_target": K, "budget_source": K_src,
        "budget_achieved": n_emit,
        "budget_abs_diff": n_emit - K,
        "budget_rel_diff": (n_emit - K) / K,
        "cbmot_threshold": info["threshold"],
        "n_tracks_kept": info.get("n_tracks_kept"),
        "n_samples": len(emitted),
        "n_emitted_frames": sum(1 for v in emitted.values() if v),
        "boxes_per_frame": n_emit / len(emitted),
        "label_switch_count_total": int(lsc),
        "n_tracks_emitted": len({b["tracking_id"] for bs in emitted.values()
                                 for b in bs}),
        "class_counts": _counts(emitted),
    }
    (axis / "cbmot_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({k: v for k, v in prov.items() if k != "class_counts"},
                     indent=1), flush=True)
    return cell


def frozen_emission_mask(pred: dict, frame: str, N: int):
    """C1's emission decision, lifted off the frozen retro arm, labels untouched.

    Needed because phase1 is M11 (gate) -> M21 (relabel) -> M31 (merge), not a
    gate alone. In the global frame M21 rewrites the class of 226,589 emitted
    boxes (fire_audit: n_relabeled_by_m21 = 293,440), so the frozen C1 arm
    differs from the baseline stream in *labels* as well as in which boxes it
    emits -- confirmed by matching on geometry+score, which is exact (0
    mismatches), against class, which is not. Comparing that arm to a pure
    emission policy would confound gating with relabelling.

    This arm keeps C1's box selection and drops its relabelling, so the
    three-way comparison is over emission policy alone. In the ego frame M21
    relabels nothing, so this must reproduce the frozen arm exactly -- which is
    how it is validated.
    """
    tj = sorted(frozen_cell(frame, N, "retro").glob("axis_*/tracks.json"))[0]
    fz = json.loads(tj.read_text())["pred"]
    geo = lambda b: (round(b["translation"][0], 6), round(b["translation"][1], 6),
                     round(b["detection_score"], 6))
    n_emit, mask = 0, {}
    for tok, boxes in pred.items():
        want = {geo(b) for b in fz.get(tok, [])}
        m = [geo(b) in want for b in boxes]
        assert sum(m) == len(want), (tok, sum(m), len(want))   # no geometry ties
        mask[tok] = m
        n_emit += sum(m)
    assert n_emit == sum(len(v) for v in fz.values()), n_emit
    return mask, {"threshold": float("nan"), "emitted": n_emit,
                  "n_tracks_kept": None, "frozen_source": str(tj.relative_to(ROOT))}


def _counts(pred: dict) -> dict:
    c: dict[str, int] = {}
    for boxes in pred.values():
        for b in boxes:
            c[b["detection_name"]] = c.get(b["detection_name"], 0) + 1
    return dict(sorted(c.items()))


def input_invariance(frames=("ego", "global")) -> dict:
    """The frozen arms must all be masks over the same baseline box stream."""
    rep = {}
    for frame in frames:
        base = json.loads(baseline_path(frame).read_text())["pred"]
        key = lambda b: (b["tracking_id"], round(b["translation"][0], 6),
                         round(b["translation"][1], 6), b["detection_name"],
                         round(b["detection_score"], 6))
        index = {tok: {key(b) for b in bs} for tok, bs in base.items()}
        for N in sorted(RUNS):
            for kind in ("retro", "ctrl"):
                cell = frozen_cell(frame, N, kind)
                tj = sorted(cell.glob("axis_*/tracks.json"))
                if not tj:
                    continue
                p = json.loads(tj[0].read_text())["pred"]
                extra = sum(len({key(b) for b in bs} - index.get(tok, set()))
                            for tok, bs in p.items())
                rep[f"{kind}_{frame}_N{N}"] = {
                    "n_boxes": sum(len(v) for v in p.values()),
                    "boxes_not_in_baseline": extra,
                    "same_tokens": set(p) == set(base),
                }
                print(f"[inv] {kind}_{frame}_N{N}: {rep[f'{kind}_{frame}_N{N}']}",
                      flush=True)
    (OUT / "out").mkdir(exist_ok=True)
    (OUT / "out" / "input_invariance.json").write_text(json.dumps(rep, indent=2))
    return rep


def provenance() -> dict:
    """Frozen-input record: §2 of the report. Read-only."""
    import hashlib
    rep = {}
    for frame in ("ego", "global"):
        bp = baseline_path(frame)
        h = hashlib.md5()
        with open(bp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        d = json.loads(bp.read_text())
        pred, ssi = d["pred"], d["sample_scene_idx"]
        sc = np.fromiter((b["detection_score"] for bs in pred.values() for b in bs),
                         dtype=np.float64)
        rep[frame] = {
            "path": str(bp), "md5": h.hexdigest(),
            "bytes": bp.stat().st_size,
            "split": "nuScenes v1.0-trainval val",
            "n_samples": len(pred), "n_scenes": len(set(ssi.values())),
            "n_boxes": int(sc.size),
            "n_tracks": len({b["tracking_id"] for bs in pred.values() for b in bs}),
            "score": {"min": float(sc.min()), "max": float(sc.max()),
                      "mean": float(sc.mean()), "median": float(np.median(sc)),
                      "p05": float(np.percentile(sc, 5)),
                      "p95": float(np.percentile(sc, 95))},
            "class_counts": _counts(pred),
            "coordinate_frame": "global (nuScenes world), association_frame="
                                + frame,
            "n_gt_boxes": sum(len(v) for v in d["gt"].values()),
        }
        print(f"[prov] {frame}: {json.dumps({k: v for k, v in rep[frame].items() if k != 'class_counts'})}",
              flush=True)
    (OUT / "out").mkdir(exist_ok=True)
    (OUT / "out" / "frozen_input_provenance.json").write_text(json.dumps(rep, indent=2))
    return rep


def run_e1(cell: Path, no_amota: bool, keep: bool) -> None:
    import subprocess
    cmd = [sys.executable, str(ROOT / "scripts/e1_gt_metrics.py"),
           "--cell", str(cell)] + (["--no-amota"] if no_amota else [])
    print("== " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    if not keep:
        (cell / "axis_cbmot" / "tracks.json").unlink()   # regenerable from mask


def run_all(frame: str, no_amota: bool, keep: bool, slice_spec: str = "") -> None:
    """Every arm of one frame from a single baseline load."""
    import time
    t0 = time.time()
    d = json.loads(baseline_path(frame).read_text())
    print(f"== baseline loaded in {time.time() - t0:.0f}s", flush=True)
    plan = [("frozenmask", N, "none", 0.0) for N in (2, 3, 4, 5)]
    plan += [("retro", N, u, P.NOISE)
             for N in (2, 3, 4, 5) for u in ("parallel_addition", "raw")]
    plan += [("causal", 3, u, P.NOISE) for u in ("parallel_addition", "raw")]
    if slice_spec:                       # k/n -> this job takes plan[k::n]
        k, n = (int(x) for x in slice_spec.split("/"))
        plan = plan[k::n]
    print(f"== plan ({len(plan)}): {plan}", flush=True)
    for arm, N, upd, noise in plan:
        tag = f"cbmot_{arm}_{frame}_N{N}_{upd}_noise{noise}"
        if (OUT / "cells" / tag / "axis_cbmot" / "e1_metrics.json").exists():
            print(f"== {tag} done, skip", flush=True)
            continue
        t = time.time()
        cell = build(frame, arm, N, upd, noise, tag, d=d)
        if cell is None:
            continue
        run_e1(cell, no_amota, keep)
        print(f"== {tag} finished in {time.time() - t:.0f}s", flush=True)


def emitted_lsc(pred: dict) -> int:
    """label_switch_count over the EMITTED stream, repo definition, unmodified.

    The frozen cells report label_switch_count_total from the evaluator's
    labeler snapshot, which is 0 on every outdoor arm. That is not a property
    of the emitted boxes: in the global frame 160,395 of 266,623 baseline
    tracks carry more than one class. Recomputing it here from what each arm
    actually emitted is the only way the three policies are comparable on this
    metric.
    """
    return int(label_switch_count(
        [{int(b["tracking_id"]): b["detection_name"] for b in bs}
         for bs in pred.values()]))


def frozen_label_switches() -> dict:
    cells = []
    for N in sorted(RUNS):
        for frame in ("ego", "global"):
            for kind in ("retro", "ctrl"):
                cells.append((f"{kind}_{frame}_N{N}", frozen_cell(frame, N, kind)))
    for frame in ("ego", "global"):
        cells.append((f"p1_{frame}_N3", GRID / f"gamma_{frame}_p1"))
        cells.append((f"unfiltered_{frame}", GRID / f"gamma_{frame}"))
        cells.append((f"ctrl_{frame}_N3_causal",
                      ROOT / "results/2026-07-28_e2_thrmatch_v01/cells" / f"ctrl_{frame}"))
    rep = {}
    for name, cell in cells:
        tj = sorted(cell.glob("axis_*/tracks.json"))
        if not tj:
            print(f"[lsc] {name}: no tracks.json", flush=True)
            continue
        pred = json.loads(tj[0].read_text())["pred"]
        n = sum(len(v) for v in pred.values())
        nf = sum(1 for v in pred.values() if v)
        rep[name] = {"label_switches": emitted_lsc(pred), "n_boxes": n,
                     "n_emitted_frames": nf,
                     "n_tracks": len({b["tracking_id"] for bs in pred.values()
                                      for b in bs}),
                     "source": str(tj[0].relative_to(ROOT))}
        print(f"[lsc] {name}: {rep[name]}", flush=True)
    (OUT / "out").mkdir(exist_ok=True)
    (OUT / "out" / "frozen_label_switches.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", choices=("ego", "global"))
    ap.add_argument("--arm", choices=("retro", "causal", "frozenmask"),
                    default="retro")
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--update", default="parallel_addition",
                    choices=sorted(P.UPDATE_FNS) + ["none"])
    ap.add_argument("--noise", type=float, default=P.NOISE)
    ap.add_argument("--tag")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--provenance", action="store_true")
    ap.add_argument("--label-switches", action="store_true")
    ap.add_argument("--plan-slice", default="", metavar="K/N")
    ap.add_argument("--all", action="store_true",
                    help="every arm of --frame from one baseline load")
    ap.add_argument("--no-amota", action="store_true")
    ap.add_argument("--keep-tracks", action="store_true")
    a = ap.parse_args()

    if a.label_switches:
        frozen_label_switches()
        return
    if a.provenance:
        provenance()
        return
    if a.verify:
        input_invariance()
        return
    if a.all:
        run_all(a.frame, a.no_amota, a.keep_tracks, a.plan_slice)
        return
    tag = a.tag or f"cbmot_{a.arm}_{a.frame}_N{a.N}_{a.update}_noise{a.noise}"
    cell = build(a.frame, a.arm, a.N, a.update, a.noise, tag)

    run_e1(cell, a.no_amota, a.keep_tracks)


if __name__ == "__main__":
    main()
