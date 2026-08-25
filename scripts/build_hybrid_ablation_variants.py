#!/usr/bin/env python
"""Reconstruct the Hybrid-Proposal bottleneck-attribution variants A/B/C/D'
from the single existing hybrid cache, with NO rebuild and NO GPU.

The hybrid cache stores, per CenterPoint box:
    cls_name   YOLO-transferred label  ('__background__' if unmatched)
    score      YOLO 2D detection score
    cp_cls_name native CenterPoint class (kept for EVERY box, incl. unmatched)
    score_cp   native CenterPoint score
    match_iou  >0  iff a YOLO ROI matched this box

On --axes baseline the native evaluator reads only `cls_name` (→ index;
'__background__' / non-NUSC10 → dropped) and `score`.  So every variant is just
a rewrite of those two fields:

  A  (baseline / native)  cls=cp_cls_name           score=score_cp   keep all
  B  (label-only)         cls=yolo if matched else cp_cls_name
                                                     score=score_cp   keep all
  C  (match-only/delete)  cls=cp_cls_name if matched else __background__
                                                     score=score_cp   drop unmatched
  D' (hybrid, CP score)   cls=yolo if matched else __background__
                                                     score=score_cp   drop unmatched

D (shipped hybrid: cls=yolo/bg, score=yolo) is the original cache, evaluated as-is.

Files are written as <token>.hybrid.pkl into one dir per variant so the
evaluator can read them with --proposal-source hybrid --cp-cache-dir <dir>.
"""
import argparse, glob, os, os.path as osp, pickle, sys, time

BG = "__background__"
# Variant G: class-aware fusion. F's global gate bled AP on large-pop classes
# (truck/trailer/car) where YOLO is wrong but still clears tau. Per-class data
# shows the ONLY label-fixable error is the bicycle<->motorcycle confusion, so
# G only trusts YOLO when its target class is in this allowlist.
G_CLASSES = frozenset({"bicycle", "motorcycle"})


def fuse_label_classaware(b, matched, tau_iou, tau_score, allow=G_CLASSES):
    """Like fuse_label, but additionally requires the YOLO target class to be in
    `allow`. Isolates the VRU confusion-pair correction and prevents large-pop
    classes from being corrupted by a confident-but-wrong YOLO label."""
    lbl = fuse_label(b, matched, tau_iou, tau_score)
    return lbl if lbl in allow else b["cp_cls_name"]


def fuse_label(b, matched, tau_iou, tau_score):
    """Arm C (confidence-aware fusion): override the CenterPoint class with the
    YOLO label ONLY when the 2D evidence is both well-localized and confident
    (match_iou >= tau_iou AND yolo_score >= tau_score). Otherwise keep CP.
    Score stays score_cp and no box is dropped (constraints). This is the gated
    version of Arm B, which relabeled blindly and lost -0.080 mAP."""
    if (matched and float(b.get("match_iou", 0.0)) >= tau_iou
            and float(b["score"]) >= tau_score and b["cls_name"] != BG):
        return b["cls_name"]          # trust YOLO
    return b["cp_cls_name"]           # keep CenterPoint


def build(src_dir: str, out_root: str, tau_iou: float, tau_score: float) -> None:
    fs = sorted(glob.glob(osp.join(src_dir, "*.hybrid.pkl")))
    if not fs:
        sys.exit(f"no *.hybrid.pkl in {src_dir!r}")
    variants = ["A", "B", "C", "Dp", "F", "G"]
    dirs = {v: osp.join(out_root, f"variant_{v}") for v in variants}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    n_box = 0
    n_matched = 0
    t0 = time.time()
    for i, fp in enumerate(fs):
        token = osp.basename(fp)[: -len(".hybrid.pkl")]
        boxes = pickle.load(open(fp, "rb"))
        outs = {v: [] for v in variants}
        for b in boxes:
            n_box += 1
            matched = float(b.get("match_iou", 0.0)) > 0.0
            n_matched += int(matched)
            cp_cls = b["cp_cls_name"]
            yolo_cls = b["cls_name"]            # '__background__' if unmatched
            s_cp = float(b["score_cp"])
            base = {k: b[k] for k in ("cls_idx", "bbox_lidar", "centroid_ego")}

            # A: native CenterPoint label + score, keep all
            outs["A"].append({**base, "cls_name": cp_cls, "score": s_cp})
            # B: relabel matched only, keep unmatched as CP, CP score, keep all
            outs["B"].append({**base,
                              "cls_name": yolo_cls if matched else cp_cls,
                              "score": s_cp})
            # C: CP label, CP score, drop unmatched (-> background)
            outs["C"].append({**base,
                              "cls_name": cp_cls if matched else BG,
                              "score": s_cp})
            # D': YOLO label, CP score, drop unmatched (yolo_cls already BG if unmatched)
            outs["Dp"].append({**base, "cls_name": yolo_cls, "score": s_cp})
            # F (Arm C): confidence-aware fusion, CP score, keep all
            outs["F"].append({**base,
                              "cls_name": fuse_label(b, matched, tau_iou, tau_score),
                              "score": s_cp})
            # G: class-aware fusion (override only into {bicycle,motorcycle})
            outs["G"].append({**base,
                              "cls_name": fuse_label_classaware(
                                  b, matched, tau_iou, tau_score),
                              "score": s_cp})

        for v in variants:
            with open(osp.join(dirs[v], f"{token}.hybrid.pkl"), "wb") as f:
                pickle.dump(outs[v], f)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(fs)} tokens  ({time.time()-t0:.0f}s)", flush=True)

    print(f"done: {len(fs)} tokens, {n_box} boxes, matched_rate="
          f"{n_matched/max(1,n_box):.4f}, {time.time()-t0:.0f}s", flush=True)
    for v in variants:
        print(f"  variant {v}: {dirs[v]}")


def demo():
    """Self-check: fusion overrides only on a tight+confident match, never drops."""
    hi = {"cls_name": "bicycle", "cp_cls_name": "motorcycle", "score": 0.6,
          "score_cp": 0.5, "match_iou": 0.7}
    loose = {**hi, "match_iou": 0.2}          # localization too loose
    weak = {**hi, "score": 0.1}               # yolo too unconfident
    unmatched = {"cls_name": BG, "cp_cls_name": "car", "score": 0.0,
                 "score_cp": 0.4, "match_iou": 0.0}
    assert fuse_label(hi, True, 0.5, 0.4) == "bicycle"        # override
    assert fuse_label(loose, True, 0.5, 0.4) == "motorcycle"  # keep CP
    assert fuse_label(weak, True, 0.5, 0.4) == "motorcycle"   # keep CP
    assert fuse_label(unmatched, False, 0.5, 0.4) == "car"    # keep CP, not dropped
    # class-aware: bicycle is in allowlist -> override; truck target is not.
    assert fuse_label_classaware(hi, True, 0.5, 0.4) == "bicycle"
    truck = {**hi, "cls_name": "truck", "cp_cls_name": "car"}
    assert fuse_label_classaware(truck, True, 0.5, 0.4) == "car"   # blocked -> keep CP
    print("demo ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="hybrid cache dir (*.hybrid.pkl)")
    ap.add_argument("--out", help="root for variant_{A,B,C,Dp,F} dirs")
    ap.add_argument("--tau-iou", type=float, default=0.5,
                    help="Arm C: min match_iou to trust the YOLO label")
    ap.add_argument("--tau-score", type=float, default=0.4,
                    help="Arm C: min YOLO 2D score to trust the YOLO label")
    ap.add_argument("--demo", action="store_true", help="run self-check and exit")
    a = ap.parse_args()
    if a.demo:
        demo()
    else:
        build(a.src, a.out, a.tau_iou, a.tau_score)
