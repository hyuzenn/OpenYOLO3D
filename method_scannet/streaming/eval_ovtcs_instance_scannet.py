"""OV-TCS metric validation on ScanNet200 — per-instance + per-scene.

EVALUATION ONLY. No method is proposed and the M22/OV-TCS-aware EMA line stays
frozen-negative — this validates OV-TCS_C purely as a *temporal-quality metric*:
does a track's per-frame class stability predict its downstream detection
quality, beyond how long it was tracked?

Why per-instance (not the nuScenes 10-variant surrogate): ScanNet instances are
fixed by the Mask3D cache and the streaming step only fuses a per-frame *label*
onto them — there is no track-survival / max_age / frag-inject knob, and
track_length is visibility-driven (invariant to any temporal-quality dial). So
the surrogate's variant design is structurally degenerate here. Instead every
3D instance is one data point; track_length varies naturally per instance,
which is exactly what the "incremental over track length" test needs.

Single baseline pass over the Mask3D cache (cache-replay, no new fusion code):
for each scene we read the evaluator's already-produced `pred_history`
(per-frame {iid: cumulative label}), final baseline preds, and GT matching.

Per instance (matched to a GT at IoU>=0.5, track_length>=2):
  OV-TCS_C  = (1 - 1/L)(1 - CSR)   over the cumulative-label sequence
              (same formula as method_22_feature_fusion._update_raw_ovtcs;
               ponytail: uses the cumulative snapshot already in pred_history —
               the same sequence label_switch_count consumes — not a fresh raw
               argmax stream, so it needs zero extra instrumentation)
  track_len = L (frames the instance was live)
  gt_frag   = # distinct pred ids matched to the instance's GT across frames
  correct   = final predicted class == GT class (1/0)  <- downstream target

Per scene (n=312): mean OV-TCS_C / track_len / gt_frag and the scene's
label-matched mask-IoU AP (metrics.mask_iou_map).

Then on BOTH tables (same analysis as the nuScenes
results/2026-06-13_ablation_ovtcs_partial_v01/analyze.py):
  Pearson/Spearman, partial corr (OV-TCS | length), nested F-test
  (y~len vs y~len+OV-TCS), residual corr, GT-fragmentation corr.

Run (inside the PBS GPU container; cache-replay Mask3D):
  python -u -m method_scannet.streaming.eval_ovtcs_instance_scannet \
    --cache-dir results/2026-05-13_mask3d_cache \
    --output results/2026-06-25_scannet_ovtcs_instance_val312_v01

Self-check (no GPU / no data):
  python -m method_scannet.streaming.eval_ovtcs_instance_scannet --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Pure metric + stats (torch-free, unit-testable via --selftest)
# --------------------------------------------------------------------------- #
def ovtcs_c_from_seq(seq) -> tuple[float | None, int, int]:
    """Causal OV-TCS_C of a cumulative-label sequence.

    Switches count only real->real label changes (a -1/unassigned on either
    side is skipped), matching metrics.label_switch_count. Returns
    (ovtcs_C | None, L, n_switches); None when L<2 (no temporal variation).
    """
    L = len(seq)
    if L < 2:
        return None, L, 0
    sw = 0
    prev = None
    for lab in seq:
        if prev is not None and prev != -1 and lab != -1 and prev != lab:
            sw += 1
        prev = lab
    csr = sw / (L - 1)
    return (1.0 - 1.0 / L) * (1.0 - csr), L, sw


def _resid(target, ctrl):
    n = len(target)
    A = np.vstack([np.ones(n), ctrl]).T
    beta, *_ = np.linalg.lstsq(A, target, rcond=None)
    return target - A @ beta


def _partial_pearson(a, b, c):
    from scipy import stats
    ra, rb = _resid(a, c), _resid(b, c)
    r, _ = stats.pearsonr(ra, rb)
    df = len(a) - 3
    t = r * np.sqrt(df / max(1e-12, (1 - r * r)))
    p = 2 * stats.t.sf(abs(t), df)
    return float(r), float(p), float(t), int(df)


def _partial_spearman(a, b, c):
    from scipy import stats
    ra, rb, rc = stats.rankdata(a), stats.rankdata(b), stats.rankdata(c)
    ea, eb = _resid(ra, rc), _resid(rb, rc)
    r, _ = stats.pearsonr(ea, eb)
    df = len(a) - 3
    t = r * np.sqrt(df / max(1e-12, (1 - r * r)))
    p = 2 * stats.t.sf(abs(t), df)
    return float(r), float(p)


def _ols_r2(y, cols):
    n = len(y)
    P = np.vstack([np.ones(n)] + cols).T
    k = P.shape[1] - 1
    beta, *_ = np.linalg.lstsq(P, y, rcond=None)
    yhat = P @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, k


def analyze(y, X, L, frag=None) -> dict:
    """The 2026-06-13 partial-correlation battery on arbitrary (y, X=OV-TCS_C,
    L=track length) arrays. Drops rows with any NaN. Returns the result dict.
    """
    from scipy import stats

    y = np.asarray(y, float)
    X = np.asarray(X, float)
    L = np.asarray(L, float)
    keep = np.isfinite(y) & np.isfinite(X) & np.isfinite(L)
    if frag is not None:
        frag = np.asarray(frag, float)
    y, X, L = y[keep], X[keep], L[keep]
    fr = frag[keep] if frag is not None else None
    n = len(y)
    out: dict = {"n": int(n)}
    if n < 4 or np.std(X) == 0 or np.std(y) == 0 or np.std(L) == 0:
        out["degenerate"] = True
        return out

    # zero-order
    out["zero_order"] = {
        "ovtcsC_y_pearson": float(stats.pearsonr(X, y)[0]),
        "ovtcsC_y_spearman": float(stats.spearmanr(X, y)[0]),
        "len_y_pearson": float(stats.pearsonr(L, y)[0]),
        "len_y_spearman": float(stats.spearmanr(L, y)[0]),
        "ovtcsC_len_pearson": float(stats.pearsonr(X, L)[0]),
    }
    # partial correlation
    pr, pp, pt, pdf = _partial_pearson(X, y, L)
    sr, sp = _partial_spearman(X, y, L)
    lr, lp, lt, ldf = _partial_pearson(L, y, X)
    out["partial_correlation"] = {
        "ovtcsC_y_given_len_pearson": {"r": pr, "p": pp, "t": pt, "df": pdf},
        "ovtcsC_y_given_len_spearman": {"rho": sr, "p": sp},
        "len_y_given_ovtcsC_pearson": {"r": lr, "p": lp, "t": lt, "df": ldf},
    }
    # nested regression: does OV-TCS add over length?
    r2_len, _ = _ols_r2(y, [L])
    r2_both, _ = _ols_r2(y, [L, X])
    df1, df2 = 1, n - 3
    F = ((1 - r2_len) - (1 - r2_both)) / df1 / ((1 - r2_both) / df2) if r2_both < 1 else float("inf")
    out["regression"] = {
        "R2_len": r2_len, "R2_len+ovtcsC": r2_both,
        "delta_R2": r2_both - r2_len,
        "F_add_ovtcsC_to_len": float(F),
        "F_p": float(stats.f.sf(F, df1, df2)),
    }
    # residual: OV-TCS vs residual(y ~ len)
    ry = _resid(y, L)
    out["residual"] = {
        "desc": "corr(OV-TCS_C, residual of y~track_length)",
        "pearson_r": float(stats.pearsonr(X, ry)[0]),
        "spearman_rho": float(stats.spearmanr(X, ry)[0]),
    }
    # fragmentation (if available): OV-TCS vs frag, and frag vs y
    if fr is not None and np.std(fr) > 0:
        out["fragmentation"] = {
            "ovtcsC_vs_frag_pearson": float(stats.pearsonr(X, fr)[0]),
            "ovtcsC_vs_frag_spearman": float(stats.spearmanr(X, fr)[0]),
            "frag_vs_y_pearson": float(stats.pearsonr(fr, y)[0]),
        }
    return out


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # OV-TCS_C formula
    o, L, sw = ovtcs_c_from_seq([5, 5, 5, 5])
    assert L == 4 and sw == 0 and abs(o - 0.75) < 1e-9, (o, L, sw)
    o, L, sw = ovtcs_c_from_seq([5, 7, 5, 7])           # 3 switches over L-1=3
    assert sw == 3 and abs(o - 0.0) < 1e-9, (o, L, sw)
    o, L, sw = ovtcs_c_from_seq([5, -1, 5])             # -1 not a switch
    assert sw == 0 and abs(o - (1 - 1 / 3)) < 1e-9, (o, L, sw)
    assert ovtcs_c_from_seq([5])[0] is None
    # analyze: construct data where OV-TCS adds signal beyond length.
    rng = np.random.default_rng(0)
    n = 400
    Larr = rng.uniform(2, 30, n)
    Xarr = rng.uniform(0, 1, n)
    yv = 0.1 * Larr + 0.8 * Xarr + rng.normal(0, 0.05, n)  # X is the real driver
    R = analyze(yv, Xarr, Larr)
    pc = R["partial_correlation"]["ovtcsC_y_given_len_pearson"]
    assert pc["r"] > 0.5 and pc["p"] < 1e-3, pc
    assert R["regression"]["delta_R2"] > 0.02 and R["regression"]["F_p"] < 1e-3, R["regression"]
    # null control: X carries nothing beyond length -> partial ~ 0, F not sig.
    yv2 = 0.3 * Larr + rng.normal(0, 0.05, n)
    R2 = analyze(yv2, Xarr, Larr)
    assert abs(R2["partial_correlation"]["ovtcsC_y_given_len_pearson"]["r"]) < 0.2
    print("selftest OK")


# --------------------------------------------------------------------------- #
# Per-scene harvest (imports torch/OpenYOLO only here, inside the GPU run)
# --------------------------------------------------------------------------- #
def _harvest_scene(oy3d, cfg, scene_name, cache_path, gt_dir):
    from method_scannet.streaming.wrapper import StreamingScanNetEvaluator
    from method_scannet.streaming import gt_matching as gtm
    from method_scannet.streaming.metrics import mask_iou_map
    from evaluate.scannet200.scannet_constants import VALID_CLASS_IDS_200_INST

    pred_id_to_id = {i: int(c) for i, c in enumerate(VALID_CLASS_IDS_200_INST)}

    ev = StreamingScanNetEvaluator(
        openyolo3d_instance=oy3d,
        scene_dir=str(Path("data/scannet200") / scene_name),
        depth_scale=cfg["openyolo3d"]["depth_scale"],
        depth_threshold=float(cfg["openyolo3d"].get("vis_depth_threshold", 0.05)),
        num_classes=len(cfg["network2d"]["text_prompts"]) + 1,
        topk=int(cfg["openyolo3d"].get("topk", 40)),
        topk_per_image=int(cfg["openyolo3d"].get("topk_per_image", 600)),
    )
    frequency = int(cfg["openyolo3d"].get("frequency", 10))
    ev.frame_indices = [f for f in ev.frame_indices if f % frequency == 0]
    ev.setup_scene(mask3d_cache_path=str(cache_path))
    for fi in ev.frame_indices:
        ev.step_frame(fi)

    preds = ev.compute_baseline_predictions()           # baseline (no method)
    mi = ev.baseline_accumulator._last_mask_idx
    mi = mi.cpu().numpy() if hasattr(mi, "cpu") else np.asarray(mi)
    pc = np.asarray(preds["pred_classes"])
    ps = np.asarray(preds["pred_scores"])
    # topk_per_image expands each proposal into many (instance,class) rows sorted
    # by vote distribution (mode class first). Keep the FIRST row per proposal =
    # its argmax class + score; iterating overwrites would pick the *worst* class.
    best: dict[int, tuple[int, float]] = {}
    for r in range(len(pc)):
        iid = int(mi[r])
        if iid not in best:
            best[iid] = (int(pc[r]), float(ps[r]))
    final_idx = {iid: c for iid, (c, _s) in best.items()}

    pred_history = list(ev.pred_history)
    ivm = np.asarray(ev.instance_vertex_masks, dtype=bool)  # (K, V)
    V = ivm.shape[1]

    gt_txt = str(Path(gt_dir) / f"{scene_name}.txt")
    gt_masks = gtm.load_gt_instance_masks(gt_txt, n_vertices=V)
    gt_ids = list(gt_masks)
    iou_g = {g: gtm.full_scene_iou(gt_masks[g], ivm) for g in gt_ids}  # (K,) each
    gm = gtm.build_gt_matching(pred_history, ivm, gt_masks, iou_threshold=0.5)
    frag_of_gt = {g: len({p for p in seq if p is not None}) for g, seq in gm.items()}

    # per-instance records (only instances ever live in pred_history)
    live_iids = sorted({iid for fm in pred_history for iid in fm})
    rows = []
    for iid in live_iids:
        seq = [fm[iid] for fm in pred_history if iid in fm]
        ov, L, sw = ovtcs_c_from_seq(seq)
        # best GT for this instance
        best_g, best_iou = None, 0.0
        for g in gt_ids:
            v = float(iou_g[g][iid])
            if v > best_iou:
                best_iou, best_g = v, g
        matched = best_iou >= 0.5
        gt_raw = (best_g // 1000) if best_g is not None else None
        fin = final_idx.get(iid)            # None => dropped from final preds
        correct = int(matched and fin is not None and pred_id_to_id.get(fin) == gt_raw)
        rows.append({
            "scene": scene_name, "iid": int(iid),
            "ovtcs_C": ov, "track_len": int(L), "n_switches": int(sw),
            "best_iou": round(best_iou, 4), "matched": bool(matched),
            "gt_frag": (frag_of_gt[best_g] if matched else None),
            "final_idx": (int(fin) if fin is not None else None),
            "correct": correct if matched else None,
        })

    # per-scene label-matched AP (raw-id label space, both sides consistent);
    # one row per proposal = its argmax class + score (see `best` above).
    pred_inst = {
        iid: {"vertex_mask": ivm[iid],
              "label": pred_id_to_id.get(c, -1), "score": s}
        for iid, (c, s) in best.items() if iid < ivm.shape[0]
    }
    gt_inst = {g: {"vertex_mask": gt_masks[g], "label": g // 1000} for g in gt_ids}
    ap = mask_iou_map(pred_inst, gt_inst, iou_thresholds=(0.5, 0.25))

    matched_ov = [r["ovtcs_C"] for r in rows if r["matched"] and r["ovtcs_C"] is not None]
    matched_tl = [r["track_len"] for r in rows if r["matched"] and r["ovtcs_C"] is not None]
    matched_fr = [r["gt_frag"] for r in rows if r["matched"] and r["gt_frag"] is not None]
    scene_row = {
        "scene": scene_name,
        "n_gt": len(gt_ids),
        "n_inst_matched": len(matched_ov),
        "mean_ovtcs_C": float(np.mean(matched_ov)) if matched_ov else None,
        "mean_track_len": float(np.mean(matched_tl)) if matched_tl else None,
        "mean_gt_frag": float(np.mean(matched_fr)) if matched_fr else None,
        "AP": ap.get("AP"), "AP_50": ap.get("AP_50"), "AP_25": ap.get("AP_25"),
    }
    return rows, scene_row


def _write_report(out_root: Path, inst_rows, scene_rows):
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "instances.json").write_text(json.dumps(inst_rows, indent=2))
    (out_root / "scenes.json").write_text(json.dumps(scene_rows, indent=2))

    # per-instance analysis (matched, track_len>=2)
    inst = [r for r in inst_rows
            if r["matched"] and r["ovtcs_C"] is not None and r["correct"] is not None]
    inst_res = analyze(
        [r["correct"] for r in inst],
        [r["ovtcs_C"] for r in inst],
        [r["track_len"] for r in inst],
        frag=[r["gt_frag"] for r in inst] if all(r["gt_frag"] is not None for r in inst) else None,
    )
    # per-scene analysis
    sc = [r for r in scene_rows if r["mean_ovtcs_C"] is not None and r["AP_50"] is not None]
    scene_res = analyze(
        [r["AP_50"] for r in sc],
        [r["mean_ovtcs_C"] for r in sc],
        [r["mean_track_len"] for r in sc],
        frag=[r["mean_gt_frag"] for r in sc] if all(r["mean_gt_frag"] is not None for r in sc) else None,
    )
    (out_root / "analysis_instance.json").write_text(json.dumps(inst_res, indent=2))
    (out_root / "analysis_scene.json").write_text(json.dumps(scene_res, indent=2))

    def _verdict(tag, R):
        if R.get("degenerate") or "partial_correlation" not in R:
            return f"### {tag}: n={R.get('n')} degenerate/insufficient\n"
        pc = R["partial_correlation"]["ovtcsC_y_given_len_pearson"]
        reg = R["regression"]
        zo = R["zero_order"]
        pass_gate = pc["p"] < 0.05 and reg["F_p"] < 0.05 and reg["delta_R2"] > 0
        return (
            f"### {tag} (n={R['n']})\n"
            f"- zero-order corr(OV-TCS_C, y): r={zo['ovtcsC_y_pearson']:.3f} "
            f"(len vs y r={zo['len_y_pearson']:.3f})\n"
            f"- partial corr(OV-TCS_C, y | len): r={pc['r']:.3f} p={pc['p']:.2e}\n"
            f"- nested F (add OV-TCS to len): ΔR²={reg['delta_R2']:.3f} "
            f"F={reg['F_add_ovtcsC_to_len']:.2f} p={reg['F_p']:.2e}\n"
            f"- **GATE {'PASS' if pass_gate else 'FAIL'}** "
            f"(OV-TCS {'adds' if pass_gate else 'does NOT add'} info beyond track length)\n"
        )

    notes = (
        "# OV-TCS metric validation — ScanNet200 (per-instance + per-scene)\n\n"
        "Does a track's per-frame class stability (OV-TCS_C) predict downstream\n"
        "detection quality beyond track length? EVALUATION ONLY; M22 EMA frozen.\n\n"
        + _verdict("Per-instance (y = label correctness)", inst_res) + "\n"
        + _verdict("Per-scene (y = AP_50)", scene_res) + "\n"
    )
    (out_root / "notes.md").write_text(notes)
    print("\n" + notes)
    print(f"wrote {out_root}/instances.json scenes.json analysis_*.json notes.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--cache-dir", type=str)
    ap.add_argument("--output", type=str)
    ap.add_argument("--config", default="pretrained/config_scannet200.yaml")
    ap.add_argument("--gt-dir", default="data/scannet200/ground_truth")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--scenes", nargs="*", default=None)
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.cache_dir or not args.output:
        sys.exit("--cache-dir and --output are required (or use --selftest)")

    from evaluate import SCENE_NAMES_SCANNET200
    from utils import OpenYolo3D
    from utils.utils_2d import load_yaml

    cfg = load_yaml(args.config)
    cache_dir = Path(args.cache_dir)
    scenes = list(args.scenes) if args.scenes else list(SCENE_NAMES_SCANNET200)
    if args.limit is not None:
        scenes = scenes[: args.limit]

    print(f"OV-TCS instance validation: {len(scenes)} scenes, cache={cache_dir}", flush=True)
    oy3d = OpenYolo3D(args.config)

    inst_rows, scene_rows = [], []
    t0 = time.time()
    for i, sc in enumerate(scenes):
        cache_path = cache_dir / f"{sc}.pt"
        if not cache_path.exists():
            print(f"  skip {sc} (no cache)", flush=True)
            continue
        try:
            rows, srow = _harvest_scene(oy3d, cfg, sc, cache_path, args.gt_dir)
            inst_rows.extend(rows)
            scene_rows.append(srow)
        except Exception as exc:
            print(f"  [{sc}] FAILED: {exc!r}", flush=True)
        if (i + 1) % 25 == 0 or (i + 1) == len(scenes):
            print(f"  [{i+1}/{len(scenes)}] insts={len(inst_rows)} "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)

    _write_report(Path(args.output), inst_rows, scene_rows)


if __name__ == "__main__":
    main()
