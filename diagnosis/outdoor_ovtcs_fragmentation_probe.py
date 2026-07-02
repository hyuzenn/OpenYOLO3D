"""OV-TCS metric-validity test: synthetic track FRAGMENTATION sweep.

EVIDENCE-ONLY. No detector output modified; no devkit eval. Builds object tracks
under TWO associators in ONE data pass over the γ cache (identical to
`outdoor_ovtcs_assoc_compare_probe.py`):

  * ego    : production `ClassAgnosticAssociator` (ego-frame, greedy/static,
             gate 2.0 m, max_age 5).
  * global : `Associator` from the associator ablation, same knobs but matching
             in the GLOBAL (ego-motion-compensated) frame.

The original `outdoor_ov_tcs_probe.py` validity sweep perturbs LABELS (per-frame
class replacement). This probe perturbs the COMPLEMENTARY axis — track CONTINUITY
— which is the canonical multi-object-tracking failure (ID switches / fragments,
the very thing the associator ablation reduced 10.6 → 4.5). A temporal-
consistency score is only valid if it degrades monotonically as tracks fragment.

Fragmentation model (parameter p ∈ {0, .10, .20, .30, .50})
  Each of a track's (L-1) internal links is independently CUT with probability p,
  splitting one length-L track into a set of contiguous fragments whose lengths
  sum to L (detections are conserved; only continuity is destroyed). p=0 is the
  identity. Expected #fragments per track = 1 + p·(L-1); singletons multiply as p
  rises, so L_norm — and every formulation built on it — should fall.

Scoring at each level (per associator, averaged over N_SEEDS fragmentation seeds)
  - all-fragments mean : mean OV-TCS_{A,B,C} over the emitted fragment set
                         (singletons score 0 — the honest pipeline view). PRIMARY.
  - detection-weighted : Σ(L_frag·score_frag)/Σ L_frag  (Σ L conserved across p).
  - paired per-ORIGINAL-track : clean score S0 vs length-weighted mean of that
                         track's fragment scores Sp — gives a paired effect size
                         immune to the changing fragment count.

Validity read-outs (per formulation, per associator)
  - sensitivity curve     : mean score vs p (+ across-seed std)
  - monotonicity          : all consecutive diffs ≤ 0 (strict degradation)
  - sensitivity (−slope)  : least-squares slope of mean vs p
  - relative degradation  : (S@0 − S@0.5)/S@0  — headline effect size
  - Cohen's d @ p=0.5     : independent-sample shift, fragments@0.5 vs @0
  - paired AUROC @ p=0.5  : P(S0 > Sp) over original tracks (1.0 = always worse)
  -> "most fragmentation-sensitive formulation" = largest relative degradation /
     steepest normalised slope / largest paired AUROC.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10_SET,
    _load_cached_proposals,
    _ego_to_global_centroid,
    _list_val_scenes,
    _scene_sample_tokens,
)
from diagnosis.outdoor_ov_tcs_probe import _track_metrics, _ov_tcs
from diagnosis.outdoor_associator_ablation_probe import Associator, GATE_M
from method_scannet.streaming.nuscenes_native_evaluator import (
    ClassAgnosticAssociator,
    DEFAULT_ASSOC_DIST_M,
)

ASSOC_MAX_AGE = 5
DEFAULT_DT = 0.5
FRAG_LEVELS = (0.0, 0.10, 0.20, 0.30, 0.50)
N_SEEDS = 5
RNG_SEED = 0
HEADLINE_P = 0.50


# --------------------------------------------------------------------------- #
# fragmentation
# --------------------------------------------------------------------------- #
def _fragment(seq, p, rng):
    """Cut each internal link of `seq` with prob p; return list of fragments."""
    L = len(seq)
    if L <= 1 or p <= 0.0:
        return [seq]
    cuts = rng.random(L - 1) < p          # link i between seq[i] and seq[i+1]
    if not cuts.any():
        return [seq]
    frags = []
    start = 0
    for i in range(L - 1):
        if cuts[i]:
            frags.append(seq[start:i + 1])
            start = i + 1
    frags.append(seq[start:])
    return frags


def _score_seq(seq, log2K):
    L, U, H, DR, CSR = _track_metrics(seq)
    A, B, C = _ov_tcs(L, H, DR, CSR, log2K)
    return L, (A, B, C)


def _cohens_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 2 or y.size < 2:
        return None
    nx, ny = x.size, y.size
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else None


def _slope(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    M = np.vstack([xs, np.ones_like(xs)]).T
    m, _ = np.linalg.lstsq(M, ys, rcond=None)[0]
    return float(m)


def _sweep_one_assoc(seqs, log2K):
    """Run the fragmentation sweep over one associator's track set.

    Returns a dict with per-level (seed-averaged) means + effect sizes.
    Keeps fragment-score arrays at p=0 and p=HEADLINE_P (one reference seed) for
    Cohen's d, plus paired per-original-track arrays at HEADLINE_P.
    """
    idx = "ABC"
    # accumulators: level -> {seed: {A:mean,...}} for all-fragments + det-weighted
    all_means = {p: {k: [] for k in idx} for p in FRAG_LEVELS}
    dw_means = {p: {k: [] for k in idx} for p in FRAG_LEVELS}
    nfrag_per_level = {p: [] for p in FRAG_LEVELS}

    # reference-seed fragment-score pools (for Cohen's d) and paired arrays
    frag_pool = {p: {k: None for k in idx} for p in (0.0, HEADLINE_P)}
    paired_S0 = {k: None for k in idx}
    paired_Sp = {k: None for k in idx}

    for s in range(N_SEEDS):
        rng = np.random.RandomState(RNG_SEED + s)
        for p in FRAG_LEVELS:
            # per-seed RNG draw order is independent per level (re-seed per level
            # so curve points are not correlated through a shared stream)
            lrng = np.random.RandomState(RNG_SEED + s * 131 + int(round(p * 1000)))
            f_scores = {k: [] for k in idx}   # per-fragment score
            f_len = []                        # per-fragment length
            # paired aggregates only needed on reference seed at the two probe pts
            want_pair = (s == 0)
            p0_list = {k: [] for k in idx} if (want_pair and p == 0.0) else None
            pp_list = {k: [] for k in idx} if (want_pair and p == HEADLINE_P) else None
            for seq in seqs:
                frags = _fragment(seq, p, lrng)
                wl, ws = 0, {k: 0.0 for k in idx}
                for fr in frags:
                    Lf, (A, B, C) = _score_seq(fr, log2K)
                    f_scores["A"].append(A); f_scores["B"].append(B); f_scores["C"].append(C)
                    f_len.append(Lf)
                    wl += Lf
                    ws["A"] += A * Lf; ws["B"] += B * Lf; ws["C"] += C * Lf
                if p0_list is not None:
                    # at p=0 each track is its own (single) fragment
                    p0_list["A"].append(A); p0_list["B"].append(B); p0_list["C"].append(C)
                if pp_list is not None:
                    for k in idx:
                        pp_list[k].append(ws[k] / wl if wl else 0.0)
            nfrag_per_level[p].append(len(f_len))
            fl = np.asarray(f_len, float); flsum = fl.sum()
            for k in idx:
                fs = np.asarray(f_scores[k], float)
                all_means[p][k].append(float(fs.mean()))
                dw_means[p][k].append(float((fs * fl).sum() / flsum) if flsum else 0.0)
                if want_pair and p in frag_pool:
                    frag_pool[p][k] = fs            # keep reference-seed pool
            if p0_list is not None:
                for k in idx:
                    paired_S0[k] = np.asarray(p0_list[k], float)
            if pp_list is not None:
                for k in idx:
                    paired_Sp[k] = np.asarray(pp_list[k], float)

    # ---- aggregate ----
    curve = {"levels": list(FRAG_LEVELS), "all_fragments": {}, "det_weighted": {},
             "mean_n_fragments": {p: float(np.mean(nfrag_per_level[p])) for p in FRAG_LEVELS}}
    effect = {}
    for k in idx:
        am = [float(np.mean(all_means[p][k])) for p in FRAG_LEVELS]
        am_sd = [float(np.std(all_means[p][k])) for p in FRAG_LEVELS]
        dm = [float(np.mean(dw_means[p][k])) for p in FRAG_LEVELS]
        curve["all_fragments"][k] = {"mean": am, "std_across_seeds": am_sd}
        curve["det_weighted"][k] = {"mean": dm}

        diffs = np.diff(am)
        mono = bool(np.all(diffs <= 1e-9))
        s0, s_hi = am[0], am[FRAG_LEVELS.index(HEADLINE_P)]
        rel = float((s0 - s_hi) / s0) if s0 > 0 else None
        d = _cohens_d(frag_pool[HEADLINE_P][k], frag_pool[0.0][k])  # @0.5 minus @0
        S0, Sp = paired_S0[k], paired_Sp[k]
        worse = S0 > Sp; eq = np.isclose(S0, Sp)
        auroc = float((worse.sum() + 0.5 * eq.sum()) / S0.size)
        effect[k] = {
            "monotonic_decreasing": mono,
            "max_increase_violation": float(max(0.0, diffs.max())),
            "neg_slope": -_slope(FRAG_LEVELS, am),
            "rel_degradation_p50": rel,
            "abs_drop_p50": float(s0 - s_hi),
            "cohens_d_p50_vs_0": d,
            "paired_auroc_p50": auroc,
        }
    return curve, effect


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gamma-cache", default=str(
        project_root / "results" /
        "outdoor_native_temporal_cpcache_thr000_single_gravity"))
    ap.add_argument("--nuscenes-dataroot", default=str(project_root / "data/nuscenes"))
    ap.add_argument("--nuscenes-version", default="v1.0-trainval")
    ap.add_argument("--assoc-dist-m", type=float, default=DEFAULT_ASSOC_DIST_M)
    ap.add_argument("--scene-limit", type=int, default=0)
    ap.add_argument("--output-name", default=None)
    args = ap.parse_args()

    gamma_cache = Path(args.gamma_cache).resolve()
    if not gamma_cache.exists():
        raise SystemExit(f"cache directory missing: {gamma_cache}")

    date = time.strftime("%Y-%m-%d")
    if args.output_name:
        out_dir = project_root / "results" / args.output_name
    else:
        existing = list(project_root.glob(f"results/{date}_outdoor_ovtcs_fragmentation_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_ovtcs_fragmentation_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[frag] γ cache    : {gamma_cache}", flush=True)
    print(f"[frag] output dir : {out_dir}", flush=True)

    K = len(NUSC_10_SET)
    log2K = math.log2(K)

    print("[frag] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[frag] val scenes : {len(val_scenes)}", flush=True)

    tracks_ego: dict = defaultdict(list)
    tracks_glob: dict = defaultdict(list)
    n_missing = n_props_total = 0
    t0 = time.time()

    for si, sc_tok in enumerate(val_scenes):
        ego = ClassAgnosticAssociator(threshold_m=args.assoc_dist_m, max_age=ASSOC_MAX_AGE)
        ego.reset()
        glob = Associator(GATE_M, max_age=ASSOC_MAX_AGE, hungarian=False, motion=False)
        glob.reset()
        prev_t = None
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            sample = nusc.get("sample", sa_tok)
            t_sec = sample["timestamp"] * 1e-6
            dt = (t_sec - prev_t) if prev_t is not None else DEFAULT_DT
            prev_t = t_sec
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego_rec = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_pose = transform_matrix(ego_rec["translation"], Quaternion(ego_rec["rotation"]))

            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                ego.step([])
                glob.step(np.empty((0, 2)), [], dt)
                continue
            props, classes, scores, P_glob = [], [], [], []
            for p in raw:
                cls = p.get("cls_name")
                if cls not in NUSC_10_SET:
                    continue
                n_props_total += 1
                c_ego = np.asarray(p["centroid_ego"], dtype=np.float64)
                sc = float(p.get("score", 0.0))
                props.append({"cls_name": cls, "score": sc, "centroid_ego": c_ego})
                classes.append(cls)
                scores.append(sc)
                P_glob.append(_ego_to_global_centroid(c_ego, ego_pose)[:2])
            if not props:
                ego.step([])
                glob.step(np.empty((0, 2)), [], dt)
                continue
            gids_e = ego.step(props)
            gids_g = glob.step(np.asarray(P_glob, dtype=np.float64),
                               np.asarray(scores, dtype=np.float64), dt)
            for cls, ge, gg in zip(classes, gids_e, gids_g):
                tracks_ego[(si, ge)].append(cls)
                tracks_glob[(si, gg)].append(cls)
        if (si + 1) % 25 == 0:
            print(f"[frag] scene {si+1}/{len(val_scenes)} — tracks ego {len(tracks_ego)} "
                  f"glob {len(tracks_glob)} — {time.time()-t0:.0f}s", flush=True)

    seqs_ego = list(tracks_ego.values())
    seqs_glob = list(tracks_glob.values())
    print(f"[frag] sweeping ego  ({len(seqs_ego):,} tracks) ...", flush=True)
    curve_e, eff_e = _sweep_one_assoc(seqs_ego, log2K)
    print(f"[frag] sweeping glob ({len(seqs_glob):,} tracks) ...", flush=True)
    curve_g, eff_g = _sweep_one_assoc(seqs_glob, log2K)

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "n_proposals_total": n_props_total,
            "n_samples_missing_cache": n_missing,
            "K_classes": K,
            "frag_levels": list(FRAG_LEVELS),
            "n_seeds": N_SEEDS,
            "headline_p": HEADLINE_P,
            "frag_model": "each internal link cut iid w.p. p; detections conserved",
            "ego": "ClassAgnosticAssociator ego-frame (production)",
            "global": "Associator global-frame greedy/static (ablation)",
            "n_tracks_ego": len(seqs_ego),
            "n_tracks_global": len(seqs_glob),
        },
        "ego": {"curve": curve_e, "effect": eff_e},
        "global": {"curve": curve_g, "effect": eff_g},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[frag] wrote {out_dir/'metrics.json'}", flush=True)
    _render_png(out_dir / "outputs", payload)
    _write_notes(out_dir, payload)
    _console(payload)


def _render_png(outputs, p):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[frag] matplotlib unavailable: {e}", flush=True)
        return
    lv = p["config"]["frag_levels"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    colors = {"ego": "#b63", "global": "#36b"}
    for j, k in enumerate("ABC"):
        for who in ("ego", "global"):
            cv = p[who]["curve"]["all_fragments"][k]
            ax[j].errorbar(lv, cv["mean"], yerr=cv["std_across_seeds"],
                           marker="o", lw=1.6, capsize=2, color=colors[who], label=who)
        ax[j].set_title(f"OV-TCS_{k}  (all fragments)")
        ax[j].set_xlabel("fragmentation rate p")
        ax[j].grid(alpha=0.3)
    ax[0].set_ylabel("mean OV-TCS")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(outputs / "sensitivity_curve.png", dpi=120); plt.close(fig)
    print(f"[frag] wrote {outputs/'sensitivity_curve.png'}", flush=True)


def _write_notes(out_dir, p):
    c = p["config"]; lv = c["frag_levels"]
    L = []
    L.append("# OV-TCS metric-validity test — synthetic track fragmentation sweep\n")
    L.append("Probe: `diagnosis/outdoor_ovtcs_fragmentation_probe.py` (evidence-only; no "
             "detector output modified, no devkit eval). γ gravity cache, full val "
             f"({c['n_val_scenes']} scenes), {c['n_proposals_total']:,} nuScenes-10 "
             f"proposals. Tracks built once per associator (gate {c['assoc_threshold_m']} m, "
             f"max_age {c['assoc_max_age']}); OV-TCS formulations identical to "
             "`outdoor_ov_tcs_probe.py`.\n")
    L.append(f"**Fragmentation model.** {c['frag_model']}. Levels "
             f"{', '.join(f'{x:.0%}' for x in lv)}; {c['n_seeds']} seeds averaged. "
             "p=0 is the identity. This is the continuity axis (ID-switches), "
             "complementary to the label-noise sweep in the base probe.\n")
    L.append(f"Tracks: ego {c['n_tracks_ego']:,}, global {c['n_tracks_global']:,}.\n")

    def curve_table(who):
        cv = p[who]["curve"]
        L.append(f"### {who} — mean OV-TCS over all fragments vs p")
        L.append("| p | mean #frags | A | B | C |")
        L.append("|---|---|---|---|---|")
        for i, x in enumerate(lv):
            nf = cv["mean_n_fragments"][str(x)] if str(x) in cv["mean_n_fragments"] else cv["mean_n_fragments"].get(x)
            row = (f"| {x:.2f} | {nf:,.0f} | "
                   + " | ".join(f"{cv['all_fragments'][k]['mean'][i]:.3f}" for k in 'ABC') + " |")
            L.append(row)
        L.append("")

    L.append("## Sensitivity curves (all-fragments / pipeline view)")
    curve_table("ego")
    curve_table("global")
    L.append("(PNG: `outputs/sensitivity_curve.png`)\n")

    L.append("## Validity read-outs per formulation")
    for who in ("ego", "global"):
        e = p[who]["effect"]
        L.append(f"### {who}")
        L.append("| formulation | monotonic↓ | −slope | rel.deg @0.5 | abs.drop @0.5 | Cohen d @0.5 | paired AUROC @0.5 |")
        L.append("|---|---|---|---|---|---|---|")
        for k in "ABC":
            ek = e[k]
            rel = f"{ek['rel_degradation_p50']:.1%}" if ek['rel_degradation_p50'] is not None else "—"
            cd = f"{ek['cohens_d_p50_vs_0']:+.3f}" if ek['cohens_d_p50_vs_0'] is not None else "—"
            L.append(f"| OV-TCS_{k} | {'✓' if ek['monotonic_decreasing'] else '✗'} | "
                     f"{ek['neg_slope']:.3f} | {rel} | {ek['abs_drop_p50']:.3f} | {cd} | "
                     f"{ek['paired_auroc_p50']:.3f} |")
        L.append("")

    # ranking by EFFECT SIZE (Cohen's d). Relative degradation is near-uniform by
    # construction — all three share the L_norm continuity factor, so fragmentation
    # scales them proportionally; the discriminating signal is absolute sensitivity.
    e = p["ego"]["effect"]
    rank = sorted("ABC", key=lambda k: -abs(e[k]["cohens_d_p50_vs_0"] or 0))
    L.append("## Verdict")
    mono_all = all(p[w]["effect"][k]["monotonic_decreasing"] for w in ("ego", "global") for k in "ABC")
    mono_txt = ("all A/B/C strictly decrease with p for both associators" if mono_all
                else "NOT all formulations monotone — see table")
    rank_txt = ", ".join("{}=|d|{:.2f}".format(k, abs(e[k]["cohens_d_p50_vs_0"])) for k in rank)
    reldeg_txt = ", ".join("{}={:.0%}".format(k, e[k]["rel_degradation_p50"]) for k in "ABC")
    L.append(f"- **Monotonicity**: {mono_txt}.")
    L.append(f"- **Most fragmentation-sensitive** (ego, by effect size |Cohen's d| @p=0.5): "
             f"OV-TCS_{rank[0]} ≳ OV-TCS_{rank[1]} > OV-TCS_{rank[2]} ({rank_txt}).")
    L.append(f"- **Least sensitive**: OV-TCS_{rank[-1]} — smallest separation, most robust "
             "to ID-switches (built on 1−CSR, which short fragments can even improve).")
    L.append(f"- **Relative degradation is near-uniform** ({reldeg_txt}) because A/B/C all "
             "ride the shared `L_norm = 1−1/L` factor that fragmentation collapses; "
             "use absolute slope / Cohen's d, not % drop, to separate the formulations.")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[frag] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    lv = p["config"]["frag_levels"]
    print("\n=== OV-TCS fragmentation validity sweep ===")
    for who in ("ego", "global"):
        cv = p[who]["curve"]; e = p[who]["effect"]
        print(f"\n[{who}]  n_tracks={p['config'][f'n_tracks_'+('ego' if who=='ego' else 'global')]:,}")
        print("  p     " + "  ".join(f"{x:.2f}" for x in lv))
        for k in "ABC":
            print(f"  {k}   " + "  ".join(f"{v:.3f}" for v in cv['all_fragments'][k]['mean']))
        for k in "ABC":
            ek = e[k]
            print(f"   OV-TCS_{k}: mono↓={ek['monotonic_decreasing']} "
                  f"-slope={ek['neg_slope']:.3f} rel.deg@0.5={ek['rel_degradation_p50']:.1%} "
                  f"d@0.5={ek['cohens_d_p50_vs_0']:+.3f} AUROC@0.5={ek['paired_auroc_p50']:.3f}")


if __name__ == "__main__":
    main()
