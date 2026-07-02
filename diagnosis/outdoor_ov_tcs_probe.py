"""OV-TCS (Open-Vocabulary Temporal Consistency Score) diagnostic.

EVIDENCE-ONLY. No detector output is modified; no devkit eval. We reuse the
existing CenterPoint γ cache and the existing streaming associator
(``ClassAgnosticAssociator``, geometry-only, 2.0 m gate, max_age 5) to build
object tracks across consecutive frames, exactly as
``outdoor_temporal_consistency_probe.py`` does. GT read-off is dropped — OV-TCS
is a property of the *predicted-label sequence* of a track, not of accuracy.

Per-track metrics
  1. track length          L  = #frames the track is observed
  2. unique label count    U  = |set(class sequence)|
  3. label entropy         H  = Shannon entropy (bits) of within-track labels
  4. dominant-label ratio  DR = max_class_count / L            in [0,1]
  5. class-switch rate      CSR = #(consecutive label changes)/(L-1)  (L>=2)

Reported
  - per-metric distribution (mean/std/percentiles)
  - corr(L, H)   and  corr(L, CSR)   (Pearson + Spearman)
  - per dominant-class statistics
  - histogram of entropy, histogram of dominant-label ratio

Candidate OV-TCS formulations (all in [0,1], 1 = maximally consistent)
  L_norm = 1 - 1/L            (parameter-free, saturating; L=1 -> 0)
  H_norm = H / log2(K)        (K = #classes = 10; global normalisation)
  OV-TCS_A = L_norm * (1 - H_norm)
  OV-TCS_B = L_norm * DominantRatio
  OV-TCS_C = L_norm * (1 - CSR)

Evaluated on three axes
  - stability      : pairwise Spearman agreement; single- vs multi-class
                     separation (Cohen's d) — a stable score puts genuinely
                     consistent tracks in a tight high cluster.
  - dynamic range  : std, IQR, robust range (p95-p5), bin coverage, histogram
                     entropy — how much of [0,1] the score actually uses.
  - noise sensitivity: inject synthetic label noise at rate rho (each frame's
                     label replaced w.p. rho by a draw from the empirical class
                     prior), recompute the score, sweep rho. Report mean-score
                     vs rho, slope (sensitivity), monotonicity, and paired
                     clean-vs-noised discrimination (AUROC).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from diagnosis.outdoor_proposal_recall_probe import (
    NUSC_10_SET,
    _load_cached_proposals,
    _list_val_scenes,
    _scene_sample_tokens,
)
from method_scannet.streaming.nuscenes_native_evaluator import (
    ClassAgnosticAssociator,
    DEFAULT_ASSOC_DIST_M,
)
from nuscenes.nuscenes import NuScenes

ASSOC_MAX_AGE = 5
NOISE_RHOS = (0.0, 0.05, 0.10, 0.20, 0.40)
NOISE_AUROC_RHO = 0.20
RNG_SEED = 0


# --------------------------------------------------------------------------- #
# per-track metric primitives
# --------------------------------------------------------------------------- #
def _entropy_bits(seq) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    h = 0.0
    for c in Counter(seq).values():
        p = c / n
        h -= p * math.log2(p)
    return float(h)


def _dominant_count(seq) -> int:
    return max(Counter(seq).values()) if seq else 0


def _switches(seq) -> int:
    return sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)


def _track_metrics(seq):
    """Return (L, U, H, DR, CSR_or_None)."""
    L = len(seq)
    U = len(set(seq))
    H = _entropy_bits(seq)
    DR = _dominant_count(seq) / L if L else 0.0
    CSR = (_switches(seq) / (L - 1)) if L >= 2 else None
    return L, U, H, DR, CSR


def _dist(arr) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def _hist(arr, lo, hi, nbins):
    counts, edges = np.histogram(np.asarray(arr, dtype=np.float64), bins=nbins, range=(lo, hi))
    return {"bin_edges": [float(x) for x in edges], "counts": [int(c) for c in counts]}


def _ascii_hist(counts, edges, width=46):
    mx = max(counts) if counts else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / mx)) if mx else ""
        lines.append(f"  [{edges[i]:.2f},{edges[i+1]:.2f}) {c:>8} |{bar}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# OV-TCS formulations
# --------------------------------------------------------------------------- #
def _l_norm(L):
    return 1.0 - 1.0 / L if L >= 1 else 0.0


def _ov_tcs(L, H, DR, CSR, log2K):
    Ln = _l_norm(L)
    Hn = H / log2K if log2K > 0 else 0.0
    csr = 0.0 if CSR is None else CSR        # L=1 -> Ln=0 anyway
    A = Ln * (1.0 - Hn)
    B = Ln * DR
    C = Ln * (1.0 - csr)
    return A, B, C


def _cohens_d(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return None
    nx, ny = x.size, y.size
    sp = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else None


def _coverage(arr, nbins=20):
    """Fraction of [0,1] bins that contain >=1 track (uses-the-range measure)."""
    counts, _ = np.histogram(np.asarray(arr), bins=nbins, range=(0.0, 1.0))
    return float(np.count_nonzero(counts) / nbins)


def _hist_entropy(arr, nbins=20):
    """Shannon entropy (bits, normalised to [0,1]) of the score histogram."""
    counts, _ = np.histogram(np.asarray(arr), bins=nbins, range=(0.0, 1.0))
    p = counts / counts.sum() if counts.sum() else counts
    h = -sum(pi * math.log2(pi) for pi in p if pi > 0)
    return float(h / math.log2(nbins))


# --------------------------------------------------------------------------- #
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
        existing = list(project_root.glob(f"results/{date}_outdoor_ov_tcs_v*"))
        out_dir = project_root / "results" / f"{date}_outdoor_ov_tcs_v{len(existing)+1:02d}"
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    print(f"[ovtcs] γ cache    : {gamma_cache}", flush=True)
    print(f"[ovtcs] output dir : {out_dir}", flush=True)

    classes_sorted = sorted(NUSC_10_SET)
    K = len(classes_sorted)
    log2K = math.log2(K)

    print("[ovtcs] loading NuScenes ...", flush=True)
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_dataroot, verbose=False)
    val_scenes = _list_val_scenes(nusc)
    if args.scene_limit > 0:
        val_scenes = val_scenes[: args.scene_limit]
    print(f"[ovtcs] val scenes : {len(val_scenes)}", flush=True)

    # ----------------------------------------------------------------------- #
    # build geometry-only tracks; collect label sequences only
    # ----------------------------------------------------------------------- #
    tracks: dict = defaultdict(list)            # (scene_idx, gid) -> [cls,...]
    cls_prior = Counter()                       # empirical proposal-class prior
    t0 = time.time()
    n_missing = n_props_total = n_props_non10 = 0
    for si, sc_tok in enumerate(val_scenes):
        assoc = ClassAgnosticAssociator(threshold_m=args.assoc_dist_m, max_age=ASSOC_MAX_AGE)
        assoc.reset()
        for sa_tok in _scene_sample_tokens(nusc, sc_tok):
            raw = _load_cached_proposals(gamma_cache, sa_tok, suffix="")
            if raw is None:
                n_missing += 1
                continue
            props = []
            for p in raw:
                n_props_total += 1
                cls = p.get("cls_name")
                if cls not in NUSC_10_SET:
                    n_props_non10 += 1
                    continue
                props.append({
                    "cls_name": cls,
                    "score": float(p.get("score", 0.0)),
                    "centroid_ego": np.asarray(p["centroid_ego"], dtype=np.float64),
                })
                cls_prior[cls] += 1
            gids = assoc.step(props)
            for p, gid in zip(props, gids):
                tracks[(si, gid)].append(p["cls_name"])
        if (si + 1) % 25 == 0:
            print(f"[ovtcs] scene {si+1}/{len(val_scenes)} — tracks {len(tracks)} "
                  f"— {time.time()-t0:.0f}s", flush=True)

    # ----------------------------------------------------------------------- #
    # per-track metrics
    # ----------------------------------------------------------------------- #
    cls_idx = {c: i for i, c in enumerate(classes_sorted)}
    L_arr, U_arr, H_arr, DR_arr, CSR_arr = [], [], [], [], []
    dom_idx = []                                # dominant class index per track
    seqs = []                                   # keep sequences for noise sweep
    A_arr, B_arr, C_arr = [], [], []
    for key, seq in tracks.items():
        L, U, H, DR, CSR = _track_metrics(seq)
        dom = Counter(seq).most_common(1)[0][0]
        A, B, C = _ov_tcs(L, H, DR, CSR, log2K)
        L_arr.append(L); U_arr.append(U); H_arr.append(H)
        DR_arr.append(DR); CSR_arr.append(CSR if CSR is not None else np.nan)
        dom_idx.append(cls_idx[dom])
        seqs.append(seq)
        A_arr.append(A); B_arr.append(B); C_arr.append(C)

    L_np = np.asarray(L_arr, dtype=np.float64)
    U_np = np.asarray(U_arr, dtype=np.float64)
    H_np = np.asarray(H_arr, dtype=np.float64)
    DR_np = np.asarray(DR_arr, dtype=np.float64)
    CSR_np = np.asarray(CSR_arr, dtype=np.float64)        # NaN for L=1
    dom_np = np.asarray(dom_idx, dtype=np.int64)
    A_np = np.asarray(A_arr, dtype=np.float64)
    B_np = np.asarray(B_arr, dtype=np.float64)
    C_np = np.asarray(C_arr, dtype=np.float64)
    multi2 = (L_np >= 2)                                  # tracks with transitions
    n_tracks = L_np.size
    n_singleton = int((L_np < 2).sum())

    # ----------------------------------------------------------------------- #
    # distributions
    # ----------------------------------------------------------------------- #
    distributions = {
        "track_length": _dist(L_np),
        "unique_label_count": _dist(U_np),
        "label_entropy_bits": _dist(H_np),
        "dominant_label_ratio": _dist(DR_np),
        "class_switch_rate_L>=2": _dist(CSR_np[multi2]),
    }

    # ----------------------------------------------------------------------- #
    # correlations
    # ----------------------------------------------------------------------- #
    def _corr(x, y):
        if x.size < 3:
            return {"n": int(x.size), "pearson_r": None, "spearman_rho": None}
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        return {"n": int(x.size),
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp)}

    correlations = {
        "length_vs_entropy_all": _corr(L_np, H_np),
        "length_vs_entropy_L>=2": _corr(L_np[multi2], H_np[multi2]),
        "length_vs_switchrate_L>=2": _corr(L_np[multi2], CSR_np[multi2]),
    }

    # ----------------------------------------------------------------------- #
    # per-class statistics (by dominant class)
    # ----------------------------------------------------------------------- #
    per_class = {}
    for c in classes_sorted:
        m = (dom_np == cls_idx[c])
        if not m.any():
            continue
        m2 = m & multi2
        per_class[c] = {
            "n_tracks": int(m.sum()),
            "mean_length": float(L_np[m].mean()),
            "mean_unique": float(U_np[m].mean()),
            "mean_entropy_bits": float(H_np[m].mean()),
            "mean_dominant_ratio": float(DR_np[m].mean()),
            "mean_switch_rate_L>=2": float(CSR_np[m2].mean()) if m2.any() else None,
            "mean_ov_tcs_A": float(A_np[m].mean()),
            "mean_ov_tcs_B": float(B_np[m].mean()),
            "mean_ov_tcs_C": float(C_np[m].mean()),
        }

    # ----------------------------------------------------------------------- #
    # histograms (entropy, dominant ratio) + PNGs
    # ----------------------------------------------------------------------- #
    h_lo, h_hi = 0.0, float(max(1.0, math.ceil(H_np.max() * 10) / 10))
    histograms = {
        "entropy_bits": _hist(H_np, h_lo, h_hi, 20),
        "dominant_label_ratio": _hist(DR_np, 0.0, 1.0, 20),
    }
    _render_pngs(out_dir / "outputs", H_np, DR_np, A_np, B_np, C_np, h_hi)

    # ----------------------------------------------------------------------- #
    # OV-TCS evaluation: distribution + dynamic range
    # ----------------------------------------------------------------------- #
    def _range_stats(arr):
        a = np.asarray(arr)
        return {
            **_dist(a),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "robust_range_p95_p5": float(np.percentile(a, 95) - np.percentile(a, 5)),
            "full_range": float(a.max() - a.min()),
            "bin_coverage_20": _coverage(a, 20),
            "hist_entropy_norm_20": _hist_entropy(a, 20),
            "frac_at_zero": float(np.mean(a <= 1e-9)),
        }

    ov_tcs = {
        "definition": {
            "L_norm": "1 - 1/L",
            "H_norm": "H / log2(K), K=10",
            "A": "L_norm * (1 - H_norm)",
            "B": "L_norm * DominantRatio",
            "C": "L_norm * (1 - CSR)",
        },
        "distribution": {
            "A": _range_stats(A_np),
            "B": _range_stats(B_np),
            "C": _range_stats(C_np),
        },
    }

    # stability: pairwise Spearman agreement (subsample for speed if huge)
    idx = np.arange(n_tracks)
    if n_tracks > 200_000:
        rs = np.random.RandomState(RNG_SEED)
        idx = rs.choice(n_tracks, 200_000, replace=False)
    ov_tcs["stability"] = {
        "spearman_A_B": float(spearmanr(A_np[idx], B_np[idx]).correlation),
        "spearman_A_C": float(spearmanr(A_np[idx], C_np[idx]).correlation),
        "spearman_B_C": float(spearmanr(B_np[idx], C_np[idx]).correlation),
    }
    # single-class (U==1, L>=2 genuinely consistent) vs multi-class separation
    stable_mask = (U_np == 1) & multi2
    unstable_mask = (U_np >= 2)
    sep = {}
    for name, arr in (("A", A_np), ("B", B_np), ("C", C_np)):
        s, u = arr[stable_mask], arr[unstable_mask]
        sep[name] = {
            "mean_singleclass": float(s.mean()) if s.size else None,
            "std_singleclass": float(s.std()) if s.size else None,
            "mean_multiclass": float(u.mean()) if u.size else None,
            "std_multiclass": float(u.std()) if u.size else None,
            "cohens_d": _cohens_d(s, u),
        }
    ov_tcs["stability"]["singleclass_vs_multiclass"] = {
        "n_singleclass_L>=2": int(stable_mask.sum()),
        "n_multiclass": int(unstable_mask.sum()),
        **sep,
    }

    # ----------------------------------------------------------------------- #
    # noise sensitivity sweep
    # ----------------------------------------------------------------------- #
    prior_classes = classes_sorted
    prior_p = np.asarray([cls_prior[c] for c in prior_classes], dtype=np.float64)
    prior_p = prior_p / prior_p.sum()
    rng = np.random.RandomState(RNG_SEED)

    # operate on multi-frame tracks (noise on length-1 is meaningless)
    seqs2 = [seqs[i] for i in range(n_tracks) if multi2[i]]

    def _noise_scores(rho):
        A_l, B_l, C_l = [], [], []
        for seq in seqs2:
            if rho > 0:
                draws = rng.random(len(seq)) < rho
                if draws.any():
                    rep = rng.choice(len(prior_classes), size=int(draws.sum()), p=prior_p)
                    seq = list(seq)
                    ri = 0
                    for j in range(len(seq)):
                        if draws[j]:
                            seq[j] = prior_classes[rep[ri]]; ri += 1
            L, U, H, DR, CSR = _track_metrics(seq)
            A, B, C = _ov_tcs(L, H, DR, CSR, log2K)
            A_l.append(A); B_l.append(B); C_l.append(C)
        return np.asarray(A_l), np.asarray(B_l), np.asarray(C_l)

    sweep = {"rho": list(NOISE_RHOS), "mean_A": [], "mean_B": [], "mean_C": []}
    clean_scores = noised_scores = None
    for rho in NOISE_RHOS:
        a, b, c = _noise_scores(rho)
        sweep["mean_A"].append(float(a.mean()))
        sweep["mean_B"].append(float(b.mean()))
        sweep["mean_C"].append(float(c.mean()))
        if rho == 0.0:
            clean_scores = (a, b, c)
        if abs(rho - NOISE_AUROC_RHO) < 1e-9:
            noised_scores = (a, b, c)

    # sensitivity = -slope of mean score vs rho (least squares); monotonic check
    rho_np = np.asarray(NOISE_RHOS)
    def _slope(ys):
        ys = np.asarray(ys)
        A_ = np.vstack([rho_np, np.ones_like(rho_np)]).T
        m, _ = np.linalg.lstsq(A_, ys, rcond=None)[0]
        return float(m)
    # paired clean-vs-noised discrimination (fraction clean>noised; AUROC-like)
    def _paired_auroc(clean, noised):
        gt = clean > noised
        eq = np.isclose(clean, noised)
        return float((gt.sum() + 0.5 * eq.sum()) / clean.size)

    ov_tcs["noise_sensitivity"] = {
        "rho_grid": list(NOISE_RHOS),
        "n_tracks_L>=2": len(seqs2),
        "noise_model": "per-frame label replaced w.p. rho by draw from empirical class prior",
        "mean_score_vs_rho": {"A": sweep["mean_A"], "B": sweep["mean_B"], "C": sweep["mean_C"]},
        "sensitivity_neg_slope": {
            "A": -_slope(sweep["mean_A"]),
            "B": -_slope(sweep["mean_B"]),
            "C": -_slope(sweep["mean_C"]),
        },
        "monotonic_decreasing": {
            "A": bool(np.all(np.diff(sweep["mean_A"]) <= 1e-9)),
            "B": bool(np.all(np.diff(sweep["mean_B"]) <= 1e-9)),
            "C": bool(np.all(np.diff(sweep["mean_C"]) <= 1e-9)),
        },
        f"paired_auroc_clean_vs_rho{NOISE_AUROC_RHO}": {
            "A": _paired_auroc(clean_scores[0], noised_scores[0]),
            "B": _paired_auroc(clean_scores[1], noised_scores[1]),
            "C": _paired_auroc(clean_scores[2], noised_scores[2]),
        },
    }

    # ----------------------------------------------------------------------- #
    # persist
    # ----------------------------------------------------------------------- #
    np.savez_compressed(
        out_dir / "outputs" / "per_track_metrics.npz",
        L=L_np, U=U_np, H=H_np, DR=DR_np, CSR=CSR_np, dom_idx=dom_np,
        A=A_np, B=B_np, C=C_np, classes=np.asarray(classes_sorted))

    payload = {
        "config": {
            "gamma_cache": str(gamma_cache),
            "n_val_scenes": len(val_scenes),
            "associator": "ClassAgnosticAssociator (geometry-only)",
            "assoc_threshold_m": args.assoc_dist_m,
            "assoc_max_age": ASSOC_MAX_AGE,
            "n_proposals_total": n_props_total,
            "n_proposals_dropped_non_nusc10": n_props_non10,
            "n_samples_missing_cache": n_missing,
            "n_tracks_total": n_tracks,
            "n_singleton_tracks": n_singleton,
            "K_classes": K,
            "noise_seed": RNG_SEED,
        },
        "distributions": distributions,
        "correlations": correlations,
        "per_class": per_class,
        "histograms": histograms,
        "ov_tcs": ov_tcs,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[ovtcs] wrote {out_dir/'metrics.json'}", flush=True)

    _write_notes(out_dir, payload, H_np, DR_np)
    _console(payload)


# --------------------------------------------------------------------------- #
def _render_pngs(outputs, H, DR, A, B, C, h_hi):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[ovtcs] matplotlib unavailable, skipping PNGs: {e}", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(H, bins=20, range=(0, h_hi), color="#3b6", edgecolor="k", lw=0.3)
    ax[0].set_title("within-track label entropy (bits)"); ax[0].set_xlabel("H"); ax[0].set_ylabel("tracks")
    ax[1].hist(DR, bins=20, range=(0, 1), color="#36b", edgecolor="k", lw=0.3)
    ax[1].set_title("dominant-label ratio"); ax[1].set_xlabel("DR")
    fig.tight_layout(); fig.savefig(outputs / "hist_entropy_dominant.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for a, arr, name in zip(ax, (A, B, C), ("OV-TCS_A", "OV-TCS_B", "OV-TCS_C")):
        a.hist(arr, bins=20, range=(0, 1), color="#b63", edgecolor="k", lw=0.3)
        a.set_title(name); a.set_xlabel("score")
    ax[0].set_ylabel("tracks")
    fig.tight_layout(); fig.savefig(outputs / "hist_ov_tcs.png", dpi=110); plt.close(fig)
    print(f"[ovtcs] wrote PNGs to {outputs}", flush=True)


def _write_notes(out_dir, p, H_np, DR_np):
    c = p["config"]; d = p["distributions"]; cor = p["correlations"]; ov = p["ov_tcs"]
    ns = ov["noise_sensitivity"]; st = ov["stability"]
    he = _hist(H_np, 0.0, p["histograms"]["entropy_bits"]["bin_edges"][-1], 20)
    dh = _hist(DR_np, 0.0, 1.0, 20)
    lines = []
    lines.append("# OV-TCS — Open-Vocabulary Temporal Consistency Score diagnostic\n")
    lines.append("Probe: `diagnosis/outdoor_ov_tcs_probe.py` (evidence-only; no detector")
    lines.append("output modified, no devkit eval). γ gravity cache, full val "
                 f"({c['n_val_scenes']} scenes), geometry-only `ClassAgnosticAssociator` "
                 f"(gate {c['assoc_threshold_m']} m, max_age {c['assoc_max_age']}), per-scene reset.\n")
    lines.append(f"Corpus: {c['n_proposals_total']:,} proposals "
                 f"({c['n_proposals_dropped_non_nusc10']:,} dropped non-nuScenes-10), "
                 f"**{c['n_tracks_total']:,} tracks** ({c['n_singleton_tracks']:,} singletons).\n")
    lines.append("## Per-track metric distributions")
    lines.append("| metric | mean | std | p5 | p50 | p95 |")
    lines.append("|---|---|---|---|---|---|")
    name_map = [("track length", "track_length"), ("unique labels", "unique_label_count"),
                ("entropy (bits)", "label_entropy_bits"), ("dominant ratio", "dominant_label_ratio"),
                ("class-switch rate (L≥2)", "class_switch_rate_L>=2")]
    for lbl, k in name_map:
        s = d[k]
        lines.append(f"| {lbl} | {s['mean']:.3f} | {s['std']:.3f} | {s['p5']:.3f} | {s['p50']:.3f} | {s['p95']:.3f} |")
    lines.append("")
    lines.append("## Correlations")
    ce = cor["length_vs_entropy_all"]; cs = cor["length_vs_switchrate_L>=2"]
    lines.append(f"- **length ↔ entropy** (all tracks): Pearson r={ce['pearson_r']:+.3f}, "
                 f"Spearman ρ={ce['spearman_rho']:+.3f}  (n={ce['n']:,})")
    lines.append(f"- **length ↔ switch-rate** (L≥2): Pearson r={cs['pearson_r']:+.3f}, "
                 f"Spearman ρ={cs['spearman_rho']:+.3f}  (n={cs['n']:,})\n")
    lines.append("## Per dominant-class")
    lines.append("| class | n | mean_L | mean_H | mean_DR | mean_CSR | A | B | C |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cls, b in sorted(p["per_class"].items(), key=lambda kv: -kv[1]["n_tracks"]):
        csr = b["mean_switch_rate_L>=2"]
        lines.append(f"| {cls} | {b['n_tracks']:,} | {b['mean_length']:.2f} | "
                     f"{b['mean_entropy_bits']:.3f} | {b['mean_dominant_ratio']:.3f} | "
                     f"{'—' if csr is None else f'{csr:.3f}'} | {b['mean_ov_tcs_A']:.3f} | "
                     f"{b['mean_ov_tcs_B']:.3f} | {b['mean_ov_tcs_C']:.3f} |")
    lines.append("")
    lines.append("## Histogram — entropy (bits)")
    lines.append("```")
    lines.append(_ascii_hist(he["counts"], he["bin_edges"]))
    lines.append("```")
    lines.append("## Histogram — dominant-label ratio")
    lines.append("```")
    lines.append(_ascii_hist(dh["counts"], dh["bin_edges"]))
    lines.append("```")
    lines.append("(PNGs: `outputs/hist_entropy_dominant.png`, `outputs/hist_ov_tcs.png`)\n")
    lines.append("## OV-TCS formulation comparison")
    lines.append("| | A = L·(1−Hₙ) | B = L·DR | C = L·(1−CSR) |")
    lines.append("|---|---|---|---|")
    dr = ov["distribution"]
    lines.append(f"| mean | {dr['A']['mean']:.3f} | {dr['B']['mean']:.3f} | {dr['C']['mean']:.3f} |")
    lines.append(f"| std | {dr['A']['std']:.3f} | {dr['B']['std']:.3f} | {dr['C']['std']:.3f} |")
    lines.append(f"| IQR | {dr['A']['iqr']:.3f} | {dr['B']['iqr']:.3f} | {dr['C']['iqr']:.3f} |")
    lines.append(f"| robust range p95−p5 | {dr['A']['robust_range_p95_p5']:.3f} | "
                 f"{dr['B']['robust_range_p95_p5']:.3f} | {dr['C']['robust_range_p95_p5']:.3f} |")
    lines.append(f"| bin coverage (20) | {dr['A']['bin_coverage_20']:.2f} | "
                 f"{dr['B']['bin_coverage_20']:.2f} | {dr['C']['bin_coverage_20']:.2f} |")
    lines.append(f"| hist entropy (norm) | {dr['A']['hist_entropy_norm_20']:.3f} | "
                 f"{dr['B']['hist_entropy_norm_20']:.3f} | {dr['C']['hist_entropy_norm_20']:.3f} |")
    sc = st["singleclass_vs_multiclass"]
    lines.append(f"| Cohen's d (stable vs multi) | {sc['A']['cohens_d']:.3f} | "
                 f"{sc['B']['cohens_d']:.3f} | {sc['C']['cohens_d']:.3f} |")
    sl = ns["sensitivity_neg_slope"]
    lines.append(f"| noise sensitivity (−slope) | {sl['A']:.3f} | {sl['B']:.3f} | {sl['C']:.3f} |")
    au = ns[f"paired_auroc_clean_vs_rho{NOISE_AUROC_RHO}"]
    lines.append(f"| clean-vs-noisy AUROC @ρ={NOISE_AUROC_RHO} | {au['A']:.3f} | {au['B']:.3f} | {au['C']:.3f} |")
    lines.append("")
    lines.append("### mean OV-TCS vs injected noise ρ")
    lines.append("| ρ | " + " | ".join(f"{r:.2f}" for r in ns["rho_grid"]) + " |")
    lines.append("|---|" + "---|" * len(ns["rho_grid"]))
    for k in ("A", "B", "C"):
        lines.append(f"| {k} | " + " | ".join(f"{v:.3f}" for v in ns["mean_score_vs_rho"][k]) + " |")
    lines.append("")
    lines.append(f"Pairwise rank agreement (Spearman): A–B {st['spearman_A_B']:+.3f}, "
                 f"A–C {st['spearman_A_C']:+.3f}, B–C {st['spearman_B_C']:+.3f}.")
    with open(out_dir / "notes.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ovtcs] wrote {out_dir/'notes.md'}", flush=True)


def _console(p):
    c = p["config"]; ov = p["ov_tcs"]; ns = ov["noise_sensitivity"]; dr = ov["distribution"]
    print("\n=== OV-TCS overall ===")
    print(f"  tracks={c['n_tracks_total']:,}  singleton={c['n_singleton_tracks']:,}")
    ce = p["correlations"]["length_vs_entropy_all"]
    cs = p["correlations"]["length_vs_switchrate_L>=2"]
    print(f"  corr(L,H)  Pearson={ce['pearson_r']:+.3f} Spearman={ce['spearman_rho']:+.3f}")
    print(f"  corr(L,CSR) Pearson={cs['pearson_r']:+.3f} Spearman={cs['spearman_rho']:+.3f}")
    print("  formulation     mean    std    p95-p5  cohens_d  noiseSens  AUROC")
    sc = ov["stability"]["singleclass_vs_multiclass"]
    au = ns[f"paired_auroc_clean_vs_rho{NOISE_AUROC_RHO}"]
    sl = ns["sensitivity_neg_slope"]
    for k in ("A", "B", "C"):
        print(f"   OV-TCS_{k}    {dr[k]['mean']:.3f}  {dr[k]['std']:.3f}  "
              f"{dr[k]['robust_range_p95_p5']:.3f}    {sc[k]['cohens_d']:+.3f}    "
              f"{sl[k]:.3f}     {au[k]:.3f}")


if __name__ == "__main__":
    main()
