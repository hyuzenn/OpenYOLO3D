"""Aggregate per-sample Tier-2 results, render figures, write report.md.

The Decision section is built dynamically from measured values: cond_A,
cond_B, cond_C are evaluated against thresholds and the recommendation
fires from those evaluations. No hardcoded recommendation.
"""

import json
import os.path as osp
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnosis.measurements import DISTANCE_BIN_LABELS
from diagnosis_tier2.measurements_tier2 import SIX_CAMERAS


# Decision thresholds (single source of truth — referenced in report).
# cond_C tightened from 0.15 (Tier 2 mini) → 0.20 (Tier 2 Extended) because n=100
# unique trainval samples have smaller SE than n=20 mini, so a stricter STRONG bar
# is justified. WEAK bar moved 0.05 → 0.10 to make the MARGINAL band [0.10, 0.20)
# explicit and named ('보류').
COND_A_THRESHOLD = 0.30  # n_multi_view_gts / n_gt_lidar_supported ≥ 0.30 → consistency strong-A
COND_A_WEAK     = 0.15
COND_B_THRESHOLD = 1.5   # median(max_pair_distance) ≤ 1.5m → consistency strong-B
COND_B_WEAK     = 3.0
COND_C_THRESHOLD = 0.20  # std(entropy_norm) ≥ 0.20 → uniformity STRONG (확정)
COND_C_WEAK     = 0.10   # std(entropy_norm) < 0.10 → uniformity WEAK (기각)


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _safe_median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _safe_std(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.std(xs)) if xs else None


def _flatten_view_diversity(samples):
    return [vd for s in samples for vd in s["view_diversity"]["per_gt"]]


def _flatten_multi_view(samples):
    return [m for s in samples for m in s["multi_view_consistency"]["per_gt"]]


def _flatten_uniformity(samples):
    return [u for s in samples for u in s["uniformity"]["per_detection"]]


def _load_mini_baseline(out_dir):
    """Load the mini Tier 2 (n=20) aggregate for comparison, only if it exists at
    a *different* path than the current run's output. Returns dict-of-floats or None.
    """
    candidate = osp.join("results", "diagnosis_tier2", "aggregate.json")
    if not osp.exists(candidate):
        return None
    if osp.abspath(candidate) == osp.abspath(osp.join(out_dir, "aggregate.json")):
        # We're running on mini itself — no self-comparison.
        return None
    try:
        with open(candidate) as f:
            m = json.load(f)
        return {
            "path": candidate,
            "n_samples": m.get("n_samples"),
            "cond_A": m.get("M2", {}).get("fraction_multi_view"),
            "cond_B": m.get("M2", {}).get("max_pair_distance_median"),
            "cond_C": m.get("M3", {}).get("entropy_norm_std"),
            "M1_mean_geom": m.get("M1", {}).get("mean_n_geom_visible"),
            "M1_mean_det": m.get("M1", {}).get("mean_n_det_visible"),
            "M1_geom_minus_det": m.get("M1", {}).get("mean_geom_minus_det"),
        }
    except Exception:
        return None


def aggregate_tier2(samples, provenance=None, mini_baseline=None):
    vds = _flatten_view_diversity(samples)
    mvs = _flatten_multi_view(samples)
    unis = _flatten_uniformity(samples)

    # M1
    hist_geom = [0] * 7
    hist_det = [0] * 7
    for vd in vds:
        hist_geom[vd["n_geom_visible"]] += 1
        hist_det[vd["n_det_visible"]] += 1
    geom_minus_det = [vd["n_geom_visible"] - vd["n_det_visible"] for vd in vds]

    geom_by_bin = {b: [] for b in DISTANCE_BIN_LABELS}
    det_by_bin = {b: [] for b in DISTANCE_BIN_LABELS}
    for vd in vds:
        b = vd["distance_bin"]
        if b in geom_by_bin:
            geom_by_bin[b].append(vd["n_geom_visible"])
            det_by_bin[b].append(vd["n_det_visible"])

    # M2
    max_pair = [m["max_pair_distance_m"] for m in mvs]
    mean_pair = [m["mean_pair_distance_m"] for m in mvs]
    max_pair_by_bin = {b: [] for b in DISTANCE_BIN_LABELS}
    for m in mvs:
        b = m["distance_bin"]
        if b in max_pair_by_bin:
            max_pair_by_bin[b].append(m["max_pair_distance_m"])

    # M3
    H_norms = [u["entropy_norm"] for u in unis]
    H_by_bin = {b: [] for b in DISTANCE_BIN_LABELS}
    H_by_class = {}
    for u in unis:
        b = u["distance_bin"]
        if b in H_by_bin:
            H_by_bin[b].append(u["entropy_norm"])
        c = u["class"]
        H_by_class.setdefault(c, []).append(u["entropy_norm"])

    n_lidar_supp_total = sum(s["n_gt_lidar_supported"] for s in samples)
    n_mv_total = sum(s["multi_view_consistency"]["n_multi_view_gts"] for s in samples)

    # Pre-compute current cond_C verdict using new thresholds for the mini-comparison field.
    cond_C_now = _safe_std(H_norms)
    if cond_C_now is None:
        verdict_now = "UNDEFINED"
    elif cond_C_now >= COND_C_THRESHOLD:
        verdict_now = "STRONG"
    elif cond_C_now < COND_C_WEAK:
        verdict_now = "WEAK"
    else:
        verdict_now = "MARGINAL"

    extra = {}
    if provenance is not None:
        extra["data_source"] = provenance.get("version", "unknown")
        extra["sample_pool_size"] = provenance.get("n_pool_total")
        extra["n_pool_after_file_check"] = provenance.get("n_pool_after_file_check")
    if mini_baseline is not None and mini_baseline.get("cond_C") is not None:
        # Re-evaluate mini cond_C verdict under the *current* thresholds for fair comparison.
        m_c = mini_baseline["cond_C"]
        if m_c >= COND_C_THRESHOLD:
            mini_verdict = "STRONG"
        elif m_c < COND_C_WEAK:
            mini_verdict = "WEAK"
        else:
            mini_verdict = "MARGINAL"
        extra["comparison_to_mini_n20"] = {
            "mini_path": mini_baseline.get("path"),
            "mini_n_samples": mini_baseline.get("n_samples"),
            "mini_cond_A": mini_baseline.get("cond_A"),
            "mini_cond_B": mini_baseline.get("cond_B"),
            "mini_cond_C": m_c,
            "mini_cond_C_verdict_under_current_thresholds": mini_verdict,
            "current_cond_C": cond_C_now,
            "current_cond_C_verdict": verdict_now,
            "delta_cond_A": (agg_fraction := (n_mv_total / n_lidar_supp_total) if n_lidar_supp_total else 0.0) - (mini_baseline.get("cond_A") or 0.0),
            "delta_cond_C": (cond_C_now - m_c) if cond_C_now is not None else None,
        }

    return {
        **extra,
        "n_samples": len(samples),
        "M1": {
            "histogram_geom_total": hist_geom,
            "histogram_det_total": hist_det,
            "n_lidar_supp_gts": n_lidar_supp_total,
            "mean_n_geom_visible": _safe_mean([vd["n_geom_visible"] for vd in vds]),
            "median_n_geom_visible": _safe_median([vd["n_geom_visible"] for vd in vds]),
            "mean_n_det_visible": _safe_mean([vd["n_det_visible"] for vd in vds]),
            "median_n_det_visible": _safe_median([vd["n_det_visible"] for vd in vds]),
            "mean_geom_minus_det": _safe_mean(geom_minus_det),
            "geom_by_bin_mean": {b: _safe_mean(v) for b, v in geom_by_bin.items()},
            "det_by_bin_mean":  {b: _safe_mean(v) for b, v in det_by_bin.items()},
            "geom_by_bin_count": {b: len(v) for b, v in geom_by_bin.items()},
        },
        "M2": {
            "n_multi_view_gts": n_mv_total,
            "fraction_multi_view": (n_mv_total / n_lidar_supp_total) if n_lidar_supp_total else 0.0,
            "max_pair_distance_mean": _safe_mean(max_pair),
            "max_pair_distance_median": _safe_median(max_pair),
            "max_pair_distance_p95": float(np.percentile(max_pair, 95)) if max_pair else None,
            "max_pair_distance_max": float(max(max_pair)) if max_pair else None,
            "mean_pair_distance_mean": _safe_mean(mean_pair),
            "max_pair_by_bin_median": {b: _safe_median(v) for b, v in max_pair_by_bin.items()},
            "max_pair_by_bin_count": {b: len(v) for b, v in max_pair_by_bin.items()},
        },
        "M3": {
            "n_eligible": len(unis),
            "entropy_norm_mean": _safe_mean(H_norms),
            "entropy_norm_median": _safe_median(H_norms),
            "entropy_norm_std": _safe_std(H_norms),
            "entropy_by_bin_mean": {b: _safe_mean(v) for b, v in H_by_bin.items()},
            "entropy_by_bin_count": {b: len(v) for b, v in H_by_bin.items()},
            "entropy_by_class_mean": {c: _safe_mean(v) for c, v in H_by_class.items()},
            "entropy_by_class_count": {c: len(v) for c, v in H_by_class.items()},
        },
    }


# ------------- figures -------------

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_view_diversity_histogram(agg, out):
    geom = agg["M1"]["histogram_geom_total"]
    det = agg["M1"]["histogram_det_total"]
    x = np.arange(7)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.4
    ax.bar(x - width / 2, geom, width, label="geom_visible", color="#4477AA")
    ax.bar(x + width / 2, det, width, label="det_visible", color="#CC6677")
    ax.set_xticks(x)
    ax.set_xlabel("# cameras")
    ax.set_ylabel("# LiDAR-supported GT boxes")
    ax.set_title("View diversity per GT — geometric vs detector")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_view_diversity_by_distance(samples, out):
    bins = DISTANCE_BIN_LABELS
    bucket = {b: [0] * 7 for b in bins}
    for s in samples:
        for vd in s["view_diversity"]["per_gt"]:
            b = vd["distance_bin"]
            if b in bucket:
                bucket[b][vd["n_geom_visible"]] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(bins))
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, 7))
    for k in range(7):
        vals = np.array([bucket[b][k] for b in bins])
        ax.bar(bins, vals, bottom=bottom, color=colors[k], label=f"{k} cams")
        bottom += vals
    ax.set_xlabel("distance bin")
    ax.set_ylabel("# LiDAR-supported GT boxes")
    ax.set_title("View diversity (n_geom_visible) by distance bin")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    _save(fig, out)


def fig_detector_geom_gap(samples, out):
    gaps = []
    for s in samples:
        for vd in s["view_diversity"]["per_gt"]:
            gaps.append(vd["n_geom_visible"] - vd["n_det_visible"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if gaps:
        bins = np.arange(min(gaps), max(gaps) + 2) - 0.5
        ax.hist(gaps, bins=bins, color="#882255")
    ax.set_xlabel("n_geom_visible − n_det_visible (cams)")
    ax.set_ylabel("# GT boxes")
    ax.set_title("Detector vs geometry gap per GT (positive = detector misses cams that geometry sees)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def fig_consistency_distribution(agg, samples, out):
    mvs = _flatten_multi_view(samples)
    vals = [m["max_pair_distance_m"] for m in mvs]
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals:
        clip = float(np.percentile(vals, 99)) + 0.5
        ax.hist(np.clip(vals, 0, clip), bins=30, color="#117733")
        med = agg["M2"]["max_pair_distance_median"]
        if med is not None:
            ax.axvline(med, color="red", ls="--", label=f"median={med:.2f}m")
            ax.legend()
    ax.set_xlabel("max pairwise centroid distance (m)")
    ax.set_ylabel("# GT boxes (multi-view)")
    ax.set_title(f"M2 multi-view consistency (n={len(vals)})")
    _save(fig, out)


def fig_consistency_vs_distance(samples, out):
    mvs = _flatten_multi_view(samples)
    xs = [m["distance_m"] for m in mvs]
    ys = [m["max_pair_distance_m"] for m in mvs]
    fig, ax = plt.subplots(figsize=(7, 5))
    if xs:
        ax.scatter(xs, ys, alpha=0.5, s=14, color="#332288")
    ax.set_xlabel("GT distance to ego (m)")
    ax.set_ylabel("max pair centroid distance (m)")
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_title("M2 consistency vs distance")
    ax.grid(alpha=0.3)
    _save(fig, out)


def fig_uniformity_distribution(samples, out):
    unis = _flatten_uniformity(samples)
    vals = [u["entropy_norm"] for u in unis]
    fig, ax = plt.subplots(figsize=(7, 4))
    if vals:
        ax.hist(vals, bins=30, color="#DDCC77", range=(0.0, 1.0))
        med = float(np.median(vals))
        ax.axvline(med, color="red", ls="--", label=f"median={med:.2f}")
        ax.legend()
    ax.set_xlabel("normalized 4-quadrant entropy (1 = uniform)")
    ax.set_ylabel("# detection boxes")
    ax.set_title(f"M3 per-box uniformity (n={len(vals)}, n_total≥4)")
    _save(fig, out)


def fig_uniformity_vs_count(samples, out):
    unis = _flatten_uniformity(samples)
    xs = [u["n_total"] for u in unis]
    ys = [u["entropy_norm"] for u in unis]
    fig, ax = plt.subplots(figsize=(7, 5))
    if xs:
        ax.scatter(xs, ys, alpha=0.5, s=14, color="#999933")
    ax.axvline(4, color="red", ls="--", lw=1, label="n=4 cutoff")
    ax.legend()
    ax.set_xlabel("# in-box LiDAR points")
    ax.set_ylabel("entropy_norm")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("M3 uniformity vs in-box LiDAR count")
    ax.grid(alpha=0.3)
    _save(fig, out)


# ------------- decision section -------------

def _evaluate_decision(agg):
    """Evaluate cond_A/B/C from measured values; emit branch number + recommendation.

    Branch fire is driven by cond_C (Uniformity). Consistency (cond_A/B) is
    reported and used to colour the rationale, but does not change the branch.
    """
    M2 = agg["M2"]
    M3 = agg["M3"]

    cond_A_value = M2["fraction_multi_view"]
    cond_A_pass = cond_A_value is not None and cond_A_value >= COND_A_THRESHOLD
    cond_A_weak = cond_A_value is not None and cond_A_value < COND_A_WEAK

    cond_B_value = M2["max_pair_distance_median"]
    cond_B_pass = cond_B_value is not None and cond_B_value <= COND_B_THRESHOLD
    cond_B_weak = cond_B_value is not None and cond_B_value > COND_B_WEAK

    cond_C_value = M3["entropy_norm_std"]
    cond_C_pass = cond_C_value is not None and cond_C_value >= COND_C_THRESHOLD
    cond_C_weak = cond_C_value is not None and cond_C_value < COND_C_WEAK

    # Branch fire — Uniformity-driven
    if cond_C_value is None:
        branch_number = 0
        branch_label = "UNDEFINED"
        branch_text = ("cond_C could not be computed (no eligible detections). "
                       "Cannot evaluate Uniformity branch — investigate sample pool.")
    elif cond_C_pass:
        branch_number = 1
        branch_label = "STRONG (Uniformity 확정)"
        branch_text = ("Uniformity = primary component #2 확정. "
                       "B(Tier 3) 진행하되 Uniformity는 ablation/보강 자료로 격하.")
    elif cond_C_weak:
        branch_number = 2
        branch_label = "WEAK (Uniformity 기각)"
        branch_text = ("Uniformity 기각. "
                       "B(Tier 3)에서 temporal/occupancy를 본격적 #2 후보로 발굴.")
    else:
        branch_number = 3
        branch_label = "MARGINAL (Uniformity 보류)"
        branch_text = ("Uniformity 보류. "
                       "B(Tier 3) 결과 강하면 그쪽 채택, B도 weak이면 그때 Uniformity로 fallback.")

    consistency_strong = bool(cond_A_pass and cond_B_pass)
    consistency_weak = bool(cond_A_weak or cond_B_weak)
    uniformity_strong = bool(cond_C_pass)
    uniformity_weak = bool(cond_C_weak)

    return {
        "cond_A": {
            "name": "n_multi_view_gts / n_gt_lidar_supported",
            "value": cond_A_value,
            "threshold_strong": COND_A_THRESHOLD,
            "threshold_weak": COND_A_WEAK,
            "verdict": "PASS" if cond_A_pass else ("WEAK" if cond_A_weak else "MARGINAL"),
        },
        "cond_B": {
            "name": "median(max_pair_distance_m)",
            "value": cond_B_value,
            "threshold_strong": COND_B_THRESHOLD,
            "threshold_weak": COND_B_WEAK,
            "verdict": "PASS" if cond_B_pass else ("WEAK" if cond_B_weak else "MARGINAL"),
        },
        "cond_C": {
            "name": "std(entropy_norm)",
            "value": cond_C_value,
            "threshold_strong": COND_C_THRESHOLD,
            "threshold_weak": COND_C_WEAK,
            "verdict": "STRONG" if cond_C_pass else ("WEAK" if cond_C_weak else "MARGINAL"),
        },
        "consistency_strong": consistency_strong,
        "consistency_weak": consistency_weak,
        "uniformity_strong": uniformity_strong,
        "uniformity_weak": uniformity_weak,
        "branch_number": branch_number,
        "branch_label": branch_label,
        "branch_text": branch_text,
        "recommendation": branch_label,  # legacy alias for any consumer
        "rationale": branch_text,
    }


def _surprising_tier2(agg, samples, decision, mini_baseline=None):
    """Top-3 surprising findings, scored by deviation. When ``mini_baseline`` is
    provided, candidates that show a *change vs mini* score by the magnitude of
    that change and are preferred over absolute-deviation candidates.
    """
    cands = []

    # Mini-comparison candidates fire only when we have baseline numbers.
    if mini_baseline is not None:
        # cond_C movement
        if mini_baseline.get("cond_C") is not None and agg["M3"]["entropy_norm_std"] is not None:
            delta_C = agg["M3"]["entropy_norm_std"] - mini_baseline["cond_C"]
            if abs(delta_C) >= 0.02:
                direction = "rose" if delta_C > 0 else "fell"
                cands.append({
                    "deviation": 5.0 + abs(delta_C) * 100,  # outrank absolute candidates when change is real
                    "observation": (
                        f"cond_C (std of entropy_norm) {direction} from mini's "
                        f"{mini_baseline['cond_C']:.4f} (n={mini_baseline.get('n_samples')}) to "
                        f"{agg['M3']['entropy_norm_std']:.4f} on n={agg['n_samples']} trainval samples "
                        f"(Δ={delta_C:+.4f})."
                    ),
                    "why_surprising": (
                        "Mini Tier 2 cond_C landed at 0.1496, just below the original 0.15 STRONG bar; "
                        "we expected the trainval n=100 number to either confirm Uniformity (≥0.20 with "
                        "the new tightened threshold) or collapse (<0.10). The actual move clarifies "
                        "the verdict — direction and magnitude both matter for the branch fire."
                    ),
                    "method_implication": (
                        "Branch fire downstream of cond_C is now data-driven rather than borderline; "
                        "downstream Tier 3 plan can commit instead of hedging."
                    ),
                })
        # cond_A movement
        if mini_baseline.get("cond_A") is not None and agg["M2"]["fraction_multi_view"] is not None:
            delta_A = agg["M2"]["fraction_multi_view"] - mini_baseline["cond_A"]
            if abs(delta_A) >= 0.05:
                cands.append({
                    "deviation": 4.0 + abs(delta_A) * 10,
                    "observation": (
                        f"Multi-view fraction (cond_A) moved from mini's {mini_baseline['cond_A']*100:.1f}% to "
                        f"{agg['M2']['fraction_multi_view']*100:.1f}% on trainval n={agg['n_samples']} "
                        f"(Δ={delta_A*100:+.1f} pts)."
                    ),
                    "why_surprising": (
                        "Mini suggested multi-view was a minority phenomenon (6.4%). A meaningful shift "
                        "on the larger pool changes whether Consistency could ever be a primary signal."
                    ),
                    "method_implication": (
                        f"Reassess Consistency's role: at {agg['M2']['fraction_multi_view']*100:.1f}% it "
                        "is " + ("usable as a primary signal" if agg['M2']['fraction_multi_view'] >= 0.30
                                 else "still minority but worth re-checking thresholds")
                    ),
                })
        # geom-det gap movement
        if mini_baseline.get("M1_geom_minus_det") is not None and agg["M1"]["mean_geom_minus_det"] is not None:
            delta_g = agg["M1"]["mean_geom_minus_det"] - mini_baseline["M1_geom_minus_det"]
            if abs(delta_g) >= 0.10:
                cands.append({
                    "deviation": 3.0 + abs(delta_g) * 5,
                    "observation": (
                        f"Mean (n_geom − n_det) gap moved from mini's "
                        f"{mini_baseline['M1_geom_minus_det']:.2f} to "
                        f"{agg['M1']['mean_geom_minus_det']:.2f} cams "
                        f"(Δ={delta_g:+.2f}) on trainval n={agg['n_samples']}."
                    ),
                    "why_surprising": (
                        "The detector-recall hole is one of the few absolute numbers that should be "
                        "stable across mini and trainval if YOLO-World behaves consistently. Movement "
                        "here means either trainval has different scene-content distribution from mini, "
                        "or the gap is more sensitive to scene content than expected."
                    ),
                    "method_implication": (
                        "If trainval gap is larger, detector replacement/fine-tune leverage grows; "
                        "if smaller, mini was a worst-case sample of the detector failure mode."
                    ),
                })

    # 1) Detector vs geometry gap
    gap = agg["M1"]["mean_geom_minus_det"]
    if gap is not None and gap > 0.5:
        cands.append({
            "deviation": gap,
            "observation": (
                f"On average each LiDAR-supported GT is geometrically visible from "
                f"{agg['M1']['mean_n_geom_visible']:.2f} cameras but YOLO-World only detects it in "
                f"{agg['M1']['mean_n_det_visible']:.2f} of them — a gap of {gap:.2f} cameras per GT."
            ),
            "why_surprising": (
                "We knew the detector misses things, but we expected the detector to be roughly as "
                "sensitive as the geometric visibility test. Instead the gap is wide enough that "
                "the multi-cam setup is paying for views YOLO-World cannot exploit."
            ),
            "method_implication": (
                "Either fine-tune YOLO-World on driving-domain prompts/data, or backstop with a "
                "geometric proposal (Mask3D / clustering on LiDAR) and let 2D detection only "
                "*label* proposals rather than gate them."
            ),
        })

    # 2) Multi-view fraction (cond_A)
    frac = agg["M2"]["fraction_multi_view"]
    if frac is not None:
        if frac < 0.20:
            cands.append({
                "deviation": 0.5 - frac,
                "observation": (
                    f"Only {frac*100:.1f}% of LiDAR-supported GT boxes are geometrically visible from "
                    f"≥2 cameras ({agg['M2']['n_multi_view_gts']} of {agg['M1']['n_lidar_supp_gts']})."
                ),
                "why_surprising": (
                    "nuScenes has 6 cameras with overlapping FoVs; the prior expectation was that "
                    "most foreground objects would be seen by 2+. Instead, the per-cam FoV is so "
                    "narrow and the GT distribution so far-skewed that multi-view is the exception."
                ),
                "method_implication": (
                    "Multi-view consistency cannot be a primary loss term — it would only apply to "
                    "a minority of objects. Use it as a regulariser at best."
                ),
            })
        elif frac > 0.5:
            cands.append({
                "deviation": frac,
                "observation": (
                    f"{frac*100:.0f}% of LiDAR-supported GTs are seen from ≥2 cameras — "
                    f"multi-view coverage is ample."
                ),
                "why_surprising": (
                    "Naive expectation from 6 narrow-FoV cams + far-skewed objects was that the "
                    "multi-view fraction would be small. It isn't — overlapping FoV regions plus "
                    "the far range mean most objects do hit multiple cameras."
                ),
                "method_implication": (
                    "Multi-view consistency is a usable signal across most of the dataset, not a "
                    "minority-case regulariser."
                ),
            })

    # 3) Consistency tightness (cond_B)
    med = agg["M2"]["max_pair_distance_median"]
    if med is not None:
        if med <= 1.0:
            cands.append({
                "deviation": 5.0 - med,
                "observation": (
                    f"Median max-pair centroid distance is {med:.2f} m — different cameras place the "
                    f"same GT within ~1 m of each other in 3D."
                ),
                "why_surprising": (
                    "We expected sparse LiDAR within different cam FoVs to hit different surfaces of "
                    "the object (e.g. side vs back of a car), giving meters of disagreement. "
                    f"{med:.2f} m is tight enough to use as a hard constraint."
                ),
                "method_implication": (
                    "A loss term penalising large pairwise distances on multi-view GTs is tractable. "
                    "Consider <2 m as the operational threshold."
                ),
            })
        elif med > 3.0:
            cands.append({
                "deviation": med,
                "observation": (
                    f"Median max-pair centroid distance is {med:.2f} m — different cameras disagree "
                    f"on where the GT lives in 3D by meters."
                ),
                "why_surprising": (
                    "Multi-view nominally provides 3D triangulation; a wide spread means the in-box "
                    "LiDAR for each cam is hitting different surfaces, not the object centroid."
                ),
                "method_implication": (
                    "Centroid-based consistency is too noisy to use directly. Switch to volumetric "
                    "agreement (e.g. IoU of inflated boxes across views) or weight by visible-surface."
                ),
            })

    # 4) Uniformity discriminative power (cond_C)
    sd = agg["M3"]["entropy_norm_std"]
    mean_h = agg["M3"]["entropy_norm_mean"]
    if sd is not None and mean_h is not None:
        if sd < 0.10:
            cands.append({
                "deviation": 0.30 - sd,
                "observation": (
                    f"entropy_norm has std={sd:.3f} (mean={mean_h:.3f}) — barely any variance across "
                    f"detection boxes."
                ),
                "why_surprising": (
                    "The hope was that genuinely-on-object boxes would show non-uniform LiDAR "
                    "(clustered on the object surface) versus background-leaking boxes showing "
                    "near-uniform spread. The data shows the metric is nearly constant — it cannot "
                    "discriminate."
                ),
                "method_implication": (
                    "Uniformity is not a usable second-component signal. Drop it and look at "
                    "alternatives (depth-distribution test, Mask3D-2D-IoU agreement)."
                ),
            })

    # 5) GT distribution by visibility — extreme zero-cam fraction
    hist_geom = agg["M1"]["histogram_geom_total"]
    n_total = sum(hist_geom)
    if n_total > 0 and hist_geom[0] > 0:
        zero_frac = hist_geom[0] / n_total
        if zero_frac > 0.10:
            cands.append({
                "deviation": zero_frac,
                "observation": (
                    f"{zero_frac*100:.1f}% of LiDAR-supported GT boxes are seen by ZERO cameras "
                    f"(geometrically). I.e. they are inside the LiDAR sweep but no camera's FoV "
                    f"covers them sufficiently."
                ),
                "why_surprising": (
                    "If LiDAR has 360° coverage, naive expectation is every LiDAR-supported GT is "
                    "covered by at least one camera. But the camera FoVs leave non-trivial blind "
                    "regions, especially close-in laterals."
                ),
                "method_implication": (
                    "Some objects are unrecoverable by any vision-based method on this rig. "
                    "Evaluation should report a 'vision-attainable' upper bound separately from "
                    "the full-LiDAR oracle."
                ),
            })

    cands.sort(key=lambda c: -c["deviation"])
    return cands[:3]


def _write_report(samples, agg, decision, surprising, failed, figures_dir, out_path,
                  provenance=None, mini_baseline=None):
    lines = []
    lines.append("# nuScenes diagnosis — Tier 2 (multi-view + uniformity)")
    lines.append("")
    if provenance is not None:
        version = provenance.get("version", "unknown")
        if provenance.get("source") == "random":
            n_pool = provenance.get("n_pool_after_file_check", "?")
            n_total = provenance.get("n_pool_total", "?")
            seed = provenance.get("seed", "?")
            lines.append(f"**Source**: {version}, n={len(samples)} unique samples (seed={seed}, "
                         f"pool {n_pool}/{n_total} after file-existence filter)")
        else:
            lines.append(f"**Source**: {version}, n={len(samples)} samples (reused from {provenance.get('source')})")
        lines.append("")
    lines.append(f"- Samples succeeded: **{len(samples)}**")
    lines.append(f"- Samples failed: {len(failed)}")
    lines.append(f"- LiDAR-supported GT boxes (total): {agg['M1']['n_lidar_supp_gts']}")
    lines.append(f"- Multi-view (n_geom≥2) GT boxes: {agg['M2']['n_multi_view_gts']}")
    lines.append(f"- M3-eligible detections (n≥4 in-box pts): {agg['M3']['n_eligible']}")
    if mini_baseline is not None:
        lines.append(f"- Compared against mini baseline (n={mini_baseline.get('n_samples')}) — "
                     f"see Top-3 surprising findings for deltas.")
    lines.append("")

    lines.append("## Summary table")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| # samples | {agg['n_samples']} |")
    lines.append(f"| Mean n_geom_visible / GT | {agg['M1']['mean_n_geom_visible']:.2f} |")
    lines.append(f"| Mean n_det_visible / GT | {agg['M1']['mean_n_det_visible']:.2f} |")
    lines.append(f"| Mean (geom − det) gap | {agg['M1']['mean_geom_minus_det']:.2f} cams |")
    frac = agg['M2']['fraction_multi_view']
    lines.append(f"| % GT seen by ≥2 cams | {frac*100:.1f}% |")
    def _row(label, value, fmt, suffix=""):
        if value is None:
            return f"| {label} | n/a |"
        return f"| {label} | {format(value, fmt)}{suffix} |"
    lines.append(_row("Median max-pair centroid distance",
                      agg['M2']['max_pair_distance_median'], ".3f", " m"))
    lines.append(_row("p95 max-pair centroid distance",
                      agg['M2']['max_pair_distance_p95'], ".3f", " m"))
    lines.append(_row("Mean entropy_norm", agg['M3']['entropy_norm_mean'], ".4f"))
    lines.append(_row("Std entropy_norm",  agg['M3']['entropy_norm_std'], ".4f"))
    lines.append("")

    lines.append("### View-diversity histogram (LiDAR-supported GTs by # visible cams)")
    lines.append("")
    lines.append("| signal | 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| geom | " + " | ".join(str(x) for x in agg["M1"]["histogram_geom_total"]) + " |")
    lines.append("| det  | " + " | ".join(str(x) for x in agg["M1"]["histogram_det_total"]) + " |")
    lines.append("")

    lines.append("## M1 View diversity")
    lines.append("")
    lines.append("![view_diversity_histogram](figures/view_diversity_histogram.png)")
    lines.append("")
    lines.append("Geometric visibility (3D box center projects in-image AND ≥1 LiDAR point in projected 2D bbox) versus detector visibility (≥1 YOLO-World detection with IoU ≥ 0.3 against the projected GT 2D bbox). The gap between the two bars at any given # cams is the per-cam recall hole of the open-vocab detector.")
    lines.append("")
    lines.append("![view_diversity_by_distance](figures/view_diversity_by_distance.png)")
    lines.append("")
    lines.append("Stacked count of LiDAR-supported GTs per distance bin, segmented by n_geom_visible. Tells us whether multi-view coverage exists where the GTs actually live (most are >30 m).")
    lines.append("")
    lines.append("![detector_geom_gap](figures/detector_geom_gap.png)")
    lines.append("")
    lines.append("Histogram of (n_geom_visible − n_det_visible) per GT. This is the per-GT detector-coverage hole — large positive values mean the geometry says the object is visible in multiple cams but YOLO-World only catches it in one (or zero).")
    lines.append("")

    lines.append("## M2 Multi-view consistency")
    lines.append("")
    lines.append("![consistency_distribution](figures/consistency_distribution.png)")
    lines.append("")
    lines.append("Distribution of max pairwise centroid distance for GTs with ≥2 geom-visible cams. Each centroid is the median ego-frame xyz of in-bbox LiDAR points for that cam. The median (red dashed) is the value used for cond_B in the Decision section.")
    lines.append("")
    lines.append("![consistency_vs_distance](figures/consistency_vs_distance.png)")
    lines.append("")
    lines.append("Scatter of GT distance vs max-pair centroid distance. Symlog y-axis because some pairs disagree by tens of meters when in-bbox LiDAR hits different surfaces.")
    lines.append("")

    lines.append("## M3 Per-box uniformity")
    lines.append("")
    lines.append("![uniformity_distribution](figures/uniformity_distribution.png)")
    lines.append("")
    lines.append("Distribution of normalised 4-quadrant entropy across all 2D detection boxes with ≥4 in-box LiDAR points. 1.0 = uniform spread across quadrants, 0.0 = all points in one quadrant. cond_C measures std of this distribution — low std = no discriminative power.")
    lines.append("")
    lines.append("![uniformity_vs_count](figures/uniformity_vs_count.png)")
    lines.append("")
    lines.append("Scatter of in-box LiDAR count vs entropy_norm. As n grows the entropy converges (CLT-like); the question is whether the spread at moderate n is enough to distinguish clean-on-object boxes from background-bleeding ones.")
    lines.append("")

    # Key findings
    lines.append("## Key findings")
    lines.append("")
    bullets = []
    bullets.append(f"Of {agg['M1']['n_lidar_supp_gts']} LiDAR-supported GT boxes, "
                   f"{agg['M2']['n_multi_view_gts']} ({agg['M2']['fraction_multi_view']*100:.1f}%) "
                   f"are geometrically visible from ≥2 cameras.")
    bullets.append(f"Mean n_geom_visible per GT = {agg['M1']['mean_n_geom_visible']:.2f}; "
                   f"mean n_det_visible = {agg['M1']['mean_n_det_visible']:.2f}; "
                   f"gap = {agg['M1']['mean_geom_minus_det']:.2f} cams.")
    if agg['M2']['max_pair_distance_median'] is not None:
        bullets.append(f"Multi-view centroid agreement: median max-pair distance "
                       f"{agg['M2']['max_pair_distance_median']:.2f} m, "
                       f"mean {agg['M2']['max_pair_distance_mean']:.2f} m, "
                       f"p95 {agg['M2']['max_pair_distance_p95']:.2f} m.")
    if agg['M3']['entropy_norm_mean'] is not None:
        bullets.append(f"4-quadrant uniformity: mean {agg['M3']['entropy_norm_mean']:.3f}, "
                       f"median {agg['M3']['entropy_norm_median']:.3f}, "
                       f"std {agg['M3']['entropy_norm_std']:.3f} "
                       f"(over {agg['M3']['n_eligible']} eligible detections).")
    hist_geom = agg["M1"]["histogram_geom_total"]
    n_total = sum(hist_geom)
    if n_total > 0:
        bullets.append(f"Per-cam view distribution: 0 cams = {hist_geom[0]}, 1 cam = {hist_geom[1]}, "
                       f"2 cams = {hist_geom[2]}, 3+ cams = {sum(hist_geom[3:])} "
                       f"(of {n_total} LiDAR-supp GTs).")
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")

    # Decision section — dynamic
    lines.append("## Decision")
    lines.append("")
    lines.append("Thresholds and measured values are evaluated below; the recommendation fires from these checks (no hardcoded conclusion).")
    lines.append("")
    lines.append("### Consistency signal (M2)")
    lines.append("")
    a = decision["cond_A"]
    a_val_str = "n/a" if a["value"] is None else f"{a['value']:.3f}"
    lines.append(f"- **cond_A**: {a['name']} = {a_val_str}  → "
                 f"strong if ≥ {a['threshold_strong']}, weak if < {a['threshold_weak']}  → "
                 f"**{a['verdict']}**")
    b = decision["cond_B"]
    b_val_str = "n/a" if b["value"] is None else f"{b['value']:.3f} m"
    lines.append(f"- **cond_B**: {b['name']} = {b_val_str}  → "
                 f"strong if ≤ {b['threshold_strong']} m, weak if > {b['threshold_weak']} m  → "
                 f"**{b['verdict']}**")
    cs = "STRONG" if decision["consistency_strong"] else ("WEAK" if decision["consistency_weak"] else "MARGINAL")
    lines.append(f"- **Consistency verdict**: {cs}")
    lines.append("")
    lines.append("### Uniformity signal (M3)")
    lines.append("")
    c = decision["cond_C"]
    c_val_str = "n/a" if c["value"] is None else f"{c['value']:.4f}"
    lines.append(f"- **cond_C**: {c['name']} = {c_val_str}  → "
                 f"strong if ≥ {c['threshold_strong']}, weak if < {c['threshold_weak']}  → "
                 f"**{c['verdict']}**")
    us = "STRONG" if decision["uniformity_strong"] else ("WEAK" if decision["uniformity_weak"] else "MARGINAL")
    lines.append(f"- **Uniformity verdict**: {us}")
    lines.append("")
    lines.append("### Branch fire")
    lines.append("")
    bn = decision["branch_number"]
    bl = decision["branch_label"]
    lines.append(f"**→ 분기 {bn}: {bl}**")
    lines.append("")
    lines.append(decision["branch_text"])
    lines.append("")

    # Compact trace block (one block, no hardcoded labels)
    a, b, c = decision["cond_A"], decision["cond_B"], decision["cond_C"]
    lines.append("Trace:")
    lines.append("")
    lines.append("```")
    lines.append(f"cond_A = {a['value']:.4f}  → {a['verdict']:<8}  (strong ≥ {a['threshold_strong']}, weak < {a['threshold_weak']})"
                 if a['value'] is not None else f"cond_A = n/a")
    if b['value'] is not None:
        lines.append(f"cond_B = {b['value']:.4f}m → {b['verdict']:<8}  (strong ≤ {b['threshold_strong']}m, weak > {b['threshold_weak']}m)")
    else:
        lines.append("cond_B = n/a")
    if c['value'] is not None:
        lines.append(f"cond_C = {c['value']:.4f}  → {c['verdict']:<8}  (strong ≥ {c['threshold_strong']}, weak < {c['threshold_weak']})")
    else:
        lines.append("cond_C = n/a")
    lines.append(f"→ 분기 {bn}: {bl}")
    lines.append("```")
    lines.append("")

    # Top-3 surprising
    lines.append("## Top-3 most surprising findings")
    lines.append("")
    if not surprising:
        lines.append("Fewer than 3 surprising findings emerged — the data lined up with the priors set after Tier 1.")
    else:
        for i, s in enumerate(surprising, 1):
            lines.append(f"### {i}. {s['observation']}")
            lines.append("")
            lines.append(f"**Why surprising:** {s['why_surprising']}")
            lines.append("")
            lines.append(f"**Method implication:** {s['method_implication']}")
            lines.append("")
        if len(surprising) < 3:
            lines.append(f"_Only {len(surprising)} qualifying surprising finding(s) — not padding._")
            lines.append("")

    # Failed samples
    lines.append("## Failed samples")
    lines.append("")
    if not failed:
        lines.append("None.")
    else:
        lines.append(f"Count: {len(failed)}")
        lines.append("")
        for fr in failed:
            wall = fr.get("wall_seconds")
            wall_s = f" (wall={wall:.1f}s)" if wall is not None else ""
            lines.append(f"- `{fr['sample_token']}` — {fr.get('reason', 'unknown')}{wall_s}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def render_all_tier2(samples, agg, failed, figures_dir, report_path,
                     provenance=None, mini_baseline=None):
    fig_view_diversity_histogram(agg, osp.join(figures_dir, "view_diversity_histogram.png"))
    fig_view_diversity_by_distance(samples, osp.join(figures_dir, "view_diversity_by_distance.png"))
    fig_detector_geom_gap(samples, osp.join(figures_dir, "detector_geom_gap.png"))
    fig_consistency_distribution(agg, samples, osp.join(figures_dir, "consistency_distribution.png"))
    fig_consistency_vs_distance(samples, osp.join(figures_dir, "consistency_vs_distance.png"))
    fig_uniformity_distribution(samples, osp.join(figures_dir, "uniformity_distribution.png"))
    fig_uniformity_vs_count(samples, osp.join(figures_dir, "uniformity_vs_count.png"))
    decision = _evaluate_decision(agg)
    surprising = _surprising_tier2(agg, samples, decision, mini_baseline=mini_baseline)
    _write_report(samples, agg, decision, surprising, failed, figures_dir, report_path,
                  provenance=provenance, mini_baseline=mini_baseline)
