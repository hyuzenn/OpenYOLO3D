"""E1 preregistered statistics — run ONCE after all 21 variants complete.

Implements results/2026-07-18_e1_prereg_v01/PREREGISTRATION.md §4-§5:
  1. Spearman/Pearson/Kendall for {OVTCS_C, L_norm, 1-CSR, A, B} x
     {AssA, HOTA, IDF1, AMOTA, DetA}, with 95% scene-bootstrap CIs
     (10,000 resamples, seed 20260718). AMOTA: point estimate only
     (official devkit eval is not scene-decomposable; noted as amendment).
  2. Steiger/Williams one-sided test: rho(OVTCS, AssA) > rho(L_norm, AssA).
  3. Partial Spearman rho(OVTCS, target | L_norm) + nested OLS dR2 F-test.
  4. Top-1/3/5 agreement vs HOTA and AssA + 10k-permutation p; Bottom-5 supp.
  5. Disagreement pairs: discordant under OVTCS vs HOTA with both bootstrap
     difference-CIs excluding 0 -> 2x2 attribution (L_norm vs 1-CSR) x
     (DetA vs AssA).
  6. Robustness: headline Spearman recomputed on scenes 0-9.

Usage: python scripts/e1_stats.py --run-dir results/2026-07-18_e1_grid_v01
"""
from __future__ import annotations

import argparse
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as st

SEED = 20260718
N_BOOT = 10_000
VARIANTS = [
    "gamma_ego", "gamma_global", "detg_ego", "detg_global", "hybrid_ego",
    "hybrid_global", "gamma_ego_a2", "gamma_ego_a10", "gamma_global_a2",
    "gamma_global_a10", "gamma_ego_ca", "gamma_global_ca", "hybrid_ego_ca",
    "hybrid_global_ca", "gamma_ego_p1", "gamma_global_p1", "hybrid_ego_p1",
    "hybrid_global_p1", "gamma_ego_m32", "gamma_global_m32", "hybrid_global_fuse",
]
PREDICTORS = ["C", "L_norm", "one_minus_csr", "A", "B"]
TARGETS = ["AssA", "HOTA", "IDF1", "AMOTA", "DetA"]


# -- per-variant loading ----------------------------------------------------
def load_cell(cell: Path) -> dict:
    axis = sorted(cell.glob("axis_*/e1_metrics.json"))
    assert len(axis) == 1, f"{cell}: {axis}"
    m = json.loads(axis[0].read_text())
    with open(axis[0].parent / "e1_perscene.pkl", "rb") as f:
        ps = pickle.load(f)
    return {"summary": m, "perscene": ps}


def hota_scene_arrays(perscene: dict, scenes: list[int]) -> dict:
    """Stack per-scene HOTA/Identity fields needed for cheap recombination."""
    te = perscene["trackeval"]
    out = {}
    for f in ("HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr", "LocA"):
        out[f] = np.stack([np.asarray(te[s]["HOTA"][f], dtype=np.float64)
                           for s in scenes])            # (S, 19)
    for f in ("IDTP", "IDFN", "IDFP"):
        out[f] = np.asarray([float(np.sum(te[s]["Identity"][f])) for s in scenes])
    return out


def combine_hota(arr: dict, idx: np.ndarray) -> dict:
    """Replicates trackeval combine_sequences: sum counts, HOTA_TP-weighted
    AssA, then final fields. Verified against trackeval on the full set."""
    tp = arr["HOTA_TP"][idx].sum(0)
    fn = arr["HOTA_FN"][idx].sum(0)
    fp = arr["HOTA_FP"][idx].sum(0)
    w = arr["HOTA_TP"][idx]                              # (s, 19)
    wsum = np.maximum(tp, 1e-10)
    assa = (arr["AssA"][idx] * w).sum(0) / wsum
    deta = tp / np.maximum(tp + fn + fp, 1e-10)
    hota = np.sqrt(deta * assa)
    idtp, idfn, idfp = (arr[f][idx].sum() for f in ("IDTP", "IDFN", "IDFP"))
    idf1 = 2 * idtp / np.maximum(2 * idtp + idfn + idfp, 1e-10)
    return {"AssA": float(assa.mean()), "DetA": float(deta.mean()),
            "HOTA": float(hota.mean()), "IDF1": float(idf1)}


def gtfree_scene_arrays(perscene: dict, scenes: list[int]) -> dict:
    """Per-scene (sum, n) per predictor -> pooled means under scene resampling."""
    gf = perscene["gt_free"]
    out = {}
    for k in PREDICTORS:
        out[k + "_sum"] = np.asarray(
            [gf[s][k].sum() if s in gf else 0.0 for s in scenes])
        out[k + "_n"] = np.asarray(
            [gf[s][k].size if s in gf else 0 for s in scenes], dtype=np.float64)
    return out


def gtfree_pooled(arr: dict, idx: np.ndarray) -> dict:
    return {k: float(arr[k + "_sum"][idx].sum() / max(arr[k + "_n"][idx].sum(), 1))
            for k in PREDICTORS}


# -- statistics -------------------------------------------------------------
def williams_steiger(r12, r13, r23, n, one_sided=True):
    """Williams' t for H1: r12 > r13 (dependent, sharing variable 1)."""
    detR = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    rbar = (r12 + r13) / 2
    denom = 2 * ((n - 1) / (n - 3)) * detR + rbar**2 * (1 - r23) ** 3
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) / denom)
    p = st.t.sf(t, n - 3) if one_sided else 2 * st.t.sf(abs(t), n - 3)
    return float(t), float(p)


def partial_spearman(x, y, z):
    rx, ry, rz = (st.rankdata(a) for a in (x, y, z))
    rxy, rxz, ryz = (st.pearsonr(a, b)[0] for a, b in
                     ((rx, ry), (rx, rz), (ry, rz)))
    pr = (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))
    n = len(x)
    t = pr * np.sqrt((n - 3) / max(1 - pr**2, 1e-12))
    return float(pr), float(2 * st.t.sf(abs(t), n - 3))


def nested_dr2(x, y, z):
    """OLS y~z vs y~z+x: dR2 + F-test (df 1, n-3)."""
    n = len(y)
    Z1 = np.column_stack([np.ones(n), z])
    Z2 = np.column_stack([np.ones(n), z, x])
    r1 = y - Z1 @ np.linalg.lstsq(Z1, y, rcond=None)[0]
    r2 = y - Z2 @ np.linalg.lstsq(Z2, y, rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    dr2 = (r1 @ r1 - r2 @ r2) / sst
    F = (r1 @ r1 - r2 @ r2) / (r2 @ r2 / (n - 3))
    return float(dr2), float(F), float(st.f.sf(F, 1, n - 3))


def topk_stats(pred_vals, gt_vals, rng):
    n = len(pred_vals)
    po = np.argsort(-pred_vals)   # best first
    go = np.argsort(-gt_vals)
    obs = {
        "top1_in_gt_top3": int(po[0] in set(go[:3])),
        "top3_overlap": len(set(po[:3]) & set(go[:3])),
        "top5_overlap": len(set(po[:5]) & set(go[:5])),
        "bottom5_overlap": len(set(po[-5:]) & set(go[-5:])),
    }
    null = {k: [] for k in obs}
    for _ in range(N_BOOT):
        rp = rng.permutation(n)
        null["top1_in_gt_top3"].append(int(rp[0] in set(go[:3])))
        null["top3_overlap"].append(len(set(rp[:3]) & set(go[:3])))
        null["top5_overlap"].append(len(set(rp[:5]) & set(go[:5])))
        null["bottom5_overlap"].append(len(set(rp[-5:]) & set(go[-5:])))
    out = {}
    for k, v in obs.items():
        nl = np.asarray(null[k])
        out[k] = {"observed": v, "chance": float(nl.mean()),
                  "perm_p": float((np.sum(nl >= v) + 1) / (N_BOOT + 1))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    rng = np.random.default_rng(SEED)

    cells = {v: load_cell(run / "cells" / v) for v in VARIANTS}
    scene_sets = [sorted(c["perscene"]["trackeval"].keys()) for c in cells.values()]
    scenes = scene_sets[0]
    assert all(s == scenes for s in scene_sets), "variants disagree on scene set"
    S = len(scenes)

    # verify manual recombination against trackeval once per variant
    harr = {v: hota_scene_arrays(cells[v]["perscene"], scenes) for v in VARIANTS}
    garr = {v: gtfree_scene_arrays(cells[v]["perscene"], scenes) for v in VARIANTS}
    full = np.arange(S)
    for v in VARIANTS:
        ours = combine_hota(harr[v], full)
        ref = cells[v]["summary"]["gt_based"]
        for k in ("AssA", "HOTA", "DetA", "IDF1"):
            assert abs(ours[k] - ref[k]) < 1e-6, (v, k, ours[k], ref[k])

    # point table
    table = {}
    for v in VARIANTS:
        s = cells[v]["summary"]
        table[v] = {**{k: s["gt_free"][k] for k in PREDICTORS},
                    **{k: s["gt_based"].get(k.lower(), s["gt_based"].get(k))
                       for k in TARGETS},
                    "n_tracks": s["n_tracks"]}
    P = {k: np.asarray([table[v][k] for v in VARIANTS]) for k in PREDICTORS}
    T = {k: np.asarray([table[v][k] for v in VARIANTS], dtype=np.float64)
         for k in TARGETS}

    # 1. correlations + bootstrap CIs
    boot_idx = rng.integers(0, S, size=(N_BOOT, S))
    boot = {p: {t: [] for t in ("AssA", "HOTA", "IDF1", "DetA")} for p in PREDICTORS}
    boot_diff = {"HOTA": {}, "C": {}}   # for disagreement CIs (pairwise diffs)
    boot_gt = np.empty((N_BOOT, len(VARIANTS), 4))
    boot_pr = np.empty((N_BOOT, len(VARIANTS), len(PREDICTORS)))
    for b in range(N_BOOT):
        idx = boot_idx[b]
        for j, v in enumerate(VARIANTS):
            h = combine_hota(harr[v], idx)
            g = gtfree_pooled(garr[v], idx)
            boot_gt[b, j] = [h["AssA"], h["HOTA"], h["IDF1"], h["DetA"]]
            boot_pr[b, j] = [g[p] for p in PREDICTORS]
    gt_cols = {"AssA": 0, "HOTA": 1, "IDF1": 2, "DetA": 3}
    corr = {}
    for pi, p in enumerate(PREDICTORS):
        corr[p] = {}
        for t in TARGETS:
            x, y = P[p], T[t]
            row = {
                "spearman": float(st.spearmanr(x, y).statistic),
                "spearman_p": float(st.spearmanr(x, y).pvalue),
                "pearson": float(st.pearsonr(x, y)[0]),
                "kendall": float(st.kendalltau(x, y).statistic),
            }
            if t in gt_cols:
                bs = np.array([st.spearmanr(boot_pr[b, :, pi],
                                            boot_gt[b, :, gt_cols[t]]).statistic
                               for b in range(N_BOOT)])
                row["spearman_ci95"] = [float(np.nanpercentile(bs, 2.5)),
                                        float(np.nanpercentile(bs, 97.5))]
            corr[p][t] = row

    # 2. Steiger/Williams (Spearman ranks + Pearson), primary: AssA
    steiger = {}
    for t in TARGETS:
        y, x1, x2 = T[t], P["C"], P["L_norm"]
        ry, r1, r2 = st.rankdata(y), st.rankdata(x1), st.rankdata(x2)
        s_t, s_p = williams_steiger(st.pearsonr(r1, ry)[0], st.pearsonr(r2, ry)[0],
                                    st.pearsonr(r1, r2)[0], len(y))
        p_t, p_p = williams_steiger(st.pearsonr(x1, y)[0], st.pearsonr(x2, y)[0],
                                    st.pearsonr(x1, x2)[0], len(y))
        steiger[t] = {"spearman_t": s_t, "spearman_p_onesided": s_p,
                      "pearson_t": p_t, "pearson_p_onesided": p_p}

    # 3. partial + nested dR2
    partial = {}
    for t in TARGETS:
        pr, pp = partial_spearman(P["C"], T[t], P["L_norm"])
        dr2, F, fp = nested_dr2(P["C"], T[t], P["L_norm"])
        partial[t] = {"partial_spearman": pr, "p": pp,
                      "dR2": dr2, "F": F, "F_p": fp}

    # 4. top-k agreement
    topk = {f"OVTCS_vs_{t}": topk_stats(P["C"], T[t], rng) for t in ("HOTA", "AssA")}
    topk["Lnorm_vs_HOTA"] = topk_stats(P["L_norm"], T["HOTA"], rng)  # comparison arm

    # 5. disagreement pairs (OVTCS vs HOTA discordant, both diff-CIs exclude 0)
    disagreements = []
    for i, j in combinations(range(len(VARIANTS)), 2):
        dC, dH = P["C"][i] - P["C"][j], T["HOTA"][i] - T["HOTA"][j]
        if dC * dH >= 0:
            continue
        bC = boot_pr[:, i, 0] - boot_pr[:, j, 0]
        bH = boot_gt[:, i, 1] - boot_gt[:, j, 1]
        ciC = np.percentile(bC, [2.5, 97.5])
        ciH = np.percentile(bH, [2.5, 97.5])
        confident = (ciC[0] > 0 or ciC[1] < 0) and (ciH[0] > 0 or ciH[1] < 0)
        # 2x2 attribution
        mL = (P["L_norm"][i] + P["L_norm"][j]) / 2
        mS = (P["one_minus_csr"][i] + P["one_minus_csr"][j]) / 2
        contrib_L = (P["L_norm"][i] - P["L_norm"][j]) * mS
        contrib_S = (P["one_minus_csr"][i] - P["one_minus_csr"][j]) * mL
        dlogDet = np.log(T["DetA"][i] / T["DetA"][j])
        dlogAss = np.log(T["AssA"][i] / T["AssA"][j])
        disagreements.append({
            "pair": [VARIANTS[i], VARIANTS[j]],
            "dC": float(dC), "dHOTA": float(dH),
            "dC_ci95": ciC.tolist(), "dHOTA_ci95": ciH.tolist(),
            "confident": bool(confident),
            "ovtcs_driver": "L_norm" if abs(contrib_L) > abs(contrib_S) else "1-CSR",
            "hota_driver": "DetA" if abs(dlogDet) > abs(dlogAss) else "AssA",
            "contrib": {"L_norm": float(contrib_L), "one_minus_csr": float(contrib_S),
                        "dlogDetA": float(dlogDet), "dlogAssA": float(dlogAss)},
        })

    # 6. robustness: scenes 0-9
    sub = np.asarray([k for k in range(min(10, S))])
    rob = {}
    for t in ("AssA", "HOTA"):
        x = np.asarray([combine_hota(harr[v], sub)[t] for v in VARIANTS])
        c = np.asarray([gtfree_pooled(garr[v], sub)["C"] for v in VARIANTS])
        rob[t] = float(st.spearmanr(c, x).statistic)

    out = {
        "n_variants": len(VARIANTS), "n_scenes": S,
        "seed": SEED, "n_boot": N_BOOT,
        "table": table, "correlations": corr, "steiger_williams": steiger,
        "partial_and_dR2": partial, "topk": topk,
        "disagreements": disagreements, "robustness_10scene_spearman": rob,
    }
    sd = run / "stats"
    sd.mkdir(exist_ok=True)
    (sd / "stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in
                      ("steiger_williams", "partial_and_dR2", "topk",
                       "robustness_10scene_spearman")}, indent=2))
    print(f"\nwrote {sd / 'stats.json'}  "
          f"({sum(1 for d in disagreements if d['confident'])} confident "
          f"disagreement pairs of {len(disagreements)} discordant)")


if __name__ == "__main__":
    main()
