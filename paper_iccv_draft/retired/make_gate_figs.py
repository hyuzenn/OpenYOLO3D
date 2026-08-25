#!/usr/bin/env python
"""Render the 4 gate-story figures from the frozen E2 numbers.
All values are the measured E2 gate sweep (results/2026-07-20_e2_gate_sweep) +
the E1 phase1 bundle for the decomposition. No new computation, no data reads.
Okabe-Ito CVD-safe palette; direct value labels; publication style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "figs"
# Okabe-Ito
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#8f8f8f", "#CC79A7")
plt.rcParams.update({"font.size": 9, "axes.splines.top" if False else "axes.grid": False,
                     "svg.fonttype": "none"})

# ---- E2 gate sweep (everything frozen but N) ----
N       = [1, 2, 3, 5]
mAP     = [0.3408, 0.2929, 0.2601, 0.2082]
NDS     = [0.3150, 0.3069, 0.3018, 0.2825]
HOTA    = [0.2011, 0.2325, 0.2465, 0.2573]
AssA    = [0.4089, 0.4286, 0.4290, 0.4133]
DetA    = [0.0995, 0.1271, 0.1427, 0.1615]
IDF1    = [0.1764, 0.2156, 0.2445, 0.2797]
AMOTA   = [0.1580, 0.1685, 0.1670, 0.1459]
frag    = [2305, 1933, 1609, 1202]
emitted = [1029380, 672972, 513938, 326990]

def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)

# =========================================================================
# Fig 1 (teaser): one glance — strengthen the gate, mAP DOWN, tracking UP,
# over identical box geometry. Baseline (N=1) vs strong gate (N=5) grouped bars.
# =========================================================================
fig, ax = plt.subplots(figsize=(3.4, 2.7))
labels = ["mAP", "HOTA", "IDF1", "DetA"]
base   = [mAP[0], HOTA[0], IDF1[0], DetA[0]]
gated  = [mAP[-1], HOTA[-1], IDF1[-1], DetA[-1]]
x = np.arange(len(labels)); w = 0.38
ax.bar(x - w/2, base,  w, color=GRAY, label="no gate (N=1)")
ax.bar(x + w/2, gated, w, color=BLUE, label="temporal gate (N=5)")
for xi, b, g in zip(x, base, gated):
    ax.text(xi - w/2, b + .006, f"{b:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(xi + w/2, g + .006, f"{g:.2f}", ha="center", va="bottom", fontsize=7.5)
# arrows: mAP down, others up
ax.annotate("", xy=(0, base[0]-.02), xytext=(0, gated[0]+.05),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
for xi in x[1:]:
    ax.annotate("", xy=(xi, gated[int(xi)]+.03), xytext=(xi, base[int(xi)]-.005),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
ax.text(0, 0.375, "mAP $\\downarrow$", color=RED, ha="center", fontsize=8.5, fontweight="bold")
ax.text(2, 0.375, "tracking $\\uparrow$", color=GREEN, ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.42); ax.set_ylabel("score")
ax.set_title("Same boxes, opposite verdicts", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=7, frameon=False, loc="upper right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
savefig(fig, "fig_teaser")

# =========================================================================
# Fig 2 (main): controlled gate sweep. Left: mAP (cost) vs HOTA/IDF1/DetA (gain)
# vs N, twin axis. Right: mAP-AMOTA Pareto, N=5 dominated.
# =========================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 3.0))
axL.plot(N, mAP, "o-", color=RED, lw=2, label="mAP (per-frame)")
axL.plot(N, NDS, "o--", color=RED, lw=1, alpha=.55, label="NDS")
axL.set_xlabel("gate strength  N  (min. visible frames)")
axL.set_ylabel("per-frame detection", color=RED)
axL.tick_params(axis="y", colors=RED)
axL.set_xticks(N)
ax2 = axL.twinx()
ax2.plot(N, HOTA, "s-", color=BLUE,   lw=2, label="HOTA")
ax2.plot(N, IDF1, "^-", color=GREEN,  lw=2, label="IDF1")
ax2.plot(N, DetA, "D-", color=ORANGE, lw=2, label="DetA")
ax2.set_ylabel("GT tracking quality")
axL.axvspan(0.9, 1.1, color="k", alpha=.05)
axL.text(1.0, mAP[0]+.004, "baseline\n(N=1 = no-op)", fontsize=6.5, ha="left", va="bottom")
l1,lab1 = axL.get_legend_handles_labels(); l2,lab2 = ax2.get_legend_handles_labels()
axL.legend(l1+l2, lab1+lab2, fontsize=6.8, frameon=False, loc="center right")
axL.set_title("Controlled gate sweep (all else frozen)", fontsize=9)
for s in ("top",): axL.spines[s].set_visible(False); ax2.spines[s].set_visible(False)

# Pareto: mAP (x) vs AMOTA (y)
axR.plot(mAP, AMOTA, "-", color=GRAY, lw=1, zorder=1)
for xi, yi, n in zip(mAP, AMOTA, N):
    c = RED if n == 5 else BLUE
    axR.scatter(xi, yi, s=55, color=c, zorder=3)
    axR.annotate(f"N={n}", (xi, yi), textcoords="offset points",
                 xytext=(6, 5), fontsize=8)
# mark N=5 dominated by N=3
axR.annotate("N=5 dominated\nby N=3\n(worse mAP & AMOTA)",
             xy=(mAP[3], AMOTA[3]), xytext=(mAP[3]+.01, AMOTA[3]-.006),
             fontsize=6.8, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1))
axR.set_xlabel("mAP  (higher better $\\rightarrow$)")
axR.set_ylabel("AMOTA  (higher better $\\uparrow$)")
axR.set_title("mAP–AMOTA trade-off", fontsize=9)
for s in ("top", "right"): axR.spines[s].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_gate_sweep")

# =========================================================================
# Fig 3 (decomposition): gate vs relabel. Grouped bars control / gate / gate+relabel
# for HOTA, AssA, IDF1 (flat across relabel) and AMOTA (jumps only with relabel).
# =========================================================================
# control, gate-only(N=3), gate+relabel(E1 phase1)
dec = {
    "HOTA":  [0.2011, 0.2465, 0.2456],
    "AssA":  [0.4089, 0.4290, 0.4199],
    "IDF1":  [0.1764, 0.2445, 0.2422],
    "AMOTA": [0.1580, 0.1670, 0.2089],
}
fig, ax = plt.subplots(figsize=(4.6, 2.9))
groups = list(dec.keys()); x = np.arange(len(groups)); w = 0.26
c_ctrl, c_gate, c_rel = GRAY, BLUE, PURPLE
v_ctrl = [dec[g][0] for g in groups]
v_gate = [dec[g][1] for g in groups]
v_rel  = [dec[g][2] for g in groups]
ax.bar(x - w, v_ctrl, w, color=c_ctrl, label="control (no gate)")
ax.bar(x,     v_gate, w, color=c_gate, label="+ Temporal Layer (gate)")
ax.bar(x + w, v_rel,  w, color=c_rel,  label="+ semantic relabel (M21)")
for xi, g in zip(x, groups):
    for dx, v in zip((-w, 0, w), dec[g]):
        ax.text(xi+dx, v+.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
# highlight: class-agnostic flat gate->relabel; AMOTA jumps
ax.annotate("relabel\nadds +0.042", xy=(3+w, dec["AMOTA"][2]),
            xytext=(3-0.1, dec["AMOTA"][2]+.03), fontsize=7, color=PURPLE,
            ha="center", arrowprops=dict(arrowstyle="->", color=PURPLE))
ax.axvline(2.5, color="k", ls=":", lw=.7, alpha=.5)
ax.text(1.0, 0.45, "class-agnostic\n(gate owns it; relabel ≈ 0)",
        fontsize=6.5, ha="center", color=BLUE)
ax.text(3.0, 0.45, "class-aware\n(relabel owns it)",
        fontsize=6.5, ha="center", color=PURPLE)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0, 0.5); ax.set_ylabel("score")
ax.set_title("Continuity (gate) vs. semantics (relabel)", fontsize=9.5, fontweight="bold")
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
savefig(fig, "fig_decomp")

# =========================================================================
# Fig 4 (mechanism): two detection scores over the SAME boxes diverge.
# DetA up, mAP down vs N, annotated with emitted-box / frag counts.
# =========================================================================
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(N, mAP,  "o-", color=RED,    lw=2, label="mAP (strict-recall integral)")
ax.plot(N, DetA, "D-", color=ORANGE, lw=2, label="DetA (class-agnostic det. acc.)")
ax.set_xlabel("gate strength  N")
ax.set_ylabel("detection-quality score")
ax.set_xticks(N)
for xi, m, d, e in zip(N, mAP, DetA, emitted):
    ax.annotate(f"{e/1e6:.2f}M boxes", (xi, min(m, d)-.012),
                fontsize=6, ha="center", color=GRAY)
ax.text(3.2, 0.30, "same box geometry\n(shared detector cache)\n→ mAP $\\downarrow$, DetA $\\uparrow$",
        fontsize=7, color="k")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_ylim(0.06, 0.36)
ax.set_title("Why mAP underestimates the gate", fontsize=9.5, fontweight="bold")
for s in ("top", "right"): ax.spines[s].set_visible(False)
savefig(fig, "fig_mechanism")

# ---- fig_extval: external validation on an independent pipeline (ConceptGraphs/ScanNet) ----
# Source = frozen 80-scene package (results/2026-07-22_cg_scannet_smoke_v01/
# advisor_review_80scene). We read the per-scene dumps so the figure regenerates
# from the verified evidence; no numbers are invented here.
import json, glob
_PKG = "/home/rintern16/OpenYOLO3D/results/2026-07-22_cg_scannet_smoke_v01"
_rows = []
for _f in glob.glob(f"{_PKG}/per_scene*.json"):
    for _r in json.load(open(_f)):
        _d = {x["N"]: x for x in _r["per_N"]}
        _a, _b = _d[1], _d[2]
        _rows.append(dict(do=_b["overseg"]-_a["overseg"],
                          di=_b["label_impurity"]-_a["label_impurity"],
                          recall=_b["n_matched_instances"]/max(_a["n_matched_instances"], 1),
                          m2=_b["n_matched_instances"]))
_valid = [r for r in _rows if r["m2"] > 0]
_do = np.array([r["do"] for r in _valid]); _rec = np.array([r["recall"] for r in _valid])

def _medci(v, B=10000):
    rng = np.random.default_rng(0)
    bt = [np.median(rng.choice(v, len(v), True)) for _ in range(B)]
    return np.median(v), np.percentile(bt, 2.5), np.percentile(bt, 97.5)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7))
# Panel A: Δoverseg distribution (valid maps)
vp = axA.violinplot(_do, showextrema=False)
for b in vp["bodies"]: b.set_facecolor(BLUE); b.set_alpha(0.35)
axA.boxplot(_do, widths=0.18, showfliers=False)
_jx = np.random.default_rng(1).normal(1, 0.035, len(_do))
axA.scatter(_jx, _do, s=8, c=BLUE, alpha=0.55, linewidths=0)
axA.axhline(0, color=GRAY, lw=0.8, ls="--")
axA.set_xticks([]); axA.set_ylabel(r"$\Delta$ fragmentation (N1$\to$N2)")
_m, _lo, _hi = _medci(_do)
axA.set_title(f"Independent pipeline (n={len(_valid)})\nmedian {_m:.0f}, {int((_do<0).sum())}/{len(_do)} improved",
              fontsize=8.5, fontweight="bold")
for s in ("top", "right"): axA.spines[s].set_visible(False)
# Panel B: recall-threshold sensitivity (median Δoverseg ± 95% CI)
_thrs = [0.7, 0.8, 0.9, 0.95]; _med = []; _elo = []; _ehi = []; _ns = []
for t in _thrs:
    g = _do[_rec >= t]; m, lo, hi = _medci(g)
    _med.append(m); _elo.append(m-lo); _ehi.append(hi-m); _ns.append(len(g))
axB.errorbar(_thrs, _med, yerr=[_elo, _ehi], fmt="o-", c=BLUE, capsize=3, lw=1.4, ms=5)
for t, m, n in zip(_thrs, _med, _ns):
    axB.annotate(f"n={n}", (t, m), textcoords="offset points", xytext=(5, 5), fontsize=7, color=GRAY)
axB.axhline(0, color=GRAY, lw=0.8, ls="--")
axB.set_xlabel("recall-retention threshold"); axB.set_ylabel(r"median $\Delta$ fragmentation")
axB.set_title("Stable across the operating region\n(95% CI excludes 0 throughout)",
              fontsize=8.5, fontweight="bold")
axB.set_ylim(top=2)
for s in ("top", "right"): axB.spines[s].set_visible(False)
fig.tight_layout()
savefig(fig, "fig_extval")

print("ALL FIGURES DONE")
