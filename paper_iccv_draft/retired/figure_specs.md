# Figure specifications — Temporal Layer paper (post-pivot, 2026-07-21)

Story: "Losing mAP, Gaining Tracks." All figures render from the frozen E2 gate
sweep (`results/2026-07-20_e2_gate_sweep/`) via `make_gate_figs.py`. No new
values are invented; every number traces to Tab. 1 (gate sweep) or Tab. 2
(decomposition).

## Fig. 1 teaser — DONE (`figs/fig_teaser.png/pdf`)
Placed Fig. 1, page 1, above the introduction (`sec/1_intro.tex`).
Grouped bars, no-gate (N=1, gray) vs strong gate (N=5, blue) on mAP / HOTA /
IDF1 / DetA, with a red down-arrow on mAP and green up-arrows on the tracking
bars. Message in <10 s: same box geometry, mAP ↓ while tracking ↑.

## Fig. 2 main result — DONE (`figs/fig_gate_sweep.png/pdf`)
Two panels (`sec/3_finalcopy.tex`, §Controlled gate isolation). Left: mAP/NDS
(red, falling) vs HOTA/IDF1/DetA (rising) against gate strength N, twin axis,
baseline N=1 shaded. Right: mAP–AMOTA Pareto; N=2 = Pareto operating point,
N=5 annotated as dominated by N=3.

## Fig. 3 decomposition — DONE (`figs/fig_decomp.png/pdf`)
Grouped bars control / +gate / +relabel for HOTA, AssA, IDF1, AMOTA
(`sec/3_finalcopy.tex`, §Separating the gate from the semantic relabel).
Class-agnostic metrics flat between gate and gate+relabel; AMOTA jumps only with
relabel. Dotted divider + "class-agnostic (gate)" vs "class-aware (relabel)"
callouts.

## Fig. 4 mechanism — DONE (`figs/fig_mechanism.png/pdf`)
mAP (falling) and DetA (rising) over the identical box-set vs N, annotated with
emitted-box counts (`sec/3_finalcopy.tex`, §Why mAP moves the other way). Shows
two detection scores over the same boxes diverging.

## Fig. qualitative — SPEC ONLY (not yet rendered; optional, high polish value)
**Message:** "Same scene, same detections: the ungated stream shatters an object
into flickering fragments; the gated stream keeps one stable, consistently
labeled track." (This is the top item from the acceptance-improvement list.)
- **Scene:** pick the nuScenes val scene with the largest gate-induced
  fragmentation drop from the E2 per-scene dumps (rule stated in caption; no
  eye cherry-pick).
- **Object rule (deterministic):** the GT instance with the highest ungated
  fragmentation among instances with ≥20 observed keyframes.
- **Layout:** 2 rows (ungated top, gated bottom) × [BEV track panel | per-keyframe
  label timeline]. Ungated row = many short colored track segments + flickering
  label cells; gated row = one long track + stable label. Annotate each row with
  its track count and per-object L / label-switches.
- **Data:** per-frame track IDs + argmax labels for both arms (cached), LiDAR
  keyframe points, GT match. CPU-only, no GPU inference.

## Statistical presentation (where CIs live)
- Abstract: headline gate deltas quoted as measured point values (controlled
  single-variable sweep; no per-arm resampling needed — bit-identical cache).
- §Decomposition: gate vs relabel attribution with explicit +0.009 / +0.042 split.
- Any per-scene correlation (OV-TCS negative result, §Exploratory) carries its
  10^4 scene-bootstrap in the supplement, not the main text.
