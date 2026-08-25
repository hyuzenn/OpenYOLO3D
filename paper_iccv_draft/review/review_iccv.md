# ICCV Review — "OV-TCS: Measuring Temporal Label Consistency in Streaming Open-Vocabulary 3D Perception"

**Recommendation: Weak Reject (4/10) as submitted.**
The problem is real, the validation methodology is unusually honest, and the
experimental chain is coherent — but the submission has no figures, validates
the metric only on the authors' own two pipelines in a low-performance regime,
and its central formulation claim is supported on one of its two axes only by a
synthetic corruption. These are fixable, which is why this is a weak rather
than clear reject.

## Summary
The paper proposes OV-TCS = L_norm·(1−CSR), a GT-free per-track temporal label
consistency score for streaming open-vocabulary 3D perception, and validates
it via (i) AP-blindness under injected fragmentation, (ii) beyond-track-length
prediction of label correctness on ScanNet200 (ΔR²=0.014, F=89.9), (iii) a
two-axis factor-necessity argument, (iv) agreement with HOTA/IDF1/AMOTA on
nuScenes, and (v) a 30-run associator-hyperparameter sweep with zero ranking
flips. It then uses the metric to separate methods with bit-identical mAP.

## Strengths
S1. The gap is genuine: no existing metric measures temporal label consistency
without GT track IDs and a closed vocabulary; the paper demonstrates (rather
than asserts) that mAP is invariant to the association axis.
S2. Validation-as-contribution framing is rigorous by metric-paper standards:
controlled corruptions with a positive control, length-controlled nested
regression, GT-based MOT cross-validation, and an evaluator-hyperparameter
sensitivity study; failures (per-scene, EMA control-signal) are reported.
S3. The MOT dissociation (correlates with AssA/IDF1, not DetA/MOTP) is exactly
the right sanity structure for a GT-free surrogate.
S4. Claims are conservative and match the evidence shown.

## Weaknesses
W1 (**major**). **No figures.** Not one. A metric paper needs, at minimum: the
fragmentation dose–response curve (AP flat, OV-TCS falling), the per-scene
OV-TCS-vs-HOTA scatter, the sensitivity sweep curves, and a qualitative
flicker-vs-fragmentation visualization. Tables alone cannot carry §5.

W2 (**major**). **Single-ecosystem validation.** Both pipelines (indoor
Mask3D-tracked, outdoor LiDAR-proposal) are the authors' own. A metric paper
must show the metric discriminates among *third-party* systems (e.g., several
published open-vocab mapping systems, or at least several distinct trackers on
nuScenes). Currently every "method pair" differs in one component of one
codebase.

W3 (**major**). **Product-form necessity rests on synthetic fragmentation.**
On the real flicker axis, stability-only (1−CSR) *beats* the product
(ΔR²=0.023 vs 0.014, the paper's own Tab. 1); on the real method pair
(ego vs global), stability-only produces the same ranking as the product (CSR
0.718 vs 0.580). The only case where the product is *necessary* is injected
fragmentation. The paper needs a real (non-injected) setting where a
single-factor metric gives the wrong answer, or must weaken the claim from
"necessary" to "necessary under a corruption model."

W4 (**moderate**). **Low-performance regime.** MOTA is −4.2 to −4.5 and AMOTA
0.03–0.16; the open-vocab stream has closed-set mAP 0.002. All agreement and
sensitivity results are established for weak systems. Does OV-TCS still agree
with HOTA when tracking is strong (e.g., a standard CenterPoint tracker at
AMOTA ~0.65)? Untested.

W5 (**moderate**). **Effect size and value-range interpretation.** ΔR²=0.014
is honest but thin as the *sole* GT-anchored predictive validation; and the
paper gives no guidance on what OV-TCS differences are meaningful (no
uncertainty: no bootstrap CIs on the 0.136→0.168 headline delta or the E2
+6.8% delta).

W6 (**moderate**). **Gameability not analyzed.** OV-TCS is averaged over
tracks with L≥2. A system that suppresses short/unstable tracks (or fragments
singletons) changes the averaged population; nothing stops score inflation by
track filtering. The n_tracks column hints at this (ego 342k vs global 267k
tracks — the *better*-scoring system also has 22% fewer tracks, which a
skeptic reads as selection) but the text never addresses it.

W7 (**minor**). **Related work misses adjacent video metrics.** STQ (STEP/
video panoptic STQ) and VPQ measure spatio-temporal consistency in video
segmentation; TETA generalizes tracking evaluation. All are GT-based/closed-
vocab, so the gap survives, but the paper must position against them.

W8 (**minor**). Indoor and outdoor validations do not overlap: correctness
prediction exists only indoor; MOT agreement and sensitivity exist only
outdoor. Any single claim is single-domain.

W9 (**minor**). §6.2: mAP/NDS "unchanged" for the open-vocab arms are quoted
from the association-invariant cached eval rather than recomputed per arm —
fine, but the footnote convention from Tab. 3 is not carried over.

## Per-section scores (1–10)
| Section | Score | Note |
|---|---|---|
| Abstract | 7 | accurate, well-scoped |
| 1 Introduction | 7 | claims match evidence; strong framing |
| 2 Related work | 5 | missing STQ/VPQ/TETA (W7) |
| 3 Metric | 6 | clear; gaming/population issues unaddressed (W6) |
| 4 Setup | 6 | honest, but single ecosystem (W2) |
| 5.1 AP-blindness | 7 | clean design, positive control; needs figure |
| 5.2 Beyond length | 6 | correct stats; thin effect (W5) |
| 5.3 Product form | 5 | necessity overstated on one axis (W3) |
| 5.4 MOT agreement | 8 | strongest section; needs scatter figure |
| 5.5 Sensitivity | 8 | thorough; rare to see in metric papers |
| 6 Applications | 6 | compelling tie-breaking; no CIs (W5) |
| 7 Limitations | 8 | unusually candid |
| Presentation | 3 | zero figures (W1) |

## Missing experiments (in decreasing importance)
M1. Multi-system evaluation: ≥3 third-party trackers/streaming systems on
nuScenes val, showing OV-TCS ranks them consistently with HOTA (extends E1).
M2. Strong-tracker regime: MOT agreement for a standard high-AMOTA tracker (W4).
M3. Bootstrap CIs (scene-level resampling) for every headline OV-TCS delta.
M4. Real-world product-necessity case: a real method pair where frag and CSR
trade off in opposite directions, or explicit reframing of the claim (W3).
M5. Gaming analysis: OV-TCS vs minimum-track-length / score-threshold
filtering curves, reported jointly with n_tracks (partially available from
the existing sweep: the p=0.3 arm already shows filtering raises OV-TCS).
M6. Indoor sensitivity or outdoor correctness-prediction (close W8).
M7. Qualitative figure: same scene, flicker vs fragmentation failure.

## Smallest changes required for acceptance
1. Add the four figures (W1) — all plottable from existing logged data; no new
   runs needed.
2. Add bootstrap CIs from the existing per-scene tables (M3) — no new runs.
3. Reframe §5.3: claim "minimal form not directionally wrong on either
   observed failure mode," not factor *necessity* in general; explicitly state
   that on the real pair stability-only agrees, and that the fragmentation
   axis is established by controlled injection (W3).
4. Add the gaming paragraph + n_tracks-aware reporting rule, using the p=0.3
   arm and the ego/global n_tracks asymmetry as the worked example (W6/M5).
5. Add STQ/VPQ/TETA positioning to related work (W7).
6. One additional third-party tracker in the E1 harness (M1-lite): even a
   single published nuScenes tracker submission replayed through the OV-TCS
   harvester would break the single-ecosystem circularity. This is the only
   suggested change that needs (CPU-only) new computation.

With changes 1–5 this moves to borderline accept; with 6 it is a solid poster.
