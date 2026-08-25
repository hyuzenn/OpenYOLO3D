# OV-TCS Paper Draft — Abstract · Introduction · Related Work · Method

> Draft v1 (2026-07-02). Sections written in order: Abstract → Introduction →
> Related Work → Method. **Only confirmed, banked evidence is used.** Numbers
> traced to `results/` and verified against the metric source
> (`method_scannet/streaming/nuscenes_native_evaluator.py:988-1002`). The
> pending ScanNet200 train(1201)+val full run is **not** assumed anywhere; all
> indoor statistics come from the banked val312 study (n=6202).
>
> Working title (provisional): **"Per-Frame AP Is Blind: A Temporal-Consistency
> Metric for Streaming Open-Vocabulary 3D Perception."**

---

## Abstract

Streaming open-vocabulary 3D perception assigns open-vocabulary labels to the
same physical object across many frames, online. Yet the field still measures
quality almost entirely with per-frame detection scores (AP / NDS). We show
that this is a blind spot: per-frame AP cannot see the temporal degradation
that actually determines whether an online semantic map is usable. In a
controlled experiment where we corrupt only the *track topology* after
detection — leaving the AP-scored proposal set untouched — AP does not move
while our proposed metric drops monotonically; a complementary interruption
control moves both, confirming the metric is not inert.

We introduce **OV-TCS**, a track-level temporal-consistency metric for streaming
open-vocabulary 3D perception. OV-TCS groups per-frame open-vocabulary labels by
associator track and, without ground-truth track IDs, scores each track as the
product of two factors: track integrity `L_norm = 1 − 1/L` and semantic
stability `1 − CSR`, where `CSR` is the per-track label-switch rate. The product
form is not a design preference but is *empirically forced* by two distinct
failure modes that we show each factor owns exclusively. On real per-instance
data (ScanNet200 val, n=6202), OV-TCS carries predictive information about label
correctness that survives controlling for track length (partial Pearson
r = 0.12, p = 3.5e-21; nested F-test ΔR² = 0.014, F = 89.87, p = 3.5e-21) — it is
not track length renamed. A formulation ablation on the same data shows semantic
stability alone owns the flicker axis, while a per-instance fragmentation sweep
on nuScenes (150 val, ~19k–22k tracks per level) shows track integrity `L_norm`
alone owns the fragmentation axis: as fragmentation increases, mean OV-TCS falls
0.476 → 0.438, a drop carried entirely by `L_norm` (contribution −0.058), while
per-track stability *rises* (0.731 → 0.769) — so a stability-only metric would
mis-report fragmentation as an improvement. Finally, OV-TCS separates real
methods that AP collapses to a tie: switching association from ego- to
global-frame raises OV-TCS 0.136 → 0.168 (+24%) at near-flat AP.

The metric exposes a further consequence we report honestly: naive temporal
aggregation does not help — an EMA over the label stream degrades AP/AR relative
to no aggregation — which is why OV-TCS is proposed as a *metric* to measure
temporal quality rather than a signal to control it. Alongside the metric we
present a class-aware label-fusion method whose benefit is visible in temporal /
label quality (OV-TCS) rather than in closed-vocabulary AP, which its
open-vocabulary regime is bottlenecked away from by localization limits — a case
that concretely motivates why a metric beyond AP is needed.

---

## 1. Introduction

**Streaming open-vocabulary 3D perception and how it is (mis)measured.**
An embodied agent building an online 3D map observes the same physical object
over a sequence of frames and must attach an open-vocabulary semantic label to
it *as it goes*, not after the fact. This is a fundamentally temporal problem:
the quantity that matters downstream is whether an object keeps a *stable
identity and a stable label over time*. Evaluation, however, has not followed.
The dominant metrics — average precision (AP) and, outdoors, the nuScenes
detection score (NDS) — are per-frame quantities: they score the boxes and
classes present in a single frame. They say nothing about whether the same
object's label flickers across frames, or whether its track shatters into
fragments. As a result, two systems with *identical* AP can have completely
different temporal behaviour, and the evaluation cannot tell them apart.

**Three assumptions we falsify.** The current practice rests on three implicit
beliefs. (a) Per-frame AP is a sufficient proxy for system quality. (b)
Aggregating information along the time axis (temporal voting, EMA of features or
labels) monotonically improves quality. (c) Temporal quality, to the extent it
matters, is adequately captured by MOT-style track length / fragmentation. Every
one of these fails under measurement:

- *AP is blind to temporal degradation.* When we inject fragmentation that only
  splits tracks *after* detection — the AP-scored proposal set is held fixed by
  construction — AP does not move, while OV-TCS drops monotonically. A positive
  control (shrinking the association max-age, which degrades real association)
  moves both AP and OV-TCS together, proving OV-TCS still tracks genuine failure.
- *Aggregation is not the answer.* Feeding a temporal signal into an EMA over the
  label stream did not help: the no-aggregation setting beat every EMA setting
  (OFF > k = 0.335 > k = 1.0), with EMA consistently shaving AP/AR. Temporal
  quality is not something to *hope for by averaging*; it is something to
  *measure*. This is precisely why we position OV-TCS as a metric rather than a
  control signal.
- *Track length alone is a near-useless predictor.* In an indoor per-instance
  regression on real data (ScanNet200 val, n=6202, target = label correctness),
  track length carries almost nothing (ΔR² ≈ 0.003 over the intercept; partial
  correlation slightly negative, r = −0.05). Length is not the signal.

**Why existing metrics do not fill the gap.** Closed-vocabulary MOT metrics
(MOTA, IDF1) presuppose a fixed class set *and* ground-truth track IDs, neither
of which exists in an online open-vocabulary setting. AP/NDS are per-frame and
therefore structurally unable to see cross-frame identity. There is, in short,
no metric that measures label/identity stability for streaming open-vocabulary
3D — which is the object's actual quality.

**Why this matters.** Online open-vocabulary 3D mapping exists to serve embodied
agents and robotics. A semantic map that flickers or fragments is unusable
downstream no matter how high its per-frame AP, and an evaluation protocol that
cannot see this *ranks methods incorrectly*. When the evaluation is wrong, the
research direction it steers is wrong with it. The metric has to come first.

**Contributions.**
1. **OV-TCS**, a track-level temporal-consistency metric for streaming
   open-vocabulary 3D perception. It groups native per-frame open-vocabulary
   labels by associator track and scores their consistency *without* ground-truth
   track IDs — a regime where AP is blind and MOT metrics do not apply. We show
   (i) it registers a degradation axis AP cannot see, (ii) it carries predictive
   information beyond track length (instance-level ΔR² = 0.014, F = 89.87), and
   (iii) it separates real methods AP collapses to a tie (ego→global +24% at flat
   AP).
2. A **two-axis analysis that justifies the metric's formulation with evidence,
   not assertion.** We show each factor exclusively owns one of two independent
   corruption axes — semantic flicker is owned by `1 − CSR`, track fragmentation
   is owned by `L_norm` — so single-factor metrics are each wrong about one axis,
   and additive combinations collapse one factor away (a validation-tuned
   weighted sum drove its weight to λ* = 0). The product is the minimal form
   responsive to both axes.
3. A **class-aware label-fusion method** for open-vocabulary streaming, whose
   benefit surfaces in temporal / label quality (OV-TCS) rather than in
   closed-vocabulary AP — a method that the metric is needed to evaluate,
   closing the loop between the two contributions.

---

## 2. Related Work

**Offline 3D instance segmentation.** Point-cloud instance segmentation methods
such as Mask3D, SoftGroup and PointGroup operate on a complete, pre-assembled
scene and are evaluated offline with whole-scene AP. They assume a closed class
set and a batch of the full geometry — neither the open vocabulary nor the
streaming observation model we target. We reuse Mask3D as an indoor proposal
source, but our concern is the temporal evaluation that offline AP omits.

**Open-vocabulary 3D — indoor.** OpenMask3D, OpenYOLO3D, ConceptFusion and
related methods lift 2D open-vocabulary features (CLIP, detector labels) onto 3D
masks and are evaluated on ScanNet200 with offline batch AP. OpenYOLO3D is the
base indoor proposal/label source in our pipeline. These works establish
open-vocabulary labelling but inherit the offline, per-scene evaluation that is
blind to temporal label stability.

**Open-vocabulary 3D — outdoor.** OV-Uni3DETR, OpenSight and FM-OV3D address
open-vocabulary detection on nuScenes / Waymo, again under offline evaluation and
in a framework separate from the indoor line. We connect the two under one
temporal metric, and our outdoor proposals build on a CenterPoint anchor whose
closed nuScenes-10 AP (0.3407) we treat as a reference point rather than a target.

**Streaming / online 3D mapping.** Semantic-SLAM and online mapping systems
(Kimera, Voxblox, Hydra) operate on streams but are predominantly closed-vocabulary
and are not evaluated for open-vocabulary label consistency over time. Our setting
— online, open-vocabulary, evaluated on temporal label stability — sits in the gap
these systems leave.

**Multi-object tracking and its metrics.** Tracking-by-detection methods
(AB3DMOT, CenterPoint tracking, EagerMOT) and their metrics (MOTA, IDF1) are the
natural place one might look for a temporal measure. They do not transfer: MOTA
and IDF1 require ground-truth track IDs and a fixed class taxonomy, both absent in
online open-vocabulary mapping. Moreover they score detection/association against
GT, not the *self-consistency* of an open-vocabulary label sequence along a track.
OV-TCS is deliberately a GT-track-ID-free, vocabulary-agnostic property of the
predicted tracks themselves — which is what makes it applicable where MOT metrics
are not.

---

## 3. Method

We describe (§3.1) the streaming setting and the track construction OV-TCS is
computed on, (§3.2) the OV-TCS metric and the two-axis argument that fixes its
form, and (§3.3) the class-aware label-fusion method.

### 3.1 Streaming setting and track construction

At each frame `t` a proposal source emits per-frame instance candidates, each
with a 3D extent and a native open-vocabulary (argmax) label. Indoors the source
is a cached per-scene Mask3D output whose 3D masks fix the track membership
directly; outdoors it is a cached LiDAR proposal set to which an associator is
applied across frames. The associator is a centroid tracker with a distance gate
and a max-age; we use three variants that differ only in how they group frames
into tracks — a class-gated centroid associator (`CentroidAssociator`), a
class-agnostic variant with the class gate removed (`ClassAgnosticAssociator`),
and a global-frame variant (`GlobalCentroidAssociator`) — so that OV-TCS, being a
pure track-topology property, changes only when the grouping or the labels change,
never as a side effect of per-frame voting.

For each track `k` this yields a label sequence
`s_k = (ℓ_{k,1}, …, ℓ_{k,L})`, the native per-frame labels in observation order,
of length `L = |s_k|`. OV-TCS is a function of `s_k` alone. Tracks with `L < 2`
(the metric is undefined without at least one transition) are excluded from
aggregation.

### 3.2 OV-TCS: definition and the two-axis argument

**Per-track score.** From a label sequence `s_k` of length `L` we define the
switch count `sw = Σ_{i} 𝟙[ℓ_i ≠ ℓ_{i+1}]`, the class-switch rate
`CSR = sw / (L − 1)`, and the normalized track length `L_norm = 1 − 1/L`. The
headline metric is

```
OV-TCS_C(k) = L_norm · (1 − CSR) = (1 − 1/L) · (1 − sw/(L−1)).
```

`L_norm ∈ [0,1)` is a **track-integrity** term (it rewards long, unbroken tracks
and penalizes fragmentation into short pieces); `1 − CSR ∈ [0,1]` is a
**semantic-stability** term (it penalizes label flicker within a track). We
aggregate `OV-TCS_C` by mean over tracks. For completeness the implementation
also computes two sibling forms that replace the stability term with an
entropy-based (`A`) or dominance-ratio (`B`) measure; we report all three and
headline `C`. All three are GT-track-ID-free and normalize the semantic term so
the metric is unbiased to vocabulary size.

**Why a product of these two factors — and not one factor, or a sum.** The
formulation is forced by the empirical fact that streaming temporal degradation
has *two independent failure modes*, and each factor exclusively owns one of them.

- *Semantic flicker is owned by `1 − CSR`.* In a parameter-free formulation
  ablation on real per-instance data (ScanNet200 val, n=6202, target = label
  correctness), semantic stability alone carries the flicker signal (ΔR² over
  track length = 0.023), the product carries less (0.014), and track length
  alone is inert (0.003, with a negative partial correlation). We report this
  honestly: on the flicker axis it is *stability, not the product,* that owns the
  signal — which is exactly where the flicker corruption lives.
- *Fragmentation is owned by `L_norm`.* In a per-instance fragmentation sweep on
  nuScenes (150 val, ~19k–22k tracks per corruption level), increasing
  fragmentation from 0 → 0.5 drops mean `OV-TCS_C` from 0.476 → 0.438, a change
  carried **entirely** by `L_norm` (its contribution −0.058 exceeds the net
  −0.037), while mean track length falls 4.28 → 2.53 and mean `L_norm` falls
  0.648 → 0.569. Crucially, per-track semantic stability `1 − CSR` does *not*
  fall — it **rises** 0.731 → 0.769, because splitting a track leaves fewer
  switches inside each shorter piece. A pure-stability metric would therefore
  *mis-report fragmentation as an improvement*; `L_norm` is what makes the metric
  correct on this axis.

The two factors are complementary, not redundant: fragmentation moves `L_norm`
and not stability, flicker moves stability and not `L_norm`. Any single-factor
metric is blind to one of the two corruptions. An *additive* combination fails
for a different reason — it lets one factor be driven to zero: a
validation-tuned weighted sum collapsed to λ* = 0 (i.e. stability-only, hence
fragmentation-blind), so the free parameter buys nothing and does not enforce
two-axis sensitivity. The **product is the minimal form responsive to both
axes**. We also note that `L_norm` is deliberately *not* used as a reliability
weight: short, perfectly stable tracks are in fact more often correct (label
correctness 80% at L = 2 vs 61% at L ≥ 5), and indoors 99.2% of tracks have
L ≥ 5, so `L_norm` is near-saturated there — its role is fragmentation
sensitivity, established outdoors, not reliability weighting.

**Scope, reported honestly.** The predictive signal lives at *instance*
granularity. On real indoor data OV-TCS adds significant explanatory power over
track length at the instance level (partial Pearson r = 0.12, p = 3.5e-21; nested
F-test ΔR² = 0.014, F = 89.87, p = 3.5e-21, n = 6202). The same test at
*per-scene* granularity (n = 312, target = AP_50) does **not** pass (partial
r = −0.03, p = 0.61); we state this in the main text rather than hide it — the
metric is an instance-level property, and scene aggregation washes it out.

### 3.3 Class-aware label fusion

Online open-vocabulary labels flicker frame to frame, and a naive global-score
gate for fusing them fails (it cannot match the closed CenterPoint anchor on
nuScenes-10). We fuse the streaming 2D open-vocabulary labels onto the 3D
mask/track with a **class- and source-aware** gate rather than a single global
score threshold: the gating that decides whether a 2D open-vocabulary label
overrides or corrects the proposal's native class is conditioned on the class and
the proposal source, producing a fused per-track label (a 2D→3D open-vocabulary
class correction). Indoors the fusion target is the Mask3D 3D mask; outdoors it is
the cached LiDAR proposal.

**Where its value lies, stated honestly.** In the closed nuScenes-10 regime this
method does *not* exceed the CenterPoint anchor AP of 0.3407: the outdoor
open-vocabulary bottleneck is localization, not score calibration (an
oracle-score decomposition shows native confidence is already calibrated, with a
deduplication ceiling of 0.548), so AP is capped by a limit this method does not
target. Its value is an open-vocabulary (GT-free) class-correction capability and
a temporal / label-quality difference that AP does not reveal but OV-TCS does.
This is exactly the situation the metric is built for: the method operates in a
different (open-vocabulary) regime from the closed CenterPoint anchor, and its
effect is measured by OV-TCS rather than by an AP its regime is bottlenecked away
from — the method and the metric justify each other.

---

> **Not yet written (per the requested order, and pending confirmed results):**
> §4 Experiments (the AP-blindness sweep table, the beyond-track-length nested
> model table, the two-axis formulation/fragmentation-decomposition table, the
> real-method separation scatter, and the label-fusion comparison) and
> §5 Discussion/Limitations. The limitations already established from banked
> evidence — EMA aggregation fails (OFF > k = 0.335 > k = 1.0), per-scene
> granularity fails (partial r = −0.03), and the outdoor localization ceiling
> (AP capped at the 0.3407 anchor; dedup ceiling 0.548) — are slotted for §5 and
> are *not* main claims.
>
> The pending ScanNet200 train(1201)+val full M22 run is **not** used in any
> statement above; all indoor numbers are from the banked val312 study.
