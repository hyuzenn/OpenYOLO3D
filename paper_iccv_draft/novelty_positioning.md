# Why the main contribution moved from "temporal layer" to "training-free retrospective confirmation"

Internal positioning note. Not part of the manuscript — it uses internal
vocabulary ("temporal layer") that is banned from the paper body.

---

## 1. Component-level novelty audit of the "temporal layer"

The temporal layer, as originally scoped, packages six mechanisms. Five of
them are prior art. This table is the reason the claim had to be narrowed.

| # | Mechanism in the layer | Already published as | Year | Novel here? |
|---|---|---|---|---|
| 1 | Nearest-centroid association across frames | SORT (Bewley et al., ICIP) | 2016 | **No** |
| 2 | Class-agnostic association in 3D | AB3DMOT (Weng et al., IROS) | 2020 | **No** |
| 3 | *M*-out-of-*N* track confirmation before reporting (our *N*=3) | SORT 2016; DeepSORT (Wojke et al., ICIP) | 2016/17 | **No** |
| 4 | Track death by max-age / gap tolerance | SORT 2016; AB3DMOT 2020 | 2016/20 | **No** |
| 5 | Center-based association on frozen 3D detections | CenterPoint (Yin et al., CVPR) | 2021 | **No** |
| 6 | Per-frame label fusion onto a persistent instance | ConceptFusion 2023; ConceptGraphs (Gu et al., ICRA) | 2023/24 | **No** |
| 7 | **Retrospective emission of the pre-confirmation prefix** | — none; SORT/DeepSORT/AB3DMOT are all *prefix-deleting* by construction | — | **Yes** |

**Reading.** Six of seven rows are citable prior work, several of them a
decade old. A contribution named after the container ("temporal layer") is
therefore a contribution whose name covers mostly rows 1–6. Row 7 is the only
mechanism a reviewer cannot point at an existing paper for — so it is the only
defensible place to put the novelty claim.

---

## 2. What each framing claims vs. what we can show

| Test a reviewer applies | "Temporal layer" as main contribution | "Training-free retrospective confirmation" |
|---|---|---|
| What is the new mechanism? | None isolated — the layer is rows 1–6 plus row 7 undifferentiated | Row 7, named in the title |
| Is the slot itself the claim? | Yes → then show ≥2 operators in the same slot compared under one protocol | No slot claim made |
| How many operators do you evaluate? | **1** (N=3 confirmation + retrospective emission) | 1, and the claim is scoped to it |
| Does "layer" mean a learned network layer? | Reader expects loss / differentiability / end-to-end training — we have none | No such expectation raised |
| Are the results robust enough for a general component? | **No** — sign reverses by association frame and emission policy (see §3) | Conditionality is *the* finding, and the name matches it |
| What survives the matched control? | Same evidence either way | Same evidence, correctly sized claim |

The decisive asymmetry: both framings rest on **identical experiments**. The
reframing costs no evidence and removes four of the six attack surfaces above.

---

## 3. The empirical fact that kills the "layer" framing

A general-purpose layer implies a robust, direction-stable effect. Ours is
conditional — the association gain **changes sign** depending on emission
policy and association frame:

| Association frame | Emission policy | ΔAssA vs matched control | 95% CI | Direction |
|---|---|---|---|---|
| Sensor | Retrospective | +0.0491 | [+0.0370, +0.0615] | gain |
| Sensor | Causal | +0.0756 | [+0.0577, +0.0922] | gain |
| World | Retrospective | +0.0139 | [+0.0086, +0.0195] | gain |
| World | **Causal** | **−0.0116** | [−0.0183, −0.0047] | **loss** |

Two consequences:

1. **Row 7 is load-bearing, not decorative.** Sensor-frame gain is *larger*
   without retrospection (+0.0756 > +0.0491); what retrospection buys is the
   world-frame sign. Remove row 7 and the "layer" loses half its result — which
   is precisely why row 7, not the layer, is the contribution.
2. **A conditional effect under an unconditional name is the mismatch
   reviewers punish.** "Retrospective confirmation" carries the condition in
   its own name; "temporal layer" does not.

---

## 4. What actually survived the narrowing

| Retained claim | Where it is stated | Verified by |
|---|---|---|
| Retrospective emission of the confirmed prefix (not standard practice) | Title, Abstract, Contribution 1–2 | Emission-policy ablation (table in §3 above) |
| Training-free application to frozen detector output | Contribution 1 | Every arm replays one frozen detection set |
| One operating point, no per-dataset tuning | Contribution 1 | *N*=3 across nuScenes and ScanNet200 |
| Matched-control protocol for output-volume-changing components | Contribution 3 | Detection-budget and identity-budget controls |
| Cross-domain transfer of the same operator | Contribution 4 | nuScenes 150 scenes + ScanNet200 312 scenes |

Everything dropped in the narrowing (rows 1–6) is now **credited to prior work
in Related Work** rather than claimed. That is what converts a novelty
liability into a positioning strength: the paper says exactly one new thing and
proves it under a control designed to remove it.

---

## 5. One-sentence version for the defense

> "Temporal layer" names a container whose contents are largely SORT-era prior
> art; the only mechanism in it without a citable precedent is retrospective
> emission of the pre-confirmation prefix, and it is also the mechanism the
> ablation shows to be load-bearing — so the contribution is named after that,
> not after the container.
