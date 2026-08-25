# 2026.08.04 — 논문 전면 재작성 → method-paper 재프레이밍

*Record of why and how the ICCV draft was restructured. Written so that it is
readable much later, with no memory of the session.*

Paper: **"Retrospective Confirmation for Identity-Consistent Streaming 3D Perception"**
Source: `~/OpenYOLO3D/paper_iccv_draft/`

---

## 1. Where we were before this change

Two earlier framings had already been tried and abandoned:

| Date | Framing | Why it died |
|---|---|---|
| ~2026-07 | **OV-TCS metric paper** — propose a GT-free open-vocab temporal-consistency score | E1 GT-validation: OV-TCS correlated only weakly with GT metrics (ΔR²≈0.014); a metric paper needs the metric to be the result |
| 2026-08-03 | **Evaluation-protocol paper** — the contribution is the matched-control protocol; the temporal layer is the case study | Whole manuscript was rewritten end-to-end under this framing, supplement deleted. But the protocol alone is a methodology note, not a 3DV/ICCV-class contribution: it proposes no mechanism, and reviewers read "we ran a fair comparison" as hygiene, not novelty |

So on 2026-08-03 we had a complete, self-consistent draft whose headline was
*the protocol*. The 08-04 change keeps that manuscript's body and evidence but
moves the headline.

---

## 2. What triggered the 08-04 reframing

A component-level novelty audit of what we internally call the "temporal
layer." The layer packages six mechanisms — and five of them are prior art:

| # | Mechanism | Already published as | Novel here? |
|---|---|---|---|
| 1 | Nearest-centroid association across frames | SORT (2016) | No |
| 2 | Class-agnostic 3D association | AB3DMOT (2020) | No |
| 3 | M-out-of-N track confirmation before reporting (our N=3) | SORT / DeepSORT (2016–17) | No |
| 4 | Track death by max-age / gap tolerance | SORT, AB3DMOT | No |
| 5 | Center-based association on frozen 3D detections | CenterPoint (2021) | No |
| 6 | Per-frame label fusion onto a persistent instance | ConceptFusion / ConceptGraphs (2023–24) | No |
| 7 | **Retrospective emission of the pre-confirmation prefix** | — none. SORT/DeepSORT/AB3DMOT are all *prefix-deleting* by construction | **Yes** |

**Reading.** Naming the contribution after the container ("temporal layer")
claims rows 1–6 as well, and rows 1–6 are a decade of citable prior work. Row 7
is the only mechanism a reviewer cannot point at an existing paper for.

The empirical fact that sealed it — the association gain **changes sign** with
emission policy and association frame, i.e. the effect is *conditional*, not the
robust general-purpose behaviour a "layer" claim implies:

| Association frame | Emission policy | ΔAssA vs matched control | 95% CI | Direction |
|---|---|---|---|---|
| Sensor | Retrospective | +0.0491 | [+0.0370, +0.0615] | gain |
| Sensor | Causal | +0.0756 | [+0.0577, +0.0922] | gain |
| World | Retrospective | +0.0139 | [+0.0086, +0.0195] | gain |
| World | **Causal** | **−0.0116** | [−0.0183, −0.0047] | **loss** |

Two consequences:

1. **Row 7 is load-bearing, not decorative.** Sensor-frame gain is actually
   *larger* without retrospection (+0.0756 > +0.0491); what retrospection buys is
   the **world-frame sign**. Delete row 7 and half the result goes with it.
2. **A conditional effect under an unconditional name is exactly the mismatch
   reviewers punish.** "Retrospective confirmation" carries the condition in its
   own name; "temporal layer" does not.

---

## 3. The reframing, precisely

> **Primary contribution = the training-free retrospective confirmation module**
> (confirmation test + retrospective emission of the confirmed prefix).
> The matched controls are the **evidence standard** behind that claim, not the
> headline.

Key definitional decisions, all deliberate:

- **"Module" is a defined term** (Method §1): the composition of associator +
  confirmation test + emission policy. Novelty rests on the *assembled operator*
  — never on the confirmation test alone (that would be row 3 = SORT).
- **The title names the mechanism, not the module.** "Retrospective
  Confirmation…" — the mechanism is what is new.
- **No slot claim.** We do not claim "a temporal slot in the pipeline" and then
  compare operators in it; that would require ≥2 operators under one protocol,
  and we evaluate exactly one.
- **Release is a reproducibility note, never a contribution bullet.**
- **Do not reintroduce the paper as an evaluation-protocol paper.** The protocol
  survives as contribution #3, not as the thesis.

**Critical property: the reframing cost zero evidence.** Both framings rest on
*identical* experiments. It removed four reviewer attack surfaces and required no
new runs.

### Contribution list as it now stands (`sec/1_intro.tex`)

1. A training-free retrospective confirmation module — frozen detector output,
   single operating point N=3, no retraining, no change to box coords/scores/labels.
2. The retrospective emission policy, isolated — causal emission narrows *where*
   the gain holds (survives sensor frame, reverses in world frame); price is
   N−1 frames of latency (1.0 s at 2 Hz for N=3).
3. A matched-control evaluation protocol — detection-budget-matched and
   identity-budget-matched, paired per-sequence statistics.
4. Cross-domain validation under that protocol — nuScenes 150 scenes + ScanNet200
   312 scenes, identical module.

### The finding is stated as an explicit trade-off

AssA and AMOTA **improve** at equal output budget; mAP, detection recall, DetA
and aggregate HOTA **degrade**. Never write that the module improves detection
quality. Both sides are presented as the result.

Headline numbers now in the abstract:
- AMOTA 0.055→0.089 and 0.166→0.203 (point estimates; consistent in sign across
  both controls and both frames — no CIs exist for AMOTA/mAP/NDS, they are
  whole-split estimators).
- Indoor: −26.9 % class-label switching vs an identity-matched control,
  309/312 scenes, segmentation AP unchanged; a random selection of equal size
  reproduces **none** of the reduction.

---

## 4. What was dropped, and where it went

- **Rows 1–6 are now credited to prior work in Related Work**, not claimed. This
  is what converts a novelty liability into positioning strength: the paper says
  exactly one new thing and proves it under a control designed to remove it.
- **Supplement deleted (08-03)** — the paper is self-contained across four
  section files; no appendix, and internal run IDs now have no permitted location
  anywhere.
- **`retired/`** holds the pre-rewrite figures (`figs_old/`),
  `make_gate_figs.py`, `figure_specs.md`. They render numbers no longer in the
  paper — never re-include.
- **ConceptGraphs external validation excluded** (no matched control exists for it).
- **The paper currently has no figures.**

### Terminology banned from the body

`OV-TCS`, `gate`, `gate sweep`, `Temporal Layer`, `Semantic Relabel`,
`E1/E2/E2b/E2c`, `M11/M21/M22/M31/M32`, `gamma`, `retro`, `detguided`, and all
internal run IDs. Use standing CV terms instead: *confirmation-based track
initialization*, *matched control*, *emission policy*.

---

## 5. Canonical number sources (unchanged by the reframing)

| Comparison | File |
|---|---|
| Detection-budget-matched, retrospective emission (main result) | `results/2026-07-30_e2c_retro_thrmatch_v01/e2_report.json` |
| Identity-budget-matched (top-K + random-K) | `results/2026-07-31_e2b_trackmatch_v01/e2b_report.json` |
| Causal-emission ablation | `results/2026-07-28_e2_thrmatch_v01/e2_report.json` |
| Indoor matched control (ScanNet200) | `results/2026-08-01_indoor_matched_control_v02/report.json` |

Pre-registrations: `experiments/preregistration_2026-07-28.md`,
`…_E2b_2026-07-31.md`, `…_indoor_matched_2026-08-01.md`.

Fixed facts previously gotten wrong: outdoor is **150 scenes** (not 146);
bootstrap = 10,000 resamples, seed `20260718`; fragmentation and track-length are
**pre-selection invariants** (identical across arms — never attribute them to the
method).

---

## 6. Open item found while writing this record

`sec/1_intro.tex:178` (end of Related Work, metrics paragraph) still reads:

> "…our contribution is the controlled experiment and the protocol that reads it."

That sentence is a leftover from the 08-03 protocol-paper framing and contradicts
the method-paper thesis stated in the abstract and contribution list. It should be
rewritten to name the module as the contribution and the protocol as the evidence
standard. Not yet changed.

---

## 7. One-sentence version for the defense

> "Temporal layer" names a container whose contents are largely SORT-era prior
> art; the only mechanism in it without a citable precedent is retrospective
> emission of the pre-confirmation prefix, and it is also the mechanism the
> ablation shows to be load-bearing — so the contribution is named after that,
> not after the container.
