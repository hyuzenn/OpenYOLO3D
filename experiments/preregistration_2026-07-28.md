# Pre-registration — Retroactive emission for the temporal confirmation gate

**Date:** 2026-07-28 (KST), written and committed **before** inspecting the gate
implementation and **before** running any retro-gate experiment.
**Status at commit time:** no retro-gate result exists. No number below is filled in.

---

## 1. Hypothesis under test

E2 (box-count-matched control, `results/2026-07-28_e2_thrmatch_v01`) showed that at an
exactly matched emitted box budget, a plain baseline score threshold beats the N=3
temporal confirmation gate on HOTA, DetA, IDF1, mAP and recall in both association
frames, with near-unanimous per-scene agreement (rank-biserial ≈ −1.0).

**Hypothesis (H1):** the current outdoor gate is implemented as a *prefix-deleting
emission mask* — a track's first N−1 observations, occurring before the track reaches
`hits >= N`, are permanently dropped and never emitted. Those frames are pure loss:
they are deleted regardless of detection confidence or localization quality. This
implementation artifact, not the temporal reasoning itself, is what distorts the E2
comparison against a confidence threshold.

**Supporting evidence available at pre-registration time:** the gated arm's GT-anchored
fragmentation is *exactly identical* to the baseline arm's (10.52 ego / 4.45 global),
i.e. the gate changes what is emitted but provably does not change the associator's
track partition — consistent with a pure emission mask.

**H1 is confirmed** if the code inspection (TASK B) shows no buffer retaining
unconfirmed observations and no path from the gate back into association, i.e. the
pre-confirmation frames are unrecoverable.

**Proposed change under test (retro-gate):** on a track reaching `hits >= N`, emit its
retained pre-confirmation observations retroactively (flush), instead of deleting them.
The gate then removes only tracks that never confirm, rather than also removing the
opening frames of tracks that do confirm.

---

## 2. Prior justification for the retroactive-emission switch

Recorded **before** seeing any retro-gate number, so that this cannot be read as a
post-hoc rescue of a failed method.

1. **Offline-evaluation convention.** Both nuScenes detection/tracking evaluation and
   TrackEval score a complete, already-recorded sequence. Under that protocol a
   confirmed track's history is available at scoring time, and retroactive emission is
   the standard treatment (cf. track initialization / birth handling in offline MOT
   evaluation). Deleting the confirmation prefix is a *streaming* constraint being
   silently imported into an *offline* benchmark.
2. **Definitional consistency with the indoor ConceptGraphs leg.** The paper claims a
   single "identical gate" is applied in both legs. The indoor leg filters *objects*
   that were confirmed in >= N frames, which is inherently a whole-object (retroactive)
   decision — the object's full observation set is kept or dropped as a unit. If the
   outdoor leg instead deletes a per-frame prefix, the two operators are **not** the
   same operator, and the paper's cross-domain claim is unsupported as written.
   TASK B must adjudicate this; the retro-gate is the variant that would make the claim
   true rather than merely asserted.
3. **Mandatory cost disclosure (binding commitment).** Retroactive emission is **not
   deployable in real time**: emitting frame *k*'s box only once confirmation arrives at
   frame *k+N−1* requires either a delayed output stream or output revision. Any paper
   text, table or figure reporting retro-gate numbers **must** carry, in the same place:
   (a) the statement that the retro variant is offline-only, and (b) the induced latency
   in frames and in seconds at the sensor rate. Reporting retro numbers without this
   disclosure is pre-committed as unacceptable, whatever the numbers turn out to be.
4. **Both variants are reported.** The streaming (prefix-deleting) gate is not deleted
   from the paper. Retro and streaming are reported side by side; the streaming variant
   remains the deployable one, the retro variant isolates the mechanism's value from the
   emission-policy artifact.

---

## 3. Decision rule (fixed in advance)

The planned dominance sweep varies the emitted-box budget (equivalently the matched
threshold t / gate parameter N) and compares **retro-gate vs. score-threshold control at
matched box count** on the E2 metric set: mAP, HOTA, AssA, DetA, IDF1, AMOTA,
fragmentation, recall. Association frames ego and global are analyzed separately; a
conclusion requires the pattern to hold in **both** frames, otherwise the weaker of the
two verdicts is adopted.

"Wins an axis" = positive Δ(control → retro-gate) whose 95% scene-bootstrap CI excludes
zero for the per-scene metrics; for the dataset-level devkit metrics (mAP, AMOTA), a
consistent sign across the swept budget points.

| Sweep outcome | Pre-committed paper conclusion |
|---|---|
| Retro-gate beats the threshold control on **no** axis | Convert to a **negative-result / evaluation-methodology paper**: report that persistence-based confirmation is dominated by confidence ranking at matched budget, and that the apparent gain in the original framing was a box-count artifact. No method contribution is claimed. |
| Retro-gate wins **only on AMOTA and/or AssA** | Shrink the contribution to **class-aware identity hygiene**: the layer suppresses spurious identities, not bad boxes. No per-frame detection-quality claim. Contingent on the track-count confound being addressed (E2b), since AMOTA/AssA are identity-count sensitive. |
| Retro-gate wins on **several axes but only within a specific budget interval** | Define that interval as the method's **operating region** and report it explicitly (with its box/recall range); state that outside it the layer is dominated by thresholding. No global superiority claim. |
| Retro-gate dominates across the **entire** swept budget range | Keep the paper as a **method paper** — but the "mAP decreases" headline is **discarded**; the claim is restated on the axes that actually dominate, with the offline-only/latency disclosure of §2.3. |

Ties and mixed frames resolve downward (to the more conservative row).

## 4. Artifacts retired regardless of outcome

Under **every** one of the four branches above, the following are pre-committed for
**removal** from the paper, because both rest on the box-count-matched comparison whose
interpretation E2 already invalidated:

- **Fig. 1 "Same boxes, opposite verdicts"** — retired.
- **Tab. 2** (in its current form) — retired; any replacement is a new table built on the
  sweep protocol above, not an edit of the existing one.

This retirement is not conditional on the retro-gate result and is not renegotiable by a
favorable sweep.

---

## 5. Analysis protocol (unchanged from E1/E2, restated for completeness)

- Same detector cache (γ gravity-corrected CenterPoint), same GT, same evaluator, same
  scene list for every arm of a comparison; scene-list provenance to be resolved in
  TASK C before any sweep is run.
- Per-scene paired Wilcoxon signed-rank + rank-biserial effect size; 95% CI from a
  10,000-resample scene bootstrap (seed 20260718) with TrackEval `combine_sequences`
  recombination. mAP and AMOTA are dataset-level devkit outputs → point estimates only.
- No paper-body edits until the sweep is complete and reported.
