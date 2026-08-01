# Pre-registration — Indoor matched control (ScanNet200 val-312)

**Date frozen:** 2026-08-01, before any computation and before the control code is written.
**Parents:** `preregistration_2026-07-28.md` (outdoor gate investigation),
`preregistration_E2b_2026-07-31.md` (track-count-matched control, PASS).
**Purpose:** the paper's cross-domain claim ("the same confirmation gate provides
identity/semantic hygiene in both legs") is currently supported outdoors only
(E2b). This experiment tests whether the indoor leg's hygiene gain survives a
non-temporal matched control.

## 1. Premise (measured, frozen)

On ScanNet200 val-312 (`results/2026-05-15_streaming_ablation_core_temporal/`):
baseline AP 0.19560 / lsc 23,385; M11 (N=3) AP 0.19540 / lsc 17,023 (−27.2%);
unique emitted instances 10,321 → 10,249 (−0.70%).

The indoor gate's measurable effect is on **label-switch count (lsc)**, not on
AP and not materially on the object count. A box- or AP-based control (the
literal E2b analogue) would therefore be degenerate by construction: both arms
would sit inside AP noise and the experiment would decide nothing. The
decision-bearing endpoint indoors must be **lsc**.

## 2. Hypothesis under test

**H0 (confound):** the gate's lsc reduction is explained by *emitting fewer
object identities* — any non-temporal rule that keeps the same number of
objects per scene would reduce lsc equally.

**H1:** the reduction depends on *which* objects temporal confirmation selects
(and on suppressing their pre-confirmation label flicker); a score-based
non-temporal selection of equal size cannot reproduce it.

## 3. Arms

All arms: identical Mask3D cache (`results/2026-05-13_mask3d_cache/`), config
(`pretrained/config_scannet200.yaml`), frame frequency, evaluator, 312 val
scenes, same GPU YOLO-World 2D pass.

- **Arm G (gate):** M11 FrameCountingGate N=3, cumulative — the production
  configuration. Rerun (not reused) so that the per-scene confirmed-set size
  K_s is recorded; its 312-scene aggregates must reproduce the frozen anchor
  (AP 0.19540, lsc 17,023) — checked before the decision is read.
- **Arm C (control, decision-bearing):** per scene, a **StaticSetGate** whose
  allowed set is the **top-K_s Mask3D proposals by cached detection score**
  (non-temporal, fixed from frame 1), K_s = |gate's end-of-scene confirmed set|
  in that scene. Same code path as M11 (installed in the `method_11` slot;
  per-frame `gate()` + finalize `_confirmed` filter), so the only difference is
  *which* identities are kept and *when* suppression applies.
- **Sanity (not decision-bearing):** random-K_s allowed sets, 3 seeds
  (20260718+i), reported as a range.

Matching target: distinct kept identities per scene, K_s exact by construction
(confirmed sets are subsets of the proposal set; asserted at run time). The
emitted-identity count in pred_history may differ marginally if a top-K
proposal is never visible; the realized match is reported, not silently fixed.

## 4. Endpoints and decision rule (pre-committed)

**Primary (decision-bearing): per-scene lsc**, gate vs. top-K control, paired
over 312 scenes. Statistics as in E2b: two-sided Wilcoxon signed-rank,
rank-biserial, 95% CI from a 10,000-resample scene bootstrap of the mean
Δ(control − gate), seed 20260718. **Gate wins** = Δ(control − gate) > 0 with
the 95% CI excluding zero (gate lsc strictly lower).

**Secondary (reported, not decision-bearing): AP** (mask-IoU, scannet200
evaluator). Expectation under either hypothesis: |ΔAP| vs. baseline < 0.005 for
both arms. If the control's AP collapses (> 0.02 loss), that is reported as a
selection-rule artifact and the lsc verdict stands but is flagged.

| Outcome | Pre-committed conclusion |
|---|---|
| Gate lsc < control lsc, CI excludes 0 | Indoor hygiene gain is **temporal-selection-specific**. The cross-domain identity-hygiene framing (E2b outdoors + this indoors) is admissible. |
| Control matches or beats gate on lsc | Indoor confound **confirmed**: the indoor gain is generic pruning. The indoor leg is demoted to a consistency note; the identity-hygiene claim is presented as **outdoor-only**. |

Ties resolve downward (to the demotion row).

## 5. Disclosed asymmetries (not escape hatches)

1. The gate suppresses the pre-confirmation frames of eventually-confirmed
   instances (streaming semantics); the static control cannot. This prefix
   suppression is *part of the mechanism under test*, not a confound — but the
   report must decompose the lsc gap into never-confirmed removal vs. prefix
   suppression if the gate wins, so the paper does not over-attribute.
2. Indoor and outdoor effect sizes differ by two orders of magnitude (0.70% vs
   ~89% identity pruning). Any cross-domain claim must state this.
3. lsc is an internal-consistency metric (no GT). The GT-anchored indoor
   check remains AP (secondary, null by premise).

## 6. Scope / stop conditions

One operating point (N=3). One selection rule (score top-K) + random sanity.
No re-tuning, no extension without a new pre-registration. If Arm G fails to
reproduce the frozen anchor (AP within ±0.002, lsc within ±2%), the run stops
and the discrepancy is reported before any control number is read.

## 7. Artifacts

`results/2026-08-01_indoor_matched_control_v01/`: `report.json`, `table.md`,
`run.log`, per-arm cells. Artifacts-before-interpretation; no manuscript edits
from within the run.
