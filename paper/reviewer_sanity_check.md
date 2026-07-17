# Reviewer sanity check after M1/M3 integration (2026-07-17)

Against the round-2 review (`review_iccv_v2.md`, Borderline 5/10). The review
stated: "With M1–M3 this is a weak accept; with M2 as well it is a solid
accept."

## Resolved by this revision

- **W2 (major — §3.3 rule indicts the headline): RESOLVED.** The rule is now
  applied to the flagship comparison inside §3.3 itself, with four controls
  (box conservation, coverage 82.4%→87.3%, within-stratum gains, GT-paired
  +0.147 [0.142, 0.152]). The skeptic's merging-deletes-bad-tracks model is
  refuted structurally (identical box population) and empirically (gains on
  fixed physical objects are *larger* than the system delta).
- **W5 (moderate — +0.032 vs +0.049 mismatch): RESOLVED.** One
  implementation (production, E1=E4), one statistic (pooled all-track mean,
  the metric's own definition), one CI (+0.052 [0.049, 0.055]); the
  scene-weighted +0.049 is retained, labeled, and consistent. The §3.2
  definition now matches the evaluator (singletons = 0, included), so the
  bootstrapped statistic is exactly the defined one. Provenance of the old
  0.168 is disclosed in §6.1.
- **W9 (minor — pooled-r inflation): PARTIALLY RESOLVED.** Both within-arm
  correlations now disclosed (ego 0.87, global 0.60, pooled 0.78). An
  arm-partialed pooled correlation is still not reported.

## Still open (ranked by expected reviewer damage)

1. **W1 (major — self-referential headline / M2).** Unchanged and still the
   single biggest threat. Every headline number remains our metric on our two
   pipelines. The fix the review demands (replay ≥1 third-party nuScenes
   tracking submission through the OV-TCS harvester, CPU-only) is still the
   highest-ROI remaining experiment; the paper still only promises it as
   future work.
2. **W3 (major — stability-only beats the product on the sole GT-anchored
   test; "5×" sentence).** Untouched. §5.2 still compares against the weakest
   alternative; Tab. 2's own first row still shows 0.023 > 0.014. The "5×"
   sentence remains vulnerable to a spin accusation.
3. **W4 (moderate — teaser's mAP tie is architectural; AMOTA separates the
   arms more).** Untouched, though the new §3.3 audit softens it slightly
   (the paper now demonstrates something AMOTA does not provide: a GT-free
   population-controlled audit).
4. **W6 (moderate — TAO/OVTrack open-vocab tracking evaluation not
   engaged).** Untouched; one related-work sentence would largely defuse it.
5. **W7 (minor — fusion contribution has no CI).** Untouched.
6. **W8 (minor — no qualitative open-vocab figure).** Untouched; spec exists
   (`figure_specs.md`), not rendered.
7. **New, small:** the E2 open-vocab replication now carries a disclosed
   provenance caveat (retired global-arm variant). Honest, but a reviewer may
   ask for the production-implementation rerun — a ~10-minute cache-replay
   PBS job — to remove the caveat entirely.

## Net assessment

M1 and M3 are done and integrated; per the review's own calculus the draft
moves from Borderline (5/10) to **weak accept territory**, with the headline
now *stronger* (+38% vs +24%) under the more defensible provenance. The
remaining gap to a solid accept is M2 (one third-party tracker replay), plus
the two cheap text fixes (W6 sentence, W3 rewording) and optionally the E2
production rerun.
