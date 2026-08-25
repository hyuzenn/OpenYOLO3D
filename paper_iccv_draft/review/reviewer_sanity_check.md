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

## Resolved by the submission-readiness pass (2026-07-17, second pass)

- **W3 (major — "5×" sentence): RESOLVED (wording half).** The "5×"
  comparison is removed from §5.2; the section now states the honest effect
  size, points forward to §5.3's open discussion of stability-only beating
  the product on the flicker axis, and the two-axis argument carries the
  formulation claim. The structural half (stability-only wins the sole
  GT-anchored test) was already conceded openly in §5.3's honesty notes.
- **W6 (moderate — TAO/OVTrack): RESOLVED.** Related work now engages both:
  they broaden the taxonomy but still score against annotated GT
  trajectories; cited.
- **New pre-empt:** Tab. 3's CLEAR-MOT Frag column (≈constant across arms)
  is now explained in the caption as a different quantity from Tab. 1's
  GT-instance fragmentation, closing an apparent-contradiction attack on the
  −58% claim.
- **Stale-anchor fix:** Tab. 1's detection anchor was silently mixing
  evaluator eras (0.3407/0.3145 May-era vs 0.3408/0.3150 production). Anchor
  rows now match the production E4/fusion-grid runs; M31/M32 rows footnoted
  with their era. Same defense class as the 0.168 provenance note.
- **E2 production rerun: DONE (PBS 104665).** The §6.2 legacy-variant caveat
  is gone: with the production global arm the open-vocab replication reads
  0.216→0.272 (+26%), frag −35%, i.e. *stronger* than the legacy +6.8%; the
  ego arm reproduced bit-identically. The abstract's replication claim is
  upgraded from "its direction replicates" to "+26%".

## Still open (ranked by expected reviewer damage)

1. **W1 (major — self-referential headline / M2).** Unchanged and still the
   single biggest threat. Every headline number remains our metric on our two
   pipelines. The fix the review demands (replay ≥1 third-party nuScenes
   tracking submission through the OV-TCS harvester, CPU-only) is still the
   highest-ROI remaining experiment; the paper still only promises it as
   future work.
2. **W4 (moderate — teaser's mAP tie is architectural; AMOTA separates the
   arms more).** Untouched, though the new §3.3 audit softens it slightly
   (the paper now demonstrates something AMOTA does not provide: a GT-free
   population-controlled audit).
3. **W7 (minor — fusion contribution has no CI).** Untouched.
4. **W8 (minor — no qualitative open-vocab figure).** Untouched; spec exists
   (`figure_specs.md`), not rendered.
5. **W9 (minor — arm-partialed pooled correlation).** Still only partially
   addressed (both within-arm r disclosed; no partialed pooled r).

## Net assessment

With W2, W3, W5, W6 resolved and every number in the manuscript traced to a
production artifact, the draft sits solidly in **weak accept territory**; the
remaining gap to a solid accept is M2 (one third-party tracker replay). W4,
W7, W8, W9 are polish-level.
