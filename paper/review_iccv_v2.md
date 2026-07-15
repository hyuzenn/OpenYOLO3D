# ICCV Review (round 2) — "OV-TCS: Measuring Temporal Label Consistency in Streaming Open-Vocabulary 3D Perception" (revised draft)

**Recommendation: Borderline (5/10).**
The revision is real: the paper now has figures, confidence intervals, an
explicit gameability rule, a clean contributions/validation separation, and
unusually candid limitations. Presentation is no longer the problem. The
problem is now sharply visible *because* the presentation is good: every
headline number is the authors' own metric, scored on the authors' own two
pipelines, in a regime where tracking barely works — and the paper's own
reporting rule, applied to the paper's own flagship result, flags that result
as suspicious. A metric paper asking the community to adopt a new column must
clear a higher bar than internal consistency.

## What the revision fixed (credit where due)

- Fig. 1 communicates the motivation in seconds; Figs. 2–4 carry §5.
- Bootstrap CI on the headline delta; effect sizes framed without hiding.
- One primary + one secondary contribution, five validation properties —
  the taxonomy is now honest and explicit.
- §3.3 (gameability + joint reporting rule) is a genuinely good addition.
- Future work states plainly that external validation has not been done.

## Weaknesses

W1 (**major, fatal if unaddressed**). **Self-referential headline.** The
flagship claim — OV-TCS +24%, CI [0.045, 0.052], 96.7% of scenes — is the
proposed metric grading a change to the proposers' own system. The CI
quantifies *precision*, not *validity*: 10^4 bootstrap resamples of a
self-scored quantity cannot tell us the metric measures something real. The
only external anchors are the MOT agreements of §5.4, which come from
**n = 2 systems** (system-level ranking agreement on two arms is one bit of
evidence) in a regime where MOTA is *negative* and AMOTA ≤ 0.16. The authors
themselves identified the fix in their own future-work paragraph — replaying
third-party nuScenes tracking submissions through the harvester, CPU-only,
no GT needed to compute OV-TCS — and did not do it. Declining the cheap
version of the decisive experiment while writing that it is "the most
important next step" reads as avoidance.

W2 (**major**). **The paper's own §3.3 rule indicts its own headline.**
§3.3 says: "comparisons that raise OV-TCS while shrinking the scored
population … must be read as filtering, not improvement." The flagship
ego→global comparison raises OV-TCS by 24% while shrinking the scored track
population by 22% (341,663 → 266,623; Tab. 1 / Tab. 5). The paper never
runs its own test on its own result. A skeptic's model — global-frame
matching merges fragments, deleting exactly the short, flicker-prone tracks
that drag the mean down — is *also* consistent with every number shown. The
authors have the data to kill this: recompute the delta on a
population-controlled comparison (e.g., matched GT instances, or OV-TCS over
GT-matched tracks only, or length-stratified deltas). Without it, Fig. 1's
right two panels and the abstract's CI have a plausible deflationary reading.

W3 (**major**). **The one GT-anchored test says a simpler metric is better.**
On the only experiment where OV-TCS is scored against ground truth
(label correctness, §5.2/Tab. 2), stability-only (1−CSR) achieves
ΔR² = 0.023 vs the product's 0.014. The revised §5.2 sells "≈5× better than
track length" — a ratio of two small numbers, and a comparison against the
*weakest* alternative in the paper's own table rather than the strongest.
The product's advantage over stability-only rests entirely on *synthetic*
fragmentation (§5.3, conceded in the text). So the honest summary of the
evidence is: "on real data, the simpler metric predicts correctness better;
our form wins only under a corruption we injected ourselves." That is a
defensible design argument, but the 5× sentence is spin, and reviewers
punish spin in papers whose brand is honesty.

W4 (**moderate**). **The teaser's "surprise" is architectural, not
empirical.** mAP is bit-identical across association because, in this
tracking-by-detection pipeline, association provably cannot change the box
set. That is true of *every* tracking-by-detection system and surprises no
one: nobody claims mAP measures tracking. The real competitor in the
teaser's scenario is not mAP but AMOTA/HOTA — which *do* separate the two
arms (Tab. 4: AMOTA 0.034 vs 0.158, a far larger separation than OV-TCS's).
On nuScenes, the paper's own showcase dataset, a practitioner would simply
run the tracking benchmark. The teaser therefore demonstrates the gap on the
one domain where the gap does not exist. The setting that actually motivates
OV-TCS — ScanNet-style streams with no GT tracks, open vocabulary — has no
teaser, no qualitative figure, and no method-comparison result.

W5 (**moderate**). **Headline arithmetic mismatch.** The abstract and Fig. 1
say +24% (0.136 → 0.168, i.e., a system-level delta of +0.032), then quote a
"mean per-scene gain +0.049." Track-weighted and scene-weighted aggregates
differ by 50% and the text never reconciles them; a careful reviewer will
stall here and wonder which aggregation the metric's definition actually
prescribes (§3.2 says mean over tracks — then the CI is computed on a
different statistic than the headline).

W6 (**moderate**). **Missing engagement with open-vocabulary tracking
evaluation.** The related-work claim is "no metric measures temporal
consistency for open vocabularies." The TAO/OVTrack line evaluates
open-vocabulary multi-object tracking with TETA on large vocabularies; it is
GT-based, so the gap (GT-free) survives — but the paper cites TETA only as a
closed-vocab video metric and never mentions that open-vocabulary *tracking*
evaluation exists. As written, the framing overstates the vacuum, and any
reviewer from the video-tracking community will notice.

W7 (**minor**). **Secondary contribution is thin and un-quantified.** The
fusion result is +0.0013 mAP from special-casing two classes (bicycle,
motorcycle), with no uncertainty (nuScenes val mAP run-to-run and
class-subset variability is not discussed). Elevating this to a named
contribution in the intro invites a "contribution does not survive a CI"
attack. It is a nice worked example of axis-orthogonality; it is not
independently a contribution.

W8 (**minor**). The qualitative flicker-vs-fragmentation figure (round-1 M7)
is still absent from the paper; it exists only as an internal spec. The
open-vocabulary regime — the paper's raison d'être — is thus never *shown*,
only tabulated.

W9 (**minor**). §5.4's strongest number (per-scene r = 0.87 with HOTA) is
the ego arm alone; pooled is 0.78, and pooling two arms can inflate
correlation via the between-arm mean gap. Report within-arm correlations for
both arms and, if pooling, a pooled-within (arm-partialed) correlation.

## Per-section scores (1–10)

| Section | Round 1 | Now | Note |
|---|---|---|---|
| Abstract | 7 | 7 | tight; but headline carries W2/W5 |
| 1 Introduction | 7 | 8 | contributions/validation split is exemplary |
| 2 Related work | 5 | 6 | STQ/VPQ/TETA added; OV-tracking line missing (W6) |
| 3 Metric | 6 | 7 | §3.3 good — but apply it to yourselves (W2) |
| 4 Setup | 6 | 6 | unchanged; single ecosystem |
| 5.1 AP-blindness | 7 | 7 | figure helps; architectural triviality (W4) |
| 5.2 Beyond length | 6 | 6 | CI framing better; 5× sentence is spin (W3) |
| 5.3 Product form | 5 | 6 | honestly bounded now; still synthetic-only |
| 5.4 MOT agreement | 8 | 7 | scatter added; n=2 systems, pooling concern (W9) |
| 5.5 Sensitivity | 8 | 8 | still the best section |
| 6 Applications | 6 | 5 | W2 + W5 land here hardest |
| 7 Limitations | 8 | 8 | candid |
| 8 Conclusion | – | 8 | strong close; future-work honesty noted |
| Presentation | 3 | 7 | figures competent; OV regime never shown (W8) |

## Experiments that would change the verdict (decreasing importance)

M1. **Population-controlled headline** (W2): recompute the ego→global OV-TCS
delta on matched populations (GT-matched tracks, or length-stratified). All
data cached; CPU-only. If the delta survives, the paper's central result
becomes robust to its own gaming critique.
M2. **≥1 third-party nuScenes tracker** through the OV-TCS harvester (W1).
CPU-only on public submission files, by the authors' own account.
M3. **Reconcile the +0.032 vs +0.049 aggregation** (W5): one paragraph +
one CI on the track-weighted delta.
M4. An open-vocabulary qualitative figure (W8/W4): the spec already exists.
M5. Within-arm correlation table for §5.4 (W9).
M6. Either add a CI to the fusion delta or demote fusion from "contribution"
to "worked example" (W7).

## Verdict

Round 1 failed on presentation; this draft fails on *provenance of
evidence*. Everything now looks right, and nearly everything is
self-supplied. M1 and M3 are days of CPU work and would defuse the two most
damaging attacks; M2 breaks the circularity outright. With M1–M3 this is a
weak accept; with M2 as well it is a solid accept, because the sensitivity
and honesty infrastructure around the metric is already better than what
most accepted metric papers ship.
