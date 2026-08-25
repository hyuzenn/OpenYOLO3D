# Revision notes (v2, addressing review_iccv.md)

## Addressed with existing data (no new experiments)
- **W1 figures** — 4 figures wired in: fragmentation dose–response + factor
  decomposition (Fig. 1, from 2026-06-26 fragdecomp run), OV-TCS vs MOT
  per-scene scatter (Fig. 2, from E1), sensitivity sweep curves + bars
  (Fig. 3, from E4).
- **M3 CIs** — scene-level paired bootstrap on the headline ego→global delta:
  +0.049, 95% CI [0.045, 0.052], 96.7% of scenes improved (10^4 resamples,
  from E1 per_scene_metrics.csv). Added to §6.1.
- **W3 product-form claim** — reframed: necessity established under controlled
  injection; explicitly states stability-only agrees on the real method pair;
  claim reduced to "minimal form never directionally wrong on either observed
  failure mode."
- **W6/M5 gaming** — new §3.3 "Reporting rule and gameability" using the
  p=0.3 arm (OV-TCS ↑ while mAP ↓) and the 342k vs 267k n_tracks asymmetry;
  joint reporting rule (n_tracks + detection metric + fixed associator
  setting).
- **W7 related work** — STQ, VPQ, TETA positioned (all GT-tube/closed-vocab).
- **W9** — E2 mAP/NDS quoting convention now explicit (footnote b carried over).
- **W4/W8** — limitations extended: low-AMOTA regime, single-ecosystem scope,
  non-overlapping domain validations stated explicitly.

## NOT addressed (requires new experiments — flagged, not silently claimed)
- **M1/M2 (third-party / strong trackers):** replaying ≥1 published nuScenes
  tracker submission through the OV-TCS harvester (CPU-only, uses public
  submission JSONs) would break single-ecosystem circularity. Recommended
  before submission; not run here.
- **M6 (indoor sensitivity / outdoor correctness):** cross-domain overlap.
- **M7 (qualitative figure):** flicker-vs-fragmentation visualization of one
  scene; needs a rendering pass over cached labels.
- **E2 delta CI:** E2 outputs are aggregate-only; a per-scene CI needs a
  re-run of the E2 harvester with per-scene dumps (cheap CPU replay).
