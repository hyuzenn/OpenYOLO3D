# E1 — MOT-metric comparison for OV-TCS validation (design + computability audit)

Date: 2026-07-12. Status: evaluator implemented (`method_scannet/streaming/
eval_mot_compare_outdoor.py`), unit-tested, smoke→gate→full via
`scripts/run_outdoor_mot_compare.pbs`. Results live in the run dir
(`results/<date>_outdoor_mot_compare_v*`), never here.

## 1. Purpose

The OV-TCS paper claims a GT-free temporal-consistency metric. The most likely
reviewer objection is: *nuScenes HAS GT track identities, so established MOT
metrics (HOTA, IDF1, AMOTA) are computable — where is the comparison?* E1
computes those metrics on the SAME tracker outputs (ego- vs global-frame
association, class-agnostic, γ CenterPoint cache replay) and measures where
OV-TCS agrees and disagrees with them. This is a truth-seeking validation, not
an advocacy exercise: disagreement is reported as found.

## 2. Metric computability audit

Data available from existing cached outputs (no new inference):

- **Per-frame predicted boxes with persistent track ids** — recoverable by
  CPU cache replay: the evaluator's associator assigns `global_id` per
  proposal; a runtime wrapper retains it as `tracking_id` on the emitted box
  dicts (the stock emission path drops it). Global-frame translations, sizes,
  rotations, velocities, per-box scores: all present.
- **GT tracks** — `sample_annotation.json` carries `instance_token` (a
  persistent GT identity) per annotation per keyframe; already loaded by
  `_load_meta()`. On disk, complete for all 150 val scenes.
- **Scores for recall sweeps** — native CenterPoint per-box scores retained
  through replay (`detection_score` → `tracking_score`).

| Metric | Computable? | Needs | Already in outputs? | How computed here |
|---|---|---|---|---|
| HOTA | YES | per-frame pred ids + GT ids + a similarity | ids via replay hook; GT on disk | own implementation of the TrackEval reference algorithm (nuScenes has no official HOTA); similarity `s = max(0, 1 − d/2 m)` BEV center distance; unit-tested against hand-computed cases |
| DetA | YES | same as HOTA | yes | same implementation (per-α detection Jaccard) |
| AssA | YES | same as HOTA | yes | same implementation (per-TP association Jaccard) |
| IDF1 | YES | per-frame pred ids + GT ids + gate | yes | `motmetrics` 1.4.0 (installed for E1), 2 m gate |
| ID Precision | YES | same | yes | `motmetrics` (`idp`) |
| ID Recall | YES | same | yes | `motmetrics` (`idr`) |
| AMOTA | YES (full val only) | tracking submission JSON w/ scores, full 150-scene split, 7 tracking classes | submission built by the evaluator | **official, unmodified** nuScenes devkit `TrackingEval` (`tracking_nips_2019`) |
| AMOTP | YES (full val only) | same | same | official devkit |
| MOTA | YES | per-frame ids + gate | yes | per-scene via `motmetrics`; dataset-level per-class via official devkit |
| MOTP | YES | matched-pair distances | yes | `motmetrics` (`motp`, meters); devkit at dataset level |

Everything on the candidate list is computable. Two structural caveats, stated
up front because they shape interpretation:

1. **Class scope.** The devkit tracking eval covers 7 classes (drops barrier,
   traffic_cone, construction_vehicle); OV-TCS was frozen on 10-class tracks.
   The per-scene HOTA/IDF1/MOTA here are therefore computed class-agnostically
   over the 10-class GT with the 2 m tracking gate (matching the class-agnostic
   association under test); AMOTA/AMOTP are the official 7-class numbers.
2. **Aggregation mismatch.** OV-TCS_C is a per-track mean; MOT metrics are
   detection-weighted per scene. Per-scene correlation is the common
   granularity; a per-track HOTA does not exist by construction.

## 3. Comparison design

- Arms: **ego** vs **global** association (the paper's flagship pair), baseline
  axis, class-agnostic, γ gravity cache — the frozen configuration bit-for-bit
  (OV-TCS side reuses `compute_variant_metrics()` unchanged).
- Detections are identical across arms; only track identity changes. So
  detection-dominated metrics (DetA, MOTA, AMOTA) should be ~flat, and
  association metrics (AssA, IDF1, IDS) carry the signal. This decomposition is
  itself the reviewer-facing test: OV-TCS claims to measure the association
  axis.
- Analyses: (a) system-level ordering (does every MOT metric rank global vs
  ego the way OV-TCS does?); (b) per-scene correlation (Pearson + Spearman,
  n=150 per arm); (c) **paired per-scene deltas** Δ(global−ego) of OV-TCS vs
  Δ HOTA/AssA/IDF1 — the strongest test, 150 matched pairs where everything
  but the association frame is held fixed; (d) top rank-divergence scenes for
  failure-mode analysis.
- Stop condition (written before launch): E1 is a measurement, not a gated
  method experiment — the full run is reported whichever way it comes out.
  The smoke gate is operational only: no crashes, OV-TCS_C(ego) <
  OV-TCS_C(global) on the 5-scene subset, all MOT values in valid ranges,
  submission format deserializes.

## 4. What E1 can and cannot establish

Can: convergent validity (OV-TCS tracks GT-based association quality where GT
exists) and divergence structure (what OV-TCS sees differently, e.g.
consistently-wrong-label tracks score high OV-TCS but do not fool HOTA's
detection term; conversely HOTA is blind to label flicker on a geometrically
stable track — OV-TCS's CSR term is the only one of the two that fires there).
Cannot: break metric–method circularity (same associator family builds the
tracks) or validate the open-vocab setting (native closed-set labels). Those
are E3/E2's jobs, respectively.
