# E4 — Associator sensitivity: static vs constant-velocity matcher (d = 2.0 m)

**Question (supervisor, 2026-08-25):** does the matched-budget association gain
hold when the associator is changed?

**Tested change:** the existing greedy nearest-centroid matcher plus
constant-velocity prediction from the detector's own velocity
(`--association-velocity`). Gate 2.0 m, gap tolerance 5, class handling and
greedy score order are unchanged. No released tracker and no class-wise gating
was implemented or tested, and nothing here is a claim about them.

**PASS criterion (fixed before the run):** confirmation − detection-budget-matched
threshold control, AssA bootstrap CI excluding zero with positive sign, in both
association frames. **Result: FAIL.**

| frame | matcher | ΔAssA (confirmation − control) | 95% CI | boxes (both arms) |
|---|---|---|---|---|
| sensor | static (canonical Tab. 1) | +0.0625 | [+0.0471, +0.0785] | 257,259 |
| sensor | velocity | **+0.0667** | [+0.0485, +0.0853] | 253,999 |
| world | static (canonical Tab. 1) | +0.0148 | [+0.0098, +0.0203] | 517,566 |
| world | velocity | **−0.0198** | [−0.0231, −0.0166] | 529,329 |

Arm values under the velocity matcher:

| frame | arm | tracks | AssA | HOTA | DetA | IDF1 |
|---|---|---|---|---|---|---|
| sensor | confirmation | 40,884 | 0.2659 | 0.2400 | 0.2189 | 0.2027 |
| sensor | control | 133,124 | 0.1992 | 0.2661 | 0.3598 | 0.2034 |
| world | confirmation | 67,774 | 0.5710 | 0.3460 | 0.2106 | 0.3586 |
| world | control | 134,453 | 0.5908 | 0.3513 | 0.2097 | 0.3694 |

Box budgets matched exactly in both frames. Wilcoxon p = 1.36e-05 (sensor),
1.88e-15 (world). Bootstrap SEED 20260718, 10,000 scene resamples, via
`scripts/e2_thrmatch_report.py` unchanged. The evaluation similarity is
clip(1 − d_xy/2.0) in every cell (`MATCH_DIST_M` in `scripts/e1_gt_metrics.py`),
independent of the associator.

## Interpretation

The observed temporal-selection effect is associator- and frame-sensitive: the
sensor-frame gain survives the tested association change, whereas the
world-frame effect reverses sign. Therefore the world-frame result should not be
presented as an associator-independent improvement.

Observation only, not tested as a mechanism: in the world frame both arms gain
AssA under the velocity matcher relative to Tab. 1 (confirmation 0.5293 → 0.5710,
control 0.5146 → 0.5908), the control by more.

## Regression: the flag is a no-op when off

Pre-change evaluator (clean worktree at e35a174) vs post-change evaluator, both
flagless, same 3 scenes, phase1 N=3 retrospective emission: `tracks.json`
byte-identical in both frames (ego `3bbf3b0d…`, world `eea162c2…`);
`metrics.json` differs only in `axis_walltime_s`. Unit tests:
`method_scannet/tests/test_velocity_associator.py`.

## Abandoned

A d = 4.0 m gate-width grid (static and velocity, world frame) was started and
abandoned on 2026-09-12: the user disk quota was exceeded before any usable cell
was produced. Its partial outputs were removed; no result exists for it.

## Provenance

- cache: `results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity_c1fix`, 6,019 files, sha256-identical to the 2026-09-07 per-file record (re-checked 2026-09-15)
- full run: PBS 123463, `scripts/run_e4_assoc_velocity.pbs`, 2026-09-12 20:27–22:29 KST, 150 scenes, CPU replay
- smoke gate: PBS 123454 (velocity) vs 123461 (`scripts/run_e4_assoc_static_ref.pbs`), 3 scenes: 4,025 vs 4,042 boxes, so the flag changes association
- regression: PBS 123477, `scripts/run_e4_regress_oldcode.pbs`
- stats: `scripts/stats.py` → `result.json`
- threshold picks: `threshold_ego.json`, `threshold_global.json`
- not versioned (large): `cells/*/axis_*/{tracks.json, e1_perscene.pkl, amota_*, nuscenes_eval/}`, smoke/staticref/oldcode_smoke cells, run logs
