# Revision log — M1/M3 audit integration (2026-07-17)

Source of truth: `results/2026-07-15_m1m3_headline_audit_v01/audit.{md,json}`,
E1 `variant_metrics_{ego,global}.json`, E4 baseline `metrics.json` (both arms),
E4 `ego_d2.0_a5_p0.3` run. One implementation everywhere: the production
evaluator (`_ovtcs`, all-track mean, singletons = 0), shared by E1 and E4.

## Numerical changes

| Quantity | Old | New | Source |
|---|---|---|---|
| Global-arm OV-TCS (headline, Tab. 1, teaser, §5.4 already had it) | 0.168 | **0.188** (0.1877) | E1 = E4 production harvester |
| Headline relative gain | +24% | **+38%** | 0.1877/0.1356 − 1 |
| Headline delta + CI | mean per-scene +0.049, CI [0.045, 0.052] | **pooled (track-weighted) +0.052, CI [0.049, 0.055]**, scene-cluster bootstrap, 10^4; scene-weighted +0.049 retained as labeled secondary; 96.7% of scenes improve (unchanged) | audit.json (P1) |
| Global track length (Tab. 1) | 3.24 | **3.86** | E1 variant metrics |
| Fragmentation ego/global (Tab. 1, teaser) | 10.70 / 4.49 | **10.64 / 4.45** (−58% unchanged) | E1 gt_fragmentation (9,690 GT instances) |
| CSR ego/global (Tab. 1, §6.1) | 0.718 / 0.580 | **0.507 / 0.457** | production `csr_mean` (all tracks, singletons = 0) — old values were the retired-variant run's convention |
| Ego-equal rows (fusion, M31, M32): Frag/CSR | 10.70 / 0.718 | 10.64 / 0.507 | must equal ego row by construction |
| Gaming example (§3.3) | 0.136→0.160, mAP 0.341→0.310 | same, **added** scored-population drop 341,663→73,184 | E4 ego p0.1/p0.3 runs (verified) |
| §5.4 per-scene correlations | HOTA r=0.87 (ego; pooled 0.78) | **ego 0.87, global 0.60, pooled 0.78** | E1 notes/correlations |
| New: GT-instance-paired audit | — | **+0.147, 95% CI [0.142, 0.152], 8,046 instances, 70.9% improved / 11.5% tied; per-instance frag 7.89→4.65** | audit.json (P2) |
| New: scored coverage | — | **82.4% → 87.3%** of 700,752 boxes (P2) | audit.json |
| New: length-standardized gain | — | **+14.7%** under ego's length distribution | audit.json |

## Modified sections

- **Abstract** — headline 38% (0.136→0.188), pooled +0.052 CI [0.049, 0.055];
  added one sentence: gain survives the anti-gaming audit incl. GT-paired
  +0.147 CI [0.142, 0.152]; open-vocab claim now "its direction replicates".
- **Fig. 1 (teaser) + caption** — middle panel 0.136 vs 0.188 (+38%), frag
  panel 10.64/4.45; caption CI → pooled +0.052 [0.049, 0.055]. Figure
  regenerated (`scripts/make_teaser_fig.py`).
- **§1 Introduction** — "+24%" → "+38%" in the metric-in-use paragraph.
- **§3.2 Definition** — fixed to match the implementation: system score =
  mean over **all** tracks; singletons (L=1) score exactly 0 by construction
  (L_norm=0) and are penalized, not dropped; CSR reporting convention stated
  (singletons contribute 0). Old text ("mean over tracks with L≥2") removed.
- **§3.3 Gameability** — threshold example strengthened with the track-count
  drop; reporting-rule sentence sharpened ("suspect until a
  population-controlled audit shows otherwise"); **new worked-example
  paragraph** applying the rule to the flagship comparison: box conservation
  (700,752 boxes per-sample-equal in 6,019 samples; 1,029,380 bit-identical
  full-population), scored coverage 82.4%→87.3%, within-stratum gains
  (L=2: 0.163 vs 0.138; +14.7% length-standardized), GT-instance-paired
  +0.147 [0.142, 0.152]. Population (7-class tracking subset) stated.
- **Tab. 1 + caption** — global row 0.188/3.86/4.45/0.457; ego-equal rows
  10.64/0.507; caption now defines OV-TCS/CSR aggregation (all tracks,
  singletons 0) and Frag (9,690 GT instances).
- **§5.4 MOT agreement** — within-arm correlations for both arms disclosed.
- **§6.1 Using the metric** — headline numbers updated; one consistent
  statistic (pooled +0.052 CI [0.049, 0.055], same all-track mean the metric
  defines) with scene-weighted +0.049 explicitly secondary; cross-ref to the
  §3.3 audit; **new provenance-note paragraph**: 0.168 came from a
  since-retired global-associator variant; production value 0.188 reproduced
  independently by E1 and E4; ego arm bit-identical; orderings/conclusions
  unchanged under both variants.
- **§6.2 Open-vocab replication** — provenance caveat added: E2 used the
  retired global-arm variant, so it is quoted as direction-level evidence
  only (numbers 0.216→0.231, +6.8% unchanged).
- **§7 Limitations** — gameability item now points to the demonstrated
  self-audit.

## Not changed (deliberately)

- E2 open-vocab magnitudes (would require a cache-replay rerun with the
  production global associator — cheap PBS job, flagged below, not run).
- §5.2 "5×" framing (review-v2 W3), TAO/OVTrack related work (W6), fusion CI
  (W7), qualitative OV figure (W8/M4) — outside the M1/M3 mandate; see
  reviewer sanity check.

## Verification

- `grep` sweep: no remaining 0.168 / +24% / 10.70 / 0.718 / 0.580 / 3.24 /
  4.49 / [0.045 outside the provenance note.
- pdflatex ×2: 0 errors, no undefined references/citations, 8 pages.
- Teaser PNG visually inspected: no label collisions, values 0.136/0.188,
  10.64/4.45.

---

# Submission-readiness pass (2026-07-17, second pass)

Full-manuscript consistency audit against production artifacts, plus
reviewer-proofing edits. Every statistic in the paper was re-verified against
its source artifact this pass (E1 `mot_compare_table.md` + `correlations.json`,
E4 `summary.md` + run metrics, fusion grid `grid_summary.json`, fragdecomp
`decomposition.json`, formulation `formulation_analysis.json`, M1/M3
`audit.json`; p-value recomputed exactly: 3.53e-21).

## Numerical changes

| Quantity | Old | New | Source |
|---|---|---|---|
| Detection anchor (Tab. 1 baseline/ego/global rows, §4 setup, §1, §6.3, teaser mAP panel) | 0.3407 / 0.3145 | **0.3408 / 0.3150** | E4 baseline metrics (0.34083/0.31503) = fusion-grid A0_native — the same production runs all OV-TCS values come from; the old values were the May-era evaluator revision |
| M31/M32 rows (Tab. 1) | unmarked 0.3407/0.3145 and 0.3198/0.3028 | same values, **new footnote $^e$** | these two rows exist only as May-era runs (`2026-05-21_outdoor_native_temporal_v01`, `outdoor_m32_sweep/dist_2.0`); footnote discloses the ≤5e-4 evaluator-revision offset |
| §6.2 open-vocab replication | 0.216→0.231 (+6.8%), legacy global arm, direction-level caveat | **0.216→0.272 (+26%)**, frag −35% (3.76→2.43), switch rate (L≥2) −0.041; nusc10 +25%/−35%; caveat removed; abstract strengthened ("it replicates … (+26%)") | E2 production rerun (PBS 104665, `results/2026-07-17_e2_openvocab_production_v01/`), production GlobalCentroidAssociator; ego arm bit-identical to the legacy-era run (0.2161/3.759) |

## Wording changes (no scientific claim altered)

- **§5.2**: removed the "exceeds by roughly 5×" comparison (review-v2 W3);
  replaced with a plain statement + forward-pointer to §5.3's stability-vs-
  product discussion. Also removed a duplicated ΔR²≈0.003.
- **§2 Related Work**: added TAO/OVTrack sentence (review-v2 W6) — they
  broaden the taxonomy but still require GT trajectories. Two new bibitems.
- **Tab. 3 (mot_system) caption**: clarified that its Frag column is
  CLEAR-MOT's interruption count (≈constant across arms), a different
  quantity from Tab. 1's GT-instance fragmentation — pre-empts an apparent
  contradiction with the −58% claim.
- **§5.1**: added "(mean over the L≥2 tracks the injection acts on)" to the
  0.476→0.438 sentence, so the population is explicit and consistent with the
  §3.2 all-track convention.

## Verified unchanged (spot-audit results)

- §3.3 audit numbers, Tab. 2 formulation ΔR² (all 6 values), Tab. 3 MOT
  values (all 14), §5.4 correlations (all 6 quoted r), §5.3 decomposition
  (4.28→2.53, 0.648→0.569, 0.731→0.769, −0.058/−0.037/+0.025), §5.5 CVs
  (0.027/0.220/0.167) and Spearman −0.018, gaming example (0.341→0.310,
  341,663→73,184), fusion per-class story, AMOTA range, IDS 2.8×.
- mAP spread 0.022 (0.3420−0.3198) and fusion delta wording unaffected.
- No supplement file exists; the manuscript is self-contained.
