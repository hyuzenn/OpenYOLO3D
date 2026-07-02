# Experiment Tracker — ScanNet200 val

Average metrics across 312 scenes. Higher is better.

| Date       | Run dir                                          | AP     | AP50   | AP25   | Notes                                       |
|------------|--------------------------------------------------|--------|--------|--------|---------------------------------------------|
| 2026-05-05 | `2026-05-05_scannet_eval_v02`                    | 0.2470 | 0.3180 | 0.3630 | baseline                                    |
| 2026-05-07 | `2026-05-07_scannet_eval_v01`                    | 0.2473 | 0.3175 | 0.3624 | baseline (re-run)                           |
| 2026-05-07 | `2026-05-07_scannet_method_21_only_v01`          | 0.2465 | 0.3176 | 0.3636 | METHOD_21 single-axis ablation              |
| 2026-05-07 | `2026-05-07_scannet_method_31_only_v01`          | 0.2442 | 0.3152 | 0.3606 | METHOD_31 only                              |
| 2026-05-07 | `2026-05-07_scannet_method_31_iou07_v01`         | 0.2468 | 0.3172 | 0.3626 | METHOD_31 iou=0.7 (near-no-op)              |
| 2026-05-07 | `2026-05-07_scannet_phase1_v02`                  | 0.2443 | 0.3157 | 0.3629 | Phase 1 (METHOD_21 + METHOD_31)             |
| 2026-05-08 | `2026-05-08_scannet_method_22_only_v02`          | 0.2188 | 0.2797 | 0.3189 | METHOD_22 only (raw class-name prompts)     |
| 2026-05-08 | `2026-05-08_scannet_method_32_only_v01`          | 0.2141 | 0.2751 | 0.3157 | METHOD_32 only                              |
| 2026-05-08 | `2026-05-08_scannet_method_22_v2_v01`            | 0.2192 | 0.2821 | 0.3230 | METHOD_22 + "a photo of a {class}" template |
| 2026-05-09 | `2026-05-09_scannet_phase2_v02`                  | 0.1698 | 0.2156 | 0.2505 | Phase 2 (METHOD_22 + METHOD_32 combined)    |

## Status summary
- **Phase 1 (METHOD_21 + METHOD_31)**: neutral / near-no-op — no clear win.
- **Phase 2 axes (METHOD_22, METHOD_32 individually)**: each regresses ~−0.03 AP.
  Prompt template tweak in METHOD_22_v2 does not recover the gap (+0.0004 AP);
  regression lives in the CLIP-image-feature re-classification logic itself, not
  in the text side.
- **Phase 2 combined (METHOD_22 + METHOD_32)**: AP 0.1698 (Δ −0.0775 vs baseline).
  Super-additive negative — worse than either axis alone (M22 −0.0285, M32 −0.0332;
  additive estimate −0.0617). Indicates the two regressions compound rather than
  cancel.

---

## Streaming evaluation (frequency=10, cached Mask3D, RunningInstanceLabeler)

312-scene ScanNet200 val. Streaming AP is computed on per-instance final-frame predictions; not directly comparable to non-streaming offline AP above. `lsc` = label switch count total. `ttc` = time-to-confirm mean (frames).

| Date       | Run dir                                                                          | Axis        | AP      | lsc total | ttc mean | Notes                                                            |
|------------|----------------------------------------------------------------------------------|-------------|---------|-----------|----------|------------------------------------------------------------------|
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | baseline    | 0.19560 | 23,385    | 6.709    | Part 3 reference                                                 |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | M11         | 0.19540 | 17,023    | 6.296    | FrameCountingGate N=3 — lsc −27 %, AP cost −0.0002              |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01`          | M12 (buggy) | 0.19540 | 17,023    | 6.296    | bit-identical to M11 (silent bug, see Task 1.4c)                |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_B_m21_m31_v01`                   | M21         | 0.19589 | 23,629    | 6.622    | WeightedVoting — lsc +1 %, near-null                            |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_B_m21_m31_v01`                   | M31         | 0.19522 | 23,359    | 6.712    | IoUMerger — null                                                |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_C_phase1_m21m31_v01`             | phase1      | 0.19507 | 17,525    | 6.345    | M11+M21+M31; benefit attributable to M11                        |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_C_phase1_m21m31_v01`             | M21+M31     | 0.19527 | 23,593    | 6.623    | phase1 vs this isolates M11 add-on                              |
| 2026-05-16 | `2026-05-15_streaming_ablation_core_temporal/pbs_D_m22m32_v01`                    | M22+M32     | 0.09979 | 103,420   | 6.633    | Phase 2 cascade — AP halves, lsc 4.4×                           |
| 2026-05-18 | `2026-05-18_streaming_ablation_m12_fixed_v01`                                    | M12 (fixed) | 0.19498 | 16,518    | 6.179    | BayesianGate silent-bug fix; slight strict-dominance over M11   |

### Streaming status summary
- **Phase 1 temporal stabilizer**: only M11 (or fixed M12) moves temporal metrics; M21 / M31 nulls.
- **M11 ≡ M12 was a silent bug** (Task 1.4c): BayesianGate.gate() missed the decay branch. Post-fix M12 reaches lsc −29 % / ttc −8 % at AP cost −0.0006.
- **Phase 2 cascade (M22+M32)**: dual failure (AP −49 %, lsc +342 %), no rescue path.
- **§5.2 framing**: "any monotonic confirmation gate at ≈ 3 visible frames" — both M11 (counting) and fixed M12 (Bayesian) qualify; difference is hyperparameter tuning, not structural.

### M11 N sensitivity sweep — Task 1.5 (paper §7.5)

50-scene subset (`seed=42`, scene0011_00 excluded — Part 3 outlier). NOT directly comparable to 312-scene rows above; subset has slightly higher AP base.
Run dir: `2026-05-19_m11_n_sweep_v02/`.

| N | AP      | lsc total | lsc mean | ttc mean | ttc n_inst | Notes                                                     |
|--:|--------:|----------:|---------:|---------:|-----------:|-----------------------------------------------------------|
| 2 | 0.21533 | 3,201     | 64.02    | 6.347    | 1,627      | bit-identical AP to N=3                                  |
| 3 | 0.21533 | 2,776     | 55.52    | 6.141    | 1,618      | current production default                               |
| 4 | 0.21507 | 2,432     | 48.64    | 6.020    | 1,611      | weak strict-dominance over N=3 (every temporal metric ↓) |
| 5 | 0.21467 | 2,186     | 43.72    | 6.263    | 1,598      | ttc starts climbing                                      |
| 7 | 0.21317 | 1,814     | 36.28    | 9.200    | 1,557      | 41 instances dropped from confirmation; AP −1 %          |

Sweet spot N=4 on this subset (same AP within 1.2e-3 vs N=3, lsc −12 %, ttc −2 %). N=3 retained as production default for direct comparability with Part 3; N=4 reported as no-cost alternative in §7.5. 312-scene re-validation of N=4 strict-dominance is the natural follow-up.

### OV-TCS diagnostic — Open-Vocabulary Temporal Consistency Score (2026-06-11)

Run dir: `2026-06-11_outdoor_ov_tcs_v01/`. Probe `diagnosis/outdoor_ov_tcs_probe.py`
(evidence-only). γ cache, full val 150, geometry-only ClassAgnosticAssociator
(2.0 m, max_age 5). 341,663 tracks (100,291 singletons), same track corpus as
`2026-06-10_outdoor_temporal_consistency_v01`.

Per-track metrics (L, unique, entropy H, dominant-ratio DR, switch-rate CSR) with
distributions, corr(L,H)/corr(L,CSR), per-class stats, entropy & DR histograms.
Three score formulations compared on stability / dynamic range / noise sensitivity:

| | A=L·(1−Hₙ) | B=L·DR | C=L·(1−CSR) |
|---|--:|--:|--:|
| std | 0.218 | 0.213 | 0.224 |
| IQR | 0.470 | 0.444 | 0.250 |
| hist-entropy(norm) | 0.635 | **0.706** | 0.446 |
| Cohen's d (1-class vs multi) | 1.737 | 1.856 | **2.383** |
| noise −slope | 0.101 | 0.112 | **0.175** |

corr(L,H) Spearman +0.727 (longer tracks accumulate more distinct labels);
corr(L,CSR) Spearman −0.325. Rank agreement A–B +0.973. **B** = best-balanced
(widest spread + smooth), **C** = sharpest clean/noisy separation but bimodal
(CSR≈{0,1}), **A** = middle. No detector edit / mAP claim.

### GT-anchored track fragmentation (2026-06-11)

Run dir: `2026-06-11_outdoor_gt_fragmentation_v01/`. Probe
`diagnosis/outdoor_gt_fragmentation_probe.py` (evidence-only). γ cache, full val
150. Predicted tracks = geometry-only ClassAgnosticAssociator (2.0 m, max_age 5);
GT = nuScenes-10 instances (instance_token); GT-centric nearest-proposal match
within 2.0 m (global BEV).

10,713 GT instances (90.5% ever detected). **GT lifetime mean 17.5 frames**, but
**matched predicted-track lifetime mean 4.7** → heavy id-fragmentation:
**10.6 fragments / detected GT**, **93.7% of detected GT split into ≥2 tracks**.
Mean fragment segment 1.33 frames → merged (defrag) 14.2 frames = **×10.6 uplift**,
ceiling 18.3 (coverage 0.784). Fragmentation is uniform across all 10 classes
(91–97% multi-track). This is the upstream cause of the short tracks
(mean L=3.0) that starve OV-TCS — the associator's per-frame nearest-match + same
geometry-only gate breaks one GT into ~10 id segments.

### Temporal sampling density vs fragmentation (2026-06-11)

Run dir: `2026-06-11_outdoor_temporal_sampling_v01/`. Probe
`diagnosis/outdoor_temporal_sampling_probe.py` (evidence-only). γ cache, full val
150. Question: is the 10.6 fragments/GT caused by sparse keyframe-only sampling
(object motion exceeds the 2.0 m gate between frames) or by the associator?

**Sampling = keyframes only** (sample['next'] chain; sweeps never visited),
**median gap 0.500 s (2 Hz)**. GT inter-frame displacement: **median 0.038 m**,
p90 2.19 m, **only 10.46% of transitions exceed the 2.0 m gate** (concentrated in
bus/car/truck/motorcycle; pedestrian/barrier/cone/trailer ~0%).

Motion decomposition: observed frag mean **10.64**, motion-floor (1+#disp>gate)
**2.78**, **excess (associator-attributable) 7.86**. Of 93,405 observed id-breaks
only **18.5% are motion-forced; 81.5% are associator**. Static control
(barrier+traffic_cone, immobile): observed frag **10.56**, excess **9.56**.
Smoking gun — id-break rate over consecutive *covered* frames at **0–0.25 m
displacement = 75.5%** (and 61–82% across all bins, flat vs displacement). Verdict:
sparse sampling is a minor contributor (~18%); **fragmentation is overwhelmingly
an associator/proposal-stability failure, not a temporal-resolution problem.**
Sweeps would not fix it. No detector edit / mAP claim.

### Associator-design ablation — what recovers GT lifetime → track (2026-06-12)

Run dir: `2026-06-12_outdoor_associator_ablation_v01/`. Probe
`diagnosis/outdoor_associator_ablation_probe.py` (evidence-only). γ cache, full
val 150, 10,713 GT instances, GT lifetime mean **17.50** frames. All 12 associator
variants run in ONE data pass; fragments = distinct predicted ids per GT instance
(GT-centric global-BEV match, 2.0 m). BASELINE (ego/greedy/static/age5) = **10.70**
reproduces production `ClassAgnosticAssociator`. cov_frames 14.20 is constant
across variants (coverage is id-independent; only fragmentation differs).

**Root cause = frame convention, not assignment/motion.** The production
associator matches in the **ego frame** with no ego-motion compensation, so
stationary objects flow out of the 2 m gate as the ego car moves. Single-knob Δ
vs baseline:
- **frame ego→global (ego-motion comp): 10.70 → 4.49 = +58.1%** (dominant)
- assignment greedy→Hungarian (ego): +0.8% (negligible)
- motion static→const-velocity (ego): +11.2% (partially absorbs ego flow)

Stacking on global frame adds nothing and often hurts (over-engineering):
global+Hungarian 4.50, global+motion 5.04 (worse), global+Hung+motion 5.36.
Persistence sweep max_age {2,5,10,20} = {4.81,4.49,4.46,4.47}: optimum ~10, only
~3% beyond age5. **Best = global/greedy/static/age10 = 4.46 (+58.3%)** → mean
fragment-segment length 1.33 → 3.18 frames (×2.4 track-length uplift) from the
frame fix alone. Residual ~4.5 fragments is detection-gap-limited
(redetection after >max_age miss forces a new id), not associator-fixable.
No detector edit / mAP claim.

---

## OV-TCS head-to-head: baseline (ego) vs global associator (2026-06-12)

Probe `diagnosis/outdoor_ovtcs_assoc_compare_probe.py` (evidence-only). γ cache,
full val (150 scenes), 1,029,380 nuScenes-10 proposals. Both associators run in
ONE pass; gate 2.0 m, max_age 5, greedy/static — only the matching FRAME differs
(ego vs global). Same OV-TCS A/B/C formulations as `outdoor_ov_tcs_probe.py`.
Result dir `results/2026-06-12_outdoor_ovtcs_assoc_compare_v01/`.

Track population: tracks base 341,663 → glob 317,350 (−24k, fewer because ego
frame spuriously chains FP noise at fixed relative positions); mean length
3.01 → 3.24; **singleton frac 0.294 → 0.440 (+0.15)** — global REFUSES those bad
ego merges, leaving honest singletons.

**The metric splits by how you weight.**
- OV-TCS over ALL tracks (singletons score 0 — penalises global's extra
  singletons): A 0.301→0.263 (−12.7%), B 0.260→0.241 (−7.4%), C 0.136→0.168
  (+24.3%). Mixed: A/B down, C up.
- OV-TCS detection-weighted (Σ L·score/Σ L, length-fair, what a random detection
  experiences): **A 0.418→0.444 (+6.3%), B 0.386→0.431 (+11.9%), C 0.255→0.355
  (+39.1%)** — global wins on all three.
- Per-frame label PURITY on multi-frame tracks (L≥2): dominant ratio
  0.577→0.630, **class-switch rate 0.718→0.580 (−0.14)**, entropy 1.064→1.001,
  frac single-class 0.146→0.204. Global tracks are unambiguously cleaner.

Verdict: the global frame produces FEWER, PURER, longer real tracks but more
honest singletons. Track-count-weighted OV-TCS_A/B drop (singleton dilution);
detection-weighted OV-TCS and all per-frame purity metrics rise. The naive
per-track mean is the wrong aggregation — it rewards the ego frame for chaining
false positives. Use detection-weighted OV-TCS (or restrict to GT-matched
tracks) for a fair comparison; under it, global is strictly better, most on
C (switch-rate-based). No detector edit / mAP claim.

## OV-TCS metric-validity — synthetic fragmentation sweep (2026-06-12)

Probe `diagnosis/outdoor_ovtcs_fragmentation_probe.py`. Result dir
`results/2026-06-12_outdoor_ovtcs_fragmentation_v01/`. Does OV-TCS actually move
when tracks get worse? Inject the canonical MOT failure — fragmentation / ID
switches — and watch the score. Each internal track link is cut iid w.p.
p ∈ {0,.1,.2,.3,.5} (detections conserved; #fragments ego 342k→685k at p=.5),
5 seeds averaged, both associators in one pass. Complementary to the base
probe's LABEL-noise sweep. p=0 reproduces the assoc-compare means exactly
(ego 0.301/0.260/0.136, glob 0.263/0.241/0.168) — correctness check.

**Sensitivity curve (ego, all-fragments mean A/B/C):**
p=0 .301/.260/.136 → .1 .272/.234/.126 → .2 .240/.207/.113 →
.3 .208/.179/.099 → .5 .145/.125/.069. Global degrades the same shape but
flatter (starts lower-fragmentation).

**Monotonicity: ✓ all A/B/C strictly decrease with p, both associators** — the
core validity claim holds (no non-monotone violation).

**Effect size @p=0.5 (ego):** A |d|0.73 AUROC .729 (−slope .312), B |d|0.67
AUROC .737 (−slope .272), C |d|0.34 AUROC .636 (−slope .135).

**Which formulation is most fragmentation-sensitive:** by effect size /
absolute slope **OV-TCS_A ≳ B ≫ C**. C (1−CSR) is ~half as sensitive — short
fragments can keep or even improve switch-rate, so it under-reacts to ID
breaks. CAUTION: *relative* degradation is near-uniform (A 52% / B 52% / C 49%)
because all three multiply the shared `L_norm=1−1/L`, which fragmentation
collapses; rank formulations by absolute slope / Cohen's d, not % drop.

**Ego vs global under fragmentation:** ego is MORE sensitive (steeper slope,
larger |d|) — it starts with longer ego-chained tracks so has more length to
lose; global's tracks are already shorter/cleaner (lower headroom), so its
curve is flatter. Both stay monotone. No detector edit / mAP claim.

---

## Global associator as a method variant — 4-metric ablation (2026-06-12)

**Dir:** `results/2026-06-12_ablation_global_associator_v01/` (ablation.json, notes.md,
cells/<frame>_<axis>/{metrics.json,nuscenes_eval}). Driver
`method_scannet/streaming/eval_global_assoc_variant.py`; new
`GlobalCentroidAssociator` in `nuscenes_native_evaluator.py`
(`--association-frame global`, `--collect-track-metrics`). Cache-only
(`outdoor_native_temporal_cpcache_thr000_single_gravity`), full val 150 scenes,
1.029M nuScenes-10 proposals. Ran in the coss_agpu A100 Singularity container
(util node kills CPU-heavy Python).

**Question.** Connect the global (ego-motion-compensated) associator to the real
streaming pipeline and measure OV-TCS, GT-fragmentation, track length, and mAP
in ONE pass — is "OV-TCS improves while mAP barely moves" / "OV-TCS and
fragmentation improve together" true?

**Design.** Single knob vs baseline = matching frame (ego→global). BOTH
associators class-agnostic (greedy / static / gate 2.0 m / max_age 5), so frame
is the only variable and the ego row reproduces the validated assoc-compare
numbers. Two axes: `baseline` (raw associator; emitted boxes carry native
class+score per proposal, track id never reaches the devkit → mAP is
associator-INVARIANT by construction) and `phase1` (M11 frame-counting gate +
M21 + M31; M11 keeps/drops boxes by track age, so track structure DOES feed mAP).

**Ablation table (full val):**

| frame | axis | mAP | NDS | OV-TCS_A | OV-TCS_B | OV-TCS_C | trk_len | GT_frag/GT | n_tracks |
|---|---|---|---|---|---|---|---|---|---|
| ego    | baseline | 0.3408 | 0.3150 | 0.3009 | 0.2598 | 0.1356 | 3.01 | 10.639 | 341,663 |
| ego    | phase1   | 0.1425 | 0.2129 | 0.3009 | 0.2598 | 0.1356 | 3.01 | 10.639 | 341,663 |
| global | baseline | 0.3408 | 0.3150 | 0.3248 | 0.2940 | 0.1877 | 3.86 | 4.455  | 266,623 |
| global | phase1   | 0.2569 | 0.2989 | 0.3248 | 0.2940 | 0.1877 | 3.86 | 4.455  | 266,623 |

Δ(global−ego): baseline mAP **+0.0000** / NDS +0.0000; phase1 mAP **+0.1144** /
NDS +0.0860. OV-TCS A +0.0239 / B +0.0342 / C +0.0521; track_len +0.85;
GT_frag −6.18 (−58%).

**Validation anchors (all exact):** ego baseline mAP 0.3408 = γ-fixed 0.3407;
ego OV-TCS 0.301/0.260/0.136 = published assoc-compare ego; ego GT_frag 10.639 =
ablation baseline 10.64; detections conserved (ego 341663×3.01 ≈ global
266623×3.86 ≈ 1.029M). Global GT_frag 4.455 reproduces the documented global
target (10.7→4.5). [Global n_tracks differs from the standalone diagnosis
probe's 317,350 — tie-break / live-vs-snapshot impl details between two matcher
implementations; both land at ~1.029M detections and GT_frag ~4.5, so
conclusions are unaffected.]

**Findings.**
1. **Baseline axis: mAP and NDS bit-identical (Δ=0.0000)** while the global
   associator improves OV-TCS on all three formulations (A +8%, B +13%, **C
   +38%**), lengthens tracks (3.01→3.86), and **more than halves GT-anchored
   fragmentation (10.64→4.46)**. → "OV-TCS improves, mAP unchanged" holds in the
   strongest form (exactly unchanged, by construction), and "OV-TCS and
   fragmentation improve together" holds (OV-TCS up, fragments −58%).
2. **Phase1 axis: global mAP +0.1144 (0.143→0.257, +80% rel).** When the
   temporal layer consumes the tracks, ego's fragmented short tracks are
   hammered by the M11 ≥3-frame gate (true detections suppressed → mAP
   collapses to 0.14); global's longer, less-fragmented tracks survive the gate,
   recovering mAP to 0.26. Better continuity directly RESCUES mAP under temporal
   gating. (Both phase1 < baseline 0.3408: the native-label temporal layer is
   net-lossy here — M21 relabel is a near no-op since native labels are stable —
   but global loses far less, −0.084 vs −0.198.)
3. OV-TCS_C gains most from the frame switch (+38%): it is built on 1−CSR, and
   the global frame's continuity most directly suppresses class-switch rate.

**Verdict.** The global associator is a free temporal-consistency upgrade in the
raw pipeline (OV-TCS↑, fragmentation↓ by half, zero mAP cost) and a net mAP WIN
once a track-age temporal gate is active. EVIDENCE-ONLY w.r.t. the detector: no
CenterPoint output modified; only the association frame (and optional temporal
layer) changed.

## Outdoor proposal ceiling — is recall the dominant bottleneck? (2026-06-13)
`results/2026-06-13_outdoor_proposal_ceiling_v01/` — consolidation of measured
results (no rerun). nuScenes val-150, γ gravity cache, CenterPoint proposals
before any temporal module. GT=187,528 (mAP anchor 0.3408).

- GT coverage: only **80.2%** have any proposal within 4m → **19.8% of GT have NO
  box** (hard floor). IoU3D recall 0.596 @0.25, **0.382 @0.5**, 0.163 @0.7.
- Missed-GT is a **range cliff**: <1% miss in near field, 66% @50-80m, **100%
  >80m**; 76% of all misses are beyond 50m. Skews far + large-volume.
- Stage levers (devkit counterfactuals): localization +0.054, classification
  +0.021, temporal relabel ≤+0.021, tracking fragmentation ~0 mAP. Native→dedup-
  oracle gap 0.207 = ~74% coverage+class / 26% localization.

**Answer: A) proposal generation is the dominant bottleneck.** Association/
tracking (93.7% split) and temporal consistency (62% switch) are severe as
tracking statistics but translate to ~0–0.02 mAP; the 19.8% no-box floor and the
>80m recall=0 cliff are unrecoverable downstream.

## Hybrid Proposal feasibility — CenterPoint as pure geometry generator (2026-06-13)
`results/2026-06-13_outdoor_hybrid_proposal_v02/` — feasibility validation (no accuracy
optimisation). Pipeline: CenterPoint 3D box (CLASS DISCARDED) → project to 6 cams → 2D ROI →
YOLO-World on ROI → open-vocab label. 30 val-150 samples, 5782 geometry proposals (score≥0.1).

- **(1) ROI projection success: 100.0% (5782/5782)**, all range bins ≤80m = 1.0; full 360°
  camera use; 5781/5782 boxes fully in front. 6-cam rig covers every LiDAR-range box.
- **(2) Runtime:** projection 0.41 ms/box, crop 5.3 ms, YOLO 52.2 ms/ROI. Per-ROI fan-out
  ≈10 s/sample vs ~0.35 s for one 6-image pass → naive crop-and-infer ≈ **28-30× overhead**.
- **(3) Labels:** 87.9% labeled; open-vocab "stuff" (building+tree+pole) = 36%; agreement with
  discarded CenterPoint class only **18.9%**; low YOLO scores on far/small ROIs.
- **(4) Viz:** 36 imgs, geometry correct; near objects label well (pedestrian 0.80), far/cluttered
  collapse to generic stuff.

**Verdict: YES — mechanically feasible (geometry→ROI→label on 100% of proposals).** Projection is
not a constraint. Two fixable, non-geometric issues: per-ROI cost (~30×; fix = one detection pass
per image + ROI↔detection IoU assignment) and noisy crop-only labels (fix = restrict vocab to the
10 scored classes, context-pad, score-gate). No downstream 3D mAP computed (by design).
