# SemWorld-3D

**OV-TCS — A Temporal-Consistency Metric for Streaming Open-Vocabulary 3D Perception**

A research project building on [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). The headline result is **OV-TCS**, a
track-level temporal-consistency metric for streaming open-vocabulary 3D perception: per-frame AP/NDS is blind to the
temporal degradation (label flicker, track fragmentation) that determines whether an online semantic map is usable, and
OV-TCS measures it without ground-truth track IDs. The streaming evaluation harness and the indoor/outdoor temporal layer
(below) are the infrastructure on which OV-TCS and a companion class-aware label-fusion method were developed and validated.

- **Indoor (ScanNet200)**: Mask3D proposals → temporal layer → per-instance OV-TCS predictive-validity study
- **Outdoor (nuScenes)**: LiDAR clustering proposals (β1 + γ hybrid) → associator tracks → OV-TCS corruption / method separation

> 📌 **Status (2026-07-02)**
> - **Streaming harness + indoor 6-axis ablation** ✅ COMPLETE (May 2026; see §3–§5, kept as infrastructure/prior findings)
> - **Outdoor diagnosis (7-stage, β1+γ hybrid, β baseline)** ✅ COMPLETE (May 2026; see §6)
> - **OV-TCS metric + two-axis formulation study** ✅ COMPLETE on banked data (see below; full write-up in [`docs/ovtcs_paper_draft.md`](docs/ovtcs_paper_draft.md) and [`docs/ovtcs_paper_storyline.md`](docs/ovtcs_paper_storyline.md))
> - **Paper drafting** 🚧 IN PROGRESS — Abstract / Introduction / Related Work / Method drafted from confirmed evidence
> - **ScanNet200 train(1201)+val full M22 run** 🚧 IN PROGRESS on PBS — *no results assumed anywhere; all indoor OV-TCS numbers below are from the banked val312 study*
>
> Targeting CVPR 2027 (Nov 13–15, 2026 deadline window).

---

## 1. Contribution

Two contributions that justify each other:

1. **OV-TCS — a track-level temporal-consistency metric.** Native per-frame open-vocabulary labels are grouped by
   associator track and scored, *without* ground-truth track IDs, as the product of two factors:
   `OV-TCS_C = L_norm · (1 − CSR)`, where `L_norm = 1 − 1/L` is track integrity and `1 − CSR` (CSR = switches/(L−1)) is
   semantic stability. It sees a degradation axis AP is blind to, carries information beyond track length, and separates
   real methods AP collapses to a tie.
2. **Class-aware label fusion** for open-vocabulary streaming — a class/source-aware 2D→3D label correction whose benefit
   surfaces in temporal/label quality (OV-TCS) rather than in closed-vocabulary AP.

The registration / label-consistency / spatial-merge temporal layer (§3) and the streaming harness (§4) are the
**infrastructure** on which these were developed: the indoor Mask3D tracks and outdoor associator tracks are what OV-TCS
is computed over. Definition source: `method_scannet/streaming/nuscenes_native_evaluator.py:988-1002`.

## 2. OV-TCS — headline evidence (banked, verified)

> Full write-up: [`docs/ovtcs_paper_draft.md`](docs/ovtcs_paper_draft.md). Numbers traced to `results/` and verified
> against the metric source. **The pending ScanNet200 train(1201)+val run is not used here; indoor numbers are from the
> banked val312 study (n=6202).**

- **AP is blind.** Injecting fragmentation that only splits tracks *after* detection (AP-scored proposals held fixed)
  leaves AP flat while OV-TCS drops monotonically; a positive control (shrinking association max-age) moves both — so
  OV-TCS is neither inert nor a restatement of AP.
- **Beyond track length.** On real indoor per-instance data (ScanNet200 val, n=6202, target = label correctness), OV-TCS
  adds significant explanatory power over track length: partial Pearson r = 0.12 (p = 3.5e-21); nested F-test ΔR² = 0.014,
  F = 89.87, p = 3.5e-21. Track length alone is inert (ΔR² ≈ 0.003, negative partial correlation). Reported honestly: the
  signal is instance-level — per-scene aggregation does **not** pass (n = 312, partial r = −0.03, p = 0.61).
- **Why the product form (two-axis).** Each factor exclusively owns one failure mode. *Flicker* is owned by `1 − CSR`
  (formulation ablation on n=6202: stability-only ΔR² 0.023 > product 0.014 > length 0.003). *Fragmentation* is owned by
  `L_norm`: a per-instance sweep on nuScenes (150 val, ~19k–22k tracks/level) drops mean OV-TCS 0.476 → 0.438, carried
  entirely by `L_norm` (−0.058), while per-track stability `1 − CSR` *rises* 0.731 → 0.769 — so a stability-only metric
  would mis-report fragmentation as improvement. An additive (weighted-sum) form collapses to λ* = 0. The product is the
  minimal form responsive to both axes.
- **Separates real methods AP ties.** Switching association from ego- to global-frame raises OV-TCS_C 0.136 → 0.168
  (+24%) at near-flat AP (nuScenes 150 val).

**Why a metric, not aggregation (honest limitation).** Feeding a temporal signal into an EMA over the label stream did not
help — no-aggregation beat every EMA setting (OFF > k = 0.335 > k = 1.0). Temporal quality is to be *measured*, not hoped
for by averaging. Outdoor open-vocabulary AP is separately bottlenecked by localization (capped near the closed
CenterPoint anchor of 0.3407), which is exactly why the label-fusion method's benefit is read via OV-TCS rather than AP.

---

> The sections below (§3–§7) document the streaming infrastructure and the indoor/outdoor evidence the OV-TCS study is
> built on. They reflect the May 2026 "proposal-agnostic temporal layer" framing and are kept as prior findings.

## 3. Method overview (6 axes, equal emphasis)

Three orthogonal axes, each with a simple (Phase 1) and an advanced (Phase 2) variant. The streaming ablation (§5.4–5.7) identified the
operative subset; the table below is the original design space.

| Axis | Phase 1 (simple) | Phase 2 (advanced) | Streaming failure mode addressed |
|---|---|---|---|
| Registration | **METHOD_11** Frame-counting "wait-and-see" gate | **METHOD_12** Bayesian probability accumulation | False positives from single-frame noise |
| Label consistency | **METHOD_21** Weighted label vote (distance + center) | **METHOD_22** CLIP visual-feature EMA fusion | Label flicker across viewpoints |
| Spatial merge | **METHOD_31** 3D IoU same-class merging | **METHOD_32** Hungarian (semantic + distance) | Identity switches / over-segmentation |

Each method is wired into the streaming wrapper via an installer in `method_scannet/streaming/hooks_streaming.py`. Post-streaming
findings (§5.4): **M11 (and the bug-fixed M12) is the sole effective temporal stabilizer** at ~zero AP cost; M21/M31 are temporally
null; M22/M32 cascade negatively. Paper §5.2 framing collapses to "any monotonic confirmation gate at ≈ 3 visible frames."

## 4. Streaming evaluation protocol

The protocol is **Mask3D per-scene + frame-visible filtering + streaming 2D label fusion** (Task 1.1 design Option (다)).

| Decision | Choice |
|---|---|
| Mask3D scope | Once per scene, offline (cached in `results/2026-05-13_mask3d_cache/`) |
| Frame visibility | **D3**: instance is processed in frame *t* iff `visible_count(k, t) ≥ 1` vertex |
| "Confirmed" criterion | **C1, K=3**: an instance is committed when ≥ 3 consecutive frames agree on its class |
| Primary mAP | Incremental against *visible GT up to t* |
| Secondary mAP | Incremental against full-scene GT (recall ceiling at *t*) |
| Temporal metrics | ID switches per object, label switch count (lsc), time-to-confirm (ttc), mask-IoU |

OpenYOLO3D core is **not modified**. The harness uses the frame-level interfaces already present in `utils/utils_2d.py`
and `utils/utils_3d.py` and adds a `StreamingScanNetEvaluator` wrapper. Each method axis is applied via
attribute-injection hooks; uninstall restores baseline byte-for-byte.

## 5. ScanNet200 results

### 5.1 Offline 8-way ablation (May 2026, single-Mask3D-draw)

Average AP across 312 ScanNet200 val scenes (May-7 Mask3D draw).

| Config | AP | AP_50 | AP_25 | Δ AP vs baseline |
|---|---|---|---|---|
| Baseline (OpenYOLO3D) | **0.2473** | 0.3175 | 0.3624 | — |
| METHOD_21 only | 0.2465 | 0.3176 | 0.3636 | −0.0008 |
| METHOD_31 only | 0.2442 | 0.3152 | 0.3606 | −0.0031 |
| METHOD_31 (IoU=0.7, near-no-op) | 0.2468 | 0.3172 | 0.3626 | −0.0005 |
| Phase 1 (M21 + M31) | 0.2443 | 0.3157 | 0.3629 | −0.0029 |
| METHOD_22 only ("{class}" prompt) | 0.2188 | 0.2797 | 0.3189 | −0.0285 |
| METHOD_22 only ("a photo of a {class}") | 0.2192 | 0.2821 | 0.3230 | −0.0281 |
| METHOD_32 only | 0.2141 | 0.2751 | 0.3157 | −0.0332 |
| Phase 2 (M22 + M32) | 0.1698 | 0.2156 | 0.2505 | **−0.0775** (super-additive negative) |

**Reading**: M21 / M31 sit on the offline noise floor (|Δ AP| ≤ 0.003, indistinguishable from baseline). M22 / M32 are
each clear single-axis negatives (≈ −0.03 AP). Their combination is super-additive negative (−0.0775 > additive estimate
−0.0617). Per §2, this is the *limit* of offline static measurement — not the method's verdict.

### 5.2 Streaming protocol validation (Step 1, May 13)

Both pipelines consume the same 312 cached Mask3D outputs (`results/2026-05-13_mask3d_cache/`), eliminating Mask3D
run-to-run non-determinism (Task 1.2c diagnosis, H6 confirmed).

| Metric | Offline (cached) | Streaming (cached) | Δ |
|---|---|---|---|
| AP | **0.195610** | **0.195595** | **−0.000015** |
| AP_50 | 0.253299 | 0.253275 | −0.000024 |
| AP_25 | 0.296141 | 0.296111 | −0.000030 |

Sanity threshold `|Δ AP| ≤ 0.005`. Observed `|Δ AP| = 0.000015` — **PASS by 333× under budget**. The remaining ~10⁻⁵ gap
is pure CPU/CUDA float-ordering precision. The streaming protocol is algorithmically equivalent to offline once Mask3D
non-determinism is held constant.

**Streaming baseline reference**: cached streaming AP = **0.1956**. All streaming ablations (§5.4–5.7) compare against this number
(same Mask3D cache), not the 0.2473 May-7 offline number (different Mask3D draw).

### 5.3 METHOD_22 / METHOD_32 failure analysis (Step 3, May 13)

Analytical — no re-inference; uses only prompt embeddings + GT. Full data in `results/2026-05-13_step3_m22_m32_failures/`.

**M22 — CLIP narrow band CONFIRMED (stronger than original memo)**

| Statistic | v1 ("{class}") | v2 ("a photo of a {class}") |
|---|---|---|
| Off-diag cosine **mean** | **0.759** | 0.750 |
| Off-diag cosine median | 0.760 | 0.751 |
| Nearest-neighbour cosine mean | **0.889** | 0.894 |
| Nearest-neighbour cosine max | **0.984** | 0.985 |

Top confused pairs: *trash bin ↔ trash can* (0.984), *keyboard piano ↔ keyboard* (0.962),
*bathroom stall ↔ bathroom stall door* (0.956). The visual-feature score cannot reliably distinguish 200 ScanNet classes
because the class prompts sit in a narrow cone of CLIP space.

**M32 — multi-instance absorption CONFIRMED**

| Statistic | Value |
|---|---|
| Same-class instance pairs (val, all scenes) | 26 802 |
| Pairs with centroid distance ≤ 2 m | 10 074 |
| **Absorption ratio under M32 (2 m, class-aware)** | **37.6 %** |
| Classes with absorption ratio = 1.0 | **51 / 134** |

Over a third of same-class instance pairs in val sit within the 2 m Hungarian threshold; the class-aware merger collapses
them into single predictions. Consistent with the −0.0332 single-axis offline AP. Fix proposals are written up in `results/2026-05-13_step3_m22_m32_failures/fix_proposals.md`; **the streaming results in §5.4–5.7 made the fixes unnecessary** because the temporal-layer story re-localised to M11/M12.

### 5.4 Streaming 8-axis ablation (Task 1.4b Part 3, May 16–17)

Full 312-scene streaming ablation with proper temporal metrics instrumentation (Task 1.3 Part 2 instrumentation fix at commit 087bbad — see `docs/task_1_3_temporal_instrumentation_fix.md`). `lsc` = label switch count total per axis; `ttc` = time-to-confirm mean (frames to instance commit).

| Axis | Mean AP | Δ AP vs base | lsc total | Δ lsc % | ttc mean | Δ ttc % | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.19560 | — | 23,385 | — | 6.709 | — | reference |
| M11 (N=3) | 0.19540 | −0.00020 | 17,023 | −27.2 % | 6.296 | −6.2 % | **sole real Phase-1 stabilizer** |
| M12 (buggy) | 0.19540 | −0.00020 | 17,023 | −27.2 % | 6.296 | −6.2 % | bit-identical to M11 — silent bug (see §5.5) |
| M21 | 0.19589 | +0.00030 | 23,629 | +1.04 % | 6.622 | −1.3 % | label voting → near-null, mild "late correction" |
| M31 | 0.19522 | −0.00037 | 23,359 | −0.11 % | 6.712 | +0.04 % | IoU merger → fully null |
| phase1 (M11+M21+M31) | 0.19507 | −0.00053 | 17,525 | −25.1 % | 6.345 | −5.4 % | benefit attributable to M11 |
| M21+M31 | 0.19527 | −0.00033 | 23,593 | +0.89 % | 6.623 | −1.3 % | phase1 vs this isolates M11 add-on |
| M22+M32 | 0.09979 | **−0.09581** | 103,420 | **+342 %** | 6.633 | −1.1 % | Phase 2 cascade — AP halves, lsc 4.4× |

**Verdict (Part 3 §3.3)**: the May-12 advisor hypothesis that M21/M31 would deliver streaming temporal stability **failed at scale**. The surprise positive is M11 — −27 % label switches at near-zero AP cost. Phase 1 contribution is **frame-counting gating**, not weighted voting or IoU merge. M21/M31 drop from core ablation; M22+M32 kept as the §5.4 negative ablation.

### 5.5 BayesianGate silent-bug diagnosis + fix (Task 1.4c, May 18)

Bit-identical M11/M12 in §5.4 traced to a two-factor cause:

1. **Silent bug**: `BayesianGate.gate()` only invoked `_update(p, True)`; the bidirectional decay branch the docstring promised was dead code.
2. **Hyperparameter coincidence**: with defaults `prior=0.5, likelihood=0.8, fpr=0.2, threshold=0.95`, the monotonic-only update converged to threshold at *exactly* the 3rd visible observation — bit-equivalent to `FrameCountingGate(N=3)` on every visibility pattern.

Fix in `method_scannet/method_12_bayesian.py` (~5 LOC) restores the bidirectional update. 312-scene re-run after fix:

| Axis | Mean AP | Δ AP vs base | lsc total | Δ lsc % | ttc mean | Δ ttc % |
|---|---:|---:|---:|---:|---:|---:|
| M12 (fixed) | 0.19498 | −0.00062 | 16,518 | −29.4 % | 6.179 | −7.9 % |

Post-fix M12 is **distinguishable from M11** and *slightly* dominates it on every temporal metric — but the practical gap is < 0.5 % everywhere. Paper §5.2 framing: *any* monotonic confirmation gate at ≈ 3 visible frames qualifies; M11 (counting) vs M12 (Bayesian, fixed) is hyperparameter tuning, not structural.

Full diagnosis: `docs/task_1_4c_m11_m12_diagnosis.md`. 51/51 unit tests pass (47 prior + 4 new for the False decay branch).

### 5.6 M11 N sensitivity sweep (Task 1.5, 50-scene subset)

Sweep on a 50-scene subset (`seed=42`, scene0011_00 excluded as Part 3 outlier). Subset AP base is higher than full population; relative N comparisons remain meaningful.

| N | AP | lsc total | lsc mean | ttc mean | ttc n_inst |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.21533 | 3,201 | 64.02 | 6.347 | 1,627 |
| **3** (current default) | **0.21533** | **2,776** | **55.52** | **6.141** | **1,618** |
| 4 | 0.21507 | 2,432 | 48.64 | 6.020 | 1,611 |
| 5 | 0.21467 | 2,186 | 43.72 | 6.263 | 1,598 |
| 7 | 0.21317 | 1,814 | 36.28 | 9.200 | 1,557 |

N=2 and N=3 yield bit-identical AP because 250-frame streams confirm the same final set regardless of these N values; lsc/ttc capture timing differences. N=4 appears to weakly dominate N=3 on this subset; ttc V-shape minimum at N=4; cliff at N=7 (41 instances drop from confirmation entirely). 312-scene anchor below.

### 5.7 M11_N4 312-scene anchor (Task 1.6, May 19)

Validating the 50-scene N=4 sweet-spot claim on the full population.

| Axis | Mean AP | Δ AP vs base | lsc total | Δ lsc % | ttc mean | Δ ttc % | ttc p90 | ttc max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.19560 | — | 23,385 | — | 6.709 | — | 9.0 | 373 |
| M11 N=3 (Part 3) | 0.19540 | −0.00020 | 17,023 | −27.2 % | 6.296 | −6.2 % | 8.0 | 371 |
| M11 N=4 | 0.19500 | −0.00060 | 15,011 | **−35.8 %** | 6.384 | −4.8 % | 7.6 | 260 |

**Strict-dominance claim is NOT preserved at scale.** N=4 vs N=3 on 312 scenes: AP −0.00040, lsc −11.8 % (matches 50-scene), ttc mean **+1.4 % (sign-flipped from −2.0 %)**, ttc p90 −5.0 %, ttc max −29.9 %. The ttc-mean increase is a *composition* effect — 64 short-life instances that just-barely cleared N=3 (ttc=3) are excluded from N=4's confirmation pool entirely, pulling the mean upward; the tails still tighten. Third documented instance of "small subset over-states streaming-metric trends" (after Part 3's scene0011_00 and Task 1.4c's M12-fixed smoke).

**Production default: N=3.** N=4 reported in paper §7.5 as a label-stability-priority variant (−35.8 % lsc), not as strict dominance.

## 6. Outdoor extension (nuScenes)

> 📦 Full diagnosis archive: [`docs/nuscenes_diagnosis.md`](docs/nuscenes_diagnosis.md). The numbers below are preserved verbatim from May 2026 diagnosis on v1.0-mini; see `results/diagnosis_*/` (untracked) for raw aggregates, and `results/diagnosis_beta_baseline*/` (tracked) for the β baseline evidence.

### 6.1 Why outdoor breaks indoor pipelines

Three core assumptions of indoor open-vocab 3D pipelines fail on nuScenes, and one additional layer module (registration) becomes load-bearing:

1. **Lifting assumption** — OpenYOLO3D lifts 2D detections to 3D using dense per-pixel depth from RGBD. nuScenes provides 32-beam sparse LiDAR; a 2D box may contain 0–few projected LiDAR points, especially at distance. Median-depth lifting is unreliable.
2. **Static-scene assumption** — Indoor instance association assumes scene geometry stable between frames. nuScenes contains moving cars, pedestrians, cyclists; naive spatial association collapses on dynamic objects.
3. **Single-view assumption** — Indoor scenes use one forward-facing camera. nuScenes uses 6 surround cameras with overlapping FoV — the same object appears in multiple cameras at different viewpoints, scales, and quality.
4. **Registration module is essential outdoor** — On ScanNet (static, per-scene), M11/M12 can be skipped. On nuScenes (online streaming with ego-motion + dynamic objects), registration becomes load-bearing and is validated outdoor.

### 6.2 Preliminary diagnosis findings (May 2026)

#### 6.2.1 Setup-level (Tier 1 / 2 / 2-Ext, 100 samples)

Quantifies how the indoor lifting assumption breaks:

- **Per-pixel depth coverage**: 0.20 % (vs. dense RGBD indoor ≈ 100 %)
- **Detection-induced GT loss**: 44.8 % (GT objects with insufficient projected LiDAR support inside their 2D detection box)

The bottleneck is at proposal generation, not at refinement.

#### 6.2.2 7-stage proposal diagnosis (W1 → γ, 50 samples)

Each stage progressively relaxes/tightens the proposal source:

- **β1 (geometric clustering)**: 36.1 % — geometric sweet spot, best near-range
- **γ (CenterPoint, learned)**: 35.2 % — comparable overall, **opposite distance bias** (better far, worse near vs. β1)
- **Ceiling**: ~ 36 % under either single-source proposal

#### 6.2.3 α hybrid simulation

Motivated by the opposite biases, simulate a **distance-aware union**:

- **β1 far + γ near, threshold = 35 m**: **46.7 %** (+10.6 pt over the 36 % ceiling)

Empirical justification for adopting a β1+γ hybrid as the outdoor proposal source.

#### 6.2.4 β baseline — direct OpenYOLO3D application

Treat OpenYOLO3D as-is on nuScenes, no outdoor adaptation:

- **mAP = 0 %** — catastrophic outdoor failure
- Root cause: Mask3D is pretrained indoor and does not transfer to outdoor LiDAR distributions. No proposal mass to feed downstream stages.

The β baseline (and its v2 GT-bug-fix re-measurement, still mAP=0) is the evidence that **Mask3D cannot serve as the outdoor proposal source**, motivating the β1+γ hybrid substitution.

### 6.3 Step 4 implementation plan

Gated on Indoor Step 1–3 (now complete). Remaining:

1. **LiDAR clustering proposal module** — β1 + γ infrastructure (starting point: `diagnosis_beta1/`, `diagnosis_gamma/`, `diagnosis_step1/`).
2. **Distance-aware fusion stage** — implement the α simulation result (β1 far + γ near, threshold ≈ 35 m) as a runtime module, not a post-hoc simulator.
3. **Integration with the indoor-validated temporal layer** — import `method_scannet.streaming.*` and reuse M11 (or fixed M12). No fork.
4. **nuScenes streaming evaluation** — same temporal layer; protocol adapted for outdoor (ego-motion, dynamic objects, multi-class nuScenes detection metrics).

Target: Jun–Aug 2026.

## 7. Repository structure

```
OpenYOLO3D/                                                 # unified main worktree (Indoor + Outdoor, post 2026-05-20 merge)
├── method_scannet/                                         # Indoor — temporal consistency layer
│   ├── method_11_frame_counting.py                         # registration Phase 1 (FrameCountingGate)
│   ├── method_12_bayesian.py                               # registration Phase 2 (BayesianGate, post Task 1.4c fix)
│   ├── method_21_weighted_voting.py                        # label Phase 1
│   ├── method_22_feature_fusion.py                         # label Phase 2 (CLIP image EMA)
│   ├── method_31_iou_merging.py                            # merge Phase 1
│   ├── method_32_hungarian_merging.py                      # merge Phase 2
│   ├── hooks.py                                            # offline install/uninstall
│   ├── clip_image_encoder.py
│   ├── extract_prompt_embeddings.py / _v2.py
│   ├── eval_phase1.py / eval_phase2.py
│   ├── eval_method_{21,22,31,32}_only.py
│   ├── streaming/                                          # Task 1.1–1.6 streaming harness
│   │   ├── wrapper.py                                      #   StreamingScanNetEvaluator + 6 method hooks
│   │   ├── visibility.py / baseline.py / metrics.py        #   D3 visibility, MVPDist-equivalent, temporal metrics
│   │   ├── running_labeler.py                              #   RunningInstanceLabeler (Task 1.3 Part 2)
│   │   ├── hooks_streaming.py                              #   install/uninstall (6 single + 6 compound)
│   │   ├── method_adapters.py                              #   May-class ↔ streaming-wrapper adapter (Task 1.4a)
│   │   ├── eval_streaming_ablation.py / eval_streaming_baseline.py
│   │   ├── run_streaming_scene.py
│   │   ├── tools/                                          #   cache, aggregate, debug, compare
│   │   └── tests/                                          #   51 unit tests (47 original + 4 from Task 1.4c)
│   └── analysis/                                           # Step 3 — M22 narrow-band + M32 absorption
│
├── adapters/                                               # Outdoor — nuScenes ↔ OpenYOLO3D bridge
│   ├── nuscenes_to_openyolo3d.py
│   ├── lidar_proposals.py
│   └── centerpoint_proposals.py
├── dataloaders/                                            # Outdoor — nuScenes frame interface
├── preprocessing/                                          # Outdoor — pillar foreground / ground filter / verticality
├── proposal/                                               # Outdoor — detection-guided clustering
├── diagnosis/                                              # Outdoor — Tier 1/2 setup-level
├── diagnosis_tier2/                                        # Outdoor — Tier 2-Ext (100 samples)
├── diagnosis_step1/, diagnosis_step_a/                     # Outdoor — 7-stage diagnosis (steps 1, A)
├── diagnosis_w1/, diagnosis_w1_5/                          # Outdoor — W1 geometric clustering check + W1.5 sweep
├── diagnosis_beta1/, diagnosis_beta1_5/                    # Outdoor — β1 geometric clustering proposal (+ variant)
├── diagnosis_gamma/                                        # Outdoor — γ CenterPoint proposal
├── diagnosis_option5/                                      # Outdoor — Option-5 variant
├── diagnosis_alpha/                                        # Outdoor — α distance-aware β1+γ hybrid simulation
├── diagnosis_beta_baseline/                                # Outdoor — β baseline v1 (mAP=0)
├── diagnosis_beta_baseline_v2/                             # Outdoor — β baseline v2 (GT bug fixed, mAP=0 confirmed)
├── configs/                                                # Outdoor — run configs (dataloader, baseline)
│
├── docs/
│   ├── nuscenes_diagnosis.md                               # Outdoor — full diagnosis archive (May 2026)
│   ├── task_1_1_pipeline_analysis.md                       # Task 1.1 Stage 1
│   ├── task_1_1_streaming_design.md                        # Task 1.1 Stage 2 (option (다))
│   ├── task_1_1_metrics_spec.md                            # Task 1.1 Stage 3
│   ├── task_1_3_temporal_instrumentation_fix.md            # Task 1.3 Part 2 instrumentation fix
│   ├── task_1_4a_method_interfaces.md / _redesign_notes.md # Task 1.4a integration design
│   ├── task_1_4c_m11_m12_diagnosis.md                      # Task 1.4c M12 silent-bug diagnosis
│   ├── phase2_integration_plan.md
│   ├── scannet200_classes_location.md / stage_b_mvpdist_location.md
│   └── BASELINE.md / CONTEXT.md / Installation.md / NUSCENES_SETUP.md / SMOKE_TEST_NUSCENES.md
│
├── results/                                                # gitignored except experiment_tracker.md + tracked β baseline dirs
│   ├── experiment_tracker.md                               # tracked aggregate ablation table (offline + streaming + N-sweep)
│   ├── 2026-05-05…2026-05-09_*                             # offline 8-way ablation (§5.1)
│   ├── 2026-05-12_…_v01                                    # initial streaming baseline (sanity FAIL, diagnosed)
│   ├── 2026-05-13_mask3d_cache/                            # 312-scene Mask3D cache
│   ├── 2026-05-13_streaming_…_cached_v01                   # Step 1 sanity PASS
│   ├── 2026-05-13_step3_m22_m32_failures/                  # Step 3 analytical results
│   ├── 2026-05-15_streaming_ablation_core_temporal/        # Task 1.4b Part 3 (8 axes × 312 scenes)
│   ├── 2026-05-18_streaming_ablation_m12_fixed_v01/        # Task 1.4c Phase 3 (M12 fixed × 312)
│   ├── 2026-05-19_m11_n_sweep_v02/                         # Task 1.5 (50-scene N sweep)
│   ├── 2026-05-19_streaming_ablation_m11_n4_v01/           # Task 1.6 (M11_N4 × 312)
│   ├── diagnosis_*/                                        # Outdoor diagnosis runs (gitignored)
│   └── diagnosis_beta_baseline{,_v2}/                      # Outdoor β baseline (tracked)
│
├── scripts/                                                # PBS scripts (Indoor + Outdoor)
│   ├── run_scannet_*.pbs                                   # Indoor offline + streaming
│   ├── run_streaming_*.pbs                                 # Indoor streaming + ablation
│   ├── run_task_1_4c_m12_fixed.pbs                         # Task 1.4c Phase 3
│   ├── run_task_1_5_m11_n_sweep.pbs                        # Task 1.5
│   ├── m11_n_sweep.py                                      # Task 1.5 sweep entry
│   ├── run_alpha.pbs / run_beta1*.pbs / run_gamma.pbs ...  # Outdoor diagnosis runs
│   ├── run_beta_baseline.pbs                               # Outdoor β baseline
│   └── (Outdoor) run_clustering_check / run_diagnosis* / run_step* / run_w1_5 / sanity_check_trainval.py
│
├── pretrained/                                             # ScanNet200 prompt embeddings (v1, v2)
├── evaluate/, models/, utils/                              # OpenYOLO3D core (untouched)
├── run_evaluation.py                                       # OpenYOLO3D evaluation entry (Indoor reference)
├── run_nuscenes.py / run_nuscenes_smoke_test.py            # Outdoor entries
└── single_scene_inference.py                               # OpenYOLO3D single-scene entry
```

## 8. Roadmap

| Step | Description | Status |
|---|---|---|
| **Step 1** | Online streaming evaluation harness + sanity PASS | ✅ Complete (May 13) |
| **Step 2** | Online streaming ablation (Task 1.4b Part 3 + Task 1.4c + 1.5 + 1.6) | ✅ Complete (May 14–19) |
| **Step 3** | M22 / M32 failure analysis | ✅ Complete (fix-not-needed verdict from Part 3) |
| **Step 4** | Outdoor extension (β1+γ hybrid + indoor-temporal-layer integration on nuScenes) | ✅ Diagnosis + native evaluator complete |
| **Step 5** | OV-TCS metric + two-axis formulation study (AP-blindness, beyond-length, fragmentation decomposition, real-method separation) | ✅ Complete on banked data |
| **Step 6** | Paper writing (Abstract/Intro/Related Work/Method drafted) + ScanNet200 train(1201)+val M22 run | 🚧 In progress |

Milestones:

| Month | Milestone |
|---|---|
| May 2026 | Indoor Step 1–3 ✅ + repository unification ✅ |
| Jun 2026 | Outdoor native evaluator + OV-TCS metric/formulation study ✅ |
| Jul 2026 | Paper drafting (Abstract → Method) 🚧 + ScanNet200 train+val M22 run 🚧 |
| Sep 2026 | Graduation thesis presentation |
| Oct 2026 | Paper writing (Experiments + Discussion) |
| Nov 13–15, 2026 | CVPR 2027 submission deadline window |

## 9. Setup

Two conda environments, one repo:

```bash
# Indoor (ScanNet200)
conda activate openyolo3d
qsub scripts/run_streaming_baseline_cached.pbs            # Step 1 streaming baseline
qsub scripts/run_task_1_5_m11_n_sweep.pbs                 # Task 1.5 N sweep

# Outdoor (nuScenes) — env adds nuscenes-devkit + mmdet3d==1.4.0
conda activate openyolo3d-dev
qsub scripts/run_alpha.pbs                                # α hybrid simulation
qsub scripts/run_beta_baseline.pbs                        # β baseline
```

A100 queue (`coss_agpu`) — every PBS script pins `Qlist=agpu`.

Single-scene streaming debug:

```bash
python -m method_scannet.streaming.run_streaming_scene \
    --scene scene0011_00 \
    --mask3d-filter --projection-index
```

Outdoor data layout + environment: see [`docs/NUSCENES_SETUP.md`](docs/NUSCENES_SETUP.md).

## 10. Branching

- **`main`** — unified, post-2026-05-20 merge (Indoor `feature/method-scannet-21-31` + Outdoor `feature/diagnosis` both integrated)
- (deprecated) `feature/method-scannet-21-31` — pre-merge Indoor branch, kept for git history reference
- (deprecated) `feature/diagnosis` — pre-merge Outdoor branch, kept for git history reference
- (historical) `feature/nuscenes-dataloader`, `backup/initial-pipeline`

All future development lands on `main` directly or via short-lived feature branches.

## 11. License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World),
[Mask3D](https://github.com/JonasSchult/Mask3D) (indoor proposal; does not transfer outdoor, see §6.2.4),
[ScanNet200](https://github.com/ScanNet/ScanNet),
[CenterPoint](https://github.com/tianweiy/CenterPoint) via [mmdet3d](https://github.com/open-mmlab/mmdetection3d),
and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

Graduation research at Soongsil University, targeting CVPR 2027.

---

**Contact**: yuha@soongsil.ac.kr
