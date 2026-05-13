# SemWorld-3D

**Streaming Open-Vocabulary 3D Instance Mapping — Proposal-Agnostic Temporal Consistency Layer**

A research project building on [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D), reframed (May 12, 2026) as a single
proposal-agnostic temporal consistency layer applied across two domains:

- **Indoor (ScanNet200)**: Mask3D proposals → temporal layer → instance map (this branch)
- **Outdoor (nuScenes)**: LiDAR clustering proposals → same temporal layer → instance map ([branch `feature/diagnosis`](https://github.com/hyuzenn/OpenYOLO3D/tree/feature/diagnosis))

> ⚠️ **Status (2026-05-13)**
> - **Step 1 — Online streaming evaluation harness** ✅ COMPLETE (sanity PASS by 333× margin)
> - **Step 2 — Online 12-way ablation** 🚧 IN PROGRESS (Task 1.4b, 6 sequential PBS jobs)
> - **Step 3 — METHOD_22 / METHOD_32 failure analysis** ✅ ANALYSIS COMPLETE (fixes pending Task 1.4b)
> - **Step 4 — Outdoor extension** ⏳ PENDING (see [`feature/diagnosis`](https://github.com/hyuzenn/OpenYOLO3D/tree/feature/diagnosis))
>
> Targeting CVPR 2027 (Nov 1, 2026 deadline).

---

## 1. Contribution

> **Proposal-agnostic temporal consistency layer for streaming open-vocabulary 3D instance mapping.**

The contribution is the *layer* itself — the registration / label consistency / spatial merge stack that turns a stream of
noisy per-frame open-vocab observations into a stable 3D instance map. The proposal source is treated as a swappable
domain-specific front-end:

| Domain | Proposal source | Temporal layer | Output |
|---|---|---|---|
| Indoor | Mask3D (per-scene, once) | M11/12 + M21/22 + M31/32 | online 3D instance map |
| Outdoor | LiDAR clustering (hybrid β1 + γ) | same layer | online 3D instance map |

Replacing the proposal source does not change the layer. This separation is what justifies "proposal-agnostic" and what
makes ScanNet200 a viable validation domain for an outdoor-targeted method.

## 2. Why this contribution

The May 2026 offline ScanNet200 ablations (8-way matrix, see §6.1) showed that the proposed methods sit between *noise
floor* and *clear negative* under static offline evaluation. The honest reading of that result, agreed with advisor
(May 12), is:

- **ScanNet200 offline static evaluation cannot measure what the method is for.** The methods (M11/12/21/22/31/32) all
  target *temporal inconsistency* across a stream of frames; an offline pipeline that processes every frame in one shot
  has no temporal inconsistency to fix.
- **The streaming protocol is the smallest setting in which the method's value is measurable.** Frames arrive one at a
  time; a label can flicker across viewpoints; an instance can be confirmed late, never, or twice. These are the failure
  modes the temporal layer is built for.

This is the project's raison d'être for the May 13 + onward work: build the streaming evaluation harness first, then
re-measure the same methods inside it. The 8-way offline numbers in §6.1 are kept as the "limit of static measurement"
reference, not as the method's verdict.

## 3. Method overview (6 axes, equal emphasis)

Three orthogonal axes, each with a simple (Phase 1) and an advanced (Phase 2) variant. Until Task 1.4b reports, all six
are treated as equally weighted ablation slots — no axis is privileged in the writing.

| Axis | Phase 1 (simple) | Phase 2 (advanced) | Streaming failure mode addressed |
|---|---|---|---|
| Registration | **METHOD_11** Frame-counting "wait-and-see" gate | **METHOD_12** Bayesian probability accumulation | False positives from single-frame noise |
| Label consistency | **METHOD_21** Weighted label vote (distance + center) | **METHOD_22** CLIP visual-feature EMA fusion | Label flicker across viewpoints |
| Spatial merge | **METHOD_31** 3D IoU same-class merging | **METHOD_32** Hungarian (semantic + distance) | Identity switches / over-segmentation |

Each method is wired into the streaming wrapper via an installer in `method_scannet/streaming/hooks_streaming.py`; the
12-way ablation (Task 1.4b) sweeps each axis independently and the four compound configurations (phase1, phase2,
M11+M21, M12+M22, M21+M31, M22+M32).

## 4. Streaming evaluation protocol

The protocol is **Mask3D per-scene + frame-visible filtering + streaming 2D label fusion** (Task 1.1 design Option (다)).

| Decision | Choice |
|---|---|
| Mask3D scope | Once per scene, offline (cached in `results/2026-05-13_mask3d_cache/`) |
| Frame visibility | **D3**: instance is processed in frame *t* iff `visible_count(k, t) ≥ 1` vertex |
| "Confirmed" criterion | **C1, K=3**: an instance is committed when ≥ 3 consecutive frames agree on its class |
| Primary mAP | Incremental against *visible GT up to t* |
| Secondary mAP | Incremental against full-scene GT (recall ceiling at *t*) |
| Temporal metrics | ID switches per object, label switch count, time-to-confirm, mask-IoU |

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

**Task 1.4b baseline reference**: cached streaming AP = **0.1956**. All Task 1.4b ablations compare against this number
(same Mask3D cache), not the 0.2473 May-7 offline number (different Mask3D draw).

### 5.3 METHOD_22 / METHOD_32 failure analysis (Step 3, May 13)

Analytical — no re-inference; uses only prompt embeddings + GT (the May 2026 result directories did not preserve
per-instance predictions). Caveat: this characterises the *failure mechanism*, not the exact failed instances. Full
data in `results/2026-05-13_step3_m22_m32_failures/`.

**M22 — CLIP narrow band CONFIRMED (stronger than original memo)**

| Statistic | v1 ("{class}") | v2 ("a photo of a {class}") |
|---|---|---|
| Off-diag cosine **mean** | **0.759** | 0.750 |
| Off-diag cosine median | 0.760 | 0.751 |
| Nearest-neighbour cosine mean | **0.889** | 0.894 |
| Nearest-neighbour cosine max | **0.984** | 0.985 |

The original Task 1.1 memo recorded the narrow band as [0.257, 0.304]. The actual band is ≈ 3× higher; the hypothesis is
*stronger*, not weaker. Top confused pairs: *trash bin ↔ trash can* (0.984), *keyboard piano ↔ keyboard* (0.962),
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
them into single predictions. This is fully consistent with the −0.0332 single-axis offline AP.

**Fix proposals** are written up in `results/2026-05-13_step3_m22_m32_failures/fix_proposals.md` (M22 a/b/c, M32 a/b/c).
None are implemented — the decision waits on Task 1.4b streaming numbers (a method that recovers under streaming may not
need a fix).

### 5.4 Streaming 12-way ablation (Step 2 / Task 1.4b — IN PROGRESS)

Sweep matrix (12 configs): `baseline`, single-axis `M11 / M12 / M21 / M22 / M31 / M32`, compounds `phase1 / phase2 /
M11+M21 / M12+M22 / M21+M31 / M22+M32`. Submitted as 6 sequential PBS jobs under `scripts/run_streaming_ablation_pbs{1..6}.pbs`
on the `coss_agpu` A100 queue. Total walltime ≈ 30 h.

Hypothesis (carried from offline + Step 3, but **not** confirmed yet): M21 and M31 — the Phase-1 axes that registered as
noise-floor offline — should produce a clean positive gain in streaming, because the inconsistency they fix is created
by the streaming protocol itself.

README will be updated with the 12-way streaming table once Task 1.4b completes.

## 6. Outdoor extension (Step 4 — PENDING)

Step 4 begins after Indoor Step 1–3 close. Intentionally short here; the diagnosis archive (7-stage hypothesis tests +
α hybrid simulation) lives on [branch `feature/diagnosis`](https://github.com/hyuzenn/OpenYOLO3D/tree/feature/diagnosis).

- **Domain change**: ScanNet200 (indoor static) → nuScenes (outdoor dynamic, multi-camera, sparse 32-beam LiDAR).
- **Proposal source change**: Mask3D → hybrid LiDAR clustering: unsupervised pillar foreground extraction (β1) for far
  range + CenterPoint (γ) for near range. May 2026 diagnosis simulated this as **+10.6 pt M-rate** (β1 0.361 / γ 0.352
  → hybrid 0.467 at 35 m threshold).
- **Temporal layer**: identical to indoor; no method change required.

## 7. Repository structure

```
OpenYOLO3D/                                              # this worktree (indoor)
├── method_scannet/
│   ├── method_11_frame_counting.py      # NEW (May 13)  registration Phase 1
│   ├── method_12_bayesian.py            # NEW (May 13)  registration Phase 2
│   ├── method_21_weighted_voting.py     # 5월             label Phase 1
│   ├── method_22_feature_fusion.py      # 5월             label Phase 2 (CLIP image EMA)
│   ├── method_31_iou_merging.py         # 5월             merge Phase 1
│   ├── method_32_hungarian_merging.py   # 5월             merge Phase 2
│   ├── hooks.py                         # 5월 offline install/uninstall
│   ├── clip_image_encoder.py            # CLIP image encoder for M22
│   ├── extract_prompt_embeddings.py     # ScanNet200 prompt embedding cache (v1)
│   ├── extract_prompt_embeddings_v2.py  # v2 prompts ("a photo of a {class}")
│   ├── eval_phase1.py / eval_phase2.py  # offline phase evaluations
│   ├── eval_method_{21,22,31,32}_only.py# offline single-axis ablations
│   ├── streaming/                       # NEW (Task 1.1–1.4a, May 12–13)
│   │   ├── visibility.py                #   D3 frame visibility
│   │   ├── wrapper.py                   #   StreamingScanNetEvaluator + 6 method hooks
│   │   ├── baseline.py                  #   MVPDist-equivalent label accumulator
│   │   ├── metrics.py                   #   4 temporal metrics + mask-IoU + ScanNet200 eval
│   │   ├── hooks_streaming.py           #   install/uninstall (6 single + 6 compound)
│   │   ├── eval_streaming_ablation.py   #   12-way runner
│   │   ├── eval_streaming_baseline.py   #   streaming baseline runner
│   │   ├── run_streaming_scene.py       #   single-scene streaming entry point
│   │   ├── tools/                       #   cache, aggregate, debug, compare
│   │   │   ├── generate_mask3d_cache.py
│   │   │   ├── aggregate_ablation_results.py
│   │   │   ├── debug_E_deterministic.py
│   │   │   ├── debug_compare.py
│   │   │   ├── compare_npy_ply.py
│   │   │   └── select_debug_scenes.py
│   │   └── tests/                       #   8 unit test files
│   ├── analysis/                        # NEW (Step 3, May 13)
│   │   ├── m22_clip_narrow_band.py      #   CLIP prompt-cosine analysis
│   │   └── m32_multi_instance.py        #   GT-centroid absorption analysis
│   └── tests/                           # offline method unit tests (7 files)
├── docs/
│   ├── task_1_1_pipeline_analysis.md    # NEW (May 12) Task 1.1 Stage 1
│   ├── task_1_1_streaming_design.md     # NEW (May 12) Task 1.1 Stage 2 (option (다))
│   ├── task_1_1_metrics_spec.md         # NEW (May 12) Task 1.1 Stage 3
│   ├── stage_b_mvpdist_location.md      # 5월 MVPDist hook location
│   ├── scannet200_classes_location.md   # 5월 class list + CLIP variants
│   └── phase2_integration_plan.md       # 5월 Phase 2 integration plan
├── results/                             # gitignored; experiment_tracker.md tracked
│   ├── 2026-05-05…2026-05-09_*          # 5월 baseline + offline 8-way ablation
│   ├── 2026-05-12_*                     # initial streaming baseline (sanity FAIL, diagnosed)
│   ├── 2026-05-13_mask3d_cache          # 312-scene shared Mask3D cache
│   ├── 2026-05-13_scannet_baseline_cached_v01            # G.2 offline cached
│   ├── 2026-05-13_scannet_streaming_baseline_cached_v01  # G.3 streaming cached — Step 1 PASS
│   ├── 2026-05-13_streaming_ablation_pbs{1..6}_v01       # Task 1.4b (IN PROGRESS)
│   ├── 2026-05-13_step3_m22_m32_failures                 # Step 3 analysis
│   └── experiment_tracker.md            # tracked aggregate ablation table
├── scripts/
│   ├── run_scannet_full_eval.pbs                # baseline
│   ├── run_scannet_phase1_eval.pbs              # offline Phase 1
│   ├── run_scannet_phase2.pbs / _v02.pbs        # offline Phase 2
│   ├── run_scannet_method_{21,22,31,32}_only.pbs# offline single-axis
│   ├── run_scannet_method_22_v2.pbs             # offline M22 with v2 prompts
│   ├── run_scannet_method_31_iou07.pbs          # offline M31 near-no-op
│   ├── run_scannet_baseline_cached.pbs          # G.2 offline cached
│   ├── run_streaming_baseline.pbs               # uncached streaming baseline
│   ├── run_streaming_baseline_cached.pbs        # G.3 streaming cached
│   ├── run_streaming_scene0011_00{,v03}.pbs     # single-scene debug
│   ├── run_streaming_debug_{E,H}.pbs            # Task 1.2c diagnosis runs
│   ├── run_streaming_ablation_pbs{1..6}.pbs     # Task 1.4b ablation jobs
│   ├── run_generate_mask3d_cache.pbs            # Mask3D cache generator
│   ├── run_regression_check{,_b,_c}.pbs         # regression sanity
│   └── smoke_method_{22,32}.pbs                 # single-scene smoke
├── pretrained/
│   ├── scannet200_prompt_embeddings.pt          # v1 prompts (ViT-B/32, L2-normalised)
│   └── scannet200_prompt_embeddings_v2.pt       # v2 prompts ("a photo of a {class}")
├── run_evaluation.py                            # extended with _maybe_dump_metrics()
└── [OpenYOLO3D core files — untouched]
```

## 8. Roadmap

| Step | Description | Status |
|---|---|---|
| **Step 1** | Online streaming evaluation harness (`method_scannet/streaming/`) + sanity PASS | ✅ Complete (May 13) |
| **Step 2** | Online 12-way streaming ablation (Task 1.4b) | 🚧 In progress (~30 h walltime) |
| **Step 3** | M22 / M32 failure analysis (CLIP narrow band + absorption) | ✅ Analysis complete (fixes pending Step 2 result) |
| **Step 4** | Outdoor extension on nuScenes (hybrid β1 + γ → temporal layer) | ⏳ Pending |

Milestones:

| Month | Milestone |
|---|---|
| May 2026 | Indoor Step 1–3 |
| Jun–Jul 2026 | Step 4 (outdoor extension) |
| Aug 2026 | Full experiments + ablation |
| Sep 2026 | Graduation thesis presentation |
| Oct 2026 | Paper writing |
| Nov 1, 2026 | CVPR 2027 submission |

## 9. Setup

```bash
conda activate openyolo3d
# A100 queue (coss_agpu) — every PBS script pins Qlist=agpu
qsub scripts/run_scannet_full_eval.pbs           # offline baseline
qsub scripts/run_streaming_baseline_cached.pbs   # streaming baseline (Step 1)
qsub scripts/run_streaming_ablation_pbs1.pbs     # Task 1.4b PBS1 (registration axis)
# … PBS2…PBS6 sequential
```

Single-scene streaming debug:

```bash
python -m method_scannet.streaming.run_streaming_scene \
    --scene scene0011_00 \
    --mask3d-filter --projection-index
```

## 10. Branching

- `main` — stable; integrated steps only.
- `feature/method-scannet-21-31` (current) — Indoor Step 1–3 + Task 1.4b.
- [`feature/diagnosis`](https://github.com/hyuzenn/OpenYOLO3D/tree/feature/diagnosis) — outdoor (nuScenes) diagnosis archive.

## 11. License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World),
[Mask3D](https://github.com/JonasSchult/Mask3D), [ScanNet200](https://github.com/ScanNet/ScanNet), and (for the outdoor
extension) [CenterPoint](https://github.com/tianweiy/CenterPoint) via [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

Conducted as a graduation research project at Soongsil University.

---

**Contact**: yuha@soongsil.ac.kr
