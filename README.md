# SemWorld-3D

**Streaming Open-Vocabulary 3D Instance Mapping — Indoor First, Outdoor Extension**

A research project building on [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D) with two phases agreed with advisor:

1. **Phase 1 (current)**: validate the proposed method modules on ScanNet200 (indoor static scenes)
2. **Phase 2**: extend to nuScenes (outdoor dynamic driving scenes)

> ⚠️ **Status**: ScanNet200 method validation in progress. METHOD_21 + METHOD_31 (Phase 1 of method axes) evaluated; ablations underway. Targeting CVPR 2027 submission (Nov 2026 deadline).

---

## Method overview

The method extends OpenYOLO3D along three orthogonal axes, each with two phases (simple → advanced). The simple phase is intended primarily as ablation; the advanced phase carries the contribution.

| Axis | Phase 1 (simple) | Phase 2 (advanced) |
|---|---|---|
| **Instance Registration** | METHOD_11: Frame-counting "wait and see" | METHOD_12: Bayesian probability accumulation |
| **Label Consistency** | METHOD_21: Weighted label voting (distance + center) | METHOD_22: Visual-feature EMA fusion |
| **Spatial Merging** | METHOD_31: 3D IoU-based merging | METHOD_32: Hungarian distance + semantic cost |

Plan agreed with advisor:
- **ScanNet200**: validate METHOD_21/22/31/32 (label consistency + spatial merging axes). METHOD_11/12 (instance registration) is skipped on ScanNet because ScanNet is static and per-scene, not online streaming — the registration axis is meaningful only on dynamic data.
- **nuScenes**: validate METHOD_11/12 in addition; adapt the ScanNet-validated method to outdoor.

## Why this method shape

The three axes target three failure modes of online open-vocab 3D mapping:

1. **Registration** — false positives from single-frame noise; resolved by requiring temporal consistency before committing to the map.
2. **Label consistency** — same instance receives different class labels across viewpoints due to open-vocab text classifier instability; resolved by weighted voting (Phase 1) or feature-level fusion (Phase 2).
3. **Spatial merging** — the same physical object gets registered as multiple instances due to label drift across frames; resolved by IoU-based merging (Phase 1) or distance + semantic cost via Hungarian assignment (Phase 2).

## Integration with OpenYOLO3D

| Method | Integration point | Strategy |
|---|---|---|
| METHOD_11/12 | Pipeline front-end (between Mask3D output and MVPDist) | Wait-and-see / Bayesian gate before label assignment |
| METHOD_21/22 | MVPDist replacement via wrapper hook | Monkey-patch: replace MVPDist with WeightedVoting / FeatureFusionEMA. Original MVPDist preserved; reverted if performance drops significantly |
| METHOD_31/32 | Output post-processing | Receive final instance map, apply 3D IoU merging or Hungarian-based merging |

OpenYOLO3D core code is **not modified**. All method modules are applied via `install_*` / `uninstall_*` hooks; baseline behavior is exactly recovered when hooks are uninstalled.

---

## ScanNet200 progress (May 2026)

### Stage 1 — Baseline reproduction (✅ done)

OpenYOLO3D ScanNet200 baseline reproduced on TITAN RTX (matches paper):

| Metric | Value |
|---|---|
| AP | 0.247 |
| AP_50 | 0.318 |
| AP_25 | 0.363 |
| AP head / common / tail | 0.278 / 0.243 / 0.216 |

Re-verified on A100: Δ ≤ 0.001 across all buckets — hardware-stable.

Infrastructure: `run_evaluation.py` extended with `_maybe_dump_metrics()` that writes a structured `metrics.json` to `RUN_DIR`. This is the single comparison reference for all method ablations.

### Stage 2 — Phase 1 evaluation (✅ done; result: NEUTRAL)

METHOD_21 + METHOD_31 with default hyperparameters (alpha=0.5, D=10m, C=300px, IoU=0.5 same-class):

| Category | Metric | Baseline | Phase 1 | Δ |
|---|---|---|---|---|
| average | AP | 0.2473 | 0.2443 | **−0.0029** |
| average | AP_50 | 0.3175 | 0.3157 | −0.0018 |
| average | AP_25 | 0.3624 | 0.3629 | +0.0005 |
| head | AP | 0.2776 | 0.2763 | −0.0013 |
| common | AP | 0.2432 | 0.2393 | −0.0038 |
| tail | AP | 0.2162 | 0.2125 | −0.0037 |
| tail | AP_50 | 0.2689 | 0.2627 | −0.0063 |

Decision branch fired: **NEUTRAL** ([-0.005, +0.005] for average AP). The combined effect is mildly negative on strict AP, concentrated in tail classes.

### Stage 3 — Single-method ablations (🚧 in progress)

The Phase 1 result mixes two methods. To isolate contributions, single-method ablations are running:

- **METHOD_31 alone** (no MVPDist replacement, only IoU post-merge): in progress
- **METHOD_21 alone** (MVPDist wrapper, no post-merge): pending

These ablations (a) clarify which axis is responsible for the regression, (b) provide ablation table data for the paper.

### Stage 4 — Phase 2 + Mix experiments (pending)

Five-experiment matrix agreed with advisor:

| Experiment | Label axis | Merge axis |
|---|---|---|
| Baseline | original MVPDist | (none) |
| Phase 1 (✅) | METHOD_21 | METHOD_31 |
| Phase 2 | METHOD_22 | METHOD_32 |
| Mix-A | METHOD_21 | METHOD_32 |
| Mix-B | METHOD_22 | METHOD_31 |

METHOD_22 (FeatureFusionEMA) and METHOD_32 (HungarianMerger) are implemented and unit-tested. Phase 2 integration requires a CLIP image encoder for per-frame visual embedding extraction (not part of the original OpenYOLO3D pipeline) — this is the main implementation work for Stage 4.

ScanNet200 prompt embeddings (200 × 512, ViT-B/32, L2-normalized) are pre-extracted and cached at `pretrained/scannet200_prompt_embeddings.pt`.

---

## nuScenes preliminary investigation (May 2026)

> Run on a separate worktree (`OpenYOLO3D-nuscenes`, branch `feature/diagnosis`). Conducted before the ScanNet method work; preserved as preliminary outdoor-extension data and to inform the eventual transition.

A 7-stage hypothesis-driven measurement campaign on nuScenes trainval keyframes characterized how the indoor pipeline fails outdoors. Each stage tested a specific hypothesis with explicit pass/fail thresholds.

### Setup-level findings (Tier 1 / 2 / 2-Ext, 100 samples)

| Finding | Value |
|---|---|
| Per-pixel depth coverage in detection box | 0.20% |
| Detection-induced GT loss | 44.8% |
| GT distribution beyond 30m | 52% |
| Multi-view GT fraction (≥2 cams visible) | 8.6% |
| Mask3D (indoor-trained) instances per frame | 2.9 |

### Proposal-level findings (7 stages, 50 samples, seed=42)

| Stage | Method | M-rate (1↔1 GT match) |
|---|---|---|
| W1 | HDBSCAN baseline | 9.1% |
| W1.5 | extended HDBSCAN sweep | 28.5% |
| Step 1 | Mask3D vs HDBSCAN | Mask3D 0.3% / HDBSCAN 23.2% |
| β1 | Pillar foreground extraction | **36.1%** (geometric sweet spot) |
| β1.5 | Verticality filter | 27.4% (over-engineering) |
| Step A | Pillar resolution sweep | 36.1% (plateau confirmed) |
| Option 5 | 2D detection-guided clustering | 15.1% (wrong axis) |
| γ | CenterPoint (learned 3D detector) | 35.2% |

Seven different proposal strategies all plateau in the 35–36% region. This is the GT-LiDAR coverage ceiling on nuScenes (sparse 32-beam LiDAR + 52% of GT > 30m).

### Hybrid simulation (α, finding)

β1 (unsupervised geometric) and γ (CenterPoint) showed **opposite distance bias**:

| Distance bin | β1 M-rate | γ M-rate |
|---|---|---|
| 0–10m | 34.8% | **71.5%** (+36.7pt) |
| 50m+ | 15.5% | 4.7% |

Per-sample paired: γ wins 19/50, β1 wins 17/50, similar 14/50.

A distance-aware union (β1 for far range, γ for near range) was simulated:

| Configuration | M-rate |
|---|---|
| β1 alone | 0.361 |
| γ alone | 0.352 |
| **Hybrid (distance-aware, threshold=35m)** | **0.467** |

Ceiling break: +10.6pt absolute, +29.4% relative. Complementarity (β1-only ∪ γ-only coverage) = 27%.

### Implications for the outdoor phase

The nuScenes preliminary data motivates two design choices for the outdoor extension (Phase 2 of the project):

- **Hybrid proposal** (geometric + learned 3D detector with distance-aware arbitration) is required to break the single-source ceiling.
- **METHOD_11/12 (instance registration)** becomes essential on nuScenes because the dataset is dynamic and online — the registration gate is meaningful only here.

These results are preserved on the `feature/diagnosis` branch. The outdoor extension work resumes after the ScanNet method validation completes.

---

## Repository structure

```
.
├── method_scannet/
│   ├── method_21_weighted_voting.py    # WeightedVoting class (MVPDist replacement, Phase 1)
│   ├── method_22_feature_fusion.py     # FeatureFusionEMA class (MVPDist replacement, Phase 2)
│   ├── method_31_iou_merging.py        # IoUMerger class (post-merge, Phase 1)
│   ├── method_32_hungarian_merging.py  # HungarianMerger class (post-merge, Phase 2)
│   ├── hooks.py                        # install_phase1 / uninstall_phase1 (monkey-patch hooks)
│   ├── eval_phase1.py                  # Phase 1 evaluation entry
│   ├── extract_prompt_embeddings.py    # CLIP text-encoder caching for METHOD_22
│   └── tests/                          # 10/10 unit tests passing
├── pretrained/
│   └── scannet200_prompt_embeddings.pt # 200 × 512, ViT-B/32, L2-normalized
├── docs/
│   ├── stage_b_mvpdist_location.md     # MVPDist hook locations identified
│   ├── scannet200_classes_location.md  # Class list + CLIP variant references
│   └── phase2_integration_plan.md      # Phase 2 integration plan + decision items
├── results/
│   ├── 2026-05-05_scannet_eval_v02/    # TITAN RTX baseline
│   ├── 2026-05-07_scannet_eval_v01/    # A100 baseline (Δ ≤ 0.001 vs TITAN)
│   └── 2026-05-07_scannet_phase1_v02/  # Phase 1 result (AP −0.0029)
├── run_evaluation.py                    # Extended with _maybe_dump_metrics() (ScanNet/Replica)
├── scripts/
│   ├── run_scannet_full_eval.pbs       # Baseline (A100 pin)
│   └── run_scannet_phase1_eval.pbs     # Phase 1 evaluation
└── [OpenYOLO3D core files — untouched]
```

The `OpenYOLO3D-nuscenes/` worktree (branch `feature/diagnosis`) holds the nuScenes preliminary investigation (7-stage diagnosis + α hybrid simulation). Not modified during ScanNet work.

## Roadmap

| Stage | Period | Status |
|---|---|---|
| ScanNet200 baseline reproduction | May 2026 | ✅ Done (A100 verified) |
| nuScenes preliminary investigation (7-stage + α) | May 2026 | ✅ Done (preliminary, archived) |
| Phase 1 evaluation (METHOD_21 + METHOD_31) | May 2026 | ✅ Done (NEUTRAL) |
| Single-method ablations (METHOD_21 alone, METHOD_31 alone) | May 2026 | 🚧 In progress |
| Phase 2 + Mix experiments | Jun 2026 | Pending |
| nuScenes adaptation (METHOD_11/12 + outdoor extension) | Jul–Aug 2026 | Pending |
| Full experiments + ablation | Aug 2026 | Pending |
| Graduation thesis presentation | Sep 2026 | Pending |
| Paper writing | Oct 2026 | Pending |
| CVPR 2027 submission | Nov 1, 2026 | Pending |

## Setup

OpenYOLO3D ScanNet evaluation environment:

```bash
conda activate openyolo3d
# A100 queue: scripts pinned with Qlist=agpu
qsub scripts/run_scannet_full_eval.pbs
```

Method evaluation:

```bash
# Phase 1
python -m method_scannet.eval_phase1 --output results/<run_dir>
```

For nuScenes preliminary work, see `OpenYOLO3D-nuscenes/` worktree (`openyolo3d-dev` env with `nuscenes-devkit` + `mmdet3d 1.4.0` added; pre-mmdet3d snapshot at `/home/rintern16/env_snapshots/`).

## Branching

- `main`: stable, integrated stages only
- `feature/method-scannet-21-31`: current ScanNet method work
- `feature/diagnosis` (separate worktree `OpenYOLO3D-nuscenes`): nuScenes preliminary investigation (7-stage + α), archived

## License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Mask3D](https://github.com/JonasSchult/Mask3D), [CenterPoint](https://github.com/tianweiy/CenterPoint) (via [mmdet3d](https://github.com/open-mmlab/mmdetection3d)) for the nuScenes preliminary work, and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

This work is conducted as a graduation research project at Soongsil University.

---

**Contact**: yuha@soongsil.ac.kr