# SemWorld-3D — Outdoor Extension

**Hybrid LiDAR Proposal × Proposal-Agnostic Temporal Consistency Layer (nuScenes)**

This worktree is the **outdoor extension** of SemWorld-3D. The authoritative project README — and the contribution definition — lives in the indoor companion repo:

> Indoor (canonical): [github.com/hyuzenn/OpenYOLO3D](https://github.com/hyuzenn/OpenYOLO3D)

> ⚠️ **Status**: Preliminary diagnosis complete (May 2026 — 7-stage proposal diagnosis + α hybrid simulation + β baseline). **Step 4 implementation pending Indoor Step 1–3 completion** (see indoor worktree for ScanNet200 validation of METHOD_11/12/21/22/31/32).

---

## 1. Context — Indoor + Outdoor architecture

After the 2026-05-12 advisor meeting, the project contribution is defined as a **proposal-agnostic temporal consistency layer**. The same layer is validated indoor and reused outdoor; only the proposal source changes:

| | Proposal source (domain-specific) | Temporal consistency layer (shared) | Open-vocab labeling (shared) |
|---|---|---|---|
| **Indoor** (ScanNet200, canonical) | Mask3D per-scene, frame-visible filtering | METHOD_11/12/21/22/31/32 | YOLO-World |
| **Outdoor** (nuScenes, this worktree) | Hybrid β1 (geometric clustering) + γ (CenterPoint), distance-aware fusion | **Same** METHOD_11/12/21/22/31/32 | **Same** YOLO-World |

The **contribution is the temporal layer itself**, not any outdoor-specific method. This worktree contributes the **outdoor proposal source** (the β1+γ hybrid) and the diagnosis evidence that justifies it. The layer itself is developed and described in the indoor worktree.

## 2. Why outdoor breaks indoor pipelines

Moving from indoor (Replica, ScanNet) to outdoor (nuScenes) is not a dataset swap. It breaks three core assumptions of indoor open-vocab 3D pipelines, and forces one additional layer module (registration) that indoor can skip:

1. **Lifting assumption** — OpenYOLO3D lifts 2D detections to 3D using dense per-pixel depth from RGBD. nuScenes provides 32-beam sparse LiDAR; a 2D bounding box may contain 0–few projected LiDAR points, especially at distance. Median-depth lifting is unreliable.
2. **Static-scene assumption** — Indoor instance association assumes scene geometry is stable between frames. nuScenes contains moving cars, pedestrians, cyclists; naive spatial association collapses on dynamic objects.
3. **Single-view assumption** — Indoor scenes use a forward-facing camera. nuScenes uses 6 surround cameras with overlapping FoV, so the same physical object appears in multiple cameras at different viewpoints, scales, and quality levels.
4. **Registration module is essential outdoor** — On ScanNet (static, per-scene), METHOD_11/12 (instance registration) can be skipped. On nuScenes (online streaming with ego-motion + dynamic objects), the registration sub-module of the temporal layer becomes load-bearing and is validated here, not in the indoor worktree.

## 3. Preliminary diagnosis (May 2026)

> **Note**: These numbers were measured on the v1.0-mini split during the May 2026 diagnosis stage. They are preserved verbatim as paper-writing reference. See `results/` for raw aggregates and per-sample data; the numbers below are not re-derived from later runs.

### 3.1 Setup-level findings (Tier 1 / 2 / 2-Ext, 100 samples)

Quantifies how the indoor lifting assumption breaks on nuScenes inputs:

- **Per-pixel depth coverage**: 0.20% (vs. dense RGBD indoor ≈ 100%)
- **Detection-induced GT loss**: 44.8% (GT objects with insufficient projected LiDAR support inside their 2D detection box)

These confirm the lifting assumption fails before any method-level module — the bottleneck is at proposal generation, not at refinement.

### 3.2 7-stage proposal diagnosis (W1 → γ, 50 samples)

Each stage progressively relaxes/tightens the proposal source to bracket the achievable recall ceiling under a depth-lifting-only regime:

- **β1 (geometric clustering)**: 36.1% — geometric sweet spot, best near-range
- **γ (CenterPoint, learned)**: 35.2% — comparable overall, **opposite distance bias** (better far, worse near vs. β1)
- **Ceiling**: ~36% under either single-source proposal — confirms that no single proposal source can pass the upper bound

### 3.3 α hybrid simulation

Motivated by the opposite distance biases of β1 and γ, simulate a **distance-aware union**:

- **β1 far + γ near, threshold = 35 m**: **46.7%** (+10.6 pt over 36% ceiling)

This is the empirical justification for adopting a hybrid β1+γ proposal as the outdoor proposal source — neither alone clears 36%, but their distance-aware combination does.

### 3.4 β baseline — direct OpenYOLO3D application (May 2026)

Treat OpenYOLO3D as-is on nuScenes, with no outdoor adaptation:

- **mAP = 0%** — catastrophic outdoor failure
- Root cause: Mask3D is pretrained indoor and does not transfer to outdoor LiDAR distributions. No proposal mass exists to feed downstream stages.

The β baseline (and its v2 GT-bug-fix re-measurement, still mAP=0) is the empirical evidence that **Mask3D cannot serve as the outdoor proposal source**, which is what motivates substituting the β1+γ hybrid in its place.

## 4. Method direction — reframed

After 2026-05-12 advisor framing, the outdoor pipeline is **not a separate method**. It is the same temporal consistency layer with a different proposal source:

**Contribution split**:
- **Temporal consistency layer (METHOD_11/12/21/22/31/32)** — defined and validated in the **indoor companion repo** ([github.com/hyuzenn/OpenYOLO3D](https://github.com/hyuzenn/OpenYOLO3D)). Outdoor reuses it as-is.
- **Outdoor proposal source (β1+γ hybrid)** — developed in this worktree. May 2026 diagnosis establishes the recall ceiling and the distance-aware hybrid as the source replacing Mask3D outdoor.

**Pipeline (outdoor)**:

1. **Proposal** — hybrid β1 + γ with distance-aware fusion (this worktree, May 2026 design).
2. **Temporal layer** — METHOD_11/12/21/22/31/32, imported from the indoor worktree without modification.
3. **Open-vocab labeling** — YOLO-World, shared with indoor.

No outdoor-specific layer module is proposed in this README. If outdoor evaluation later reveals a gap that the indoor-validated layer cannot close, that gap will be addressed in a follow-up note, not by silently introducing a new module here.

## 5. Step 4 implementation plan

Step 4 is the **outdoor extension implementation**. It is gated on the indoor side:

- **Precondition**: Indoor Step 1–3 (METHOD_11/12/21/22/31/32 validation on ScanNet200) complete — see Task 1.4b result in the indoor worktree.
- **Tasks**:
  1. LiDAR clustering proposal module — β1 + γ infrastructure (May 2026 diagnosis code under `diagnosis_beta1/`, `diagnosis_gamma/`, `diagnosis_step1/` is the starting point).
  2. Distance-aware fusion stage — implements the α simulation result (β1 far + γ near, threshold ≈ 35 m) as a runtime module, not a post-hoc simulator.
  3. Integration with the indoor-validated temporal layer — import as library; no fork.
  4. nuScenes streaming evaluation — same temporal layer as indoor, analogous protocol adapted for outdoor (online streaming with ego-motion + dynamic objects, multi-class nuScenes detection metrics — not identical to ScanNet200's static per-scene instance segmentation protocol).
- **Target window**: Jul–Aug 2026, after the indoor evaluation cycle completes.

## 6. Repository structure

```
.
├── adapters/                          # Bridges nuScenes dataloader → OpenYOLO3D
├── configs/                           # Run configs (dataloader, baseline)
├── data -> <nuScenes dataset root>     # symlink, set per-machine
├── dataloaders/                       # nuScenes per-frame dict interface + sanity check
├── diagnosis/                         # Stage-1 diagnosis (setup-level Tier 1/2)
├── diagnosis_tier2/                   # Stage-1 diagnosis (Tier 2-Ext, 100 samples)
├── diagnosis_step1/                   # Step 1 of 7-stage proposal diagnosis (Mask3D runner + hybrid simulator)
├── diagnosis_step_a/                  # Step A of 7-stage proposal diagnosis
├── diagnosis_w1/                      # W1 — geometric clustering check
├── diagnosis_w1_5/                    # W1.5 — ground filter + clustering sweep
├── diagnosis_beta1/                   # β1 — geometric clustering proposal
├── diagnosis_beta1_5/                 # β1.5 — β1 refinement variant
├── diagnosis_gamma/                   # γ — CenterPoint (learned) proposal
├── diagnosis_option5/                 # Option-5 variant in 7-stage sequence
├── diagnosis_alpha/                   # α — distance-aware hybrid simulation (β1+γ)
├── diagnosis_beta_baseline/           # β baseline — direct OpenYOLO3D application infra (May 2026)
├── diagnosis_beta_baseline_v2/        # β baseline — GT-bug fix + re-measurement (May 2026, mAP=0 confirmed)
├── docs/                              # Setup docs (BASELINE, CONTEXT, Installation, NUSCENES_SETUP, SMOKE_TEST)
├── evaluate/                          # OpenYOLO3D evaluation harness (untouched)
├── models/                            # OpenYOLO3D model code (untouched)
├── preprocessing/                     # Preprocessing utilities
├── pretrained/                        # Pretrained checkpoint configs
├── proposal/                          # Proposal stage code
├── results/                           # Per-run outputs (gitignored except experiment_tracker.md)
├── scripts/                           # Run scripts
├── utils/                             # Shared utilities
├── run_evaluation.py                  # OpenYOLO3D evaluation entry (indoor reference)
├── run_nuscenes.py                    # nuScenes dataloader entry
├── run_nuscenes_smoke_test.py         # Smoke test entry
└── single_scene_inference.py          # OpenYOLO3D single-scene entry (indoor reference)
```

**Diagnosis run outputs** (under `results/`, May 2026):

```
results/
├── smoke_nuscenes/                    # Stage-2 OpenYOLO3D integration smoke test
├── smoke_regression/                  # Regression check for smoke test
├── sanity_overlay.png                 # LiDAR→camera projection sanity visual
├── sanity_trainval/                   # Sanity check on trainval split
├── diagnosis/                         # Tier 1/2 setup-level findings
├── diagnosis_tier2/                   # Tier 2-Ext (100 samples)
├── diagnosis_tier2_trainval/          # Tier 2-Ext on trainval
├── w1_clustering_check/               # W1 clustering check
├── w1_5_diagnostic_sweep/             # W1.5 parameter sweep (phase_a/b/c)
├── diagnosis_step1/                   # 7-stage Step 1
├── diagnosis_step_a/                  # 7-stage Step A
├── diagnosis_beta1/                   # β1 — 36.1% recall, geometric sweet spot
├── diagnosis_beta1_5/                 # β1.5 variant
├── diagnosis_gamma/                   # γ — 35.2% recall, opposite distance bias to β1
├── diagnosis_option5/                 # Option-5 variant
├── diagnosis_alpha/                   # α — 46.7% recall (β1 far + γ near, threshold=35m)
├── diagnosis_beta_baseline/           # β baseline v1 — mAP=0 (Mask3D indoor pretrain does not transfer)
└── diagnosis_beta_baseline_v2/        # β baseline v2 — GT bug fixed, mAP=0 confirmed structurally
```

## 7. Roadmap

| Stage | Period | Status |
|-------|--------|--------|
| 1. nuScenes dataloader | May 2026 | ✅ Done |
| 2. OpenYOLO3D integration smoke test | May 2026 | ✅ Done |
| 3. Diagnosis (7-stage + α + β) | May 2026 | ✅ Done (preliminary) |
| 4. Step 4 — Outdoor extension (LiDAR proposal + temporal layer integration) | Jul–Aug 2026 | ⏳ Pending Indoor Step 1–3 completion |
| 5. Full nuScenes experiments | Aug 2026 | Pending |
| 6. Graduation thesis + CVPR 2027 submission | Sep–Nov 2026 | Pending (cross-worktree) |

## 8. Setup

See [`docs/NUSCENES_SETUP.md`](docs/NUSCENES_SETUP.md) for nuScenes data layout and environment setup. The development environment `openyolo3d-dev` is a clone of OpenYOLO3D's `openyolo3d` env with `nuscenes-devkit` and `mmdet3d==1.4.0` added (the latter for γ / CenterPoint diagnosis).

## 9. Branching policy

- `main` — stable, OpenYOLO3D base + verified integration only
- `feature/diagnosis` — current; holds May 2026 diagnosis code + results
- (historical) `feature/nuscenes-dataloader`, `backup/initial-pipeline`

Stages merge to `main` only after end-to-end verification.

## 10. License and acknowledgments

Built on top of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D). Uses [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [Mask3D](https://github.com/JonasSchult/Mask3D) (indoor proposal reference only — does not transfer outdoor; see §3.4), [CenterPoint](https://github.com/tianweiy/CenterPoint) via `mmdet3d` (γ outdoor proposal), and the [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit).

Indoor companion repo (canonical): [github.com/hyuzenn/OpenYOLO3D](https://github.com/hyuzenn/OpenYOLO3D).

This work is conducted as a graduation thesis project.
