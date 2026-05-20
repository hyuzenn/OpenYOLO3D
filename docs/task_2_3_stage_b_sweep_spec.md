# Task 2.3 — Stage B (β1+α hybrid) + M12 sweep — design + Stage 1 findings

Date: 2026-05-20
Branch: main @ ef93063 (post Task 2.2)
Env: openyolo3d-dev

## ⚠️ Stage 1 critical finding — the α "union" is a GT oracle, not a deployable merger

The spec assumes `diagnosis_alpha/union_strategies.py` has a deployable
`distance_aware_union(threshold=35m)` that merges β1 + γ proposals into one set.
**It does not.** What exists is `strategy_distance_aware(beta1_masks, gamma_masks,
gt_boxes, ego_pose, pc_xyz, distance_threshold_m)` — a **diagnostic** function that:

- takes **ground-truth boxes** as input,
- matches each GT against both sources,
- returns per-GT M/L/D/miss case counts (the "M-rate"),
- picks, per GT, whichever source matched it (γ near / β1 far) — *using GT distance*.

So the **May 46.7% M-rate is an oracle upper bound**: it presupposes you know each
GT's distance to choose the right source. It is **not achievable at runtime** and
**cannot be compared to a detection mAP** the way spec §4.1 plans. Treating "Stage B
mAP vs 46.7%" as a consistency check would be a category error.

### Two more β1-as-proposal-source limitations

1. **β1 boxes are axis-aligned** (`cluster_bbox` = [xmin..zmax] AABB in ego frame,
   no yaw). nuScenes detection mAP is orientation-aware, so β1-sourced vehicle
   boxes will score poorly on heading regardless of recall.
2. **β1 has no class and no score** — only geometry (centroid, AABB, point count).
   Class comes from YOLO-World downstream (consistent with γ); score must be
   synthesised (we use a point-count pseudo-score).

### Decision (autonomous)

- Implement a **deployable, GT-free distance-aware union** that follows the *same
  principle* the diagnosis validated (γ preferred < 35 m, β1 preferred ≥ 35 m, with
  cross-source dedup), but does **not** use GT. This is the honest runtime version.
- Run Stage B as a legitimate experiment: *does a deployable β1+γ union beat γ-only
  at runtime?* — while **explicitly not** claiming the 46.7% oracle figure.
- Run the **M12 threshold sweep regardless of Stage B's outcome** (decoupling the
  spec's "PBS B only if PBS A scenario P" gate): the sweep is independent of Stage B
  and runs on the γ pipeline already validated in v03, so the gate's rationale
  (don't run dependent work on a broken pipeline) does not apply. This guarantees
  the high-value, low-risk result lands.

This premise correction is the headline item for the user's review.

## Design

### Stage B — hybrid proposal source

`nuscenes_evaluator.py` gains `proposal_source ∈ {gamma, beta1, hybrid}`:
- `gamma`: CenterPoint only (Stage A path; unchanged).
- `beta1`: HDBSCAN clusters → proposal dicts (axis-aligned, yaw=0, pseudo-score).
- `hybrid`: γ ∪ β1 via `_hybrid_distance_aware_union` (deployable, GT-free):
  - near (<35 m): keep all γ; add β1 only where no γ within `dedup_dist_m`.
  - far (≥35 m): keep all β1; add γ only where no β1 within `dedup_dist_m`.
- β1 config: adapters default `ClusteringConfig` (ground z_threshold −1.4,
  min_cluster_size 20, eps 0.5, max_distance 100 m).
- Axes: baseline, M11, M12 (threshold 0.95) — same as Stage A.

### M12 sweep

- `proposal_source = gamma` (Stage A path).
- Axes `M12_thr080 / M12_thr085 / M12_thr090`: each installs `BayesianGate` with
  `threshold ∈ {0.80, 0.85, 0.90}`, `prior=0.5`, `detection_likelihood=0.8`
  (Indoor defaults). Lower threshold → more lenient → more confirmations →
  expected mAP recovery, at the cost of higher lsc.
- Indoor `method_12_bayesian.py` unchanged — only `BayesianGate(threshold=X)`
  instantiation differs, handled in the evaluator's axis dispatch.

## PBS split

| PBS | content | output |
|---|---|---|
| A (`run_task_2_3_stage_b.pbs`) | proposal_source=hybrid, axes baseline/M11/M12, 10 mini scenes | `results/2026-05-20_outdoor_stage_b_v01/` |
| B (`run_task_2_3_m12_sweep.pbs`) | proposal_source=gamma, axes M12_thr080/085/090, 10 mini scenes | `results/2026-05-20_m12_sweep_outdoor_v01/` |

Both run (gate decoupled, see Decision above); sequential per cluster policy.

## Code-change scope (only nuscenes_evaluator.py)

- `+ _beta1_clusters_to_proposals(beta1_out)`
- `+ _hybrid_distance_aware_union(gamma, beta1, threshold_m, dedup_dist_m)`
- evaluator: `proposal_source`, optional β1 generator handle, axis-dispatch for
  `M12_thrXXX`, `install_axis(method_id=…)` override.
- 2 PBS scripts (new), this spec doc (new).
- Indoor classes / streaming infra / adapters / dataloaders / diagnosis_alpha:
  **unchanged**.
