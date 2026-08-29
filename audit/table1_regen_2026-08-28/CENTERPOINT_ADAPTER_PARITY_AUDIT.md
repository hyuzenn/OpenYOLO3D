<title>CenterPoint Adapter Parity Audit</title>

# CenterPoint Adapter Parity Audit — official mmdet3d v1.4.0 vs `adapters/centerpoint_proposals.py`

Written 2026-08-29. git `a282f821df69353fadced4a24d6738ad4ec5be96`.

**Nothing was modified.** No code, no config, no checkpoint, no threshold, no
cache, no manuscript. No inference was rerun. Two new files were *added*
(`audit/table1_regen_2026-08-28/parity_probe/probe_parity.py`,
`scripts/run_parity_probe.pbs`) to run one CPU-only, model-free probe (PBS job
120238, `coss_a6gpu`, `ngpus=0`, exit 0).

All paths below are relative to `/home/rintern16/OpenYOLO3D`. mmdet3d source
paths are relative to
`~/miniconda3/envs/openyolo3d-dev/lib/python3.10/site-packages/mmdet3d/`.

---

## 1. Executive diagnosis

The adapter does **not** reimplement the mmdet3d decode/NMS path — it calls
`mmdet3d.apis.inference_detector`, so decode, score threshold, circle NMS,
top-K, class mapping and the z-convention are all the official code and all
match. The defect is one layer earlier, and it is a **single line of
consequence**:

> `inference_detector` re-runs the model's **full configured test pipeline**,
> including `LoadPointsFromMultiSweeps`. The adapter hands it a point cloud
> that has *already* been 10-sweep aggregated by the nuScenes devkit. The
> pipeline therefore (a) **zeroes the timestamp channel** and (b) **replicates
> the whole cloud ~10×** via `pad_empty_sweeps`.

Measured, not inferred (probe P1, synthetic 20,000-point cloud with 10 distinct
Δt values pushed through the *unmodified* `cfg.test_dataloader.dataset.pipeline`):

| | in | out |
|---|---:|---:|
| points | 20,000 | **199,892** (×9.9946) |
| Δt channel distinct values | 10 (0.00 … 0.45 s) | **1** |
| Δt channel max | 0.45 s | **0.0** |

So the checkpoint — which was trained with a per-point Δt channel — is served,
at every sample in every cache we have ever built, a ~10×-duplicated cloud with
**no temporal information at all**. That is a train/test input mismatch strictly
worse than the single-sweep one this regeneration was launched to fix.

Consequences, all measured:

- **1.486× excess detections** over the anchor after the class-range filter is
  applied to both sides (739,122 vs 497,475), with systematically lower scores
  (mean 0.233 vs 0.293). A degraded input makes the heatmap fire more, weaker.
- **mAVE 1.0716 vs 0.2951**. Velocity *magnitudes* are near-identical to the
  anchor (mean 0.583 vs 0.513 m/s), so this is not a dead head or a frame/unit
  bug — the velocity vectors are simply wrong, which is what a
  timestamp-blind network produces.
- **mAOE 0.5410 vs 0.3046**, **mAP 0.3922 vs 0.5580** — detection-quality
  symptoms of the same cause.

Two *formatting* differences exist independently of the above:

- **Box size ordering is not swapped.** Official swaps `dims[:, [1,0,2]]` at
  nuScenes serialization; our harness does not. NDS-only, worth +0.0443 NDS.
  This lives in the **harness**, not the adapter.
- **The per-class range filter is not applied** on our side. This is
  **metric-neutral** — the devkit's `filter_eval_boxes` applies it to both
  sides — and explains only the raw-JSON count difference, not the metrics.

---

## 2. Official mmdet3d call chain

Actual call chain used to produce the anchor (`scripts/run_official_centerpoint_test_only.pbs`
→ `mmdetection3d_v1.4.0/tools/test.py`):

| # | Stage | Source |
|---|---|---|
| 1 | `LoadPointsFromFile(load_dim=5, use_dim=5)` on `samples/LIDAR_TOP/*.bin` | `datasets/transforms/loading.py` |
| 2 | `LoadPointsFromMultiSweeps(sweeps_num=9, pad_empty_sweeps=True, remove_close=True, use_dim=[0..4])` — loads **9 real prior sweeps** from `info['lidar_sweeps']`, transforms each by `lidar2sensor`, writes `points_sweep[:,4] = ts - sweep_ts` | `loading.py:398-453`; keyframe zeroed at `:413`, sweep Δt at `:446` |
| 3 | `MultiScaleFlipAug3D` → identity `GlobalRotScaleTrans` + `RandomFlip3D` + `PointsRangeFilter([-54,-54,-5, 54,54,3])` | config `test_pipeline` |
| 4 | `Pack3DDetInputs(keys=['points'])` | |
| 5 | `Det3DDataPreprocessor` hard voxelization, `voxel_size=[.075,.075,.2]`, `max_num_points=10`, `max_voxels=(90000,120000)` | config `model.data_preprocessor` |
| 6 | `HardSimpleVFE(num_features=5)` — per-voxel **mean over the first ≤10 points** | `models/voxel_encoders/voxel_encoder.py:42-43` |
| 7 | `SparseEncoder` → `SECOND` → `SECONDFPN` | config |
| 8 | `CenterHead.predict_by_feat` | `models/dense_heads/centerpoint_head.py:693` |
| 9 | `CenterPointBBoxCoder.decode`: top-K `max_num=500` **per task**, `score > 0.1`, `post_center_range ±61.2` | `models/task_modules/coders/centerpoint_bbox_coders.py:154, 203-218` |
| 10 | `circle_nms(min_radius[task_id], post_max_size=83)` per task → ≤ 6×83 = 498 boxes | `centerpoint_head.py:755-776`; `models/layers/box3d_nms.py:186-226` |
| 11 | merge tasks, **`bboxes[:,2] -= bboxes[:,5]*0.5`** (gravity → bottom centre), label offset by task | `centerpoint_head.py:793, 799-803` |
| 12 | `NuScenesMetric._format_lidar_bbox` → `output_to_nusc_box` | `evaluation/metrics/nuscenes_metric.py:485, 560` |
| 13 | `gravity_center` (z + h/2); **`nus_box_dims = box_dims[:, [1,0,2]]`**; `quat = Quaternion(axis=[0,0,1], radians=yaw)`; `velocity = (vx, vy, 0)` in LiDAR frame | `nuscenes_metric.py:582-604` (**size swap at :590**) |
| 14 | `lidar_nusc_box_to_global`: rotate+translate by `lidar2ego`, **drop if `‖centre_xy‖ > class_range[cls]`**, then rotate+translate by `ego2global` | `nuscenes_metric.py:633-668` (**range filter at :657-661**) |
| 15 | attribute rule from `‖velocity‖ > 0.2` | `nuscenes_metric.py:517-536` (already ported, commit `438f121`) |

## 3. Our adapter call chain

| # | Stage | Source |
|---|---|---|
| 1 | `NuScenesLoader._load_lidar_ego` → devkit `LidarPointCloud.from_file_multisweep(nsweeps=10, ref_chan=LIDAR_TOP)`, returns `(x,y,z,intensity)` in **ego** frame + per-point Δt as a 5th column | `dataloaders/nuscenes_loader.py:73-101` |
| 2 | `CenterPointProposalGenerator.generate`: ego→lidar via `inv(T_lidar_to_ego)`, assemble `(N,5) = [x,y,z,intensity,Δt]`, `tofile(tmp.bin)` | `adapters/centerpoint_proposals.py:102-114` |
| 3 | `inference_detector(model, tmp.bin)` | `adapters/centerpoint_proposals.py:118` |
| 3a | → **the entire official test pipeline runs again** on that .bin: `LoadPointsFromFile`, then `LoadPointsFromMultiSweeps` with **no `lidar_sweeps` key present** ⇒ `points.tensor[:,4] = 0` and 9 extra copies of the cloud appended | `apis/inference.py:150-171`; `loading.py:412-422` |
| 4 | steps 5-11 of §2 — **identical official code** | |
| 5 | `scores >= self.config.score_threshold` (0.0 for the cache) | `centerpoint_proposals.py:130` |
| 6 | `bboxes_lidar[:,2] += bboxes_lidar[:,5]*0.5` (bottom → gravity centre) | `centerpoint_proposals.py:144-145` |
| 7 | centre lidar→ego; serialize `bbox_lidar = [x,y,z,dx,dy,dz,yaw,vx,vy]` + `centroid_ego` | `centerpoint_proposals.py:147-175` |
| 8 | harness: `box_q_ego = lidar_to_ego_q · Rz(yaw)`, `global_q = ego_quat · box_q_ego`, `v_g = R_ego2global · R_lidar2ego · [vx,vy,0]` | `method_scannet/streaming/nuscenes_native_evaluator.py:771-779` |
| 9 | harness: `_detection_box_dict` emits `size = [dx, dy, dz]` — **no `[1,0,2]` swap** | `method_scannet/streaming/nuscenes_evaluator.py:456` |
| 10 | evaluator: `add_center_dist` + `filter_eval_boxes` (class range, `num_pts==0`, bike-rack) then devkit `accumulate`/`calc_ap`/`calc_tp` | `diagnosis_beta_baseline/evaluate_nuscenes.py:96-101` |

## 4. Parity table

| Stage | Official mmdet3d | Our adapter | Exact difference | Expected impact | Evidence | Class |
|---|---|---|---|---|---|---|
| Input channels | `(x,y,z,intensity,ring)` from disk, ch.4 later overwritten with Δt | `(x,y,z,intensity,Δt)` written by us, **then overwritten with 0** | ch.4 reaches the model as **all-zero** instead of real sweep lag | Velocity head loses its only temporal cue; heatmap degrades | probe P1: `dt_channel_out.n_distinct = 1`, `max = 0.0`; `loading.py:413` | **REAL DIFFERENCE** |
| Sweep representation | 9 real prior sweeps, each with its own Δt, motion-compensated by `lidar2sensor` | devkit `from_file_multisweep(nsweeps=10)` aggregate, **then duplicated ~10×** by `pad_empty_sweeps` | ~10× redundant points, single timestamp | Voxel means biased toward the first copies; density statistics off-distribution | probe P1: 20,000 → 199,892 points (×9.99); `loading.py:416-422` | **REAL DIFFERENCE** |
| Point aggregation frame | keyframe LiDAR frame | keyframe LiDAR frame (devkit `ref_chan=LIDAR_TOP`), round-tripped lidar→ego→lidar | float32 round-trip only | negligible | `nuscenes_loader.py:90-92`, `centerpoint_proposals.py:103-105` | MATCH |
| Close-point removal | `remove_close(1.0)` on sweeps only, keyframe keeps them | devkit `min_distance=1.0` on **all** sweeps incl. reference | a few thousand points near the ego | negligible | devkit `from_file_multisweep` default | IRRELEVANT |
| Voxelization | `Det3DDataPreprocessor`, same cfg | same object, same cfg | none | — | config `model.data_preprocessor` | MATCH |
| VFE | `HardSimpleVFE` mean of first ≤10 pts/voxel | same | none in code; **input differs** (see duplication) | per-voxel mean subsamples the duplicated block | `voxel_encoder.py:42-43` | MATCH (code) |
| Model forward | official | official (`init_model` + same ckpt) | none | — | `centerpoint_proposals.py:73` | MATCH |
| Heatmap / regression decode | `CenterPointBBoxCoder.decode` | same object | none | — | `centerpoint_bbox_coders.py:152-218` | MATCH |
| Score threshold | `score > 0.1` in the coder | same coder; adapter then applies `>= 0.0` | adapter filter is a strict **no-op** | none | probe P3: cache min score = **0.10000002** | MATCH |
| NMS | `circle_nms(min_radius[task], post_max_size=83)` | same | none | — | `centerpoint_head.py:763-769` | MATCH |
| Top-K | coder `max_num=500`/task; NMS `post_max_size=83`/task | same | none | — | `centerpoint_bbox_coders.py:154` | MATCH |
| Class mapping | task order flattened: car / truck,cv / bus,trailer / barrier / moto,bicycle / ped,cone | `NUSC_10` identical tuple | none | — | `centerpoint_proposals.py:37-40` vs config `tasks` | MATCH |
| z convention | head subtracts h/2, `gravity_center` adds it back | head subtracts h/2, adapter adds it back | none | — | `centerpoint_head.py:793` vs `centerpoint_proposals.py:145` | MATCH |
| Box dimensions | `nus_box_dims = dims[:, [1,0,2]]` → `(w,l,h)` | `size = [dx,dy,dz]` → `(l,w,h)` | **length/width swapped** | mASE 0.7101 vs 0.2537; **NDS-only** (mAP is centre-distance matched) | `nuscenes_metric.py:590` vs `nuscenes_evaluator.py:456`; arm C: mASE → 0.2675, NDS +0.0443 | **REAL DIFFERENCE** (harness) |
| Yaw | `Quaternion(axis=z, radians=yaw_lidar)`, then `·lidar2ego`, then `·ego2global` | `ego_quat · lidar_to_ego_q · Rz(yaw_lidar)` | none | — | `nuscenes_metric.py:592, 653-665` vs `nuscenes_native_evaluator.py:771-773` | MATCH |
| Velocity | `(vx,vy,0)` LiDAR-frame, rotated by `lidar2ego` then `ego2global` (translation does not touch velocity) | `R_ego2global · R_lidar2ego · [vx,vy,0]` | none | — | `nuscenes_metric.py:593, 653-665` vs `nuscenes_native_evaluator.py:777-779` | MATCH |
| Coordinate frame | lidar → ego → global | lidar → ego → global | none | — | as above | MATCH |
| Per-class range filter | dropped before serialization (`nuscenes_metric.py:657-661`) | not applied | raw JSON keeps far boxes | **none** — devkit `filter_eval_boxes` applies it to both sides | `evaluate_nuscenes.py:96-101`; probe P3 (below) | IRRELEVANT TO GAP |
| Attribute rule | speed-based rule at `nuscenes_metric.py:517-536` | ported verbatim in `438f121` | none | — | 0/497,475 mismatches | MATCH |

## 5. Detection-count discrepancy — resolved

Raw counts: cache 1,111,841 (184.72/sample) vs anchor 497,475 (82.65/sample).
Two separate effects, measured (probe P3):

| | total | per sample |
|---|---:|---:|
| cache, raw | 1,111,841 | 184.72 |
| cache, after per-class range filter (ego-frame, exactly `nuscenes_metric.py:657-661`) | **739,122** | 122.80 |
| anchor (already range-filtered by mmdet3d) | 497,475 | 82.65 |
| **residual excess ratio** | **1.4857×** | |

- **33.5 % of the raw gap is the class-range filter**, which official applies
  before writing JSON and we do not. It is metric-neutral: the corrected
  evaluator re-applies it to both sides (arm C's own post-filter count is
  738,802, within 0.04 % of the 739,122 computed here from `centroid_ego`).
- **The remaining 1.486× is a genuine excess of detections.** It is *not* a
  threshold, NMS, top-K or budget difference: every one of those is the same
  official code (§4), the coder's `score > 0.1` is binding (cache min score
  0.10000002, so the adapter's 0.0 threshold does nothing), and the per-task
  cap of 83 is untouched. It is the model firing more on a degraded input.

Per class, cache-after-range vs anchor:

| class | cache raw | cache ranged | anchor | ranged/anchor |
|---|---:|---:|---:|---:|
| construction_vehicle | 70,848 | 53,136 | 20,869 | **2.55** |
| bus | 24,619 | 13,188 | 5,490 | **2.40** |
| motorcycle | 106,376 | 73,920 | 34,341 | **2.15** |
| trailer | 39,424 | 27,916 | 14,855 | 1.88 |
| pedestrian | 250,184 | 146,898 | 92,523 | 1.59 |
| truck | 67,856 | 53,889 | 37,565 | 1.44 |
| car | 196,356 | 153,119 | 114,647 | 1.34 |
| bicycle | 110,163 | 83,507 | 65,878 | 1.27 |
| traffic_cone | 119,779 | 81,709 | 65,554 | 1.25 |
| barrier | 126,236 | 51,840 | 45,753 | 1.13 |

Score distribution is shifted down accordingly (cache mean 0.233 / p90 0.492;
anchor 0.293 / p90 0.711) — more, weaker detections, the signature of an
off-distribution input rather than of a post-processing difference. Note the
excess is worst in the rare classes that share a head task with a common one
(construction_vehicle with truck, bus with trailer, motorcycle with bicycle):
the pre-NMS top-500 budget is per *task*, so extra low-quality peaks on one
class crowd its task partner.

## 6. Yaw / orientation analysis

**Conventions MATCH at source level.** Official: `box_yaw = bbox3d.yaw` (=
`tensor[:,6]`, no negation for `LiDARInstance3DBoxes`), `quat =
Quaternion(axis=[0,0,1], radians=box_yaw)` (`nuscenes_metric.py:584, 592`),
then `box.rotate(Quaternion(matrix=lidar2ego))` and
`box.rotate(Quaternion(matrix=ego2global))` (`:653, :664`). Ours:
`global_q = ego_quat · lidar_to_ego_q · Rz(yaw_lidar)`
(`nuscenes_native_evaluator.py:771-773`) — the same composition in the same
order. No `-yaw`, no `+π`, no camera-convention leakage on either side.

Therefore **mAOE 0.5410 vs 0.3046 is not a convention bug**; it is a
detection-quality symptom of the input defect. A convention error would produce
a bimodal error concentrated at π/2 or π and would also wreck classes with
near-square footprints; instead the error is a broad degradation.

## 7. Velocity analysis

Conventions MATCH (§4). The decisive measurement is the *distribution*, not the
frame (probe P3):

| speed ‖v‖ (m/s) | cache | anchor |
|---|---:|---:|
| mean | 0.583 | 0.513 |
| p90 | 1.262 | 1.193 |
| max | 18.50 | 17.32 |
| fraction < 0.2 m/s | 0.784 | 0.834 |

The magnitudes are essentially the same. A frame error (missing or doubled
`lidar2ego`/`ego2global` rotation) preserves magnitude but rotates direction —
it *could* look like this. But the composition is identical to official line
for line, and a frame error would be a fixed rotation per sample that the
attribute rule (speed-thresholded, magnitude-only) would not notice, whereas
our attribute error is *also* inflated (0.2700 vs 0.1844) — which requires the
speeds themselves to be wrong per box, not merely rotated.

The reading consistent with everything: with Δt ≡ 0 the network can still infer
a plausible speed *magnitude* from the motion smear present in the aggregated
cloud, but cannot resolve direction or sign. mAVE ≈ 1.07 m/s against a mean GT
speed of the same order is what an essentially uncorrelated velocity vector
gives. **The cache velocity is in the LiDAR frame** (as official expects) —
this was verified by construction: the harness's rotation is the only
transform applied, and it matches official.

**Do not add a second rotation.** The rotation path is correct.

## 8. Existing box-size defect

Restated from `DIAGNOSIS_ANCHOR_GAP.md`, now with the official source
citation: `nuscenes_metric.py:590` performs `nus_box_dims = box_dims[:, [1,0,2]]`
because mmdet3d's `LiDARInstance3DBoxes.dims` is `(l, w, h)` while
nuScenes `Box.wlh` is `(w, l, h)`. `nuscenes_evaluator.py:456` omits the swap
and its docstring asserts the opposite. Measured effect (job 120230, arm C):
mASE 0.7101 → 0.2675 (anchor 0.2537), NDS +0.0443, **mAP unchanged to six
decimals** — mAP is matched by 2-D centre distance and cannot move with size.

## 9. Sample-level evidence

Existing artifacts only; nothing rerun.

- All 6,019 tokens present on both sides; 6,019 in common. GT identical between
  arm C and the anchor evaluation (`n_gt_boxes = 121,861` in both
  `diag_cache_direct/C_cache_direct_wlh/eval_summary.json` and
  `audit/official_centerpoint_ref/evaluator_fix_validation/test_D_our_10sweep/eval_summary.json`).
- Sample `000868a7…`, score > 0.3: cache 9 construction_vehicle / 4 bicycle vs
  anchor 2 / 0; cache top score 0.842 vs anchor 0.878. Consistent with the
  population-level excess and score deflation above.
- Cache min score 0.10000002 over all 1,111,841 boxes ⇒ the head threshold is
  what bounds the set, not the adapter.
- Per-class and per-metric residuals: `diag_cache_direct/*/per_class.json`.

## 10. Confirmed vs suspected

### Confirmed bugs

1. **`inference_detector` re-runs `LoadPointsFromMultiSweeps` on an
   already-aggregated cloud** → Δt zeroed and cloud duplicated ~10×.
   `adapters/centerpoint_proposals.py:118` + `apis/inference.py:150-171` +
   `loading.py:412-422`. Directly measured (probe P1). Explains the detection
   excess, the score deflation, mAVE, mAOE and the mAP gap.
2. **Box size ordering not swapped** at nuScenes serialization.
   `nuscenes_evaluator.py:456` vs `nuscenes_metric.py:590`. Directly measured
   (job 120230 arm C). NDS-only, +0.0443.

### Suspected / not isolated

3. The **relative weight** of the two sub-effects of (1) — Δt-zeroing versus
   10× duplication — is not separated. Both are present in every cache. Test
   T2 (§12) separates them if it matters; it does not need separating to apply
   the fix, since the correct pipeline removes both at once.
4. Whether, after fixing (1), any residual gap to 0.5580/0.6458 remains from
   the devkit-vs-mmdet3d aggregation difference (devkit `from_file_multisweep`
   removes close points from the reference sweep, mmdet3d does not; devkit
   uses 10 sweeps incl. keyframe, mmdet3d 1 + 9). Expected small; measurable
   by T1.

### Harmless implementation differences

- Adapter `score_threshold=0.0`: strict no-op below the coder's 0.1.
- `nms_iou_threshold=0.20` field: documented as informational, never used.
- Missing per-class range filter: metric-neutral, the devkit applies it.
- ego↔lidar float round-trip; devkit reference-sweep close-point removal.

## 11. Minimum correction set

**Not implemented. Listed for a decision.**

### C1 — stop the test pipeline from re-aggregating (required for parity)

- **File** `adapters/centerpoint_proposals.py`
- **Function** `CenterPointProposalGenerator.__init__` / `.generate`
- **Lines** 71-73 (build) and 118 (call)
- **Change**: stop using `inference_detector`. Build the pipeline once in
  `__init__` from `model.cfg.test_dataloader.dataset.pipeline` **with the
  `LoadPointsFromMultiSweeps` entry removed**, and call `model.test_step` on it
  — because the adapter's input is *already* a 10-sweep aggregate carrying the
  correct Δt. Everything else in the pipeline (`LoadPointsFromFile`,
  `MultiScaleFlipAug3D`/`PointsRangeFilter`, `Pack3DDetInputs`) must be kept
  byte-identical.
- **Why**: `loading.py:413` unconditionally zeroes ch.4 and `:416-422`
  duplicates the cloud when no `lidar_sweeps` key is present, which is always
  the case on the `inference_detector` path.
- **Expected effect**: the model receives the intended 10-sweep input with real
  Δt. Detection count should fall toward the anchor's, scores rise, mAVE and
  mAOE drop sharply, mAP rise toward 0.5580.
- **Test**: T1 (§12).
- **Rerun required**: **yes** — the detection set changes, so the phase-1
  proposal cache must be rebuilt (6,019 samples, one A100, ~8 h).

### C2 — swap box dimensions at nuScenes serialization (required for NDS parity)

- **File** `method_scannet/streaming/nuscenes_evaluator.py`
- **Function** `_detection_box_dict`
- **Line** 456 (and the docstring at 434-437, which states the wrong convention)
- **Change**: emit `size = [bbox_lidar[4], bbox_lidar[3], bbox_lidar[5]]`.
- **Why**: mmdet3d dims are `(l,w,h)`; nuScenes `size` is `(w,l,h)`;
  `nuscenes_metric.py:590` does exactly this swap.
- **Expected effect**: mASE 0.7101 → ~0.2675, NDS +0.0443, mAP unchanged.
- **Test**: already measured — job 120230 arm C.
- **Rerun required**: **no**. It is a CPU replay off the existing cache.

### Explicitly NOT in the correction set

- The per-class range filter (metric-neutral).
- Any yaw or velocity transform (both already match official).
- Any threshold, NMS, top-K, budget or class-map change.

**Ordering note.** C2 alone is safe and cheap but produces a table whose
absolute numbers are still not comparable to published CenterPoint. C1 changes
every number in Table 1 and every downstream arm. Doing C2 without C1 is a
partial fix; doing C1 requires redoing the whole regeneration.

## 12. Decisive follow-up tests

Cheapest-first. None of these is a full 6,019-sample eval.

- **T1 — pipeline parity smoke (GPU, ~15 min, ~200 samples).** *Genuinely needs
  a GPU*: it is the only way to see the corrected model output, and no cached
  artifact contains it. Run the corrected adapter (C1) over ~200 val samples,
  then compare **box for box against the anchor JSON on the same tokens**:
  count/sample, score distribution, per-class counts, and greedy 2 m centre
  matching to report translation / yaw / velocity residuals. Gate: count within
  ~10 % of anchor and median velocity residual < 0.3 m/s. Only after PASS spend
  the 8 GPU hours on the full cache.
- **T2 — Δt vs duplication attribution (GPU, ~10 min, optional).** Same 200
  samples, three arms: (a) pipeline-fixed with real Δt, (b) pipeline-fixed with
  Δt forced to 0, (c) current path. Separates suspected item 3. Skip unless the
  attribution is wanted for the write-up.
- **T3 — C2 replay (CPU, already done).** Job 120230 arm C.
- **T4 — range-filter neutrality (CPU, already done).** Probe P3: 739,122
  computed from `centroid_ego` vs 738,802 from the evaluator's own filter,
  0.04 % apart — confirms the filter is not a metric-relevant difference.

## 13. Must `centerpoint_proposals.py` be modified?

**Yes.** The defect is on line 118 (`inference_detector`), which silently
re-applies a sweep-aggregation transform the adapter has already performed. It
cannot be fixed anywhere else: the pipeline is chosen inside
`inference_detector` from the model config, and passing a numpy array instead
of a path does not help (`apis/inference.py:139-141` only swaps the *loader*
for `LoadPointsFromDict`; `LoadPointsFromMultiSweeps` still runs and still
zeroes ch.4).

But it is **not the only file**: the box-size defect is in
`method_scannet/streaming/nuscenes_evaluator.py:456`, not in the adapter.

## 14. Is an inference rerun required?

**Yes, for C1.** The detection set itself changes, so the frozen proposal cache
must be rebuilt for all 6,019 val samples on an A100 (~8 h), and phases 2-4 of
the Table 1 regeneration re-run off the new cache (CPU replays). Gate that
spend behind T1.

**No, for C2.** CPU replay off the existing cache.

## 15. What was NOT changed

- No source file was edited. `adapters/centerpoint_proposals.py`,
  `method_scannet/streaming/*.py`, `diagnosis_beta_baseline/*`, all configs and
  all PBS scripts for phases 1-4 are byte-identical to `a282f82`.
- No checkpoint, config, score threshold, NMS parameter, association threshold,
  temporal window, budget or class set was touched.
- No cache was modified or deleted. No inference was run.
- No manuscript file was touched. Table 1 LaTeX untouched.
- No previous audit artifact was overwritten; everything new lives in
  `audit/table1_regen_2026-08-28/parity_probe/`.
- Added files (new, additive only): `parity_probe/probe_parity.py`,
  `parity_probe/probe_results.json`, `scripts/run_parity_probe.pbs`,
  `logs/parity_probe.120238.log`, this report.

---

## Acceptance criteria

1. **Why 1,111,841 vs 497,475?** 33.5 % is the per-class range filter official
   applies before serialization and we do not (metric-neutral); the remaining
   **1.486×** is a genuine detection excess caused by the degraded input
   (Δt≡0, cloud duplicated ~10×). Not a threshold/NMS/top-K difference — all
   of those are the same official code and the coder's 0.1 threshold is
   binding (cache min score 0.10000002).
2. **Why mAP 0.3922 vs 0.5580?** Same cause: the checkpoint is served an
   off-distribution, timestamp-free input, producing more and weaker
   detections (mean score 0.233 vs 0.293).
3. **Why mAOE 0.541 vs 0.305?** Not a convention bug — yaw composition is
   identical to official line for line. Detection-quality symptom of (1).
4. **Why mAVE 1.072 vs 0.295?** Velocity frame and rotation match official;
   speed *magnitudes* match the anchor (0.583 vs 0.513 mean). The vectors are
   wrong because the network never receives Δt.
5. **Which differences originate in `centerpoint_proposals.py`?** Exactly one:
   the `inference_detector` call at line 118.
6. **Which are only formatting?** The box size ordering (harness, NDS-only) and
   the missing class-range filter (metric-neutral).
7. **Proven vs suspected?** Proven: the pipeline defect (probe P1) and the size
   swap (job 120230). Suspected: the split between Δt-zeroing and duplication,
   and any small residual from devkit-vs-mmdet3d aggregation.
8. **Minimum correction?** C1 (adapter pipeline) + C2 (size ordering). Nothing
   else.
9. **Validatable without a full rerun?** The *fix* can be validated on ~200
   samples (T1). The *table* cannot — C1 requires a full cache rebuild.
10. **Is `centerpoint_proposals.py` the right file?** Yes for the dominant
    defect, but not the only one: C2 is in `nuscenes_evaluator.py:456`.

---

# Verdict

> **MODIFY `adapters/centerpoint_proposals.py`** — the dominant defect is its
> `inference_detector` call, which re-applies `LoadPointsFromMultiSweeps` to an
> already-aggregated cloud, zeroing the timestamp channel and duplicating the
> point cloud ~10× (measured: 20,000 → 199,892 points, Δt 10 distinct values →
> 1, all zero).
>
> **Also modify `method_scannet/streaming/nuscenes_evaluator.py:456`** for the
> `(l,w,h)` → `(w,l,h)` swap.
>
> **Run T1 (≈200-sample GPU parity smoke) before spending the ~8 GPU hours on a
> full cache rebuild.**

Nothing was modified in this task. No Table 1 regeneration, no manuscript
change, no checkpoint/config change, no threshold tuning.
