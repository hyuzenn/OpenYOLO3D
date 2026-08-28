# Table 1 regeneration — experiment matrix

Written **2026-08-28**, before any regenerated metric was read.
Repo `/home/rintern16/OpenYOLO3D`, branch `master`, git HEAD at audit time
`438f121` (contains `ad94732` evaluator correction + `438f121` official
attribute rule).

Table 1 is `\label{tab:detmatch}` in `paper_iccv_draft/sec/3_finalcopy.tex:47`,
"Detection-budget-matched comparison", nuScenes val, 150 scenes / 6,019 samples.

---

## 1. Rows as currently published

| # | Frame | Arm | Artifact that produced the published number | Boxes | mAP | NDS |
|---|---|---|---|---:|---:|---:|
| 1 | Sensor (ego) | Baseline (unfiltered) | `results/2026-07-18_e1_grid_v01/cells/gamma_ego/axis_baseline` | 1,029,380 | 0.3408 | 0.3150 |
| 2 | Sensor | Control (threshold) | `results/2026-07-30_e2c_retro_thrmatch_v01/cells/ctrl_ego` | 360,309 | 0.3324 | 0.3110 |
| 3 | Sensor | Control (accumulation) | `audit/cbmot/cells/cbmot_retro_ego_N3_parallel_addition_noise0.05` | 360,309 | — | — |
| 4 | Sensor | Confirmation (N=3) | `results/2026-07-30_e2c_retro_thrmatch_v01/cells/retro_ego` | 360,309 | 0.2023 | 0.2518 |
| 5 | World (global) | Baseline (unfiltered) | `results/2026-07-18_e1_grid_v01/cells/gamma_global/axis_baseline` | 1,029,380 | 0.3408 | 0.3150 |
| 6 | World | Control (threshold) | `results/2026-07-30_e2c_retro_thrmatch_v01/cells/ctrl_global` | 754,500 | 0.3396 | 0.3143 |
| 7 | World | Control (accumulation) | `audit/cbmot/cells/cbmot_retro_global_N3_parallel_addition_noise0.05` | 754,500 | — | — |
| 8 | World | Confirmation (N=3) | `results/2026-07-30_e2c_retro_thrmatch_v01/cells/retro_global` | 754,500 | 0.2900 | 0.3096 |

mAP/NDS are deliberately blank for the accumulation control in the published
table and stay blank here — the CBMOT arm is scored on tracking metrics only.

## 2. Fixed experimental conditions (unchanged by this regeneration)

| Condition | Value | Where fixed |
|---|---|---|
| Checkpoint | `pretrained/centerpoint_nuscenes/centerpoint_0075voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220810_011659-04cb3a3b.pth` | unchanged |
| Detector config | `centerpoint_voxel0075_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py` | unchanged |
| Split | nuScenes v1.0-trainval, `val`, 150 scenes, 6,019 samples | `--scene-split val` |
| Class set | nuScenes-10 | `NUSC_10` |
| Detector score threshold | 0.0 (cache is unfiltered) | `--score-threshold 0.0` |
| Association distance | 2.0 m | `--association-threshold-m 2.0` |
| Association max_age | 5 | default |
| Confirmation window | N = 3 | `--m11-N 3` |
| Emission | retrospective | `--retro-emission` |
| Matched-control rule | rank by score, cut at K = retro arm's box count | `scripts/e2_thrmatch_pick.py` |
| CBMOT variant | `parallel_addition`, noise 0.05, max_age 5 | `audit/cbmot/PRESPECIFICATION.md` |
| Evaluator | corrected custom (`ad94732`) + official attribute rule (`438f121`) | HEAD |

## 3. The one condition that changes, and why

**Every published Table 1 row was fed by a single-sweep CenterPoint cache.**

Evidence, all pre-existing:

- `method_scannet/streaming/nuscenes_native_evaluator.py:1328-1329` (before this
  regeneration) hardcoded `loader.multi_sweep = False; loader.num_sweeps = 1`,
  overriding the YAML.
- `configs/nuscenes_trainval.yaml:13-14` — `multi_sweep: false`, `num_sweeps: 1`.
- `audit/official_centerpoint_ref/verify_sweep_count.log` measures that path at
  **34,720 points / 4 channels / no Δt**, against **250,091 points / 5 channels
  / 10 distinct Δt** under `configs/nuscenes_trainval_multisweep.yaml`.
- `audit/BASELINE_INTEGRITY_RESULTS.md` §1: "the single-sweep input problem is
  untouched and remains the dominant cause."

The checkpoint was trained with `LoadPointsFromMultiSweeps(sweeps_num≈10)`, so
single-sweep input is a train/test distribution mismatch. The validated 10-sweep
anchor is **mAP 0.5580 / NDS 0.6458**; the published Table 1 baseline is 0.3408.

**Change made:** the hardcode at `nuscenes_native_evaluator.py:1328-1329` is
removed so the sweep setting follows the config. `configs/nuscenes_trainval.yaml`
is untouched (still single-sweep), so every historical run reproduces
byte-for-byte; the regeneration selects
`configs/nuscenes_trainval_multisweep.yaml`. No other code, config, checkpoint
or threshold is modified.

Derived quantities that necessarily move with the new detections: the emitted
box budgets, and therefore the matched-control threshold values. The matching
*rule* is preserved exactly.

## 4. Execution phases

| Phase | Job | Resource | Output |
|---|---|---|---|
| 1 | `scripts/run_table1_cpcache_10sweep.pbs` | A100 (`coss_agpu`), GPU gate asserts A100, `ece-tgpu3` excluded | `results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity/` (6,019 pkl) |
| 2 | `scripts/run_table1_arms_10sweep.pbs` | CPU (`coss_a6gpu`, ngpus=0) | `audit/table1_regen_2026-08-28/phase2_arms/cells/{gamma,retro,ctrl}_{ego,global}` |
| 3 | CBMOT accumulation controls | CPU | accumulation rows 3 and 7 |
| 4 | aggregation + sanity checks | CPU | `table1_regenerated_results.json`, `TABLE1_REGENERATION_REPORT.md` |

Phase 2 replays hit the proposal cache, so `loader._load()` is never called and
no point cloud is read; the sweep setting is inert during replay and is passed
only to document provenance.

The old artifacts (`results/2026-07-18_e1_grid_v01`,
`results/2026-07-30_e2c_retro_thrmatch_v01`, `audit/cbmot/cells`, and the
single-sweep cache) are **not** overwritten.
