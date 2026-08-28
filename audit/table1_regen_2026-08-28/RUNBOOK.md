# Table 1 regeneration — runbook

Everything is scripted and syntax-checked. Phase 1 is queued; phases 2–4 are
submitted in order as each predecessor completes.

## Order of operations

| # | Command | Queue | Gate before running |
|---|---|---|---|
| 0 | `qsub scripts/run_table1_preflight.pbs` | `coss_a6gpu` (CPU) | none |
| 1 | `qsub scripts/run_table1_cpcache_10sweep.pbs` | `coss_agpu` (A100) | preflight PASS; **commit the sweep change** |
| 2 | `qsub scripts/run_table1_arms_10sweep.pbs` | `coss_a6gpu` (CPU) | cache has 6,019 pkl (asserted in-script) |
| 3 | `qsub scripts/run_table1_cbmot_10sweep.pbs` | `coss_a6gpu` (CPU) | phase-2 `retro_*` cells exist (asserted in-script) |
| 4 | `python audit/table1_regen_2026-08-28/aggregate_table1.py` | CPU | phases 2–3 complete |

Phase 0 runs on `coss_a6gpu` on purpose: `coss_agpu` allows one running job per
user (`max_run = [u:PBS_GENERIC=1]`), and that slot is reserved for phase 1.

## Commit gate (phase 1)

Per instruction, the sweep change is **not** committed yet. Immediately before
phase 1 actually starts running, re-read the diff and commit **only**:

- `method_scannet/streaming/nuscenes_native_evaluator.py` — remove the
  `multi_sweep=False / num_sweeps=1` hardcode so the setting follows the config
- `audit/cbmot/run_cbmot_matched_control.py` — `CBMOT_OUT/GRID/RUN_N3` env
  overrides, defaults byte-identical to the frozen run
- the new `scripts/run_table1_*.pbs` and `audit/table1_regen_2026-08-28/*`

**Do not stage** these pre-existing, unrelated working-tree changes, which were
dirty before this task began:

- `README.md` (indoor AP-gap provenance write-up)
- `paper_iccv_draft/figs/fig3_semantic_stability{.pdf,.tex,_body.tex}`
  (Fig. 3 replay, PBS job 119908)

Stage by explicit path; never `git add -A`.

## What each phase proves

- **Phase 0** — P1 config plumbing, P2 loader input, P3 split (150/6,019),
  P4 checkpoint sha, P5 env/git, P6 attribute rule live, P7 corrected evaluator
  live, P8 disk, P9 sweep-chain availability over **all 6,019** val samples.
  Also dry-runs the phase-4 aggregator against the old cells, which must
  reproduce the published mAP/NDS and fire the single-sweep and NDS<mAP flags.
- **Phase 1** — probes the *runtime* input (5 channels / >150k points / 10
  distinct Δt) and aborts before inference if it is not 10-sweep; also aborts if
  the scheduler placed it on anything other than an A100, or on `ece-tgpu3`.
  Resumable: `<token>.pkl` already present is skipped, so a walltime kill is
  simply resubmitted.
- **Phase 2** — the six mAP/NDS-bearing arms. Threshold-matching rule unchanged;
  threshold value re-derived because the budget is a property of the new
  detections. Count match asserted within ±1 %.
- **Phase 3** — the two accumulation controls, `parallel_addition` / noise 0.05
  / `max_age` 5, all pre-specified in `audit/cbmot/PRESPECIFICATION.md`.
- **Phase 4** — collects all eight rows with the same `read_axis` reader that
  built the published CBMOT table, compares to the published values, runs the
  sanity checks, and writes `table1_regenerated_results.json` +
  `TABLE1_REGENERATION_REPORT.md`. Exits nonzero if anything is flagged.

## Preserved, never overwritten

`results/outdoor_native_temporal_cpcache_thr000_single_gravity`,
`results/2026-07-18_e1_grid_v01`, `results/2026-07-30_e2c_retro_thrmatch_v01`,
`audit/cbmot/cells`. The regeneration writes only to
`results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity` and
`audit/table1_regen_2026-08-28/`.
