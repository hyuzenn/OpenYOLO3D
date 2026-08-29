# C1/C2 parity smoke — artifact guide

PBS job 120240, A100-SXM4-80GB, 5 val scenes = 200 keyframes.

| file | what it is |
|---|---|
| `validate_pipeline.json` | Stage A, 8/8 PASS. Input-pipeline assertions on the live runtime objects. |
| `parity_smoke_results.json` | Stage C, as the job emitted it. **Its `counts` block and `gates.count_within_10pct` are WRONG** — see below. Its `metrics` and `matched_residuals_2m` blocks are correct. |
| `recount.json` | The corrected count comparison. This supersedes the `counts` block above. |
| `eval_ours_corrected/`, `eval_anchor/` | Both sides scored through the same corrected evaluator against the same GT. |

## The count-block defect (in the probe, not in the pipeline)

`compare_to_anchor.py` originally computed the per-class range filter as
`‖ego_translation‖`. The harness stores `ego_translation` as the ego's
**absolute global position** (`nuscenes_native_evaluator`:
`ego_translation = ego_pose[:3, 3]`); the devkit's `add_center_dist` replaces it
with the relative vector only later, inside the evaluator. Every box therefore
read as hundreds of metres from the origin and was filtered out, giving
`ours_after_class_range = 0` and `count_rel_err = -1.0`, so the job exited 3.

The correct distance is `‖translation − ego_translation‖`. `recount.py`
recomputes it read-only from the same `tracks.json`; `compare_to_anchor.py` has
since been fixed at the same place. Nothing in the adapter, the harness or the
evaluator was involved, and no metric in `parity_smoke_results.json` depended on
the faulty quantity.

Corrected result: **66.90 vs 65.475 boxes/sample, +2.18 %** — gate PASS.
