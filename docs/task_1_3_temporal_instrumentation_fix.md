# Task 1.3 — Temporal-metric instrumentation fix (Part 2)

**Date:** 2026-05-15
**Scope:** Fix the 2-layer instrumentation defect that blocks
temporal-metric reporting in Task 1.4b. Part 2 (this doc): instrument +
1-scene verify. Part 3 (re-run 7 axes on 312 scenes) is a separate user
decision based on Part 2's verdict.

---

## Stage 1 — defect map

Two independent bugs make `label_switch_count` and `time_to_confirm`
return degenerate values across the streaming codebase. Both must be
fixed; fixing only one keeps the metric uninformative.

### Layer 2 — `wrapper.py:383` writes `-1` placeholder labels

`method_scannet/streaming/wrapper.py:378-386`:

```python
# Lightweight per-frame snapshot (label-only). Confirmed visible
# instances get a placeholder -1 label here; the final-frame label
# comes from compute_baseline_predictions(). Task 1.2b deliberately
# does NOT call the accumulator each frame (O(F·V·K) cost) — Task
# 1.3 / 1.4 will add checkpoint-based per-frame mAP if needed.
current_instance_map: dict[int, int] = {int(k): -1 for k in confirmed_visible}

# --- Step F: history bookkeeping for Stage 3 metrics ------------
self.pred_history.append(current_instance_map)
```

Every per-frame label is `-1`. `metrics.py` consumers reject `-1` on
purpose (`label_switch_count` line 113 `if last != -1 and label != -1`,
`time_to_confirm` line 141 `if label == -1: continue`). Result:
`label_switch_count` returns 0 and `time_to_confirm` returns `{}` for
every scene — by design but on degenerate input. The comment correctly
calls this out ("placeholder…Task 1.3 / 1.4 will add").

### Layer 1 — `eval_streaming_ablation.py` does not call the metric helpers at all

`method_scannet/streaming/eval_streaming_ablation.py`:

```bash
$ grep -n "temporal\|label_switch\|id_switch\|time_to_confirm" method_scannet/streaming/eval_streaming_ablation.py
6:  ... and aggregates AP + temporal metrics per axis.   ← docstring only
```

Only the docstring mentions temporal metrics; `run_one_axis()` writes
`metrics.json` (AP) + `summary.json` (axis AP / walltime). No call to
`label_switch_count`, `time_to_confirm`, or `id_switch_count` anywhere.
No `temporal_metrics.json` ever gets written.

### Empirical confirmation

`results/2026-05-13_scannet_streaming_baseline_cached_v01/temporal_metrics.json`
(Task 1.2c — only place temporal metrics *were* computed, on the
same `-1` placeholder input):

```json
{
  "label_switch_count": {"n_scenes": 312, "total": 0, "mean_per_scene": 0.0},
  "time_to_confirm":    {"n_instances": 0, "mean": null}
}
```

Zero switches in 312 scenes is impossible for real measurement.

### What `pred_history` should carry

`metrics.py` consumers expect `pred_history` to be a list of per-frame
maps `{instance_id: predicted_label_at_that_frame}`. The label is the
*cumulative-up-to-t* prediction (what the model would output if it had
to commit at frame *t*). That is exactly the quantity `metrics.py` was
designed for — `label_switch_count` counts label transitions in that
cumulative prediction, `time_to_confirm` counts how many frames the
cumulative prediction needs to stabilize.

The wrapper needs to compute and emit that cumulative-up-to-t label per
visible/confirmed instance per frame, without disturbing the Task 1.4a
method-integration logic (Step C/D/F hook calls + scene-end finalize).

---

## Stage 2 — fix layer 2 in the wrapper

Add a separate `RunningInstanceLabeler` that mirrors the
`BaselineLabelAccumulator`'s per-frame data ingestion but maintains a
*running* per-instance label histogram instead of deferring to a single
scene-end pass.

- Class lives in `method_scannet/streaming/running_labeler.py` (new
  file). State: `per_instance_counts: dict[int, np.ndarray(num_classes)]`.
- Per-frame `update_frame(...)` looks up label-map pixels at the
  projected positions of each visible instance (same procedure as the
  baseline accumulator's final-frame pass, applied to just *this* frame).
- `snapshot(instance_ids)` returns `{iid: argmax(counts[iid])}` for
  every requested id, falling back to `-1` if no labels have ever been
  observed for that id.
- When M21 (`WeightedVoting`) is installed, the labeler swaps in a
  weighting function that mirrors May's offline `frame_weight` formula
  (`alpha · exp(-d3/D) + (1-alpha) · exp(-d2/C)`, weighted by per-frame
  IoU confidence). Same hyperparameters as `voter.distance_weight_decay`
  / `voter.center_weight_decay` / `voter.spatial_alpha`.
- When M22 is installed, snapshot directly calls
  `method_22.predict_label(iid)` (M22's EMA fusion already carries the
  cumulative state).

The wrapper's `step_frame` calls the running labeler *after*
`baseline_accumulator.add_frame` (same frame data, no extra disk I/O)
and writes the labeler's snapshot into `pred_history`. The existing
hook calls (Step C: M11/M12 gate, Step D: M22 per-frame CLIP encode)
stay untouched.

## Stage 3 — fix layer 1 in the eval script

`run_one_axis()` in `eval_streaming_ablation.py` already loops the
evaluator over every scene. Add per-scene temporal-metric collection
after the `compute_method_predictions()` call:

- Read `evaluator.pred_history` (list of frame snapshots).
- Compute `label_switch_count(pred_history)` and
  `time_to_confirm(pred_history, K=3)` per scene.
- Aggregate into `axis-level` temporal stats: total switches,
  per-scene mean/median/p90, time-to-confirm distribution.
- Write `temporal_metrics.json` sibling to `metrics.json` per axis.

`id_switch_count` needs `gt_matching`, which is not available in this
loop — skip it for now (documented limitation; the Task 1.1 streaming
design also lists `id_switch_count` as gated on a per-scene GT matcher
that isn't built yet).

Also add a `--scenes` CLI flag so we can run on a single scene
(`scene0011_00`) for verification before any full re-run.

## Stage 4 — verification gate (scene0011_00, 3 axes)

PASS conditions (all required):
1. `pred_history[t]` contains real class labels (sample non-`-1` value).
2. `label_switch_count.total > 0` for baseline on scene0011_00.
3. `time_to_confirm.n_instances > 0` and `mean` is a number (not null).
4. M11 raises `time_to_confirm.mean` (registration delays confirmation).
5. M21 reduces `label_switch_count.total` vs baseline (weighted vote
   stabilizes labels — advisor-hypothesis direction).

If any of (1)-(3) fails the instrumentation is still degenerate. Stop
and report; do NOT proceed to Part 3.
