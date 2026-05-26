# Task 3.1 — lsc=0 root-cause diagnosis (Native γ CenterPoint baseline)

Date: 2026-05-21 · **diagnosis only, 0 code changes** · CPU, no GPU
(Step 2b PBS 95045/A left untouched on `ece-agpu7`).

**Verdict: scenario (Q) — association mechanism. And scenario (P) is REFUTED.**
`lsc=0` on the native γ baseline is a *structural artifact of the class-aware
`CentroidAssociator`*, not evidence that the LiDAR detector is temporally
stable. Measured against ground-truth tracks, native CenterPoint (γ) is in fact
**temporally unstable** (≈23% of frame-to-frame transitions switch class; 81% of
multi-frame instances switch at least once at a 2 m gate). The prior recorded
finding ("native CenterPoint class labels are temporally stable per track →
legitimate null") is **wrong** and is corrected below.

Artifacts:
- script `results/2026-05-21_task_3_1_lsc_diagnosis_v01/diagnose_lsc_zero.py`
- raw `results/2026-05-21_task_3_1_lsc_diagnosis_v01/lsc_diagnosis.json`
- data: Step 2b smoke cached proposals (3 val scenes, 120 samples, single-sweep,
  score_thr=0.0) — `results/2026-05-21_outdoor_native_temporal_smoke_v01/cpcache_thr000_single/`

---

## Stage 1 — Association mechanism (code reading)

### 1.1 NuScenesRunningLabeler + the track key

`NativeTemporalNuScenesEvaluator.step_sample` builds the temporal history as
(`nuscenes_native_evaluator.py:294-392`):

```
global_ids = associator.step(proposals)           # track id per proposal
for p, gid in ...:  labeler.add_vote(gid, native_idx, weight=...)   # native γ class
confirmed = gate(gids_present)                     # M11/M12 (or all, baseline)
pred_history.append(labeler.snapshot(confirmed))   # {gid: argmax class}
```

`label_switch_count(pred_history)` (`metrics.py:99`) counts, **per `gid`**, the
frames where that track's snapshot label flips. So `lsc>0` needs (a) a `gid`
present in ≥2 frames **and** (b) its cumulative-argmax class changing.

- `NuScenesRunningLabeler` (`nuscenes_evaluator.py:93`) is a per-`gid` class
  histogram; `snapshot` = `argmax` of accumulated votes.
- The track key `gid` comes from **`CentroidAssociator`**, which is documented
  and implemented as **class-AWARE** (`nuscenes_evaluator.py:137`, line 193:
  `if st["cls"] != cls: continue`). A proposal can only join an existing track
  of the **same class**; otherwise a fresh `gid` is allocated, and the track's
  class is frozen at creation (line 208).

**Therefore every track is class-homogeneous by construction** → its histogram
has weight in exactly one bin → `snapshot` returns one fixed class for all
frames → `label_switch_count = 0` is *structurally forced*, independent of how
much γ actually flickers. The same applies to M21: its votes are still the
per-proposal `native_idx` of a single-class track, so `best_label` never differs
from `native_idx` (`n_relabeled_by_m21 = 0` is the same artifact, not stability).

### 1.2 Native γ classifier output

Cached proposal dict keys: `cls_name, cls_idx, centroid_ego, score, bbox_lidar`
(score_thr=0.0 → ~200 proposals/sample). γ emits a per-detection class per
sample. The class-aware associator then *bins* those detections so that any
class disagreement is expressed as **two separate tracks**, never as one track
switching. Whether γ is per-object stable is thus invisible to the native `lsc`.

### 1.3 Indoor vs Outdoor — why Indoor shows lsc≫0 and native Outdoor shows 0

| | Indoor (ScanNet) | Outdoor native (nuScenes) |
|---|---|---|
| Track key | Mask3D proposal index — **fixed, class-agnostic geometry** (`RunningInstanceLabeler`, `running_labeler.py:36`) | `CentroidAssociator` `gid` — **class-aware centroid** |
| Per-frame label | YOLO pixel labels painted into the proposal's histogram (any class) | native γ class of the matched detection (one class per track) |
| Histogram | **multi-class** → argmax can flip across frames | **single-class** by construction → argmax fixed |
| `lsc` | ≫0 (prior: ≈23k) — measures real flicker | **0, structural** — cannot express a flip |

The two domains compute `lsc` on **incompatible track definitions**. The Indoor
number and the Outdoor number are not comparable, and the Outdoor 0 is not a
detector property.

---

## Stage 2 — Diagnostic experiments (smoke: 3 scenes / 120 samples)

> Faithfulness check: replicating the baseline axis with the **real**
> `CentroidAssociator` at the default 2.0 m reproduces the smoke output exactly
> — `lsc=0`, `ttc_n=1263`. The offline harness is byte-faithful to the pipeline.

### 2.1 Per-instance γ class variance — GT-anchored (perfect association)

For each GT `instance_token`, over the samples it appears in, match the nearest
γ proposal in **global frame, class-agnostically**, and record γ's class.
"Switch" = consecutive differing γ class. This isolates γ's true per-physical-
object stability (no pipeline association in the loop).

| Match gate | coverage | inst (≥2 samples) | inst with ≥1 switch | per-instance switch rate | per-transition switch rate | distinct γ classes / instance |
|---|---|---|---|---|---|---|
| 1.0 m | 0.641 | 70 | 48 | **0.686** | 0.166 | up to 7 |
| 2.0 m | 0.730 | 75 | 61 | **0.813** | **0.229** | up to 7 |
| 4.0 m | 0.822 | 81 | 72 | **0.889** | 0.286 | up to 8 |

**γ is NOT per-track stable.** At a 2 m gate, 81% of multi-frame GT instances
receive ≥1 γ class switch and ~23% of all frame-to-frame transitions flip class
(typical confusions: car↔truck↔construction_vehicle; pedestrian↔barrier↔
traffic_cone at range). Refutes scenario (P).

### 2.2 Association sweep — class-aware vs class-agnostic, looser/stricter

Same cached γ proposals, same vote/snapshot/`lsc` path; only the associator
varies.

| Associator | thr | lsc | ttc_n | tracks | multi-frame tracks | multi-class tracks | max classes/track |
|---|---|---|---|---|---|---|---|
| class-aware (real) | 0.5 | **0** | 331 | 17793 | 1375 | 0 | 1 |
| class-aware (real) | 2.0 | **0** | 1263 | 12074 | 3727 | 0 | 1 |
| class-aware (real) | 4.0 | **0** | 2000 | 8268 | 4252 | 0 | 1 |
| class-agnostic (counterfactual) | 0.5 | **744** | 403 | 14809 | 3698 | 2910 | 5 |
| class-agnostic (counterfactual) | 2.0 | **2637** | 1404 | 7082 | 4772 | 4092 | 10 |
| class-agnostic (counterfactual) | 4.0 | **3152** | 1673 | 3878 | 3307 | 3000 | 9 |

- **The class-aware associator yields `lsc=0` at every threshold** (stricter and
  looser): the threshold is not the lever, **class-awareness is**. Max distinct
  classes/track = 1 always.
- **Dropping only the class gate makes `lsc` jump to hundreds–thousands** on the
  identical γ proposals. → `lsc=0` is caused by the association mechanism (Q).

(Caveat: the class-agnostic counterfactual also merges some genuinely-different
nearby objects, so its absolute `lsc` overstates pure γ flicker — but Stage 2.1,
which uses GT tracks, independently proves γ flicker is real, so the conclusion
is robust either way.)

### 2.3 lsc-calculation sanity (rules out scenario R)

- `label_switch_count` works: synthetic histories with 1 and 2 switches return
  exactly 1 and 2.
- Native (real, class-aware) tracks: **3727 multi-frame tracks exist** (the
  metric *has* opportunity to count switches) but **all 12074 tracks are
  single-class** (`max classes/track = 1`). So `lsc=0` is the correct output of a
  class-aware design, not a metric/wiring bug. Refutes scenario (R).

---

## Stage 3 — Conclusion

### Scenario verdict

| | Scenario | Verdict |
|---|---|---|
| **(P)** | Fundamental — LiDAR γ per-track stable | **REJECTED.** GT-anchored γ switches on 81% of multi-frame instances (2 m gate); ~23% per-transition. γ is temporally unstable. |
| **(Q)** | Association mechanism — Indoor vs Outdoor difference | **CONFIRMED (root cause).** Class-aware `CentroidAssociator` makes every native track single-class → `lsc` structurally 0. Class-agnostic association on the same γ proposals gives `lsc` = 744–3152. |
| **(R)** | Bug / wiring | **REJECTED.** Metric verified correct; native evaluator does call it on 3727 multi-frame tracks; 0 is the correct output of the design. |

**One-line:** `lsc=0` is a *deliberate-design artifact* of class-aware
association, not a fundamental property of the LiDAR detector — and the detector
itself is actually quite label-unstable.

### Impact on the paper §6 narrative

- **Remove** any claim that "the native LiDAR classifier baseline has no label
  switching / is temporally stable." It is false; replace with the honest,
  measured statement: native γ exhibits substantial per-track class instability
  (≈23% of transitions at a 2 m GT gate), but the native pipeline's class-aware
  association makes that instability *unobservable* to `lsc`, and *unfixable* by
  the label-stabilization axes (M21/M22 vote inside a single-class track).
- **`lsc/ttc` cannot be the outdoor contribution as currently wired** — `lsc` is
  structurally pinned to 0. The honest outdoor value of the temporal layer stays
  what Step 2b already shows: **registration gating (M11/M12) = FP suppression at
  an mAP cost** ([[project_step2b_native_temporal]]). `ttc` is still meaningful
  (it measures how long the registration gate takes to confirm a track).
- **Main-contribution framing:** keep temporal label/ID stability as the
  **indoor (image-based open-vocab)** contribution, where the track key is
  class-agnostic (Mask3D) and flicker is real and correctable. Outdoor is a
  proposal-source-swap demonstration + a registration-gating result, not an
  lsc-stabilization result.
- **New opportunity (do not auto-implement):** Stage 2.1+2.2 show there *is*
  real, fixable label flicker in native γ — it is only hidden by the class-aware
  association. A **class-agnostic association + temporal label voting (M21/M22)**
  on outdoor could turn this into a genuine outdoor `lsc` contribution. That
  needs a code change and is a separate decision (see below).

### Recommended next steps

1. **(P rejected / Q confirmed) → §6 rewrite** with the cross-baseline honest
   framing above. No code change required for the honest report.
2. **Confirm magnitude at population scale** when Step 2b PBS A/B/C land
   (150 scenes): re-run this script against the 150-scene cache to replace the
   3-scene/75-instance smoke numbers (mechanism conclusion is threshold-robust
   and won't change; only the percentages will firm up).
3. **Decision required (separate task):** whether to add a *class-agnostic*
   outdoor association variant so M21/M22 can act on the now-exposed γ flicker.
   This is a code change to the association path → per project rules, hold for
   your explicit go-ahead before implementing; do not touch the running Step 2b
   evaluator.

### Honest caveats

- Smoke scale: 3 scenes, 120 samples, 75 multi-sample GT instances at the 2 m
  gate. Conclusions (P-rejected, Q-confirmed, R-rejected) are mechanism-level and
  threshold-robust; the exact switch percentages are smoke estimates.
- score_thr=0.0 means ~200 γ proposals/sample (heavy low-score clutter); this is
  the actual Step 2b config, so the diagnosis matches the pipeline being studied.
