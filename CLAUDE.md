# CLAUDE.md — Operational Handbook (SemWorld-3D / OpenYOLO3D)

This file is the operating manual for working in this repository with zero
prior history. Read it fully before touching anything. It contains rules and
procedures only — **no experimental results, no mAP/AP/FPS numbers, ever**
(see "Things that must NEVER be changed"). All established numbers live in
`STATE.md` and `results/`.

Reading order for a fresh session:
1. This file.
2. `STATE.md` — frozen numbers, closed research lines, open items.
3. `results/experiment_tracker.md` — full-scale run history.
4. The `docs/` note relevant to your task (see index below).

---

## 1. Project overview

**SemWorld-3D** — real-time open-vocabulary 3D instance mapping. The repo is a
heavily extended fork of OpenYOLO3D (upstream README preserved as `README.md`;
it describes the original paper, not this project's current direction).

Current architecture decision (final): **Mask3D 3D instance masks + streaming
2D label fusion**. The earlier 2D→3D depth-lifting direction was abandoned;
any lifter/refit/stitch code you find is dead. Do not build on it.

Two experimental domains:
- **Indoor**: ScanNet200 val (312 scenes), streaming evaluation over cached
  Mask3D masks with a running instance labeler.
- **Outdoor**: nuScenes val (150 scenes), CenterPoint proposals (cached,
  gravity-corrected) with the same temporal-method family injected.

This project **is** an undergraduate thesis. Every experiment ultimately
serves the paper. Language convention: the user instructs in Korean; all
reports, generated documents, paper tables, and summaries are written in
**English**.

## 2. Research goal

Streaming (frame-by-frame, no future frames) open-vocabulary 3D instance
segmentation that keeps instance labels **temporally consistent** without
sacrificing accuracy. Concretely:

1. Quantify temporal label instability with a purpose-built metric (OV-TCS).
2. Fix it with lightweight temporal methods (the M-series) layered on top of a
   frozen proposal backbone — no retraining of the 3D backbone.
3. Show the same method family transfers indoor (ScanNet200) → outdoor
   (nuScenes).

## 3. Thesis contribution

- **OV-TCS metric** (paper core): `OV-TCS_C = L_norm × (1 − CSR)`.
  Two-axis justification: label flicker → `(1 − CSR)`; track fragmentation →
  `L_norm`. The product formulation is RESOLVED — do not reopen or redefine
  it as stability-only (that is fragmentation-blind and collapses the §1
  argument). Validated as a downstream surrogate for detection quality.
  It is **dead as a control signal** (EMA-control line is closed).
- **Temporal method family (M11/M12/M21/M22/M31/M32)** as a streaming label
  stabilization layer over frozen proposals, indoor and outdoor.
- **Class-aware label fusion** (outdoor, gated VRU override) as the 2D→3D
  class-correction method.
- Storyline: `docs/ovtcs_paper_storyline.md`. Final tables:
  `results/nuscenes_final/paper_table.md` and
  `results/scannet_val312_final/summary.md`. Draft work lives in the
  `ovtcs-paper-draft` worktree.

## 4. Repository architecture

```
OpenYOLO3D/
├── CLAUDE.md                  # this handbook (rules only, no results)
├── STATE.md                   # frozen numbers & decisions — single source of truth
├── run_evaluation.py          # offline (non-streaming) ScanNet200/Replica eval
├── configs/                   # nuScenes + OpenYOLO3D-on-nuScenes YAML configs
├── pretrained/                # checkpoints, config_scannet200.yaml, prompt embeddings
├── data/                      # datasets (see §5) — never committed
├── adapters/                  # nuScenes/CenterPoint → project format adapters
│   ├── centerpoint_proposals.py   # CenterPoint boxes → proposals (z-convention fix here)
│   └── nuscenes_to_openyolo3d.py
├── dataloaders/               # nuscenes_loader.py, nuscenes_stream_loader.py
├── preprocessing/             # frustum / pillar / verticality filters (outdoor geometry)
├── proposal/                  # detection_guided_clustering.py (detguided), hybrid_proposal.py
├── method_scannet/            # THE method code (name is historical; used by both domains)
│   ├── method_11_frame_counting.py    # M11: N-frame confirmation gate
│   ├── method_12_bayesian.py          # M12: Bayesian label filtering
│   ├── method_21_weighted_voting.py   # M21: confidence-weighted label voting
│   ├── method_22_feature_fusion.py    # M22: CLIP-feature EMA fusion
│   ├── method_31_iou_merging.py       # M31: IoU-based instance merging
│   ├── method_32_hungarian_merging.py # M32: Hungarian matching merge
│   ├── streaming/
│   │   ├── run_streaming_scene.py         # per-scene streaming driver
│   │   ├── running_labeler.py             # RunningInstanceLabeler (core state machine)
│   │   ├── eval_streaming_ablation.py     # INDOOR entry point (cached Mask3D)
│   │   ├── nuscenes_native_evaluator.py   # OUTDOOR entry point (7-axis) — USE THIS
│   │   ├── nuscenes_evaluator.py          # LEGACY — M21/M22/M31/M32 are silent no-ops here
│   │   ├── metrics.py, gt_matching.py     # streaming AP, lsc, ttc, OV-TCS pieces
│   │   └── eval_ovtcs_*.py                # OV-TCS metric evaluation scripts
│   └── tests/                 # pytest unit tests for M22/M32/bbox utils
├── evaluate/                  # upstream ScanNet200/Replica AP evaluators (do not modify)
├── scripts/                   # all PBS job scripts + aggregation/analysis scripts
├── results/                   # experiment outputs (gitignored except experiment_tracker.md)
├── docs/                      # design notes & diagnoses, one file per task
├── models/Mask3D/             # Mask3D backbone (third-party)
├── third_party/               # other vendored deps
└── diagnosis*/                # historical debugging snapshots — read-only archaeology
```

`docs/` index convention: `task_<phase>_<n>_*.md` are design/diagnosis notes
per task; `ovtcs_*` are metric/paper notes; `NUSCENES_SETUP.md`,
`Data_prep.md`, `Installation.md` are setup guides.

## 5. Dataset layout

```
data/
├── scannet200/                # preprocessed ScanNet200 scenes (scene####_##/), 1515 dirs
│   ├── ground_truth/          # GT instance masks
│   └── label_database.yaml
├── raw/scannet/               # raw .sens + scannetv2-labels.combined.tsv
└── nuscenes/
    └── v1.0-trainval/         # metadata JSONs (full trainval metadata)
```

Critical dataset facts:
- **ScanNet200**: only **val (312 scenes)** has extracted RGB-D. train1201 has
  NO extracted RGB-D on disk; fetching requires ~1.8 TB of .sens AND the
  Mask3D checkpoint is in-sample on train, so a train1201 run is scientifically
  invalid for headline numbers anyway. Do not "just run train".
- **nuScenes**: the download was keyframe-only — **sweeps are missing for
  146/150 val scenes**. Everything runs single-sweep. Any plan that assumes
  multi-sweep accumulation is blocked on data, not code.
- **Replica**: supported by `run_evaluation.py` (legacy upstream path).

Key caches (expensive to rebuild — treat as data):
- `results/2026-05-13_mask3d_cache/` — per-scene Mask3D predictions (.pt),
  the input for ALL indoor streaming runs.
- `results/outdoor_native_temporal_cpcache_thr000_single_gravity/` — the
  gravity-corrected CenterPoint cache (γ). This is the outdoor anchor cache.
- `results/outdoor_detguided_cpcache_thr000_full150/` — detguided proposal
  cache (open-vocab track).
- Build tool: `scripts/build_hybrid_cache.py` (+ `.pbs`).

## 6. Indoor pipeline (ScanNet200)

Flow: cached Mask3D masks → per-frame streaming replay →
`RunningInstanceLabeler` assigns/updates labels → optional M-axes modify
labeling/merging → per-instance final-frame predictions → AP + temporal
metrics.

Entry point:
```bash
python -u -m method_scannet.streaming.eval_streaming_ablation \
  --cache-dir results/2026-05-13_mask3d_cache \
  --output "$RUN_DIR/outputs" \
  --config pretrained/config_scannet200.yaml \
  --axes baseline m11 ... \
  [--limit N | --scenes scene0011_00 ...] \
  --idsw-iou 0.5
```

Method axes (shared vocabulary, both domains):
| Axis | Module | What it does |
|---|---|---|
| baseline | — | RunningInstanceLabeler as-is |
| M11 | frame counting | instance confirmed only after N frames (default N=3) |
| M12 | Bayesian | probabilistic label filter (threshold 0.85) |
| M21 | weighted voting | confidence-weighted label votes via `WeightedVoting.frame_weight` |
| M22 | feature fusion | CLIP-feature EMA; indoor data-driven gate: merge 0.006, dist 0.5 m, sem 0.95 |
| M31 | IoU merging | merge instances above IoU threshold (default 0.5) |
| M32 | Hungarian | Hungarian matching merge; distance 0.5 m indoor / 1.0 m outdoor |

Indoor facts you must not re-derive:
- `lsc` (label switch count) is computed **per-frame**, independent of merging.
- M22 alone affects lsc only; it does not recover AP. M32 (data-driven
  parameters above) is the AP-recovering component indoor.
- Streaming AP and offline AP (`run_evaluation.py`) are **not comparable** —
  never mix them in one table.

Offline (non-streaming, upstream-style) eval:
```bash
python run_evaluation.py --dataset_name scannet200 [--path_to_3d_masks ...]
```

## 7. Outdoor pipeline (nuScenes)

Flow: CenterPoint boxes from cache (γ) or DetectionGuidedClusterer (detguided)
→ adapter (`adapters/centerpoint_proposals.py`) applies the **z-convention
fix** (CenterPoint z is bottom-center; use `.gravity_center`, per-class
`ez = −h/2`) → streaming association (ego or global frame; class-aware or
class-agnostic) → M-axes → native nuScenes mAP/NDS + track metrics.

Entry point (**always the native evaluator**):
```bash
python -u -m method_scannet.streaming.nuscenes_native_evaluator \
  --output "$RUN_DIR/outputs" \
  --axes baseline [m11 m12 m21 m31 m32 phase1] \
  --scene-split val \
  --proposal-source gamma|detguided|hybrid \
  --cp-cache-dir <cache dir from §5> \
  [--association-class-agnostic] \
  [--association-frame ego|global] \
  [--m32-distance 1.0] \
  [--collect-track-metrics] \
  [--fuse-allow bicycle,motorcycle --fuse-tau-iou 0.5 --fuse-tau-score 0.4] \
  [--scene-limit N | --scenes ...]        # smoke only
```

Proposal-source semantics (framing matters for the paper):
- `gamma` — closed-set nuScenes-10 **anchor** (CenterPoint native labels).
- `detguided` — **open-vocab capability** track (GT-free proposals). Evaluated
  for capability, not to beat the closed-set anchor.
- `hybrid` — CP geometry + YOLO ROI labels. This line is CLOSED (see STATE.md).

Outdoor facts you must not re-derive:
- Class-aware association makes CSR ≡ 0 (degenerate) — temporal-consistency
  metrics require `--association-class-agnostic`.
- Native CenterPoint labels give lsc = 0 under class-aware association, so the
  temporal-labeling axis is null there; the temporal layer's value outdoor is
  registration/association gating and label fusion.
- The known bottleneck is **proposal generation and localization**, not score
  ranking. Score calibration is a closed line. See STATE.md §1 and §4.
- Label Fusion G is a **per-class VRU correction** (allowlisted classes with
  IoU/score gates), not a global-mAP method. Frame it that way.
- `--no-gpu` exists for cache-replay runs that need no CUDA.

## 8. Evaluation pipeline

- **Indoor AP**: `evaluate/scannet200/eval_semantic_instance.py` (upstream —
  do not modify). Streaming AP is computed inside
  `method_scannet/streaming/metrics.py` on final-frame predictions.
- **Streaming temporal metrics**: `lsc` (label switches, per-frame), `ttc`
  (time-to-confirm), id-switch count (`--idsw-iou`).
- **OV-TCS**: `eval_ovtcs_instance_scannet.py` (indoor),
  `eval_ovtcs_track_outdoor.py` (outdoor), `eval_ovtcs_surrogate.py`
  (surrogate-validity analysis). Formulation is frozen (§3).
- **Outdoor**: `nuscenes_native_evaluator.py` computes native nuScenes
  mAP/NDS plus track metrics (`--collect-track-metrics`: OV-TCS_C, track
  length, fragmentation, CSR). The legacy `nuscenes_evaluator.py` silently
  no-ops M21/M22/M31/M32 — never use it for method comparisons.
- **Aggregation**: each experiment family has a `scripts/aggregate_*.py`
  that turns raw per-scene outputs into the summary table. Write one per new
  experiment family; never aggregate by hand.

## 9. PBS workflow

**Never run CPU/GPU-heavy Python on the util node (ECE-util2)** — a watchdog
kills it. Everything heavy goes through `qsub` to the `coss_agpu` A100
Singularity container.

Standard PBS header (copy from any recent `scripts/run_*.pbs`):
```bash
#PBS -q coss_agpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb:Qlist=agpu
#PBS -l walltime=08:00:00
#PBS -j oe
```

Standard body:
```bash
cd /home/rintern16/OpenYOLO3D
source /home/rintern16/miniconda3/etc/profile.d/conda.sh
conda activate openyolo3d-dev
export CUDA_HOME=/tools/cuda/cuda11.7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

DATE=$(date +%F)
EXP=<experiment_name>              # snake_case
N=$(printf '%02d' $(($(ls -d results/${DATE}_${EXP}_v* 2>/dev/null | wc -l) + 1)))
RUN_DIR=results/${DATE}_${EXP}_v${N}
mkdir -p "$RUN_DIR/outputs"
LOG="$RUN_DIR/run.log"
exec >"$LOG" 2>&1                  # redirect INSIDE the script — PBS stdout is unreliable
```

Operational notes:
- Environment: `conda activate openyolo3d-dev` (Python 3.10). Other envs
  (`openyolo3d`, `semworld3d`, ...) exist for other purposes — default to
  `openyolo3d-dev` unless a script says otherwise.
- Monitor with `qstat -u rintern16`. Job output lands in `$RUN_DIR/run.log`,
  not the PBS `.o` file.
- Light work allowed on the util node: file inspection, grep, aggregation of
  small JSONs, git. Anything that loads torch/point clouds → qsub.
- Session skills automate this: `/auto-eval` (scaffold → smoke → gate → full
  submit), `/pbs-watch` (watch a job, aggregate on completion), `/state`
  (load frozen state), `/sync-tracker` (fold a finished full run into
  `experiment_tracker.md` + `STATE.md`). Prefer them over hand-rolling.

## 10. Common commands

```bash
# Environment
conda activate openyolo3d-dev

# Unit tests (safe on util node)
python -m pytest method_scannet/tests/ -q

# Indoor streaming smoke (2 scenes) — inside a PBS job
python -u -m method_scannet.streaming.eval_streaming_ablation \
  --cache-dir results/2026-05-13_mask3d_cache --output "$RUN_DIR/outputs" \
  --axes baseline --limit 2 --idsw-iou 0.5

# Outdoor native smoke (5 scenes, γ cache replay) — inside a PBS job
python -u -m method_scannet.streaming.nuscenes_native_evaluator \
  --output "$RUN_DIR/outputs" --axes baseline --scene-limit 5 \
  --proposal-source gamma \
  --cp-cache-dir results/outdoor_native_temporal_cpcache_thr000_single_gravity

# Outdoor open-vocab (detguided, class-agnostic, full val)
python -u -m method_scannet.streaming.nuscenes_native_evaluator \
  --output "$RUN_DIR/outputs" --axes baseline \
  --proposal-source detguided --association-class-agnostic \
  --cp-cache-dir results/outdoor_detguided_cpcache_thr000_full150

# Submit / monitor
qsub scripts/run_<experiment>.pbs
qstat -u rintern16

# Offline (upstream-style) eval
python run_evaluation.py --dataset_name scannet200
```

## 11. Coding conventions

- Python 3.10, no formatter enforced — match surrounding style (4-space,
  snake_case, argparse CLIs with `--kebab-case` flags).
- Every long-running script prints progress with `flush=True` / `python -u`.
- New evaluator options default to **no-op** (backward compatible): a run with
  no new flags must reproduce the previous behavior bit-for-bit. This is how
  every M-axis and fusion flag was added; keep it that way.
- Reuse before writing: the M-axis adapters (`streaming/method_adapters.py`),
  metrics, and aggregation patterns already exist — extend, don't fork.
- Tests: `method_scannet/tests/` uses plain pytest, no fixtures framework.
  Non-trivial new logic gets one small test or a one-scene smoke script
  (pattern: `smoke_method_XX_one_scene.py`).
- Do not modify `evaluate/` (upstream evaluators) or `models/Mask3D/` —
  correctness comparability depends on them being untouched.
- `diagnosis*/` directories are frozen archaeology; never import from them.

## 12. Naming conventions

- **Run dirs**: `results/<YYYY-MM-DD>_<experiment>_v<NN>/` — date of launch,
  snake_case experiment name, version zero-padded from 01. Never overwrite;
  re-runs bump the version.
- **Methods**: `M<phase><n>` — M1x = label gating (M11 frame counting,
  M12 Bayesian), M2x = label refinement (M21 voting, M22 feature EMA),
  M3x = instance merging (M31 IoU, M32 Hungarian). `phase1` = M21+M31 combo
  axis. Files: `method_scannet/method_<nn>_<name>.py`.
- **PBS scripts**: `scripts/run_<task-or-experiment>.pbs`; historical ones are
  `run_task_<phase>_<n>_*.pbs`. Multi-part jobs split as `_pbs_a/_pbs_b/...`.
- **Docs**: `docs/task_<phase>_<n>_<topic>.md` for task notes;
  `docs/ovtcs_*.md` for metric/paper material.
- **Caches**: descriptive stable names outside the dated scheme
  (`outdoor_*_cpcache_*`, `2026-05-13_mask3d_cache`) — they are inputs, not
  experiment outputs.
- **Aggregators**: `scripts/aggregate_<experiment>.py`.

## 13. Experiment conventions

These are hard rules, learned the expensive way:

1. **Minimal decisive comparison first.** A new hypothesis gets the cheapest
   2-arm experiment that can falsify it, with an explicit stop condition
   written down *before* launch. Promote to a sweep only on a positive signal.
   Before ablating a control signal, verify its mechanism correlation first.
2. **Smoke → gate → full.** Every full run is preceded by a small smoke
   (2–5 scenes). The gate criterion is explicit (e.g. "smoke Δ within X of
   expectation, no crashes"). `/auto-eval` encodes this.
3. **Never interrupt training/evaluation mid-epoch or mid-run for wall-clock
   reasons.** Completeness beats time budget. Never propose time-based
   auto-cutoffs.
4. **Verify premises before executing.** If an instruction's stated diagnosis
   or number conflicts with what the code/data actually shows, measure and
   push back with the measurement before running. Raw measurements first;
   verdicts/interpretation only when asked.
5. **One variable at a time.** The user prefers incremental,
   variable-controlled experiments. No multi-change runs.
6. **Check STATE.md §4 (closed lines) before proposing anything.** Reopening a
   closed line requires new evidence, stated explicitly.
7. Full-scale runs (and only those — never smokes or pilots) get synced into
   `results/experiment_tracker.md` and `STATE.md` via `/sync-tracker`.

## 14. How results are organized

- `results/` is **gitignored** except `results/experiment_tracker.md`.
- Standard run-dir contents:
  ```
  results/2026-05-05_scannet_eval_v01/
  ├── run.log        # full stdout/stderr
  ├── config.yaml    # config snapshot for reproducibility
  ├── metrics.json   # final metrics
  ├── outputs/       # artifacts (per-scene json/npy/ply)
  └── notes.md       # what/why/verdict (write this for every full run)
  ```
- `results/experiment_tracker.md` — append-only table of **full-scale** runs
  (312-scene indoor / 150-scene outdoor). Numbers never live anywhere else
  except run dirs and STATE.md.
- `STATE.md` — frozen anchors and decisions, each with a `results/` source
  citation. Update only when a full-scale run changes a frozen value.
- Canonical final artifacts: `results/nuscenes_final/paper_table.md`
  (outdoor) and `results/scannet_val312_final/summary.md` (indoor).
- Named non-dated dirs (`nuscenes_final`, `scannet_val312_final`,
  `*_cpcache_*`, `2026-05-13_mask3d_cache`) are canonical/cache — never
  delete or overwrite them.

## 15. Common pitfalls

- **Util-node watchdog** kills CPU-heavy Python. Symptom: process dies
  silently mid-run. Fix: qsub (§9). This is the #1 wasted-afternoon bug.
- **Legacy outdoor evaluator**: `nuscenes_evaluator.py` accepts M21/M22/M31/
  M32 axis names and silently does nothing. Use `nuscenes_native_evaluator.py`.
- **CenterPoint z-convention**: raw CP boxes use bottom-center z. Treating it
  as center collapses IoU3D recall to noise. The fix lives in the adapter
  (`.gravity_center` / per-class `ez=−h/2`); the gravity-corrected cache
  already has it. Never mix corrected and uncorrected caches in one
  comparison.
- **Class-aware association ⇒ CSR ≡ 0** outdoor. Temporal-consistency numbers
  from a class-aware run are degenerate, not "perfectly stable".
- **Streaming AP ≠ offline AP** indoor. Different protocols; never compare
  across them.
- **nuScenes sweeps missing** (146/150 val keyframe-only). Multi-sweep ideas
  are data-blocked.
- **train1201**: no RGB-D on disk + in-sample checkpoint. Headline indoor
  results are val-312 only.
- **PBS stdout**: rely on the in-script `exec > "$LOG" 2>&1` redirect; the
  PBS `.o` file may be empty or truncated.
- **M32 distance is domain-specific**: 0.5 m indoor, 1.0 m outdoor. Don't
  carry one across.
- **detguided vs γ framing**: detguided is not expected to beat the
  closed-set anchor; comparing them as rivals misstates the paper's claim.
- **Global score gates fail** across proposal sources — thresholds must be
  class/source-aware.
- Duplicate/legacy files exist (`download-scannet.py.1`,
  `extract_prompt_embeddings.py` vs `_v2`, numbered `eval_method_*` variants).
  When in doubt, the PBS scripts show which variant is live.

## 16. Things that must NEVER be changed

1. **No results in CLAUDE.md.** No mAP/AP/FPS/lsc values, no experiment
   change logs, ever. Numbers go in `results/` + `STATE.md`.
2. **OV-TCS formulation** (`L_norm × (1 − CSR)`, product form). §2c is
   RESOLVED. Do not redefine, do not make it stability-only.
3. **Frozen anchors in STATE.md** — updated only by a completed full-scale
   run, with the source run dir cited.
4. **The z-convention / gravity correction** in the CenterPoint adapter, and
   the gravity-corrected cache as the outdoor anchor.
5. **`evaluate/` upstream evaluators and `models/Mask3D/`** — comparability
   depends on them.
6. **Existing `results/` run dirs and caches** — never overwrite or delete;
   new runs get new version numbers.
7. **Backward-compatible defaults** in evaluators — a flagless run must
   reproduce prior behavior.
8. **Epoch/run completeness** — never truncate a run to save wall-clock time.
9. **Closed lines in STATE.md §4** — not restarted without new evidence.
10. **`results/experiment_tracker.md` is append-only** — never rewrite
    historical rows.

## 17. Checklist before any commit

- [ ] `python -m pytest method_scannet/tests/ -q` passes (util node OK).
- [ ] No files from `data/`, `results/` (except `experiment_tracker.md`),
      `pretrained/checkpoints/`, or logs staged. Check `git status` output.
- [ ] No numbers/results added to CLAUDE.md.
- [ ] New evaluator flags are no-op by default; a flagless smoke reproduces
      previous output.
- [ ] Nothing in `evaluate/`, `models/Mask3D/`, or `diagnosis*/` modified.
- [ ] Commit message follows the existing style: `feat:/fix:/docs:/refactor:`
      prefix + one-line scope of what completed.
- [ ] If a full-scale run finished: tracker row appended and STATE.md updated
      (or `/sync-tracker` run) in the same commit.

## 18. Checklist before any experiment

- [ ] Read `STATE.md` (or run `/state`) — is this line already closed (§4)?
- [ ] Hypothesis written as a 2-arm minimal comparison with an explicit stop
      condition.
- [ ] Exactly one variable differs from the reference arm.
- [ ] Premises verified against code/data (measure, don't trust the prompt).
- [ ] Correct entry point: indoor `eval_streaming_ablation`, outdoor
      `nuscenes_native_evaluator` (never the legacy evaluator).
- [ ] Correct cache for the comparison (γ gravity cache vs detguided) — and
      the same cache across both arms.
- [ ] Class-agnostic association enabled if temporal metrics are measured
      outdoor.
- [ ] Run dir scaffolded per §14 (or via `/auto-eval`); config snapshot
      copied; log redirect inside the script.
- [ ] Smoke (2–5 scenes) submitted first; gate criterion written down; full
      run only after gate PASS.
- [ ] Submitted via `qsub`, not run on the util node.

## 19. Checklist before writing the paper

- [ ] All headline numbers come from `results/nuscenes_final/paper_table.md`,
      `results/scannet_val312_final/summary.md`, or a STATE.md-cited run dir —
      never from memory or chat history.
- [ ] Storyline follows `docs/ovtcs_paper_storyline.md`; drafts in the
      `ovtcs-paper-draft` worktree.
- [ ] OV-TCS presented in product form with the two-axis justification
      (flicker → 1−CSR, fragmentation → L_norm) and its surrogate validation;
      the EMA-control negative result reported as such.
- [ ] Framing check: γ = closed-set anchor; detguided = open-vocab capability
      (GT-free); Label Fusion G = per-class VRU correction; hybrid/calibration
      /EMA lines reported as closed with their evidence.
- [ ] Indoor scope stated as val-312 with the train1201 data/in-sample caveat.
- [ ] Streaming vs offline AP never mixed in one table; class-aware CSR
      degeneracy footnoted where relevant.
- [ ] Every table states: dataset split, scene count, proposal source, cache,
      association mode, and the run dir it came from.
- [ ] Written in English.
