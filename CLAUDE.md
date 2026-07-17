# CLAUDE.md — SemWorld-3D Operational Handbook

This is the single source of operational truth for this repository. A new
contributor (human or model) should be able to work here using only this file
plus the code. **Rule zero: this file must never contain measured results
(mAP/AP/FPS numbers, deltas, experiment logs). Results live in `results/`
and in the paper. Only procedures, conventions, and pointers belong here.**

---

## 1. Project overview

SemWorld-3D is a research fork of [OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D).
It builds a **streaming, open-vocabulary 3D instance mapping system** and — as
the current paper focus — a **GT-free temporal consistency metric (OV-TCS)**
for evaluating such systems.

Two domains share one temporal layer:

| Domain  | Dataset     | Proposal front-end                      | Temporal layer |
|---------|-------------|-----------------------------------------|----------------|
| Indoor  | ScanNet200  | Mask3D 3D masks + streaming 2D labels   | shared         |
| Outdoor | nuScenes    | CenterPoint boxes / detection-guided clustering | shared |

The 2D→3D depth-lifting path from the original OpenYOLO3D was abandoned;
`lifter`/`refit`/`stitch` code paths are dead. The live architecture is:
3D proposals → streaming 2D label fusion → temporal consistency layer.

## 2. Research goal

Real-time open-vocabulary 3D semantic mapping from streaming sensor data,
with a principled way to *measure* temporal label stability when no
ground-truth track identities exist (the open-vocabulary setting).

## 3. Thesis contribution

This project is an undergraduate thesis (user: Yujeong). Contributions, in
priority order:

1. **OV-TCS metric** — `L_norm · (1 − CSR)` per track, system score = mean
   over **all** tracks (singletons score exactly 0 and are *included*).
   `L_norm = 1 − 1/L`; CSR = adjacent-frame label-switch fraction (defined
   for L≥2; singletons contribute 0 to the reported `csr_mean`). Validated
   via injection studies, MOT-metric agreement, an anti-gaming self-audit,
   and an open-vocabulary replication. Paper: `paper/main.tex` (ICCV draft).
2. **Proposal-agnostic temporal consistency layer** — the association /
   label-consistency / spatial-merge stack (axes M11…M32 below), applied
   unchanged across indoor and outdoor front-ends.
3. **2D→3D label fusion** (CLIP-adapter / prompt-tuning direction) — the
   thesis "main method" track; see `results/*labelfusion*` writeups.

The user prefers incremental, variable-controlled experiments and minimal
decisive 2-arm comparisons with explicit stop/go gates (see §13).

## 4. Repository architecture

```
adapters/            dataset/detector adapters (e.g. CenterPoint box → internal box;
                     z-convention handling lives here — see Pitfalls)
configs/             nuScenes run configs (baseline / stream / trainval / nusc10)
data/                datasets (see §5) — never committed
dataloaders/         dataset loaders
diagnosis*/          one-off diagnostic probes, grouped by campaign
                     (diagnosis/, diagnosis_alpha/, diagnosis_gamma/, …).
                     ⚠ Most are UNTRACKED and exist only in the main checkout
                     /home/rintern16/OpenYOLO3D — not in git worktrees.
docs/                design notes
evaluate/            legacy OpenYOLO3D evaluation code
method_scannet/      the method itself (misleading name — also hosts outdoor code)
  method_11_frame_counting.py     M11: temporal label voting (frame counting)
  method_12_bayesian.py           M12: Bayesian label update
  method_21_weighted_voting.py    M21: weighted label voting / relabel
  method_22_feature_fusion.py     M22: EMA feature fusion
  method_31_iou_merging.py        M31: IoU-based spatial merge
  method_32_hungarian_merging.py  M32: Hungarian spatial merge
  streaming/
    nuscenes_native_evaluator.py  ★ PRODUCTION outdoor evaluator (7-axis)
    nuscenes_evaluator.py         legacy — M21/M22/M31/M32 are SILENT NO-OPS here
    eval_streaming_*.py           indoor streaming harness
    metrics.py, gt_matching.py    metric implementations
models/, proposal/, preprocessing/, utils/   OpenYOLO3D inherited code
paper/               ICCV manuscript (main.tex, figs/, revision logs, review docs)
pretrained/          checkpoints, prompt embeddings, ScanNet/Replica configs
results/             all experiment outputs (gitignored except experiment_tracker.md)
scripts/             PBS job scripts + aggregation/figure scripts
run_evaluation.py    indoor ScanNet200/Replica entry point
run_nuscenes.py      outdoor entry point (legacy path)
```

Delivery copy of the paper for the advisor lives at
`/home/rintern16/OpenYOLO3D/paper_ovtcs_iccv_draft/` (sync via `cp`, not git).

## 5. Dataset layout

```
data/
  scannet200/        ScanNet200. Only val (312 scenes) has extracted RGB-D.
                     ⚠ train1201 has NO extracted RGB-D; extracting needs
                     ~1.8 TB of .sens first, and train-set eval would be
                     in-sample anyway. Official results = val312.
  nuscenes/          nuScenes. ⚠ 146/150 val scenes are missing non-keyframe
                     sweeps (keyframe-only download). Anything needing full
                     sweeps is blocked; single-sweep is the operating regime.
  raw/               raw downloads
```

Proposal caches (expensive to regenerate, treat as read-only inputs):
- `results/outdoor_detguided_cpcache_thr000_full150/` — detguided proposal
  cache for all 150 val scenes (used by E2-style cache replays, ~15 min CPU).
- `results/2026-05-13_mask3d_cache/` — indoor Mask3D proposal cache.

## 6. Indoor pipeline (ScanNet200)

- Proposals: Mask3D 3D instance masks (cached; `scripts/run_generate_mask3d_cache.pbs`).
- Labels: streaming 2D open-vocab labels fused onto 3D masks
  (`method_scannet/streaming/running_labeler.py`, `hooks_streaming.py`).
- Temporal axes M11/M12/M21/M22/M31/M32 applied per-frame in streaming order
  (`eval_streaming_ablation.py`, `eval_streaming_baseline.py`).
- Domain constants that differ from outdoor: M32 distance gate **0.5 m**
  (outdoor: 1.0 m for AP, 2.0 m used in some metric-paper runs — check the
  run's own config snapshot, never assume). M22 EMA momentum m=0.006,
  semantic gate 0.95.
- Known behavior: label-switch counting (`lsc`) is per-frame and independent
  of merging; M22 affects lsc only (does not recover AP); M32 recovers AP.

Entry points:
```
python run_evaluation.py --dataset scannet200 ...          # full eval
python -m method_scannet.streaming.eval_streaming_ablation # ablation harness
```

## 7. Outdoor pipeline (nuScenes)

- Proposal sources (`--proposal-source`):
  - `gamma` — CenterPoint-derived closed nuScenes-10 anchor (the "closed" arm).
  - `detguided` — DetectionGuidedClusterer, GT-free open-vocabulary capability.
  - `hybrid` — combination; historically not competitive on closed classes.
- Association (the E2/E4 subject):
  - `ClassAgnosticAssociator` (in `nuscenes_native_evaluator.py`): greedy,
    score-ordered, static, ego-frame, gate 2.0 m, max_age 5.
  - `GlobalCentroidAssociator`: subclass with a byte-identical matcher; only
    the gating frame is global. **Requires `set_ego_pose(ego_pose)` before
    every `step(proposals)`.** Selected via `--association-frame global`.
  - The old `diagnosis.outdoor_associator_ablation_probe.Associator` is
    RETIRED. Never use it for new numbers; it exists only for provenance.
- Class-agnostic association (`--association-class-agnostic`) is required
  for meaningful track statistics; class-aware association structurally
  zeroes label-switch counts.
- Native CenterPoint labels have zero label switches by construction, so
  temporal-labeling axes are null on the native stream; the temporal layer's
  outdoor value is association/registration, measured by OV-TCS.

## 8. Evaluation pipeline

### Production outdoor evaluator (use this, not `nuscenes_evaluator.py`)

```
python -u -m method_scannet.streaming.nuscenes_native_evaluator \
  --output "$RUN_DIR/outputs" \
  --axes baseline m11 m12 m21 m22 m31 m32 \
  --scene-split val \
  --proposal-source detguided \
  --association-class-agnostic \
  --association-frame global \
  --m32-distance 1.0
```

Key flags: `--scene-limit N` / `--scenes ...` for smokes;
`--score-threshold` / `--proposal-score-threshold` for gating.

### Metric conventions (must match the paper exactly)

- OV-TCS system score: mean over ALL tracks; singletons included at 0.
- `csr_mean`: all tracks, singletons contribute 0. A diagnostic "CSR over
  L≥2 tracks" also exists — always label which one you are reporting.
- GT-instance fragmentation: predicted tracks per GT instance (distinct
  from CLEAR-MOT Frag, which counts interruptions of GT-matched tracks).
- Bootstrap CIs: scene-cluster bootstrap, 10^4 resamples; the primary
  statistic is the pooled (track-weighted) mean; scene-weighted mean is a
  labeled secondary. Never mix the two in one comparison.

### Evaluator-era provenance

The detection evaluator was revised between the May-era runs and the
June/July production runs (offset ≤ 5×10⁻⁴ on the anchor). Any table mixing
eras must footnote it. When quoting a number, name the run directory it
came from. The paper's provenance footnotes (`main.tex`, Tab. 1) are the
template.

## 9. PBS workflow

**Never run CPU/GPU-heavy Python on the util node (ECE-util2) — a watchdog
kills it.** Everything heavy goes through PBS `qsub` to the `coss_agpu`
A100 Singularity container.

Standard header (see any `scripts/run_*.pbs`):
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
exec >"$LOG" 2>&1     # redirect output INSIDE the script; PBS -o is unreliable here
```

Job lifecycle:
```bash
qsub scripts/run_foo.pbs        # submit; note the job id
qstat -u $USER                  # queue state
# monitor by grepping the run.log for an explicit terminal string
# (echo 'FOO DONE' at the end of every PBS script; grep for it)
```

Rules:
- Every PBS script ends with a unique terminal echo (`... DONE`) so
  completion is grep-detectable.
- Smoke gate first: run 3–5 scenes with hard `assert`s in the same script
  before the full run, or as a separate short job. No full run without a
  passed smoke.
- CPU-only cache replays (e.g. E2-style, ~15 min) still go through PBS —
  the watchdog does not distinguish CPU-heavy from GPU-heavy.
- Never interrupt training mid-epoch. Completion beats wall-clock budget;
  do not propose time-based cutoffs.

## 10. Common commands

```bash
conda activate openyolo3d-dev                      # Python 3.10, the only env

# Outdoor production eval (see §8 for full flags)
python -u -m method_scannet.streaming.nuscenes_native_evaluator --output ... --axes baseline

# Indoor full eval
python run_evaluation.py --dataset scannet200 --config pretrained/config_scannet200.yaml ...

# Paper build (from paper/)
pdflatex main.tex && pdflatex main.tex            # run twice for refs
python scripts/make_teaser_fig.py                 # regenerate Fig. 1

# Aggregations
python scripts/e4_aggregate.py                    # E4 sensitivity tables
python scripts/m1_m3_headline_audit.py            # headline audit

# PBS
qsub scripts/run_<name>.pbs && qstat -u $USER
```

## 11. Coding conventions

- Match the surrounding code's style; this is research code — plain
  functions and scripts, no speculative abstractions, no new dependencies
  for what a few lines can do.
- New experiment code goes next to what it varies: evaluator changes in
  `method_scannet/streaming/`, one-off probes in a `diagnosis*/` dir or the
  experiment's own `results/.../` dir.
- Prefer *reusing an existing knob* over new code (e.g. a sweep over an
  existing `const_scale` beats a new mechanism).
- Every non-trivial probe/replay script prints explicit progress lines and
  a terminal string, and writes `metrics.json` — logs are the only UI on PBS.
- Comments state constraints, not narration. Korean comments exist in older
  code; write new ones in English.

## 12. Naming conventions

- Method axes: `M<phase><variant>` — M11 frame counting, M12 Bayesian,
  M21 weighted voting, M22 feature fusion, M31 IoU merge, M32 Hungarian.
  "phase1/phase2" = axis bundles.
- Experiments: E1 = MOT-metric comparison, E2 = open-vocab replication,
  E4 = associator sensitivity (these labels are load-bearing in the paper
  and results dirs; do not reuse them for new things).
- PBS scripts: `scripts/run_<experiment>_<detail>.pbs`.
- Result dirs: see §14. Branches: `<experiment-slug>` kebab-case.

## 13. Experiment conventions

- **Minimal decisive first**: every new hypothesis gets the cheapest 2-arm
  comparison that could falsify it, with an explicit stop condition written
  down *before* running. Promote to a sweep/full run only on a positive
  signal. Control signals get a mechanism-correlation check before any
  ablation built on them.
- **Verify premises before executing**: if a task's stated diagnosis or
  numbers conflict with what the code/data show, measure and report the
  discrepancy before running the requested job.
- **One variable per comparison.** If an arm changes two things, add the
  control arm or don't run it.
- **Report ETA**: when starting any run, state expected duration and the
  expected finish time in KST.
- **Reports in English**: instructions may arrive in Korean; all reports,
  paper text, tables, and generated docs are English.
- Reruns of the same experiment bump the version suffix; never overwrite an
  existing results directory.

## 14. How results are organized

```
results/<YYYY-MM-DD>_<experiment>_v<NN>/
  run.log         full stdout/stderr
  config.yaml     config snapshot (reproducibility)
  metrics.json    final metrics
  outputs/        artifacts (ply, npy, json, figures)
  notes.md        what/why/how + job id + validation checks (write this)
```

- `<experiment>` is snake_case (`scannet_eval`, `ablation_<name>`, …);
  `NN` is zero-padded from 01, incremented per rerun on the same day.
- Standard header for run scripts (auto-versioning):
```bash
DATE=$(date +%F); EXP=<experiment>
N=$(printf '%02d' $(($(ls -d results/${DATE}_${EXP}_v* 2>/dev/null | wc -l) + 1)))
RUN_DIR=results/${DATE}_${EXP}_v${N}; mkdir -p "$RUN_DIR/outputs"
```
- `results/` is gitignored except `results/experiment_tracker.md` (keep it
  updated: one line per experiment with date, dir, one-line outcome).
- Some pre-convention dirs exist (`results/e4_associator_sensitivity/`,
  `results/outdoor_m32_sweep/`, …); treat them as read-only provenance.
- Every number in the paper must trace to a `metrics.json` (or equivalent
  artifact) in `results/`. `paper/revision_log_m1m3.md` shows the
  traceability format; keep extending it.

## 15. Common pitfalls

1. **Util-node watchdog** kills heavy processes silently. PBS everything (§9).
2. **`nuscenes_evaluator.py` vs `nuscenes_native_evaluator.py`**: the former
   silently no-ops M21/M22/M31/M32. Production = native evaluator only.
3. **CenterPoint z-convention**: CenterPoint boxes are bottom-centered;
   treating z as center silently destroys IoU3D. Use the adapter's
   `.gravity_center` path; never hand-roll the shift.
4. **Worktrees**: `diagnosis/*` probes and `results/` are untracked and
   exist only in the main checkout. Jobs importing them must
   `cd /home/rintern16/OpenYOLO3D` and set
   `PYTHONPATH=/home/rintern16/OpenYOLO3D`, even when launched from a
   worktree. The Write tool is blocked outside the active worktree; use
   `cp` for delivery folders.
5. **Shared stash stack** across worktrees: never bare `git stash`/`pop`.
   Prefer a WIP commit; if stashing, `git stash push -u -m "<tag>"`, apply
   by SHA, drop by tag.
6. **Aggregation conventions drift**: pooled vs scene-weighted means, all-track
   vs L≥2 CSR, retired vs production associator — every historical number
   discrepancy in this project came from one of these. Always state the
   convention next to the number.
7. **nuScenes sweeps missing** (146/150 val scenes): anything assuming full
   sweep stacks fails quietly with degraded inputs.
8. **Class-aware association** zeroes label-switch counts structurally —
   an lsc of 0 there is not "perfect stability".
9. **PBS `-o` output**: rely on the in-script `exec >"$LOG"` redirect, not
   PBS's own output file.
10. **Global associator without `set_ego_pose`** silently degrades to
    nonsense gating; the call is required before every `step()`.

## 16. Things that must NEVER be changed

- **OV-TCS definition and aggregation** (product form `L_norm·(1−CSR)`;
  all-track mean with singletons at 0). Re-formulating (e.g. stability-only)
  is a settled question — see `results/2026-06-26_ablation_ovtcs_formulation_v01/`
  and the paper's §5.3. Do not reopen without explicit user instruction.
- **Published/paper-quoted result artifacts** under `results/` — read-only
  provenance. Never edit, regenerate-in-place, or delete; rerun into a new
  versioned dir instead.
- **Proposal caches** (§5) — regeneration is expensive and breaks
  replay-comparability.
- **The retired legacy `Associator`** must never produce new paper numbers.
- **`results/experiment_tracker.md`** is the only tracked file in `results/`;
  do not gitignore it or move it.
- **This file's rule zero**: no measured results in CLAUDE.md.
- **Epoch completeness**: never add time-based training cutoffs.

## 17. Checklist before any commit

- [ ] `git status` — only intended files staged; no `data/`, no `results/`
      payloads (tracker file excepted), no `.aux/.log` unless intentional.
- [ ] Code imports/runs from a clean shell (`conda activate openyolo3d-dev;
      python -c "import <module>"` or the relevant smoke).
- [ ] If the paper changed: `pdflatex` twice, 0 errors, no undefined refs,
      page count checked; delivery folder re-synced if the user expects it.
- [ ] Commit message: `type(scope): summary` (see `git log` for the house
      style: `feat(audit): …`, `docs(paper): …`).
- [ ] No experiment results pasted into CLAUDE.md or other tracked docs
      outside `results/` and the paper.

## 18. Checklist before any experiment

- [ ] Hypothesis + the cheapest falsifying 2-arm design + explicit stop/go
      condition, written into the run dir's `notes.md` *before* submission.
- [ ] Premises checked against code/data (does the flag exist? does the
      cache cover the scenes? which evaluator era?).
- [ ] Exactly one variable differs between arms.
- [ ] New `results/<date>_<exp>_v<NN>/` dir; config snapshot copied; no
      overwrite of an existing dir.
- [ ] PBS script: standard header, env block, in-script log redirect,
      smoke gate with asserts, unique terminal echo.
- [ ] Worktree import check (pitfall 4) if the job touches `diagnosis/*`.
- [ ] ETA + KST finish time reported to the user at submission.

## 19. Checklist before writing the paper

- [ ] Every number traces to a named artifact (`results/.../metrics.json`);
      record the mapping in `paper/revision_log_*.md`.
- [ ] One statistic per claim: pooled vs scene-weighted, all-track vs L≥2 —
      named explicitly; every CI belongs to the exact statistic quoted.
- [ ] All numbers from the *production* evaluator/associator; any
      legacy-era value carries a provenance footnote.
- [ ] Apply the paper's own anti-gaming rule to any headline comparison
      (population control: box conservation, coverage, within-stratum,
      GT-paired) before quoting it.
- [ ] Stale-number grep sweep after any renumbering (grep the old values
      across the whole .tex).
- [ ] Figures regenerated from scripts (`scripts/make_teaser_fig.py`), PNG
      visually inspected; figure numbers match the tables.
- [ ] Adversarial self-review pass (`paper/reviewer_sanity_check.md` is the
      running document); every known weakness either fixed or explicitly
      acknowledged in Limitations.
- [ ] Build clean twice; delivery folder synced.
