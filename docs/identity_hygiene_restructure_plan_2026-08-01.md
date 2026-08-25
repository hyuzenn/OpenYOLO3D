# Manuscript restructuring plan v2 — identity-hygiene framing

**Status: PLAN ONLY. No manuscript file is modified by this document.**
Target: `paper_iccv_draft/` (canonical). Rewrite begins only after the
indoor matched control completes and the plan is re-frozen against it.

v2 incorporates the writing rules of 2026-08-01: standard terminology only, no
development history, no abandoned directions, every number traced to a
canonical machine-generated file, claims narrowed where evidence is thin.

---

## 1. Writing rules in force

1. **Community-standard terminology only.** No internal identifiers anywhere in
   the main paper. §2 gives the required mapping.
2. **No development history.** No account of how the method evolved, no audits,
   no "we initially found…". The paper describes the final method.
3. **No abandoned directions or failed exploratory work.** §3 lists what this
   removes.
4. **Only experiments that support the final contribution.** Each must answer a
   reviewer question (§6).
5. **Every number verified against the canonical JSON**, not against notes,
   summaries, or generated markdown tables. §5 is the single transcription
   source, with file + key path for each value.
6. **Self-contained.** A reviewer needs no project history, no experiment IDs.
7. **Narrow, don't over-claim**, wherever a claim rests on one experiment or an
   open confound (§7).

## 2. Terminology mapping (internal → paper)

| Internal | Paper wording |
|---|---|
| M11 / FrameCountingGate / "the gate" | confirmation-based track initialization; a track is emitted once it has been observed in at least $N$ frames |
| retro / retroactive emission | offline (retrospective) emission: a confirmed track's pre-confirmation detections are emitted at their own timestamps |
| streaming / prefix-deleting emission | causal (online) emission: detections before confirmation are not emitted |
| M21 | per-track majority voting over frame-wise class predictions |
| M31 | IoU-based duplicate suppression across tracks |
| M32 | Hungarian centroid matching for duplicate merging |
| phase1 | the full temporal layer (confirmation + label voting + duplicate suppression) |
| E2/E2c control | detection-budget-matched control: the same detector thresholded to emit the same number of detections |
| E2b control | identity-budget-matched control: the highest-scoring $K$ tracks per sequence, $K$ = number the temporal layer emits |
| indoor control | identity-budget-matched control on ScanNet200 |
| gamma / γ cache | a frozen, pre-computed CenterPoint detector |
| ego / global frame | sensor-frame vs. ego-motion-compensated world-frame association |
| lsc | class-label switches: frames in which an instance's aggregated class label changes |
| ttc | frames to confirmation |
| OV-TCS | (removed — see §3) |
| val312, n=6202, cell names, job IDs | ScanNet200 validation split (312 scenes); no IDs in the paper |

Experiment IDs and script paths may appear **only** in supplementary
reproducibility text, never in the main paper.

## 3. Removed content

**Retired by pre-registration (non-negotiable):**
- Fig. 1 "Same boxes, opposite verdicts" (`sec/1_intro.tex:1-11`).
- Table `tab:gate` (`sec/3_finalcopy.tex:7-27`) and the monotone gate-sweep
  narrative built on it.

**Removed under the no-abandoned-directions rule:**
- The GT-free consistency surrogate: entire supplementary section
  (`sec/4_supp.tex`) and the "Exploratory: a GT-free surrogate (negative
  result)" subsection (`sec/3_finalcopy.tex:258`). This is an abandoned
  direction and is additionally unsupportable: the quantity is computed before
  the temporal layer runs and is therefore numerically identical across all
  settings of $N$. **This deletes the manuscript's original headline
  contribution — confirm before the rewrite (§9).**

**Removed under the no-history rule:**
- Any narrative about emission-policy being corrected during development. The
  online/offline distinction survives *only* as a method design choice with an
  ablation, stated in the present tense, with its latency cost.

**Removed as unsupportable:**
- Every sentence attributing fragmentation, track-length, or surrogate-metric
  changes to the temporal layer. These are computed pre-gate and are invariant
  to it.
- "Identical gate in both domains" as a bare claim; replaced by an explicit
  operator definition plus the disclosed effect-size asymmetry.

## 4. Final claim, stated once

> Confirmation-based track initialization, with offline retrospective emission,
> improves association accuracy (AssA) and class-aware tracking accuracy
> (AMOTA) over non-temporal controls matched on **detection budget** and on
> **identity budget**, while reducing per-frame detection quality (mAP, DetA,
> recall, HOTA) at equal budget. The contribution is identity hygiene, not
> detection quality.

## 5. Verified numbers — single transcription source

All values re-read from canonical JSON on 2026-08-01. **Transcribe from this
table only.**

### 5.1 nuScenes, detection-budget-matched control
Source: `results/2026-07-30_e2c_retro_thrmatch_v01/e2_report.json`,
key `arms.<frame>.rows.<Baseline|Control|Gate>`.

| frame | arm | boxes | mAP | HOTA | AssA | DetA | IDF1 | AMOTA |
|---|---|---|---|---|---|---|---|---|
| sensor | Baseline | 1,029,380 | 0.34082783841011366 | 0.1371550707251852 | 0.18963950823702655 | 0.10074368820012089 | 0.07587850930323649 | 0.05006365443219036 |
| sensor | Control | 360,309 | 0.33240680007473494 | 0.209797311163527 | 0.20435369286245866 | 0.21835808207292098 | 0.15174923921497557 | 0.05471525631228811 |
| sensor | Temporal | 360,309 | 0.20230112142111337 | 0.19600551893665696 | 0.25348327424350314 | 0.15367896887283902 | 0.1531262738032531 | 0.08866254504985036 |
| world | Baseline | 1,029,380 | 0.34082783841011366 | 0.20107043131855798 | 0.40886323057470636 | 0.09951745727532552 | 0.1763889123014968 | 0.15798266676067801 |
| world | Control | 754,500 | 0.3395899207508847 | 0.2306523482507052 | 0.4174570712840486 | 0.12823959861222403 | 0.2190111704975454 | 0.1655428493887176 |
| world | Temporal | 754,500 | 0.28998945307427004 | 0.22633601781382018 | 0.4313962140421516 | 0.11956992748341255 | 0.2120558838146778 | 0.20334507457404835 |

Budget match, `arms.<frame>.threshold`: sensor target 360,309, threshold
0.187537282705307, `rel_err` 0.0; world target 754,500, threshold
0.12113966792821884, `rel_err` 0.0.

Bootstrap CIs of the combined Δ(Control→Temporal),
`arms.<frame>.bootstrap_ci_combined_delta` (seed 20260718, 10,000 resamples):

| metric | sensor | world |
|---|---|---|
| AssA | [+0.037012859662367295, +0.061492206094542756] | [+0.0086096315440935, +0.01953602487942332] |
| HOTA | [−0.01801607603177061, −0.010269435646390377] | [−0.0059243990955994115, −0.0027070762648551136] |
| DetA | [−0.07003479650345124, −0.059285278367346386] | [−0.010110490961788152, −0.007195640679204607] |
| IDF1 | [−0.0045993307427198635, +0.006447686118310751] | [−0.00912647097657286, −0.004730882242129872] |
| recall | [−0.1483210019188161, −0.11696636512405845] | [−0.040435065112426456, −0.028591703286894022] |

Note: sensor-frame IDF1 CI **spans zero** — report as no difference, not a win.

### 5.2 nuScenes, identity-budget-matched control
Source: `results/2026-07-31_e2b_trackmatch_v01/e2b_report.json`.

| quantity | sensor | world | key |
|---|---|---|---|
| identities matched | 64,428 = 64,428 | 108,225 = 108,225 | `arms.<f>.track_match.n_tracks_{total,target}`, `track_match_exact` true |
| control detections | 190,708 | 608,168 | `track_match.n_pred_boxes_total` |
| AssA control / temporal | 0.23461833766997747 / 0.25348327424350314 | 0.41733204938444785 / 0.4313962140421516 | `combined.{Control,Gate}.AssA` |
| AssA Δ 95% CI | [+0.01293020847600491, +0.025763164280633435] | [+0.010112735653526108, +0.018491480777181567] | `bootstrap_ci_combined_delta.AssA` |
| AssA Wilcoxon p | 2.153016315808772e-16 | 4.068848160102626e-11 | `paired_stats.AssA.p` |
| rank-biserial | 0.7730684326710817 | 0.6213686534216335 | `paired_stats.AssA.rank_biserial` |
| sequences favouring temporal | 117/150 | 111/150 | `paired_stats.AssA.n_scenes_gate_better` |
| AMOTA control / temporal | 0.05471279768612512 / 0.08866254504985036 | 0.15866449153714332 / 0.20334507457404835 | `amota.{Control,Gate}` |
| random-$K$ AssA range | 0.23210673625863687–0.2459747531403993 | 0.38780977138537726–0.393409727070321 | `random_K_AssA` |

The two sensor-frame controls give AMOTA 0.05471525631228811 (detection-matched)
and 0.05471279768612512 (identity-matched): **agree to five decimals, not
identical**. Write "≈".

### 5.3 ScanNet200 reference values
Source: `results/2026-05-15_streaming_ablation_core_temporal/pbs_A_baseline_m11_m12_v01/axis_{baseline,M11}/`,
keys `summary.json:AP`, `temporal_metrics.json:{label_switch_count.total,
n_unique_instances_total, time_to_confirm.n_instances}`. 312 scenes.

| quantity | baseline | temporal layer |
|---|---|---|
| AP | 0.19559525675483355 | 0.19539843680338323 |
| class-label switches | 23,385 | 17,023 (−27.20%) |
| emitted instances | 10,475 | 10,413 (−0.59%) |
| confirmed-instance population | 10,321 | 10,249 |

Correction carried into the paper: the emitted-identity reduction is
**−0.59%** (10,475→10,413). Earlier internal text quoted −0.70%, which is the
confirmation-population counter, a different quantity.

### 5.4 Indoor matched control
Pending (`results/2026-08-01_indoor_matched_control_v02/report.json`). Nothing
transcribed until it exists and reproduces the §5.3 anchors.

## 6. Experiment inventory — each must earn its place

| Experiment | Reviewer question answered | Keep? |
|---|---|---|
| Detection-budget-matched control (§5.1) | "Does it only look good because it emits fewer detections?" | Yes — headline |
| Identity-budget-matched control (§5.2) | "Is the gain just emitting fewer identities?" | Yes — decisive |
| Emission-policy ablation (online vs offline) | "Why offline? What does it cost?" | Yes — one table row + latency statement |
| Association-frame factor (sensor/world) | "Does it hold under a different association regime?" | Yes — every result reported in both |
| Temporal layer components (confirmation vs label voting vs duplicate suppression) | "Which component does the work?" | Yes — compact ablation |
| ScanNet200 identity-budget-matched control (§5.4) | "Does it generalize beyond one pipeline?" | Conditional (§8) |
| Gate-strength sweep over $N$ | superseded by matched controls | **No** |
| GT-free surrogate validation | supports no surviving claim | **No** |
| Fragmentation / track-length reporting | invariant to the method | **No** |

## 7. Required narrowing

- **AssA gain.** One confirmation threshold ($N{=}3$); the identity-matched
  control leaves detection budgets unequal (sensor 190,708 vs 360,309; world
  608,168 vs 754,500). Write: *"at a single confirmation threshold and matched
  identity budget, association accuracy improves; we do not isolate whether
  this reflects the selection of better identities or the greater length of
  confirmed tracks."* State the open control explicitly.
- **AMOTA.** Dataset-level, no interval. Write: *"consistent in sign and
  magnitude across both association regimes"*, never "significant".
- **Sensor-frame IDF1.** CI spans zero → "no measurable difference".
- **Random-$K$ proximity.** Sensor-frame random selection reaches AssA up to
  0.2460 against 0.2535; disclose in-text.
- **Cross-domain.** Effect sizes differ by two orders of magnitude (0.59%
  identity reduction indoors vs ~89% outdoors); frame as a consistency check,
  not equivalence.
- **Offline emission.** Every table reporting it carries: offline-only,
  $N{-}1 = 2$ frames = 1.0 s at 2 Hz, at track initialization only.

## 8. Indoor branches

- **Temporal layer wins on class-label switches:** keep as a cross-pipeline
  consistency section with the effect-size asymmetry stated.
- **Control matches or wins:** demote to one paragraph; all identity-hygiene
  claims become single-dataset; add limitation.

## 9. Decisions

1. **Removing the GT-free surrogate section — APPROVED 2026-08-01.** The
   supplementary section and its main-text subsection are deleted outright, as
   is any other material that does not support the final contribution. Only the
   control experiments a reviewer needs to follow the argument remain.
2. **Title** currently promises "What mAP Misses". Consistent with the final
   claim; re-decide once the abstract is drafted.
3. **Hardware provenance.** The indoor run executes on RTX A6000 while the
   frozen ScanNet200 reference (§5.3) was produced on A100-SXM4-80GB (the
   pipeline auto-seeds, so seeds differ across runs regardless). The
   pre-registered anchor check decides: **pass** → the indoor run is canonical
   and the reference row is quoted with a hardware footnote; **fail** → control
   numbers are not interpreted at all and the whole indoor study, including a
   no-gate baseline arm, is re-run on A100.

## 10. Execution order (after indoor completes + approval)

Re-freeze plan against the indoor report → Experiments → Method → Protocol →
Intro → Abstract → Discussion/Limitations → title → compile → self-review
against the reviewer-risk checklist. No number enters the manuscript that is
not in §5.
