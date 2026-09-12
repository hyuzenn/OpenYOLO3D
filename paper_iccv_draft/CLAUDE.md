# Claude Agent Instructions for Paper Revision

You are an expert AI researcher and LaTeX typesetter. You maintain the ICCV/CVPR
layout for the paper **"Retrospective Confirmation for Identity-Consistent
Streaming 3D Perception."**

> **Full rewrite (2026-08-03).** The manuscript was rewritten end to end, then
> reframed as a **method paper**: the primary contribution is the training-free
> **retrospective confirmation module** (confirmation test + retrospective
> emission of the confirmed prefix; novelty rests on the assembled operator,
> never on the confirmation test alone). "Module" is a defined term (Method
> §1: composition of associator + confirmation test + emission policy), the
> title deliberately names the mechanism, not the module. Release is a
> reproducibility note, never a contribution bullet., and the
> matched controls are the evidence standard behind its claim, not the headline.
> Do not reintroduce the paper as an evaluation protocol. The finding remains an
> **explicit trade-off**:
> association accuracy (AssA) and class-aware tracking accuracy (AMOTA) improve
> at equal output budget, while mAP, detection recall, DetA, and aggregate HOTA
> degrade. Do not write that the module improves detection quality.

## 1. Writing rules (in force for every edit)

1. Keep only the final contribution and the experiments that support it.
2. **Every number is verified against canonical JSON** (§4), never quoted from
   a summary document.
3. **No internal terminology in the body.** Forbidden: OV-TCS, gate, gate
   sweep, Temporal Layer, Semantic Relabel, E1/E2/E2b/E2c, M11/M21/M22/M31/M32,
   gamma, retro, detguided. Use standard CV terms: *confirmation-based track
   initialization*, *matched control*, *emission policy*. Internal run IDs are not permitted
   anywhere (the supplement that once held them is gone).
4. No GT-free surrogate, no abandoned directions, no development history.
5. **Never present a non-significant result as a win.** A difference is an
   improvement only when its bootstrap CI excludes zero. Sensor-frame IDF1
   spans zero → "no detectable difference"; world-frame IDF1 is a small loss.
6. Central message: temporal confirmation improves identity consistency and
   semantic stability under matched controls, and explicitly trades away
   detection-oriented metrics.
7. Indoor and outdoor are the same contribution on different datasets. State
   the effect-size difference factually; do not generalise the indoor
   zero-AP-cost result to the outdoor setting.
8. Keep only tables needed for the final claim. **Figures are no longer capped
   at one** (rule retired 2026-08-26): the supervisor asked for qualitative
   figures, and more than one is expected. `fig:overview` (TikZ, single-column,
   top of Method) stays the anchor: panel (a) emission policies, panel (b) the
   two matched controls. **Panel (a) is now a measured example**, not a
   schematic — one pedestrian over six frames of nuScenes `scene-0966`, threshold
   identities `…007/…041/…041/…041/…121/…121` (the control's own association
   pass) vs. one identity under confirmation, with the emitted box identical
   across arms. Row 4 (provisional) draws no revision: the running vote is
   pedestrian throughout and confirmation changes status only. Re-mined
   2026-09-11 from canonical c1fix cells; the old `scene-0925` example came from
   a superseded run and is retired. Provenance and the only quotable values:
   `results/2026-09-11_canonical_figure_remining_v01/figure_values.json`; extra
   figure assets live in `figs/`. Any new qualitative figure must come from
   stored output — no schematic passed off as data. Do not re-add
   `retired/` figures.
9. **"Pre-registration" appears exactly once in the body** (Sec. Statistics),
   per supervisor feedback that the term reads as defensive to CV reviewers.
   Do not reintroduce it into endpoint descriptions or table captions; say
   "primary endpoint" / "exploratory". The documents go to the code release.

## 2. File tree
```
.
├── main.tex               # Structural driver (title, style, section inputs)
├── preamble.tex           # Global custom macros (currently empty — no macros needed)
├── main.bib               # BibTeX bibliography
├── figs/                  # Figure assets + paste-ready TikZ bodies (see figs/README.md)
├── retired/               # Dead assets from the pre-rewrite narrative. Do not reuse.
└── sec/
    ├── 0_abstract.tex     # Abstract only
    ├── 1_intro.tex        # Introduction + Related Work
    ├── 2_formatting.tex   # Method + Evaluation Protocol
    ├── 3_finalcopy.tex    # Experiments, Discussion, Limitations, Conclusion
    └── 9_supp_nsweep.tex  # Supplementary: full N-sweep numbers + the indoor
                           # waiting-vs-revision table (tab:provretro-supp).
```
**The supplement is reinstated, narrowly** (2026-08-27, supervisor
instruction: the pre-registration documents go to supplementary + the code
release, not the body). It had been removed on 2026-08-03. The original intent
still binds: **every claim must be supported inside the four section files**,
and the supplement may hold only the full numeric backing for a claim the body
already makes in prose --- never an argument, never detail moved there to dodge
a cut. Currently: `sec/9_supp_nsweep.tex` (full N-sweep table, whose claims are
stated in `sec:nsweep` and in the caption of `fig:nsweep`),
`tab:provretro-supp` (per-arm backing for `fig:provretro`),
`tab:consumer-robust` (backing for the robustness sentence of `sec:consumer`),
and the
pre-registration documents. Internal run IDs still have no permitted location
anywhere in the paper.
`retired/` holds the old figures (`figs_old/`), `make_gate_figs.py`, and
`figure_specs.md`. They render numbers that are no longer in the paper —
**never re-include them**.

The supplement is loaded after `\bibliography{main}` behind
`\clearpage\appendix`, renumbered `S1, S2, ...`.

## 3. Section contents
- `sec/1_intro.tex`: Introduction (confound → two matched controls → trade-off
  → indoor result → contributions) + Related Work (streaming perception, MOT,
  open-vocabulary 3D, evaluation metrics, controlled evaluation).
- `sec/2_formatting.tex`: Method (`sec:pipeline`, `sec:confirmation`,
  `sec:emission`) + Evaluation Protocol (`sec:datasets`, `sec:controls`,
  `sec:metrics`, `sec:stats`).
- `sec/3_finalcopy.tex`: six experiment subsections — `sec:detmatch`
  (Tab.~1), `sec:idmatch` (Tab.~2), `sec:emissionabl` (causal-emission table
  and `tab:policy`), `sec:consumer` (`tab:consumer`, what zero latency buys a
  proxy consumer), `sec:nsweep` (confirmation-window sensitivity, prose only;
  numbers in Tab.~S1), `sec:indoor` — then Discussion, Limitations, Conclusion.
  `sec:consumer` is evidence for the provisional policy, never a contribution
  bullet: keep the false-alert column beside the miss column, keep the
  world-frame tie with the score-ranked control visible, and never state that
  revision improves what the consumer holds.

## 4. Canonical number sources
Every experimental value in the paper traces to exactly one of the files
below. **Do not change a reported number without re-reading its JSON.**

All paths are relative to the repository root (`~/OpenYOLO3D`) unless the row
says otherwise.

| Comparison | File |
|---|---|
| Tab. 1 — detection-budget-matched, retrospective emission (main result) | `audit/table1_regen_c1fix_2026-08-29/table1_regenerated_results.json`; the accumulation arm's mAP/NDS only in `audit/table1_regen_c1fix_2026-08-29/accumulation_map_nds.json` |
| Tab. 2 — identity-budget-matched (top-$K$ + random-$K$) | `audit/tables23s1fig3_regen_2026-09-07/t2_trackmatch/e2b_report.json` |
| Tab. 3 — causal-emission ablation | `audit/tables23s1fig3_regen_2026-09-07/t3_causal/e2_report.json` |
| Tab. 4 / Tab. S1 / `fig:nsweep` — confirmation-window sensitivity | `audit/tables23s1fig3_regen_2026-09-07/nsweep_N{2,3,4,5}/e2_report.json`, aggregated in `nsweep_rows.json` (N=3 is its own cell here, no longer reused from another run) |
| Tab. `tab:policy` — provisional emission and revision | `results/2026-09-02_emission_policy_phase1_v01/policy_ledger.json` (box counts, invariants) and `score_policy.json` (mAP/NDS); the $H$ sweep incl. $H{=}0$ in `results/2026-09-02_emission_policy_phase2/sweep_summary.json` |
| Tab. 5 — indoor matched control (ScanNet200) | `results/2026-08-01_indoor_matched_control_v02/report.json` |
| `fig:provretro` / Tab. S2 — indoor waiting-vs-revision frontier | `audit/indoor_provretro/frontiers.json` and `scannet_provisional_summary.json` (one streaming pass; confirmation rows recomputed from that same pass, never mixed with the Tab. 5 run) |
| `tab:consumer` / `sec:consumer` / `tab:consumer-robust` — proximity-critical consumer | `results/2026-09-10_downstream_d6_v01/d6_paper_numbers.json`, generated by `scripts/downstream_d6_paper_numbers.py` (never hand-edited); design and frozen endpoints in `docs/downstream_d6_prereg.md`. The score-ranked row is the identity-budget peak-score control, whose selection uses whole-track information |
| Cross-detector stress test, BEVFusion-L (supplement + the cross-detector paragraph in Limitations) | `~/pretrained/bevfusion_nuscenes/q6_paper_numbers.json` — **home-relative, not under the repo** |

**Superseded, and no longer canonical** (kept on disk as historical audit
record; they were computed on the pre-c1fix single-sweep cache and their box
counts — 360,309 / 754,500 / 1,029,380 — contradict the current manuscript):
`results/2026-07-30_e2c_retro_thrmatch_v01/`,
`results/2026-07-31_e2b_trackmatch_v01/`,
`results/2026-07-28_e2_thrmatch_v01/`,
`results/2026-08-04_nsweep_N{2,4,5}_v01/`. Do not quote a number from them.
The canonical detection cache for every row above is
`results/outdoor_native_temporal_cpcache_thr000_10sweep_gravity_c1fix`
(6,019 files, 718,786 boxes); cross-table agreement is checked by
`audit/tables23s1fig3_regen_2026-09-07/consistency_audit.py`.

`q6_paper_numbers.json` is an aggregate: it carries every BEVFusion-L value the
manuscript cites together with the upstream artifact each one came from, so a
single file closes the provenance chain the way the regeneration reports do for
the CenterPoint comparisons. It is **generated, not hand-written** — rebuild it from
the frozen artifacts rather than editing it, and note that it is the one
canonical source living under `pretrained/` rather than `results/`, because the
BEVFusion runs are detector artifacts rather than a dated experiment directory.
Upstream artifacts it reads: `q6_full/` (PBS 121986), `q6_phase2/` (PBS 122055
then 122118) and `q6_ci/` (PBS 122249).

Pre-registration documents: `experiments/preregistration_2026-07-28.md`,
`experiments/preregistration_E2b_2026-07-31.md`,
`experiments/preregistration_indoor_matched_2026-08-01.md`.

Fixed facts that have been gotten wrong before:
- The outdoor associator is **class-aware in the sensor frame** (a detection
  continues only a track of its own class; `CentroidAssociator`, the default,
  no canonical run passes `--association-class-agnostic`) and **class-agnostic
  in the world frame** (`GlobalCentroidAssociator`). Evidence: label-switch
  count 0 in every sensor-frame cell, nonzero in every world-frame cell, for
  both detectors. Never write "class-agnostic associator" without the frame.
- **Every identity-budget control is a whole-track, budget-matched reference,
  not a streaming policy**: Tab.~2 top-$ (whole-track mean score), random-$,
  the accumulation control at identity budget (whole-track peak refined score),
  the `sec:consumer` score-ranked row (whole-track peak score), and the indoor
  top-$ all use a per-sequence budget known only at sequence end. Say so with
  the word "whole-track"; never call one a baseline or a streaming policy.
- Outdoor is **150 scenes**, not 146.
- Bootstrap: 10,000 resamples, seed `20260718`.
- **No intervals exist for AMOTA, mAP, or NDS** — all three are whole-split
  estimators. Quote them as point estimates only; never say "all N intervals"
  about a list that includes one of them.
- Fragmentation and track-length are **pre-selection invariants** (identical
  between the baseline and confirmation arms) — never attribute them to the
  method.
- ConceptGraphs external validation is **excluded** from the rewritten paper
  (no matched control). Do not reintroduce it without one.
- The cross-detector arm under test is **confirmation with retrospective
  emission**, the same arm as Tab.~1 — not the provisional policy. The
  supplement bounds the premise of `sec:detmatch`, and no cross-detector claim
  may be made about provisional emission, which runs on one detector only.
- The confidence-accumulation control at the **emission** budget is
  track-granular: it never exceeds $K$ and a small shortfall is reported. Never
  write that it "matches $K$ exactly" — only the score-threshold control does.
  (`audit/cbmot/CBMOT_CI_AUDIT.md` asserted the opposite until 2026-09-05 and
  carries a dated correction.)
- All **world-frame** BEVFusion AMOTA figures are emission-only, taken from the
  frozen-mask arm; the confirmation arm's own world AMOTA also carries the
  track-voted relabelling and is higher. Do not quote the pre-control values --
  both are in `q6_paper_numbers.json` under `frozen_mask_control`.

## 5. Constraints
- **LaTeX math:** indicator as `\mathbb{1}`.
- **Bibliography:** proper `author={Last, First and others}`.
- **Verify every change with a full build:**
  `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
  — must end with 0 undefined references and 0 errors.
  Current build (2026-09-11, after the canonical Fig. 1(a)/Fig. 2 re-mining):
  **15 pages**, 0 errors, 0 undefined references, 0 overfull
  and 0 underfull boxes. Body must end by the bottom of page 8 (ICCV limit,
  references excluded); it ends at **9.91 pages, an overrun of 1.91** (the
  figure re-mining added +0.05; before it the identity-budget control
  definition was added to the Protocol, +0.14). `fig:nsweep` was
  removed from the body on 2026-09-11 (its numbers are all in Tab.~S1 and
  `sec:nsweep` states its claims in prose); the file stays in `figs/`, do not
  move it into the supplement. `sec:consumer` cost 0.22 pages and the removal
  saved 0.50. Remaining cuts are deferred pending supervisor feedback.
  **Do not judge the page count from `main.log` page markers** — they cannot
  distinguish a body ending on p8 from one spilling a few lines onto p9.
  Measure the last body line with `\pdfsavepos`: the `bodyend` label in
  `main.aux` gives (page, y-in-sp), and body pages
  `= (page - 1) + (720 - y/65536) / 648`.
