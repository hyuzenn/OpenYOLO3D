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
   schematic — one pedestrian over six frames of nuScenes `scene-0925`, control
   identities `…000/…056/…000/…000/…000/…131` vs. one identity under
   retrospective confirmation, with the emitted box identical across arms.
   Provenance and alternative candidates:
   `results/2026-08-26_qualitative_figure_mining_v01/CANDIDATES.md`; extra
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
    └── 9_supp_nsweep.tex  # Supplementary: full N-sweep numbers. NOT \input by
                           # main.tex — the body shows fig:nsweep instead.
```
**The supplement is reinstated, narrowly** (2026-08-27, supervisor
instruction: the pre-registration documents go to supplementary + the code
release, not the body). It had been removed on 2026-08-03. The original intent
still binds: **every claim must be supported inside the four section files**,
and the supplement may hold only the full numeric backing for a claim the body
already makes in prose --- never an argument, never detail moved there to dodge
a cut. Currently: `sec/9_supp_nsweep.tex` (full N-sweep table, whose claims are
stated in `sec:nsweep` and in the caption of `fig:nsweep`) and the
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
- `sec/3_finalcopy.tex`: five experiment subsections — `sec:detmatch`
  (Tab.~1), `sec:idmatch` (Tab.~2), `sec:emissionabl` (Tab.~3), `sec:nsweep`
  (Tab.~4, confirmation-window sensitivity), `sec:indoor` (Tab.~5) — then
  Discussion, Limitations, Conclusion.

## 4. Canonical number sources
Every experimental value in the paper traces to exactly one of these four
files. **Do not change a reported number without re-reading its JSON.**

| Comparison | File |
|---|---|
| Detection-budget-matched, retrospective emission (main result) | `results/2026-07-30_e2c_retro_thrmatch_v01/e2_report.json` |
| Identity-budget-matched (top-$K$ + random-$K$) | `results/2026-07-31_e2b_trackmatch_v01/e2b_report.json` |
| Causal-emission ablation | `results/2026-07-28_e2_thrmatch_v01/e2_report.json` |
| Indoor matched control (ScanNet200) | `results/2026-08-01_indoor_matched_control_v02/report.json` |
| Confirmation-window sensitivity (Tab.~4) | `results/2026-08-04_nsweep_N{2,4,5}_v01/e2_report.json` (+ N=3 reused from the E2c file above) |

Pre-registration documents: `experiments/preregistration_2026-07-28.md`,
`experiments/preregistration_E2b_2026-07-31.md`,
`experiments/preregistration_indoor_matched_2026-08-01.md`.

Fixed facts that have been gotten wrong before:
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

## 5. Constraints
- **LaTeX math:** indicator as `\mathbb{1}`.
- **Bibliography:** proper `author={Last, First and others}`.
- **Verify every change with a full build:**
  `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
  — must end with 0 undefined references and 0 errors.
  Current build: **10 pages**, 0 undefined references, 0 overfull boxes.
  Body must end by the bottom of page 8 (ICCV limit, references excluded);
  it currently overruns by ~1.05 pages and cuts are deferred pending
  supervisor feedback. **Do not judge the page count from `main.log` page
  markers** — they cannot distinguish a body ending on p8 from one spilling a
  few lines onto p9. Measure the last body line with `\pdfsavepos`.
