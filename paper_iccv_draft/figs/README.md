# figs/ — qualitative figure assets

**Status (2026-09-11): both figures are in the body.** `fig:overview` panel (a)
in `sec/2_formatting.tex` is the four-row real-data version (score threshold /
causal / retrospective / provisional), and `fig2_identity_consistency_body.tex`
is `\input` in `sec/3_finalcopy.tex` (Sec. `sec:detmatch`) as a full-width float.
Both draw the canonical examples re-mined on 2026-09-11 (scene-0966 pedestrian,
scene-0104 car); provenance below.

Added 2026-08-26. Answers the supervisor's manuscript comment #6 ("qualitative
figure 없음 → 추가") and 2-week TODO #7.

## Files
| File | Use |
|---|---|
| `fig2_identity_consistency_body.tex` | **paste-ready**: `\input{figs/fig2_identity_consistency_body}` from a `sec/*.tex` |
| `fig2_identity_consistency.tex` | standalone, compiles on its own (for quick preview) |
| `fig2_identity_consistency.pdf` / `.png` | rendered preview |
| `fig3_semantic_stability_body.tex` | indoor figure, in the body (see Fig. 3 status below) |
| `fig3_semantic_stability.tex` | standalone preview at the CVPR `\textwidth` |
| `fig3_semantic_stability.pdf` / `.png` | rendered preview |

Verified: the body compiles under `main.tex`'s exact preamble and typesets as a
full-width float in the two-column ICCV layout. It needs
`\usetikzlibrary{positioning, arrows.meta, backgrounds, fit}` (included at the
top of the body file) and `amssymb` for `\checkmark` (already in `main.tex`).

## Fig. 3 status (updated 2026-09-11)
Two separate facts; do not conflate them.

- **Inclusion:** in the body. `sec/3_finalcopy.tex` inputs it under
  `sec:indoor`. It is no longer a template: no `?` chips, no banner. Whether it
  stays in the body is an open editorial decision, not settled here.
- **Validation:** the replay has run (PBS 119908,
  `results/2026-08-26_qualitative_figure_mining_v01/fig3_replay/`). The figure
  draws scene0696_02 instance 28 from that replay, not the earlier template
  candidate scene0655_00. It used a regenerated Mask3D cache
  (`results/2026-08-26_mask3d_cache_regen_v01`) because the frozen Tab. 5
  cache is gone. `lsc_check.json` against the frozen run
  (`results/2026-08-01_indoor_matched_control_v02`) is **FAIL, 16/17 scenes
  differ**: regeneration is not bit-reproducible. The mechanism drawn is exact
  (602/604 instances have gate output identical to a suffix of the baseline, 0
  have a mid-sequence label change), but the frame numbers come from the
  replay, not from Tab. 5. Full provenance is in the header of
  `fig3_semantic_stability_body.tex`. Never fill label chips with guessed
  class names.

## Data provenance — every value is from stored output
**Re-mined 2026-09-11 from canonical c1fix cells.** The 2026-08-26 values came from
the superseded `results/2026-07-30_e2c_retro_thrmatch_v01/` run and are retired; on
the canonical stream the old `scene-0925` example reverses the claim.
Run directory: `results/2026-09-11_canonical_figure_remining_v01/` (`notes.md`,
selection criteria, cell md5s). **The only file the figures may quote is its
`figure_values.json`.** Sensor frame; cells
`audit/table1_regen_c1fix_2026-08-29/phase2_arms/cells/{ctrl_ego/axis_baseline,
retro_ego/axis_phase1, gamma_ego/axis_baseline}` (Tab. 1 arms, 257,259 boxes).
The threshold control re-runs association on its thresholded boxes, so its
identities come from a separate pass; the other rows share the frozen association.

Fig. 1(a) and Fig. 2(a) — `scene-0966`, GT pedestrian `46af915cdf034fc093620f883ddcdf6b`,
frames 0–5. Selected by the original rule plus two recorded additions (zero class
errors; pre-confirmation prefix inside the window). Threshold IDs
`129000007 → …041 → …041 → …041 → …121 → …121` (2 switches); confirmation and
frozen stream `129000007` throughout. Displayed IDs are the last three digits.
Scores 0.7076/0.5883/0.8067/0.7246/0.7323/0.8692, centre distance ≤ 0.168 m,
pedestrian in every frame. The emitted box is identical in all three streams in
every frame. Ledger: confirmed at frame 2, running vote pedestrian at every
frame, 0 relabels, never retracted — confirmation changes status, not label.

Fig. 2(b) — `scene-0104`, GT car `ad00b4de161548a09912a35d9ebca4c2`, frames 33–34
(kept by decision; canonical rank 4 under the original ranking). Threshold emits
`24000967` (0.7443, 0.237 m) then `24000996` (0.7357, 0.262 m). Late release never
emits it. The frozen stream holds it as two one-observation tracks, `24002581`
(t=33) and `24002664` (t=34): neither confirms; provisional emission emits both and
retracts both by t=38 (the first by the H=4 rule, the second at scene end).

Superseded history: `results/2026-08-26_qualitative_figure_mining_v01/CANDIDATES.md`.

## Open issues
1. ~~One-figure rule~~ — retired in `CLAUDE.md` §1.8 on 2026-08-26.
2. **Page budget.** Making panel (a) real-data grew the build from 10 to 11 pages.
   Deferred by agreement until the remaining experiments land and the revision is
   assembled in one pass. **Do not touch this yet.**
3. ~~`\resizebox` shrinks the fonts~~ — fixed 2026-08-26. The drawing is now laid
   out at 17.3 cm, just under `\textwidth` (6.875 in = 17.46 cm), and `\resizebox`
   is gone, so every label typesets at its true point size. The standalone preview
   now sets the same `textwidth` and only `\input`s the body, so preview and body
   cannot drift. Compiles with 0 overfull boxes. **Do not re-wrap in `\resizebox`.**
   The repeated inline "switch" word was removed (it did not fit at true size);
   the switch count now appears once per row in the left label, and the caption
   states that a dashed red arrow marks an identity switch.
4. ~~Frame~~ — resolved 2026-08-26 in favour of keeping the **sensor frame**.
   World-frame mining (PBS 119325, `strict_global.json`) returned 62 strict
   candidates vs. 31 for ego, but under the figure filter (5–8 frames, ≥2 control
   switches, 0 class errors) only 7 survive and **all 7 have just 2 switches**;
   the only world-frame pedestrian among them (scene 96) has a 0.70 m centre
   error. (That comparison was made on the superseded run; the 2026-09-11
   canonical re-mining keeps the sensor frame, see the provenance section.)
5. **Detector soundness.** These sequences come from the same pipeline whose detector
   numbers are under review (TODO #1). If that changes `tracks.json`, re-mine before
   final rendering — the scripts are saved next to the candidate report.
6. ~~Caption still says "Test"~~ — fixed 2026-08-26. Note that `CANDIDATES.md`
   never contained a drafted caption sentence, only the verified per-frame facts;
   the caption now in the body was written from those facts (§4 of the report).
