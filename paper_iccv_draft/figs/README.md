# figs/ — qualitative figure assets

**Status (2026-08-26): `fig:overview` panel (a) in `sec/2_formatting.tex` is now the
real-data version** — the same measured pedestrian described below, drawn in the
existing single-column three-row style (score threshold / causal / retrospective).
That replaced the old schematic in place, so the standalone full-width figure in
this folder is **not currently included in the body**. It is kept as the
two-panel alternative (identity strip + failure inset) for the additional
qualitative figures planned for the next revision.

Added 2026-08-26. Answers the supervisor's manuscript comment #6 ("qualitative
figure 없음 → 추가") and 2-week TODO #7.

## Files
| File | Use |
|---|---|
| `fig2_identity_consistency_body.tex` | **paste-ready**: `\input{figs/fig2_identity_consistency_body}` from a `sec/*.tex` |
| `fig2_identity_consistency.tex` | standalone, compiles on its own (for quick preview) |
| `fig2_identity_consistency.pdf` / `.png` | rendered preview |
| `fig3_semantic_stability_body.tex` | **paste-ready** indoor layout --- currently a TEMPLATE, see below |
| `fig3_semantic_stability.tex` | standalone preview at the CVPR `\textwidth` |
| `fig3_semantic_stability.pdf` / `.png` | rendered preview |

Verified: the body compiles under `main.tex`'s exact preamble and typesets as a
full-width float in the two-column ICCV layout. It needs
`\usetikzlibrary{positioning, arrows.meta, backgrounds, fit}` (included at the
top of the body file) and `amssymb` for `\checkmark` (already in `main.tex`).

## Fig. 3 is a template, not a result
`fig3_semantic_stability_body.tex` draws the intended indoor layout with every
label chip left as `?`. This is deliberate: **no stored run persists the
per-frame label sequence**, so the class names do not exist yet. The number of
label runs drawn matches the measured switch count of the leading candidate
(scene0655_00 instance 4: 4 switches over 20 frames, mask IoU 1.000). Recovering
the real sequence needs
`results/2026-08-26_qualitative_figure_mining_v01/mining_scripts/run_fig3_replay.pbs`
(three arms, no production-code change, cross-checked against the frozen
per-scene switch counts). **Never fill the chips with guessed class names, and
do not put this in the body until the replay has run and the banner is removed.**
Candidate instances: `.../FIGURE_SHORTLIST.md`.

## Data provenance — every value is from stored output
Source run: `results/2026-07-30_e2c_retro_thrmatch_v01/` (150 nuScenes val
scenes), detection-budget-matched arms, **sensor frame**:

- control = score threshold t = 0.187537, `cells/ctrl_ego/axis_baseline/tracks.json`
- ours    = confirmation N = 3 + retrospective emission, `cells/retro_ego/axis_phase1/tracks.json`
- box budget matched exactly: 360,309 = 360,309

Panel (a) — `scene-0925`, GT pedestrian `1a2b708d2e2240c987b17c823713fd24`, frames 0–5.
Control IDs `120000000 → …056 → …000 → …000 → …000 → …131`; ours `120000000` throughout.
Displayed IDs are the last three digits. Scores 0.726/0.779/0.792/0.754/0.789/0.833,
centre distance ≤ 0.27 m, correct class in every frame. **The emitted box is identical
between the two arms in all six frames** (same translation/size/rotation/score), so the
panel isolates identity with no detection-quality confound.

Panel (b) — `scene-0104`, GT car `ad00b4de161548a09912a35d9ebca4c2`, frames 33–34 only
(mid-scene). Control emits `24001399` (0.753) then `24001450` (0.470), both correctly
classified, ≤ 0.35 m. Ours never emits it: 2 frames < N = 3.

Full candidate mining, ranked alternatives and the failure-case pool:
`results/2026-08-26_qualitative_figure_mining_v01/CANDIDATES.md`.

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
   error. `scene-0925` (ego) remains the sole 3-switch candidate and is far more
   tightly localised (0.27 m), so it stays.
5. **Detector soundness.** These sequences come from the same pipeline whose detector
   numbers are under review (TODO #1). If that changes `tracks.json`, re-mine before
   final rendering — the scripts are saved next to the candidate report.
6. ~~Caption still says "Test"~~ — fixed 2026-08-26. Note that `CANDIDATES.md`
   never contained a drafted caption sentence, only the verified per-frame facts;
   the caption now in the body was written from those facts (§4 of the report).
