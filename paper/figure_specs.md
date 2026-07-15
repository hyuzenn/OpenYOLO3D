# Figure specifications (camera-ready presentation plan)

## Fig. teaser — DONE (generated, in draft)

`figs/fig_teaser.png/pdf`, from `scripts/make_teaser_fig.py`. Three panels
(mAP / OV-TCS / Fragmentation), Ego (gray #8f8f8f) vs Global (blue #0072B2,
Okabe–Ito CVD-safe), direct value labels on every bar, panel annotations
"bit-identical" / "+24%" / "−58%". All numbers from the frozen main table
(Tab. 1); no new values. Placed as Fig. 1, page 1, above the introduction.
Goal: a reviewer understands the motivation in <10 seconds without reading
any text.

## Fig. qualitative (SPEC ONLY — needs one CPU rendering pass over cached labels; not executed)

**Message:** "Same scene, same detections: the ego-frame system shatters an
object into flickering fragments; the global-frame system keeps one stable,
consistently-labeled track." This is review item M7.

**Scene:** nuScenes `scene-0632` (token `b789de07180846cc972118ee6d1fb027`,
"Rain, industrial, turn right, parked cars", 40 keyframes) — the largest
measured ego→global per-scene OV-TCS delta in E1
(`results/e1_outdoor_mot_compare_v05/per_scene_metrics.csv`): 0.081 → 0.189
(+0.108). Backup with the same construction: `scene-0634` (token
`7210f928860043b5a7e0d3dd4b3e80ff`, 0.148 → 0.243) if scene-0632's parked-car
clutter renders poorly.

**Object selection (deterministic, no cherry-picking by eye):** from the
cached track dumps of the two arms at the default operating point (d=2.0 m,
a=5, p=0.0), pick the GT instance in scene-0632 with the highest
fragmentation count under ego among instances with ≥20 observed keyframes.
Report the selection rule in the caption.

**Layout:** 2 rows (Ego top, Global bottom) × 1 BEV panel + a label timeline.
- **Left (BEV, ~60% width):** bird's-eye-view LiDAR points (light gray) of
  the scene accumulated in the global frame; overlay the selected object's
  predicted track(s): one color per predicted track ID (Okabe–Ito order), box
  centers connected by a line per track. Ego row shows many short colored
  segments (fragments); Global row shows one long track. Annotate each row
  with its track count for this object (e.g., "7 tracks" vs "1 track").
- **Right (label timeline, ~40% width):** a horizontal strip per row, one
  cell per keyframe (40 cells), cell color = predicted argmax label at that
  keyframe (categorical, fixed label→color map shared across rows; legend
  below). Label switches appear as color changes; track breaks as thin black
  gaps. Under each strip print the per-object numbers: `L`, `CSR`, OV-TCS.
- **Annotations:** a single callout arrow on the ego strip at the densest
  switch region: "label flicker + fragmentation"; on the global strip: "one
  track, stable label".

**Caption (draft):** "Qualitative comparison on nuScenes scene-0632, the
scene with the largest measured ego→global OV-TCS gap (0.081→0.189).
Detections are identical in both rows; only the association frame differs.
Top: ego-frame association splits the highlighted parked vehicle into
short-lived fragments whose labels flicker (timeline right). Bottom:
global-frame association maintains a single track with a stable label.
Per-frame mAP is bit-identical between the two rows; OV-TCS is the only
reported metric that separates them. Object selected as the most-fragmented
≥20-frame GT instance under ego (rule stated in Sec. X, no manual pick)."

**Data needed (all cached, CPU-only):** per-frame track IDs + argmax labels
for both arms (already produced by the E1 harvester when run with per-frame
dumps), LiDAR keyframe points for scene-0632 (on disk), GT match for the
selection rule (E1 matching output). No GPU inference.

## Statistical presentation (where CIs live)

- Abstract: headline delta quoted with 95% CI.
- Fig. teaser caption: CI + % of scenes improved.
- §6.1 text: full bootstrap statement (10^4 scene-level paired resamples).
- §5.2: effect size framed against the length-only baseline (~5×), F=89.9.
- Not invented anywhere: E2 (+6.8%) has no per-scene dump, so no CI is
  quoted for it (flagged in revision_notes.md).
