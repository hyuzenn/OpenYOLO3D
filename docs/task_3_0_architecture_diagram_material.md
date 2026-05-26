# Task 3.0 — Architecture diagram material (for image generation)

Date: 2026-05-21 · docs-only, no code changes.
Source of truth: `docs/task_3_0_architecture_inventory.md` +
`docs/task_3_0_method_implementation_audit.md`. All labels/metrics below are
verified against code; the diagram is honest (Outdoor temporal layer = only
M11/M12 wired; M21/M22/M31/M32 = negative ablation / dimmed).

---

## 1. ASCII diagram — unified system (Indoor + Outdoor)

```
LEGEND   ░░ upstream OpenYOLO3D (frozen, not a contribution)
         ██ user contribution (highlighted)
         ▒▒ negative ablation (dimmed)        ✗ bypassed

           STEP A               STEP B            ┌── STEP C / D / F ──┐         OUTPUT
        3D proposals          2D labels             temporal layer            (metric)
 ┌──────────┐ ┌─────────────┐ ┌──────────────┐  ╔═══════════════════════════╗ ┌───────────────┐
 │ ScanNet  │►│ ░░ Mask3D ░░│►│ ░░ YOLO-World │► ║ TEMPORAL CONSISTENCY LAYER║►│ confirmed map │
 │  RGB-D   │ │  (Step A)   │ │ ░░ (Step B)   │  ║      (contribution)       ║ │  lsc · ttc    │
 └──────────┘ └─────────────┘ └──────────────┘  ║                           ║ │  Mean AP      │
   INDOOR                                        ║ C: ██ M11 / M12 ██        ║ └───────────────┘
                                                 ║    PHASE-1 STABILIZER     ║
 ┌──────────┐ ┌─────────────┐ ┌──────────────┐  ║    (sole positive axis)   ║ ┌───────────────┐
 │ nuScenes │►│ ██ γ Center-│►│ ░░ YOLO-World │► ║                           ║►│ mAP 0.0526    │
 │ 6cam+LiD │ │ ██ Point ██ │ │ ░░ relabel    │  ║ D: ▒▒ M21 / M22 ▒▒  ┐     ║ │ open-vocab    │
 └────┬─────┘ │(replaces A) │ └──────────────┘  ║ F: ▒▒ M31 / M32 ▒▒  ┘     ║ │ (capability)  │
   OUTDOOR    └─────────────┘   MODE 1          ║   negative ablation       ║ └───────────────┘
   Mode 1                                        ║                           ║
      │       ┌─────────────┐ ┌──────────────┐  ║ ██ StreamingNuScenes-     ║ ┌───────────────┐
      │       │ ██ γ native │►│  ✗ bypass     │► ║ ██ Evaluator ██           ║►│ mAP 0.3407    │
      └──────►│ ██ class  ██ │ │  ✗ YOLO       │  ║ (Outdoor: only M11/M12    ║ │ detection     │
   OUTDOOR    │ (native head)│ └──────────────┘  ║  are actually wired)      ║ │ (Step 2a/2b)  │
   Mode 2     └─────────────┘   MODE 2           ╚═══════════════════════════╝ └───────────────┘
```

### Compact linear traces (unambiguous reading)

```
INDOOR        : ScanNet RGB-D ─► ░Mask3D (A)░ ─► ░YOLO-World (B)░
                ─► ██TEMPORAL██ { C: M11/M12 gate · D: M21/M22 · F: M31/M32 }
                ─► confirmed instance map ─► lsc, ttc, Mean AP

OUTDOOR Mode 1: nuScenes (6cam+LiDAR) ─► ██γ CenterPoint██ (replaces A)
   Stage C       ─► ░YOLO-World relabel (B)░  [γ native class discarded]
   open-vocab    ─► ██TEMPORAL: M11/M12 only██ ─► mAP 0.0526 (capability)

OUTDOOR Mode 2: nuScenes ─► ██γ CenterPoint native██ ─► ✗ bypass YOLO
   native        ─► ██native class + TEMPORAL: M11/M12██
   Step 2a/2b    ─► mAP 0.3407 (detection; Step 2b in progress)
```

### OpenYOLO3D streaming 6-step A–F skeleton (per-frame loop)

Shows exactly which steps of the upstream pipeline host the contribution.
Steps A, B, E are frozen upstream (gray ░); the temporal layer attaches at
C, D, F. (Source: `docs/task_1_1_streaming_design.md`.)

```
 ░ A  instance-level visibility (D3)       ◄ consumes 3D proposals (Mask3D / γ)   upstream
 ░ B  2D detection (YOLO-World)            ◄ open-vocab labels   [Mode 2: ✗ bypassed]
 █ C  instance registration gate          ◄ ██ M11 / M12 ██   PHASE-1 stabilizer (positive)
 ▒ D  label vote accumulation             ◄ ▒▒ M21 / M22 ▒▒   negative ablation (dimmed)
 ░ E  per-instance current class + score   ◄ upstream readout                     upstream
 ▒ F  spatial merging                      ◄ ▒▒ M31 / M32 ▒▒   negative ablation (dimmed)
```

### Emphasis callouts (for the figure)

- **M11 / M12 = Phase-1 sole stabilizer** — the only axis with a positive
  effect; the "proposal-agnostic temporal consistency layer" headline. (M12
  ≡ M11 until the Task 1.4c silent-bug fix.)
- **M21 / M22 / M31 / M32 = negative ablation (dimmed)** — correct
  implementations, but: M21 lsc +1.04% (wrong sign), M22 AP −27% (CLIP narrow
  band), M31 ~null (indoor lacks duplicates), M32 AP −49% cascade (2 m
  over-merge). Drawn faded with a small "negative ablation" tag.
- **Shared γ source, two class assignments** — Mode 1 and Mode 2 use the
  *same* γ CenterPoint proposals; the only difference is who labels them
  (YOLO-World vs native head). Show the nuScenes input splitting into both.
- **Outdoor temporal honesty** — only M11/M12 are wired in
  `StreamingNuScenesEvaluator`; M21/M22/M31/M32 install but are never called
  (silent no-op). The dimmed D/F row carries an Outdoor "(no-op)" marker.

---

## 2. Image generation prompt (ML paper figure style)

> Paste into Google Nano Banana / image tool. Style block first, then the
> structured content. Edit color names to taste.

**Style:** A clean, professional academic machine-learning system
architecture diagram in the style of a CVPR/NeurIPS paper figure. Flat
vector illustration, white background, thin rounded-rectangle node boxes,
crisp directional arrows, modern sans-serif labels, generous whitespace,
subtle soft shadows, no photorealism, no 3D bevels, no clutter. Horizontal
left-to-right data flow. High resolution, presentation-ready.

**Color semantics (must be visually distinct):**
- **Light gray / muted, low-contrast boxes** = frozen upstream components
  (not the author's contribution): Mask3D, YOLO-World, the OpenYOLO3D A–F base.
- **Saturated accent (teal/blue) solid boxes with a subtle glow** = the
  author's contributions: the γ CenterPoint adapter, the central Temporal
  Consistency Layer, and the StreamingNuScenesEvaluator.
- **A distinct strong color (green) for the M11/M12 sub-box**, labeled
  "Phase-1 stabilizer (primary)".
- **Faded / desaturated, dashed-outline, slightly transparent boxes with a
  diagonal hatch** = negative ablation: M21, M22, M31, M32, tagged
  "negative ablation".
- **A dashed red "bypass ✗" marker** on the Mode-2 YOLO-World step.

**Layout — three horizontal swimlanes that converge on one shared central
"Temporal Consistency Layer" block:**

1. **Top lane — Indoor:** Input box "ScanNet (RGB-D)" → gray box "Mask3D
   — 3D proposals (Step A)" → gray box "YOLO-World — 2D open-vocab labels
   (Step B)" → into the central Temporal Layer → output box "Confirmed
   instance map" with a metrics badge "lsc · ttc · Mean AP".

2. **Middle lane — Outdoor Mode 1 (open-vocab, Stage C):** Input box
   "nuScenes (6 cameras + LiDAR)" → teal box "γ CenterPoint adapter
   (replaces Step A)" → gray box "YOLO-World relabel (Step B)" with a small
   note "γ native class discarded" → into the central Temporal Layer →
   output box with badge "mAP 0.0526 — open-vocab capability".

3. **Bottom lane — Outdoor Mode 2 (native, Step 2a/2b):** the same
   "nuScenes" input splits down to a teal box "γ CenterPoint — native class
   head" → a box "YOLO-World" crossed out with a red dashed "✗ bypass" →
   into the central Temporal Layer → output box with badge "mAP 0.3407 —
   detection (Step 2a; 2b in progress)".

**Central block — "Temporal Consistency Layer (our contribution)":** a large
teal rounded rectangle spanning all three lanes, containing three stacked
sub-rows:
- Row C (green, emphasized): "M11 / M12 — registration gate · PHASE-1
  STABILIZER (sole positive axis)".
- Row D (faded, hatched): "M21 / M22 — label assignment".
- Row F (faded, hatched): "M31 / M32 — spatial merge".
- A small caption inside: "Outdoor: only M11/M12 wired (M21/M22/M31/M32 =
  negative ablation)".
- A teal sub-label at the block's base: "StreamingNuScenesEvaluator
  (Outdoor) / StreamingScanNetEvaluator (Indoor)".

**Annotations:** an arrow from the "nuScenes" input fans out to both the
Mode-1 and Mode-2 γ boxes, with a small tag "same γ proposals, different
class assignment". A compact legend box in a corner: gray = "frozen
upstream", teal = "our contribution", green = "primary (stabilizer)", faded
hatched = "negative ablation".

**Title (top, small):** "Open-vocabulary 3D instance segmentation with a
proposal-agnostic temporal consistency layer — Indoor (ScanNet) + Outdoor
(nuScenes)".

**Do not include:** numeric tables, code, photographs, dense paragraphs.
Keep all text short labels only.

---

## 3. Per-element fact sheet (so the figure stays accurate when edited)

| Element | Type | Color | Exact label | Metric |
|---------|------|-------|-------------|--------|
| ScanNet | input | neutral | "ScanNet (RGB-D)" | — |
| nuScenes | input | neutral | "nuScenes (6 cam + LiDAR)" | — |
| Mask3D | upstream | gray | "Mask3D — 3D proposals (A)" | — |
| YOLO-World | upstream | gray | "YOLO-World — 2D labels (B)" | — |
| γ CenterPoint | contribution | teal | "γ CenterPoint adapter (replaces A)" | — |
| γ native | contribution | teal | "γ CenterPoint — native class head" | — |
| YOLO bypass (Mode 2) | bypass | red dashed ✗ | "YOLO-World — bypassed" | — |
| M11 / M12 | contribution+ | green | "M11/M12 registration — Phase-1 stabilizer" | primary |
| M21 / M22 | neg. ablation | faded hatch | "M21/M22 label (negative)" | dimmed |
| M31 / M32 | neg. ablation | faded hatch | "M31/M32 merge (negative)" | dimmed |
| Temporal layer | contribution | teal block | "Temporal Consistency Layer" | — |
| StreamingNuScenesEvaluator | contribution | teal | "StreamingNuScenesEvaluator" | — |
| Indoor output | output | neutral | "Confirmed instance map" | lsc · ttc · Mean AP |
| Mode 1 output | output | neutral | "open-vocab capability" | mAP 0.0526 |
| Mode 2 output | output | neutral | "detection" | mAP 0.3407 |

Honesty guardrails baked in: (1) γ is a contribution (teal), Mask3D/YOLO are
upstream (gray); (2) only M11/M12 are highlighted positive; (3)
M21/M22/M31/M32 are dimmed negatives; (4) Mode 1 and Mode 2 share the γ
source; (5) Outdoor temporal = M11/M12 only.
