# STATE.md — Frozen project state (SemWorld-3D / OpenYOLO3D)

> Purpose: single source of established numbers and decisions, so prompts never
> need to restate them. **Read this before any experiment work.**
> Every number below is copied from a file in `results/` (source cited).
> Update ONLY when a full-scale run changes a frozen value. Results never go in CLAUDE.md.
> Last updated: 2026-07-09.

## 1. Frozen anchors — Outdoor (nuScenes val-150, γ gravity-corrected CenterPoint cache)

Source: `results/nuscenes_final/paper_table.md` (final paper table, cache replay, job 100932).

| Method | mAP | NDS | OV-TCS_C | TrackLen | Frag | CSR |
|---|---|---|---|---|---|---|
| Baseline (native, per-frame) | 0.3407 | 0.3145 | — (degenerate: class-aware assoc ⇒ CSR≡0) | — | — | — |
| Ego Association | 0.3407 | 0.3145 | 0.1356 | 3.013 | 10.70 | 0.7182 |
| Global Association | 0.3407 | 0.3145 | **0.1685** | 3.244 | **4.488** | 0.5797 |
| Class-aware Label Fusion (G) | **0.3420** | **0.3159** | 0.1356 | 3.013 | 10.70 | 0.7182 |
| M31 (IoU merge) | 0.3407 | 0.3145 | =ego | =ego | =ego | =ego |
| M32 (Hungarian, 1.0 m) | 0.3198 | 0.3028 | =ego | =ego | =ego | =ego |

- γ-fixed native anchor 0.3407; fusion A-anchor reproduces 0.3408 (same thing, rounding).
- Global assoc under phase1 (M11 ≥3-frame gate): ego 0.1425 → global 0.2569 (**+0.1144**); both < baseline (native-label temporal layer is net-lossy). Source: `results/experiment_tracker.md` tail.
- Bottleneck decomposition (frozen, `2026-06-13_outdoor_proposal_ceiling_v01`): **proposal generation dominates** — 19.8% of GT has no proposal within 4 m; 100% miss >80 m; IoU3D recall@0.5 = 0.382 (z-fix raised it from 0.0053). Oracle score 0.3749, dedup oracle 0.5477 (geometric ceiling). Calibration path is dead (rank-invariance).
- Label Fusion G = gated VRU override (allowlist bicycle/motorcycle, τ_iou/τ_score): bicycle ≈+16%, zero regression. Framing: per-class VRU correction, not a global-mAP method. Source: `2026-06-24_outdoor_labelfusion_writeup_v01/notes.md`.

## 2. Frozen anchors — Indoor (ScanNet200 val, 312 scenes)

Source: `results/scannet_val312_final/summary.md` (final, n=6202 matched instances, GATE PASS).

| Metric | Value |
|---|---|
| AP / AP50 / AP25 (mask-IoU, label-matched) | **0.3797 / 0.3645 / 0.3949** |
| mean OV-TCS_C | **0.9001** |
| mean Track Length | 45.16 frames |
| mean CSR | 0.0665 |
| mean GT Fragmentation | 1.024 |

Streaming ablation reference (312 scenes, `results/experiment_tracker.md`):
baseline streaming AP 0.1956, lsc 23,385; M11 (N=3) lsc −27% at AP −0.0002;
M12-fixed lsc −29% / ttc −8% at AP −0.0006; M22+M32 naive cascade AP 0.0998 (dead);
M32 data-driven fix (merge 0.006, dist 0.5 m, sem 0.95) is what recovers AP indoor.
Offline (non-streaming) baseline AP 0.2470.

## 3. OV-TCS metric — final status (paper core)

- **Final formulation: OV-TCS_C = L_norm × (1 − CSR).** Product justified (two-axis argument: flicker → 1−CSR, fragmentation → L_norm); §2c RESOLVED — do not reopen. Source: `results/2026-06-26_ablation_ovtcs_formulation_v01`.
- Validated as downstream surrogate: phase1-mAP corr Pearson 0.948 / Spearman 0.988; survives track-length control (partial r 0.818, p=0.007).
- **Dead as an EMA control signal** (const-scale control beat it; OV-TCS-specific ΔAP negative).

## 4. Closed lines — do not restart without new evidence

| Line | Verdict | Key number | Source |
|---|---|---|---|
| M22 EMA (weighted, τ_skip, k-sweep, OV-TCS-aware) | CLOSED — OFF > k=0.335 > k=1.0 | 50-scene gate FAIL | `2026-06-23_scannet_m22_off_vs_ema_50_v01` |
| Hybrid proposal (CP geometry + YOLO ROI label) | CLOSED for closed-set mAP | 0.0667 vs 0.3407; label agreement 18.9%, ~30× ROI overhead | `2026-06-22_outdoor_hybrid_eval_v01` |
| Score calibration / temperature scaling | CLOSED — rank-invariant | oracle +0.034 only | `2026-06-23_outdoor_calibration_diagnosis_v01` |
| Blind relabel (B) / global-gate (F) | CLOSED | B −0.080 mAP; F 50-scene FAIL | labelfusion writeup |
| ScanNet train-1201 full run | BLOCKED + invalid | no .sens on disk (~1.8 TB); ckpt in-sample on train | `scannet_val312_final/summary.md` scope note |

## 5. Open items

1. Paper writing (OV-TCS): storyline `docs/ovtcs_paper_storyline.md`; drafts in worktree `ovtcs-paper-draft`; final tables = `results/nuscenes_final/paper_table.md` + `results/scannet_val312_final/summary.md`.
2. train1201 fetch/extract PBS chain prepared — submit only after storage verification; in-sample caveat stands.
3. Outdoor secondary track: localization refinement (max realizable +0.05–0.10, HIGH risk) — parked.

## 6. Conventions (pointers, not copies)

- Env/PBS header/evaluator CLI: see `CLAUDE.md` (never run heavy Python on util node).
- Run dirs: `results/<YYYY-MM-DD>_<experiment>_v<NN>/` with `notes.md`; append full-scale rows to `results/experiment_tracker.md`.
- Experiment protocol: 2-arm minimal decisive comparison, smoke → gate → full; explicit stop condition.
