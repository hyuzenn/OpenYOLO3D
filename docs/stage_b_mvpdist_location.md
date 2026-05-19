# Stage B.1 — MVPDist hook location in OpenYOLO3D

## Naming note

The OpenYOLO3D codebase does **not** contain a class or module literally named
`MVPDist` / `mvpdist` / `prompt_distribution`. The Multi-View Prompt
Distribution mechanism is implemented **inline** inside the main wrapper class
in `utils/__init__.py`.

`grep -rn "MVPDist\|mvpdist\|prompt_distribution" --include='*.py'` returns
**zero matches**. The matching code is located by name only: the structure
that aggregates per-frame 2D detections into per-instance 3D class
distributions is the body of `OpenYolo3D.label_3d_masks_from_label_maps`.

## Hook point

- File: `utils/__init__.py`
- Class: `OpenYolo3D`
- Method: `label_3d_masks_from_label_maps`
- Lines: 158 – 262

The method takes the 3D mask proposals + per-frame 2D bounding boxes and
returns `(predicted_masks, pred_classes, pred_scores)` — the final per-instance
class assignments. This is the point at which METHOD_21 replaces the original
voting rule.

## Existing voting logic (what we replace)

For each 3D mask proposal `mask_id`:

1. `representitive_frame_ids = where(visibility_matrix[mask_id])` — top-k most
   visible frames for this instance (`topk=40` from config).
2. For each representative frame:
   - Project mesh points onto the frame to get pixel `(x, y)` coords for the
     mask's vertices.
   - Look up `label_maps[frame, y, x]` to get a per-pixel label_id list.
   - Append all pixel-labels to `labels_distribution`.
   - Compute IoU between the projected instance AABB and the frame's 2D bboxes
     (collected in `iou_vals`).
3. `labels_distribution = np.concatenate(...)` — a flat array of pixel-labels
   pooled across all representative frames with **uniform per-pixel weight**.
4. Final class assignment, two parallel branches:
   - **Mode branch (line 218)**: `class_label = mode(labels_distribution)`.
     Used when `topk_per_image == -1` or `is_gt`.
   - **Distribution branch (lines 207–216, 243–260)**: build a per-class count
     histogram, normalize by max → `distribution`. After the per-instance loop,
     stack into `(n_inst, num_classes)` and take global top-k. Used when
     `topk_per_image != -1` and `not is_gt` — **this is the active branch for
     ScanNet200 evaluation** (`topk_per_image=600` per
     `pretrained/config_scannet200.yaml`).

In both branches, every pixel-label contributes weight 1. Frames that see more
mask vertices pull harder simply by contributing more pixel-labels — there is
no explicit distance / center / confidence weighting.

## Input / output interface for METHOD_21

For the WeightedVoting wrapper, the natural inputs assembled per
`(instance, frame)` pair are:

| Field | Source in existing code |
|---|---|
| `pixel_labels` (vector of label_ids, -1 = no box) | `selected_labels` at line 201 |
| `camera_pos` (3D, world frame) | `world2cam.poses[frame_id]` → `pose[:3, 3]` |
| `instance_centroid` (3D, world frame) | `vertex_coords[mask].mean(0)` (vertex_coords loaded from `world2cam.mesh` ply; cached on `world2cam` to avoid re-loading) |
| `bbox_2d_center` (2D pixel) | `((x_l+x_r)/2, (y_t+y_b)/2)` from `instance_x_y_coords` AABB at line 196 |
| `image_size` (W, H pixel) | `(world2cam.width, world2cam.height)` from `world2cam.depth_resolution` |
| `confidence` (scalar) | `iou_values.max().item()` from line 199 (per-frame projected-bbox vs detected-bbox IoU). Falls back to 1.0 when no 2D bbox available. |

Output (matches existing return contract):

- `predicted_masks` — (n_vertices, K) bool, possibly re-indexed by top-k.
- `pred_classes` — (K,) long.
- `pred_scores` — (K,) float, weighted-distribution scores after top-k.

## Hook strategy (no core edits)

We monkey-patch `OpenYolo3D.label_3d_masks_from_label_maps` from
`method_scannet/hooks.py` at runtime. The original method is preserved as
`OpenYolo3D._original_label_3d_masks_from_label_maps` so `uninstall()` restores
the baseline. No file in `utils/`, `models/`, `evaluate/`, or `run_evaluation.py`
is modified.

METHOD_31 (3D IoU merging) is similarly applied as a runtime patch on
`OpenYolo3D.label_3d_masks_from_2d_bboxes` (which packages the per-scene
return tuple) — the patched version calls the original then runs `IoUMerger`
on `(masks, classes, scores)` before returning.
