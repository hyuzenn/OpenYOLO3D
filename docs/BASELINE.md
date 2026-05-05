\# BASELINE.md



\## 1. Baseline definition



\*\*Baseline = OpenYOLO3D, run as-is on nuScenes through a new data path.\*\*



We are not reproducing OpenYOLO3D's indoor numbers. The OpenYOLO3D paper already validated those. The baseline for this project is OpenYOLO3D's stock inference applied to nuScenes — with no method modifications — to establish:



1\. That the integration works end-to-end (dataloader → OpenYOLO3D → outputs).

2\. Where and how it fails on outdoor data.



The expected outcome of running this baseline is \*\*degraded performance compared to indoor\*\*, not a number we try to match. The failures are the input to method design.



\## 2. What OpenYOLO3D does (unchanged in baseline)



For each frame:

1\. \*\*2D open-vocab detection\*\*: YOLO-World on RGB, producing boxes + class-text scores.

2\. \*\*3D mask proposals\*\*: Class-agnostic 3D mask generator on the point cloud.

3\. \*\*Box-based 2D-to-3D lifting\*\*: 2D box → 3D mask association via depth.

4\. \*\*Per-frame open-vocab classification\*\*: Each 3D mask gets a label from the associated 2D detections.



OpenYOLO3D processes frames independently — no temporal memory, no instance tracking across frames. This is fine for static indoor reconstruction. It is the first thing that breaks on outdoor streaming input.



\## 3. nuScenes integration spec (this is what the dataloader implements)



\### Inputs the dataloader must produce per frame



\- `point\_cloud`: `(N, 4)` float array — `(x, y, z, intensity)` from the LiDAR `.bin` file. Coordinate frame: ego frame at the LiDAR sweep timestamp. Document the frame in code comments.

\- `images`: `dict\[cam\_name -> np.ndarray]` for 6 cameras (`CAM\_FRONT`, `CAM\_FRONT\_LEFT`, `CAM\_FRONT\_RIGHT`, `CAM\_BACK`, `CAM\_BACK\_LEFT`, `CAM\_BACK\_RIGHT`). Original resolution preserved.

\- `intrinsics`: `dict\[cam\_name -> (3, 3) np.ndarray]`.

\- `cam\_to\_ego`: `dict\[cam\_name -> (4, 4) np.ndarray]`. Convention: transforms a point from \*\*camera frame\*\* to \*\*ego frame\*\* (i.e., multiplying a homogeneous point in camera coordinates by this matrix yields its position in the ego frame). This matches the raw nuScenes `calibrated\_sensor` records, which store the sensor's pose in the ego frame. To project ego-frame points into the image, invert this matrix first. Document this clearly.

\- `ego\_pose`: `(4, 4) np.ndarray` — ego frame to global frame at the sample timestamp.

\- `timestamp`: int (microseconds, from the LiDAR sweep).

\- `sample\_token`: nuScenes sample token (str).

\- `gt\_boxes` (eval only): list of dicts with category, 3D box in ego frame, instance token.



\### Calibration / projection contract (this must be unit-tested)



A LiDAR point `p\_lidar` projects into camera `c` as:

```

p\_ego = T\_lidar\_to\_ego @ p\_lidar

p\_cam = T\_ego\_to\_cam\[c] @ p\_ego

p\_pix = K\[c] @ p\_cam\[:3] / p\_cam\[2]   (only if p\_cam\[2] > 0)

```

The smoke test must verify that for at least one frame, projecting the LiDAR sweep into `CAM\_FRONT` produces points that mostly fall inside image bounds and visually align with image content (sanity check via a saved overlay image).



\### Sweep handling



\- Single-sweep mode (default): use only the keyframe LiDAR sweep paired with the keyframe images.

\- Multi-sweep accumulation (configurable, off for now): accumulate ±N non-keyframe sweeps with ego-motion compensation. Implement the option but keep it disabled in baseline.



\### Split



\- Start with `v1.0-mini` for development.

\- `v1.0-trainval` support is required but not exercised yet.



\## 4. Out of scope for the baseline step



Do not implement, even if tempting:



\- Any change to OpenYOLO3D's lifter, mask predictor, detector, or classifier.

\- Temporal memory across frames.

\- Multi-view fusion logic.

\- Any "Phase 1" / "Phase 2" method module from earlier drafts.

\- Replica or ScanNet code paths.

\- Training of any kind.



If the baseline produces poor numbers on nuScenes, that is the expected and useful result. We do not "fix" it at this step.



\## 5. Diagnosis to perform after the baseline runs



Once OpenYOLO3D runs end-to-end on nuScenes via the new dataloader, the following measurements drive method design (Section 6 of CONTEXT.md):



1\. \*\*Lifting reliability\*\*: Per detected 2D box, count the number of LiDAR points falling inside the box's frustum. Plot the distribution. Measure how often median-depth lifting fails or yields outlier depth. Break down by object distance from ego.

2\. \*\*Dynamic-object failure\*\*: For tracked ground-truth instances that move between frames, measure how often the per-frame 3D output for that instance is spatially inconsistent (e.g., centroid jumps that exceed plausible motion).

3\. \*\*Multi-view inconsistency\*\*: For ground-truth instances visible in two cameras simultaneously, measure how often the per-frame outputs disagree in classification or fail to merge spatially.

4\. \*\*Real-time profile\*\*: Per-frame latency breakdown (2D detection, mask proposal, lifting, classification). Identify the dominant cost on nuScenes input.



The output of this diagnosis is a short measurement report. That report — not the existing method drafts — is what authorizes the next stage of method design.



\## 6. Success criteria for the baseline step



The baseline step is complete when all of the following hold:



\- Dataloader passes its unit tests (calibration projection, image / LiDAR sync, all 6 cameras load).

\- OpenYOLO3D runs without crashing on at least 10 consecutive nuScenes mini frames.

\- A measurement report covering the four diagnoses in Section 5 is checked in under `results/baseline\_diagnosis.md`.

\- No core OpenYOLO3D file has been modified (verified by `git diff` against the OpenYOLO3D upstream commit).



Only after these are satisfied do we start writing METHOD.md.

