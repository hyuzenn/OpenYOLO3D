"""V1–V4 sanity check for nuScenes v1.0-trainval.

Risk-removal step before Tier-2 (n=100) expansion. Verifies that the
dataloader and adapter, both validated on v1.0-mini, also work on the
larger trainval split. NO downloads, NO auto-fixes, NO dataloader edits.

Outputs (relative to repo root):
    results/sanity_trainval/report.md
    results/sanity_trainval/tested_samples.json

Acceptance: 4/4 verifications must pass. Final verdict line:
    GO          — trainval ready for Tier-2
    FIX_NEEDED  — issues found that need a code change
    BLOCKED     — environmental issue (data missing, etc.)
"""

import argparse
import json
import os
import os.path as osp
import random
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataloaders.nuscenes_loader import NuScenesLoader
from adapters.nuscenes_to_openyolo3d import adapt_sample, _DEFAULT_NUSC_CAMERAS


OUT_DIR = osp.join(REPO_ROOT, "results", "sanity_trainval")
SEED = 42
N_SAMPLES = 5
TIME_BUDGET_S = 600


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class CheckResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.notes = []
        self.details = {}

    def add(self, msg):
        self.notes.append(msg)

    def fail(self, msg):
        self.passed = False
        self.notes.append(f"✗ {msg}")

    def ok(self, msg):
        self.passed = True
        self.notes.append(f"✓ {msg}")


def _summarize_frame(frame):
    """Compact, JSON-safe shape/dtype summary for cross-split comparison."""
    return {
        "keys": sorted(frame.keys()),
        "point_cloud_shape": list(frame["point_cloud"].shape),
        "point_cloud_dtype": str(frame["point_cloud"].dtype),
        "ego_pose_shape": list(frame["ego_pose"].shape),
        "ego_pose_dtype": str(frame["ego_pose"].dtype),
        "cameras": sorted(frame["images"].keys()),
        "image_shapes": {c: list(frame["images"][c].shape) for c in frame["images"]},
        "image_dtypes": {c: str(frame["images"][c].dtype) for c in frame["images"]},
        "intrinsics_shape": list(next(iter(frame["intrinsics"].values())).shape),
        "cam_to_ego_shape": list(next(iter(frame["cam_to_ego"].values())).shape),
        "n_gt_boxes": len(frame["gt_boxes"]),
        "timestamp_type": type(frame["timestamp"]).__name__,
        "sample_token_type": type(frame["sample_token"]).__name__,
    }


# ---------------------------------------------------------------------------
# V1: NuScenes init
# ---------------------------------------------------------------------------

def v1_nuscenes_init(trainval_loader):
    r = CheckResult("V1: NuScenes(version='v1.0-trainval') init")
    n = len(trainval_loader)
    r.details["n_samples"] = n
    if n >= 5000:
        r.ok(f"loader holds {n} samples (≥ 5000)")
    else:
        r.fail(f"only {n} samples, expected ≥ 5000 for trainval")
    return r


# ---------------------------------------------------------------------------
# V2: 5 random samples — 8-key dict structure
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"point_cloud", "images", "intrinsics", "cam_to_ego",
                 "ego_pose", "timestamp", "sample_token", "gt_boxes"}


def v2_eight_key_dict(trainval_loader, mini_summary, picks):
    r = CheckResult("V2: 8-key dict on 5 random samples")
    diffs_vs_mini = []
    per_sample = []
    all_ok = True

    for tok in picks:
        try:
            frame = trainval_loader._load(tok)
        except Exception as e:
            r.fail(f"_load({tok[:8]}…) crashed: {e}")
            return r

        s = _summarize_frame(frame)
        s["sample_token"] = tok
        per_sample.append(s)

        missing = REQUIRED_KEYS - set(frame.keys())
        if missing:
            r.fail(f"sample {tok[:8]}… missing keys: {missing}")
            all_ok = False
            continue

        if frame["point_cloud"].shape[0] < 10000:
            r.fail(f"sample {tok[:8]}… point_cloud has only {frame['point_cloud'].shape[0]} points (<10000)")
            all_ok = False

        if set(frame["images"].keys()) != set(_DEFAULT_NUSC_CAMERAS):
            r.fail(f"sample {tok[:8]}… cameras mismatch: {set(frame['images'].keys())}")
            all_ok = False

        if len(frame["gt_boxes"]) < 1:
            r.fail(f"sample {tok[:8]}… empty gt_boxes")
            all_ok = False

        # Compare structural fields with mini (dtype/shape only, NOT counts).
        for field in ("point_cloud_dtype", "ego_pose_dtype", "ego_pose_shape",
                      "intrinsics_shape", "cam_to_ego_shape", "cameras",
                      "timestamp_type", "sample_token_type"):
            if s[field] != mini_summary[field]:
                diffs_vs_mini.append({"token": tok, "field": field,
                                       "trainval": s[field], "mini": mini_summary[field]})

        # image dtype must match mini per-camera.
        for c in s["image_dtypes"]:
            if s["image_dtypes"][c] != mini_summary["image_dtypes"].get(c):
                diffs_vs_mini.append({"token": tok, "field": f"image_dtype[{c}]",
                                       "trainval": s["image_dtypes"][c],
                                       "mini": mini_summary["image_dtypes"].get(c)})

    r.details["per_sample"] = per_sample
    r.details["diffs_vs_mini"] = diffs_vs_mini
    if diffs_vs_mini:
        r.add(f"  {len(diffs_vs_mini)} structural differences vs mini (see details)")
    if all_ok and not diffs_vs_mini:
        r.ok(f"all {len(picks)} samples valid; structurally identical to mini")
    elif all_ok:
        r.ok(f"all {len(picks)} samples valid (with {len(diffs_vs_mini)} structural diffs)")
    return r


# ---------------------------------------------------------------------------
# V3: adapter cameras="all" + cameras=None backward-compat
# ---------------------------------------------------------------------------

def v3_adapter_paths(trainval_loader, picks, tmp_root):
    r = CheckResult("V3: adapter cameras='all' + cameras=None")
    per_sample = []
    all_ok = True

    for tok in picks:
        try:
            frame = trainval_loader._load(tok)
        except Exception as e:
            r.fail(f"_load({tok[:8]}…) crashed: {e}")
            return r

        # multi-cam path
        multi_dir = osp.join(tmp_root, f"multi_{tok[:8]}")
        try:
            multi_stats = adapt_sample(frame, multi_dir, cameras="all")
        except Exception as e:
            r.fail(f"adapt_sample(cameras='all') crashed on {tok[:8]}…: {e}\n{traceback.format_exc(limit=2)}")
            return r

        if multi_stats.get("mode") != "multi_camera":
            r.fail(f"multi-cam stats missing 'mode'='multi_camera'")
            all_ok = False

        per_cam = multi_stats.get("per_cam", {})
        if set(per_cam.keys()) != set(_DEFAULT_NUSC_CAMERAS):
            r.fail(f"multi-cam path missing cams: {set(_DEFAULT_NUSC_CAMERAS) - set(per_cam.keys())}")
            all_ok = False

        cams_with_zero_depth = []
        for cam, st in per_cam.items():
            for f in (st["image_path"], st["depth_path"], st["pose_path"], st["intrinsics_path"]):
                if not osp.exists(f):
                    r.fail(f"missing file {f}")
                    all_ok = False
            if st["n_depth_pixels_filled"] < 1:
                cams_with_zero_depth.append(cam)
        if cams_with_zero_depth:
            r.fail(f"cams with 0 depth pixels: {cams_with_zero_depth}")
            all_ok = False

        # backward-compat: cameras=None must still produce single-cam layout
        single_dir = osp.join(tmp_root, f"single_{tok[:8]}")
        try:
            single_stats = adapt_sample(frame, single_dir, cameras=None)
        except Exception as e:
            r.fail(f"adapt_sample(cameras=None) crashed on {tok[:8]}…: {e}")
            return r

        for f in ("color/0.jpg", "depth/0.png", "poses/0.txt", "intrinsics.txt", "lidar.ply"):
            if not osp.exists(osp.join(single_dir, f)):
                r.fail(f"single-cam layout missing {f}")
                all_ok = False

        per_sample.append({
            "token": tok,
            "multi_per_cam_filled": {c: st["n_depth_pixels_filled"] for c, st in per_cam.items()},
            "single_filled": single_stats["n_depth_pixels_filled"],
        })

    r.details["per_sample"] = per_sample
    if all_ok:
        r.ok(f"adapter both paths (all/None) work on all {len(picks)} samples")
    return r


# ---------------------------------------------------------------------------
# V4: deterministic loading
# ---------------------------------------------------------------------------

def v4_deterministic(trainval_loader, picks):
    r = CheckResult("V4: same sample_token → bytewise-identical reload")
    mismatches = []
    for tok in picks[:3]:  # subset to save time
        a = trainval_loader._load(tok)
        b = trainval_loader._load(tok)
        pc_eq = (a["point_cloud"].shape == b["point_cloud"].shape
                 and (a["point_cloud"] == b["point_cloud"]).all())
        gtA = [(g["category"], g["instance_token"], tuple(g["translation"])) for g in a["gt_boxes"]]
        gtB = [(g["category"], g["instance_token"], tuple(g["translation"])) for g in b["gt_boxes"]]
        gt_eq = gtA == gtB
        if not pc_eq:
            mismatches.append({"token": tok, "field": "point_cloud"})
        if not gt_eq:
            mismatches.append({"token": tok, "field": "gt_boxes"})
    r.details["checked_tokens"] = picks[:3]
    r.details["mismatches"] = mismatches
    if not mismatches:
        r.ok(f"all {len(picks[:3])} tokens reload deterministically")
    else:
        r.fail(f"{len(mismatches)} mismatches across reloads")
    return r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def write_report(checks, picks, n_trainval, n_mini, elapsed_s, verdict):
    os.makedirs(OUT_DIR, exist_ok=True)
    report_path = osp.join(OUT_DIR, "report.md")
    samples_path = osp.join(OUT_DIR, "tested_samples.json")

    with open(report_path, "w") as f:
        f.write("# nuScenes v1.0-trainval sanity check\n\n")
        f.write(f"- elapsed: {elapsed_s:.1f}s (budget {TIME_BUDGET_S}s)\n")
        f.write(f"- branch: feature/diagnosis\n")
        f.write(f"- env: openyolo3d-dev\n")
        f.write(f"- trainval samples: {n_trainval}\n")
        f.write(f"- mini samples (reference): {n_mini}\n")
        f.write(f"- random pick seed: {SEED}, picked {N_SAMPLES} tokens\n\n")

        passed_n = sum(c.passed for c in checks)
        f.write(f"## Summary: {passed_n}/{len(checks)} passed\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")

        for c in checks:
            mark = "✓" if c.passed else "✗"
            f.write(f"### {mark} {c.name}\n\n")
            for n in c.notes:
                f.write(f"- {n}\n")
            if c.name.startswith("V2") and c.details.get("diffs_vs_mini"):
                f.write("\nStructural differences vs mini (each row = one diff):\n\n")
                for d in c.details["diffs_vs_mini"][:20]:
                    f.write(f"- `{d['field']}` token=`{d['token'][:8]}…`: trainval=`{d['trainval']}` mini=`{d['mini']}`\n")
            f.write("\n")

    with open(samples_path, "w") as f:
        json.dump({
            "seed": SEED,
            "n_trainval": n_trainval,
            "tokens": picks,
            "checks": [
                {"name": c.name, "passed": c.passed, "notes": c.notes, "details_keys": list(c.details.keys())}
                for c in checks
            ],
        }, f, indent=2)

    return report_path, samples_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainval-config", default="configs/nuscenes_trainval.yaml")
    parser.add_argument("--mini-config", default="configs/nuscenes_baseline.yaml")
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== nuScenes v1.0-trainval sanity check (budget {TIME_BUDGET_S}s) ===")

    print("\n[V1] initializing trainval loader ... (NuScenes JSON parse)")
    t0 = time.time()
    try:
        trainval_loader = NuScenesLoader(config_path=args.trainval_config)
    except Exception as e:
        print(f"✗ BLOCKED: {e}")
        traceback.print_exc()
        return 1
    print(f"  trainval init in {time.time() - t0:.1f}s, {len(trainval_loader)} samples")

    print("\n  loading mini for structural reference ...")
    t0 = time.time()
    mini_loader = NuScenesLoader(config_path=args.mini_config)
    mini_first = mini_loader._load(mini_loader.sample_tokens[0])
    mini_summary = _summarize_frame(mini_first)
    print(f"  mini init in {time.time() - t0:.1f}s, {len(mini_loader)} samples")

    rng = random.Random(SEED)
    picks = rng.sample(trainval_loader.sample_tokens, N_SAMPLES)
    print(f"\n  picked {N_SAMPLES} random samples (seed={SEED})")
    for tok in picks:
        print(f"    {tok}")

    checks = []

    print("\n[V1] check len ...")
    checks.append(v1_nuscenes_init(trainval_loader))

    print("\n[V2] 8-key dict on picked samples ...")
    checks.append(v2_eight_key_dict(trainval_loader, mini_summary, picks))

    print("\n[V3] adapter all + None paths ...")
    with tempfile.TemporaryDirectory(prefix="trainval_sanity_") as tmp_root:
        checks.append(v3_adapter_paths(trainval_loader, picks, tmp_root))

    print("\n[V4] deterministic reload ...")
    checks.append(v4_deterministic(trainval_loader, picks))

    elapsed = time.time() - t_start
    if elapsed > TIME_BUDGET_S:
        print(f"\n✗ TIME BUDGET EXCEEDED: {elapsed:.1f}s > {TIME_BUDGET_S}s")

    n_passed = sum(c.passed for c in checks)
    if n_passed == len(checks):
        verdict = "GO"
    elif any("BLOCKED" in n or "missing" in n.lower() for c in checks for n in c.notes):
        verdict = "BLOCKED"
    else:
        verdict = "FIX_NEEDED"

    report, samples = write_report(checks, picks, len(trainval_loader),
                                    len(mini_loader), elapsed, verdict)

    print(f"\n=== {n_passed}/{len(checks)} passed — verdict: {verdict} ===")
    print(f"  report: {report}")
    print(f"  tokens: {samples}")
    print(f"  total : {elapsed:.1f}s")
    return 0 if n_passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
