#!/usr/bin/env python3
"""Minimum-viable Hybrid-Proposal feasibility validation (outdoor / nuScenes).

Question: can CenterPoint be used purely as a GEOMETRY proposal generator, with
the open-vocabulary class assigned by YOLO-World on the projected 2D ROI?

Pipeline per box (CenterPoint class is DISCARDED):
    bbox_lidar -> project to 6 cams -> best-camera 2D ROI -> crop -> YOLO-World
    -> open-vocab label.

Measures (feasibility only, no accuracy optimisation):
  1. ROI projection success rate (fraction of geometry proposals that yield a
     valid in-image ROI), with per-camera and per-distance breakdown.
  2. Runtime overhead: per-box projection time, crop time, per-ROI YOLO-World
     time, vs the cost of running YOLO-World once on the 6 full images.
  3. Label distribution of the assigned open-vocab labels.
  4. Example visualizations (projected box + ROI + assigned label).

Inputs reused (no recompute of proposals):
  - gamma cache: results/outdoor_native_temporal_cpcache_thr000_single_gravity/<token>.pkl
    (list of {cls_idx, cls_name, score, bbox_lidar, centroid_ego}; gravity-corrected z)
  - calibration / GT: nuScenes v1.0-trainval metadata via the devkit (data/nuscenes)
  - camera pixels: local WebDataset shards /home/rintern16/nuscenes_shards/*.tar
    (members "{token}.cam_front.jpg" etc.), scene->shard from manifest.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import tarfile
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "models/YOLO-World"))  # so cfg custom_imports=['yolo_world'] resolves

from proposal.hybrid_proposal import project_box, box_corners_lidar, _homog  # noqa: E402

CAM_MEMBER = {
    "CAM_FRONT": "cam_front.jpg",
    "CAM_FRONT_LEFT": "cam_front_left.jpg",
    "CAM_FRONT_RIGHT": "cam_front_right.jpg",
    "CAM_BACK": "cam_back.jpg",
    "CAM_BACK_LEFT": "cam_back_left.jpg",
    "CAM_BACK_RIGHT": "cam_back_right.jpg",
}
CAMERAS = list(CAM_MEMBER.keys())
DIST_BINS = [(0, 15), (15, 30), (30, 50), (50, 80), (80, 1e9)]


def dist_bin(r):
    for lo, hi in DIST_BINS:
        if lo <= r < hi:
            return f"{lo}-{hi if hi < 1e9 else 'inf'}m"
    return "inf"


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def build_scene_shard(manifest_path):
    m = {}
    with open(manifest_path) as f:
        for line in f:
            d = json.loads(line)
            m[d["scene_name"]] = d["shard"]
    return m


def load_images_for_tokens(shard_path, tokens, cv2):
    """Extract {token: {cam: np.ndarray BGR}} for the given tokens from one tar."""
    want = {}
    for tok in tokens:
        for cam, suff in CAM_MEMBER.items():
            want[f"{tok}.{suff}"] = (tok, cam)
    out = defaultdict(dict)
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if member.name in want:
                tok, cam = want[member.name]
                raw = tf.extractfile(member).read()
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                out[tok][cam] = arr
    return out


# ---------------------------------------------------------------------------
# YOLO-World ROI labeler (reuses the repo's Network_2D runner; per-crop inference)
# ---------------------------------------------------------------------------
class YoloRoiLabeler:
    def __init__(self, config_path, pretrained_path, vocab, th, nms, use_amp):
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmengine.runner import Runner
        from mmyolo.registry import RUNNERS
        self.torch = torch
        from torchvision.ops import nms as tvnms
        from mmengine.runner.amp import autocast
        self._nms = tvnms
        self._autocast = autocast
        self.vocab = list(vocab)
        self.texts = [[t] for t in self.vocab] + [[" "]]
        self.th, self.nms_iou, self.use_amp = th, nms, use_amp

        cfg = Config.fromfile(os.path.join(ROOT, config_path))
        cfg.work_dir = os.path.join(ROOT, "models/YOLO-World/yolo_world/work_dirs",
                                    os.path.splitext(config_path)[0].split("/")[-1])
        cfg.load_from = os.path.join(ROOT, pretrained_path)
        self.runner = (Runner.from_cfg(cfg) if "runner_type" not in cfg
                       else RUNNERS.build(cfg))
        self.runner.call_hook("before_run")
        self.runner.load_or_resume()
        self.runner.pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
        self.runner.model.eval()

    def label_crop_path(self, path):
        """Run YOLO-World on a saved crop. Return (label_name, score, n_det)."""
        data_info = self.runner.pipeline(dict(img_id=0, img_path=path, texts=self.texts))
        data_batch = dict(inputs=self.torch.stack([data_info["inputs"]]),
                          data_samples=[data_info["data_samples"]])
        with self._autocast(enabled=self.use_amp), self.torch.no_grad():
            output = self.runner.model.test_step(data_batch)
        pi = output[0].pred_instances
        if len(pi.scores) == 0:
            return ("__background__", 0.0, 0)
        keep = self._nms(pi.bboxes, pi.scores, iou_threshold=self.nms_iou)
        pi = pi[keep]
        n_det = int((pi.scores.float() > self.th).sum())
        # the dominant label for this ROI = highest-scoring detection over the vocab
        j = int(pi.scores.float().argmax())
        score = float(pi.scores[j])
        lab = int(pi.labels[j])
        name = self.vocab[lab] if 0 <= lab < len(self.vocab) else "__background__"
        if score < self.th:
            return ("__background__", score, n_det)
        return (name, score, n_det)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="results/outdoor_native_temporal_cpcache_thr000_single_gravity")
    ap.add_argument("--shards", default="/home/rintern16/nuscenes_shards")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--score-thr", type=float, default=0.10,
                    help="CenterPoint objectness floor (score is kept; only CLASS is discarded)")
    ap.add_argument("--max-rois-per-sample", type=int, default=0, help="0 = no cap")
    ap.add_argument("--n-viz", type=int, default=6)
    ap.add_argument("--no-yolo", action="store_true", help="geometry/projection only (no GPU)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cv2
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "outputs"), exist_ok=True)
    tmp_crop = os.path.join(args.out, "_tmp_crop.jpg")

    # ---- devkit + scene->shard ----
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion
    nusc = NuScenes(version=args.version, dataroot=os.path.join(ROOT, args.dataroot), verbose=False)
    scene_shard = build_scene_shard(os.path.join(args.shards, "manifest.jsonl"))
    sample2scene = {s["token"]: s["scene_token"] for s in nusc.sample}
    scene_name = {s["token"]: s["name"] for s in nusc.scene}

    def q2m(rot, trans):
        return transform_matrix(translation=trans, rotation=Quaternion(rot))

    # ---- pick samples from the gamma cache that resolve to a shard ----
    cache_dir = os.path.join(ROOT, args.cache)
    all_tokens = sorted(x[:-4] for x in os.listdir(cache_dir) if x.endswith(".pkl"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_tokens)
    chosen = []
    for tok in all_tokens:
        sc = sample2scene.get(tok)
        if sc is None:
            continue
        nm = scene_name.get(sc)
        if nm in scene_shard:
            chosen.append(tok)
        if len(chosen) >= args.n_samples:
            break

    # group by shard to minimise tar scans
    by_shard = defaultdict(list)
    for tok in chosen:
        by_shard[scene_shard[scene_name[sample2scene[tok]]]].append(tok)

    labeler = None
    if not args.no_yolo:
        labeler = YoloRoiLabeler(
            config_path="pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
            pretrained_path="pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth",
            vocab=["car", "truck", "bus", "trailer", "construction vehicle", "pedestrian",
                   "motorcycle", "bicycle", "traffic cone", "barrier",
                   "tree", "building", "pole", "traffic sign", "traffic light"],
            th=0.05, nms=0.5, use_amp=False)

    # ---- accumulators ----
    REC = {
        "n_samples": 0, "n_boxes": 0, "n_success": 0,
        "by_cam": Counter(), "by_dist": defaultdict(lambda: [0, 0]),  # bin -> [success, total]
        "corners_in_front_hist": Counter(),
        "labels": Counter(), "label_score_sum": defaultdict(float),
        "n_labeled": 0, "n_background": 0,
        "t_project": [], "t_crop": [], "t_yolo": [], "t_yolo_fullimg": [],
        "centerpoint_vs_yolo_agree": 0, "centerpoint_vs_yolo_total": 0,
    }
    viz_done = 0
    per_box_rows = []

    for shard, toks in by_shard.items():
        shard_path = os.path.join(args.shards, shard)
        imgs_by_tok = load_images_for_tokens(shard_path, toks, cv2)
        for tok in toks:
            imgs = imgs_by_tok.get(tok, {})
            if len(imgs) < 6:
                continue
            sample = nusc.get("sample", tok)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
            T_l2e = q2m(lidar_cs["rotation"], lidar_cs["translation"])
            intr, c2e, hw = {}, {}, {}
            for cam in CAMERAS:
                cam_sd = nusc.get("sample_data", sample["data"][cam])
                cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
                intr[cam] = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
                c2e[cam] = q2m(cam_cs["rotation"], cam_cs["translation"])
                hh, ww = imgs[cam].shape[:2]
                hw[cam] = (hh, ww)

            props = pickle.load(open(os.path.join(cache_dir, f"{tok}.pkl"), "rb"))
            props = [p for p in props if p["score"] >= args.score_thr]
            if args.max_rois_per_sample > 0:
                props = props[:args.max_rois_per_sample]

            # reference cost: YOLO-World once on each of the 6 full images
            if labeler is not None:
                t0 = time.perf_counter()
                for cam in CAMERAS:
                    fp = os.path.join(args.out, "_tmp_full.jpg")
                    cv2.imwrite(fp, imgs[cam])
                    labeler.label_crop_path(fp)
                REC["t_yolo_fullimg"].append(time.perf_counter() - t0)

            REC["n_samples"] += 1
            sample_viz = (viz_done < args.n_viz)
            viz_imgs = {cam: imgs[cam].copy() for cam in CAMERAS} if sample_viz else None

            for p in props:
                bbox = p["bbox_lidar"]
                r = float(np.hypot(bbox[0], bbox[1]))  # range in lidar frame
                db = dist_bin(r)
                REC["n_boxes"] += 1
                REC["by_dist"][db][1] += 1

                t0 = time.perf_counter()
                roi = project_box(bbox, T_l2e, c2e, intr, hw)
                REC["t_project"].append(time.perf_counter() - t0)
                REC["corners_in_front_hist"][roi.n_corners_in_front] += 1

                if not roi.success:
                    continue
                REC["n_success"] += 1
                REC["by_cam"][roi.cam] += 1
                REC["by_dist"][db][0] += 1

                if labeler is None:
                    continue
                x0, y0, x1, y1 = [int(round(v)) for v in roi.roi_xyxy]
                t0 = time.perf_counter()
                crop = imgs[roi.cam][y0:y1, x0:x1]
                cv2.imwrite(tmp_crop, crop)
                REC["t_crop"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                name, score, ndet = labeler.label_crop_path(tmp_crop)
                REC["t_yolo"].append(time.perf_counter() - t0)

                REC["labels"][name] += 1
                REC["label_score_sum"][name] += score
                if name == "__background__":
                    REC["n_background"] += 1
                else:
                    REC["n_labeled"] += 1
                # how often does the discarded CenterPoint class agree with YOLO?
                cp = p["cls_name"].replace("_", " ")
                if name != "__background__":
                    REC["centerpoint_vs_yolo_total"] += 1
                    if cp == name:
                        REC["centerpoint_vs_yolo_agree"] += 1

                if len(per_box_rows) < 4000:
                    per_box_rows.append({
                        "token": tok, "cam": roi.cam, "range_m": round(r, 1),
                        "centerpoint_cls": p["cls_name"], "cp_score": round(p["score"], 3),
                        "yolo_label": name, "yolo_score": round(score, 3),
                        "roi_xyxy": [round(v, 1) for v in roi.roi_xyxy],
                    })

                if viz_imgs is not None:
                    col = (0, 255, 0) if name != "__background__" else (0, 0, 255)
                    cv2.rectangle(viz_imgs[roi.cam], (x0, y0), (x1, y1), col, 2)
                    cv2.putText(viz_imgs[roi.cam], f"{name}:{score:.2f}", (x0, max(0, y0 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            if viz_imgs is not None:
                for cam in CAMERAS:
                    op = os.path.join(args.out, "outputs", f"viz_{viz_done:02d}_{tok[:8]}_{cam}.jpg")
                    cv2.imwrite(op, viz_imgs[cam])
                viz_done += 1

    # ---- summarize ----
    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": int(a.size), "mean_ms": round(float(a.mean()) * 1e3, 3),
                "median_ms": round(float(np.median(a)) * 1e3, 3),
                "p95_ms": round(float(np.percentile(a, 95)) * 1e3, 3),
                "total_s": round(float(a.sum()), 3)}

    n_boxes = max(1, REC["n_boxes"])
    out = {
        "config": {
            "n_samples": REC["n_samples"], "score_thr": args.score_thr,
            "max_rois_per_sample": args.max_rois_per_sample,
            "yolo": labeler is not None,
            "vocab": (labeler.vocab if labeler else None),
        },
        "1_roi_projection_success": {
            "n_boxes": REC["n_boxes"], "n_success": REC["n_success"],
            "success_rate": round(REC["n_success"] / n_boxes, 4),
            "by_camera": dict(REC["by_cam"]),
            "by_distance": {k: {"success": v[0], "total": v[1],
                                "rate": round(v[0] / max(1, v[1]), 4)}
                            for k, v in sorted(REC["by_dist"].items())},
            "corners_in_front_hist": dict(sorted(REC["corners_in_front_hist"].items())),
        },
        "2_runtime": {
            "projection_per_box": stats(REC["t_project"]),
            "crop_per_roi": stats(REC["t_crop"]),
            "yolo_per_roi": stats(REC["t_yolo"]),
            "yolo_full6img_per_sample": stats(REC["t_yolo_fullimg"]),
            "note": "overhead = per-ROI YOLO calls vs one 6-image pass; see notes.md",
        },
        "3_label_distribution": {
            "labels": dict(REC["labels"].most_common()),
            "n_labeled": REC["n_labeled"], "n_background": REC["n_background"],
            "label_rate": round(REC["n_labeled"] / max(1, REC["n_labeled"] + REC["n_background"]), 4),
            "mean_score_per_label": {k: round(REC["label_score_sum"][k] / REC["labels"][k], 3)
                                     for k in REC["labels"]},
            "centerpoint_yolo_agreement": (
                round(REC["centerpoint_vs_yolo_agree"] / max(1, REC["centerpoint_vs_yolo_total"]), 4)),
            "centerpoint_yolo_n": REC["centerpoint_vs_yolo_total"],
        },
        "4_visualizations": {"n_viz_samples": viz_done,
                             "dir": os.path.join(args.out, "outputs")},
    }
    json.dump(out, open(os.path.join(args.out, "analysis.json"), "w"), indent=2)
    json.dump(per_box_rows, open(os.path.join(args.out, "per_box.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    for f in (tmp_crop, os.path.join(args.out, "_tmp_full.jpg")):
        if os.path.exists(f):
            os.remove(f)


if __name__ == "__main__":
    main()
