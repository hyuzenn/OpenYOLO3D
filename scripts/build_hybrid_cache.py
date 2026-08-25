#!/usr/bin/env python3
"""Build the Hybrid-Proposal v2 cache (outdoor / nuScenes val-150).

Hybrid Proposal v2 = CenterPoint as a PURE GEOMETRY proposal generator:
    * keep every CenterPoint 3D box (geometry preserved verbatim),
    * discard the CenterPoint class,
    * run YOLO-World ONCE per camera image (not per-ROI — the v1 fan-out fix),
    * project each 3D box into image space (all 6 cams),
    * match the projected ROI to the YOLO detections by 2D IoU / containment,
    * transfer the open-vocabulary label + score from the matched detection.

Vocabulary is RESTRICTED to the 10 evaluated nuScenes classes (configs/
openyolo3d_nuscenes_nusc10.yaml text_prompts) so the relabel is directly
comparable to the native CenterPoint class on the scored mAP.

Output: one ``<token>.hybrid.pkl`` per sample, a list of proposal dicts with the
SAME geometry contract as the gamma cache
    {cls_name, cls_idx, score, bbox_lidar, centroid_ego}
but cls_name/cls_idx/score taken from the matched YOLO detection (or
cls_name="__background__"/cls_idx=-1/score=0 for an unmatched box). Each dict
also carries ``score_cp`` (the original CenterPoint objectness) and ``match_iou``
so the downstream eval can run a label-only ablation (YOLO label + CP score)
without rebuilding the cache.

The native evaluator reads these via ``--proposal-source hybrid`` (cache-only),
so mAP / NDS / per-class AP / temporal / OV-TCS are computed by the SAME code
path as the gamma baseline — only the proposal cls/score differ. Geometry is
identical to the gamma gravity cache, by construction.

Inputs reused (no recompute of geometry):
  - gamma gravity cache: results/outdoor_native_temporal_cpcache_thr000_single_gravity
  - calibration / scenes: nuScenes devkit (data/nuscenes)
  - camera pixels: local WebDataset shards (/home/rintern16/nuscenes_shards/*.tar)
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, os.path.join(ROOT, "models/YOLO-World"))  # cfg custom_imports=['yolo_world']

from proposal.hybrid_proposal import project_box  # noqa: E402

CAM_MEMBER = {
    "CAM_FRONT": "cam_front.jpg",
    "CAM_FRONT_LEFT": "cam_front_left.jpg",
    "CAM_FRONT_RIGHT": "cam_front_right.jpg",
    "CAM_BACK": "cam_back.jpg",
    "CAM_BACK_LEFT": "cam_back_left.jpg",
    "CAM_BACK_RIGHT": "cam_back_right.jpg",
}
CAMERAS = list(CAM_MEMBER.keys())

# Canonical nuScenes-10 (underscored), matching the gamma cache + native eval.
NUSC_10 = ("car", "truck", "construction_vehicle", "bus", "trailer",
           "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone")
NAME_TO_IDX = {n: i for i, n in enumerate(NUSC_10)}

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
# YOLO-World full-image detector (ONE inference per image; nusc10 vocab)
# ---------------------------------------------------------------------------
class YoloImageDetector:
    """YOLO-World run once per full image; returns all detections (xyxy, label
    index into vocab, score) after NMS + score threshold. Mirrors the repo's
    Network_2D runner setup (the proven path from run_hybrid_feasibility)."""

    def __init__(self, config_path, pretrained_path, vocab, th, nms, use_amp):
        import torch
        from mmengine.config import Config
        from mmengine.dataset import Compose
        from mmengine.runner import Runner
        from mmyolo.registry import RUNNERS
        from torchvision.ops import nms as tvnms
        from mmengine.runner.amp import autocast
        self.torch = torch
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

    def detect_path(self, path):
        """Run YOLO-World on a full image saved at ``path``. Returns
        (xyxy (N,4) float, labels (N,) int, scores (N,) float) above threshold."""
        torch = self.torch
        data_info = self.runner.pipeline(dict(img_id=0, img_path=path, texts=self.texts))
        data_batch = dict(inputs=torch.stack([data_info["inputs"]]),
                          data_samples=[data_info["data_samples"]])
        with self._autocast(enabled=self.use_amp), torch.no_grad():
            output = self.runner.model.test_step(data_batch)
        pi = output[0].pred_instances
        if len(pi.scores) == 0:
            return (np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,)))
        keep = self._nms(pi.bboxes, pi.scores, iou_threshold=self.nms_iou)
        pi = pi[keep]
        sc = pi.scores.float()
        m = sc > self.th
        bx = pi.bboxes[m].detach().cpu().numpy().astype(np.float64)
        lb = pi.labels[m].detach().cpu().numpy().astype(int)
        sn = sc[m].detach().cpu().numpy().astype(np.float64)
        return (bx, lb, sn)


# ---------------------------------------------------------------------------
# 2D matching
# ---------------------------------------------------------------------------
def iou_and_containment(roi, det):
    """roi=(x0,y0,x1,y1), det=(x0,y0,x1,y1). Returns (iou, containment) where
    containment = inter / area(roi) (how much of the projected box the detection
    covers)."""
    ax0, ay0, ax1, ay1 = roi
    bx0, by0, bx1, by1 = det
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-6, (bx1 - bx0) * (by1 - by0))
    iou = inter / (area_a + area_b - inter)
    return float(iou), float(inter / area_a)


def match_box_to_dets(roi_result, dets_by_cam, tau_iou, tau_contain):
    """Best YOLO detection over all cameras the box projects into.

    dets_by_cam[cam] = (xyxy (N,4), labels (N,), scores (N,)).
    Returns (label_idx, score, match_iou, cam) or (None, 0.0, 0.0, None)."""
    best = (None, 0.0, 0.0, None)   # (label, score, iou, cam); ranked by iou
    best_key = -1.0
    for cam, rec in roi_result.per_cam.items():
        if not rec["in_image"] or rec["roi_xyxy"] is None:
            continue
        roi = rec["roi_xyxy"]
        bx, lb, sn = dets_by_cam.get(cam, (np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,))))
        for j in range(bx.shape[0]):
            iou, cont = iou_and_containment(roi, tuple(bx[j]))
            if iou < tau_iou and cont < tau_contain:
                continue
            # rank candidates by IoU primarily; a containment-only hit (large
            # detection swallowing a small far ROI) ranks below any real IoU hit.
            key = iou if iou >= tau_iou else (tau_iou * cont)
            if key > best_key:
                best_key = key
                best = (int(lb[j]), float(sn[j]), float(iou), cam)
    return best


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-cache",
                    default="results/outdoor_native_temporal_cpcache_thr000_single_gravity")
    ap.add_argument("--shards", default="/home/rintern16/nuscenes_shards")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--out-cache", required=True,
                    help="directory for <token>.hybrid.pkl files")
    ap.add_argument("--stats-out", required=True, help="transfer-stats json path")
    ap.add_argument("--scene-limit", type=int, default=0, help="0 = all val scenes")
    ap.add_argument("--tau-iou", type=float, default=0.3)
    ap.add_argument("--tau-contain", type=float, default=0.5)
    ap.add_argument("--yolo-th", type=float, default=0.1)
    ap.add_argument("--yolo-nms", type=float, default=0.3)
    args = ap.parse_args()

    import cv2
    os.makedirs(args.out_cache, exist_ok=True)
    tmp_img = os.path.join(args.out_cache, "_tmp_full.jpg")

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.splits import val as VAL_SCENES
    from pyquaternion import Quaternion

    nusc = NuScenes(version=args.version,
                    dataroot=os.path.join(ROOT, args.dataroot), verbose=False)
    scene_shard = build_scene_shard(os.path.join(args.shards, "manifest.jsonl"))
    name2scene = {s["name"]: s for s in nusc.scene}
    val_scene_tokens = [name2scene[n]["token"] for n in VAL_SCENES if n in name2scene]
    if args.scene_limit > 0:
        val_scene_tokens = val_scene_tokens[: args.scene_limit]

    def q2m(rot, trans):
        return transform_matrix(translation=trans, rotation=Quaternion(rot))

    def scene_sample_tokens(scene_token):
        toks, cur = [], nusc.get("scene", scene_token)["first_sample_token"]
        while cur:
            toks.append(cur)
            cur = nusc.get("sample", cur)["next"]
        return toks

    cache_dir = os.path.join(ROOT, args.gamma_cache)

    print("Loading YOLO-World (nusc10 vocab) ...", flush=True)
    yolo = YoloImageDetector(
        config_path="pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
        pretrained_path="pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth",
        vocab=list(NUSC_10), th=args.yolo_th, nms=args.yolo_nms, use_amp=False)

    # ---- accumulators (transfer stats) ----
    S = {
        "n_scenes": 0, "n_samples": 0, "n_boxes": 0,
        "n_projected": 0,        # box has >=1 in-image ROI
        "n_matched": 0,          # box matched a YOLO det (label transferred)
        "n_background": 0,       # projected but no det match
        "n_not_projected": 0,    # box never lands in any image
        "labels": Counter(),     # transferred-label distribution
        "label_score_sum": defaultdict(float),
        "match_iou": [],
        "cp_agree": 0, "cp_total": 0,   # transferred label vs discarded CP class
        "by_dist": defaultdict(lambda: [0, 0, 0]),  # bin -> [matched, projected, total]
        "n_yolo_det_total": 0,
        "t_yolo_per_sample": [], "t_match_per_sample": [],
    }

    t_start = time.time()
    for si, sc_tok in enumerate(val_scene_tokens):
        sc_name = nusc.get("scene", sc_tok)["name"]
        shard = scene_shard.get(sc_name)
        if shard is None:
            print(f"  [scene {si+1}] {sc_name}: NO SHARD — skipped", flush=True)
            continue
        toks = scene_sample_tokens(sc_tok)
        toks = [t for t in toks if os.path.exists(os.path.join(cache_dir, f"{t}.pkl"))]
        if not toks:
            continue
        imgs_by_tok = load_images_for_tokens(os.path.join(args.shards, shard), toks, cv2)
        S["n_scenes"] += 1

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

            # ---- YOLO once per camera image ----
            t0 = time.perf_counter()
            dets_by_cam = {}
            for cam in CAMERAS:
                cv2.imwrite(tmp_img, imgs[cam])
                bx, lb, sn = yolo.detect_path(tmp_img)
                dets_by_cam[cam] = (bx, lb, sn)
                S["n_yolo_det_total"] += int(bx.shape[0])
            S["t_yolo_per_sample"].append(time.perf_counter() - t0)

            props = pickle.load(open(os.path.join(cache_dir, f"{tok}.pkl"), "rb"))

            hybrid_props = []
            t0 = time.perf_counter()
            for p in props:
                bbox = p["bbox_lidar"]
                r = float(np.hypot(bbox[0], bbox[1]))
                db = dist_bin(r)
                S["n_boxes"] += 1
                S["by_dist"][db][2] += 1

                roi = project_box(bbox, T_l2e, c2e, intr, hw)
                projected = any(rec["in_image"] for rec in roi.per_cam.values())
                if projected:
                    S["n_projected"] += 1
                    S["by_dist"][db][1] += 1
                else:
                    S["n_not_projected"] += 1

                label_idx, score, miou, cam = (None, 0.0, 0.0, None)
                if projected:
                    label_idx, score, miou, cam = match_box_to_dets(
                        roi, dets_by_cam, args.tau_iou, args.tau_contain)

                if label_idx is not None:
                    name = NUSC_10[label_idx]
                    cls_idx = NAME_TO_IDX[name]
                    S["n_matched"] += 1
                    S["by_dist"][db][0] += 1
                    S["labels"][name] += 1
                    S["label_score_sum"][name] += score
                    S["match_iou"].append(miou)
                    S["cp_total"] += 1
                    if name == p.get("cls_name"):
                        S["cp_agree"] += 1
                else:
                    name = "__background__"
                    cls_idx = -1
                    if projected:
                        S["n_background"] += 1

                hybrid_props.append({
                    "cls_name": name,
                    "cls_idx": cls_idx,
                    "score": float(score),              # YOLO open-vocab score
                    "score_cp": float(p.get("score", 0.0)),  # CP objectness (ablation)
                    "bbox_lidar": p["bbox_lidar"],      # geometry preserved verbatim
                    "centroid_ego": p["centroid_ego"],
                    "match_iou": float(miou),
                    "cp_cls_name": p.get("cls_name"),
                })
            S["t_match_per_sample"].append(time.perf_counter() - t0)

            with open(os.path.join(args.out_cache, f"{tok}.hybrid.pkl"), "wb") as f:
                pickle.dump(hybrid_props, f)
            S["n_samples"] += 1

        if (si + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  [scene {si+1}/{len(val_scene_tokens)}] samples={S['n_samples']} "
                  f"boxes={S['n_boxes']} matched={S['n_matched']} elapsed={el:.0f}s",
                  flush=True)

    # ---- summarize ----
    def stats(xs):
        if not xs:
            return {"n": 0}
        a = np.array(xs)
        return {"n": int(a.size), "mean": round(float(a.mean()), 4),
                "median": round(float(np.median(a)), 4),
                "p95": round(float(np.percentile(a, 95)), 4)}

    nb = max(1, S["n_boxes"])
    out = {
        "config": {
            "gamma_cache": args.gamma_cache, "out_cache": args.out_cache,
            "n_val_scenes": len(val_scene_tokens), "scene_limit": args.scene_limit,
            "tau_iou": args.tau_iou, "tau_contain": args.tau_contain,
            "yolo_th": args.yolo_th, "yolo_nms": args.yolo_nms, "vocab": list(NUSC_10),
        },
        "counts": {
            "n_scenes": S["n_scenes"], "n_samples": S["n_samples"],
            "n_boxes": S["n_boxes"], "n_projected": S["n_projected"],
            "n_matched": S["n_matched"], "n_background": S["n_background"],
            "n_not_projected": S["n_not_projected"],
            "n_yolo_det_total": S["n_yolo_det_total"],
            "projected_rate": round(S["n_projected"] / nb, 4),
            "matched_rate": round(S["n_matched"] / nb, 4),
            "matched_rate_of_projected": round(
                S["n_matched"] / max(1, S["n_projected"]), 4),
        },
        "label_distribution": {
            "labels": dict(S["labels"].most_common()),
            "mean_score_per_label": {k: round(S["label_score_sum"][k] / S["labels"][k], 4)
                                     for k in S["labels"]},
            "match_iou": stats(S["match_iou"]),
            "cp_label_agreement": round(S["cp_agree"] / max(1, S["cp_total"]), 4),
            "cp_label_agreement_n": S["cp_total"],
        },
        "by_distance": {k: {"matched": v[0], "projected": v[1], "total": v[2],
                            "match_rate": round(v[0] / max(1, v[2]), 4)}
                        for k, v in sorted(S["by_dist"].items())},
        "runtime": {
            "yolo_per_sample_s": stats(S["t_yolo_per_sample"]),
            "match_per_sample_s": stats(S["t_match_per_sample"]),
            "total_walltime_s": round(time.time() - t_start, 1),
        },
    }
    os.makedirs(os.path.dirname(args.stats_out) or ".", exist_ok=True)
    json.dump(out, open(args.stats_out, "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)
    if os.path.exists(tmp_img):
        os.remove(tmp_img)


if __name__ == "__main__":
    main()
