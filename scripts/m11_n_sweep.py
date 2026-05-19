"""Task 1.5 — M11 (FrameCountingGate) N sensitivity sweep on a 50-scene subset.

Sweeps N in {2, 3, 4, 5, 7} on a deterministic 50-scene random sample
(seed=42) from SCENE_NAMES_SCANNET200, using the cached Mask3D outputs
from Task 1.2c. Reuses ``run_one_axis`` from eval_streaming_ablation so
no method class or evaluator code is modified.

Per-axis output dirs are named ``axis_M11_N{N}`` to remain consistent
with the existing aggregate report tooling.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from evaluate import SCENE_NAMES_SCANNET200
from method_scannet.streaming.eval_streaming_ablation import run_one_axis
from utils import OpenYolo3D
from utils.utils_2d import load_yaml


N_VALUES = (2, 3, 4, 5, 7)


def pick_subset(seed: int = 42, n: int = 50) -> list[str]:
    all_scenes = list(SCENE_NAMES_SCANNET200)
    rng = random.Random(seed)
    return sorted(rng.sample(all_scenes, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--config", default="pretrained/config_scannet200.yaml",
                        type=str)
    parser.add_argument("--n-values", type=int, nargs="+", default=list(N_VALUES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-scenes", type=int, default=50)
    args = parser.parse_args()

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    scenes = pick_subset(seed=args.seed, n=args.n_scenes)
    (out_root / "subset.json").write_text(
        json.dumps({
            "seed": args.seed,
            "n_scenes_total": len(SCENE_NAMES_SCANNET200),
            "n_scenes_subset": len(scenes),
            "scenes": scenes,
        }, indent=2)
    )

    cfg = load_yaml(args.config)
    print(f"=== M11 N-sweep ===")
    print(f"N values  : {args.n_values}")
    print(f"subset    : {len(scenes)} scenes (seed={args.seed})")
    print(f"cache     : {cache_dir}")
    print(f"output    : {out_root}")

    print("Constructing OpenYolo3D ...", flush=True)
    oy3d = OpenYolo3D(args.config)

    summaries: list[dict] = []
    t_total = time.time()
    for N in args.n_values:
        name = f"M11_N{N}"
        try:
            summaries.append(run_one_axis(
                name=name,
                method_id="M11",
                method_kwargs={"N": N},
                oy3d=oy3d,
                cfg=cfg,
                cache_dir=cache_dir,
                out_root=out_root,
                scenes=scenes,
            ))
        except Exception as exc:
            print(f"[axis {name}] FAILED: {exc!r}", flush=True)
            summaries.append({"axis": name, "error": str(exc)})

    (out_root / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    print(f"\ntotal wallclock = {(time.time() - t_total) / 60:.1f} min")
    print(f"wrote {out_root / 'all_summaries.json'}")


if __name__ == "__main__":
    main()
