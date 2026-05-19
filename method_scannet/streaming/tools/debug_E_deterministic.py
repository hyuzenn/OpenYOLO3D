"""Task 1.2c Option H — run the same offline/streaming dump as Option E,
but with cuDNN deterministic + fixed seeds + ``warn_only`` deterministic
algorithms. Captures (a) whether the Mask3D non-determinism collapses
and (b) which ops PyTorch flags as non-deterministic for follow-up.

Reuses ``debug_compare.run_one_scene`` so the comparison is apples-to-apples
with the Option E run.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Determinism setup MUST happen before importing torch-heavy code.
# --------------------------------------------------------------------------
import os

# CUBLAS deterministic workspace (required by torch.use_deterministic_algorithms).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ``warn_only=True`` keeps non-deterministic ops alive but emits a warning
# for each. We capture those warnings to enumerate the residual sources.
torch.use_deterministic_algorithms(True, warn_only=True)

_NONDET_WARNINGS: list[dict] = []


def _capture_nondeterministic_warning(message, category, filename, lineno, file=None, line=None):
    msg = str(message)
    cat_name = getattr(category, "__name__", str(category))
    # Track everything but tag whether it's a deterministic-related warning.
    is_nondet = (
        "deterministic" in msg.lower()
        or "non-deterministic" in msg.lower()
        or "cublas" in msg.lower()
    )
    _NONDET_WARNINGS.append(
        {
            "is_nondet": is_nondet,
            "category": cat_name,
            "message": msg,
            "filename": filename,
            "lineno": lineno,
        }
    )


warnings.showwarning = _capture_nondeterministic_warning  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Now import the rest. We deliberately reuse Option E's helpers so the
# only knob being tested here is determinism.
# --------------------------------------------------------------------------

from method_scannet.streaming.tools.debug_compare import run_one_scene  # noqa: E402
from utils import OpenYolo3D  # noqa: E402
from utils.utils_2d import load_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", required=True, type=str)
    parser.add_argument(
        "--output",
        default="results/2026-05-13_streaming_debug_H_deterministic",
        type=str,
    )
    parser.add_argument(
        "--config", default="pretrained/config_scannet200.yaml", type=str
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(args.config)
    print("=== Task 1.2c Option H — cuDNN deterministic + seeds ===", flush=True)
    print(f"SEED={SEED}", flush=True)
    print(f"cudnn.deterministic={torch.backends.cudnn.deterministic}", flush=True)
    print(f"cudnn.benchmark={torch.backends.cudnn.benchmark}", flush=True)
    print(
        f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}",
        flush=True,
    )
    print(
        f"torch.are_deterministic_algorithms_enabled()="
        f"{torch.are_deterministic_algorithms_enabled()}",
        flush=True,
    )

    print("Constructing OpenYolo3D ...", flush=True)
    t0 = time.time()
    oy3d = OpenYolo3D(args.config)
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    summaries = []
    for scene_name in args.scenes:
        try:
            summaries.append(run_one_scene(scene_name, out, oy3d, cfg))
        except Exception as exc:
            print(f"  scene {scene_name} raised: {exc!r}", flush=True)
            summaries.append({"scene_name": scene_name, "error": str(exc)})

    # --- Warnings audit -------------------------------------------------
    nondet_only = [w for w in _NONDET_WARNINGS if w["is_nondet"]]
    warning_audit = {
        "total_warnings_captured": len(_NONDET_WARNINGS),
        "nondeterministic_related": len(nondet_only),
        "samples": _NONDET_WARNINGS[:40],
        "nondet_samples": nondet_only[:20],
    }
    (out / "deterministic_warnings.json").write_text(
        json.dumps(warning_audit, indent=2)
    )

    print(f"\n=== Determinism warnings captured ===", flush=True)
    print(f"  total                  : {warning_audit['total_warnings_captured']}", flush=True)
    print(f"  deterministic-related  : {warning_audit['nondeterministic_related']}", flush=True)
    if nondet_only:
        seen = set()
        print("  Unique nondet messages (first lines):")
        for w in nondet_only:
            first_line = w["message"].split("\n")[0][:120]
            if first_line not in seen:
                seen.add(first_line)
                print(f"    - {first_line}", flush=True)

    (out / "scenes_run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {out / 'scenes_run_summary.json'}")
    print(f"wrote {out / 'deterministic_warnings.json'}")


if __name__ == "__main__":
    main()
