"""Smoke test: SCANNET200_SCENES_FILE scene-list injection.

No inference. Verifies only: import -> scene-list build -> subset-filter pass.
Runs each scenario in this single process by re-importing `evaluate` fresh
under a controlled env var, since the list is materialized at import time.

Usage:
    python -m method_scannet.streaming.tools.smoke_scene_list_injection
"""
import argparse
import importlib
import os
import sys

VAL_TXT = "data/scannet200/splits/scannetv2_val.txt"
TRAIN_TXT = "data/scannet200/splits/scannetv2_train.txt"


def _fresh_import_scene_names():
    """Drop cached `evaluate` and re-import so the env var is re-read."""
    for m in [k for k in sys.modules if k == "evaluate" or k.startswith("evaluate.")]:
        del sys.modules[m]
    import evaluate as _eval_mod
    return _eval_mod.SCENE_NAMES_SCANNET200


def _read_txt(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def test_default_is_val_312():
    os.environ.pop("SCANNET200_SCENES_FILE", None)
    names = _fresh_import_scene_names()
    val_txt = _read_txt(VAL_TXT)
    import evaluate as _eval_mod
    assert len(names) == 312, f"default len={len(names)} (expected 312)"
    # backward-compat: default must be the ORIGINAL hardcoded literal, byte-for-byte
    assert names is _eval_mod._SCENE_NAMES_SCANNET200_VAL, "default is not the preserved literal"
    # same 312 scenes as the canonical val split (order differs: literal is
    # scene-id sorted, val.txt is not — order is irrelevant to evaluation)
    assert set(names) == set(val_txt), "default scene set != scannetv2_val.txt set"
    print(f"[A] default (env unset)        -> {len(names)} scenes  OK "
          f"(original literal preserved; == val.txt as a set)")
    return names


def test_train_override_1201():
    os.environ["SCANNET200_SCENES_FILE"] = TRAIN_TXT
    names = _fresh_import_scene_names()
    train_txt = _read_txt(TRAIN_TXT)
    assert len(names) == 1201, f"train len={len(names)} (expected 1201)"
    assert names == train_txt, "override list != scannetv2_train.txt"
    print(f"[B] SCANNET200_SCENES_FILE=train -> {len(names)} scenes  OK (== train.txt)")
    return names


def test_conf_skip_subset_filter(train_names):
    """eval_method_22_conf_skip._maybe_filter_scenes must accept train scenes
    when the train list is active, and still reject genuinely-unknown ids."""
    os.environ["SCANNET200_SCENES_FILE"] = TRAIN_TXT
    _fresh_import_scene_names()  # ensure module-level list is the train list
    from method_scannet.eval_method_22_conf_skip import _maybe_filter_scenes

    # pick 3 real train scenes that are NOT in the val set
    val_set = set(_read_txt(VAL_TXT))
    train_only = [s for s in train_names if s not in val_set][:3]
    assert train_only, "no train-only scenes found"

    opt = argparse.Namespace(scenes=",".join(train_only), scene_limit=0)
    _maybe_filter_scenes(opt)  # raises SystemExit if any scene is unknown
    import evaluate as _eval_mod
    assert _eval_mod.SCENE_NAMES_SCANNET200 == train_only, "filter did not patch subset"
    print(f"[C] conf_skip --scenes <train>  -> subset filter PASS for {train_only}")

    # negative control: a bogus scene must still be rejected
    bogus = argparse.Namespace(scenes="scene9999_99", scene_limit=0)
    try:
        _maybe_filter_scenes(bogus)
    except SystemExit:
        print("[C] conf_skip --scenes <bogus>  -> correctly REJECTED  OK")
    else:
        raise AssertionError("bogus scene was NOT rejected")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../..")
    print(f"cwd={os.getcwd()}")
    val_names = test_default_is_val_312()
    train_names = test_train_override_1201()
    test_conf_skip_subset_filter(train_names)
    # restore clean state
    os.environ.pop("SCANNET200_SCENES_FILE", None)
    assert len(_fresh_import_scene_names()) == 312, "post-test default regressed"
    print("\nALL SMOKE CHECKS PASSED (no inference run).")


if __name__ == "__main__":
    main()
