#!/usr/bin/env python3
"""
make_shards.py — Pack nuScenes keyframes into ~1GB WebDataset .tar shards.

WebDataset layout
-----------------
One WebDataset *sample* == one nuScenes keyframe sample token. All members of a
sample are written contiguously so a downstream ``webdataset.WebDataset`` reader
groups them correctly. Member names::

    <sample_token>.cam_front.jpg
    <sample_token>.cam_back.jpg
    ... (6 cameras)
    <sample_token>.lidar_top.pcd.bin
    <sample_token>.meta.json          # token, scene, timestamp, src paths

Reader side::

    import webdataset as wds
    ds = wds.WebDataset("nuscenes_shards/nuscenes-{000000..000NNN}.tar")

Safety model (read this before using --delete-originals)
--------------------------------------------------------
This packs what may be the *only* copy of the dataset on a shared, near-full NFS.
Deletion is therefore strictly gated:

  1. A shard is written, flushed and fsync'd, then CLOSED.
  2. The closed shard is RE-OPENED and VERIFIED: every expected member must be
     present, readable, and byte-size identical to its source file.
  3. Only sources belonging to a fully-verified, committed shard become eligible
     for deletion.
  4. Deletion happens only with --delete-originals, and removes the *exact
     verified files* one by one with os.remove(). It NEVER rmtree()s a directory
     speculatively. Empty sensor dirs are left in place.

Default run = pack only (no deletion). Use --dry-run to preview.

Crash-safety & resume
---------------------
* Shards are scene-aligned: a scene is never split across two shards, so a
  committed shard always contains whole scenes. A scene larger than the shard
  size simply produces one oversized shard (does not happen for nuScenes).
* A shard is written to ``<name>.tar.tmp`` and only renamed to the final
  ``<name>.tar`` after it passes verification, so a crash never leaves a
  half-written shard that a reader could pick up.
* After each shard commits+verifies, the scenes it contains are appended to
  ``manifest.jsonl`` (fsync'd). On restart, scenes already in the manifest are
  skipped and new shards are numbered after the existing ones (no overwrite).
  Leftover ``*.tar.tmp`` from a crashed run are removed at startup.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

# Keyframe sample_data channels we pack. fileformat is informational only.
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]
LIDAR = ["LIDAR_TOP"]
RADARS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #
def load_tables(meta_dir: Path):
    """Load the few nuScenes tables we need and build lookup structures.

    Returns scenes (ordered), and per-scene ordered list of
    (sample_token, timestamp, {channel: (filename, sd_token)}).
    """
    log(f"loading metadata from {meta_dir} ...")
    with open(meta_dir / "scene.json") as f:
        scenes = json.load(f)
    with open(meta_dir / "sample.json") as f:
        samples = json.load(f)
    with open(meta_dir / "sample_data.json") as f:  # ~1.3 GB for full trainval
        sample_data = json.load(f)
    with open(meta_dir / "sensor.json") as f:
        sensors = json.load(f)
    with open(meta_dir / "calibrated_sensor.json") as f:
        calib = json.load(f)

    # sensor_token -> channel name (e.g. "CAM_FRONT")
    sensor_channel = {s["token"]: s["channel"] for s in sensors}
    # calibrated_sensor_token -> sensor_token
    calib_sensor = {c["token"]: c["sensor_token"] for c in calib}

    # sample_token -> ordered samples; chain via "next"
    sample_by_token = {s["token"]: s for s in samples}

    # keyframe sample_data only: sample_token -> {channel: (filename, sd_token)}
    kf: dict[str, dict[str, tuple[str, str]]] = {}
    for sd in sample_data:
        if not sd.get("is_key_frame"):
            continue
        ch = sensor_channel.get(calib_sensor.get(sd["calibrated_sensor_token"]))
        if ch is None:
            continue
        kf.setdefault(sd["sample_token"], {})[ch] = (sd["filename"], sd["token"])

    # build per-scene ordered sample lists by following first_sample_token.next
    scene_samples: list[tuple[dict, list[tuple[str, int, dict]]]] = []
    for scene in scenes:
        ordered = []
        tok = scene["first_sample_token"]
        while tok:
            s = sample_by_token.get(tok)
            if s is None:
                break
            ordered.append((s["token"], s["timestamp"], kf.get(s["token"], {})))
            tok = s["next"]
        scene_samples.append((scene, ordered))
    log(f"loaded {len(scenes)} scenes, "
        f"{sum(len(o) for _, o in scene_samples)} keyframe samples")
    return scene_samples


# --------------------------------------------------------------------------- #
# Shard writer with verify-before-delete
# --------------------------------------------------------------------------- #
class ShardPacker:
    def __init__(self, out_dir: Path, src_root: Path, prefix: str,
                 max_bytes: int, channels: list[str], dry_run: bool,
                 start_idx: int, manifest_path: Path):
        self.out_dir = out_dir
        self.src_root = src_root
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.channels = channels
        self.dry_run = dry_run
        self.manifest_path = manifest_path

        self.shard_idx = start_idx
        self.tar: tarfile.TarFile | None = None
        self.tar_path: Path | None = None      # final name
        self.tmp_path: Path | None = None       # name being written
        self.cur_bytes = 0
        # files added to the *current* (not yet committed) shard:
        # list of (member_name, src_path, size)
        self.pending: list[tuple[str, Path, int]] = []
        # whole scenes packed into the current (uncommitted) shard
        self.shard_scenes: list[dict] = []
        # all source paths verified & safe to delete, across committed shards
        self.verified_sources: list[Path] = []
        self.samples_written = 0
        self.done_scene_count = 0

    # -- shard lifecycle ---------------------------------------------------- #
    def _open_new(self):
        self.tar_path = self.out_dir / f"{self.prefix}-{self.shard_idx:06d}.tar"
        self.tmp_path = self.out_dir / f"{self.prefix}-{self.shard_idx:06d}.tar.tmp"
        if not self.dry_run:
            self.tar = tarfile.open(self.tmp_path, "w")
        self.cur_bytes = 0
        self.pending = []
        self.shard_scenes = []
        log(f"  -> opening shard {self.tar_path.name}")

    def _verify(self, path: Path) -> bool:
        """Every pending member present, extractable, size == source."""
        try:
            with tarfile.open(path, "r") as t:
                members = {m.name: m for m in t.getmembers()}
                for name, _src, size in self.pending:
                    m = members.get(name)
                    if m is None or m.size != size:
                        log(f"  !! VERIFY FAIL: {name} missing/size-mismatch")
                        return False
                    f = t.extractfile(m)
                    if f is None:
                        log(f"  !! VERIFY FAIL: {name} not extractable")
                        return False
                    f.read(1)  # touch first byte
        except Exception as e:  # noqa: BLE001
            log(f"  !! VERIFY ERROR on {path.name}: {e}")
            return False
        return True

    def _write_manifest(self):
        """Append the current shard's scenes to the manifest (fsync'd)."""
        with open(self.manifest_path, "a") as f:
            for sc in self.shard_scenes:
                f.write(json.dumps({
                    "scene_token": sc["scene_token"],
                    "scene_name": sc["scene_name"],
                    "shard": self.tar_path.name,
                    "samples": sc["samples"],
                    "verified": True,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.done_scene_count += len(self.shard_scenes)

    def _commit_current(self) -> bool:
        """Close, fsync, verify, rename .tmp->final, record manifest."""
        if self.tar_path is None or not self.pending:
            return True
        if self.dry_run:
            log(f"  [dry-run] would commit {self.tar_path.name} "
                f"({self.cur_bytes/1e9:.2f} GB, {len(self.pending)} files, "
                f"{len(self.shard_scenes)} scenes)")
            self.verified_sources.extend(p for _, p, _ in self.pending)
            return True

        self.tar.close()
        fd = os.open(self.tmp_path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

        if not self._verify(self.tmp_path):
            log(f"  ✗ shard {self.tar_path.name} FAILED verification — "
                f"leaving {self.tmp_path.name}, not recorded, sources kept")
            return False

        # atomically publish the shard, then durably record it
        os.replace(self.tmp_path, self.tar_path)
        dfd = os.open(self.out_dir, os.O_RDONLY)
        os.fsync(dfd)
        os.close(dfd)
        self._write_manifest()
        log(f"  ✓ committed+verified {self.tar_path.name} "
            f"({self.cur_bytes/1e9:.2f} GB, {len(self.pending)} files, "
            f"{len(self.shard_scenes)} scenes)")
        self.verified_sources.extend(p for _, p, _ in self.pending)
        return True

    # -- scene-aligned packing --------------------------------------------- #
    def _sample_members(self, token: str, channel_files: dict[str, tuple[str, str]]):
        """Build (member_name, src, size) list + meta dict for one sample."""
        members: list[tuple[str, Path, int]] = []
        meta_members: dict[str, str] = {}
        for ch in self.channels:
            entry = channel_files.get(ch)
            if entry is None:
                continue
            filename, _sd_token = entry
            src = self.src_root / filename
            if not src.is_file():
                log(f"  (skip missing {filename})")
                continue
            ext = "jpg" if filename.endswith(".jpg") else \
                  "pcd.bin" if filename.endswith(".pcd.bin") else \
                  Path(filename).suffix.lstrip(".") or "bin"
            member_name = f"{token}.{ch.lower()}.{ext}"
            members.append((member_name, src, src.stat().st_size))
            meta_members[member_name] = filename
        return members, meta_members

    def pack_scene(self, scene_token: str, scene_name: str,
                   ordered: list[tuple[str, int, dict]]):
        """Pack a whole scene into a single shard (rolling over first if needed)."""
        # plan the scene up front so we know its size before choosing a shard
        plan = []  # (token, ts, members, meta)
        scene_bytes = 0
        for token, ts, chfiles in ordered:
            members, meta_members = self._sample_members(token, chfiles)
            if not members:
                continue
            meta = {"sample_token": token, "scene": scene_name,
                    "timestamp": ts, "members": meta_members}
            scene_bytes += sum(s for _, _, s in members)
            plan.append((token, members, meta))
        if not plan:
            log(f"  (scene {scene_name}: no files on disk, skipped)")
            return

        # scene-aligned rollover: never split a scene across shards
        if self.tar_path is None:
            self._open_new()
        elif self.pending and self.cur_bytes + scene_bytes > self.max_bytes:
            self._commit_current()
            self.shard_idx += 1
            self._open_new()
        if scene_bytes > self.max_bytes:
            log(f"  (note: scene {scene_name} ~{scene_bytes/1e9:.2f} GB exceeds "
                f"shard size; producing one oversized shard)")

        n = 0
        for token, members, meta in plan:
            for member_name, src, size in members:
                if not self.dry_run:
                    self.tar.add(str(src), arcname=member_name)
                self.cur_bytes += size
                self.pending.append((member_name, src, size))
            meta_bytes = json.dumps(meta).encode()
            if not self.dry_run:
                info = tarfile.TarInfo(f"{token}.meta.json")
                info.size = len(meta_bytes)
                self.tar.addfile(info, io.BytesIO(meta_bytes))
            self.cur_bytes += len(meta_bytes)
            self.samples_written += 1
            n += 1
        self.shard_scenes.append({"scene_token": scene_token,
                                  "scene_name": scene_name, "samples": n})

    def finalize(self) -> list[Path]:
        """Commit the last shard. Returns verified-deletable source paths."""
        self._commit_current()
        return self.verified_sources


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path,
                    default=Path.home() / "OpenYOLO3D/data/nuscenes",
                    help="nuScenes root (contains samples/, v1.0-trainval/)")
    ap.add_argument("--version", default="v1.0-trainval",
                    help="metadata subdir under --src")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "nuscenes_shards",
                    help="output dir for .tar shards")
    ap.add_argument("--shard-size-gb", type=float, default=1.0,
                    help="approx max bytes per shard (default 1.0 GB)")
    ap.add_argument("--sensors", default="cam,lidar",
                    help="comma list from {cam,lidar,radar}")
    ap.add_argument("--limit-scenes", type=int, default=0,
                    help="pack at most N scenes (0 = all); use for smoke tests")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan only: no tar written, no deletion")
    ap.add_argument("--delete-originals", action="store_true",
                    help="DELETE source files AFTER their shard is verified. "
                         "Off by default. Per-file os.remove only; never rmtree.")
    ap.add_argument("--i-have-a-backup", action="store_true",
                    help="required acknowledgement to actually delete")
    args = ap.parse_args()

    channels: list[str] = []
    sel = {s.strip() for s in args.sensors.split(",")}
    if "cam" in sel:
        channels += CAMERAS
    if "lidar" in sel:
        channels += LIDAR
    if "radar" in sel:
        channels += RADARS
    if not channels:
        sys.exit("no sensors selected")

    meta_dir = args.src / args.version
    if not meta_dir.is_dir():
        sys.exit(f"metadata dir not found: {meta_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    prefix = "nuscenes"

    # remove leftover half-written shards from a crashed run
    for tmp in args.out.glob(f"{prefix}-*.tar.tmp"):
        log(f"removing stale temp shard {tmp.name}")
        tmp.unlink()

    # resume: scenes already recorded as done, and next free shard index
    done_scenes: set[str] = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    done_scenes.add(json.loads(line)["scene_token"])
                except Exception:  # noqa: BLE001
                    pass
        log(f"resume: {len(done_scenes)} scenes already in manifest")

    existing = list(args.out.glob(f"{prefix}-*.tar"))
    start_idx = 0
    if existing:
        start_idx = max(int(p.stem.split("-")[-1]) for p in existing) + 1
        if not done_scenes:
            log(f"WARNING: {len(existing)} shard(s) present in {args.out} but "
                f"manifest has no entries — these scenes are NOT marked done and "
                f"may be re-packed into new shards. For a clean full run, clear "
                f"the output dir first. New shards start at index {start_idx}.")

    # delete-safety gate
    will_delete = args.delete_originals and not args.dry_run
    if args.delete_originals and not args.i_have_a_backup:
        sys.exit("Refusing to delete originals without --i-have-a-backup. "
                 "Confirm a verified copy exists (e.g. completed rsync to local) "
                 "and that no labmate depends on this nuScenes copy first.")

    scene_samples = load_tables(meta_dir)
    if args.limit_scenes:
        scene_samples = scene_samples[: args.limit_scenes]

    max_bytes = int(args.shard_size_gb * 1e9)
    packer = ShardPacker(args.out, args.src, prefix, max_bytes,
                         channels, args.dry_run, start_idx, manifest_path)

    log(f"packing {len(scene_samples)} scenes -> {args.out} "
        f"(shard~{args.shard_size_gb}GB, sensors={sorted(sel)}, "
        f"delete={'YES' if will_delete else 'no'})")

    total_deleted = 0
    total_freed = 0
    skipped = 0
    for scene, ordered in scene_samples:
        if scene["token"] in done_scenes:
            skipped += 1
            continue
        log(f"scene {scene['name']} ({len(ordered)} samples)")
        # whole scene goes into one shard; its sources become deletable only
        # once that shard commits+verifies (recorded in the manifest).
        packer.pack_scene(scene["token"], scene["name"], ordered)

    # finalize + collect everything verified
    verified = packer.finalize()
    log(f"done packing: {packer.samples_written} samples, "
        f"{packer.done_scene_count} scenes committed"
        + (f", {skipped} skipped (resume)" if skipped else ""))

    if will_delete:
        log(f"deleting {len(verified)} verified source files ...")
        for src in verified:
            try:
                sz = src.stat().st_size
                os.remove(src)
                total_deleted += 1
                total_freed += sz
            except FileNotFoundError:
                pass
            except OSError as e:
                log(f"  delete failed {src}: {e}")
        log(f"deleted {total_deleted} files, freed {total_freed/1e9:.1f} GB")
    elif args.delete_originals and args.dry_run:
        log(f"[dry-run] would delete {len(verified)} verified source files")
    else:
        log(f"pack-only: {len(verified)} sources verified & deletable "
            f"(re-run with --delete-originals --i-have-a-backup to remove them)")


if __name__ == "__main__":
    main()
