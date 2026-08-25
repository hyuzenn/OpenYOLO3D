#!/bin/bash
# =============================================================================
# download_scannet200_filtered.sh
#
# Minimal-footprint ScanNet downloader for the OpenYOLO3D ScanNet200 pipeline.
# Downloads ONLY the native ScanNet files this repo's loader + Mask3D
# preprocessing actually consume (verified against source — see the comment
# block "WHY EACH TYPE" below). Skips the 1.3 TB of unused ScanNet assets
# (2D label/instance zips, the high-res _vh_clean.ply / _vh_clean.segs.json,
# task_data, etc.).
#
# This script ONLY downloads native ScanNet files into the official
# `scans/<scene>/` layout. It does NOT extract .sens or run Mask3D
# preprocessing — those conversion steps are documented at the bottom.
#
# ---------------------------------------------------------------------------
# WHY EACH TYPE (traceability — do not add types without a code reason):
#
#   .sens                            -> ScanNet SensReader extracts this into
#                                       color/<i>.jpg, depth/<i>.png,
#                                       poses/<i>.txt, intrinsics.txt, which
#                                       utils/__init__.py:WORLD_2_CAM reads at
#                                       runtime. (LARGE: ~1-2 GB/scene)
#   _vh_clean_2.ply                  -> the mesh. Read at runtime
#                                       (WORLD_2_CAM.load_ply, glob "*.ply")
#                                       AND by Mask3D preprocessing (coords +
#                                       RGB features -> <id>.npy).
#   _vh_clean_2.labels.ply           -> semantic labels. Mask3D preprocessing
#                                       (scannet_preprocessing.process_file)
#                                       reads it to build the <id>.npy GT
#                                       columns + ground_truth/<scene>.txt.
#                                       (train/val only — not for test)
#   .aggregation.json                -> instance grouping. Read by Mask3D
#                                       preprocessing (glob "*.aggregation.json").
#                                       (train/val only)
#   _vh_clean_2.0.010000.segs.json   -> oversegmentation. Read by Mask3D
#                                       preprocessing (glob "*[0-9].segs.json").
#                                       (train/val only)
#   .txt                             -> per-scene metadata (scene_type line).
#                                       Read by Mask3D preprocessing.
#                                       (train/val only; tiny)
#   --label_map                      -> scannetv2-labels.combined.tsv, mapping
#                                       raw_category -> scannet200 id. Required
#                                       once by scannet_preprocessing for
#                                       scannet200=True. (downloaded once)
#
# EXCLUDED (no code path reads them):
#   _vh_clean.ply, _vh_clean.segs.json, _vh_clean.aggregation.json,
#   _2d-instance(.filt).zip, _2d-label(.filt).zip, --task_data,
#   scans_test 2D data. These are the bulk of the 1.3 TB and are skipped.
# ---------------------------------------------------------------------------
#
# USAGE (idempotent; safe to re-run — already-present files are skipped):
#
#   # 1. Point at your TUM download-scannet.py and pick a split:
#   DOWNLOAD_SCRIPT=/path/to/download-scannet.py \
#   SPLIT=val INCLUDE_SENS=0 bash scripts/download_scannet200_filtered.sh
#
#   # 2. Geometry+labels only (no .sens), first 5 scenes — a cheap smoke:
#   DOWNLOAD_SCRIPT=... SPLIT=val INCLUDE_SENS=0 LIMIT=5 bash scripts/download_scannet200_filtered.sh
#
#   # 3. Full runtime download incl. RGB-D frames (LARGE — see disk warning):
#   DOWNLOAD_SCRIPT=... SPLIT=val INCLUDE_SENS=1 bash scripts/download_scannet200_filtered.sh
#
# ENV VARS:
#   DOWNLOAD_SCRIPT  (required) path to the official TUM download-scannet.py
#   SPLIT            val | train | test            (default: val)
#   INCLUDE_SENS     1=download .sens, 0=skip      (default: 0)
#   LIMIT            download only first N scenes   (default: all)
#   SCENES_FILE      override split file            (default: data/scannet200/splits/scannetv2_<SPLIT>.txt)
#   RAW_ROOT         download destination           (default: data/raw/scannet)
#   FORCE_DISK       1=skip the free-space guard    (default: 0)
#   DRY_RUN          1=print commands, download nothing (default: 0)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SPLIT="${SPLIT:-val}"
INCLUDE_SENS="${INCLUDE_SENS:-0}"
LIMIT="${LIMIT:-0}"
RAW_ROOT="${RAW_ROOT:-data/raw/scannet}"
FORCE_DISK="${FORCE_DISK:-0}"
DRY_RUN="${DRY_RUN:-0}"
DOWNLOAD_SCRIPT="${DOWNLOAD_SCRIPT:-}"
SCENES_FILE="${SCENES_FILE:-data/scannet200/splits/scannetv2_${SPLIT}.txt}"

LOG_DIR="${RAW_ROOT}/_download_logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/download_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- pre-flight ------------------------------------------------------------
if [ -z "$DOWNLOAD_SCRIPT" ] || [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "ERROR: set DOWNLOAD_SCRIPT=/path/to/download-scannet.py (not found: '$DOWNLOAD_SCRIPT')" >&2
    exit 2
fi
if [ ! -f "$SCENES_FILE" ]; then
    echo "ERROR: split file not found: $SCENES_FILE" >&2
    exit 2
fi

case "$SPLIT" in
    train|val) NEED_LABELS=1 ;;
    test)      NEED_LABELS=0 ;;
    *) echo "ERROR: SPLIT must be val|train|test (got '$SPLIT')" >&2; exit 2 ;;
esac

# Build the per-scene type list.
TYPES=("_vh_clean_2.ply")
if [ "$NEED_LABELS" = "1" ]; then
    TYPES+=("_vh_clean_2.labels.ply" ".aggregation.json" "_vh_clean_2.0.010000.segs.json" ".txt")
fi
if [ "$INCLUDE_SENS" = "1" ]; then
    TYPES+=(".sens")
fi

# Scene list (strip blanks; apply LIMIT).
mapfile -t SCENES < <(grep -vE '^\s*$' "$SCENES_FILE")
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    SCENES=("${SCENES[@]:0:$LIMIT}")
fi
N=${#SCENES[@]}

# ---- disk guard ------------------------------------------------------------
# Rough per-scene estimate (GB): geometry+labels ~0.05, .sens ~1.5.
EST_PER=1
[ "$INCLUDE_SENS" = "1" ] && EST_PER=2
EST_TOTAL=$(( N * EST_PER ))
AVAIL_GB=$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')

log "=== ScanNet200 filtered download ==="
log "split=$SPLIT  scenes=$N  include_sens=$INCLUDE_SENS  types=[${TYPES[*]}]"
log "dest=$RAW_ROOT  est_need~${EST_TOTAL}GB  avail~${AVAIL_GB}GB  log=$LOG"

if [ "$FORCE_DISK" != "1" ] && [ "$AVAIL_GB" -lt "$EST_TOTAL" ]; then
    log "ABORT: estimated need (~${EST_TOTAL}GB) exceeds available (~${AVAIL_GB}GB)."
    log "       Reduce with LIMIT=N, set INCLUDE_SENS=0, or FORCE_DISK=1 to override."
    exit 3
fi

mkdir -p "$RAW_ROOT"

run() {
    if [ "$DRY_RUN" = "1" ]; then echo "DRY: $*" | tee -a "$LOG"; else "$@"; fi
}

# ---- label map (once) ------------------------------------------------------
TSV="${RAW_ROOT}/scannetv2-labels.combined.tsv"
if [ -f "$TSV" ]; then
    log "[skip] label map present: $TSV"
else
    log "[get ] label map (scannetv2-labels.combined.tsv)"
    # download-scannet.py prompts for ToS agreement -> auto-accept via piped input.
    run bash -c "yes | python '$DOWNLOAD_SCRIPT' -o '$RAW_ROOT' --label_map" || \
        log "[warn] label_map download returned nonzero (check $LOG)"
fi

# ---- per-scene downloads ---------------------------------------------------
SCANS_DIR="scans"; [ "$SPLIT" = "test" ] && SCANS_DIR="scans_test"
i=0
for scene in "${SCENES[@]}"; do
    i=$((i+1))
    log "[$i/$N] $scene"
    for t in "${TYPES[@]}"; do
        target="${RAW_ROOT}/${SCANS_DIR}/${scene}/${scene}${t}"
        if [ -s "$target" ]; then
            log "    [skip] ${scene}${t}"
            continue
        fi
        log "    [get ] ${scene}${t}"
        # --id <scene> --type <t> downloads exactly one file into scans/<scene>/.
        # download-scannet.py resumes/overwrites partial files on re-run.
        run bash -c "yes | python '$DOWNLOAD_SCRIPT' -o '$RAW_ROOT' --id '$scene' --type '$t'" || \
            log "    [warn] failed ${scene}${t} (will retry on next run)"
    done
done

log "=== done: $i/$N scenes processed (split=$SPLIT) ==="

# =============================================================================
# POST-DOWNLOAD CONVERSION (NOT run here — documented for reference)
#
# The OpenYOLO3D runtime expects a FLAT per-scene layout:
#     data/scannet200/<scene>/{color,depth,poses}/  intrinsics.txt
#                              <scene>_vh_clean_2.ply  <id>.npy
# whereas this script downloads the official nested layout:
#     data/raw/scannet/scans/<scene>/<scene>.sens (+ .ply/.json/.txt)
#
# To convert (per scene):
#   1) Extract RGB-D frames from .sens using ScanNet's SensReader:
#        python <ScanNetRepo>/SensReader/python/reader.py \
#            --filename data/raw/scannet/scans/<scene>/<scene>.sens \
#            --output_path data/scannet200/<scene> \
#            --export_color_images --export_depth_images \
#            --export_poses --export_intrinsics
#      then rename intrinsic/intrinsic_depth.txt -> intrinsics.txt and copy
#      <scene>_vh_clean_2.ply into data/scannet200/<scene>/.
#   2) Generate <id>.npy + ground_truth/<scene>.txt with Mask3D preprocessing:
#        cd models/Mask3D && python -m datasets.preprocessing.scannet_preprocessing \
#            preprocess --data_dir data/raw/scannet --save_dir <save> --scannet200 true
#   3) Generate the Mask3D class-agnostic proposal cache
#        output/scannet200/scannet200_masks/<scene>.pt
#      by running Mask3D inference. (NOT downloadable — must be computed.)
#
# NOTE: the repo ALREADY contains the full 312-scene validation set in runtime
# form (+ ground_truth + .pt caches), so a val download is usually unnecessary.
# =============================================================================
