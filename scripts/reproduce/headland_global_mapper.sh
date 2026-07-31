#!/usr/bin/env bash
# Controlled headland A/B: cached stereo front end -> Global Mapper -> strict
# metric gates -> pose-matched cloud -> identical full GS training.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_ARGS=("$@")
PYTHON_BIN="${RTK_SPLAT_PYTHON:-python3}"
CONFIG="${RTK_SPLAT_CONFIG:-$REPO_ROOT/configs/reproductions/headland_stereo_ba.yaml}"
POSE_ARTIFACT="${RTK_SPLAT_POSE_ARTIFACT:-colmap_global_headland_reduced_v1}"
RUN_NAME="${RTK_SPLAT_RUN:-tile_turn4_global_mapper_reduced_v1}"
COLMAP_EXE="${COLMAP_BIN:-${RTK_SPLAT_COLMAP:-colmap}}"
MODE="run"
RESUME=0
POSE_ONLY=1

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/headland_global_mapper.sh [options]

Options:
  --check                 Read-only validation; run no mapping or training.
  --resume                Continue fully prepared/solved pose stages only.
                          Interrupted mapper and GS runs are not resumable.
  --pose-only             Stop after the gated Global Mapper pose export.
  --with-gs               After pose integrity passes, build its cloud and run
                          the full 65,000-iteration exploratory GS arm.
  --config PATH           Configuration YAML.
  --pose-artifact NAME    New destination pose artifact.
  --run NAME              New Gaussian run name.
  --python PATH           Python interpreter.
  --colmap PATH           COLMAP 4.1.1 executable.
  -h, --help              Show this help.

The source stereo artifact is read from global_mapper.source_pose_artifact.
Feature extraction and matching are never rerun. GS starts only if every pose
quality gate passes.
EOF
}

while (($#)); do
    case "$1" in
        --check)
            MODE="check"
            shift
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --pose-only)
            POSE_ONLY=1
            shift
            ;;
        --with-gs)
            POSE_ONLY=0
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --pose-artifact)
            POSE_ARTIFACT="$2"
            shift 2
            ;;
        --run)
            RUN_NAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --colmap)
            COLMAP_EXE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "FATAL: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$POSE_ARTIFACT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "FATAL: invalid pose artifact name: $POSE_ARTIFACT" >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "FATAL: invalid run name: $RUN_NAME" >&2
    exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "FATAL: config does not exist: $CONFIG" >&2
    exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "FATAL: Python is unavailable: $PYTHON_BIN" >&2
    exit 1
fi
if ! command -v "$COLMAP_EXE" >/dev/null 2>&1; then
    echo "FATAL: COLMAP is unavailable: $COLMAP_EXE" >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export COLMAP_BIN="$COLMAP_EXE"

cd "$REPO_ROOT"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import cv2
import numpy
import rosbags
import scipy
import yaml
PY
then
    echo "FATAL: $PYTHON_BIN lacks the isolated RTK-Splat dependencies." >&2
    echo "Pass --python /path/to/the/rtk-splat/environment/bin/python." >&2
    exit 1
fi
if [[ "$POSE_ONLY" -ne 1 ]] && ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import gsplat
import torch
import torchmetrics
PY
then
    echo "FATAL: $PYTHON_BIN lacks the GPU training dependencies." >&2
    echo "Pass --python /path/to/the/rtk-splat/environment/bin/python." >&2
    exit 1
fi

readarray -t CONFIG_VALUES < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from rtk_splat.configio import load_config
cfg = load_config(sys.argv[1])
print(cfg.paths.workdir)
print(cfg.global_mapper.source_pose_artifact)
print(getattr(cfg.global_mapper, "minimum_free_space_gb", 15.0))
PY
)
WORKDIR="${CONFIG_VALUES[0]}"
SOURCE_ARTIFACT="${CONFIG_VALUES[1]}"
MIN_FREE_GB="${CONFIG_VALUES[2]}"
SEGMENT="$WORKDIR/segment"
SOURCE_DIR="$SEGMENT/pose_artifacts/$SOURCE_ARTIFACT"
SOURCE_DB="$SOURCE_DIR/colmap/database.db"
POSE_DIR="$SEGMENT/pose_artifacts/$POSE_ARTIFACT"
RUN_DIR="$WORKDIR/runs/$RUN_NAME"

pose_export_complete() {
    "$PYTHON_BIN" - "$CONFIG" "$POSE_ARTIFACT" <<'PY' >/dev/null
import json
import sys
from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import load_pose_artifact
cfg = load_config(sys.argv[1])
cfg.pose.artifact = sys.argv[2]
quality_path = cfg.paths.workdir / "segment" / "pose_artifacts" \
    / sys.argv[2] / "quality.json"
if not quality_path.is_file():
    raise SystemExit(1)
quality = json.loads(quality_path.read_text())
if not quality.get("accepted_for_gs"):
    raise SystemExit(1)
load_pose_artifact(cfg.paths.workdir / "segment", cfg)
for name in ("frame_ids.npy", "registered.npy"):
    if not (quality_path.parent / name).is_file():
        raise SystemExit(1)
PY
}

cloud_complete() {
    "$PYTHON_BIN" - "$CONFIG" "$POSE_ARTIFACT" <<'PY' >/dev/null
import sys
from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import (
    cloud_path, load_pose_artifact, verify_cloud_matches_poses)
cfg = load_config(sys.argv[1])
cfg.pose.artifact = sys.argv[2]
segment = cfg.paths.workdir / "segment"
viewmats, _ = load_pose_artifact(segment, cfg)
verify_cloud_matches_poses(
    cloud_path(segment, cfg), viewmats, require_fingerprint=True)
PY
}

for required in \
    "$SEGMENT/segment_meta.json" \
    "$SEGMENT/manifest.json" \
    "$SEGMENT/viewmats.npy" \
    "$SEGMENT/cam_centers.npy" \
    "$SOURCE_DB" \
    "$SOURCE_DIR/quality.json"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: required input is missing: $required" >&2
        exit 1
    fi
done
if [[ ! -d "$SOURCE_DIR/colmap/images" ]]; then
    echo "FATAL: source COLMAP image tree is missing." >&2
    exit 1
fi

COLMAP_HELP="$("$COLMAP_EXE" -h 2>&1)"
COLMAP_BANNER="${COLMAP_HELP%%$'\n'*}"
if [[ "$COLMAP_BANNER" != "COLMAP 4.1.1"*"(Commit "*" with CUDA)" ]]; then
    echo "FATAL: expected COLMAP 4.1.1 with CUDA; got: $COLMAP_BANNER" >&2
    exit 1
fi
if [[ "$COLMAP_HELP" != *"global_mapper"* ]]; then
    echo "FATAL: this COLMAP build has no integrated global_mapper." >&2
    exit 1
fi
FREE_GB="$(df --output=avail -BG "$WORKDIR" | tail -n 1 | tr -dc '0-9')"
MIN_FREE_GB_INT="${MIN_FREE_GB%.*}"
if [[ "$FREE_GB" -lt "$MIN_FREE_GB_INT" ]]; then
    echo "FATAL: only ${FREE_GB}G free; require ${MIN_FREE_GB}G." >&2
    exit 1
fi

if [[ -e "$POSE_DIR" && "$RESUME" -ne 1 ]]; then
    echo "FATAL: refusing to overwrite pose artifact: $POSE_DIR" >&2
    echo "Choose a new --pose-artifact name or use --resume intentionally." >&2
    exit 1
fi
if [[ "$POSE_ONLY" -ne 1 && -e "$RUN_DIR" ]]; then
    echo "FATAL: refusing to overwrite GS run: $RUN_DIR" >&2
    exit 1
fi

echo "Repository:      $REPO_ROOT"
echo "Config:          $CONFIG"
echo "Workdir:         $WORKDIR"
echo "Source artifact: $SOURCE_ARTIFACT"
echo "Pose artifact:   $POSE_ARTIFACT"
echo "GS run:          $RUN_NAME"
echo "COLMAP:          $COLMAP_BANNER"
echo "Free space:      ${FREE_GB}G"

if [[ "$MODE" == "check" ]]; then
    "$PYTHON_BIN" -m unittest discover -s tests -v
    echo "READY: read-only checks passed; no artifact or solve was created."
    exit 0
fi

if [[ -z "${RTK_SPLAT_INHIBITED:-}" ]] &&
   command -v systemd-inhibit >/dev/null 2>&1; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle \
        --why="RTK-Splat headland Global Mapper A/B" \
        bash "$0" "${ORIGINAL_ARGS[@]}"
fi

mkdir -p "$WORKDIR/logs"
LOG="$WORKDIR/logs/headland_global_mapper_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
{
    echo "=== Global Mapper A/B started $(date --iso-8601=seconds) ==="
    if [[ ! -f "$POSE_DIR/source.json" ]]; then
        "$PYTHON_BIN" -m rtk_splat.cli global-prepare \
            --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    else
        echo "global-prepare: already complete"
    fi
    "$PYTHON_BIN" -m rtk_splat.cli global-solve \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    if ! pose_export_complete; then
        "$PYTHON_BIN" -m rtk_splat.cli global-export \
            --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    else
        echo "global-export: already complete"
    fi
    if ! pose_export_complete; then
        echo "FATAL: global-export did not publish a complete pose artifact." >&2
        exit 1
    fi
    if [[ "$POSE_ONLY" -eq 1 ]]; then
        echo "=== gated pose-only A/B complete $(date --iso-8601=seconds) ==="
        exit 0
    fi
    if [[ ! -e "$POSE_DIR/init_cloud.npz" ]]; then
        "$PYTHON_BIN" -m rtk_splat.cli cloud \
            --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    elif cloud_complete; then
        echo "cloud: already complete and pose fingerprint verified"
    else
        echo "FATAL: existing cloud is incomplete or mismatched; choose a new " \
             "pose artifact rather than overwriting it." >&2
        exit 1
    fi
    "$PYTHON_BIN" -m rtk_splat.cli train \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" --run "$RUN_NAME"
    "$PYTHON_BIN" -m rtk_splat.cli diagnose \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME" || true
    "$PYTHON_BIN" -m rtk_splat.cli evalonly \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME" --split train --max-frames 64 || true
    echo "=== Global Mapper + full GS A/B complete $(date --iso-8601=seconds) ==="
    echo "The machine was deliberately left on."
} 2>&1 | tee "$LOG"
