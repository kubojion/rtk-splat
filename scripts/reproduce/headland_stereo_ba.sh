#!/usr/bin/env bash
# Reproduce the calibrated-stereo headland experiment from an existing
# canonical segment. This script never selects, extracts, or computes depth.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_ARGS=("$@")
PYTHON_BIN="${RTK_SPLAT_PYTHON:-python3}"
CONFIG="${RTK_SPLAT_CONFIG:-$REPO_ROOT/configs/reproductions/headland_stereo_ba.yaml}"
POSE_ARTIFACT="${RTK_SPLAT_POSE_ARTIFACT:-colmap_stereo_repro}"
RUN_NAME="${RTK_SPLAT_RUN:-headland_stereo_ba_repro}"
COLMAP_EXE="${COLMAP_BIN:-${RTK_SPLAT_COLMAP:-colmap}}"
MIN_FREE_GB="${RTK_SPLAT_MIN_FREE_GB:-20}"
MODE="run"
RESUME=0

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/headland_stereo_ba.sh [options]

Options:
  --check                 Read-only validation; run no mapping or training.
  --resume                Resume an interrupted COLMAP sidecar. The GS run
                          directory must still be absent.
  --config PATH           Configuration YAML.
  --pose-artifact NAME    New pose artifact name.
  --run NAME              New Gaussian run name.
  --python PATH           Python interpreter.
  --colmap PATH           COLMAP executable.
  -h, --help              Show this help.

Environment equivalents:
  RTK_SPLAT_CONFIG, RTK_SPLAT_POSE_ARTIFACT, RTK_SPLAT_RUN,
  RTK_SPLAT_PYTHON, RTK_SPLAT_COLMAP, COLMAP_BIN
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
WORKDIR="$("$PYTHON_BIN" -c \
    'import sys; from rtk_splat.configio import load_config; print(load_config(sys.argv[1]).paths.workdir)' \
    "$CONFIG")"
SEGMENT="$WORKDIR/segment"
POSE_DIR="$SEGMENT/pose_artifacts/$POSE_ARTIFACT"
RUN_DIR="$WORKDIR/runs/$RUN_NAME"

for required in \
    "$SEGMENT/segment_meta.json" \
    "$SEGMENT/manifest.json" \
    "$SEGMENT/viewmats.npy" \
    "$SEGMENT/cam_centers.npy"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: canonical segment is incomplete: $required" >&2
        exit 1
    fi
done
if [[ ! -d "$SEGMENT/images" || ! -d "$SEGMENT/depth" ]]; then
    echo "FATAL: canonical images or depth are missing under $SEGMENT" >&2
    exit 1
fi

N_FRAMES="$("$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["n_frames"])' \
    "$SEGMENT/segment_meta.json")"
N_LEFT="$(find "$SEGMENT/images" -maxdepth 1 -type f -name 'left_*.jpg' | wc -l)"
N_RIGHT="$(find "$SEGMENT/images" -maxdepth 1 -type f -name 'right_*.jpg' | wc -l)"
N_DEPTH="$(find "$SEGMENT/depth" -maxdepth 1 -type f -name '*.npz' | wc -l)"
if [[ "$N_LEFT" -ne "$N_FRAMES" || "$N_RIGHT" -ne "$N_FRAMES" ||
      "$N_DEPTH" -ne "$N_FRAMES" ]]; then
    echo "FATAL: expected $N_FRAMES left/right/depth artifacts; found " \
         "$N_LEFT/$N_RIGHT/$N_DEPTH" >&2
    exit 1
fi

COLMAP_BANNER="$("$COLMAP_EXE" -h 2>&1 | head -n 1)"
if [[ "$COLMAP_BANNER" != "COLMAP 4.1.1"*"(Commit "*" with CUDA)" ]]; then
    echo "FATAL: golden run used COLMAP 4.1.1 with CUDA; got: " \
         "$COLMAP_BANNER" >&2
    exit 1
fi
FREE_GB="$(df --output=avail -BG "$WORKDIR" | tail -n 1 | tr -dc '0-9')"
if [[ "$FREE_GB" -lt "$MIN_FREE_GB" ]]; then
    echo "FATAL: only ${FREE_GB}G free at $WORKDIR; require " \
         "${MIN_FREE_GB}G" >&2
    exit 1
fi

if [[ -e "$RUN_DIR" ]]; then
    echo "FATAL: refusing to overwrite GS run: $RUN_DIR" >&2
    echo "Choose a new --run name." >&2
    exit 1
fi
if [[ -e "$POSE_DIR" && "$RESUME" -ne 1 ]]; then
    echo "FATAL: refusing to overwrite pose artifact: $POSE_DIR" >&2
    echo "Choose a new --pose-artifact name or use --resume intentionally." >&2
    exit 1
fi

echo "Repository:    $REPO_ROOT"
echo "Config:        $CONFIG"
echo "Workdir:       $WORKDIR"
echo "Frames:        $N_FRAMES stereo pairs"
echo "Pose artifact: $POSE_ARTIFACT"
echo "GS run:        $RUN_NAME"
echo "Python:        $("$PYTHON_BIN" --version 2>&1)"
echo "COLMAP:        $COLMAP_BANNER"
echo "Free space:    ${FREE_GB}G"

if [[ "$MODE" == "check" ]]; then
    "$PYTHON_BIN" -m unittest discover -s tests -v
    echo "READY: checks were read-only; no COLMAP or GS stage was run."
    exit 0
fi

if [[ -z "${RTK_SPLAT_INHIBITED:-}" ]] &&
   command -v systemd-inhibit >/dev/null 2>&1; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="RTK-Splat stereo BA" \
        bash "$0" "${ORIGINAL_ARGS[@]}"
fi

mkdir -p "$WORKDIR/logs"
LOG="$WORKDIR/logs/headland_stereo_ba_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
{
    echo "=== stereo BA reproduction started $(date --iso-8601=seconds) ==="
    "$PYTHON_BIN" -m rtk_splat.cli stereo-prepare \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    "$PYTHON_BIN" -m rtk_splat.cli stereo-solve \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    "$PYTHON_BIN" -m rtk_splat.cli stereo-export \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    "$PYTHON_BIN" -m rtk_splat.cli cloud \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    "$PYTHON_BIN" -m rtk_splat.cli train \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" --run "$RUN_NAME"
    "$PYTHON_BIN" -m rtk_splat.cli diagnose \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME" || true
    "$PYTHON_BIN" -m rtk_splat.cli evalonly \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME" --split train --max-frames 64 || true
    echo "=== stereo BA reproduction complete $(date --iso-8601=seconds) ==="
    echo "The machine was deliberately left on."
} 2>&1 | tee "$LOG"
