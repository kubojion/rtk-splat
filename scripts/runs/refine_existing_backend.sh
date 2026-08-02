#!/usr/bin/env bash
# Run only the optional RTK refinement and refined pose export for a completed
# mapper artifact. No bag, depth, feature, match, mapper, cloud, or GS stage is
# repeated. Nothing is launched in tmux.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON="${RTK_SPLAT_PYTHON:-/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python}"
COLMAP="${COLMAP_BIN:-/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap}"
CONFIG=""
WORKDIR=""
BACKEND_NAME=""
BACKEND_KIND="global"
REFINEMENT_NAME=""
POSE_NAME=""

die() { echo "FATAL: $*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: refine_existing_backend.sh --config FILE --workdir DIR \
  --backend-name NAME --refinement-name NAME --pose-name NAME

The source backend must already have completed solve, registration, and
quality stages. The refinement and pose names must be new. Re-running after an
interruption resumes the refinement stages. If the pose was already published,
this launcher refuses it instead of making an incomplete verification claim.
EOF
}

while (($#)); do
    case "$1" in
        --config) CONFIG="${2:-}"; shift 2 ;;
        --workdir) WORKDIR="${2:-}"; shift 2 ;;
        --backend-name) BACKEND_NAME="${2:-}"; shift 2 ;;
        --backend) BACKEND_KIND="${2:-}"; shift 2 ;;
        --refinement-name) REFINEMENT_NAME="${2:-}"; shift 2 ;;
        --pose-name) POSE_NAME="${2:-}"; shift 2 ;;
        --python) PYTHON="${2:-}"; shift 2 ;;
        --colmap) COLMAP="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

for value in CONFIG WORKDIR BACKEND_NAME REFINEMENT_NAME POSE_NAME; do
    [[ -n "${!value}" ]] || die "--${value,,} is required"
done
[[ -r "$CONFIG" ]] || die "cannot read config: $CONFIG"
[[ -x "$PYTHON" ]] || die "Python is unavailable: $PYTHON"
[[ -x "$COLMAP" ]] || die "COLMAP is unavailable: $COLMAP"
[[ "$BACKEND_KIND" == "global" || "$BACKEND_KIND" == "incremental" ]] \
    || die "--backend must be global or incremental"

WORKDIR="$(realpath -m "$WORKDIR")"
BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
REFINEMENT="$WORKDIR/refinement_artifacts/$REFINEMENT_NAME"
POSE="$WORKDIR/pose_artifacts/$POSE_NAME"
[[ -s "$BACKEND/stages/quality.json" ]] \
    || die "completed source backend quality is absent: $BACKEND"
[[ ! -e "$POSE" ]] \
    || die "pose artifact already exists; refusing an unverifiable shortcut: $POSE"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export COLMAP_BIN="$COLMAP"
mkdir -p "$WORKDIR/logs"
LOG="$WORKDIR/logs/rtk-refine-${REFINEMENT_NAME}-$(date '+%Y%m%d_%H%M%S').log"
CLI=("$PYTHON" -m rtk_splat.workflows.cli)
COMMON=(--config "$CONFIG" --workdir "$WORKDIR" --backend "$BACKEND_KIND" \
    --backend-name "$BACKEND_NAME" --refinement-name "$REFINEMENT_NAME")

{
    echo "RTK refinement source: $BACKEND"
    echo "RTK refinement output: $REFINEMENT"
    "${CLI[@]}" backend-refine-rtk "${COMMON[@]}"
    "${CLI[@]}" backend-export "${COMMON[@]}" --pose-name "$POSE_NAME"
} 2>&1 | tee "$LOG"

echo "COMPLETE: accepted refined pose -> $POSE"
echo "Log: $LOG"
