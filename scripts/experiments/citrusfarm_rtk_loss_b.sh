#!/usr/bin/env bash
# Run only the covariance-weighted quadratic RTK-refinement candidate (B),
# then compare it with the already sealed Cauchy control (A). The script never
# exports poses, builds a cloud, starts training, or creates a PLY.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PYTHON="${RTK_SPLAT_PYTHON:-/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python}"
COLMAP="${COLMAP_BIN:-/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap}"
CONFIG="${RTK_SPLAT_CONFIG:-$REPO_ROOT/configs/sequences/citrusfarm_05_13d_uturn.yaml}"
WORKDIR="${RTK_SPLAT_WORKDIR:-/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_v1}"
BACKEND_NAME="citrus-05-13d-543-735-global-bounded-v1"
CONTROL_NAME="citrus-05-13d-543-735-rtk-refined-v1"
CANDIDATE_NAME="citrus-05-13d-543-735-rtk-l2-v1"
EXPECTED_COLMAP_SHA256="0f48cab4aa671569ef2a4be6dc4ecadd2027b08f8fa9128430b2661aad8a9fd9"
MINIMUM_FREE_KIB=$((15 * 1024 * 1024))

die() {
    echo "FATAL: $*" >&2
    exit 1
}

on_error() {
    local status=$?
    echo "FATAL: RTK loss B experiment stopped (status $status)." >&2
    echo "No downstream mapping or GS stage was started." >&2
    exit "$status"
}
trap on_error ERR

if (($#)); then
    case "$1" in
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/experiments/citrusfarm_rtk_loss_b.sh

Reuses the completed Cauchy arm A, runs only the quadratic arm B, verifies
that the arms differ only in the loss switch, and writes a paired report.
Environment overrides: RTK_SPLAT_PYTHON, COLMAP_BIN, RTK_SPLAT_CONFIG,
RTK_SPLAT_WORKDIR.
EOF
            exit 0
            ;;
        *) die "this launcher accepts no positional arguments" ;;
    esac
fi

[[ -x "$PYTHON" ]] || die "Python is unavailable: $PYTHON"
[[ -x "$COLMAP" ]] || die "COLMAP is unavailable: $COLMAP"
[[ -r "$CONFIG" ]] || die "config is unavailable: $CONFIG"
[[ -d "$WORKDIR" ]] || die "cached Citrus workdir is unavailable: $WORKDIR"
command -v flock >/dev/null || die "flock is required for single-run locking"

WORKDIR="$(realpath "$WORKDIR")"
COLMAP="$(realpath "$COLMAP")"
CONTROL="$WORKDIR/refinement_artifacts/$CONTROL_NAME"
CANDIDATE="$WORKDIR/refinement_artifacts/$CANDIDATE_NAME"
BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
REPORT="$WORKDIR/experiment_reports/citrusfarm_rtk_loss_ab_v1.json"
[[ -s "$BACKEND/stages/quality.json" ]] || die "source backend is incomplete"
[[ -s "$CONTROL/stages/quality.json" ]] || die "sealed control A is incomplete"

mkdir -p "$WORKDIR/logs" "$WORKDIR/experiment_reports"
LOCK="$WORKDIR/logs/citrus-rtk-loss-b.lock"
exec 9>"$LOCK"
flock -n 9 || die "another Citrus RTK loss experiment is already running"

CONTROL_COLMAP="$("$PYTHON" - "$CONTROL/reports/refine.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
command = report.get("command")
if not isinstance(command, list) or not command or not isinstance(command[0], str):
    raise SystemExit("invalid control refinement command")
print(command[0])
PY
)"
[[ "$COLMAP" == "$CONTROL_COLMAP" ]] ||
    die "COLMAP path must exactly match the sealed control command: $CONTROL_COLMAP"

ACTUAL_COLMAP_SHA256="$(sha256sum "$COLMAP" | awk '{print $1}')"
[[ "$ACTUAL_COLMAP_SHA256" == "$EXPECTED_COLMAP_SHA256" ]] ||
    die "COLMAP binary hash differs from the audited A/B executable"

AVAILABLE_KIB="$(df -Pk "$WORKDIR" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || die "cannot measure free disk space"
if [[ ! -e "$CANDIDATE" ]]; then
    ((AVAILABLE_KIB >= MINIMUM_FREE_KIB)) ||
        die "at least 15 GiB free is required to create candidate B"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export COLMAP_BIN="$COLMAP"
LOG="$WORKDIR/logs/citrus-rtk-loss-b-$(date '+%Y%m%d_%H%M%S').log"
CLI=("$PYTHON" -m rtk_splat.workflows.cli)

{
    echo "Control A (reused): $CONTROL"
    echo "Candidate B:        $CANDIDATE"
    echo "Loss change:        Cauchy -> covariance-weighted quadratic"
    echo "COLMAP sha256:       $ACTUAL_COLMAP_SHA256"
    echo "No pose export or GS stage is part of this experiment."

    "${CLI[@]}" backend-refine-rtk \
        --config "$CONFIG" \
        --workdir "$WORKDIR" \
        --backend global \
        --backend-name "$BACKEND_NAME" \
        --refinement-name "$CANDIDATE_NAME" \
        --prior-position-loss trivial

    "$PYTHON" -m rtk_splat.diagnostics.refinement_ab \
        --control "$CONTROL" \
        --candidate "$CANDIDATE" \
        --output "$REPORT" \
        --expected-frames 1495 \
        --expected-evaluation-priors 1495 \
        --expected-calibration-priors 897 \
        --expected-holdout-priors 598
} 2>&1 | tee "$LOG"

echo "COMPLETE: sealed A/B report -> $REPORT"
echo "Log: $LOG"
echo "No pose, cloud, GS run, or PLY was produced."
