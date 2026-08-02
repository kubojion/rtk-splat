#!/usr/bin/env bash
# Run one fresh, quadratic pose-prior mapper arm from the sealed Citrus cache.
# This experiment deliberately stops after mapping, quality, and paired audit.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
readonly DEFAULT_PYTHON="/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python"
readonly EXPECTED_COLMAP="/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap"
readonly DEFAULT_CONFIG="$REPO_ROOT/configs/sequences/citrusfarm_05_13d_uturn.yaml"
readonly DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_v1"
readonly BACKEND_NAME="citrus-05-13d-543-735-global-bounded-v1"
readonly CONTROL_NAME="citrus-05-13d-543-735-rtk-l2-v1"
readonly CANDIDATE_NAME="citrus-05-13d-543-735-pose-prior-fresh-l2-v1"
readonly REPORT_NAME="citrusfarm_pose_prior_initialization_ab_v1.json"
readonly EXPECTED_COLMAP_SHA256="0f48cab4aa671569ef2a4be6dc4ecadd2027b08f8fa9128430b2661aad8a9fd9"
readonly MINIMUM_FREE_KIB=$((20 * 1024 * 1024))
readonly MINIMUM_AVAILABLE_MEMORY_KIB=$((16 * 1024 * 1024))

ACTION="${1:-}"
PYTHON="${RTK_SPLAT_PYTHON:-$DEFAULT_PYTHON}"
COLMAP="${COLMAP_BIN:-$EXPECTED_COLMAP}"
CONFIG="${RTK_SPLAT_CONFIG:-$DEFAULT_CONFIG}"
WORKDIR="${RTK_SPLAT_WORKDIR:-$DEFAULT_WORKDIR}"

die() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  bash scripts/experiments/$(basename "$SCRIPT_PATH") preflight
  bash scripts/experiments/$(basename "$SCRIPT_PATH") run

This reuses the sealed Citrus frontend/backend and continuation-L2 control,
then tests fresh COLMAP pose-prior initialization with the same features,
matches, rig, RTK factors, loss, and temporal holdout split. The run stops
after the fresh mapper, its quality audit, and the paired comparison report.

Default workdir: $DEFAULT_WORKDIR
Expected runtime on this machine: approximately 4-10 hours; allow overnight.
Expected extra storage: approximately 7-12 GiB (20 GiB is required to start).

Environment overrides: RTK_SPLAT_PYTHON, RTK_SPLAT_CONFIG,
RTK_SPLAT_WORKDIR. COLMAP_BIN is accepted only when it resolves to the pinned
audited executable.
EOF
}

on_error() {
    local status=$?
    echo "FATAL: fresh pose-prior experiment stopped (status $status)." >&2
    echo "Prepared inputs are reusable, but an interrupted mapper restarts from zero." >&2
    echo "No downstream reconstruction was started." >&2
    exit "$status"
}

case "$ACTION" in
    -h|--help)
        usage
        exit 0
        ;;
    preflight|run)
        (($# == 1)) || die "this launcher accepts only one action"
        ;;
    *)
        usage >&2
        die "choose exactly one action: preflight or run"
        ;;
esac

trap on_error ERR

[[ -x "$PYTHON" ]] || die "Python is unavailable: $PYTHON"
[[ -x "$COLMAP" ]] || die "COLMAP is unavailable: $COLMAP"
[[ -r "$CONFIG" ]] || die "config is unavailable: $CONFIG"
[[ -d "$WORKDIR" ]] || die "sealed Citrus workdir is unavailable: $WORKDIR"
command -v flock >/dev/null || die "flock is required for single-run locking"
command -v sha256sum >/dev/null || die "sha256sum is required"

PYTHON="$(readlink -f "$PYTHON")"
COLMAP="$(readlink -f "$COLMAP")"
CONFIG="$(readlink -f "$CONFIG")"
WORKDIR="$(readlink -f "$WORKDIR")"
[[ "$COLMAP" == "$EXPECTED_COLMAP" ]] ||
    die "COLMAP must resolve exactly to $EXPECTED_COLMAP"

BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
CONTROL="$WORKDIR/refinement_artifacts/$CONTROL_NAME"
CANDIDATE="$WORKDIR/refinement_artifacts/$CANDIDATE_NAME"
REPORT="$WORKDIR/experiment_reports/$REPORT_NAME"
[[ -d "$BACKEND" ]] || die "source backend is absent: $BACKEND"
[[ -d "$CONTROL" ]] || die "continuation-L2 control is absent: $CONTROL"

mkdir -p "$WORKDIR/logs" "$WORKDIR/experiment_reports"
LOCK="$WORKDIR/logs/citrus-pose-prior-fresh-l2.lock"
exec 9>"$LOCK"
flock -n 9 || die "another fresh Citrus pose-prior experiment is running"

ACTUAL_COLMAP_SHA256="$(sha256sum "$COLMAP" | awk '{print $1}')"
[[ "$ACTUAL_COLMAP_SHA256" == "$EXPECTED_COLMAP_SHA256" ]] ||
    die "COLMAP binary hash differs from the audited executable"

AVAILABLE_KIB="$(df -Pk "$WORKDIR" | awk 'NR == 2 {print $4}')"
[[ "$AVAILABLE_KIB" =~ ^[0-9]+$ ]] || die "cannot measure free disk space"
if [[ ! -e "$CANDIDATE" ]]; then
    ((AVAILABLE_KIB >= MINIMUM_FREE_KIB)) ||
        die "at least 20 GiB free is required to create the fresh sidecar"
fi

MEMORY_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)"
[[ "$MEMORY_AVAILABLE_KIB" =~ ^[0-9]+$ ]] ||
    die "cannot measure available memory"
((MEMORY_AVAILABLE_KIB >= MINIMUM_AVAILABLE_MEMORY_KIB)) ||
    die "at least 16 GiB MemAvailable is required before starting"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export COLMAP_BIN="$COLMAP"

# This audit intentionally hashes only small sealed control records. The core
# command performs the full database/source verification immediately before it
# prepares or resumes the candidate, and the final comparator audits both arms.
"$PYTHON" - "$BACKEND" "$CONTROL" "$CANDIDATE" "$COLMAP" <<'PY'
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from rtk_splat.backends.rtk_refinement import (
    RtkRefinementConfig,
    _normalized_refinement_config_record,
    _rtk_refinement_command,
    _verify_stage_output_hashes,
)
from rtk_splat.frontends.artifact import sha256_file


backend, control, candidate, colmap = map(
    lambda value: Path(value).expanduser().resolve(), sys.argv[1:]
)


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"sealed JSON is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


plan = load(control / "refinement_plan.json")
require(Path(str(plan.get("source_backend", ""))).resolve() == backend,
        "control references a different source backend")

# Verify the small files covered by every completed control stage marker.
_verify_stage_output_hashes(
    control,
    "prepare",
    (
        "refinement_plan.json",
        "source_database_snapshot.json",
        "optimizer_database.json",
        "prior_split.json",
        "all_images.txt",
        "constant_cameras.txt",
        "constant_rigs.txt",
        "calibration_prior_names.txt",
        "holdout_prior_names.txt",
    ),
)
_verify_stage_output_hashes(
    control, "refine", ("reports/refine.json", "refined_model_manifest.json")
)
_verify_stage_output_hashes(
    control, "quality", ("reports/quality.json", "refined_text_manifest.json")
)

# Reconnect the control's source-evidence seal to the named backend without
# doing a second multi-gigabyte database pass during preflight.
source_evidence = plan.get("source_evidence", {})
source_files = {
    "backend_plan_sha256": backend / "backend_plan.json",
    "quality_marker_sha256": backend / "stages" / "quality.json",
    "quality_report_sha256": backend / "reports" / "quality.json",
    "registered_model_manifest_sha256": backend / "registered_model_manifest.json",
    "text_model_manifest_sha256": backend / "text_model_manifest.json",
}
for field, path in source_files.items():
    require(path.is_file(), f"source backend is incomplete: {path}")
    require(source_evidence.get(field) == sha256_file(path),
            f"source backend seal changed: {field}")
source_quality = load(backend / "reports" / "quality.json")
require(source_quality.get("passed") is True,
        "source backend did not pass its visual quality audit")
require(source_quality.get("registration_fraction") == 1.0,
        "source backend is not fully registered")

expected_counts = {
    "n_images": 2990,
    "n_frames": 1495,
    "n_evaluation_priors": 1495,
    "n_calibration_database_priors": 897,
    "n_holdout_database_priors": 598,
}
for field, expected in expected_counts.items():
    require(plan.get(field) == expected,
            f"unexpected sealed {field}: {plan.get(field)!r} != {expected}")
require(len(plan.get("camera_ids", [])) == 2, "expected exactly two cameras")
require(len(plan.get("rig_ids", [])) == 1, "expected exactly one stereo rig")

split = load(control / "prior_split.json")
require(split.get("calibration_block_ids") == [0, 2, 4],
        "unexpected calibration temporal blocks")
require(split.get("holdout_block_ids") == [1, 3],
        "unexpected held-out temporal blocks")

try:
    control_config = RtkRefinementConfig(
        **_normalized_refinement_config_record(plan["config"])
    )
except (KeyError, TypeError, ValueError) as exc:
    raise SystemExit(f"invalid sealed control configuration: {exc}") from exc
require(control_config.initialization_mode == "continuation",
        "control is not continuation initialization")
require(control_config.prior_position_loss == "trivial",
        "control is not covariance-weighted quadratic/L2")

refine_report = load(control / "reports" / "refine.json")
control_command = refine_report.get("command")
require(isinstance(control_command, list), "control has no recorded mapper command")
require(len(control_command) >= 2 and control_command[1] == "pose_prior_mapper",
        "control did not use pose_prior_mapper")
require(Path(str(control_command[0])).resolve() == colmap,
        "control used a different COLMAP executable")
require(control_command.count("--input_path") == 1,
        "continuation control must contain exactly one mapper input model")
input_index = control_command.index("--input_path")
require(input_index + 1 < len(control_command), "control input model has no value")
require(Path(str(control_command[input_index + 1])).resolve()
        == Path(str(plan["input_model"])).resolve(),
        "control command and plan disagree on their input model")
loss_index = control_command.index("--use_robust_loss_on_prior_position")
require(control_command[loss_index + 1] == "0",
        "continuation control is not quadratic/L2")

fresh_config = replace(control_config, initialization_mode="fresh")
fresh_plan = plan
if candidate.exists():
    require(candidate.is_dir(), "fresh candidate path is not a directory")
    candidate_plan_path = candidate / "refinement_plan.json"
    require(candidate_plan_path.is_file(),
            "fresh candidate exists without a resumable refinement plan")
    fresh_plan = load(candidate_plan_path)
    try:
        candidate_config = RtkRefinementConfig(
            **_normalized_refinement_config_record(fresh_plan["config"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid candidate configuration: {exc}") from exc
    require(asdict(candidate_config) == asdict(fresh_config),
            "existing candidate has different experiment settings")
    require(Path(str(fresh_plan.get("source_backend", ""))).resolve() == backend,
            "existing candidate references a different source backend")
    for field, expected in expected_counts.items():
        require(fresh_plan.get(field) == expected,
                f"existing candidate has unexpected {field}")
else:
    leftovers = list(candidate.parent.glob(f".{candidate.name}.prepare-*"))
    require(not leftovers,
            "an unpublished preparation directory remains; inspect it before retrying")

fresh_command = list(
    _rtk_refinement_command(candidate, fresh_plan, fresh_config, colmap)
)
require(len(fresh_command) >= 2 and fresh_command[1] == "pose_prior_mapper",
        "fresh command is not pose_prior_mapper")
require("--input_path" not in fresh_command,
        "fresh mapper command unexpectedly contains --input_path")
fresh_loss_index = fresh_command.index("--use_robust_loss_on_prior_position")
require(fresh_command[fresh_loss_index + 1] == "0",
        "fresh mapper command is not quadratic/L2")
require(Path(fresh_command[0]).resolve() == colmap,
        "fresh command does not use the pinned COLMAP executable")

print("Artifact audit: sealed source backend and continuation-L2 control")
print("Dataset audit: 2990 images / 1495 frames / 1495 RTK priors")
print("Split audit: 897 calibration / 598 held out / 2 cameras / 1 rig")
print("Command audit: fresh pose_prior_mapper, L2 priors, no --input_path")
print("Resume state: " + ("existing candidate" if candidate.exists() else "new candidate"))
PY

echo "Resource audit: $((AVAILABLE_KIB / 1024 / 1024)) GiB free, $((MEMORY_AVAILABLE_KIB / 1024 / 1024)) GiB MemAvailable"
echo "COLMAP audit: $COLMAP ($ACTUAL_COLMAP_SHA256)"

if [[ "$ACTION" == "preflight" ]]; then
    echo "PREFLIGHT PASSED: the overnight experiment is ready; nothing was started."
    echo "Run: bash $SCRIPT_PATH run"
    exit 0
fi

LOG="$WORKDIR/logs/citrus-pose-prior-fresh-l2-$(date '+%Y%m%d_%H%M%S').log"
CLI=("$PYTHON" -m rtk_splat.workflows.cli)

{
    echo "=== fresh pose-prior mapper started $(date --iso-8601=seconds) ==="
    echo "Source backend: $BACKEND"
    echo "Control (reused): $CONTROL"
    echo "Fresh candidate: $CANDIDATE"
    echo "Expected duration: approximately 4-10 hours"
    echo "Only mapper, quality audit, and paired comparison are enabled."

    "${CLI[@]}" backend-refine-rtk \
        --config "$CONFIG" \
        --workdir "$WORKDIR" \
        --backend global \
        --backend-name "$BACKEND_NAME" \
        --refinement-name "$CANDIDATE_NAME" \
        --prior-position-loss trivial \
        --initialization-mode fresh

    "$PYTHON" -m rtk_splat.diagnostics.initialization_ab \
        --control "$CONTROL" \
        --candidate "$CANDIDATE" \
        --output "$REPORT" \
        --expected-frames 1495 \
        --expected-evaluation-priors 1495 \
        --expected-calibration-priors 897 \
        --expected-holdout-priors 598

    echo "=== fresh pose-prior mapper completed $(date --iso-8601=seconds) ==="
} 2>&1 | tee "$LOG"

echo "COMPLETE: paired report -> $REPORT"
echo "Log: $LOG"
echo "No pose artifact, point cloud, Gaussian run, or PLY was produced."
