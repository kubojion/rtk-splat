#!/usr/bin/env bash
# Calibrated RGB-D transfer and diagnostic colour-GS run for the sealed
# Rosario v2 Sequence 5 [140,250] s pilot. No bag replay, COLMAP, or IMU.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
readonly CONFIG="$REPO_ROOT/configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml"
readonly SOURCE_SEGMENT="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment"
readonly SOURCE_POSE="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_overnight_v1/pose_artifacts/rosario-seq5-ppk-140-250-global-v1"
readonly RGB_OBSERVATIONS="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment.rgb_observations"
readonly DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v1"

readonly POSE_NAME="rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1"
readonly RUN_NAME="rosario-seq5-ppk-140-250-rgb-gs-diagnostic-v1"
readonly EXPECTED_SOURCE_TREE_SHA256="f195dd803624ffc6b114f8ef051f888de03bfe988f6e15dce08286f028d76e6f"
readonly EXPECTED_SOURCE_FILES=3048
readonly EXPECTED_SOURCE_MANIFEST_SHA256="465f692be2f639c07e1c2cba136e259b863cb12781aa55de0f3b9def2289da78"
readonly EXPECTED_RGB_MANIFEST_SHA256="3a1eaf904ed24205dcd38f273acfddd7e4a4b51b5a87e83a04b8bb552e5d4fc7"
readonly EXPECTED_RGB_FILES=1649
readonly EXPECTED_POSE_MANIFEST_SHA256="5bfba99ad2a6e4c54f867d0e2eba2f1918a3c35310c1409dbea5c945138f5956"
readonly EXPECTED_PROFILE_SHA256="4e2f92b27ed0bee505f7699456b3c6b9ee468a837e80c61c7098745e211b9ddf"
readonly EXPECTED_ROBOT_SHA256="a44a7c9a7b940adf65ac82fde7ac65492c9469b0e1cd434773fc9347d26e5b71"
readonly EXPECTED_SEQUENCE_SHA256="e8fc24a7e38fccc1280be58db14ff1f573ce7c1f1b3cf1d49e06b99aaf3d9d8e"
readonly EXPECTED_ACQUISITION_ID="20db0f906e9a4761a07f85b6908f80a5"
readonly EXPECTED_FRAMES=1012
readonly EXPECTED_TRAIN=886
readonly EXPECTED_VAL=126
readonly EXPECTED_ITERATIONS=44300

ACTION=plan
WORKDIR="$DEFAULT_WORKDIR"
if [[ -n "${RTK_SPLAT_PYTHON:-}" ]]; then
    PYTHON="$RTK_SPLAT_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
else
    PYTHON="$(command -v python3 || true)"
fi
ACKNOWLEDGED=0
RESUME=0

die() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [plan]
  $(basename "$0") preflight [--workdir PATH]
  $(basename "$0") run --acknowledge-diagnostic-render-only [--workdir PATH]
      [--resume-existing]

This run is a colour-quality experiment. The output is deliberately named
splat.DIAGNOSTIC_ONLY.ply and is not eligible for a metric RGB georeferencing
claim until independent RGB extrinsic/clock registration evidence exists.
EOF
}

print_plan() {
    cat <<EOF
Rosario RGB-D diagnostic overnight run

Inputs (already on NVMe):
  1,014 accepted IR/depth poses: $SOURCE_SEGMENT
  1,645 sealed RGB observations: $RGB_OBSERVATIONS
  accepted production source pose: $SOURCE_POSE

Phase 1 selects 1,012 genuinely synchronous RGB/depth observations, transfers
the accepted pose to the RGB optical frame, and motion-compensates depth into
each RGB image. It rejects two ~66.7 ms endpoint mismatches.

Phase 2 builds a colour/depth initial cloud and trains 44,300 presentations.
No bag replay, feature extraction, COLMAP, mapper, or IMU is used.

Expected time:
  transfer: 15-35 min
  cloud:    under 5 min
  GS:       about 3.2-4 h
  total:    about 4-5 h; allow 6 h

Expected output:
  $WORKDIR/runs/$RUN_NAME/splat.DIAGNOSTIC_ONLY.ply

Nothing runs from plan. First run:
  $SCRIPT_PATH preflight

Then start the experiment:
  $SCRIPT_PATH run --acknowledge-diagnostic-render-only
EOF
}

while (($#)); do
    case "$1" in
        plan|preflight|run) ACTION="$1"; shift ;;
        --workdir)
            (($# >= 2)) || die "--workdir needs a path"
            WORKDIR="$2"; shift 2 ;;
        --python)
            (($# >= 2)) || die "--python needs a path"
            PYTHON="$2"; shift 2 ;;
        --acknowledge-diagnostic-render-only) ACKNOWLEDGED=1; shift ;;
        --resume-existing) RESUME=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ "$ACTION" == plan ]] && { print_plan; exit 0; }
if [[ "$ACTION" == run ]] && ((ACKNOWLEDGED == 0)); then
    die "run requires --acknowledge-diagnostic-render-only before writing anything"
fi
[[ -x "$PYTHON" ]] || die "Python environment is unavailable: $PYTHON"
PYTHON="$(readlink -f "$PYTHON")"
[[ "$WORKDIR" == /* ]] || WORKDIR="$(pwd -P)/$WORKDIR"
WORKDIR="$(realpath -m "$WORKDIR")"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

readonly DEST_SEGMENT="$WORKDIR/segment"
readonly DEST_POSE="$WORKDIR/pose_artifacts/$POSE_NAME"
readonly DEST_CLOUD="$WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz"
readonly DEST_RUN="$WORKDIR/runs/$RUN_NAME"
readonly LOG_DIR="$WORKDIR/logs"
readonly STATE_DIR="$WORKDIR/run_state"

nearest_existing_parent() {
    local path="$1"
    while [[ ! -d "$path" && "$path" != / ]]; do path="$(dirname "$path")"; done
    printf '%s\n' "$path"
}

verify_inputs() {
    [[ -d "$SOURCE_SEGMENT" && -d "$SOURCE_POSE" && -d "$RGB_OBSERVATIONS" ]] \
        || die "one or more sealed input artifacts are missing"
    printf '%s  %s\n' "$EXPECTED_PROFILE_SHA256" \
        "$REPO_ROOT/configs/profiles/quality_v1.yaml" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_ROBOT_SHA256" \
        "$REPO_ROOT/configs/robots/rosario_v2_dual_m2_d435.yaml" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_SEQUENCE_SHA256" "$CONFIG" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" \
        "$SOURCE_SEGMENT/manifest.json" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_RGB_MANIFEST_SHA256" \
        "$RGB_OBSERVATIONS/manifest.json" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_POSE_MANIFEST_SHA256" \
        "$SOURCE_POSE/manifest.json" | sha256sum -c -

    local count digest
    count="$(find "$SOURCE_SEGMENT" -type f | wc -l)"
    [[ "$count" -eq "$EXPECTED_SOURCE_FILES" ]] \
        || die "source segment file count changed: $count"
    digest="$(
        cd "$SOURCE_SEGMENT"
        find . -type f -print0 | LC_ALL=C sort -z \
            | xargs -0 sha256sum | sha256sum | awk '{print $1}'
    )"
    [[ "$digest" == "$EXPECTED_SOURCE_TREE_SHA256" ]] \
        || die "source segment tree changed: $digest"
    count="$(find "$RGB_OBSERVATIONS" -type f | wc -l)"
    [[ "$count" -eq "$EXPECTED_RGB_FILES" ]] \
        || die "RGB observation file count changed: $count"

    "$PYTHON" - "$SOURCE_SEGMENT" "$SOURCE_POSE" "$RGB_OBSERVATIONS" \
        "$EXPECTED_ACQUISITION_ID" "$EXPECTED_FRAMES" \
        "$EXPECTED_TRAIN" "$EXPECTED_VAL" <<'PY'
import sys
import numpy as np

from rtk_splat.adapters.rgb_observations import load_rectified_rgb_observations
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.rgbd_transfer import (
    _associate_depth_to_rgb,
    _interpolate_viewmats,
)

source_path, pose_path, rgb_path, acquisition = sys.argv[1:5]
expected_n, expected_train, expected_val = map(int, sys.argv[5:8])
source = SegmentReader(source_path).validate()
rgb = load_rectified_rgb_observations(rgb_path)
pose = verify_pose_georeferencing_artifact(
    pose_path, expected_name="rosario-seq5-ppk-140-250-global-v1"
)
if not (
    pose["artifact_class"] == "production"
    and pose["georeferencing_status"] == "PASSED"
    and pose["metric_georeferencing_claim_eligible"] is True
):
    raise SystemExit("source pose is not accepted production georeferencing")
if source.meta.get("acquisition_id") != acquisition \
        or rgb.meta.get("acquisition_id") != acquisition:
    raise SystemExit("source and RGB acquisition IDs disagree")
if source.meta["capabilities"].get("imu_present"):
    raise SystemExit("source unexpectedly declares IMU use")
depth_indices, rgb_indices, residuals, report = _associate_depth_to_rgb(
    source.frames["depth_timestamp_ns"], rgb.frames["timestamp_ns"], 100_000
)
if len(depth_indices) != expected_n or int(np.max(np.abs(residuals))) > 100_000:
    raise SystemExit("sealed RGB/depth association changed")
split = np.empty(len(source.frames["frame_id"]), dtype="<U5")
for name in ("train", "val", "test"):
    split[np.asarray(source.manifest[name], dtype=np.int64)] = name
target = split[depth_indices]
if int(np.count_nonzero(target == "train")) != expected_train \
        or int(np.count_nonzero(target == "val")) != expected_val \
        or np.any(target == "test"):
    raise SystemExit("derived RGB-D split changed")
timestamps = np.load(f"{pose_path}/timestamps_ns.npy", allow_pickle=False)
viewmats = np.load(f"{pose_path}/viewmats.npy", allow_pickle=False)
_interpolate_viewmats(
    timestamps, viewmats, rgb.frames["timestamp_ns"][rgb_indices],
    max_gap_ns=300_000_000,
)
print(
    f"Sealed inputs passed: {len(rgb.frames['frame_id'])} RGB observations, "
    f"{len(depth_indices)} synchronous RGB-D outputs, "
    f"max residual {int(np.max(np.abs(residuals)))} ns"
)
PY
}

verify_machine() {
    local parent available required active free_gpu
    parent="$(nearest_existing_parent "$WORKDIR")"
    [[ "$(findmnt -n -o SOURCE -T "$parent")" == /dev/nvme* ]] \
        || die "the output must stay on internal NVMe storage"
    available="$(df -PB1 "$parent" | awk 'NR==2 {print $4}')"
    required=$((20 * 1024 * 1024 * 1024))
    ((available >= required)) || die "at least 20 GiB free output space is required"
    free_gpu="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | head -1 | tr -d '[:space:]')"
    [[ "$free_gpu" =~ ^[0-9]+$ ]] || die "cannot inspect GPU memory"
    ((free_gpu >= 6000)) || die "at least 6000 MiB free GPU memory is required"
    [[ -s /home/jion_kubo/.cache/torch/hub/checkpoints/vgg16-397923af.pth ]] \
        || die "cached VGG weights required by final LPIPS evaluation are absent"
    active="$(pgrep -af '[r]tk_splat\.(workflows|backends).*(rgbd_transfer|cloud|train)' || true)"
    [[ -z "$active" ]] || die "another RGB-D/cloud/train stage is active: $active"
    "$PYTHON" - "$REPO_ROOT" <<'PY'
import pathlib, sys
import cv2, gsplat, numpy, torch, torchmetrics  # noqa: F401
import rtk_splat
repo = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(rtk_splat.__file__).resolve()
if repo not in package.parents:
    raise SystemExit(f"rtk_splat resolves outside this checkout: {package}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
print(f"Package: {package}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")
PY
    echo "Resources passed: $((available / 1024 / 1024 / 1024)) GiB disk, ${free_gpu} MiB GPU free"
}

preflight() {
    verify_inputs
    verify_machine
    echo "Preflight passed. No RGB-D transfer, cloud, or GS stage was started."
}

if [[ "$ACTION" == preflight ]]; then preflight; exit 0; fi
[[ "$ACTION" == run ]] || die "unsupported action: $ACTION"
preflight

if [[ -e "$WORKDIR" ]]; then
    ((RESUME)) || die "workdir exists; choose another path or use --resume-existing"
else
    ((RESUME == 0)) || die "--resume-existing requires an existing workdir"
    mkdir -p "$WORKDIR"
fi
mkdir -p "$LOG_DIR" "$STATE_DIR"
exec 9>"$WORKDIR/.rgbd-diagnostic.lock"
flock -n 9 || die "another launcher owns this workdir"

CONFIG_FINGERPRINT="$({
    sha256sum "$SCRIPT_PATH" "$CONFIG" \
        "$REPO_ROOT/configs/profiles/quality_v1.yaml" \
        "$REPO_ROOT/configs/robots/rosario_v2_dual_m2_d435.yaml"
    find "$REPO_ROOT/src/rtk_splat" -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum
    printf '%s\n' "$EXPECTED_SOURCE_TREE_SHA256" \
        "$EXPECTED_RGB_MANIFEST_SHA256" "$EXPECTED_POSE_MANIFEST_SHA256"
} | sha256sum | awk '{print $1}')"
export RTK_SPLAT_CONFIG_SHA256="$CONFIG_FINGERPRINT"

execute_logged() {
    local stage="$1" timestamp log resources status
    shift
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    log="$LOG_DIR/${stage}_${timestamp}.log"
    resources="$LOG_DIR/${stage}_${timestamp}.resources.txt"
    echo "RUN: $stage"
    printf '  %q' "$@"; printf '\n  log: %s\n' "$log"
    set +e
    /usr/bin/time -v -o "$resources" "$@" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    ((status == 0)) || die "$stage failed; inspect $log"
}

verify_transfer() {
    "$PYTHON" - "$DEST_SEGMENT" "$DEST_POSE" "$POSE_NAME" \
        "$EXPECTED_FRAMES" "$EXPECTED_TRAIN" "$EXPECTED_VAL" <<'PY'
import json, sys
import numpy as np
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
from rtk_splat.core.segment import SegmentReader

segment_path, pose_path, pose_name = sys.argv[1:4]
expected_n, expected_train, expected_val = map(int, sys.argv[4:7])
segment = SegmentReader(segment_path).validate()
evidence = verify_pose_georeferencing_artifact(pose_path, expected_name=pose_name)
expected_status = {
    "artifact_class": "diagnostic_render_only",
    "georeferencing_status": "PASSED",
    "metric_georeferencing_claim_eligible": False,
    "diagnostic_export_requested": True,
    "diagnostic_export_override_used": False,
}
if any(evidence.get(key) != value for key, value in expected_status.items()):
    raise SystemExit("RGB pose status is not the declared diagnostic-only state")
if len(segment.frames["frame_id"]) != expected_n \
        or len(segment.manifest["train"]) != expected_train \
        or len(segment.manifest["val"]) != expected_val \
        or segment.manifest["test"]:
    raise SystemExit("derived segment frame/split count changed")
caps = segment.meta["capabilities"]
if not caps.get("rgbd") or caps.get("stereo") or caps.get("imu_present"):
    raise SystemExit("derived segment capability declaration is invalid")
derived = segment.meta["derived_segment"]
association = derived["rgb_depth_association"]
projected = derived["projected_depth_quality"]
held_out = derived["held_out_registration_quality"]
if association.get("n_matched") != expected_n or association.get("passed") is not True:
    raise SystemExit("RGB-depth association gate did not pass exactly")
if projected.get("passed") is not True or projected.get("n_zero_projected_frames") != 0:
    raise SystemExit("motion-compensated depth gate failed")
if held_out.get("status") != "NOT_SUPPLIED" \
        or held_out.get("production_promotion_allowed") is not False:
    raise SystemExit("unexpected held-out RGB registration status")
evaluation = json.load(open(f"{pose_path}/transfer_evaluation.json"))
if evaluation["quality"].get("passed") is not True \
        or evaluation.get("passed") is not False:
    raise SystemExit("RTK transfer or diagnostic classification changed")
if derived.get("no_imu_consumed") is not True:
    raise SystemExit("transfer did not preserve the no-IMU declaration")
print(
    f"Transfer verified: {expected_n} RGB-D views; coverage "
    f"{projected['projected_depth_coverage_fraction']:.4f}; retained "
    f"{projected['projected_depth_retained_fraction']:.4f}"
)
PY
}

verify_cloud() {
    "$PYTHON" - "$DEST_POSE" "$POSE_NAME" "$DEST_CLOUD" <<'PY'
import sys, numpy as np
from rtk_splat.backends.pose_evidence import (
    cloud_georeferencing_evidence, verify_pose_georeferencing_artifact,
)
pose, name, cloud = sys.argv[1:4]
expected = verify_pose_georeferencing_artifact(pose, expected_name=name)
stored = cloud_georeferencing_evidence(
    cloud, expected, allow_failed_georeferencing_for_render=True
)
with np.load(cloud, allow_pickle=False) as data:
    xyz, rgb = np.asarray(data["xyz"]), np.asarray(data["rgb"])
if stored["artifact_class"] != "diagnostic_render_only" \
        or xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape \
        or len(xyz) == 0 or not np.isfinite(xyz).all():
    raise SystemExit("diagnostic initial cloud is invalid")
print(f"Cloud verified: {len(xyz):,} coloured points")
PY
}

verify_train() {
    "$PYTHON" - "$DEST_POSE" "$POSE_NAME" "$DEST_RUN" \
        "$EXPECTED_ITERATIONS" <<'PY'
import hashlib, json, sys
from pathlib import Path
from rtk_splat.backends.pose_evidence import (
    verify_pose_georeferencing_artifact, verify_training_run_georeferencing,
)
pose, name, run_path, expected_steps = sys.argv[1:5]
run, expected_steps = Path(run_path), int(expected_steps)
evidence = verify_pose_georeferencing_artifact(pose, expected_name=name)
verify_training_run_georeferencing(run, evidence)
selection = json.loads((run / "best_checkpoint.json").read_text())
if selection.get("status") != "complete" \
        or selection.get("completed_training_steps") != expected_steps:
    raise SystemExit("GS completion record is invalid")
ply = run / "splat.DIAGNOSTIC_ONLY.ply"
if not ply.is_file() or ply.stat().st_size == 0 or (run / "splat.ply").exists():
    raise SystemExit("diagnostic PLY naming/integrity is invalid")
sidecar = json.loads((run / "splat.georeferencing.json").read_text())
digest = hashlib.sha256(ply.read_bytes()).hexdigest()
if sidecar.get("splat_file") != ply.name or sidecar.get("splat_sha256") != digest:
    raise SystemExit("diagnostic PLY hash evidence is invalid")
print(
    f"Training verified: {expected_steps:,} presentations; "
    f"best {selection['criterion']}={selection['best_metric']:.3f}"
)
PY
}

stage_done() {
    local stage="$1" marker="$STATE_DIR/$1.done"
    [[ -f "$marker" ]] || return 1
    [[ "$(<"$marker")" == "$CONFIG_FINGERPRINT" ]] \
        || die "$stage marker belongs to different code or inputs"
    case "$stage" in
        transfer) verify_transfer ;;
        cloud) verify_cloud ;;
        train) verify_train ;;
        *) die "unknown stage: $stage" ;;
    esac
}

complete_stage() {
    local stage="$1" marker="$STATE_DIR/$1.done"
    (set -o noclobber; printf '%s\n' "$CONFIG_FINGERPRINT" > "$marker") \
        || die "refusing to replace stage marker: $marker"
}

if stage_done transfer; then
    ((RESUME)) || die "transfer is already complete; use --resume-existing"
    echo "RESUME: verified transfer"
else
    if ((RESUME)) && [[ -e "$DEST_SEGMENT" || -e "$DEST_POSE" ]]; then
        verify_transfer || die "partial/untrusted transfer output exists"
        complete_stage transfer
        echo "RESUME: adopted verified transfer"
    else
        execute_logged transfer "$PYTHON" -m rtk_splat.workflows.rgbd_transfer \
            --source-segment "$SOURCE_SEGMENT" \
            --source-pose-artifact "$SOURCE_POSE" \
            --rgb-observations "$RGB_OBSERVATIONS" \
            --destination-segment "$DEST_SEGMENT" \
            --destination-pose-artifact "$DEST_POSE" \
            --max-depth-sync-residual-ns 100000 \
            --max-pose-interpolation-gap-ns 300000000 \
            --min-depth-rgb-association-fraction 0.995 \
            --min-projected-depth-coverage-fraction 0.30 \
            --min-projected-depth-retained-fraction 0.50
        verify_transfer
        complete_stage transfer
    fi
fi

CLI=("$PYTHON" -m rtk_splat.workflows.cli)
COMMON=(--config "$CONFIG" --workdir "$WORKDIR" --segment "$DEST_SEGMENT" \
    --expected-frames "$EXPECTED_FRAMES")
"${CLI[@]}" validate "${COMMON[@]}"

if stage_done cloud; then
    ((RESUME)) || die "cloud is already complete; use --resume-existing"
    echo "RESUME: verified cloud"
else
    if ((RESUME)) && [[ -e "$DEST_CLOUD" ]]; then
        verify_cloud || die "partial/untrusted cloud exists"
        complete_stage cloud
        echo "RESUME: adopted verified cloud"
    else
        execute_logged cloud "${CLI[@]}" cloud "${COMMON[@]}" \
            --pose-name "$POSE_NAME" --allow-failed-georeferencing-for-render
        verify_cloud
        complete_stage cloud
    fi
fi

if stage_done train; then
    ((RESUME)) || die "training is already complete; use --resume-existing"
    echo "RESUME: verified training"
else
    if ((RESUME)) && [[ -e "$DEST_RUN" ]]; then
        verify_train || die "training directory exists but is not complete"
        complete_stage train
        echo "RESUME: adopted verified training"
    else
        execute_logged train "${CLI[@]}" train "${COMMON[@]}" \
            --pose-name "$POSE_NAME" --run-name "$RUN_NAME" \
            --allow-failed-georeferencing-for-render
        verify_train
        complete_stage train
    fi
fi

echo "RGB diagnostic run complete: $DEST_RUN/splat.DIAGNOSTIC_ONLY.ply"
echo "This artifact is for colour-quality evaluation, not a metric RGB georeferencing claim."
