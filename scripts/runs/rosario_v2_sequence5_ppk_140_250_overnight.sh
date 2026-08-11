#!/usr/bin/env bash
# Pose-to-GS overnight baseline for the sealed Rosario v2 Sequence 5 pilot.
#
# The source segment is immutable and fully materialized on NVMe. This launcher
# never replays the ROS bags, never consumes IMU/PGT/conventional GNSS, and
# never needs the Buffalo disk. It tries a production pose export first. Only
# the exact held-out RTK residual-gate failure may fall back to separately
# labelled diagnostic-render pose/cloud/GS artifacts.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
readonly DEFAULT_CONFIG="$REPO_ROOT/configs/sequences/rosario_v2_sequence5_ppk_140_250.yaml"
readonly DEFAULT_SOURCE_SEGMENT="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_v1/segment"
readonly DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_overnight_v1"
readonly DEFAULT_PYTHON="/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python"
readonly DEFAULT_COLMAP="/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap"

readonly FRONTEND_NAME="rosario-seq5-ppk-140-250-all-gpu-v1"
readonly BACKEND_NAME="rosario-seq5-ppk-140-250-global-v1"
readonly PRODUCTION_POSE_NAME="$BACKEND_NAME"
readonly DIAGNOSTIC_POSE_NAME="${BACKEND_NAME}-diagnostic-render"
readonly PRODUCTION_TRAIN_NAME="rosario-seq5-ppk-140-250-gs-v1"
readonly DIAGNOSTIC_TRAIN_NAME="${PRODUCTION_TRAIN_NAME}-diagnostic-render"

readonly EXPECTED_SEGMENT_TREE_SHA256="f195dd803624ffc6b114f8ef051f888de03bfe988f6e15dce08286f028d76e6f"
readonly EXPECTED_SEGMENT_FILE_COUNT="3048"
readonly EXPECTED_ACQUISITION_ID="20db0f906e9a4761a07f85b6908f80a5"
readonly EXPECTED_PROFILE_SHA256="4e2f92b27ed0bee505f7699456b3c6b9ee468a837e80c61c7098745e211b9ddf"
readonly EXPECTED_ROBOT_SHA256="a44a7c9a7b940adf65ac82fde7ac65492c9469b0e1cd434773fc9347d26e5b71"
readonly EXPECTED_SEQUENCE_SHA256="e8fc24a7e38fccc1280be58db14ff1f573ce7c1f1b3cf1d49e06b99aaf3d9d8e"
readonly EXPECTED_AUTHORED_CONFIG_SHA256="24a85fd4f8c6f2fa3de71cd4203fcc1390de9b35c53b7fef9970299309f5ab09"
readonly EXPECTED_FRAMES="1014"
readonly EXPECTED_TRAIN_FRAMES="888"
readonly EXPECTED_VAL_FRAMES="126"
readonly MIN_OUTPUT_FREE_GIB="40"
readonly MIN_AVAILABLE_RAM_GIB="8"
readonly MIN_FREE_GPU_MIB="6000"

ACTION="plan"
CONFIG="$DEFAULT_CONFIG"
SOURCE_SEGMENT="$DEFAULT_SOURCE_SEGMENT"
WORKDIR="$DEFAULT_WORKDIR"
PYTHON="$DEFAULT_PYTHON"
COLMAP="$DEFAULT_COLMAP"
RESUME=0

die() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [plan]
  $(basename "$0") preflight [options]
  $(basename "$0") run [--workdir NEW_DIR] [options]
  $(basename "$0") run --workdir EXISTING_DIR --resume-existing [options]

Options:
  --config PATH          Frozen Rosario sequence config
  --source-segment PATH  Frozen immutable 1,014-frame segment
  --workdir PATH         Fresh downstream artifact root
  --python PATH          rtk-splat Python environment
  --colmap PATH          Tested COLMAP 4.1.1 executable
  --resume-existing      Verify stage receipts and continue an interrupted run
  --dry-run              Alias for plan
  -h, --help             Show this help

No bag ingest or depth computation is performed. The first GS is grayscale IR.
The launcher does not bypass missing registrations, poor visual geometry,
observability failures, or corrupt artifacts. It bypasses only a measured RTK
georeferencing-gate rejection, and stamps that fallback DIAGNOSTIC_ONLY.
EOF
}

print_plan() {
    cat <<EOF
Rosario v2 Sequence 5 [140,250] s overnight pose-to-GS baseline

Frozen input:
  segment:        $SOURCE_SEGMENT
  frames:         1,014 rectified IR stereo + recorded metric depth
  split:          888 train / 126 validation
  method input:   offline dual-M2 PPK; no IMU, PGT, conventional GNSS, or wheel
  Buffalo disk:   not used

Stages:
  validate sealed segment
  build all-frame frontend (2,028 images / 9,106 requested pairs)
  GPU SIFT + matching, calibrated rig, and PPK position priors
  bounded Global Mapper + exact all-image registration/quality gate
  production fixed-scale ENU export
  if and only if held-out RTK residual gates fail: diagnostic pose export
  pose-matched recorded-depth cloud
  grayscale IR Gaussian training and PLY export

Expected automatic controls on this RTX 3080 Laptop GPU:
  initial cloud cap:  820,841 points
  Gaussian cap:       2,462,524
  training:           44,400 image presentations

Expected elapsed time: approximately 4-8 hours; allow 8-10 hours.
Expected extra storage: approximately 8-15 GiB.

Production output:
  $WORKDIR/runs/$PRODUCTION_TRAIN_NAME/splat.ply

Georeferencing-failed visualization fallback:
  $WORKDIR/runs/$DIAGNOSTIC_TRAIN_NAME/splat.DIAGNOSTIC_ONLY.ply

Nothing runs from plan. Verify the machine and frozen input first:
  $SCRIPT_PATH preflight
EOF
}

while (($#)); do
    case "$1" in
        plan|preflight|run)
            ACTION="$1"
            shift
            ;;
        --dry-run)
            ACTION="plan"
            shift
            ;;
        --config)
            (($# >= 2)) || die "--config needs a path"
            CONFIG="$2"
            shift 2
            ;;
        --source-segment)
            (($# >= 2)) || die "--source-segment needs a path"
            SOURCE_SEGMENT="$2"
            shift 2
            ;;
        --workdir)
            (($# >= 2)) || die "--workdir needs a path"
            WORKDIR="$2"
            shift 2
            ;;
        --python)
            (($# >= 2)) || die "--python needs a path"
            PYTHON="$2"
            shift 2
            ;;
        --colmap)
            (($# >= 2)) || die "--colmap needs a path"
            COLMAP="$2"
            shift 2
            ;;
        --resume-existing)
            RESUME=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ "$ACTION" == "plan" ]] && { print_plan; exit 0; }

resolve_executable() {
    local requested="$1"
    if [[ "$requested" == */* ]]; then
        [[ -x "$requested" ]] || die "executable is unavailable: $requested"
        readlink -f "$requested"
    else
        command -v "$requested" || die "executable is unavailable: $requested"
    fi
}

nearest_existing_parent() {
    local candidate="$1"
    while [[ ! -d "$candidate" ]]; do
        [[ "$candidate" != "/" ]] || break
        candidate="$(dirname "$candidate")"
    done
    printf '%s\n' "$candidate"
}

CONFIG="$(readlink -f "$CONFIG")"
SOURCE_SEGMENT="$(readlink -f "$SOURCE_SEGMENT")"
if [[ "$WORKDIR" != /* ]]; then
    WORKDIR="$(pwd -P)/$WORKDIR"
fi
WORKDIR="$(realpath -m "$WORKDIR")"
PYTHON="$(resolve_executable "$PYTHON")"
COLMAP="$(resolve_executable "$COLMAP")"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export COLMAP_BIN="$COLMAP"

readonly FRONTEND="$WORKDIR/frontend_artifacts/$FRONTEND_NAME"
readonly BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
readonly PRODUCTION_POSE="$WORKDIR/pose_artifacts/$PRODUCTION_POSE_NAME"
readonly DIAGNOSTIC_POSE="$WORKDIR/pose_artifacts/$DIAGNOSTIC_POSE_NAME"
readonly PRODUCTION_CLOUD="$WORKDIR/cloud_artifacts/$PRODUCTION_POSE_NAME/init_cloud.npz"
readonly DIAGNOSTIC_CLOUD="$WORKDIR/cloud_artifacts/$DIAGNOSTIC_POSE_NAME/init_cloud.npz"
readonly PRODUCTION_TRAIN="$WORKDIR/runs/$PRODUCTION_TRAIN_NAME"
readonly DIAGNOSTIC_TRAIN="$WORKDIR/runs/$DIAGNOSTIC_TRAIN_NAME"
readonly LOG_DIR="$WORKDIR/logs"
readonly STATE_DIR="$WORKDIR/run_state"

verify_pinned_files() {
    [[ -r "$CONFIG" ]] || die "missing config: $CONFIG"
    [[ -d "$SOURCE_SEGMENT" ]] || die "missing source segment: $SOURCE_SEGMENT"
    [[ "$CONFIG" == "$DEFAULT_CONFIG" ]] \
        || die "this frozen launcher requires $DEFAULT_CONFIG"
    [[ "$SOURCE_SEGMENT" == "$DEFAULT_SOURCE_SEGMENT" ]] \
        || die "this frozen launcher requires $DEFAULT_SOURCE_SEGMENT"
    [[ "$WORKDIR" != "$SOURCE_SEGMENT" \
        && "$WORKDIR" != "$SOURCE_SEGMENT/"* \
        && "$SOURCE_SEGMENT" != "$WORKDIR/"* ]] \
        || die "workdir must be outside the immutable source segment"

    printf '%s  %s\n' "$EXPECTED_PROFILE_SHA256" \
        "$REPO_ROOT/configs/profiles/quality_v1.yaml" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_ROBOT_SHA256" \
        "$REPO_ROOT/configs/robots/rosario_v2_dual_m2_d435.yaml" | sha256sum -c -
    printf '%s  %s\n' "$EXPECTED_SEQUENCE_SHA256" "$CONFIG" | sha256sum -c -

    local file_count tree_digest
    file_count="$(find "$SOURCE_SEGMENT" -type f | wc -l)"
    [[ "$file_count" == "$EXPECTED_SEGMENT_FILE_COUNT" ]] \
        || die "source segment file count changed: $file_count"
    if find "$SOURCE_SEGMENT" -type l -print -quit | grep -q .; then
        die "source segment unexpectedly contains symlinks"
    fi
    tree_digest="$(
        cd "$SOURCE_SEGMENT"
        find . -type f -print0 | LC_ALL=C sort -z \
            | xargs -0 sha256sum | sha256sum | awk '{print $1}'
    )"
    [[ "$tree_digest" == "$EXPECTED_SEGMENT_TREE_SHA256" ]] \
        || die "source segment tree hash changed: $tree_digest"
    echo "Frozen segment tree: $tree_digest ($file_count files)"
}

verify_config_and_segment() {
    "$PYTHON" - \
        "$CONFIG" "$SOURCE_SEGMENT" "$EXPECTED_ACQUISITION_ID" \
        "$EXPECTED_AUTHORED_CONFIG_SHA256" \
        "$EXPECTED_FRAMES" "$EXPECTED_TRAIN_FRAMES" "$EXPECTED_VAL_FRAMES" <<'PY'
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config

config, segment_path, acquisition, authored_hash = sys.argv[1:5]
expected_n, expected_train, expected_val = map(int, sys.argv[5:8])
cfg = load_config(config)
if cfg.adapter != "ros1_rosario_v2":
    raise SystemExit(f"wrong adapter: {cfg.adapter}")
if cfg.adapter_options.gnss_source != "offline_ppk":
    raise SystemExit("overnight baseline requires the explicit offline PPK input")
if list(map(float, cfg.segment.window_s)) != [140.0, 250.0]:
    raise SystemExit("Rosario window changed")
if cfg.segment.window_epoch_source != "first_bag_log":
    raise SystemExit("Rosario epoch semantics changed")
if cfg.segment.frame_spacing_m != "auto":
    raise SystemExit("metric frame spacing must remain runtime-derived")
if cfg.frontend.keyframes.preset != "all":
    raise SystemExit("overnight baseline requires the all-frame topology")
if cfg.frontend.features.profile != "gpu":
    raise SystemExit("overnight baseline requires the tested GPU feature profile")
if cfg.mapper.backend != "global":
    raise SystemExit("overnight baseline requires Global Mapper")
if cfg.mapper.name != "rosario-seq5-ppk-140-250-global-v1":
    raise SystemExit("backend name changed")
if cfg.mapper.pose_artifact_name != cfg.mapper.name:
    raise SystemExit("production pose/backend names disagree")
if cfg.train.run_name != "rosario-seq5-ppk-140-250-gs-v1":
    raise SystemExit("training name changed")
if cfg.train.iterations != "auto" or cfg.train.max_gaussians != "auto":
    raise SystemExit("training controls must remain runtime-derived")
if cfg.train.use_right_camera:
    raise SystemExit("this baseline is left-IR GS; right supervision is unvalidated")
if cfg.pose.use_imu_tilt:
    raise SystemExit("IMU tilt must remain disabled")

segment = SegmentReader(segment_path).validate()
frames = segment.frames
caps = segment.meta["capabilities"]
expected_caps = {
    "stereo": True,
    "images_rectified": True,
    "depth_recorded": True,
    "depth_computed": False,
    "dual_rtk": True,
    "single_rtk": False,
    "imu_present": False,
    "rgbd": False,
}
for key, value in expected_caps.items():
    if caps.get(key) is not value:
        raise SystemExit(f"unexpected capability {key}={caps.get(key)!r}")
if segment.meta.get("acquisition_id") != acquisition:
    raise SystemExit("source acquisition ID changed")
if int(segment.meta["n_frames"]) != expected_n:
    raise SystemExit("source frame count changed")
if len(segment.manifest["train"]) != expected_train \
        or len(segment.manifest["val"]) != expected_val \
        or segment.manifest["test"]:
    raise SystemExit("source train/validation split changed")
if not np.asarray(frames["pose_valid"], dtype=bool).all():
    raise SystemExit("not every source frame has an initial PPK pose")
if any(not str(path) for path in frames["depth_path"]):
    raise SystemExit("not every source frame has recorded depth")
if np.max(np.abs(frames["stereo_sync_residual_ns"])) != 0 \
        or np.max(np.abs(frames["depth_sync_residual_ns"])) != 0:
    raise SystemExit("source stereo/depth synchronization changed")
policy = segment.meta["provenance"]["input_policy"]
if policy.get("gnss_source") != "offline_ppk" or not policy.get("postprocessed_input"):
    raise SystemExit("source PPK provenance changed")
if set(policy.get("excluded", [])) != {"PGT", "conventional GNSS", "IMU", "wheel odometry"}:
    raise SystemExit("source excluded-input policy changed")
configuration = segment.meta["provenance"]["configuration"]
if configuration.get("authored_config_sha256") != authored_hash:
    raise SystemExit("source authored configuration changed")

heading = segment.observations("heading")
if not np.asarray(heading["valid"], dtype=bool).all():
    raise SystemExit("not every frame retains valid dual-position heading")
if not any(name.startswith("secondary_raw_") for name in heading):
    raise SystemExit("complete secondary GNSS evidence is absent")

probe = expected_n // 2
image = cv2.imread(str(segment.root / str(frames["left_image_path"][probe])))
if image is None or image.shape != (720, 1280, 3):
    raise SystemExit("OpenCV cannot expand the IR image to three channels")
if not np.array_equal(image[..., 0], image[..., 1]) \
        or not np.array_equal(image[..., 1], image[..., 2]):
    raise SystemExit("IR channel expansion is not grayscale-preserving")
with np.load(segment.root / str(frames["depth_path"][probe]), allow_pickle=False) as depth:
    if depth["depth"].shape != image.shape[:2] or depth["valid"].shape != image.shape[:2]:
        raise SystemExit("recorded depth/image shapes disagree")
    if depth["depth"].dtype != np.float32 or depth["valid"].dtype != np.bool_:
        raise SystemExit("recorded depth semantics changed")

print(json.dumps({
    "passed": True,
    "frames": expected_n,
    "train": expected_train,
    "val": expected_val,
    "acquisition_id": acquisition,
    "capabilities": caps,
    "pose_source": "offline dual-position PPK; no IMU",
    "rendering_input": "left grayscale IR + recorded optical-z depth",
}, indent=2, sort_keys=True))
PY
}

verify_environment_and_resources() {
    "$PYTHON" - "$REPO_ROOT" <<'PY'
import importlib
import pathlib
import sys

for name in ("cv2", "gsplat", "numpy", "torch", "torchmetrics", "yaml"):
    importlib.import_module(name)
import rtk_splat
import torch

repo = pathlib.Path(sys.argv[1]).resolve()
package = pathlib.Path(rtk_splat.__file__).resolve()
if repo not in package.parents:
    raise SystemExit(f"rtk_splat resolves outside this checkout: {package}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
print(f"Python package: {package}")
print(f"PyTorch CUDA: {torch.cuda.get_device_name(0)}")
PY
    "$COLMAP" -h 2>&1 | grep -q 'COLMAP 4\.1\.1' \
        || die "this baseline requires the tested COLMAP 4.1.1 build"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    command -v flock >/dev/null || die "flock is unavailable"
    [[ -x /usr/bin/time ]] || die "/usr/bin/time is unavailable"
    [[ -s /home/jion_kubo/.cache/torch/hub/checkpoints/vgg16-397923af.pth ]] \
        || die "cached VGG weights are absent; LPIPS could fail offline"

    if pgrep -x colmap >/dev/null; then
        die "another COLMAP process is already running"
    fi
    local active
    active="$(pgrep -af '[r]tk_splat\.workflows\.cli.*(frontend|backend|cloud|train)' || true)"
    [[ -z "$active" ]] || die "another rtk-splat compute stage is running: $active"

    local output_parent available_bytes required_bytes source_device output_device
    output_parent="$(nearest_existing_parent "$WORKDIR")"
    source_device="$(findmnt -n -o SOURCE -T "$SOURCE_SEGMENT")"
    output_device="$(findmnt -n -o SOURCE -T "$output_parent")"
    [[ "$source_device" == /dev/nvme* && "$output_device" == /dev/nvme* ]] \
        || die "source/output must remain on internal NVMe storage"
    available_bytes="$(df -PB1 "$output_parent" | awk 'NR==2 {print $4}')"
    required_bytes=$((MIN_OUTPUT_FREE_GIB * 1024 * 1024 * 1024))
    ((available_bytes >= required_bytes)) \
        || die "need at least ${MIN_OUTPUT_FREE_GIB} GiB free for the overnight run"

    local available_kib required_kib
    available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    required_kib=$((MIN_AVAILABLE_RAM_GIB * 1024 * 1024))
    ((available_kib >= required_kib)) \
        || die "need at least ${MIN_AVAILABLE_RAM_GIB} GiB available RAM"

    verify_gpu_headroom

    local mains_online=0 supply
    for supply in /sys/class/power_supply/*; do
        if [[ -r "$supply/type" && -r "$supply/online" ]] \
                && [[ "$(<"$supply/type")" == "Mains" ]] \
                && [[ "$(<"$supply/online")" == "1" ]]; then
            mains_online=1
        fi
    done
    ((mains_online)) || die "connect AC power before an overnight run"

    echo "Output storage: $((available_bytes / 1024 / 1024 / 1024)) GiB free on $output_device"
    echo "Available RAM: $((available_kib / 1024 / 1024)) GiB"
    echo "AC power: connected"
}

verify_gpu_headroom() {
    local total_mib free_mib
    IFS=',' read -r total_mib free_mib < <(
        nvidia-smi --query-gpu=memory.total,memory.free \
            --format=csv,noheader,nounits | head -n 1
    )
    total_mib="${total_mib//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    [[ "$total_mib" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] \
        || die "cannot read GPU memory state"
    ((free_mib >= MIN_FREE_GPU_MIB)) || die \
        "need ${MIN_FREE_GPU_MIB} MiB free GPU memory; found ${free_mib} MiB (close GPU-heavy apps)"
    echo "GPU memory: ${free_mib}/${total_mib} MiB free"
}

preflight() {
    cd "$REPO_ROOT"
    verify_pinned_files
    verify_config_and_segment
    verify_environment_and_resources
    echo "Preflight passed. No COLMAP, cloud, or GS stage was started."
}

if [[ "$ACTION" == "preflight" ]]; then
    preflight
    exit 0
fi
[[ "$ACTION" == "run" ]] || die "unsupported action: $ACTION"

preflight

if [[ -e "$WORKDIR" ]]; then
    ((RESUME)) || die "workdir exists; choose a new path or pass --resume-existing"
else
    ((RESUME == 0)) || die "--resume-existing requires an existing workdir"
    mkdir -p "$WORKDIR"
fi
mkdir -p "$LOG_DIR" "$STATE_DIR"
exec 9>"$WORKDIR/.overnight.lock"
flock -n 9 || die "another launcher owns $WORKDIR"

CONFIG_FINGERPRINT="$({
    sha256sum "$SCRIPT_PATH"
    sha256sum \
        "$REPO_ROOT/configs/profiles/quality_v1.yaml" \
        "$REPO_ROOT/configs/robots/rosario_v2_dual_m2_d435.yaml" \
        "$CONFIG"
    find "$REPO_ROOT/src/rtk_splat" -type f -name '*.py' -print0 \
        | LC_ALL=C sort -z | xargs -0 sha256sum
    printf '%s\n' "$EXPECTED_SEGMENT_TREE_SHA256"
} | sha256sum | awk '{print $1}')"
export RTK_SPLAT_CONFIG_SHA256="$CONFIG_FINGERPRINT"

command_digest() {
    printf '%q\000' "$@" | sha256sum | awk '{print $1}'
}

write_marker() {
    local stage="$1" digest="$2" extra="${3:-}"
    local marker="$STATE_DIR/$stage.done"
    (set -o noclobber; printf '%s %s %s\n' \
        "$CONFIG_FINGERPRINT" "$digest" "$extra" > "$marker") \
        || die "refusing to replace stage marker: $marker"
}

LAST_LOG=""
execute_logged() {
    local stage="$1"
    shift
    local timestamp resource status
    timestamp="$(date '+%Y%m%d_%H%M%S')"
    LAST_LOG="$LOG_DIR/${stage}_${timestamp}.log"
    resource="$LOG_DIR/${stage}_${timestamp}.resources.txt"
    echo "RUN: $stage"
    printf '  %q' "$@"
    printf '\n  log: %s\n  resources: %s\n' "$LAST_LOG" "$resource"
    set +e
    /usr/bin/time -v -o "$resource" "$@" 2>&1 | tee "$LAST_LOG"
    status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

verify_pose() {
    local path="$1" name="$2" class="$3" status="$4"
    "$PYTHON" - "$path" "$name" "$class" "$status" <<'PY'
import sys
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact

path, name, expected_class, expected_status = sys.argv[1:5]
evidence = verify_pose_georeferencing_artifact(path, expected_name=name)
if evidence["artifact_class"] != expected_class:
    raise SystemExit("pose artifact class disagrees with launcher mode")
if evidence["georeferencing_status"] != expected_status:
    raise SystemExit("pose georeferencing status disagrees with launcher mode")
eligible = expected_class == "production" and expected_status == "PASSED"
if evidence["metric_georeferencing_claim_eligible"] is not eligible:
    raise SystemExit("pose metric-claim eligibility disagrees with launcher mode")
PY
}

verify_cloud() {
    local pose="$1" pose_name="$2" cloud="$3" diagnostic="$4"
    "$PYTHON" - "$pose" "$pose_name" "$cloud" "$diagnostic" <<'PY'
import sys
import numpy as np
from rtk_splat.backends.pose_evidence import (
    cloud_georeferencing_evidence,
    verify_pose_georeferencing_artifact,
)

pose, pose_name, cloud, diagnostic = sys.argv[1:5]
diagnostic = bool(int(diagnostic))
expected = verify_pose_georeferencing_artifact(pose, expected_name=pose_name)
stored = cloud_georeferencing_evidence(
    cloud,
    expected,
    allow_failed_georeferencing_for_render=diagnostic,
)
with np.load(cloud, allow_pickle=False) as data:
    xyz = np.asarray(data["xyz"])
    rgb = np.asarray(data["rgb"])
if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 \
        or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
    raise SystemExit("initial cloud arrays are invalid")
expected_class = "diagnostic_render_only" if diagnostic else "production"
if stored is None or stored["artifact_class"] != expected_class:
    raise SystemExit("cloud artifact class disagrees with launcher mode")
PY
}

verify_train() {
    local pose="$1" pose_name="$2" run="$3" diagnostic="$4"
    "$PYTHON" - "$pose" "$pose_name" "$run" "$diagnostic" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from rtk_splat.backends.pose_evidence import (
    verify_pose_georeferencing_artifact,
    verify_training_run_georeferencing,
)

pose, pose_name, run, diagnostic = sys.argv[1:5]
diagnostic = bool(int(diagnostic))
run = Path(run)
evidence = verify_pose_georeferencing_artifact(pose, expected_name=pose_name)
verify_training_run_georeferencing(run, evidence)
expected_class = "diagnostic_render_only" if diagnostic else "production"
if evidence["artifact_class"] != expected_class:
    raise SystemExit("training artifact class disagrees with launcher mode")
required = (
    "metrics.json", "params.pt", "best_checkpoint.json",
    "run_provenance.json", "georeferencing.json",
    "splat.georeferencing.json",
)
for name in required:
    if not (run / name).is_file() or (run / name).stat().st_size == 0:
        raise SystemExit(f"missing completed training artifact: {name}")
selection = json.loads((run / "best_checkpoint.json").read_text())
if selection.get("status") != "complete" \
        or selection.get("completed_training_steps") != 44400:
    raise SystemExit("training completion record is invalid")
metrics = json.loads((run / "metrics.json").read_text())
if not metrics or metrics[-1].get("selected_for_export") is not True \
        or metrics[-1].get("step") != 44400:
    raise SystemExit("final selected-model metrics are absent")
sidecar = json.loads((run / "splat.georeferencing.json").read_text())
ply = run / str(sidecar.get("splat_file", ""))
expected_ply = "splat.DIAGNOSTIC_ONLY.ply" if diagnostic else "splat.ply"
if ply.name != expected_ply or not ply.is_file() or ply.stat().st_size == 0:
    raise SystemExit("expected completed PLY is absent")
digest = hashlib.sha256(ply.read_bytes()).hexdigest()
if digest != sidecar.get("splat_sha256"):
    raise SystemExit("PLY hash disagrees with its georeferencing sidecar")
PY
}

verify_stage() {
    local stage="$1"
    local -a required=()
    case "$stage" in
        frontend-build)
            required=("$FRONTEND/frame_manifest.json" "$FRONTEND/rig_config.json" \
                "$FRONTEND/keyframes.json" "$FRONTEND/pairs.txt" \
                "$FRONTEND/provenance.json" "$FRONTEND/quality.json") ;;
        frontend-features)
            required=("$FRONTEND/database.db" "$FRONTEND/stages/features.json" \
                "$FRONTEND/stage_reports/features.json") ;;
        frontend-rig)
            required=("$FRONTEND/stages/rig.json" "$FRONTEND/stage_reports/rig.json") ;;
        frontend-priors)
            required=("$FRONTEND/stages/pose_priors.json" \
                "$FRONTEND/stage_reports/pose_priors.json") ;;
        frontend-match)
            required=("$FRONTEND/frontend_seal.json" \
                "$FRONTEND/stages/matching.json" \
                "$FRONTEND/stage_reports/matching.json") ;;
        backend-prepare)
            required=("$BACKEND/backend_plan.json" "$BACKEND/database.db" \
                "$BACKEND/stages/prepare.json") ;;
        backend-solve)
            required=("$BACKEND/stages/solve.json" "$BACKEND/reports/solve.json" \
                "$BACKEND/solve_model_manifest.json") ;;
        backend-register)
            required=("$BACKEND/stages/register.json" \
                "$BACKEND/reports/register.json" \
                "$BACKEND/registered_model_manifest.json") ;;
        backend-quality)
            required=("$BACKEND/stages/quality.json" \
                "$BACKEND/reports/quality.json" \
                "$BACKEND/text_model_manifest.json") ;;
        backend-export-production)
            verify_pose "$PRODUCTION_POSE" "$PRODUCTION_POSE_NAME" production PASSED
            return ;;
        backend-export-diagnostic)
            verify_pose "$DIAGNOSTIC_POSE" "$DIAGNOSTIC_POSE_NAME" \
                diagnostic_render_only FAILED
            return ;;
        cloud-production)
            verify_cloud "$PRODUCTION_POSE" "$PRODUCTION_POSE_NAME" \
                "$PRODUCTION_CLOUD" 0
            return ;;
        cloud-diagnostic)
            verify_cloud "$DIAGNOSTIC_POSE" "$DIAGNOSTIC_POSE_NAME" \
                "$DIAGNOSTIC_CLOUD" 1
            return ;;
        train-production)
            verify_train "$PRODUCTION_POSE" "$PRODUCTION_POSE_NAME" \
                "$PRODUCTION_TRAIN" 0
            return ;;
        train-diagnostic)
            verify_train "$DIAGNOSTIC_POSE" "$DIAGNOSTIC_POSE_NAME" \
                "$DIAGNOSTIC_TRAIN" 1
            return ;;
        *) die "unknown stage verifier: $stage" ;;
    esac
    local path
    for path in "${required[@]}"; do
        [[ -s "$path" ]] || return 1
    done
    if [[ "$stage" == "frontend-match" ]]; then
        "$PYTHON" - "$FRONTEND" <<'PY'
import sys
from rtk_splat.frontends.artifact import verify_frontend_seal
verify_frontend_seal(sys.argv[1])
PY
    elif [[ "$stage" == "backend-quality" ]]; then
        "$PYTHON" - "$BACKEND/reports/quality.json" "$EXPECTED_FRAMES" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
expected_frames = int(sys.argv[2])
if report.get("passed") is not True or report.get("registration_fraction") != 1.0:
    raise SystemExit("visual registration/quality gate failed")
if report.get("n_registered_images") != 2 * expected_frames:
    raise SystemExit("backend did not retain every left/right image")
PY
    fi
}

run_stage() {
    local stage="$1"
    shift
    local marker="$STATE_DIR/$stage.done" digest marked_fingerprint marked_digest
    digest="$(command_digest "$@")"
    if [[ -f "$marker" ]]; then
        ((RESUME)) || die "stage marker already exists without --resume-existing: $marker"
        read -r marked_fingerprint marked_digest _ < "$marker"
        [[ "$marked_fingerprint" == "$CONFIG_FINGERPRINT" ]] \
            || die "stage $stage belongs to a different code/config snapshot"
        [[ "$marked_digest" == "$digest" ]] \
            || die "stage $stage command changed"
        verify_stage "$stage" || die "stage $stage receipt exists but output is invalid"
        echo "RESUME: verified completed stage $stage"
        return
    fi
    if ((RESUME)) && verify_stage "$stage" >/dev/null 2>&1; then
        write_marker "$stage" "$digest" adopted_after_receipt_gap
        echo "RESUME: adopted verified completed stage $stage"
        return
    fi
    execute_logged "$stage" "$@" || die "$stage failed; inspect $LAST_LOG"
    verify_stage "$stage" || die "$stage returned success but output is incomplete"
    write_marker "$stage" "$digest"
}

CLI=("$PYTHON" -m rtk_splat.workflows.cli)
COMMON=(--config "$CONFIG" --workdir "$WORKDIR" --segment "$SOURCE_SEGMENT")

"${CLI[@]}" validate "${COMMON[@]}" --expected-frames "$EXPECTED_FRAMES"

run_stage frontend-build \
    "${CLI[@]}" frontend-build "${COMMON[@]}" \
    --frontend-name "$FRONTEND_NAME" --keyframe-preset all
run_stage frontend-features \
    "${CLI[@]}" frontend-features "${COMMON[@]}" \
    --frontend-name "$FRONTEND_NAME" --feature-profile gpu
run_stage frontend-rig \
    "${CLI[@]}" frontend-rig "${COMMON[@]}" --frontend-name "$FRONTEND_NAME"
run_stage frontend-priors \
    "${CLI[@]}" frontend-priors "${COMMON[@]}" --frontend-name "$FRONTEND_NAME"
run_stage frontend-match \
    "${CLI[@]}" frontend-match "${COMMON[@]}" --frontend-name "$FRONTEND_NAME"

run_stage backend-prepare \
    "${CLI[@]}" backend-prepare "${COMMON[@]}" \
    --frontend-name "$FRONTEND_NAME" --backend global --backend-name "$BACKEND_NAME"
run_stage backend-solve \
    "${CLI[@]}" backend-solve "${COMMON[@]}" \
    --backend global --backend-name "$BACKEND_NAME"
run_stage backend-register \
    "${CLI[@]}" backend-register "${COMMON[@]}" \
    --backend global --backend-name "$BACKEND_NAME"
run_stage backend-quality \
    "${CLI[@]}" backend-quality "${COMMON[@]}" \
    --backend global --backend-name "$BACKEND_NAME"

PRODUCTION_EXPORT=(
    "${CLI[@]}" backend-export "${COMMON[@]}"
    --backend global --backend-name "$BACKEND_NAME"
    --pose-name "$PRODUCTION_POSE_NAME"
)
PRODUCTION_EXPORT_DIGEST="$(command_digest "${PRODUCTION_EXPORT[@]}")"
PRODUCTION_EXPORT_MARKER="$STATE_DIR/backend-export-production.done"
RTK_FAILURE_MARKER="$STATE_DIR/backend-export-production.rtk_failed"

POSE_NAME=""
TRAIN_NAME=""
DIAGNOSTIC_MODE=0

if [[ -f "$PRODUCTION_EXPORT_MARKER" ]]; then
    ((RESUME)) || die "production export marker exists without --resume-existing"
    read -r marked_fingerprint marked_digest _ < "$PRODUCTION_EXPORT_MARKER"
    [[ "$marked_fingerprint" == "$CONFIG_FINGERPRINT" \
        && "$marked_digest" == "$PRODUCTION_EXPORT_DIGEST" ]] \
        || die "production export marker belongs to different inputs"
    verify_stage backend-export-production
    echo "RESUME: verified production pose export"
    POSE_NAME="$PRODUCTION_POSE_NAME"
    TRAIN_NAME="$PRODUCTION_TRAIN_NAME"
elif [[ -f "$RTK_FAILURE_MARKER" ]]; then
    ((RESUME)) || die "RTK failure receipt exists without --resume-existing"
    read -r marked_fingerprint marked_digest report_path < "$RTK_FAILURE_MARKER"
    [[ "$marked_fingerprint" == "$CONFIG_FINGERPRINT" \
        && "$marked_digest" == "$PRODUCTION_EXPORT_DIGEST" ]] \
        || die "RTK failure receipt belongs to different inputs"
    "$PYTHON" - "$report_path" "$PRODUCTION_POSE_NAME" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if report.get("pose_artifact_name") != sys.argv[2] \
        or report.get("passed") is not False \
        or report.get("georeferencing_status") != "FAILED":
    raise SystemExit("stored RTK rejection report is invalid")
PY
    echo "RESUME: verified prior held-out RTK georeferencing rejection"
elif ((RESUME)) && [[ -e "$PRODUCTION_POSE" ]] \
        && verify_stage backend-export-production >/dev/null 2>&1; then
    write_marker backend-export-production "$PRODUCTION_EXPORT_DIGEST" \
        adopted_after_receipt_gap
    echo "RESUME: adopted verified production pose export"
    POSE_NAME="$PRODUCTION_POSE_NAME"
    TRAIN_NAME="$PRODUCTION_TRAIN_NAME"
else
    [[ ! -e "$PRODUCTION_POSE" ]] \
        || die "unmarked production pose exists; refusing to overwrite it"
    if execute_logged backend-export-production "${PRODUCTION_EXPORT[@]}"; then
        verify_stage backend-export-production \
            || die "production export returned success but output is invalid"
        write_marker backend-export-production "$PRODUCTION_EXPORT_DIGEST"
        POSE_NAME="$PRODUCTION_POSE_NAME"
        TRAIN_NAME="$PRODUCTION_TRAIN_NAME"
    else
        if ! grep -Fq 'fixed-scale ENU alignment failed RTK residual gates' "$LAST_LOG"; then
            die "pose export failed for a non-georeferencing reason; inspect $LAST_LOG"
        fi
        [[ ! -e "$PRODUCTION_POSE" ]] \
            || die "failed production export unexpectedly published a pose"
        report_path="$(
            find "$BACKEND/reports" -maxdepth 1 -type f \
                -name "pose_export_alignment_${PRODUCTION_POSE_NAME}_*.json" \
                -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
        )"
        [[ -n "$report_path" && -s "$report_path" ]] \
            || die "RTK failure log has no immutable rejection report"
        "$PYTHON" - "$report_path" "$PRODUCTION_POSE_NAME" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if report.get("pose_artifact_name") != sys.argv[2] \
        or report.get("passed") is not False \
        or report.get("georeferencing_status") != "FAILED":
    raise SystemExit("RTK rejection report does not prove the expected failure")
PY
        (set -o noclobber; printf '%s %s %s\n' \
            "$CONFIG_FINGERPRINT" "$PRODUCTION_EXPORT_DIGEST" "$report_path" \
            > "$RTK_FAILURE_MARKER") \
            || die "refusing to replace RTK rejection receipt"
        echo "Held-out RTK gates failed. Continuing only as DIAGNOSTIC_RENDER_ONLY."
    fi
fi

if [[ -z "$POSE_NAME" ]]; then
    run_stage backend-export-diagnostic \
        "${CLI[@]}" backend-export "${COMMON[@]}" \
        --backend global --backend-name "$BACKEND_NAME" \
        --pose-name "$DIAGNOSTIC_POSE_NAME" \
        --allow-failed-georeferencing-for-render
    POSE_NAME="$DIAGNOSTIC_POSE_NAME"
    TRAIN_NAME="$DIAGNOSTIC_TRAIN_NAME"
    DIAGNOSTIC_MODE=1
fi

if ((DIAGNOSTIC_MODE)); then
    run_stage cloud-diagnostic \
        "${CLI[@]}" cloud "${COMMON[@]}" --pose-name "$POSE_NAME" \
        --allow-failed-georeferencing-for-render
    if [[ -e "$DIAGNOSTIC_TRAIN" \
            && ! -f "$STATE_DIR/train-diagnostic.done" ]] \
            && { ((RESUME == 0)) \
                || ! verify_stage train-diagnostic >/dev/null 2>&1; }; then
        die "partial diagnostic training cannot be resumed; preserve it and use a new workdir"
    fi
    verify_gpu_headroom
    run_stage train-diagnostic \
        "${CLI[@]}" train "${COMMON[@]}" \
        --pose-name "$POSE_NAME" --run-name "$TRAIN_NAME" \
        --allow-failed-georeferencing-for-render
    FINAL_POSE="$DIAGNOSTIC_POSE"
    FINAL_TRAIN="$DIAGNOSTIC_TRAIN"
else
    run_stage cloud-production \
        "${CLI[@]}" cloud "${COMMON[@]}" --pose-name "$POSE_NAME"
    if [[ -e "$PRODUCTION_TRAIN" \
            && ! -f "$STATE_DIR/train-production.done" ]] \
            && { ((RESUME == 0)) \
                || ! verify_stage train-production >/dev/null 2>&1; }; then
        die "partial production training cannot be resumed; preserve it and use a new workdir"
    fi
    verify_gpu_headroom
    run_stage train-production \
        "${CLI[@]}" train "${COMMON[@]}" \
        --pose-name "$POSE_NAME" --run-name "$TRAIN_NAME"
    FINAL_POSE="$PRODUCTION_POSE"
    FINAL_TRAIN="$PRODUCTION_TRAIN"
fi

"$PYTHON" - \
    "$BACKEND/reports/quality.json" "$FINAL_POSE/quality.json" \
    "$FINAL_TRAIN/metrics.json" "$FINAL_TRAIN/georeferencing.json" \
    "$FINAL_TRAIN/splat.georeferencing.json" <<'PY'
import json
import sys
visual = json.load(open(sys.argv[1]))
pose = json.load(open(sys.argv[2]))
history = json.load(open(sys.argv[3]))
georef = json.load(open(sys.argv[4]))
splat = json.load(open(sys.argv[5]))
selected = history[-1]
residual = pose["holdout_rtk_residual_m"]
print(json.dumps({
    "artifact_class": georef["artifact_class"],
    "georeferencing_status": georef["georeferencing_status"],
    "metric_georeferencing_claim_eligible": georef[
        "metric_georeferencing_claim_eligible"
    ],
    "registered_images": visual["n_registered_images"],
    "registration_fraction": visual["registration_fraction"],
    "mean_reprojection_error_px": visual["checks"][
        "mean_reprojection_error_px"
    ]["value"],
    "heldout_rtk_median_residual_m": residual["median"],
    "heldout_rtk_p95_inlier_residual_m": residual[
        "p95_euclidean_inliers"
    ],
    "selected_model_step": selected["model_step"],
    "completed_training_steps": selected["step"],
    "psnr_masked": selected["psnr_masked"],
    "psnr_masked_cc": selected["psnr_masked_cc"],
    "ssim": selected["ssim"],
    "lpips_cc": selected["lpips_cc"],
    "splat_file": splat["splat_file"],
}, indent=2, sort_keys=True))
if not georef["metric_georeferencing_claim_eligible"]:
    print(
        "\n*** VISUALIZATION ONLY: NOT ELIGIBLE FOR A METRIC "
        "GEOREFERENCING CLAIM ***"
    )
PY

echo "COMPLETE: $WORKDIR"
