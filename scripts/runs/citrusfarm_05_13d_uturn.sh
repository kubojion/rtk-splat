#!/usr/bin/env bash
# Reproducible CitrusFarm 05_13D window [543,735] pipeline.
#
# This script is deliberately stage-oriented. It never overwrites an artifact,
# never starts in tmux, and only resumes stages carrying matching script-owned
# and native rtk_splat completion evidence.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
readonly DEFAULT_CONFIG="$REPO_ROOT/configs/sequences/citrusfarm_05_13d_uturn.yaml"
readonly ROBOT_CONFIG="$REPO_ROOT/configs/robots/citrusfarm_jackal_zed2i_piksi.yaml"
readonly DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_v2"
readonly DEFAULT_PYTHON="/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python"
readonly DEFAULT_COLMAP="/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap"
readonly DATASET_ROOT="/media/jion_kubo/Buffalo SSD/CitrusFarm"
readonly SEQUENCE_ROOT="$DATASET_ROOT/05_13D_Jackal"
readonly GT_CSV="$DATASET_ROOT/ground_truth/05_13D_Jackal/gt.csv"
readonly FRONTEND_NAME="citrus-05-13d-543-735-all-gpu-v2"
readonly BACKEND_NAME="citrus-05-13d-543-735-global-bounded-v2"
readonly POSE_NAME="$BACKEND_NAME"
readonly TRAIN_NAME="citrus-05-13d-543-735-gs-v2"
readonly SOURCE_SEGMENT_NAME="citrus-05-13d-543-735-rgb-v2"
readonly DERIVED_SEGMENT_NAME="citrus-05-13d-543-735-sgbm-v2"
readonly MIN_OUTPUT_FREE_GIB=65

ACTION="plan"
CONFIG="$DEFAULT_CONFIG"
WORKDIR="$DEFAULT_WORKDIR"
PYTHON="$DEFAULT_PYTHON"
COLMAP="$DEFAULT_COLMAP"
RESUME=0
STORAGE_PROBE=1

die() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  $(basename "$0") [plan]
  $(basename "$0") preflight [options]
  $(basename "$0") run --workdir NEW_DIR [options]
  $(basename "$0") run --workdir EXISTING_DIR --resume-existing [options]

Options:
  --config PATH          Sequence configuration (default: $DEFAULT_CONFIG)
  --workdir PATH         Internal-SSD artifact root (default: $DEFAULT_WORKDIR)
  --python PATH          rtk-splat Python (default: $DEFAULT_PYTHON)
  --colmap PATH          COLMAP 4.1 executable (default: $DEFAULT_COLMAP)
  --resume-existing      Resume only verified completed stages in WORKDIR
  --skip-storage-probe   Skip the small read-only Buffalo stability probe
  --dry-run              Alias for plan; no dataset or artifact I/O
  -h, --help             Show this help

No process is launched in tmux. The source bags are read-only. A failed or
partial training directory is never silently reused.
EOF
}

print_plan() {
    cat <<EOF
CitrusFarm 05_13D end-to-end plan

Input window:
  relative time:       [543, 735] s from first Piksi bag-log timestamp
  duration/path:       192 s / approximately 229 m
  frame policy:        0.15 m metric spacing (frame count is measured, not fixed)
  expected scale:      roughly 1,500 stereo pairs; exact count is an output
  sensors used:        rectified stereo RGB + computed stereo depth + Piksi RTK
  pose source:         GNSS course, with per-recording audited clock correction
  mapper:              bounded Global Mapper; incremental is not auto-started
  GS supervision:      left RGB, same proven 65k/2.5M quality profile as headland

Stages and updated estimates on this machine (first full window measured):
  drive/config/timing preflight        2-5 min
  ROS1 ingest + lossless extraction   10-15 min  (measured 9m53s)
  immutable SGBM depth                  4-8 min  (measured 3m29s)
  frontend build/features/rig           3-6 min  (measured 2m49s)
  RTK-guided matching                  14-25 min  (measured 13m39s)
  bounded Global Mapper                7-15 min  (measured solve 6m54s)
  registration/quality/export           4-8 min
  pose-matched initialization cloud     5-20 min
  65k Gaussian training + final eval   3-5 h
  total if every pose gate passes       approximately 4-6 h

The first visual-only solve registered every image but failed the independent
held-out RTK export gate. Repeating the same mapper is not expected to cure
that time-dependent pose disagreement; preserve the failed evidence and use a
new workdir only for a provenance-complete reproduction after pose refinement.

The headland 24.8 dB result is a regression reference, not a cross-dataset
threshold: CitrusFarm has different cameras, resolution, motion, foliage and
validation views. This run enforces complete stereo registration, held-out RTK
gates and identical high-quality GS settings; its PSNR is measured honestly.

Nothing runs from plan. Start with:
  $SCRIPT_PATH preflight

Then launch into a new internal directory:
  $SCRIPT_PATH run --workdir $DEFAULT_WORKDIR
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
        --skip-storage-probe)
            STORAGE_PROBE=0
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

CONFIG="$(readlink -f "$CONFIG")"
if [[ "$WORKDIR" != /* ]]; then
    WORKDIR="$(pwd -P)/$WORKDIR"
fi
WORKDIR="$(realpath -m "$WORKDIR")"
SOURCE_SEGMENT="$WORKDIR/segments/$SOURCE_SEGMENT_NAME"
SEGMENT="$WORKDIR/segments/$DERIVED_SEGMENT_NAME"
FRONTEND="$WORKDIR/frontend_artifacts/$FRONTEND_NAME"
BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
POSE="$WORKDIR/pose_artifacts/$POSE_NAME"
CLOUD="$WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz"
TRAIN="$WORKDIR/runs/$TRAIN_NAME"
LOG_DIR="$WORKDIR/logs"
STATE_DIR="$WORKDIR/run_state"

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

verify_required_files() {
    [[ -r "$CONFIG" ]] || die "missing config: $CONFIG"
    [[ -r "$ROBOT_CONFIG" ]] || die "missing robot profile: $ROBOT_CONFIG"
    [[ -d "$DATASET_ROOT" ]] || die "CitrusFarm is not mounted at $DATASET_ROOT"
    [[ -r "$GT_CSV" ]] || die "missing evaluation trajectory: $GT_CSV"

    local -a camera_bags gnss_bags
    mapfile -d '' camera_bags < <(
        find "$SEQUENCE_ROOT" -maxdepth 1 -type f -name 'zed_*.bag' -print0 \
            | sort -zV
    )
    mapfile -d '' gnss_bags < <(
        find "$SEQUENCE_ROOT" -maxdepth 1 -type f -name 'base_*.bag' -print0 \
            | sort -zV
    )
    [[ ${#camera_bags[@]} -eq 27 ]] \
        || die "expected 27 ordered ZED chunks, found ${#camera_bags[@]}"
    [[ ${#gnss_bags[@]} -eq 2 ]] \
        || die "expected 2 ordered base/GNSS chunks, found ${#gnss_bags[@]}"
    local index suffix
    for index in "${!camera_bags[@]}"; do
        suffix="${camera_bags[$index]##*_}"
        suffix="${suffix%.bag}"
        [[ "$suffix" == "$index" ]] \
            || die "ZED chunk ordering is incomplete at index $index: ${camera_bags[$index]}"
        [[ -r "${camera_bags[$index]}" && -s "${camera_bags[$index]}" ]] \
            || die "unreadable/empty ZED chunk: ${camera_bags[$index]}"
    done
    local bag
    for bag in "${gnss_bags[@]}"; do
        [[ -r "$bag" && -s "$bag" ]] || die "unreadable/empty GNSS bag: $bag"
    done
}

verify_mount_and_space() {
    local source target fstype options
    source="$(findmnt -n -o SOURCE -T "$DATASET_ROOT")" \
        || die "cannot resolve the Buffalo source device"
    target="$(findmnt -n -o TARGET -T "$DATASET_ROOT")" \
        || die "cannot resolve the Buffalo mount point"
    fstype="$(findmnt -n -o FSTYPE -T "$DATASET_ROOT")" \
        || die "cannot resolve the Buffalo filesystem"
    options="$(findmnt -n -o OPTIONS -T "$DATASET_ROOT")" \
        || die "cannot resolve the Buffalo mount options"
    [[ "$source" == /dev/* ]] || die "dataset source is not a block-device mount: $source"
    [[ "$fstype" == "exfat" ]] || die "unexpected Buffalo filesystem: $fstype"
    [[ ",$options," != *,ro,* ]] || die "Buffalo unexpectedly mounted read-only"

    case "$WORKDIR/" in
        "$DATASET_ROOT"/*)
            die "workdir must be on internal storage, not inside the Buffalo dataset"
            ;;
    esac
    local output_parent available_bytes required_bytes output_source
    output_parent="$(nearest_existing_parent "$WORKDIR")"
    output_source="$(findmnt -n -o SOURCE -T "$output_parent")"
    [[ "$output_source" != "$source" ]] \
        || die "workdir resolves to the Buffalo source drive; choose internal storage"
    available_bytes="$(df -PB1 "$output_parent" | awk 'NR==2 {print $4}')"
    required_bytes=$((MIN_OUTPUT_FREE_GIB * 1024 * 1024 * 1024))
    ((available_bytes >= required_bytes)) || die \
        "need at least ${MIN_OUTPUT_FREE_GIB} GiB free on output filesystem"

    echo "Buffalo mount: $source -> $target ($fstype)"
    echo "Output free: $((available_bytes / 1024 / 1024 / 1024)) GiB on $output_source"

    local disk sys_path speed="unknown"
    disk="$(lsblk -no PKNAME "$source" 2>/dev/null | head -1)"
    [[ -n "$disk" ]] || disk="${source#/dev/}"
    sys_path="$(readlink -f "/sys/class/block/$disk/device" 2>/dev/null || true)"
    while [[ -n "$sys_path" && "$sys_path" != "/" ]]; do
        if [[ -r "$sys_path/speed" ]]; then
            speed="$(<"$sys_path/speed")"
            break
        fi
        sys_path="${sys_path%/*}"
        [[ -n "$sys_path" ]] || sys_path="/"
    done
    echo "USB negotiated speed: ${speed} Mbit/s (USB2/480 is expected for the stable cable)"
}

probe_storage() {
    ((STORAGE_PROBE)) || { echo "Storage probe explicitly skipped."; return; }
    local start_time bag size blocks skip
    start_time="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "Read-only Buffalo probe: three 32 MiB samples..."
    for bag in \
        "$SEQUENCE_ROOT/zed_2023-07-16-17-13-51_18.bag" \
        "$SEQUENCE_ROOT/zed_2023-07-16-17-15-18_21.bag" \
        "$SEQUENCE_ROOT/zed_2023-07-16-17-17-13_25.bag"; do
        size="$(stat -c '%s' "$bag")"
        blocks=$((size / 4194304))
        skip=$((blocks / 2))
        dd if="$bag" of=/dev/null bs=4M skip="$skip" count=8 \
            iflag=direct status=none \
            || die "Buffalo read probe failed: $bag"
    done
    if journalctl -k --since "$start_time" --no-pager -q >/dev/null 2>&1; then
        local errors
        errors="$(journalctl -k --since "$start_time" --no-pager -q \
            | grep -Ei '(I/O error|Buffer I/O error|blk_update_request|DID_ERROR|uas_eh_device_reset|USB disconnect|device offline|reset (high-speed|SuperSpeed) USB device)' \
            || true)"
        [[ -z "$errors" ]] || die "fresh storage/kernel error during probe: $errors"
    else
        echo "WARNING: kernel journal is not readable; dd probe passed but kernel errors could not be audited."
    fi
    echo "Buffalo read probe passed."
}

verify_environment() {
    PYTHON="$(resolve_executable "$PYTHON")"
    COLMAP="$(resolve_executable "$COLMAP")"
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export COLMAP_BIN="$COLMAP"
    "$PYTHON" - <<'PY'
import importlib
import pathlib

required = ("cv2", "gsplat", "numpy", "pymap3d", "rosbags", "torch", "yaml")
for name in required:
    importlib.import_module(name)
import rtk_splat
import torch
repo = pathlib.Path.cwd().resolve()
package = pathlib.Path(rtk_splat.__file__).resolve()
if repo not in package.parents:
    raise SystemExit(f"rtk_splat resolves outside this checkout: {package}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
print(f"Python package: {package}")
print(f"PyTorch CUDA: {torch.cuda.get_device_name(0)}")
PY
    "$COLMAP" -h 2>&1 | grep -q 'COLMAP 4\.1' \
        || die "this reproduction requires the tested COLMAP 4.1 build"
    command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
    nvidia-smi --query-gpu=name,memory.total,memory.free \
        --format=csv,noheader
}

verify_config() {
    "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
import numpy as np
from rtk_splat.core.configio import load_config

cfg = load_config(sys.argv[1])
if cfg.adapter != "ros1_citrusfarm":
    raise SystemExit(f"wrong adapter: {cfg.adapter}")
if list(map(float, cfg.segment.window_s)) != [543.0, 735.0]:
    raise SystemExit("window must remain [543,735] seconds")
if cfg.segment.window_epoch_source != "first_gnss_log":
    raise SystemExit("window epoch must be first_gnss_log")
if float(cfg.segment.frame_spacing_m) != 0.15:
    raise SystemExit("frame spacing must remain 0.15 m")
if len(cfg.paths.camera_bags) != 27 or len(cfg.paths.gnss_bags) != 2:
    raise SystemExit("config does not declare the complete ordered bag chain")
if cfg.pose.source != "gnss_course":
    raise SystemExit("Citrus primary method must use pose.source=gnss_course")
if cfg.topics.receiver_state != "/piksi/debug/receiver_state":
    raise SystemExit("Citrus receiver-state evidence topic changed")
if cfg.gnss_quality.receiver_state_required is not True:
    raise SystemExit("Citrus receiver-state evidence must be required")
if cfg.gnss_quality.covariance_provenance != "driver_static_nominal":
    raise SystemExit("Citrus covariance provenance must remain explicit")
if cfg.gnss_quality.covariance_is_live_per_epoch is not False:
    raise SystemExit("static Citrus covariance must not be labelled live")
if int(cfg.pose.minimum_position_carrier_status) != 2:
    raise SystemExit("Citrus pose construction must require receiver-fixed RTK")
if int(cfg.frontend.pose_priors.min_carrier_status) != 2:
    raise SystemExit("Citrus COLMAP priors must require receiver-fixed RTK")
if abs(float(cfg.pose.time_offset_s) - 0.072548749) > 1e-12:
    raise SystemExit("audited sequence clock correction must remain +0.072548749 s")
if cfg.mapper.backend != "global" or cfg.mapper.name != "citrus-05-13d-543-735-global-bounded-v2":
    raise SystemExit("unexpected mapper backend/artifact")
if cfg.depth.backend != "sgbm":
    raise SystemExit("this primary reproduction requires an immutable SGBM depth segment")
if cfg.mapper.rtk_covariance_gate_mode != "diagnostic_only":
    raise SystemExit("static approximate covariance cannot enforce the chi-square gate")
t = np.asarray(cfg.sensor_geometry.T_camera_primary_antenna, dtype=float)
if t.shape != (4, 4) or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-8):
    raise SystemExit("camera/GPS transform is not rigid")
expected_t = np.asarray([
    [0.026234841320, -0.999648165195, 0.003908826209, 0.060535141536],
    [-0.009163740751, -0.004150497990, -0.999949398331, -0.309152932606],
    [0.999613804905, 0.026197694323, -0.009269404284, -0.424344392950],
    [0.0, 0.0, 0.0, 1.0],
])
if not np.allclose(t, expected_t, atol=1e-12, rtol=0.0):
    raise SystemExit("Citrus official camera/GPS transform changed")
if cfg.sensor_geometry.extrinsic_camera_geometry != "raw_left":
    raise SystemExit("published calibration must be declared in raw_left geometry")
print(json.dumps({
    "adapter": cfg.adapter,
    "camera_bags": len(cfg.paths.camera_bags),
    "gnss_bags": len(cfg.paths.gnss_bags),
    "window_s": cfg.segment.window_s,
    "window_epoch_source": cfg.segment.window_epoch_source,
    "frame_spacing_m": cfg.segment.frame_spacing_m,
    "clock_offset_s": cfg.pose.time_offset_s,
    "receiver_state_required": cfg.gnss_quality.receiver_state_required,
    "covariance_provenance": cfg.gnss_quality.covariance_provenance,
    "camera_gps_lever_norm_m": float(np.linalg.norm(t[:3, 3])),
}, indent=2))
PY
}

adapter_preflight() {
    # The adapter owns ROS1/chunk/timestamp semantics. Its preflight samples
    # headers/log stamps without publishing images or starting COLMAP.
    "$PYTHON" - "$CONFIG" <<'PY'
import dataclasses
import json
import sys
from rtk_splat.core.configio import load_config
from rtk_splat.adapters.ros1_citrusfarm import preflight_config

report = preflight_config(load_config(sys.argv[1]))
if dataclasses.is_dataclass(report):
    report = dataclasses.asdict(report)
if not isinstance(report, dict):
    raise SystemExit("Citrus adapter preflight returned no report mapping")
if report.get("passed") is not True:
    raise SystemExit("Citrus adapter timestamp/chunk preflight did not pass")
quality = report.get("gnss_quality", {})
receiver = quality.get("receiver_state") or {}
if receiver.get("associated_navsatfix_count") != receiver.get("navsatfix_count"):
    raise SystemExit("receiver-state evidence is incomplete")
if set(receiver.get("fix_mode_counts", {})) != {"FIXED_RTK"}:
    raise SystemExit("selected Citrus evidence is not entirely receiver-fixed RTK")
covariance = quality.get("covariance") or {}
if covariance.get("configured_provenance") != "driver_static_nominal":
    raise SystemExit("Citrus covariance provenance was not retained")
if covariance.get("is_live_per_epoch") is not False:
    raise SystemExit("Citrus static covariance was incorrectly labelled live")
def chain_summary(chain):
    gaps = [int(value) for value in chain["gaps_ns"]]
    return {
        "chunks": len(chain["chunks"]),
        "largest_observed_gap_ns": max([0, *gaps]),
        "largest_observed_overlap_ns": max([0, *(-value for value in gaps)]),
    }
print(json.dumps({
    "passed": True,
    "camera_chain": chain_summary(report["camera_chain"]),
    "gnss_chain": chain_summary(report["gnss_chain"]),
    "window": report["window"],
    "clock": report["clock"],
    "gnss_quality": quality,
    "stereo": report["stereo"],
    "extrinsic_resolution": report["extrinsic_resolution"],
    "sampling": report["sampling"],
    "optional_recorded_inputs": report["optional_recorded_inputs"],
}, indent=2, sort_keys=True))
PY
}

preflight() {
    cd "$REPO_ROOT"
    verify_required_files
    verify_mount_and_space
    verify_environment
    verify_config
    probe_storage
    adapter_preflight
    echo "Preflight passed; no artifact was created."
}

if [[ "$ACTION" == "preflight" ]]; then
    preflight
    exit 0
fi
[[ "$ACTION" == "run" ]] || die "unsupported action: $ACTION"

preflight

if [[ -e "$WORKDIR" ]]; then
    ((RESUME)) || die \
        "workdir already exists; choose a new path or pass --resume-existing"
else
    ((RESUME == 0)) || die "--resume-existing requires an existing workdir"
    mkdir -p "$WORKDIR"
fi
mkdir -p "$LOG_DIR" "$STATE_DIR"

CONFIG_FINGERPRINT="$({
    sha256sum "$CONFIG" "$ROBOT_CONFIG" "$SCRIPT_PATH"
    find "$REPO_ROOT/rtk_splat" -type f -name '*.py' -print0 \
        | sort -z | xargs -0 sha256sum
    git -C "$REPO_ROOT" rev-parse HEAD
    git -C "$REPO_ROOT" diff --no-ext-diff --binary
} | sha256sum | awk '{print $1}')"
export RTK_SPLAT_CONFIG_SHA256="$CONFIG_FINGERPRINT"

command_digest() {
    printf '%q\000' "$@" | sha256sum | awk '{print $1}'
}

verify_outputs() {
    local stage="$1"
    local -a required=()
    case "$stage" in
        ingest)
            required=("$SOURCE_SEGMENT/segment_meta.json" \
                "$SOURCE_SEGMENT/manifest.json" "$SOURCE_SEGMENT/frames.npz" \
                "$SOURCE_SEGMENT/calibration.json") ;;
        depth)
            required=("$SEGMENT/segment_meta.json" "$SEGMENT/manifest.json" \
                "$SEGMENT/frames.npz" "$SEGMENT/calibration.json") ;;
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
            required=("$FRONTEND/frontend_seal.json" "$FRONTEND/stages/matching.json" \
                "$FRONTEND/stage_reports/matching.json") ;;
        backend-prepare)
            required=("$BACKEND/backend_plan.json" "$BACKEND/database.db" \
                "$BACKEND/stages/prepare.json") ;;
        backend-solve)
            required=("$BACKEND/stages/solve.json" "$BACKEND/reports/solve.json" \
                "$BACKEND/solve_model_manifest.json") ;;
        backend-register)
            required=("$BACKEND/stages/register.json" "$BACKEND/reports/register.json" \
                "$BACKEND/registered_model_manifest.json") ;;
        backend-quality)
            required=("$BACKEND/stages/quality.json" "$BACKEND/reports/quality.json" \
                "$BACKEND/text_model_manifest.json") ;;
        backend-export)
            required=("$POSE/viewmats.npy" "$POSE/cam_centers.npy" \
                "$POSE/quality.json") ;;
        cloud)
            required=("$CLOUD") ;;
        train)
            required=("$TRAIN/metrics.json" "$TRAIN/params.pt" "$TRAIN/splat.ply" \
                "$TRAIN/run_provenance.json") ;;
        *) die "unknown stage verifier: $stage" ;;
    esac
    local path
    for path in "${required[@]}"; do
        [[ -s "$path" ]] || return 1
    done
}

run_stage() {
    local stage="$1"
    shift
    local marker="$STATE_DIR/$stage.done"
    local digest log timestamp
    digest="$(command_digest "$@")"
    if [[ -f "$marker" ]]; then
        ((RESUME)) || die "stage marker already exists without --resume-existing: $marker"
        local marked_config marked_command
        IFS=' ' read -r marked_config marked_command < "$marker"
        [[ "$marked_config" == "$CONFIG_FINGERPRINT" ]] \
            || die "stage $stage was created by a different code/config snapshot"
        [[ "$marked_command" == "$digest" ]] \
            || die "stage $stage command changed; use a new workdir/artifact name"
        verify_outputs "$stage" \
            || die "stage $stage marker exists but required outputs are incomplete"
        echo "RESUME: verified completed stage $stage"
        return
    fi

    timestamp="$(date '+%Y%m%d_%H%M%S')"
    log="$LOG_DIR/${stage}_${timestamp}.log"
    echo "RUN: $stage"
    printf '  %q' "$@"
    printf '\n  log: %s\n' "$log"
    "$@" 2>&1 | tee "$log"
    verify_outputs "$stage" || die "stage $stage returned success but outputs are incomplete"
    (set -o noclobber; printf '%s %s\n' "$CONFIG_FINGERPRINT" "$digest" > "$marker") \
        || die "refusing to replace stage marker: $marker"
}

CLI=("$PYTHON" -m rtk_splat.workflows.cli)
SOURCE_COMMON=(--config "$CONFIG" --workdir "$WORKDIR" --segment "$SOURCE_SEGMENT")
COMMON=(--config "$CONFIG" --workdir "$WORKDIR" --segment "$SEGMENT")

run_stage ingest \
    "${CLI[@]}" ingest "${SOURCE_COMMON[@]}"
"${CLI[@]}" validate "${SOURCE_COMMON[@]}"
run_stage depth \
    "${CLI[@]}" depth "${SOURCE_COMMON[@]}" --derived-segment "$SEGMENT"
"${CLI[@]}" validate "${COMMON[@]}"
"$PYTHON" - "$SEGMENT" <<'PY'
import sys
from rtk_splat.core.segment import SegmentReader
reader = SegmentReader(sys.argv[1]).validate()
caps = reader.meta["capabilities"]
if not (caps["stereo"] and caps["single_rtk"] and caps["depth_computed"]):
    raise SystemExit(f"required Citrus capabilities absent: {caps}")
if caps["dual_rtk"]:
    raise SystemExit("Citrus sequence unexpectedly declares dual RTK")
print(f"Measured segment size: {reader.meta['n_frames']} stereo pairs")
PY

run_stage frontend-build \
    "${CLI[@]}" frontend-build "${COMMON[@]}" \
    --frontend-name "$FRONTEND_NAME" --keyframe-preset all --feature-profile gpu
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
run_stage backend-export \
    "${CLI[@]}" backend-export "${COMMON[@]}" \
    --backend global --backend-name "$BACKEND_NAME" --pose-name "$POSE_NAME"

run_stage cloud \
    "${CLI[@]}" cloud "${COMMON[@]}" --pose-name "$POSE_NAME"
run_stage train \
    "${CLI[@]}" train "${COMMON[@]}" \
    --pose-name "$POSE_NAME" --run-name "$TRAIN_NAME"

"$PYTHON" - "$TRAIN/metrics.json" "$BACKEND/reports/quality.json" <<'PY'
import json
import sys
metrics = json.load(open(sys.argv[1]))[-1]
quality = json.load(open(sys.argv[2]))
summary = {
    "registration_fraction": quality["registration_fraction"],
    "quality_passed": quality["passed"],
    "psnr": metrics["psnr"],
    "psnr_masked": metrics["psnr_masked"],
    "psnr_masked_cc": metrics["psnr_masked_cc"],
    "ssim": metrics["ssim"],
    "lpips": metrics["lpips"],
    "lpips_cc": metrics["lpips_cc"],
}
if "psnr_masked_aligned" in metrics:
    summary["psnr_masked_aligned"] = metrics["psnr_masked_aligned"]
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "COMPLETE: $WORKDIR"
