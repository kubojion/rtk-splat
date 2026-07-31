#!/usr/bin/env bash
# Shared guarded runner for the sealed-frontend experiments. Source this file
# from a wrapper that defines the AB_* arrays below; do not invoke it directly.

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly AB_EXPECTED_FRAMES=1344
readonly AB_EXPECTED_TRAIN=1176
readonly AB_EXPECTED_VAL=168
readonly AB_PROXY_ITERS=15000
readonly AB_FULL_ITERS=65000

AB_PHASE="plan"
AB_CONFIG="${RTK_SPLAT_CONFIG:-}"
AB_SEGMENT_OVERRIDE="${RTK_SPLAT_SEGMENT:-}"
AB_EXPERIMENT_ROOT="${RTK_SPLAT_EXPERIMENT_ROOT:-}"
AB_PYTHON="${RTK_SPLAT_PYTHON:-python3}"
AB_COLMAP="${COLMAP_BIN:-${RTK_SPLAT_COLMAP:-colmap}}"
AB_WINNER=""
AB_RESUME=0
AB_ACCEPT_POSE=0
AB_ACCEPT_PROXY=0
AB_SEGMENT=""
AB_CONFIG_WORKDIR=""
AB_CONFIG_SHA256=""
AB_LOG_DIR=""

ab_die() {
    echo "FATAL: $*" >&2
    exit 1
}

ab_on_error() {
    local status=$?
    echo "FATAL: ${AB_KIND:-experiment} stopped at line ${BASH_LINENO[0]} (status $status)." >&2
    echo "No later arm or phase was started. Inspect ${AB_LOG_DIR:-the experiment logs}." >&2
    exit "$status"
}
trap ab_on_error ERR

ab_join() {
    local separator="$1"
    shift
    local output=""
    local value
    for value in "$@"; do
        if [[ -n "$output" ]]; then
            output+="$separator"
        fi
        output+="$value"
    done
    printf '%s' "$output"
}

ab_usage() {
    local winners
    winners="$(ab_join "|" "${AB_ARMS[@]}")"
    cat <<EOF
Usage:
  $(basename "$0") [plan]
  $(basename "$0") pose  --config CFG --experiment-root NEW_DIR [options]
  $(basename "$0") pose  --config CFG --experiment-root DIR --resume-existing
  $(basename "$0") proxy --config CFG --experiment-root DIR --resume-existing \\
      --accept-pose-gates
  $(basename "$0") full  --config CFG --experiment-root DIR --resume-existing \\
      --winner {$winners} --accept-proxy-gates

Phases:
  plan    Print the arms, measured reference timings, and manual gates. No I/O.
  pose    Build each sealed frontend, solve with Global Mapper, register every
          non-keyframe, run quality gates, and export exactly 1,344 poses.
  proxy   After manual pose acceptance, build pose-matched clouds and run one
          15,000-presentation GS proxy for every arm.
  full    After manual proxy acceptance, run one 65,000-presentation GS job for
          the explicitly selected winner only.

Options:
  --config PATH            Config whose immutable v2 segment and GS settings
                           are shared by every arm.
  --segment PATH           Explicit immutable contract-v2 segment. This
                           overrides paths.segment in the config.
  --experiment-root PATH   New experiment directory for pose; the exact same
                           directory for later phases.
  --python PATH            Python executable (default: \$RTK_SPLAT_PYTHON or python3).
  --colmap PATH            COLMAP executable (default: \$COLMAP_BIN or colmap).
  --resume-existing        Explicitly permit a known experiment root. Required
                           for proxy/full and a pose-stage retry. Each stage
                           still refuses unsafe or unverifiable partial output.
  --accept-pose-gates      Confirm that pose-only reports were reviewed.
  --accept-proxy-gates     Confirm that all proxy metrics were reviewed.
  --winner NAME            Required only for full; one of: $winners.
  -h, --help               Show this message.

The script runs in the foreground, without tmux, and writes one log per stage.
It never selects a winner or advances across a manual gate automatically.
EOF
}

ab_print_plan() {
    local index
    echo "${AB_TITLE}"
    echo
    echo "Arms:"
    for index in "${!AB_ARMS[@]}"; do
        printf '  %-12s features=%-14s keyframes=%-9s reference solve frames=%s\n' \
            "${AB_ARMS[$index]}" \
            "${AB_FEATURE_PROFILES[$index]}" \
            "${AB_KEYFRAME_PRESETS[$index]}" \
            "${AB_REFERENCE_KEYFRAMES[$index]}"
    done
    cat <<'EOF'

Manual sequence:
  1. pose: all arms, no GS
  2. inspect registration/reprojection/RTK/scale/pose-delta reports
  3. proxy: all accepted arms at exactly 15,000 presentations
  4. inspect equal-presentation image metrics and runtime/storage
  5. full: exactly one explicitly named winner at 65,000 presentations

Measured headland reference timings (not extrapolated promises):
  GPU all-frame SIFT feature extraction:    ~6 min
  CPU-reference all-frame SIFT extraction:  62.7 min
  all-frame sequential matching:             18.7 min
  all-frame Global Mapper:                    18.1 min
  incremental mapper reference:              581.0 min (not used here)
  one 15k GS proxy:                          ~1.2 h
  one 65k GS winner:                         ~245 min

Adaptive matching, registration, and mapper times are intentionally reported as
unknown until measured. The script does not invent a speedup from frame counts.
EOF
}

ab_resolve_executable() {
    local requested="$1"
    local resolved
    if [[ "$requested" == */* ]]; then
        [[ -x "$requested" ]] || ab_die "executable is unavailable: $requested"
        resolved="$(readlink -f "$requested")"
    else
        resolved="$(command -v "$requested" || true)"
        [[ -n "$resolved" && -x "$resolved" ]] ||
            ab_die "executable is unavailable: $requested"
        resolved="$(readlink -f "$resolved")"
    fi
    printf '%s\n' "$resolved"
}

ab_parse() {
    if (($#)) && [[ "$1" != -* ]]; then
        AB_PHASE="$1"
        shift
    fi
    case "$AB_PHASE" in
        plan|pose|proxy|full) ;;
        *) ab_die "unknown phase '$AB_PHASE'; use plan, pose, proxy, or full" ;;
    esac
    while (($#)); do
        case "$1" in
            --config)
                (($# >= 2)) || ab_die "--config requires a path"
                AB_CONFIG="$2"
                shift 2
                ;;
            --experiment-root)
                (($# >= 2)) || ab_die "--experiment-root requires a path"
                AB_EXPERIMENT_ROOT="$2"
                shift 2
                ;;
            --segment)
                (($# >= 2)) || ab_die "--segment requires a path"
                AB_SEGMENT_OVERRIDE="$2"
                shift 2
                ;;
            --python)
                (($# >= 2)) || ab_die "--python requires a path"
                AB_PYTHON="$2"
                shift 2
                ;;
            --colmap)
                (($# >= 2)) || ab_die "--colmap requires a path"
                AB_COLMAP="$2"
                shift 2
                ;;
            --winner)
                (($# >= 2)) || ab_die "--winner requires an arm name"
                AB_WINNER="$2"
                shift 2
                ;;
            --resume-existing)
                AB_RESUME=1
                shift
                ;;
            --accept-pose-gates)
                AB_ACCEPT_POSE=1
                shift
                ;;
            --accept-proxy-gates)
                AB_ACCEPT_PROXY=1
                shift
                ;;
            -h|--help)
                ab_usage
                exit 0
                ;;
            *)
                ab_die "unknown option: $1"
                ;;
        esac
    done
}

ab_check_definition() {
    local count="${#AB_ARMS[@]}"
    ((count >= 2)) || ab_die "experiment needs at least two arms"
    ((${#AB_FEATURE_PROFILES[@]} == count)) ||
        ab_die "internal feature-profile arm mismatch"
    ((${#AB_KEYFRAME_PRESETS[@]} == count)) ||
        ab_die "internal keyframe-preset arm mismatch"
    ((${#AB_REFERENCE_KEYFRAMES[@]} == count)) ||
        ab_die "internal reference-count arm mismatch"
    local value
    for value in "${AB_ARMS[@]}"; do
        [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] ||
            ab_die "unsafe internal arm name: $value"
    done
}

ab_validate_cli() {
    local help
    help="$("$AB_PYTHON" -m workflows.cli --help)"
    local stage
    for stage in \
        frontend-build frontend-features frontend-rig frontend-priors \
        frontend-match backend-prepare backend-solve backend-register \
        backend-quality backend-export cloud train; do
        [[ "$help" == *"$stage"* ]] ||
            ab_die "workflows CLI does not provide required stage: $stage"
    done
    for option in \
        --segment --workdir --frontend-name --feature-profile \
        --keyframe-preset --backend-name --pose-name --run-name \
        --train-iters --expected-frames; do
        [[ "$help" == *"$option"* ]] ||
            ab_die "workflows CLI does not provide required option: $option"
    done
}

ab_validate_tools() {
    local help banner required
    if [[ "$AB_PHASE" == "pose" ]]; then
        help="$("$AB_COLMAP" -h 2>&1)"
        banner="${help%%$'\n'*}"
        [[ "$banner" == "COLMAP 4.1.1"*"(Commit "*" with CUDA)" ]] ||
            ab_die "expected the measured COLMAP 4.1.1 CUDA build; got: $banner"
        for required in feature_extractor matches_importer global_mapper \
            image_registrator model_analyzer model_converter; do
            [[ "$help" == *"$required"* ]] ||
                ab_die "COLMAP build lacks required command: $required"
        done
    fi
    command -v nvidia-smi >/dev/null 2>&1 ||
        ab_die "nvidia-smi is unavailable; these controlled arms require CUDA"
    if [[ "$AB_PHASE" == "proxy" || "$AB_PHASE" == "full" ]]; then
        "$AB_PYTHON" - <<'PY'
import gsplat
import torch
import torchmetrics

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
PY
    fi
}

ab_read_segment() {
    local info
    mapfile -t info < <(
        "$AB_PYTHON" - "$AB_CONFIG" "$AB_EXPECTED_FRAMES" \
            "$AB_EXPECTED_TRAIN" "$AB_EXPECTED_VAL" \
            "$AB_SEGMENT_OVERRIDE" <<'PY'
import sys
from pathlib import Path

from rtk_splat.configio import load_config
from rtk_splat.segment import SegmentReader

cfg = load_config(sys.argv[1])
configured = getattr(cfg.paths, "segment", None)
segment = (
    Path(sys.argv[5]).expanduser()
    if sys.argv[5]
    else (
        Path(configured).expanduser()
        if configured is not None
        else Path(cfg.paths.workdir).expanduser() / "segment"
    )
).resolve()
reader = SegmentReader(segment).validate()
expected, expected_train, expected_val = map(int, sys.argv[2:])
if reader.meta["n_frames"] != expected:
    raise SystemExit(
        f"expected {expected} headland frames, found {reader.meta['n_frames']}"
    )
manifest = reader.manifest
counts = tuple(len(manifest[name]) for name in ("train", "val", "test"))
if counts != (expected_train, expected_val, 0):
    raise SystemExit(
        f"expected train/val/test {expected_train}/{expected_val}/0, found "
        f"{counts[0]}/{counts[1]}/{counts[2]}"
    )
if not reader.meta["capabilities"]["stereo"]:
    raise SystemExit("experiment requires stereo capability")
if not reader.meta["capabilities"]["images_rectified"]:
    raise SystemExit("experiment requires rectified images")
print(segment)
print(counts[0])
print(counts[1])
print(Path(cfg.paths.workdir).expanduser().resolve())
PY
    )
    ((${#info[@]} == 4)) || ab_die "could not resolve the canonical segment"
    AB_SEGMENT="${info[0]}"
    AB_CONFIG_WORKDIR="${info[3]}"
    "$AB_PYTHON" - "$AB_CONFIG" <<'PY'
import sys
from rtk_splat.configio import load_config

cfg = load_config(sys.argv[1])
if getattr(cfg.pose, "artifact_root", None) is not None:
    raise SystemExit(
        "A/B config must not override pose.artifact_root; arm isolation owns it"
    )
cloud = getattr(cfg, "cloud", None)
if cloud is not None and getattr(cloud, "artifact_root", None) is not None:
    raise SystemExit(
        "A/B config must not override cloud.artifact_root; arm isolation owns it"
    )
PY
}

ab_write_manifest() {
    "$AB_PYTHON" - \
        "$AB_EXPERIMENT_ROOT/experiment.json" \
        "$AB_KIND" "$AB_CONFIG" "$AB_CONFIG_SHA256" "$AB_SEGMENT" \
        "$(ab_join "," "${AB_ARMS[@]}")" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
record = {
    "schema_version": 1,
    "experiment": sys.argv[2],
    "config_path": str(Path(sys.argv[3]).resolve()),
    "config_sha256": sys.argv[4],
    "segment": str(Path(sys.argv[5]).resolve()),
    "expected_frames": 1344,
    "expected_train_frames": 1176,
    "expected_val_frames": 168,
    "arms": sys.argv[6].split(","),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "manual_gates": ["pose", "proxy", "one_full_winner"],
}
payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
with os.fdopen(descriptor, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
}

ab_validate_manifest() {
    "$AB_PYTHON" - \
        "$AB_EXPERIMENT_ROOT/experiment.json" \
        "$AB_KIND" "$AB_CONFIG_SHA256" "$AB_SEGMENT" \
        "$(ab_join "," "${AB_ARMS[@]}")" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"existing output is not a recognized experiment: {path}")
record = json.loads(path.read_text())
expected = {
    "experiment": sys.argv[2],
    "config_sha256": sys.argv[3],
    "segment": str(Path(sys.argv[4]).resolve()),
    "expected_frames": 1344,
    "expected_train_frames": 1176,
    "expected_val_frames": 168,
    "arms": sys.argv[5].split(","),
}
for key, value in expected.items():
    if record.get(key) != value:
        raise SystemExit(
            f"existing experiment mismatch for {key}: "
            f"{record.get(key)!r} != {value!r}"
        )
PY
}

ab_prepare() {
    [[ -n "$AB_CONFIG" ]] || ab_die "--config is required for $AB_PHASE"
    [[ -f "$AB_CONFIG" ]] || ab_die "config does not exist: $AB_CONFIG"
    AB_CONFIG="$(readlink -f "$AB_CONFIG")"
    AB_PYTHON="$(ab_resolve_executable "$AB_PYTHON")"
    if [[ "$AB_PHASE" == "pose" ]]; then
        AB_COLMAP="$(ab_resolve_executable "$AB_COLMAP")"
        export COLMAP_BIN="$AB_COLMAP"
    else
        AB_COLMAP="not used in $AB_PHASE phase"
        unset COLMAP_BIN
    fi
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    cd "$AB_REPO_ROOT"
    ab_validate_cli
    ab_validate_tools
    ab_read_segment
    # Bind the experiment to the fully merged robot+sequence configuration,
    # not just to the small sequence YAML that names the robot profile.
    AB_CONFIG_SHA256="$(
        "$AB_PYTHON" - "$AB_CONFIG" <<'PY'
import hashlib
import json
import sys

from rtk_splat.configio import load_config
from workflows.cli import _plain

resolved = _plain(load_config(sys.argv[1]))
payload = json.dumps(
    resolved,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode()
print(hashlib.sha256(payload).hexdigest())
PY
    )"
    export RTK_SPLAT_CONFIG_SHA256="$AB_CONFIG_SHA256"

    [[ -n "$AB_EXPERIMENT_ROOT" ]] ||
        ab_die "--experiment-root is required for $AB_PHASE"
    AB_EXPERIMENT_ROOT="$(realpath -m "$AB_EXPERIMENT_ROOT")"
    [[ "$AB_EXPERIMENT_ROOT" != "/" ]] ||
        ab_die "experiment root cannot be /"
    [[ "$AB_EXPERIMENT_ROOT" != "$AB_REPO_ROOT" &&
       "$AB_EXPERIMENT_ROOT" != "$AB_REPO_ROOT/"* ]] ||
        ab_die "experiment outputs must stay outside the git repository"
    [[ "$AB_EXPERIMENT_ROOT" != "$AB_SEGMENT" &&
       "$AB_EXPERIMENT_ROOT" != "$AB_SEGMENT/"* &&
       "$AB_SEGMENT" != "$AB_EXPERIMENT_ROOT/"* ]] ||
        ab_die "experiment root and immutable segment must be separate trees"
    [[ "$AB_EXPERIMENT_ROOT" != "$AB_CONFIG_WORKDIR" &&
       "$AB_EXPERIMENT_ROOT" != "$AB_CONFIG_WORKDIR/"* &&
       "$AB_CONFIG_WORKDIR" != "$AB_EXPERIMENT_ROOT/"* ]] ||
        ab_die "experiment root must not modify the config's historical workdir"

    if [[ "$AB_PHASE" == "pose" && ! -e "$AB_EXPERIMENT_ROOT" ]]; then
        mkdir -p "$(dirname "$AB_EXPERIMENT_ROOT")"
        mkdir "$AB_EXPERIMENT_ROOT"
        ab_write_manifest
    else
        [[ -d "$AB_EXPERIMENT_ROOT" ]] ||
            ab_die "experiment root does not exist: $AB_EXPERIMENT_ROOT"
        ((AB_RESUME == 1)) ||
            ab_die "refusing existing output without --resume-existing"
        ab_validate_manifest
    fi
    AB_LOG_DIR="$AB_EXPERIMENT_ROOT/logs"
    mkdir -p "$AB_LOG_DIR"
}

ab_arm_index() {
    local wanted="$1"
    local index
    for index in "${!AB_ARMS[@]}"; do
        if [[ "${AB_ARMS[$index]}" == "$wanted" ]]; then
            printf '%s\n' "$index"
            return 0
        fi
    done
    return 1
}

ab_frontend_name() {
    printf '%s-%s-frontend\n' "$AB_KIND" "$1"
}

ab_backend_name() {
    printf '%s-%s-global\n' "$AB_KIND" "$1"
}

ab_pose_name() {
    printf '%s-%s-pose\n' "$AB_KIND" "$1"
}

ab_arm_workdir() {
    printf '%s/arms/%s\n' "$AB_EXPERIMENT_ROOT" "$1"
}

ab_run_cli() {
    local arm="$1"
    local stage="$2"
    local profile="$3"
    local preset="$4"
    shift 4
    local workdir frontend backend pose log start end status
    local pipeline_statuses=()
    workdir="$(ab_arm_workdir "$arm")"
    frontend="$(ab_frontend_name "$arm")"
    backend="$(ab_backend_name "$arm")"
    pose="$(ab_pose_name "$arm")"
    mkdir -p "$workdir"
    log="$AB_LOG_DIR/$arm-$stage.log"
    local command=(
        "$AB_PYTHON" -m workflows.cli "$stage"
        --config "$AB_CONFIG"
        --segment "$AB_SEGMENT"
        --workdir "$workdir"
        --expected-frames "$AB_EXPECTED_FRAMES"
        --frontend-name "$frontend"
        --feature-profile "$profile"
        --keyframe-preset "$preset"
        --backend global
        --backend-name "$backend"
        --pose-name "$pose"
        "$@"
    )
    printf 'RUN:'
    printf ' %q' "${command[@]}"
    printf '\nLOG: %s\n' "$log"
    start="$(date +%s)"
    status=0
    if "${command[@]}" 2>&1 | tee -a "$log"; then
        status=0
    else
        pipeline_statuses=("${PIPESTATUS[@]}")
        status="${pipeline_statuses[0]}"
        if ((status == 0)); then
            status="${pipeline_statuses[1]}"
        fi
    fi
    end="$(date +%s)"
    printf 'ELAPSED: %s stage=%s arm=%s status=%s\n' \
        "$((end - start))" "$stage" "$arm" "$status" | tee -a "$log"
    ((status == 0)) || return "$status"
}

ab_verify_pose() {
    local arm="$1"
    local workdir pose
    workdir="$(ab_arm_workdir "$arm")"
    pose="$(ab_pose_name "$arm")"
    "$AB_PYTHON" - "$AB_CONFIG" "$AB_SEGMENT" "$workdir" "$pose" \
        "$AB_EXPECTED_FRAMES" <<'PY'
import json
import sys
from pathlib import Path

from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import load_pose_artifact

cfg = load_config(sys.argv[1])
segment = Path(sys.argv[2])
cfg.paths.workdir = Path(sys.argv[3])
cfg.paths.segment = segment
cfg.pose.artifact = sys.argv[4]
expected = int(sys.argv[5])
viewmats, centers = load_pose_artifact(segment, cfg)
quality_path = cfg.paths.workdir / "pose_artifacts" / cfg.pose.artifact / "quality.json"
quality = json.loads(quality_path.read_text())
if len(viewmats) != expected or centers.shape != (expected, 3):
    raise SystemExit("pose artifact does not retain every frame")
if quality.get("n_frames") != expected:
    raise SystemExit("pose quality frame count mismatch")
if quality.get("registration_fraction") != 1.0:
    raise SystemExit("pose artifact is not 100% registered")
if not quality.get("rtk_alignment_passed"):
    raise SystemExit("pose artifact failed fixed-scale RTK gates")
PY
}

ab_compare_golden_pose() {
    local arm="$1"
    local workdir pose candidate output reference
    workdir="$(ab_arm_workdir "$arm")"
    pose="$(ab_pose_name "$arm")"
    candidate="$workdir/pose_artifacts/$pose"
    output="$workdir/reports/pose_delta_vs_golden.json"
    reference="$(
        "$AB_PYTHON" - \
            "$AB_REPO_ROOT/docs/experiments/golden/headland_stereo_ba.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
layout = manifest["artifact_layout"]
print(
    Path(layout["recorded_workdir"]).expanduser()
    / "segment"
    / "pose_artifacts"
    / layout["pose_artifact"]
)
PY
    )"
    [[ -d "$reference" ]] ||
        ab_die "accepted golden pose artifact is unavailable: $reference"
    if [[ ! -e "$output" ]]; then
        "$AB_PYTHON" -m diagnostics.compare_poses \
            --reference "$reference" \
            --candidate "$candidate" \
            --output "$output"
    else
        echo "RESUME: pose-delta report already exists: $output"
    fi
}

ab_verify_frontend_plan() {
    local arm="$1"
    local preset="$2"
    local artifact
    artifact="$(ab_arm_workdir "$arm")/frontend_artifacts/$(ab_frontend_name "$arm")"
    "$AB_PYTHON" - "$artifact" "$preset" "$AB_EXPECTED_FRAMES" <<'PY'
import json
import math
import sys
from pathlib import Path

artifact = Path(sys.argv[1])
preset = sys.argv[2]
expected_frames = int(sys.argv[3])
keyframes = json.loads((artifact / "keyframes.json").read_text())
quality = json.loads((artifact / "quality.json").read_text())
manifest = json.loads((artifact / "frame_manifest.json").read_text())
if len(manifest.get("frames", [])) != expected_frames:
    raise SystemExit("frontend frame manifest lost input frames")
if quality.get("n_frames") != expected_frames:
    raise SystemExit("frontend quality frame count mismatch")
if quality.get("mandatory_stereo_pairs") != expected_frames:
    raise SystemExit("frontend lacks one mandatory stereo pair per frame")
if not quality.get("all_frames_have_registration_path"):
    raise SystemExit("frontend cannot register every non-keyframe")
if not quality.get("all_frames_retained_for_gs"):
    raise SystemExit("frontend does not retain every GS view")
config = keyframes.get("config", {})
expected = {
    "dense": (False, 0.08, 1.0, 1.5),
    "balanced": (False, 0.10, 2.0, 1.5),
    "sparse": (False, 0.15, 3.0, 1.5),
    "all": (True, None, None, None),
}[preset]
if config.get("select_all_frames") is not expected[0]:
    raise SystemExit(f"keyframe preset {preset!r} was overridden")
if preset == "all":
    if len(keyframes.get("frame_ids", [])) != expected_frames:
        raise SystemExit("all-frame arm did not select every solve frame")
else:
    values = (
        float(config.get("translation_m", math.nan)),
        float(config.get("rotation_deg", math.nan)),
        float(config.get("max_elapsed_s", math.nan)),
    )
    if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for a, b in zip(values, expected[1:])):
        raise SystemExit(
            f"keyframe preset {preset!r} was overridden: {values}"
        )
    count = len(keyframes.get("frame_ids", []))
    if not 1 < count < expected_frames:
        raise SystemExit(f"adaptive preset selected an invalid count: {count}")
PY
}

ab_verify_features() {
    local arm="$1"
    local expected_profile="$2"
    local report
    report="$(ab_arm_workdir "$arm")/frontend_artifacts/$(ab_frontend_name "$arm")/stage_reports/features.json"
    "$AB_PYTHON" - "$report" "$expected_profile" \
        "$((2 * AB_EXPECTED_FRAMES))" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
profile = sys.argv[2]
expected_images = int(sys.argv[3])
if report.get("profile") != profile:
    raise SystemExit(
        f"feature profile mismatch: {report.get('profile')!r} != {profile!r}"
    )
if report.get("n_images") != expected_images:
    raise SystemExit("feature extraction did not cover every stereo image")
if min(report.get("n_keypoints", 0), report.get("n_descriptors", 0)) <= 0:
    raise SystemExit("feature extraction produced no usable features")
command = report.get("command", [])
options = dict(zip(command[2::2], command[3::2]))
if options.get("--FeatureExtraction.use_gpu") != (
    "1" if profile == "gpu" else "0"
):
    raise SystemExit("feature command does not match the declared profile")
for option in (
    "--SiftExtraction.estimate_affine_shape",
    "--SiftExtraction.domain_size_pooling",
):
    if profile == "gpu" and option in options:
        raise SystemExit(f"GPU arm unexpectedly enables {option}")
    if profile == "cpu_reference" and options.get(option) != "1":
        raise SystemExit(f"CPU-reference arm does not enable {option}")
PY
}

ab_pose_arm() {
    local arm="$1"
    local index="$2"
    local profile="${AB_FEATURE_PROFILES[$index]}"
    local preset="${AB_KEYFRAME_PRESETS[$index]}"
    local workdir frontend pose_dir
    workdir="$(ab_arm_workdir "$arm")"
    frontend="$workdir/frontend_artifacts/$(ab_frontend_name "$arm")"
    pose_dir="$workdir/pose_artifacts/$(ab_pose_name "$arm")"
    if [[ ! -d "$frontend" ]]; then
        ab_run_cli "$arm" frontend-build "$profile" "$preset"
    else
        echo "RESUME: sealed frontend foundation already exists: $frontend"
    fi
    ab_verify_frontend_plan "$arm" "$preset"
    ab_run_cli "$arm" frontend-features "$profile" "$preset"
    ab_verify_features "$arm" "$profile"
    ab_run_cli "$arm" frontend-rig "$profile" "$preset"
    ab_run_cli "$arm" frontend-priors "$profile" "$preset"
    ab_run_cli "$arm" frontend-match "$profile" "$preset"
    ab_run_cli "$arm" backend-prepare "$profile" "$preset"
    ab_run_cli "$arm" backend-solve "$profile" "$preset"
    ab_run_cli "$arm" backend-register "$profile" "$preset"
    ab_run_cli "$arm" backend-quality "$profile" "$preset"
    if [[ ! -d "$pose_dir" ]]; then
        ab_run_cli "$arm" backend-export "$profile" "$preset"
    else
        echo "RESUME: pose artifact already exists: $pose_dir"
    fi
    ab_verify_pose "$arm"
    ab_compare_golden_pose "$arm"
}

ab_verify_cloud() {
    local arm="$1"
    local workdir pose
    workdir="$(ab_arm_workdir "$arm")"
    pose="$(ab_pose_name "$arm")"
    "$AB_PYTHON" - "$AB_CONFIG" "$AB_SEGMENT" "$workdir" "$pose" <<'PY'
import sys
from pathlib import Path

from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import (
    cloud_path,
    load_pose_artifact,
    verify_cloud_matches_poses,
)

cfg = load_config(sys.argv[1])
segment = Path(sys.argv[2])
cfg.paths.workdir = Path(sys.argv[3])
cfg.paths.segment = segment
cfg.pose.artifact = sys.argv[4]
viewmats, _ = load_pose_artifact(segment, cfg)
verify_cloud_matches_poses(
    cloud_path(segment, cfg), viewmats, require_fingerprint=True
)
PY
}

ab_ensure_cloud() {
    local arm="$1"
    local index="$2"
    local workdir pose cloud
    workdir="$(ab_arm_workdir "$arm")"
    pose="$(ab_pose_name "$arm")"
    cloud="$workdir/cloud_artifacts/$pose/init_cloud.npz"
    if [[ ! -f "$cloud" ]]; then
        ab_run_cli "$arm" cloud \
            "${AB_FEATURE_PROFILES[$index]}" \
            "${AB_KEYFRAME_PRESETS[$index]}"
    else
        echo "RESUME: cloud artifact already exists: $cloud"
    fi
    ab_verify_cloud "$arm"
}

ab_verify_training() {
    local arm="$1"
    local run_name="$2"
    local expected="$3"
    local run_dir
    run_dir="$(ab_arm_workdir "$arm")/runs/$run_name"
    "$AB_PYTHON" - "$run_dir" "$expected" "$AB_EXPECTED_FRAMES" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
expected_presentations = int(sys.argv[2])
expected_frames = int(sys.argv[3])
for name in ("run_provenance.json", "metrics.json", "params.pt"):
    if not (run / name).is_file():
        raise SystemExit(f"incomplete training run: {run / name}")
record = json.loads((run / "run_provenance.json").read_text())
if record.get("training_image_presentations") != expected_presentations:
    raise SystemExit(
        "training presentation mismatch: "
        f"{record.get('training_image_presentations')} != {expected_presentations}"
    )
if record.get("all_segment_frames_retain_poses") != expected_frames:
    raise SystemExit("training did not retain poses for every segment frame")
metrics = json.loads((run / "metrics.json").read_text())
if not metrics or metrics[-1].get("step") != expected_presentations:
    raise SystemExit("training metrics do not contain the expected final step")
PY
}

ab_train_arm() {
    local arm="$1"
    local index="$2"
    local iterations="$3"
    local suffix="$4"
    local run_name="$AB_KIND-$arm-$suffix"
    local run_dir
    run_dir="$(ab_arm_workdir "$arm")/runs/$run_name"
    if [[ ! -e "$run_dir" ]]; then
        ab_run_cli "$arm" train \
            "${AB_FEATURE_PROFILES[$index]}" \
            "${AB_KEYFRAME_PRESETS[$index]}" \
            --run-name "$run_name" \
            --train-iters "$iterations"
    elif [[ ! -f "$run_dir/run_provenance.json" ||
            ! -f "$run_dir/metrics.json" ||
            ! -f "$run_dir/params.pt" ]]; then
        ab_die "refusing incomplete, non-resumable GS directory: $run_dir"
    else
        echo "RESUME: completed training run already exists: $run_dir"
    fi
    ab_verify_training "$arm" "$run_name" "$iterations"
}

ab_verify_equal_proxies() {
    local arguments=()
    local arm
    for arm in "${AB_ARMS[@]}"; do
        arguments+=(
            "$(ab_arm_workdir "$arm")/runs/$AB_KIND-$arm-proxy-15k/run_provenance.json"
            "$(ab_arm_workdir "$arm")/pose_artifacts/$(ab_pose_name "$arm")/timestamps_ns.npy"
        )
    done
    "$AB_PYTHON" - "$AB_PROXY_ITERS" "$AB_EXPECTED_FRAMES" "${arguments[@]}" <<'PY'
import json
import sys

import numpy as np

expected_presentations = int(sys.argv[1])
expected_frames = int(sys.argv[2])
paths = sys.argv[3:]
records = [json.load(open(paths[index])) for index in range(0, len(paths), 2)]
timestamps = [np.load(paths[index]) for index in range(1, len(paths), 2)]
for record in records:
    if record.get("training_image_presentations") != expected_presentations:
        raise SystemExit("proxy arms have unequal training presentations")
    if record.get("all_segment_frames_retain_poses") != expected_frames:
        raise SystemExit("a proxy arm lost full-frame pose coverage")
for field in (
    "available_training_views",
    "manifest_sha256",
    "frames_sha256",
    "calibration_sha256",
    "segment_meta_sha256",
    "training_implementation_sha256",
    "train_seed",
    "launcher_config_sha256",
):
    values = {record.get(field) for record in records}
    if len(values) != 1:
        raise SystemExit(f"proxy arms differ in {field}: {values}")
normalized = []
for record in records:
    config = json.loads(json.dumps(record["effective_training_config"]))
    config.get("pose", {}).pop("artifact", None)
    config.get("pose", {}).pop("artifact_root", None)
    config.get("cloud", {}).pop("artifact_root", None)
    config.get("train", {}).pop("run_name", None)
    normalized.append(
        json.dumps(config, sort_keys=True, separators=(",", ":"))
    )
if len(set(normalized)) != 1:
    raise SystemExit("proxy arms differ in normalized effective GS config")
if any(len(value) != expected_frames for value in timestamps):
    raise SystemExit("a pose timestamp array lost frames")
if any(not np.array_equal(timestamps[0], value) for value in timestamps[1:]):
    raise SystemExit("evaluation/frame timestamps differ across arms")
print(
    f"EQUAL-PRESENTATION GATE PASSED: {len(records)} arms, "
    f"{expected_presentations} presentations each, "
    f"{records[0]['available_training_views']} available training views, "
    f"{expected_frames} timestamped poses each"
)
PY
}

ab_pose_phase() {
    local index
    for index in "${!AB_ARMS[@]}"; do
        echo
        echo "=== POSE ARM: ${AB_ARMS[$index]} ==="
        ab_pose_arm "${AB_ARMS[$index]}" "$index"
    done
    cat <<EOF

POSE-ONLY PHASE COMPLETE.
No GS training was started.
Review each backend and pose quality report under:
  $AB_EXPERIMENT_ROOT/arms/<arm>/{backend_artifacts,pose_artifacts}
Then compare against the fixed golden pose/metrics and invoke the proxy phase
with --resume-existing --accept-pose-gates. The script will not advance itself.
EOF
}

ab_proxy_phase() {
    ((AB_ACCEPT_POSE == 1)) ||
        ab_die "proxy requires the explicit --accept-pose-gates acknowledgment"
    local index arm
    for index in "${!AB_ARMS[@]}"; do
        arm="${AB_ARMS[$index]}"
        ab_verify_pose "$arm"
        ab_ensure_cloud "$arm" "$index"
        echo
        echo "=== 15K PROXY ARM: $arm ==="
        ab_train_arm "$arm" "$index" "$AB_PROXY_ITERS" "proxy-15k"
    done
    ab_verify_equal_proxies
    cat <<EOF

PROXY PHASE COMPLETE.
Every arm used exactly $AB_PROXY_ITERS training-image presentations.
Review final metrics and measured runtime/storage before choosing one winner.
Invoke full with --winner NAME --resume-existing --accept-proxy-gates.
The script will not select or start the full run itself.
EOF
}

ab_full_phase() {
    ((AB_ACCEPT_PROXY == 1)) ||
        ab_die "full requires the explicit --accept-proxy-gates acknowledgment"
    [[ -n "$AB_WINNER" ]] || ab_die "full requires --winner"
    local index
    index="$(ab_arm_index "$AB_WINNER")" ||
        ab_die "unknown winner '$AB_WINNER'"
    ab_verify_equal_proxies
    ab_verify_pose "$AB_WINNER"
    ab_verify_cloud "$AB_WINNER"
    echo
    echo "=== 65K FULL WINNER: $AB_WINNER ==="
    ab_train_arm "$AB_WINNER" "$index" "$AB_FULL_ITERS" "winner-65k"
    echo
    echo "FULL WINNER COMPLETE: $AB_WINNER; no other 65k arm was launched."
}

ab_main() {
    ab_check_definition
    ab_parse "$@"
    if [[ "$AB_PHASE" == "plan" ]]; then
        ab_print_plan
        exit 0
    fi
    if [[ "$AB_PHASE" != "full" && -n "$AB_WINNER" ]]; then
        ab_die "--winner is valid only for the full phase"
    fi
    ab_prepare
    ab_print_plan
    echo
    echo "Config:          $AB_CONFIG"
    echo "Immutable input: $AB_SEGMENT"
    echo "Experiment root: $AB_EXPERIMENT_ROOT"
    echo "Python:          $AB_PYTHON"
    echo "COLMAP:          $AB_COLMAP"
    case "$AB_PHASE" in
        pose) ab_pose_phase ;;
        proxy) ab_proxy_phase ;;
        full) ab_full_phase ;;
    esac
}
