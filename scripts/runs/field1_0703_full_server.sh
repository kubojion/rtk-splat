#!/usr/bin/env bash
# Full 77-minute field build on the RTX 3090 server.
#
# Input is the portable, checksum-sealed derived segment prepared locally. The
# launcher performs the visual pose solve once, plans an arbitrary number of
# visibility tiles, trains those tiles sequentially, and publishes one
# production scene without requiring a monolithic GS reference.
#
# Recovery is deliberately stage based. Native sealed frontend/backend stages
# are re-verified before they are skipped. An interrupted GS optimizer is not
# resumed approximately: its directory is preserved and the same tile starts
# in a new immutable attempt directory on the next invocation.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

CONFIG="$REPO/configs/sequences/field1_0703_full_77min.yaml"
PREFLIGHT_TOOL="$REPO/scripts/tools/server_preflight.py"
SEGMENT_DEFAULT="/data/jkobo/datasets/field1_0703_full77/segment"
WORKDIR_DEFAULT="/data/jkobo/runs/field1_0703_full77_v1"

SEGMENT="$SEGMENT_DEFAULT"
WORKDIR="$WORKDIR_DEFAULT"
if [[ -n "${RTK_SPLAT_PYTHON:-}" ]]; then
    PY="$RTK_SPLAT_PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PY="$CONDA_PREFIX/bin/python"
else
    PY="$(command -v python3 || true)"
fi
COLMAP="${COLMAP_BIN:-$HOME/miniconda3/envs/colmap-rtk/bin/colmap}"
MINIMUM_FREE_GIB=500
MINIMUM_RAM_GIB=120
MAXIMUM_VISIBLE_GAUSSIANS=12000000
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    ACTION=help
    shift
else
    ACTION="${1:-plan}"
    if (($#)); then shift; fi
fi

FRONTEND_NAME="field1-0703-full77-all-gpu-v1"
BACKEND_NAME="field1-0703-full77-global-v1"
POSE_NAME="field1-0703-full77-global-pose-v1"
PLAN_NAME="field1-0703-full77-visibility-v1"
RUN_BASE="field1-0703-full77-tile-gs-v1"
SCENE_NAME="field1-0703-full77-production-scene-v1"

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<EOF
Usage:
  bash scripts/runs/field1_0703_full_server.sh plan
  bash scripts/runs/field1_0703_full_server.sh preflight [options]
  bash scripts/runs/field1_0703_full_server.sh prepare [options]
  bash scripts/runs/field1_0703_full_server.sh smoke [options]
  bash scripts/runs/field1_0703_full_server.sh run [options]
  bash scripts/runs/field1_0703_full_server.sh status [options]

Options:
  --segment PATH                 Portable derived segment
  --workdir PATH                 Server artifact root
  --python PATH                  Python in the tested GPU environment
  --colmap PATH                  COLMAP 4.1.1 CUDA executable
  --minimum-free-gib N           Free-space gate (default: 500)
  --minimum-ram-gib N            Physical-RAM gate (default: 120)
  --max-visible-gaussians N      Per-view final evaluation cap (default: 12000000)
  -h, --help                     Show this help

prepare: visual frontend -> Global Mapper -> production pose -> automatic plan
smoke:   prepare, then fully train the most demanding planned tile
run:     prepare, train every planned tile sequentially, publish final scene

The launcher uses no session manager and never shuts the host down. It is safe
to keep in the foreground or wrap with nohup; systemd-inhibit blocks sleep when
available. Re-running the same action verifies completed work and continues.
EOF
}

while (($#)); do
    case "$1" in
        --segment)
            (($# >= 2)) || fail "--segment needs a path"
            SEGMENT="$(readlink -m "$2")"
            shift 2
            ;;
        --workdir)
            (($# >= 2)) || fail "--workdir needs a path"
            WORKDIR="$(readlink -m "$2")"
            shift 2
            ;;
        --python)
            (($# >= 2)) || fail "--python needs a path"
            PY="$(readlink -m "$2")"
            shift 2
            ;;
        --colmap)
            (($# >= 2)) || fail "--colmap needs a path"
            COLMAP="$(readlink -m "$2")"
            shift 2
            ;;
        --minimum-free-gib)
            (($# >= 2)) || fail "--minimum-free-gib needs a number"
            MINIMUM_FREE_GIB="$2"
            shift 2
            ;;
        --minimum-ram-gib)
            (($# >= 2)) || fail "--minimum-ram-gib needs a number"
            MINIMUM_RAM_GIB="$2"
            shift 2
            ;;
        --max-visible-gaussians)
            (($# >= 2)) || fail "--max-visible-gaussians needs an integer"
            MAXIMUM_VISIBLE_GAUSSIANS="$2"
            shift 2
            ;;
        -h|--help)
            ACTION=help
            shift
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

case "$ACTION" in
    plan|preflight|prepare|smoke|run|status|help) ;;
    *) fail "action must be plan, preflight, prepare, smoke, run, or status" ;;
esac
[[ "$MINIMUM_FREE_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fail "--minimum-free-gib must be a non-negative number"
[[ "$MINIMUM_RAM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fail "--minimum-ram-gib must be a positive number"
[[ "$MINIMUM_RAM_GIB" != 0 && "$MINIMUM_RAM_GIB" != 0.0 ]] \
    || fail "--minimum-ram-gib must be a positive number"
[[ "$MAXIMUM_VISIBLE_GAUSSIANS" =~ ^[1-9][0-9]*$ ]] \
    || fail "--max-visible-gaussians must be a positive integer"

FRONTEND="$WORKDIR/frontend_artifacts/$FRONTEND_NAME"
BACKEND="$WORKDIR/backend_artifacts/$BACKEND_NAME"
POSE_PARENT="$WORKDIR/pose_artifacts"
POSE="$POSE_PARENT/$POSE_NAME"
PLAN="$WORKDIR/tile_plan_artifacts/$PLAN_NAME"
SCENE="$WORKDIR/scene_artifacts/$SCENE_NAME"
LOGS="$WORKDIR/logs"
ORCHESTRATION="$WORKDIR/orchestration"
RUN_SPEC="$ORCHESTRATION/server_run_spec.json"
PREFLIGHT_RECORD="$ORCHESTRATION/server_preflight.json"
EXPECTED_FRAMES=""
PREFLIGHT_JSON=""

print_plan() {
    cat <<EOF
Full 77-minute field production build (server side only)

  config:             $CONFIG
  portable segment:   $SEGMENT
  workdir:            $WORKDIR
  pose solver:        stereo GPU features/matches + bounded Global Mapper
  tiling:             automatic visibility/workload plan (tile count is data-derived)
  tile training:      65,000 iterations, 2,500,000 Gaussian cap, sequential
  final publication:  absolute whole-scene + seam evaluation; no monolithic reference

Conservative RTX 3090 timing after the segment is on server:
  preflight and full transfer hash          5-20 min
  frontend feature/match graph              2-6 h
  Global Mapper, registration and export    8-24 h (largest uncertainty)
  TilePlan                                  0.5-2 h
  each complete tile                        1.5-3.5 h
  final scene publication                   1-4 h

The planner, not this script, decides the number and IDs of tiles. Expecting
roughly 10-15 tiles is only a capacity estimate, so the complete job may take
about 2-4 days. Start with prepare, then smoke, then run. Every completed tile
is reused after full verification. Incomplete attempts remain on disk and only
that tile is restarted under a new attempt name.

Nothing is started by plan. No directory is created.
EOF
}

if [[ "$ACTION" == help ]]; then usage; exit 0; fi
if [[ "$ACTION" == plan ]]; then print_plan; exit 0; fi

semantic_segment_check() {
    EXPECTED_FRAMES="$("$PY" - "$CONFIG" "$SEGMENT" <<'PY'
import sys
from pathlib import Path

import numpy as np

from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config

config, segment = map(Path, sys.argv[1:3])
cfg = load_config(config)
reader = SegmentReader(segment).validate()
frames = reader.frames
meta = reader.meta
caps = meta["capabilities"]
if meta.get("adapter") != "ros2_zed_ublox":
    raise SystemExit(f"unexpected segment adapter: {meta.get('adapter')!r}")
origin = meta.get("coordinate_frame", {}).get("origin_wgs84", {})
latitude = float(origin.get("latitude_deg", float("nan")))
longitude = float(origin.get("longitude_deg", float("nan")))
if (
    not np.isfinite([latitude, longitude]).all()
    or abs(latitude - 52.29136) > 0.002
    or abs(longitude - 16.39219) > 0.002
):
    raise SystemExit(
        f"segment origin ({latitude}, {longitude}) is not field 1 on 2026-07-03"
    )
required = {
    "stereo": True,
    "dual_rtk": True,
    "depth_computed": True,
    "images_rectified": True,
}
wrong = {name: caps.get(name) for name, expected in required.items()
         if caps.get(name) is not expected}
if wrong:
    raise SystemExit(f"full-field segment capabilities are wrong: {wrong}")
n_frames = int(meta["n_frames"])
timestamps = np.asarray(frames["timestamp_ns"], dtype=np.int64)
duration_s = float((timestamps[-1] - timestamps[0]) / 1e9)
pose_valid = np.asarray(frames["pose_valid"], dtype=bool)
centers = np.asarray(frames["initial_camera_center_m"], dtype=np.float64)
valid = pose_valid & np.isfinite(centers).all(axis=1)
if float(valid.mean()) < 0.99:
    raise SystemExit(
        f"only {float(valid.mean()):.3%} of full-field frames have initial poses"
    )
valid_indices = np.flatnonzero(valid)
sampled = [int(valid_indices[0])]
for index in valid_indices[1:]:
    if int(timestamps[index]) - int(timestamps[sampled[-1]]) >= 2_000_000_000:
        sampled.append(int(index))
if sampled[-1] != int(valid_indices[-1]):
    sampled.append(int(valid_indices[-1]))
sampled_centers = centers[np.asarray(sampled, dtype=np.int64)]
steps_m = np.linalg.norm(np.diff(sampled_centers, axis=0), axis=1)
if len(steps_m) < 100 or not np.isfinite(steps_m).all() or float(steps_m.max()) > 2.0:
    raise SystemExit("initial camera trajectory has gaps or implausible 2 s jumps")
path_length_m = float(steps_m.sum())
trajectory_span_m = float(np.linalg.norm(np.ptp(sampled_centers, axis=0)))
if not (10_000 <= n_frames <= 20_000):
    raise SystemExit(f"expected a full-field frame count, found {n_frames}")
if not (4_400.0 <= duration_s <= 4_700.0):
    raise SystemExit(f"expected the 77-minute duration, found {duration_s:.3f} s")
if not (300.0 <= path_length_m <= 550.0) or trajectory_span_m < 30.0:
    raise SystemExit(
        "expected the full-field trajectory; measured a robust 2 s polyline of "
        f"{path_length_m:.3f} m and span {trajectory_span_m:.3f} m"
    )
if list(cfg.segment.window_s) != [0.0, 4638.0] or int(cfg.segment.frame_stride) != 5:
    raise SystemExit("sequence config no longer names the frozen full window/stride")
if str(cfg.frontend.name) != "field1-0703-full77-all-gpu-v1":
    raise SystemExit("frontend name changed")
if str(cfg.frontend.keyframes.preset) != "all":
    raise SystemExit("full-field frontend must retain all selected frames")
if str(cfg.mapper.name) != "field1-0703-full77-global-v1":
    raise SystemExit("backend name changed")
if str(cfg.mapper.pose_artifact_name) != "field1-0703-full77-global-pose-v1":
    raise SystemExit("pose name changed")
if str(cfg.tiles.name) != "field1-0703-full77-visibility-v1":
    raise SystemExit("TilePlan name changed")
if int(cfg.train.iterations) != 65_000 or int(cfg.train.max_gaussians) != 2_500_000:
    raise SystemExit("first production transfer must retain the accepted 65k/2.5M policy")
if bool(cfg.pose.use_imu_tilt):
    raise SystemExit("this no-IMU experiment unexpectedly enables IMU tilt")
print(n_frames)
print(
    f"verified full field semantics: {n_frames} frames, {duration_s:.1f} s, "
    f"{path_length_m:.1f} m robust 2 s pose polyline, no IMU tilt",
    file=sys.stderr,
)
PY
)" || fail "full-field segment/config semantic check failed"
}

preflight() {
    [[ -x "$PY" ]] || fail "Python is not executable: $PY"
    [[ -x "$COLMAP" ]] || fail "COLMAP is not executable: $COLMAP"
    [[ -f "$CONFIG" ]] || fail "sequence config is missing: $CONFIG"
    [[ -f "$PREFLIGHT_TOOL" ]] || fail "server preflight tool is missing: $PREFLIGHT_TOOL"
    [[ -d "$SEGMENT" ]] || fail "portable segment is missing: $SEGMENT"
    command -v flock >/dev/null || fail "flock is required"
    [[ -x /usr/bin/time ]] || fail "/usr/bin/time is required"
    PREFLIGHT_JSON="$("$PY" "$PREFLIGHT_TOOL" \
        --segment "$SEGMENT" --work-root "$WORKDIR" --colmap "$COLMAP" \
        --minimum-free-gib "$MINIMUM_FREE_GIB" \
        --minimum-ram-gib "$MINIMUM_RAM_GIB")" \
        || fail "server hardware/environment preflight failed"
    printf '%s\n' "$PREFLIGHT_JSON"
    semantic_segment_check
    echo "Preflight passed for exactly $EXPECTED_FRAMES frames; nothing was written."
}

if [[ "$ACTION" == preflight ]]; then preflight; exit 0; fi

status() {
    echo "segment: $SEGMENT"
    echo "workdir: $WORKDIR"
    [[ -f "$RUN_SPEC" ]] && echo "run identity: sealed" \
        || echo "run identity: missing"
    [[ -f "$FRONTEND/frontend_seal.json" ]] && echo "frontend: sealed" \
        || echo "frontend: pending/incomplete"
    [[ -f "$BACKEND/stages/quality.json" ]] && echo "global pose solve: quality stage present" \
        || echo "global pose solve: pending/incomplete"
    [[ -f "$POSE/manifest.json" ]] && echo "production pose: published" \
        || echo "production pose: pending"
    if [[ -f "$PLAN/tile_plan.json" ]]; then
        "$PY" - "$PLAN/tile_plan.json" "$WORKDIR" "$PLAN_NAME" "$RUN_BASE" <<'PY'
import json, sys
from pathlib import Path

plan_file, workdir, plan_name, run_base = sys.argv[1:]
plan = json.loads(Path(plan_file).read_text())
tiles = [str(item["tile_id"]) for item in plan["tiles"]]
complete = incomplete = pending = 0
for tile in tiles:
    root = Path(workdir) / "tile_runs" / plan_name / tile
    attempts = sorted(root.glob(f"{run_base}-attempt-[0-9][0-9][0-9][0-9]"))
    completed = [path for path in attempts if (path / "params.pt").is_file()]
    partial = [path for path in attempts if not (path / "params.pt").is_file()]
    if completed:
        state = "completion marker present"
        complete += 1
    elif partial:
        state = f"incomplete attempts preserved: {len(partial)}"
        incomplete += 1
    else:
        state = "pending"
        pending += 1
    cloud = (
        Path(workdir) / "cloud_artifacts" / "field1-0703-full77-global-pose-v1"
        / "tile_plans" / plan_name / tile / "manifest.json"
    )
    print(f"{tile}: cloud={'present' if cloud.is_file() else 'pending'}; {state}")
print(f"tiles: {len(tiles)} planned, {complete} marked complete, "
      f"{incomplete} interrupted, {pending} pending (status does not rehash them)")
PY
    else
        echo "TilePlan: pending"
    fi
    [[ -f "$SCENE/manifest.json" ]] && echo "production scene: published" \
        || echo "production scene: pending"
    local active
    active="$(pgrep -af '[p]ython.*rtk_splat.*workflows\.cli.*(frontend|backend|tiles-plan|cloud|train|scene-publish)' || true)"
    [[ -z "$active" ]] && echo "active server mapping stage: none" \
        || echo "active server mapping stage: $active"
}

if [[ "$ACTION" == status ]]; then status; exit 0; fi

# Long actions remain foreground jobs. This wrapper only prevents the host from
# sleeping; callers may safely put the command under nohup if desired.
if [[ -z "${RTK_SPLAT_INHIBITED:-}" ]] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    argv=("$ACTION" --segment "$SEGMENT" --workdir "$WORKDIR" \
        --python "$PY" --colmap "$COLMAP" \
        --minimum-free-gib "$MINIMUM_FREE_GIB" \
        --minimum-ram-gib "$MINIMUM_RAM_GIB" \
        --max-visible-gaussians "$MAXIMUM_VISIBLE_GAUSSIANS")
    exec systemd-inhibit --what=sleep:idle --mode=block \
        --why="RTK-Splat full-field production build" \
        bash "$SCRIPT_PATH" "${argv[@]}"
fi

lock_directory() {
    local candidate
    if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "$XDG_RUNTIME_DIR" \
        && ! -L "$XDG_RUNTIME_DIR" && -O "$XDG_RUNTIME_DIR" ]]; then
        candidate="$XDG_RUNTIME_DIR/rtk-splat-run-locks"
    else
        candidate="$(dirname "$WORKDIR")/run_locks"
    fi
    if [[ -e "$candidate" ]]; then
        [[ -d "$candidate" && ! -L "$candidate" && -O "$candidate" ]] \
            || fail "run-lock directory is not a private owned directory: $candidate"
    else
        mkdir -p -- "$candidate"
    fi
    chmod 700 -- "$candidate"
    [[ "$(stat -c '%a' -- "$candidate")" == 700 ]] \
        || fail "cannot enforce mode 700 on run-lock directory: $candidate"
    printf '%s\n' "$candidate"
}

LOCK_DIR="$(lock_directory)"
LOCK="$LOCK_DIR/field1-full-$(printf '%s' "$WORKDIR" | sha256sum | cut -c1-16).lock"
exec 9>"$LOCK"
flock -n 9 || fail "another launcher owns this workdir: $WORKDIR"

preflight

initialize_run_identity() {
    if [[ -e "$WORKDIR" && ! -d "$WORKDIR" ]]; then
        fail "workdir is not a directory: $WORKDIR"
    fi
    if [[ -d "$WORKDIR" && ! -f "$RUN_SPEC" ]] \
        && [[ -n "$(find "$WORKDIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        local unexpected
        unexpected="$(find "$WORKDIR" -mindepth 1 \
            ! -path "$ORCHESTRATION" ! -path "$PREFLIGHT_RECORD" \
            -print -quit)"
        [[ -z "$unexpected" ]] || \
            fail "non-empty workdir has no server run identity; choose a new workdir"
    fi
    [[ -n "$PREFLIGHT_JSON" ]] || fail "no passing preflight inventory to seal"
    mkdir -p "$ORCHESTRATION"
    local preflight_temporary status
    preflight_temporary="$(mktemp "$ORCHESTRATION/.server_preflight.current.XXXXXX")"
    printf '%s\n' "$PREFLIGHT_JSON" > "$preflight_temporary"
    "$PY" - "$RUN_SPEC" "$REPO" "$SCRIPT_PATH" "$CONFIG" "$SEGMENT" \
        "$WORKDIR" "$COLMAP" "$MAXIMUM_VISIBLE_GAUSSIANS" \
        "$MINIMUM_RAM_GIB" "$MINIMUM_FREE_GIB" "$PREFLIGHT_RECORD" \
        "$preflight_temporary" <<'PY' || {
import copy, importlib.metadata, json, os, subprocess, sys
from pathlib import Path

from rtk_splat.frontends.artifact import (
    collect_git_state,
    collect_package_state,
    sha256_file,
)
from rtk_splat.workflows.configio import load_config

target, repo, launcher, config, segment, workdir, colmap = (
    Path(value).expanduser().resolve() for value in sys.argv[1:8]
)
maximum_visible = int(sys.argv[8])
minimum_ram_gib = float(sys.argv[9])
minimum_free_gib = float(sys.argv[10])
preflight_record = Path(sys.argv[11]).resolve()
current_preflight_path = Path(sys.argv[12]).resolve()
git = collect_git_state(repo)
if git["dirty"]:
    raise SystemExit(
        "production server runs require a clean committed Git checkout; "
        "commit the exact tested code before starting"
    )
cfg = load_config(config)
sources = [
    {"role": item.role, "path": item.path, "sha256": item.sha256}
    for item in cfg.runtime_resolution.source_files
]
transfer_manifest = json.loads(
    (segment / "transfer_manifest.json").read_text(encoding="utf-8")
)
if (
    transfer_manifest.get("schema_version") != 1
    or transfer_manifest.get("artifact_type") != "rtk_splat_portable_segment"
    or not isinstance(transfer_manifest.get("files"), list)
):
    raise SystemExit("portable transfer manifest identity is invalid")
transfer = {
    "segment": str(segment),
    "verified_by_immediately_preceding_server_preflight": True,
    "n_frames": int(transfer_manifest["n_frames"]),
    "n_files": len(transfer_manifest["files"]),
    "inventory_sha256": str(transfer_manifest["inventory_sha256"]),
    "size_bytes": int(sum(
        int(record["size_bytes"]) for record in transfer_manifest["files"]
    )),
}
current_preflight = json.loads(
    current_preflight_path.read_text(encoding="utf-8")
)
if (
    current_preflight.get("schema_version") != 1
    or current_preflight.get("status") != "PASSED"
    or current_preflight.get("requirements") != {
        "minimum_free_gib": minimum_free_gib,
        "minimum_ram_gib": minimum_ram_gib,
    }
):
    raise SystemExit("current server preflight inventory is invalid")


def stable_preflight_inventory(document):
    """Remove only capacity values that legitimately change between resumes."""
    stable = copy.deepcopy(document)
    stable["host"]["memory"].pop("available_gib", None)
    stable["host"]["storage"].pop("free_gib", None)
    stable["cuda"].pop("free_gib", None)
    return stable


if preflight_record.exists():
    stored_preflight = json.loads(preflight_record.read_text(encoding="utf-8"))
    if stable_preflight_inventory(stored_preflight) != stable_preflight_inventory(
        current_preflight
    ):
        raise SystemExit(
            "server hardware/environment identity changed since the first preflight; "
            "use a new workdir"
        )
else:
    os.link(current_preflight_path, preflight_record)
    stored_preflight = current_preflight
packages = {
    name: importlib.metadata.version(name)
    for name in (
        "torch", "torchvision", "gsplat", "torchmetrics", "numpy",
        "opencv-python-headless", "scipy", "pymap3d", "PyYAML",
    )
}
banner = subprocess.run(
    [str(colmap), "-h"], check=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True,
).stdout.splitlines()[0]
identity = {
    "artifact_type": "rtk_splat_full_field_server_run_spec",
    "package": collect_package_state(),
    "git": git,
    "launcher": {"path": str(launcher), "sha256": sha256_file(launcher)},
    "configuration_sources": sources,
    "segment": {
        **transfer,
        "transfer_manifest_sha256": sha256_file(segment / "transfer_manifest.json"),
        "contract": {
            name: sha256_file(segment / name)
            for name in (
                "frames.npz", "calibration.json", "segment_meta.json", "manifest.json"
            )
        },
    },
    "environment": {
        "python": str(Path(sys.executable).resolve()),
        "packages": packages,
        "colmap": {
            "path": str(colmap),
            "sha256": sha256_file(colmap),
            "banner": banner,
        },
        "server_preflight": {
            "path": str(preflight_record),
            "sha256": sha256_file(preflight_record),
            "inventory": stored_preflight,
        },
    },
    "execution": {
        "workdir": str(workdir),
        "frontend_name": "field1-0703-full77-all-gpu-v1",
        "backend_name": "field1-0703-full77-global-v1",
        "pose_name": "field1-0703-full77-global-pose-v1",
        "tile_plan_name": "field1-0703-full77-visibility-v1",
        "tile_run_base": "field1-0703-full77-tile-gs-v1",
        "scene_name": "field1-0703-full77-production-scene-v1",
        "tile_selection": "arbitrary_count_from_sealed_tile_plan",
        "training_order": "sequential",
        "within_tile_resume": False,
        "interrupted_tile_policy": "preserve_and_start_new_immutable_attempt",
        "scene_publication": "production_absolute_no_monolithic_reference",
        "maximum_visible_gaussians_per_evaluation_view": maximum_visible,
        "minimum_physical_ram_gib": minimum_ram_gib,
        "minimum_storage_free_gib": minimum_free_gib,
    },
}
document = {"schema_version": 1, "identity": identity}
if target.exists():
    stored = json.loads(target.read_text(encoding="utf-8"))
    if stored != document:
        raise SystemExit(
            "server run identity changed (code/config/segment/environment/path); "
            "use a new workdir"
        )
    print(f"verified immutable server run identity: {target}")
else:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"published immutable server run identity: {target}")
PY
        status=$?
        rm -f -- "$preflight_temporary"
        return "$status"
    }
    rm -f -- "$preflight_temporary"
    mkdir -p "$LOGS"
    export RTK_SPLAT_CONFIG_SHA256
    RTK_SPLAT_CONFIG_SHA256="$(sha256sum "$RUN_SPEC" | awk '{print $1}')"
}

initialize_run_identity
export COLMAP_BIN="$COLMAP"

execute_logged() {
    local label="$1"
    shift
    local log="$LOGS/${label}_$(date '+%Y%m%d_%H%M%S_%N').log"
    echo
    echo "RUN/VERIFY: $label"
    printf '  %q' "$@"
    printf '\n  log: %s\n' "$log"
    /usr/bin/time -v "$@" 2>&1 | tee "$log"
}

verify_frontend_foundation() {
    "$PY" - "$SEGMENT" "$WORKDIR" "$FRONTEND" \
        "$FRONTEND_NAME" "$COLMAP" <<'PY'
import json, sys
from pathlib import Path

from rtk_splat.core.segment import SegmentReader
from rtk_splat.frontends.artifact import (
    _validate_provenance,
    collect_package_state,
    frontend_image_record,
    image_inventory,
    probe_colmap_identity,
)

segment, workdir, frontend, name, colmap = sys.argv[1:]
reader = SegmentReader(segment).validate()
root = Path(frontend)
required = (
    "frame_manifest.json", "rig_config.json", "keyframes.json",
    "pairs.txt", "provenance.json", "quality.json",
)
if not root.is_dir() or any(not (root / item).is_file() for item in required):
    raise SystemExit("frontend foundation is incomplete")
provenance = json.loads((root / "provenance.json").read_text())
inventory = image_inventory(reader)
_validate_provenance(reader, provenance, inventory)
frontend_image_record(root)
package = collect_package_state()
if provenance.get("source", {}).get("python_tree_sha256") != package["python_tree_sha256"]:
    raise SystemExit("frontend was made by different package code")
if provenance.get("colmap") != probe_colmap_identity(colmap):
    raise SystemExit("frontend was made with a different COLMAP executable")
resolved = provenance.get("resolved_config", {})
overrides = resolved.get("experiment_overrides", {})
if overrides.get("frontend_name") != name or overrides.get("keyframe_preset") != "all":
    raise SystemExit("frontend experiment identity disagrees")
if Path(resolved["paths"]["workdir"]).expanduser().resolve() != Path(workdir).resolve():
    raise SystemExit("frontend workdir binding disagrees")
if Path(resolved["paths"]["segment"]).expanduser().resolve() != Path(segment).resolve():
    raise SystemExit("frontend segment binding disagrees")
keys = json.loads((root / "keyframes.json").read_text())["frame_ids"]
if keys != reader.frames["frame_id"].astype(int).tolist():
    raise SystemExit("all-frame frontend did not select every frame")
print(f"verified frontend foundation and {len(inventory)} image links")
PY
}

verify_frontend_complete() {
    "$PY" - "$FRONTEND" <<'PY'
import sys
from rtk_splat.frontends.artifact import verify_frontend_seal
seal = verify_frontend_seal(sys.argv[1])
print(f"verified terminal frontend seal: {seal['images']['count']} images")
PY
}

verify_production_pose() {
    "$PY" - "$POSE" "$POSE_NAME" <<'PY'
import sys
from rtk_splat.backends.mapper import _verify_pose_artifact
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact

_verify_pose_artifact(sys.argv[1])
evidence = verify_pose_georeferencing_artifact(
    sys.argv[1], expected_name=sys.argv[2]
)
if (
    evidence.get("artifact_class") != "production"
    or evidence.get("georeferencing_status") != "PASSED"
    or evidence.get("metric_georeferencing_claim_eligible") is not True
):
    raise SystemExit("pose is not a PASSED production georeferencing artifact")
print("verified PASSED production pose and held-out georeferencing gate")
PY
}

verify_plan() {
    "$PY" - "$PLAN" "$SEGMENT" "$POSE" "$PLAN_NAME" <<'PY'
import sys
from rtk_splat.workflows.tiles import verify_tile_plan

plan = verify_tile_plan(
    sys.argv[1], segment=sys.argv[2], pose_root=sys.argv[3], rehash_sources=True
)
if plan.get("name") != sys.argv[4]:
    raise SystemExit("TilePlan name changed")
if plan.get("metric_georeferencing_claim_eligible") is not True:
    raise SystemExit("TilePlan is not eligible for production georeferencing")
tiles = plan.get("tiles", [])
if len(tiles) < 2:
    raise SystemExit("full-field production requires at least two planned tiles")
print(f"verified automatic TilePlan: {len(tiles)} tiles")
for tile in tiles:
    summary = tile["summary"]
    print(
        f"  {tile['tile_id']}: {summary['n_train']} train / "
        f"{summary['n_val']} val"
    )
PY
}

CLI=("$PY" -m rtk_splat.workflows.cli)
COMMON=(--config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
    --expected-frames "$EXPECTED_FRAMES")

prepare_pipeline() {
    if [[ -e "$FRONTEND" ]]; then
        verify_frontend_foundation
    else
        execute_logged frontend-build \
            "${CLI[@]}" frontend-build "${COMMON[@]}" \
            --frontend-name "$FRONTEND_NAME"
        verify_frontend_foundation
    fi
    execute_logged frontend-features \
        "${CLI[@]}" frontend-features "${COMMON[@]}" \
        --frontend-name "$FRONTEND_NAME"
    execute_logged frontend-rig \
        "${CLI[@]}" frontend-rig "${COMMON[@]}" \
        --frontend-name "$FRONTEND_NAME"
    execute_logged frontend-priors \
        "${CLI[@]}" frontend-priors "${COMMON[@]}" \
        --frontend-name "$FRONTEND_NAME"
    execute_logged frontend-match \
        "${CLI[@]}" frontend-match "${COMMON[@]}" \
        --frontend-name "$FRONTEND_NAME"
    verify_frontend_complete

    execute_logged backend-prepare \
        "${CLI[@]}" backend-prepare "${COMMON[@]}" \
        --frontend-name "$FRONTEND_NAME" --backend global \
        --backend-name "$BACKEND_NAME"
    execute_logged backend-solve \
        "${CLI[@]}" backend-solve "${COMMON[@]}" \
        --backend global --backend-name "$BACKEND_NAME"
    execute_logged backend-register \
        "${CLI[@]}" backend-register "${COMMON[@]}" \
        --backend global --backend-name "$BACKEND_NAME"
    execute_logged backend-quality \
        "${CLI[@]}" backend-quality "${COMMON[@]}" \
        --backend global --backend-name "$BACKEND_NAME"

    if [[ -e "$POSE" ]]; then
        verify_production_pose
    else
        execute_logged backend-export \
            "${CLI[@]}" backend-export "${COMMON[@]}" \
            --backend global --backend-name "$BACKEND_NAME" \
            --pose-name "$POSE_NAME"
        verify_production_pose
    fi

    if [[ -e "$PLAN" ]]; then
        verify_plan
    else
        execute_logged tiles-plan \
            "${CLI[@]}" tiles-plan "${COMMON[@]}" \
            --pose-name "$POSE_NAME" --pose-artifact-root "$POSE_PARENT" \
            --tile-plan-name "$PLAN_NAME"
        verify_plan
    fi
}

tile_ids() {
    "$PY" - "$PLAN/tile_plan.json" <<'PY'
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
for tile in plan["tiles"]:
    print(tile["tile_id"])
PY
}

smoke_tile_id() {
    "$PY" - "$PLAN/tile_plan.json" <<'PY'
import json, sys
from pathlib import Path
tiles = json.loads(Path(sys.argv[1]).read_text())["tiles"]
selected = max(
    tiles,
    key=lambda item: (
        int(item["summary"]["n_train"]),
        int(item["summary"]["n_val"]),
        str(item["tile_id"]),
    ),
)
print(selected["tile_id"])
PY
}

cloud_path() {
    printf '%s\n' "$WORKDIR/cloud_artifacts/$POSE_NAME/tile_plans/$PLAN_NAME/$1/init_cloud.npz"
}

verify_cloud() {
    local tile="$1" cloud="$2"
    "$PY" - "$CONFIG" "$SEGMENT" "$WORKDIR" "$POSE_PARENT" \
        "$POSE_NAME" "$PLAN" "$tile" "$cloud" <<'PY'
import sys
from pathlib import Path

from rtk_splat.backends.pose_evidence import pose_georeferencing_evidence
from rtk_splat.core.pose_artifacts import load_pose_artifact
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.cloud import verify_tile_cloud
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.tiles import load_tile_execution

config, segment, workdir, pose_parent, pose_name, plan, tile, cloud = sys.argv[1:]
cfg = load_config(Path(config))
cfg.paths.segment = Path(segment)
cfg.paths.workdir = Path(workdir)
cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(segment).validate()
execution = load_tile_execution(plan, tile, reader, cfg, rehash_sources=False)
viewmats, _ = load_pose_artifact(reader.root, cfg)
points, masks = verify_tile_cloud(
    cloud, execution, viewmats,
    georeferencing=pose_georeferencing_evidence(reader.root, cfg),
)
print(f"verified {tile} cloud: {points:,} points / {masks} context masks")
PY
}

ensure_cloud() {
    local tile="$1" cloud
    cloud="$(cloud_path "$tile")"
    if [[ -e "$cloud" || -e "$(dirname "$cloud")/manifest.json" ]]; then
        [[ -f "$cloud" && -f "$(dirname "$cloud")/manifest.json" ]] \
            || fail "$tile cloud is incomplete; preserve it and use a new workdir"
        verify_cloud "$tile" "$cloud"
        return
    fi
    execute_logged "cloud_${tile}" \
        "${CLI[@]}" cloud "${COMMON[@]}" \
        --pose-name "$POSE_NAME" --pose-artifact-root "$POSE_PARENT" \
        --tile-plan "$PLAN" --tile-id "$tile"
    verify_cloud "$tile" "$cloud"
}

verify_tile_run() {
    local tile="$1" run="$2"
    "$PY" - "$CONFIG" "$SEGMENT" "$WORKDIR" "$POSE_PARENT" \
        "$POSE_NAME" "$PLAN" "$tile" "$run" "$RTK_SPLAT_CONFIG_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path

from rtk_splat.backends.pose_evidence import pose_georeferencing_evidence
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.export_splat import _completed_run_evidence
from rtk_splat.workflows.tile_scene import _completed_tile_run
from rtk_splat.workflows.tiles import load_tile_execution

(
    config, segment, workdir, pose_parent, pose_name, plan, tile, run,
    launcher_hash,
) = sys.argv[1:]
cfg = load_config(Path(config))
cfg.paths.segment = Path(segment)
cfg.paths.workdir = Path(workdir)
cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(segment).validate()
execution = load_tile_execution(plan, tile, reader, cfg, rehash_sources=False)
georef = pose_georeferencing_evidence(reader.root, cfg)
_completed_tile_run(Path(run), execution.binding, georef)
_, best, provenance, _, params_hash = _completed_run_evidence(
    Path(run), allow_failed_georeferencing_for_render=False
)
if int(best["completed_training_steps"]) != int(cfg.train.iterations):
    raise SystemExit("tile completed a different iteration count")
if int(provenance["max_gaussians"]) != int(cfg.train.max_gaussians):
    raise SystemExit("tile used a different Gaussian cap")
if provenance.get("launcher_config_sha256") != launcher_hash:
    raise SystemExit("tile belongs to a different sealed server run")
implementation = Path(__import__(
    "rtk_splat.backends.gsplat", fromlist=["x"]
).__file__)
if provenance.get("training_implementation_sha256") != hashlib.sha256(
    implementation.read_bytes()
).hexdigest():
    raise SystemExit("tile was trained by different implementation code")
print(
    f"verified {tile}: {best['completed_training_steps']:,} steps, "
    f"best cc PSNR {best['best_metric']:.4f}, params {params_hash[:12]}..."
)
PY
}

TILE_RUN_RESULT=""
ensure_tile_run() {
    local tile="$1"
    local root="$WORKDIR/tile_runs/$PLAN_NAME/$tile"
    local completed="" highest=0 path base number
    local -a entries=()
    if [[ -d "$root" ]]; then
        shopt -s nullglob
        entries=("$root"/*)
        shopt -u nullglob
    fi
    for path in "${entries[@]}"; do
        [[ -d "$path" && ! -L "$path" ]] \
            || fail "unexpected non-directory tile artifact: $path"
        base="$(basename "$path")"
        [[ "$base" =~ ^${RUN_BASE}-attempt-([0-9]{4})$ ]] \
            || fail "unexpected run directory below $tile: $base"
        number=$((10#${BASH_REMATCH[1]}))
        ((number > highest)) && highest=$number
        if [[ -f "$path/params.pt" ]]; then
            verify_tile_run "$tile" "$path" \
                || fail "$tile has a completion marker that failed verification: $path"
            [[ -z "$completed" ]] \
                || fail "$tile has more than one completed attempt; select explicitly in a new publication workdir"
            completed="$path"
        else
            echo "PRESERVED: incomplete $tile attempt: $path"
        fi
    done
    if [[ -n "$completed" ]]; then
        echo "SKIP: $tile already has one verified completed attempt"
        TILE_RUN_RESULT="$completed"
        return
    fi
    ((highest < 9999)) || fail "$tile exhausted four-digit attempt names"
    local next=$((highest + 1)) run_name
    printf -v run_name '%s-attempt-%04d' "$RUN_BASE" "$next"
    path="$root/$run_name"
    execute_logged "train_${tile}_attempt_$(printf '%04d' "$next")" \
        "${CLI[@]}" train "${COMMON[@]}" \
        --pose-name "$POSE_NAME" --pose-artifact-root "$POSE_PARENT" \
        --tile-plan "$PLAN" --tile-id "$tile" --run-name "$run_name"
    verify_tile_run "$tile" "$path"
    TILE_RUN_RESULT="$path"
}

publish_scene() {
    local -a ids=() arguments=()
    mapfile -t ids < <(tile_ids)
    ((${#ids[@]} >= 2)) || fail "production publication needs at least two tiles"
    local tile
    for tile in "${ids[@]}"; do
        ensure_cloud "$tile"
        ensure_tile_run "$tile"
        arguments+=(--scene-tile-run "$tile=$TILE_RUN_RESULT")
    done
    if [[ -e "$SCENE" ]]; then
        "$PY" - "$SCENE" "${#ids[@]}" <<'PY'
import json, sys
from pathlib import Path
from rtk_splat.workflows.tile_scene import verify_tiled_scene

root = Path(sys.argv[1])
scene = verify_tiled_scene(root)
metrics = json.loads((root / "metrics.json").read_text())
expected = int(sys.argv[2])
if (
    scene.get("publication_mode") != "production"
    or scene.get("quality_passed") is not True
    or scene.get("metric_georeferencing_claim_eligible") is not True
    or int(metrics["completeness"]["n_completed_tiles"]) != expected
):
    raise SystemExit("existing final scene is not a complete production artifact")
print(f"verified existing production scene with {expected} tiles")
PY
        return
    fi
    execute_logged scene-publish \
        "${CLI[@]}" scene-publish --config "$CONFIG" --segment "$SEGMENT" \
        --workdir "$WORKDIR" --pose-name "$POSE_NAME" \
        --pose-artifact-root "$POSE_PARENT" --tile-plan "$PLAN" \
        "${arguments[@]}" --scene-name "$SCENE_NAME" \
        --scene-mode production --scene-opacity-threshold 0.05 \
        --max-combined-gaussians "$MAXIMUM_VISIBLE_GAUSSIANS" \
        --scene-device cuda
    "$PY" - "$SCENE" "${#ids[@]}" <<'PY'
import json, sys
from pathlib import Path
from rtk_splat.workflows.tile_scene import verify_tiled_scene

root = Path(sys.argv[1])
expected = int(sys.argv[2])
scene = verify_tiled_scene(root)
metrics = json.loads((root / "metrics.json").read_text())
if scene.get("metric_georeferencing_claim_eligible") is not True:
    raise SystemExit("published scene is not metric-georeferencing claim eligible")
if int(metrics["completeness"]["n_completed_tiles"]) != expected:
    raise SystemExit("published scene omitted a tile")
absolute = metrics["absolute"]
print("=== FULL-FIELD PRODUCTION SCENE ===")
print(f"tiles: {expected}")
print(f"PLY: {root / scene['splat_file']}")
print(
    f"absolute held-out: masked PSNR {absolute['psnr_masked']:.3f} dB; "
    f"corrected {absolute['psnr_masked_cc']:.3f} dB; "
    f"SSIM {absolute['ssim']:.4f}; LPIPS-cc {absolute['lpips_cc']:.4f}"
)
PY
}

case "$ACTION" in
    prepare)
        prepare_pipeline
        echo "PREPARE COMPLETE: $PLAN"
        ;;
    smoke)
        prepare_pipeline
        SMOKE_TILE="$(smoke_tile_id)"
        echo "SMOKE TILE: $SMOKE_TILE (highest planned train/validation workload)"
        ensure_cloud "$SMOKE_TILE"
        ensure_tile_run "$SMOKE_TILE"
        echo "SMOKE COMPLETE: $TILE_RUN_RESULT"
        ;;
    run)
        prepare_pipeline
        publish_scene
        echo "RUN COMPLETE: $SCENE"
        ;;
esac
