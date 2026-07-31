#!/usr/bin/env bash
# Complete, guarded GS evaluation of the 1,344-pair headland-turn segment.
# This script consumes an accepted reduced-Global-Mapper pose artifact. It
# never starts or retries COLMAP itself.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/reproduce/headland_global_mapper.sh"
CONFIG="${RTK_SPLAT_CONFIG:-$REPO_ROOT/configs/reproductions/headland_stereo_ba.yaml}"
POSE_ARTIFACT="${RTK_SPLAT_POSE_ARTIFACT:-colmap_global_headland_reduced_v1}"
RUN_NAME="${RTK_SPLAT_RUN:-tile_turn4_global_mapper_reduced_v1}"
PYTHON_BIN="${RTK_SPLAT_PYTHON:-/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python}"
COLMAP_EXE="${COLMAP_BIN:-${RTK_SPLAT_COLMAP:-/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap}}"
WAIT_HOURS=12
MODE="run"

usage() {
    cat <<'EOF'
Usage: scripts/reproduce/headland_global_gs_overnight.sh [options]

Options:
  --check             Validate static inputs and report pose readiness only.
                      No mapping, cloud construction, or training is started.
  --no-wait           Fail instead of waiting for the active pose job.
  --wait-hours N      Maximum time to wait for the current pose job (default 12).
  --python PATH       Isolated RTK-Splat Python interpreter.
  --colmap PATH       Validated COLMAP 4.1.1 executable.
  --config PATH       Headland reproduction configuration.
  --pose-artifact N   Accepted pose artifact to consume.
  --run N             New GS run name.
  -h, --help          Show this help.

The run covers the already extracted 1,344-pair (~450 s) headland-turn
segment. It does not scan the 142 GB source bag and it does not represent the
separate 77-minute full-field sequence.
EOF
}

NO_WAIT=0
while (($#)); do
    case "$1" in
        --check)
            MODE="check"
            shift
            ;;
        --no-wait)
            NO_WAIT=1
            shift
            ;;
        --wait-hours)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --wait-hours requires a value." >&2
                exit 2
            }
            WAIT_HOURS="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --python requires a path." >&2
                exit 2
            }
            PYTHON_BIN="$2"
            shift 2
            ;;
        --colmap)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --colmap requires a path." >&2
                exit 2
            }
            COLMAP_EXE="$2"
            shift 2
            ;;
        --config)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --config requires a path." >&2
                exit 2
            }
            CONFIG="$2"
            shift 2
            ;;
        --pose-artifact)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --pose-artifact requires a name." >&2
                exit 2
            }
            POSE_ARTIFACT="$2"
            shift 2
            ;;
        --run)
            [[ $# -ge 2 ]] || {
                echo "FATAL: --run requires a name." >&2
                exit 2
            }
            RUN_NAME="$2"
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

if [[ ! "$WAIT_HOURS" =~ ^[0-9]+$ ]] || [[ "$WAIT_HOURS" -gt 48 ]]; then
    echo "FATAL: --wait-hours must be an integer from 0 to 48." >&2
    exit 2
fi
if [[ ! "$POSE_ARTIFACT" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "FATAL: invalid pose artifact name: $POSE_ARTIFACT" >&2
    exit 2
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "FATAL: invalid run name: $RUN_NAME" >&2
    exit 2
fi
for required in "$CONFIG" "$LAUNCHER" "$PYTHON_BIN" "$COLMAP_EXE"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: required path is missing: $required" >&2
        exit 1
    fi
done
CONFIG="$(readlink -f "$CONFIG")"
PYTHON_BIN="$(readlink -f "$PYTHON_BIN")"
COLMAP_EXE="$(readlink -f "$COLMAP_EXE")"
if [[ ! -f "$CONFIG" ]]; then
    echo "FATAL: configuration is not a regular file: $CONFIG" >&2
    exit 1
fi
for executable in "$PYTHON_BIN" "$COLMAP_EXE"; do
    if [[ ! -x "$executable" ]]; then
        echo "FATAL: required executable is not executable: $executable" >&2
        exit 1
    fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi is unavailable; the GS run requires CUDA." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"

readarray -t CONFIG_VALUES < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from rtk_splat.configio import load_config

cfg = load_config(sys.argv[1])
print(cfg.paths.workdir)
print(int(cfg.train.iterations))
print(int(cfg.train.seed))
PY
)
WORKDIR="${CONFIG_VALUES[0]}"
ITERATIONS="${CONFIG_VALUES[1]}"
TRAIN_SEED="${CONFIG_VALUES[2]}"
CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
export RTK_SPLAT_CONFIG_SHA256="$CONFIG_SHA256"
SEGMENT="$WORKDIR/segment"
POSE_DIR="$SEGMENT/pose_artifacts/$POSE_ARTIFACT"
RUN_DIR="$WORKDIR/runs/$RUN_NAME"

check_segment() {
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from rtk_splat.configio import load_config

cfg = load_config(sys.argv[1])
segment = Path(cfg.paths.workdir) / "segment"
meta = json.loads((segment / "segment_meta.json").read_text())
n_frames = int(meta["n_frames"])
if n_frames != 1344:
    raise SystemExit(
        f"expected the 1,344-frame headland segment, found {n_frames}")

for name, expected in (
    ("viewmats.npy", (n_frames, 4, 4)),
    ("cam_centers.npy", (n_frames, 3)),
):
    array = np.load(segment / name, mmap_mode="r")
    if array.shape != expected:
        raise SystemExit(f"{name} shape {array.shape}, expected {expected}")

expected_ids = set(range(n_frames))
manifest = json.loads((segment / "manifest.json").read_text())
splits = [set(map(int, manifest[name])) for name in ("train", "val", "test")]
if any(a & b for index, a in enumerate(splits) for b in splits[index + 1:]):
    raise SystemExit("manifest train/val/test splits overlap")
if set().union(*splits) != expected_ids:
    raise SystemExit("manifest does not cover every headland frame exactly once")
expected_val = set(range(0, n_frames, int(cfg.train.holdout_every)))
if splits[1] != expected_val or splits[2]:
    raise SystemExit(
        "manifest is not the configured interleaved train/validation A/B split")

expected_files = {
    "left images": {
        segment / "images" / f"left_{index:06d}.jpg"
        for index in expected_ids
    },
    "right images": {
        segment / "images" / f"right_{index:06d}.jpg"
        for index in expected_ids
    },
    "depth maps": {
        segment / "depth" / f"{index:06d}.npz"
        for index in expected_ids
    },
}
for label, paths in expected_files.items():
    missing = [
        path for path in paths
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise SystemExit(
            f"{label}: {len(missing)} missing/empty; first is {missing[0]}")

print(
    f"segment verified: {n_frames} stereo pairs, {n_frames} depth maps, "
    f"split {len(splits[0])}/{len(splits[1])}/{len(splits[2])}")
PY
}

pose_ready() {
    "$PYTHON_BIN" - "$CONFIG" "$POSE_ARTIFACT" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

import numpy as np

from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import (
    load_pose_artifact, pose_fingerprint)

cfg = load_config(sys.argv[1])
cfg.pose.artifact = sys.argv[2]
segment = Path(cfg.paths.workdir) / "segment"
root = segment / "pose_artifacts" / sys.argv[2]
if not (root / "global" / ".global_mapper.done.json").is_file():
    raise SystemExit(1)
quality = json.loads((root / "quality.json").read_text())
if not quality.get("accepted_for_gs"):
    raise SystemExit(1)
viewmats, _ = load_pose_artifact(segment, cfg)
if len(viewmats) != 1344:
    raise SystemExit(1)
if quality.get("pose_fingerprint") != pose_fingerprint(viewmats):
    raise SystemExit(1)
frame_ids = np.load(root / "frame_ids.npy")
registered = np.load(root / "registered.npy")
if not np.array_equal(frame_ids, np.arange(1344, dtype=frame_ids.dtype)):
    raise SystemExit(1)
if registered.shape != (1344,) or not np.asarray(registered).all():
    raise SystemExit(1)
PY
}

verify_pose_for_consumption() {
    "$PYTHON_BIN" - "$CONFIG" "$POSE_ARTIFACT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

from rtk_splat.colmap_global import (
    _settings, _sha256_file, _verify_prepared_provenance,
    database_inventory)
from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import (
    load_pose_artifact, pose_fingerprint)

cfg = load_config(sys.argv[1])
cfg.pose.artifact = sys.argv[2]
segment = Path(cfg.paths.workdir) / "segment"
root = segment / "pose_artifacts" / sys.argv[2]
work = root / "global"
settings = _settings(cfg)
record = _verify_prepared_provenance(
    root, settings, verify_database=True)

if _sha256_file(root / "rtk_camera_centers.npy") != \
        record["rtk_centers_snapshot_sha256"]:
    raise SystemExit("immutable RTK-centre snapshot changed")
if _sha256_file(segment / "cam_centers.npy") != \
        record["source_segment_rtk_centers_sha256"]:
    raise SystemExit("canonical segment RTK centres changed")
if database_inventory(work / "database.db") != \
        record["database_inventory"]:
    raise SystemExit("isolated COLMAP database inventory changed unexpectedly")

done = json.loads((work / ".global_mapper.done.json").read_text())
solve_result = json.loads((work / "solve_result.json").read_text())
if done != solve_result:
    raise SystemExit("Global Mapper completion records disagree")
if not done.get("structural_gates_passed"):
    raise SystemExit("Global Mapper structural gates did not pass")
model_name = str(done["candidate_model_name"])
for model_root in ("sparse_global", "models_text"):
    model = work / model_root / model_name
    if not model.is_dir():
        raise SystemExit(f"completed model is missing: {model}")

quality = json.loads((root / "quality.json").read_text())
if not quality.get("accepted_for_gs"):
    raise SystemExit("pose quality record is not accepted for GS")
if not quality.get("metric_scale_preserved"):
    raise SystemExit("pose quality record does not preserve metric scale")
if quality.get("source_artifacts_modified"):
    raise SystemExit("pose record reports modified source artifacts")
failed = [
    name for name, check in quality["quality_gates"].items()
    if check.get("enforced", True) and not check["passed"]
]
if failed:
    raise SystemExit("enforced pose gates failed: " + ", ".join(failed))
if quality["model_stats"] != done["candidate_model_stats"]:
    raise SystemExit("pose/model completion records disagree")

viewmats, _ = load_pose_artifact(segment, cfg)
fingerprint = pose_fingerprint(viewmats)
if fingerprint != quality.get("pose_fingerprint"):
    raise SystemExit("published pose fingerprint does not match viewmats.npy")
frame_ids = np.load(root / "frame_ids.npy")
registered = np.load(root / "registered.npy")
if not np.array_equal(frame_ids, np.arange(len(viewmats),
                                           dtype=frame_ids.dtype)):
    raise SystemExit("pose frame IDs are incomplete or reordered")
if registered.shape != (len(viewmats),) or not registered.all():
    raise SystemExit("pose registration mask is incomplete")
print(
    f"pose verified for consumption: {len(viewmats)} frames, "
    f"fingerprint {fingerprint[:12]}..., all enforced gates passed")
if not quality.get("diagnostic_targets_passed", False):
    alignment = quality.get("fixed_scale_alignment", {})
    model = quality.get("model_stats", {})
    print(
        "WARNING: pose is structurally valid but diagnostic targets did not "
        "all pass; this is an exploratory A/B, not a claimed improvement.")
    print(
        "  fixed-scale RTK residual: "
        f"median={100.0 * float(alignment.get('median_m', float('nan'))):.1f} cm, "
        f"p95={100.0 * float(alignment.get('p95_m', float('nan'))):.1f} cm; "
        "reprojection="
        f"{float(model.get('mean_reprojection_error_px', float('nan'))):.3f} px")
PY
}

cloud_ready() {
    "$PYTHON_BIN" - "$CONFIG" "$POSE_ARTIFACT" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path

import numpy as np

from rtk_splat.configio import load_config
from rtk_splat.pose_artifacts import (
    cloud_path, load_pose_artifact, verify_cloud_matches_poses)

cfg = load_config(sys.argv[1])
cfg.pose.artifact = sys.argv[2]
segment = Path(cfg.paths.workdir) / "segment"
viewmats, _ = load_pose_artifact(segment, cfg)
path = cloud_path(segment, cfg)
verify_cloud_matches_poses(path, viewmats, require_fingerprint=True)
with np.load(path) as cloud:
    if not {"xyz", "rgb", "pose_fingerprint"} <= set(cloud.files):
        raise SystemExit("pose-specific cloud is missing required arrays")
    xyz = cloud["xyz"]
    rgb = cloud["rgb"]
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape
            or len(xyz) == 0):
        raise SystemExit(
            f"invalid pose-specific cloud shapes: xyz={xyz.shape}, rgb={rgb.shape}")
    if len(xyz) > int(cfg.cloud.max_points):
        raise SystemExit(
            f"cloud has {len(xyz)} points, above configured cap "
            f"{int(cfg.cloud.max_points)}")
    if not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise SystemExit("pose-specific cloud contains NaN/Inf")
PY
}

pose_process_active() {
    local processes
    processes="$(pgrep -af \
        'rtk_splat.cli global-(prepare|solve|export)|colmap global_mapper' \
        2>/dev/null || true)"
    if [[ "$processes" == *"$POSE_ARTIFACT"* ]]; then
        return 0
    fi
    if [[ "$POSE_ARTIFACT" == "colmap_global_headland_reduced_v1" ]] &&
       [[ "$CONFIG" == "$REPO_ROOT/configs/reproductions/headland_stereo_ba.yaml" ]]; then
        processes="$(pgrep -af 'headland_global_mapper.sh' 2>/dev/null || true)"
        [[ "$processes" == *"--pose-only"* ]]
        return
    fi
    return 1
}

check_ac_power() {
    local seen=0
    local online=0
    local supply
    for supply in /sys/class/power_supply/*; do
        [[ -f "$supply/type" ]] || continue
        if [[ "$(<"$supply/type")" == "Mains" ]]; then
            seen=1
            if [[ -f "$supply/online" ]] && [[ "$(<"$supply/online")" == "1" ]]; then
                online=1
            fi
        fi
    done
    if [[ "$seen" -eq 1 ]] && [[ "$online" -ne 1 ]]; then
        echo "FATAL: AC power is disconnected." >&2
        return 1
    fi
}

check_static_resources() {
    check_ac_power
    local free_gb
    local available_ram_gb
    free_gb="$(df --output=avail -BG "$WORKDIR" | tail -n 1 | tr -dc '0-9')"
    if [[ "$free_gb" -lt 30 ]]; then
        echo "FATAL: only ${free_gb}G free; require at least 30G." >&2
        return 1
    fi
    available_ram_gb="$(
        awk '/^MemAvailable:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo
    )"
    if [[ -z "$available_ram_gb" ]] || [[ "$available_ram_gb" -lt 8 ]]; then
        echo "FATAL: only ${available_ram_gb:-unknown} GiB RAM available; " \
             "require at least 8 GiB for cloud fusion." >&2
        return 1
    fi
    if [[ -e "$RUN_DIR" ]]; then
        echo "FATAL: refusing to overwrite existing GS run: $RUN_DIR" >&2
        return 1
    fi
    echo "resources verified: AC connected, ${free_gb}G disk free, " \
         "${available_ram_gb} GiB RAM available"
}

check_gpu_idle() {
    local compute_apps
    local free_vram_mib
    if ! compute_apps="$(nvidia-smi \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader 2>/dev/null)"; then
        echo "FATAL: nvidia-smi failed while checking CUDA ownership." >&2
        return 1
    fi
    if [[ -n "${compute_apps//[[:space:]]/}" ]]; then
        echo "FATAL: another CUDA compute process is active:" >&2
        echo "$compute_apps" >&2
        return 1
    fi
    if ! free_vram_mib="$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits 2>/dev/null | head -n 1 \
            | tr -dc '0-9')"; then
        echo "FATAL: nvidia-smi failed while checking free VRAM." >&2
        return 1
    fi
    if [[ -z "$free_vram_mib" ]] || [[ "$free_vram_mib" -lt 6000 ]]; then
        echo "FATAL: only ${free_vram_mib:-unknown} MiB VRAM free; " \
             "require at least 6000 MiB." >&2
        return 1
    fi
    echo "GPU verified: no compute process, ${free_vram_mib} MiB VRAM free"
}

postflight() {
    "$PYTHON_BIN" - \
        "$CONFIG" "$CONFIG_SHA256" "$RUN_DIR" "$POSE_ARTIFACT" \
        "$ITERATIONS" "$TRAIN_SEED" "$POSE_DIR/init_cloud.npz" \
        "$REPO_ROOT/rtk_splat/train.py" \
        "$SEGMENT/manifest.json" "$SEGMENT/segment_meta.json" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from rtk_splat.pose_artifacts import pose_fingerprint

(
    cfg_path, expected_cfg_hash, run_path, pose_artifact,
    expected_iterations, expected_seed, cloud_path, train_source,
    manifest_path, segment_meta_path,
) = sys.argv[1:]
run = Path(run_path)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if sha256(cfg_path) != expected_cfg_hash:
    raise SystemExit("configuration changed during the overnight run")
for name in ("run_provenance.json", "metrics.json", "params.pt", "splat.ply"):
    path = run / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"overnight output is missing or empty: {path}")
provenance = json.loads((run / "run_provenance.json").read_text())
if provenance.get("pose_artifact") != pose_artifact:
    raise SystemExit("run provenance references the wrong pose artifact")
if int(provenance.get("iterations", -1)) != int(expected_iterations):
    raise SystemExit("run provenance has the wrong iteration count")
if int(provenance.get("train_seed", -1)) != int(expected_seed):
    raise SystemExit("run provenance has the wrong training seed")
if provenance.get("launcher_config_sha256") != expected_cfg_hash:
    raise SystemExit("run provenance has the wrong launcher configuration hash")
if provenance.get("initial_cloud_sha256") != sha256(cloud_path):
    raise SystemExit("run provenance has the wrong initial-cloud hash")
if provenance.get("training_implementation_sha256") != sha256(train_source):
    raise SystemExit("training implementation changed during the run")
if provenance.get("manifest_sha256") != sha256(manifest_path):
    raise SystemExit("training manifest changed or has the wrong provenance")
if provenance.get("segment_meta_sha256") != sha256(segment_meta_path):
    raise SystemExit("segment metadata changed or has the wrong provenance")
effective = provenance.get("effective_training_config")
if not isinstance(effective, dict):
    raise SystemExit("effective training configuration was not recorded")
effective_hash = hashlib.sha256(json.dumps(
    effective, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if provenance.get("effective_training_config_sha256") != effective_hash:
    raise SystemExit("effective training configuration hash is invalid")

pose_root = Path(cloud_path).parent
quality = json.loads((pose_root / "quality.json").read_text())
viewmats = np.load(pose_root / "viewmats.npy")
actual_pose_fingerprint = pose_fingerprint(viewmats)
if provenance.get("pose_fingerprint") != actual_pose_fingerprint:
    raise SystemExit("run provenance has the wrong pose fingerprint")
if quality.get("pose_fingerprint") != actual_pose_fingerprint:
    raise SystemExit("pose quality record changed during the run")

history = json.loads((run / "metrics.json").read_text())
if not history or int(history[-1]["step"]) != int(expected_iterations):
    raise SystemExit("metrics do not contain the configured final iteration")
final = history[-1]
required_metrics = (
    "psnr", "psnr_masked", "psnr_near", "psnr_masked_cc",
    "lpips", "lpips_cc", "ssim", "loss",
)
for name in required_metrics:
    value = final.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SystemExit(f"final metric is missing or non-finite: {name}={value!r}")
manifest = json.loads(Path(manifest_path).read_text())
if int(final.get("n_eval", -1)) != len(manifest["val"]):
    raise SystemExit("final metrics do not cover the full validation split")
per_frame = final.get("per_frame", {})
if per_frame.get("eval_ids") != manifest["val"]:
    raise SystemExit("per-frame metrics do not match the validation frame IDs")
for name in ("brightness", "psnr_masked"):
    values = per_frame.get(name)
    if (not isinstance(values, list) or len(values) != len(manifest["val"])
            or not all(math.isfinite(float(value)) for value in values)):
        raise SystemExit(f"invalid per-frame metric payload: {name}")

checkpoint = torch.load(run / "params.pt", map_location="cpu",
                        weights_only=True)
required_params = ("means", "quats", "scales", "opacities", "sh0", "shN")
if any(name not in checkpoint for name in required_params):
    raise SystemExit("checkpoint is missing required Gaussian tensors")
n_gaussians = int(final.get("n_gaussians", -1))
if n_gaussians <= 0:
    raise SystemExit("final Gaussian count is invalid")
for name in required_params:
    tensor = checkpoint[name]
    if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != n_gaussians:
        raise SystemExit(f"checkpoint tensor has invalid shape: {name}")
    if not bool(torch.isfinite(tensor).all()):
        raise SystemExit(f"checkpoint tensor contains NaN/Inf: {name}")

vertex_count = None
vertex_properties = 0
in_vertex_element = False
header_bytes = None
ply_format = None
with (run / "splat.ply").open("rb") as stream:
    for _ in range(100):
        line = stream.readline()
        if not line:
            break
        decoded = line.decode("ascii", errors="strict").strip()
        if decoded.startswith("format "):
            ply_format = decoded.split()[1]
        if decoded.startswith("element vertex "):
            vertex_count = int(decoded.split()[-1])
            in_vertex_element = True
        elif decoded.startswith("element "):
            in_vertex_element = False
        elif decoded.startswith("property ") and in_vertex_element:
            fields = decoded.split()
            if len(fields) != 3 or fields[1] != "float":
                raise SystemExit(
                    f"unexpected PLY vertex property: {decoded!r}")
            vertex_properties += 1
        if decoded == "end_header":
            header_bytes = stream.tell()
            break
if vertex_count is None or not (0 < vertex_count <= n_gaussians):
    raise SystemExit(f"PLY has invalid vertex count: {vertex_count!r}")
if (ply_format != "binary_little_endian" or not vertex_properties
        or header_bytes is None):
    raise SystemExit("PLY header is incomplete or has an unexpected format")
expected_ply_bytes = header_bytes + vertex_count * vertex_properties * 4
if (run / "splat.ply").stat().st_size != expected_ply_bytes:
    raise SystemExit(
        "PLY payload size does not match its vertex count/properties")
print(
    f"OVERNIGHT COMPLETE: step={final['step']}, "
    f"gaussians={n_gaussians}, PLY_vertices={vertex_count}, "
    f"masked_psnr={final['psnr_masked']:.4f}, "
    f"ssim={final['ssim']:.4f}, "
    f"lpips={final['lpips']:.4f}")
PY
}

if [[ "$MODE" == "run" ]] &&
   [[ -z "${RTK_SPLAT_OVERNIGHT_INHIBITED:-}" ]]; then
    if ! command -v systemd-inhibit >/dev/null 2>&1; then
        echo "FATAL: systemd-inhibit is required for an unattended run." >&2
        exit 1
    fi
    export RTK_SPLAT_OVERNIGHT_INHIBITED=1
    REEXEC_ARGS=(
        --wait-hours "$WAIT_HOURS"
        --python "$PYTHON_BIN"
        --colmap "$COLMAP_EXE"
        --config "$CONFIG"
        --pose-artifact "$POSE_ARTIFACT"
        --run "$RUN_NAME"
    )
    if [[ "$NO_WAIT" -eq 1 ]]; then
        REEXEC_ARGS+=(--no-wait)
    fi
    exec systemd-inhibit --what=sleep:idle \
        --why="RTK-Splat complete headland GS evaluation" \
        bash "$SCRIPT_PATH" "${REEXEC_ARGS[@]}"
fi

if [[ "$MODE" == "run" ]]; then
    if ! command -v flock >/dev/null 2>&1; then
        echo "FATAL: flock is required to prevent duplicate overnight runs." >&2
        exit 1
    fi
    mkdir -p "$WORKDIR/logs"
    exec 9>"$WORKDIR/logs/.headland_global_gs_overnight.lock"
    if ! flock -n 9; then
        echo "FATAL: another headland overnight wrapper is already active." >&2
        exit 1
    fi
    OVERNIGHT_LOG="$WORKDIR/logs/headland_global_gs_overnight_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee -a "$OVERNIGHT_LOG") 2>&1
    echo "Logging to $OVERNIGHT_LOG"
    if [[ -z "${TMUX:-}" ]]; then
        echo "NOTICE: keep this terminal open for the full run, or launch the " \
             "wrapper inside tmux."
    fi
fi

echo "Configuration:  $CONFIG"
echo "Segment:        $SEGMENT"
echo "Pose artifact:  $POSE_ARTIFACT"
echo "GS run:         $RUN_NAME"
echo "Training:       $ITERATIONS iterations, seed $TRAIN_SEED"
echo "Source bag:     not required (canonical segment is complete)"

check_segment
check_static_resources

# Exercise the exact environments, source artifact, tests, and launcher without
# starting any long stage.
bash "$LAUNCHER" --check --resume --with-gs \
    --config "$CONFIG" \
    --pose-artifact "$POSE_ARTIFACT" \
    --run "$RUN_NAME" \
    --python "$PYTHON_BIN" \
    --colmap "$COLMAP_EXE"

if [[ "$MODE" == "check" ]]; then
    check_gpu_idle
    if pose_ready && ! pose_process_active; then
        verify_pose_for_consumption
        echo "READY: accepted pose is complete; the overnight GS run can start."
    elif pose_process_active; then
        echo "PENDING: static checks passed; the protected pose job is still active."
    else
        echo "PENDING: static checks passed, but no accepted pose is available."
    fi
    exit 0
fi

started="$(date +%s)"
deadline="$((started + WAIT_HOURS * 3600))"
while true; do
    pose_is_ready=0
    pose_is_active=0
    if pose_ready; then
        pose_is_ready=1
    fi
    if pose_process_active; then
        pose_is_active=1
    fi
    if [[ "$pose_is_ready" -eq 1 ]] && [[ "$pose_is_active" -eq 0 ]]; then
        break
    fi
    if [[ "$NO_WAIT" -eq 1 ]]; then
        echo "FATAL: accepted pose is not ready and --no-wait was requested." >&2
        exit 1
    fi
    if [[ "$pose_is_active" -eq 0 ]]; then
        # Close the solve->export process-transition race before declaring
        # that the protected producer died.
        sleep 2
        if pose_ready && ! pose_process_active; then
            break
        fi
        if pose_process_active; then
            continue
        fi
        echo "FATAL: pose producer stopped without publishing an accepted pose." >&2
        echo "Inspect: $POSE_DIR/global/global.log" >&2
        exit 1
    fi
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
        echo "FATAL: pose was not ready within ${WAIT_HOURS} hours." >&2
        exit 1
    fi
    echo "waiting for the protected pose solve/export: $(date --iso-8601=seconds)"
    sleep 30
done

verify_pose_for_consumption
check_gpu_idle
echo "Pose accepted and GPU idle; starting complete headland GS run."

if [[ -e "$POSE_DIR/init_cloud.npz" ]]; then
    if ! cloud_ready; then
        echo "FATAL: existing pose-specific cloud is incomplete or mismatched." >&2
        exit 1
    fi
    echo "Pose-specific initial cloud already exists and matches the pose."
else
    "$PYTHON_BIN" -m rtk_splat.cli cloud \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT"
    if ! cloud_ready; then
        echo "FATAL: cloud construction did not publish a valid artifact." >&2
        exit 1
    fi
fi

"$PYTHON_BIN" -m rtk_splat.cli train \
    --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" --run "$RUN_NAME"
if ! "$PYTHON_BIN" -m rtk_splat.cli diagnose \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME"; then
    echo "WARNING: brightness diagnostic failed; primary training is complete." >&2
fi
if ! "$PYTHON_BIN" -m rtk_splat.cli evalonly \
        --config "$CONFIG" --pose-artifact "$POSE_ARTIFACT" \
        --run "$RUN_NAME" --split train --max-frames 64; then
    echo "WARNING: train-view eval-only pass failed; primary metrics remain." >&2
fi

postflight
