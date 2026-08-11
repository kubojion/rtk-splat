#!/usr/bin/env bash
# Controlled Option-A seam A/B on the cached 1,344-frame headland segment.
#
# This launcher deliberately does NOT ingest or inspect the 77-minute bag.  It
# consumes the immutable cached segment, accepted AB03 global poses, and the
# already-published two-tile TilePlan. One same-code monolithic control and the
# two 65k tile jobs run sequentially on the 8 GB laptop GPU. The core-owned
# merged model is then evaluated on the original 168 held-out frames against
# that control. The older accepted 24.36 dB model remains an absolute target,
# not the causal tiling control, because it used an earlier training backend.
#
#   bash scripts/experiments/headland_two_tile_overnight.sh plan
#   bash scripts/experiments/headland_two_tile_overnight.sh preflight
#   bash scripts/experiments/headland_two_tile_overnight.sh run
#   bash scripts/experiments/headland_two_tile_overnight.sh status
#
# A rerun may skip only fully verified stages:
#   bash scripts/experiments/headland_two_tile_overnight.sh run \
#     --workdir /path/to/the/existing/workdir --resume-existing
#
# There is no within-tile optimizer resume.  An interrupted tile directory is
# preserved and rejected; use a new workdir to restart that tile experiment.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"
CONFIG="$REPO/configs/reproductions/headland_stereo_ba.yaml"
SEGMENT="$HOME/agromap4d_work/field_turn_contract_v2_normalized/segment"
POSE_PARENT="$HOME/agromap4d_work/headland_feature_ab_03/arms/gpu/pose_artifacts"
POSE_NAME="feature-profile-ab-gpu-pose"
POSE_ARTIFACT="$POSE_PARENT/$POSE_NAME"
PLAN="$HOME/agromap4d_work/headland_tile_validation_v1/tile_plan_artifacts/headland-two-tile-seam-v1"
PLAN_NAME="headland-two-tile-seam-v1"
HISTORICAL_REFERENCE_RUN="$HOME/agromap4d_work/headland_feature_ab_03/arms/gpu/runs/headland_v2_gpu_65k"
FROZEN_CONTROL_CLOUD="$HOME/agromap4d_work/headland_feature_ab_03/arms/gpu/cloud_artifacts/$POSE_NAME/init_cloud.npz"
WORKDIR_DEFAULT="$HOME/agromap4d_work/headland_two_tile_controlled_overnight_v1"
WORKDIR="$WORKDIR_DEFAULT"
RUN_NAME="headland-two-tile-65k-v1"
CONTROL_RUN_NAME="headland-current-tree-monolith-65k-v1"
SCENE_NAME="headland-two-tile-seam-controlled-v1"
TILES=(tile-0000 tile-0001)

# Frozen input identities for this controlled comparison.  The TilePlan also
# seals every image/depth/pose byte and is re-verified against the live source.
CONFIG_SHA256="1a72b29a245cdbf8a597e8707f013a26ac67eafc7e1444dc70a28fe5d92513a8"
PLAN_MANIFEST_SHA256="4b7f2c5c1490b923743d547f2d0a61b68a0e6d2ffb88d891f05793072b109e5d"
PLAN_JSON_SHA256="400a2ab2f600676f06d3f795604d50e93814ac368fa304c77704c2a19dfdb07d"
HISTORICAL_PARAMS_SHA256="ffea25e6ac02d4fc3728cfba64d6aa67ad9b206435ad4cf8aa20be626ea4c59a"
HISTORICAL_METRICS_SHA256="65b2409acce214a3042c9c48f2a141f232d1782566da244aea97fd00b393129b"
HISTORICAL_PROVENANCE_SHA256="08500b7a64ea007c21c7bc7190056721f29ff502ce8048f1bc50b5c096fc036c"
FROZEN_CONTROL_CLOUD_SHA256="4e1ccc4acbd9094565520643802a6c05b9a9cbb36175f092e8d9081784618f32"

fail() { echo "FATAL: $*" >&2; exit 1; }

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

ACTION="${1:-plan}"
if [ $# -gt 0 ]; then shift; fi
RESUME_EXISTING=0
while [ $# -gt 0 ]; do
    case "$1" in
        --workdir)
            [ $# -ge 2 ] || fail "--workdir needs a path"
            WORKDIR="$(readlink -m "$2")"
            shift 2
            ;;
        --resume-existing)
            RESUME_EXISTING=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

case "$ACTION" in
    plan|preflight|run|status) ;;
    *) usage; fail "action must be plan, preflight, run, or status" ;;
esac
if [ "$ACTION" != run ] && [ "$RESUME_EXISTING" -ne 0 ]; then
    fail "--resume-existing is valid only with run"
fi

cloud_path() {
    printf '%s\n' "$WORKDIR/cloud_artifacts/$POSE_NAME/tile_plans/$PLAN_NAME/$1/init_cloud.npz"
}

control_cloud_path() {
    printf '%s\n' "$WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz"
}

run_path() {
    printf '%s\n' "$WORKDIR/tile_runs/$PLAN_NAME/$1/$RUN_NAME"
}

control_run_path() {
    printf '%s\n' "$WORKDIR/runs/$CONTROL_RUN_NAME"
}

scene_path() {
    printf '%s\n' "$WORKDIR/scene_artifacts/$SCENE_NAME"
}

print_plan() {
    cat <<EOF
Cached headland two-tile validation (the 77-minute bag is not touched)

  TilePlan:   $PLAN
  Tile 0000:  919 train / 132 validation frames
  Tile 0001:  879 train / 126 validation frames
  Control:    one current-code monolithic 65k model on all 1,176 train views
  Tiles:      two 65k models, each capped at 2,500,000 Gaussians
  Execution:  sequential on the 8 GB GPU; no tmux and no shutdown
  Expected:   about 12 hours; allow 14-16 hours
  New space:  normally 3-6 GiB; preflight requires at least 20 GiB free
  Output:     $WORKDIR

The final evaluation renders the merged, uniquely core-owned model on all 168
source validation frames. It compares against the same-code monolithic control
and does not average overlapping per-tile scores. The frozen 24.364/25.733 dB
model is reported separately as the historical headland-quality target.
The cached pose is legacy_unassessed, so every result remains PROVISIONAL and
is not eligible for a new metric-georeferencing claim.

Nothing is running.  Inspect only:
  bash scripts/experiments/headland_two_tile_overnight.sh preflight${WORKDIR:+ --workdir "$WORKDIR"}

Start when ready:
  bash scripts/experiments/headland_two_tile_overnight.sh run${WORKDIR:+ --workdir "$WORKDIR"}
EOF
}

preflight() {
    [ -x "$PY" ] || fail "Python is not executable: $PY"
    for path in "$CONFIG" "$SEGMENT/manifest.json" "$POSE_ARTIFACT/manifest.json" \
                "$PLAN/manifest.json" "$HISTORICAL_REFERENCE_RUN/params.pt" \
                "$FROZEN_CONTROL_CLOUD"; do
        [ -e "$path" ] || fail "required input is missing: $path"
    done
    [ "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA256" ] \
        || fail "frozen reproduction config changed"
    [ "$(sha256sum "$PLAN/manifest.json" | awk '{print $1}')" = "$PLAN_MANIFEST_SHA256" ] \
        || fail "selected TilePlan manifest changed"
    [ "$(sha256sum "$PLAN/tile_plan.json" | awk '{print $1}')" = "$PLAN_JSON_SHA256" ] \
        || fail "selected TilePlan JSON changed"
    [ "$(sha256sum "$FROZEN_CONTROL_CLOUD" | awk '{print $1}')" = "$FROZEN_CONTROL_CLOUD_SHA256" ] \
        || fail "frozen monolithic control cloud changed"

    if [ "$ACTION" = run ]; then
        if [ "$RESUME_EXISTING" -eq 0 ] && [ -e "$WORKDIR" ]; then
            fail "fresh workdir already exists: $WORKDIR (choose a new path)"
        fi
        if [ "$RESUME_EXISTING" -eq 1 ] && [ ! -d "$WORKDIR" ]; then
            fail "--resume-existing workdir does not exist: $WORKDIR"
        fi
    fi

    local active
    active="$(pgrep -af '[p]ython.*rtk_splat.*(workflows\.cli.*train|scene-publish)' || true)"
    [ -z "$active" ] || fail "another RTK-Splat train/scene stage is active: $active"

    local free_gib available_kib
    free_gib="$(df --output=avail -BG "$(dirname "$WORKDIR")" | tail -1 | tr -dc '0-9')"
    [ "$free_gib" -ge 20 ] || fail "only ${free_gib} GiB free; need at least 20 GiB"
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    [ "$available_kib" -ge $((8 * 1024 * 1024)) ] \
        || fail "less than 8 GiB RAM is available"
    local storage
    storage="$(findmnt -no SOURCE,FSTYPE --target "$(dirname "$WORKDIR")")"
    [[ "$storage" == /dev/nvme* ]] \
        || fail "overnight workdir is not on local NVMe: $storage"
    if [ -f /sys/class/power_supply/ADP0/online ]; then
        [ "$(cat /sys/class/power_supply/ADP0/online)" = 1 ] \
            || fail "laptop AC power is disconnected"
    fi
    [ -f "$HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth" ] \
        || fail "cached VGG LPIPS weights are missing; do not discover this offline overnight"

    "$PY" - "$CONFIG" "$SEGMENT" "$POSE_ARTIFACT" "$PLAN" \
        "$HISTORICAL_REFERENCE_RUN" "$HISTORICAL_PARAMS_SHA256" \
        "$HISTORICAL_METRICS_SHA256" "$HISTORICAL_PROVENANCE_SHA256" <<'PY'
import hashlib, json, sys
from pathlib import Path

import gsplat
import torch

from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.tiles import verify_tile_plan

cfg_path, segment, pose, plan_root, reference = map(Path, sys.argv[1:6])
expected_reference = {
    "params.pt": sys.argv[6],
    "metrics.json": sys.argv[7],
    "run_provenance.json": sys.argv[8],
}

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

reader = SegmentReader(segment).validate()
assert len(reader.frames["frame_id"]) == 1344
assert len(reader.manifest["train"]) == 1176
assert len(reader.manifest["val"]) == 168
plan = verify_tile_plan(
    plan_root, segment=segment, pose_root=pose, rehash_sources=True
)
assert plan["name"] == "headland-two-tile-seam-v1"
assert plan["provisional"] is True
assert plan["metric_georeferencing_claim_eligible"] is False
assert [len(t["frame_ids"]["train"]) for t in plan["tiles"]] == [919, 879]
assert [len(t["frame_ids"]["val"]) for t in plan["tiles"]] == [132, 126]

cfg = load_config(cfg_path)
assert int(cfg.train.iterations) == 65000
assert int(cfg.train.max_gaussians) == 2500000
assert int(cfg.cloud.max_points) == 4000000
assert abs(float(cfg.train.max_scale_m) - 0.20) < 1e-12
assert cfg.train.pose_opt.enabled is False
assert cfg.train.use_right_camera is False
for name, expected in expected_reference.items():
    assert sha(reference / name) == expected, f"frozen reference changed: {name}"

assert torch.__version__ == "2.4.1+cu121", torch.__version__
assert gsplat.__version__ == "1.5.3+pt24cu121", gsplat.__version__
assert torch.cuda.is_available(), "CUDA is unavailable"
free, total = torch.cuda.mem_get_info()
assert free >= 6000 * 1024 * 1024, f"only {free >> 20} MiB GPU memory is free"
print(
    f"verified TilePlan and all live sources; GPU {torch.cuda.get_device_name(0)} "
    f"has {free >> 20}/{total >> 20} MiB free"
)
print("exact schedule: 3 x 65,000 iterations, 2,500,000 Gaussian cap, 4,000,000 cloud cap")
print("scientific status: PROVISIONAL / metric-georeferencing claim INELIGIBLE")
PY
    echo "Preflight passed. No training was started."
}

write_or_verify_run_spec() {
    mkdir -p "$WORKDIR"
    "$PY" - "$WORKDIR/run_spec.json" "$REPO" "$0" "$CONFIG" "$SEGMENT" \
        "$POSE_ARTIFACT" "$PLAN" "$HISTORICAL_REFERENCE_RUN" \
        "$FROZEN_CONTROL_CLOUD" "$WORKDIR" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from rtk_splat.frontends.artifact import collect_package_state

target, repo, launcher, config, segment, pose, plan, reference, cloud, workdir = (
    Path(value).expanduser().resolve() for value in sys.argv[1:11]
)
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

identity = {
    "artifact_type": "headland_two_tile_overnight_spec",
    "package": collect_package_state(),
    "launcher_sha256": sha(launcher),
    "config": {"path": str(config), "sha256": sha(config)},
    "segment": {
        "path": str(segment),
        "manifest_sha256": sha(segment / "manifest.json"),
        "frames_sha256": sha(segment / "frames.npz"),
    },
    "pose": {"path": str(pose), "manifest_sha256": sha(pose / "manifest.json")},
    "tile_plan": {
        "path": str(plan),
        "manifest_sha256": sha(plan / "manifest.json"),
        "tile_plan_sha256": sha(plan / "tile_plan.json"),
    },
    "historical_reference": {
        "path": str(reference),
        "params_sha256": sha(reference / "params.pt"),
        "metrics_sha256": sha(reference / "metrics.json"),
        "provenance_sha256": sha(reference / "run_provenance.json"),
    },
    "frozen_control_cloud": {"path": str(cloud), "sha256": sha(cloud)},
    "workdir": str(workdir),
    "control_run_name": "headland-current-tree-monolith-65k-v1",
    "tiles": ["tile-0000", "tile-0001"],
    "control_iterations": 65000,
    "iterations_per_tile": 65000,
    "max_gaussians_per_tile": 2500000,
    "execution": "sequential",
    "within_tile_resume": False,
}
document = {"schema_version": 1, "identity": identity}
if target.exists():
    stored = json.loads(target.read_text())
    if stored != document:
        raise SystemExit("existing run_spec.json differs; use a new workdir")
    print("verified existing run specification")
else:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.link(temporary, target)
    temporary.unlink()
    print(f"published immutable run specification: {target}")
PY
}

verify_control_cloud() {
    local cloud="$1"
    "$PY" - "$CONFIG" "$SEGMENT" "$POSE_PARENT" "$POSE_NAME" \
        "$cloud" "$WORKDIR" "$FROZEN_CONTROL_CLOUD_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path

import numpy as np

from rtk_splat.backends.pose_evidence import (
    cloud_georeferencing_evidence,
    pose_georeferencing_evidence,
)
from rtk_splat.core.pose_artifacts import (
    load_pose_artifact,
    verify_cloud_matches_poses,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config

config, segment, pose_parent, pose_name, cloud, workdir, expected_hash = sys.argv[1:]
cfg = load_config(Path(config)); cfg.paths.workdir = Path(workdir)
cfg.paths.segment = Path(segment); cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(Path(segment)).validate()
viewmats, _ = load_pose_artifact(reader.root, cfg)
expected = pose_georeferencing_evidence(reader.root, cfg)
verify_cloud_matches_poses(
    Path(cloud),
    viewmats,
    require_fingerprint=True,
)
cloud_georeferencing_evidence(
    Path(cloud), expected,
    allow_failed_georeferencing_for_render=False,
)
with np.load(cloud, allow_pickle=False) as archive:
    points = int(len(archive["xyz"]))
if points <= 0 or points > 4_000_000:
    raise RuntimeError(f"invalid control cloud point count: {points}")
if hashlib.sha256(Path(cloud).read_bytes()).hexdigest() != expected_hash:
    raise RuntimeError("control cloud no longer matches the frozen SHA-256")
print(f"verified current-code monolithic control cloud: {points:,} points")
PY
}

stage_control_cloud() {
    local destination="$1"
    mkdir -p "$(dirname "$destination")"
    "$PY" - "$FROZEN_CONTROL_CLOUD" "$destination" \
        "$FROZEN_CONTROL_CLOUD_SHA256" <<'PY'
import hashlib, os, shutil, sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:3])
expected = sys.argv[3]
temporary = destination.with_name(f".{destination.name}.copying-{os.getpid()}")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

try:
    shutil.copyfile(source, temporary)
    if sha(temporary) != expected:
        raise RuntimeError("copied control cloud failed its frozen SHA-256")
    os.link(temporary, destination)
finally:
    temporary.unlink(missing_ok=True)
print(f"published isolated frozen control cloud: {destination}")
PY
}

verify_cloud() {
    local tile="$1" cloud="$2"
    "$PY" - "$CONFIG" "$SEGMENT" "$POSE_PARENT" "$POSE_NAME" "$PLAN" \
        "$tile" "$cloud" "$WORKDIR" <<'PY'
import sys
from pathlib import Path
from rtk_splat.backends.pose_evidence import pose_georeferencing_evidence
from rtk_splat.core.pose_artifacts import load_pose_artifact
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.cloud import verify_tile_cloud
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.tiles import load_tile_execution

config, segment, pose_parent, pose_name, plan, tile, cloud, workdir = sys.argv[1:]
cfg = load_config(Path(config)); cfg.paths.workdir = Path(workdir)
cfg.paths.segment = Path(segment); cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(Path(segment)).validate()
execution = load_tile_execution(Path(plan), tile, reader, cfg)
viewmats, _ = load_pose_artifact(reader.root, cfg)
points, masks = verify_tile_cloud(
    Path(cloud), execution, viewmats,
    georeferencing=pose_georeferencing_evidence(reader.root, cfg),
)
print(f"verified {tile} cloud: {points:,} points, {masks} packed context masks")
PY
}

verify_run() {
    local tile="$1" run="$2"
    "$PY" - "$CONFIG" "$SEGMENT" "$POSE_PARENT" "$POSE_NAME" "$PLAN" \
        "$tile" "$run" "$WORKDIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
from rtk_splat.backends.pose_evidence import (
    canonical_georeferencing_json, pose_georeferencing_evidence,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.export_splat import _completed_run_evidence
from rtk_splat.workflows.tiles import load_tile_execution

config, segment, pose_parent, pose_name, plan, tile, run, workdir = sys.argv[1:]
cfg = load_config(Path(config)); cfg.paths.workdir = Path(workdir)
cfg.paths.segment = Path(segment); cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(Path(segment)).validate()
execution = load_tile_execution(Path(plan), tile, reader, cfg)
expected = pose_georeferencing_evidence(reader.root, cfg)
georef, best, provenance, params, params_hash = _completed_run_evidence(
    Path(run), allow_failed_georeferencing_for_render=False
)
assert canonical_georeferencing_json(georef) == canonical_georeferencing_json(expected)
assert provenance.get("tile_plan") == execution.binding
assert best["completed_training_steps"] == 65000
implementation = Path(__import__(
    "rtk_splat.backends.gsplat", fromlist=["x"]
).__file__)
assert provenance["training_implementation_sha256"] == hashlib.sha256(
    implementation.read_bytes()
).hexdigest()
print(
    f"verified {tile} completed run: best step {best['best_step']:,}, "
    f"cc PSNR {best['best_metric']:.4f}, params {params_hash[:12]}..."
)
PY
}

verify_control_run() {
    local run="$1"
    "$PY" - "$CONFIG" "$SEGMENT" "$POSE_PARENT" "$POSE_NAME" \
        "$run" "$WORKDIR" "$FROZEN_CONTROL_CLOUD_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path

from rtk_splat.backends.pose_evidence import (
    canonical_georeferencing_json,
    pose_georeferencing_evidence,
)
from rtk_splat.core.segment import SegmentReader
from rtk_splat.workflows.configio import load_config
from rtk_splat.workflows.export_splat import _completed_run_evidence

config, segment, pose_parent, pose_name, run, workdir, cloud_hash = sys.argv[1:]
cfg = load_config(Path(config)); cfg.paths.workdir = Path(workdir)
cfg.paths.segment = Path(segment); cfg.pose.artifact = pose_name
cfg.pose.artifact_root = Path(pose_parent)
reader = SegmentReader(Path(segment)).validate()
expected = pose_georeferencing_evidence(reader.root, cfg)
georef, best, provenance, _, params_hash = _completed_run_evidence(
    Path(run), allow_failed_georeferencing_for_render=False
)
assert canonical_georeferencing_json(georef) == canonical_georeferencing_json(expected)
assert "tile_plan" not in provenance
assert best["completed_training_steps"] == 65000
assert provenance["available_training_views"] == 1176
assert provenance["initial_cloud_sha256"] == cloud_hash
implementation = Path(__import__(
    "rtk_splat.backends.gsplat", fromlist=["x"]
).__file__)
assert provenance["training_implementation_sha256"] == hashlib.sha256(
    implementation.read_bytes()
).hexdigest()
print(
    f"verified current-code monolithic control: best step {best['best_step']:,}, "
    f"cc PSNR {best['best_metric']:.4f}, params {params_hash[:12]}..."
)
PY
}

verify_controlled_training_identity() {
    "$PY" - "$(control_run_path)" "$(run_path tile-0000)" \
        "$(run_path tile-0001)" <<'PY'
import copy, hashlib, json, sys
from pathlib import Path

provenances = [
    json.loads((Path(root) / "run_provenance.json").read_text())
    for root in sys.argv[1:]
]
implementations = {
    item.get("training_implementation_sha256") for item in provenances
}
if len(implementations) != 1:
    raise SystemExit("control and tile runs used different training implementations")

def comparable_config(item):
    value = copy.deepcopy(item["effective_training_config"])
    value["train"].pop("run_name", None)
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()

if len({comparable_config(item) for item in provenances}) != 1:
    raise SystemExit("control and tile runs used different scientific settings")
print(
    "verified controlled A/B identity: same training code and settings; "
    "only TilePlan selection/context/ownership differs"
)
PY
}

run_stage() {
    local label="$1"; shift
    echo
    echo "--- $label ---"
    /usr/bin/time -v "$@"
}

run_pipeline() {
    write_or_verify_run_spec
    export RTK_SPLAT_CONFIG_SHA256="$CONFIG_SHA256"

    local tile cloud run control_cloud control_run
    control_cloud="$(control_cloud_path)"
    if [ -e "$control_cloud" ]; then
        [ "$RESUME_EXISTING" -eq 1 ] \
            || fail "control cloud already exists without --resume-existing"
        verify_control_cloud "$control_cloud" \
            || fail "control cloud is incomplete/tampered; use a new workdir"
    else
        stage_control_cloud "$control_cloud"
        verify_control_cloud "$control_cloud"
    fi

    for tile in "${TILES[@]}"; do
        cloud="$(cloud_path "$tile")"
        if [ -e "$cloud" ]; then
            [ "$RESUME_EXISTING" -eq 1 ] \
                || fail "$tile cloud already exists without --resume-existing"
            verify_cloud "$tile" "$cloud" \
                || fail "$tile cloud is incomplete/tampered; use a new workdir"
        else
            run_stage "$tile: context-cropped initial cloud" \
                "$PY" -m rtk_splat.workflows.cli cloud \
                --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
                --expected-frames 1344 --pose-name "$POSE_NAME" \
                --pose-artifact-root "$POSE_PARENT" \
                --tile-plan "$PLAN" --tile-id "$tile"
            verify_cloud "$tile" "$cloud"
        fi
    done

    control_run="$(control_run_path)"
    if [ -e "$control_run" ]; then
        [ "$RESUME_EXISTING" -eq 1 ] \
            || fail "control run already exists without --resume-existing"
        verify_control_run "$control_run" \
            || fail "control run is incomplete; use a new workdir"
    else
        run_stage "current-code monolithic control: 65k Gaussian training" \
            "$PY" -m rtk_splat.workflows.cli train \
            --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
            --expected-frames 1344 --pose-name "$POSE_NAME" \
            --pose-artifact-root "$POSE_PARENT" \
            --run-name "$CONTROL_RUN_NAME"
        verify_control_run "$control_run"
    fi

    for tile in "${TILES[@]}"; do
        run="$(run_path "$tile")"
        if [ -e "$run" ]; then
            [ "$RESUME_EXISTING" -eq 1 ] \
                || fail "$tile run already exists without --resume-existing"
            verify_run "$tile" "$run" \
                || fail "$tile run is incomplete; there is no exact optimizer resume. Use a new workdir."
        else
            run_stage "$tile: 65k Gaussian training (sequential)" \
                "$PY" -m rtk_splat.workflows.cli train \
                --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
                --expected-frames 1344 --pose-name "$POSE_NAME" \
                --pose-artifact-root "$POSE_PARENT" --run-name "$RUN_NAME" \
                --tile-plan "$PLAN" --tile-id "$tile"
            verify_run "$tile" "$run"
        fi
    done

    verify_controlled_training_identity

    local scene
    scene="$(scene_path)"
    if [ -e "$scene" ]; then
        [ "$RESUME_EXISTING" -eq 1 ] \
            || fail "scene already exists without --resume-existing"
        "$PY" - "$scene" <<'PY'
import sys
from rtk_splat.workflows.tile_scene import verify_tiled_scene
scene = verify_tiled_scene(sys.argv[1])
print(f"verified existing scene: quality_passed={scene['quality_passed']}")
PY
    else
        run_stage "merged core-owned scene: 168-frame held-out evaluation" \
            "$PY" -m rtk_splat.workflows.cli scene-publish \
            --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
            --pose-name "$POSE_NAME" --pose-artifact-root "$POSE_PARENT" \
            --tile-plan "$PLAN" \
            --scene-tile-run "tile-0000=$(run_path tile-0000)" \
            --scene-tile-run "tile-0001=$(run_path tile-0001)" \
            --scene-name "$SCENE_NAME" --reference-run "$control_run" \
            --reference-params-sha256 "$(sha256sum "$control_run/params.pt" | awk '{print $1}')" \
            --reference-metrics-sha256 "$(sha256sum "$control_run/metrics.json" | awk '{print $1}')" \
            --reference-provenance-sha256 "$(sha256sum "$control_run/run_provenance.json" | awk '{print $1}')" \
            --scene-opacity-threshold 0.05 --max-psnr-loss-db 0.30 \
            --max-ssim-loss 0.015 --max-lpips-cc-increase 0.030 \
            --max-combined-gaussians 5000000 --scene-device cuda
    fi

    "$PY" - "$(scene_path)" "$HISTORICAL_REFERENCE_RUN" <<'PY'
import json, sys
from pathlib import Path
from rtk_splat.workflows.tile_scene import verify_tiled_scene
root = Path(sys.argv[1])
historical = Path(sys.argv[2])
scene = verify_tiled_scene(root)
quality = json.loads((root / "quality.json").read_text())
metrics = json.loads((root / "metrics.json").read_text())
history = json.loads((historical / "metrics.json").read_text())[-1]
candidate = metrics["candidate"]
print()
print("=== TWO-TILE VERDICT ===")
print(f"quality passed: {quality['passed']}")
print(f"merged PLY: {root / scene['splat_file']}")
print(f"core-owned Gaussians: {scene['ownership']['core_owned_gaussians']:,}")
print(
    f"controlled tiling loss vs current-code monolith: "
    f"masked {metrics['loss_db']['psnr_masked']:+.3f} dB; "
    f"corrected {metrics['loss_db']['psnr_masked_cc']:+.3f} dB"
)
print(
    f"absolute delta vs historical headland target: "
    f"masked {history['psnr_masked'] - candidate['psnr_masked']:+.3f} dB; "
    f"corrected {history['psnr_masked_cc'] - candidate['psnr_masked_cc']:+.3f} dB"
)
print("status: PROVISIONAL / NOT ELIGIBLE FOR A NEW METRIC-GEOREFERENCING CLAIM")
PY
}

status() {
    echo "workdir: $WORKDIR"
    [ -f "$(control_cloud_path)" ] && echo "control cloud: present" \
        || echo "control cloud: missing"
    if [ -f "$(control_run_path)/params.pt" ]; then
        echo "current-code monolithic control: complete marker present"
    elif [ -d "$(control_run_path)" ]; then
        echo "current-code monolithic control: INCOMPLETE (not exactly resumable)"
    else
        echo "current-code monolithic control: missing"
    fi
    for tile in "${TILES[@]}"; do
        [ -f "$(cloud_path "$tile")" ] && echo "$tile cloud: present" \
            || echo "$tile cloud: missing"
        if [ -f "$(run_path "$tile")/params.pt" ]; then
            echo "$tile run: complete marker present"
        elif [ -d "$(run_path "$tile")" ]; then
            echo "$tile run: INCOMPLETE (not exactly resumable)"
        else
            echo "$tile run: missing"
        fi
    done
    [ -f "$(scene_path)/manifest.json" ] && echo "scene: published" \
        || echo "scene: missing"
    local active
    active="$(pgrep -af '[p]ython.*rtk_splat.*(workflows\.cli.*train|scene-publish)' || true)"
    [ -z "$active" ] && echo "active RTK-Splat stage: none" \
        || echo "active RTK-Splat stage: $active"
}

case "$ACTION" in
    plan)
        print_plan
        exit 0
        ;;
    status)
        status
        exit 0
        ;;
    preflight)
        preflight
        exit 0
        ;;
esac

# RUN: block suspend/lid sleep, but do not use tmux and do not power off.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    argv=(run --workdir "$WORKDIR")
    [ "$RESUME_EXISTING" -eq 1 ] && argv+=(--resume-existing)
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="RTK-Splat cached headland two-tile validation" \
        bash "$0" "${argv[@]}"
fi

LOCK="/tmp/rtk-splat-headland-two-tile-$(printf '%s' "$WORKDIR" | sha256sum | cut -c1-16).lock"
exec 9>"$LOCK"
flock -n 9 || fail "another launcher owns $WORKDIR"
preflight

mkdir -p "$WORKDIR/logs"
LOG="$WORKDIR/logs/overnight_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"
{
    echo "=== cached headland two-tile run started $(date -Is) ==="
    echo "workdir: $WORKDIR"
    echo "no tmux; suspend inhibited; no shutdown requested"
    run_pipeline
    echo "=== completed $(date -Is) ==="
} 2>&1 | tee "$LOG"
