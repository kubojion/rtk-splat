#!/usr/bin/env bash
# Prepare the immutable full-field segment locally (bag -> stereo -> SGBM ->
# portable checksum-sealed segment). This does not run COLMAP or GS.
set -Eeuo pipefail

# Keep a multi-hour local ingest alive when the terminal is unattended. The
# same argv is re-executed exactly once under systemd-logind; no tmux/session
# manager is required. Plan, preflight, and status do not take an inhibitor.
if [[ "${1:-plan}" == run && -z "${RTK_SPLAT_INHIBITED:-}" ]] \
    && command -v systemd-inhibit >/dev/null 2>&1; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit \
        --what=sleep:idle:handle-lid-switch \
        --who=rtk-splat-full-field-preprocess \
        --why="immutable full-field ingest and SGBM depth" \
        --mode=block -- "$0" "$@"
fi

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"
CONFIG="$REPO/configs/sequences/field1_0703_full_77min.yaml"
WORKDIR_DEFAULT="$HOME/agromap4d_work/field1_0703_full_77min"
WORKDIR="$WORKDIR_DEFAULT"
RESUME=0
ACTION="${1:-plan}"
if (($#)); then shift; fi

fail() { echo "FATAL: $*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --workdir)
            (($# >= 2)) || fail "--workdir needs a path"
            WORKDIR="$(readlink -m "$2")"
            shift 2
            ;;
        --resume-existing)
            RESUME=1
            shift
            ;;
        -h|--help)
            ACTION=help
            shift
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

case "$ACTION" in
    plan|preflight|run|status|help) ;;
    *) fail "action must be plan, preflight, run, or status" ;;
esac
if [[ "$ACTION" != run && "$RESUME" -ne 0 ]]; then
    fail "--resume-existing is valid only with run"
fi

SOURCE="$WORKDIR/segments/field1-0703-full77-source-v1"
DEPTH="$WORKDIR/segments/field1-0703-full77-sgbm-v1"
PORTABLE="$WORKDIR/transfer/field1-0703-full77-portable-v1"
LOGS="$WORKDIR/logs"
STATE="$WORKDIR/local_prepare_state"

usage() {
    cat <<EOF
Usage:
  bash scripts/runs/field1_0703_full_local_prepare.sh plan
  bash scripts/runs/field1_0703_full_local_prepare.sh preflight
  bash scripts/runs/field1_0703_full_local_prepare.sh run
  bash scripts/runs/field1_0703_full_local_prepare.sh status

Options:
  --workdir PATH       New local artifact root (default: $WORKDIR_DEFAULT)
  --resume-existing    Reuse only strongly verified completed stages

The run stays in the foreground, uses no tmux, inhibits suspend, and writes
one log per stage. It never starts COLMAP or Gaussian training.
EOF
}

print_plan() {
    cat <<EOF
Full-field local preprocessing only

  config:    $CONFIG
  workdir:   $WORKDIR
  source:    $SOURCE
  SGBM:      $DEPTH
  transfer:  $PORTABLE

Expected selected stereo pairs: about 13,800 (measured count is authoritative)
Expected final unique data:     about 40 GB / 37 GiB
USB 2 estimate:                 2 h 10-40 min ingest + 30-55 min depth
USB 3 estimate:                 25-60 min ingest + 30-55 min depth

The transfer directory contains regular files only and a terminal SHA-256
inventory. Hardlinks avoid duplicating local image/depth blocks; rsync sends
normal file contents to the server.
EOF
}

bag_path() {
    "$PY" - "$CONFIG" <<'PY'
import sys
from rtk_splat.workflows.configio import load_config
cfg = load_config(sys.argv[1])
if len(cfg.paths.bags) != 1:
    raise SystemExit("full-field config must resolve exactly one bag")
print(cfg.paths.bags[0])
PY
}

orphaned_staging() {
    local output parent base candidate
    for output in "$SOURCE" "$DEPTH" "$PORTABLE"; do
        parent="$(dirname "$output")"
        base="$(basename "$output")"
        [[ -d "$parent" ]] || continue
        while IFS= read -r candidate; do
            printf '%s\n' "$candidate"
        done < <(
            find "$parent" -mindepth 1 -maxdepth 1 -type d \
                \( -name ".$base.payloads-*" -o -name ".$base.writing-*" \) \
                -print
        )
    done
}

preflight() {
    [[ -x "$PY" ]] || fail "Python is not executable: $PY"
    [[ -f "$CONFIG" ]] || fail "config is missing: $CONFIG"
    local bag
    bag="$(bag_path)"
    [[ -e "$bag" ]] || fail "bag is missing: $bag"
    "$PY" - <<'PY'
import cv2, numpy, rosbags, yaml
from rtk_splat.adapters.records import StagedPayload
assert cv2.__version__
assert numpy.__version__
assert StagedPayload
print("verified memory-bounded ROS2 ingest dependencies")
PY
    local active
    active="$(pgrep -af '[p]ython.*rtk_splat.*workflows\.cli.*(ingest|depth|segment-materialize)' || true)"
    [[ -z "$active" ]] || fail "another preprocessing stage is active: $active"
    local orphaned
    orphaned="$(orphaned_staging)"
    [[ -z "$orphaned" ]] || fail \
        "orphaned staging from a hard interruption exists; inspect it and remove only after confirming no process is active: $orphaned"

    local space_target free_gib available_kib
    space_target="$(dirname "$WORKDIR")"
    while [[ ! -e "$space_target" ]]; do space_target="$(dirname "$space_target")"; done
    free_gib="$(df --output=avail -BG "$space_target" | tail -1 | tr -dc '0-9')"
    [[ "$free_gib" -ge 50 ]] || fail "only ${free_gib} GiB free; need at least 50 GiB"
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    [[ "$available_kib" -ge $((4 * 1024 * 1024)) ]] \
        || fail "less than 4 GiB RAM is currently available"

    echo "Bag storage: $(findmnt -no SOURCE,FSTYPE --target "$bag")"
    echo "USB topology (Mass Storage should show 5000M or 10000M for the real run):"
    lsusb -t | sed -n '/Mass Storage/p' || true
    if lsusb -t | grep -q 'Mass Storage.*480M'; then
        echo "WARNING: storage is visible at USB 2 speed; reconnecting at USB 3 is strongly recommended."
    fi
    if [[ -f /sys/class/power_supply/ADP0/online ]]; then
        [[ "$(< /sys/class/power_supply/ADP0/online)" == 1 ]] \
            || fail "laptop AC power is disconnected"
    fi
    echo "Preflight passed; no artifact was created."
}

verify_stage() {
    local stage="$1" path="$2"
    case "$stage" in
        ingest|depth)
            "$PY" - "$path" <<'PY'
import sys
from rtk_splat.core.segment import SegmentReader
reader = SegmentReader(sys.argv[1]).validate()
print(f"verified {reader.meta['n_frames']} frames at {reader.root}")
PY
            ;;
        materialize)
            "$PY" - "$path" <<'PY'
import json, sys
from rtk_splat.workflows.segment_transfer import verify_portable_segment
print(json.dumps(verify_portable_segment(sys.argv[1]), indent=2, sort_keys=True))
PY
            ;;
        *) fail "unknown stage verifier: $stage" ;;
    esac
}

fingerprint() {
    {
        sha256sum "$0"
        # Bind every authored configuration layer (sequence, inherited
        # profile, and robot calibration), not merely the leaf sequence YAML.
        "$PY" - "$CONFIG" <<'PY'
import sys
from rtk_splat.workflows.configio import load_config

cfg = load_config(sys.argv[1])
for item in cfg.runtime_resolution.source_files:
    print(f"{item.role}\0{item.path}\0{item.sha256}")
PY
        find "$REPO/src/rtk_splat" -type f -name '*.py' -print0 \
            | LC_ALL=C sort -z | xargs -0 sha256sum
    } | sha256sum | awk '{print $1}'
}

command_digest() {
    printf '%q\000' "$@" | sha256sum | awk '{print $1}'
}

run_stage() {
    local stage="$1" output="$2"
    shift 2
    local marker="$STATE/$stage.done" digest code_hash stamp log
    digest="$(command_digest "$@")"
    code_hash="$(fingerprint)"
    if [[ -f "$marker" ]]; then
        [[ "$RESUME" -eq 1 ]] || fail "stage marker exists; use --resume-existing: $marker"
        local recorded_code recorded_command
        read -r recorded_code recorded_command < "$marker"
        [[ "$recorded_code" == "$code_hash" ]] || fail "$stage code/config changed"
        [[ "$recorded_command" == "$digest" ]] || fail "$stage command changed"
        verify_stage "$stage" "$output"
        echo "RESUME: verified completed stage $stage"
        return
    fi
    [[ ! -e "$output" ]] || fail "$stage output exists without a receipt: $output"
    stamp="$(date '+%Y%m%d_%H%M%S')"
    log="$LOGS/${stage}_${stamp}.log"
    echo "RUN: $stage"
    printf '  %q' "$@"; printf '\n  log: %s\n' "$log"
    /usr/bin/time -v "$@" 2>&1 | tee "$log"
    verify_stage "$stage" "$output"
    (set -o noclobber; printf '%s %s\n' "$code_hash" "$digest" > "$marker") \
        || fail "refusing to replace $marker"
}

status() {
    local stage output
    for pair in "ingest:$SOURCE" "depth:$DEPTH" "materialize:$PORTABLE"; do
        stage="${pair%%:*}"; output="${pair#*:}"
        if [[ -f "$STATE/$stage.done" && -d "$output" ]]; then
            echo "$stage: receipt and output present"
        elif [[ -e "$output" || -f "$STATE/$stage.done" ]]; then
            echo "$stage: INCOMPLETE (inspect before retry)"
        else
            echo "$stage: pending"
        fi
    done
    pgrep -af '[p]ython.*rtk_splat.*workflows\.cli.*(ingest|depth|segment-materialize)' || true
}

if [[ "$ACTION" == help ]]; then usage; exit 0; fi
if [[ "$ACTION" == plan ]]; then print_plan; exit 0; fi
if [[ "$ACTION" == status ]]; then status; exit 0; fi
if [[ "$ACTION" == preflight ]]; then preflight; exit 0; fi

preflight
if [[ -e "$WORKDIR" ]]; then
    [[ "$RESUME" -eq 1 ]] || fail "workdir exists; choose a new one or use --resume-existing"
else
    [[ "$RESUME" -eq 0 ]] || fail "--resume-existing needs an existing workdir"
fi
mkdir -p "$LOGS" "$STATE" "$(dirname "$SOURCE")" "$(dirname "$PORTABLE")"

CLI=("$PY" -m rtk_splat.workflows.cli)
COMMON=(--config "$CONFIG" --workdir "$WORKDIR")
run_stage ingest "$SOURCE" \
    "${CLI[@]}" ingest "${COMMON[@]}" --segment "$SOURCE"
run_stage depth "$DEPTH" \
    "${CLI[@]}" depth "${COMMON[@]}" --segment "$SOURCE" \
    --derived-segment "$DEPTH"
run_stage materialize "$PORTABLE" \
    "${CLI[@]}" segment-materialize "${COMMON[@]}" --segment "$DEPTH" \
    --portable-segment "$PORTABLE" --link-mode auto

echo "COMPLETE: portable segment ready for transfer: $PORTABLE"
