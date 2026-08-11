#!/usr/bin/env bash
# Delayed launcher for the Rosario RGB-D diagnostic colour-GS run.
#
# Validates the machine and the sealed inputs immediately, so a broken setup
# fails in seconds instead of in an hour. Then waits DELAY_SECONDS and
# re-validates before starting the real run.
#
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_delayed.sh
#
# GPU headroom is the tight resource here -- the target requires 6000 MiB free
# and this machine reports about 6371 MiB with nothing running. A browser or
# IDE left open will fail the launch-time preflight, so that one retries a few
# times instead of abandoning the night on a transient.
#
# Override the wait:    DELAY_SECONDS=1800 bash scripts/runs/..._delayed.sh
# Override the target:  RTK_SPLAT_WORKDIR=/path bash scripts/runs/..._delayed.sh
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_overnight.sh"
WORKDIR="${RTK_SPLAT_WORKDIR:-/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v1}"
DELAY_SECONDS="${DELAY_SECONDS:-3600}"
PREFLIGHT_RETRIES="${PREFLIGHT_RETRIES:-6}"
PREFLIGHT_RETRY_SECONDS="${PREFLIGHT_RETRY_SECONDS:-300}"

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rgbd_diagnostic_delayed_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }
[ -e "$WORKDIR" ] && { echo "FATAL: workdir already exists: $WORKDIR" >&2; exit 1; }

# Hold sleep off for the wait AND the run, and survive a closed lid. The guard
# stops systemd-inhibit re-entering itself after the exec.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario RGB-D diagnostic delayed run" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== delayed RGB-D diagnostic launcher started $(date -Is) ==="
echo "target  : $TARGET"
echo "workdir : $WORKDIR"
echo "delay   : ${DELAY_SECONDS}s"

echo
echo "--- [1/4] immediate preflight (fail fast) ---"
bash "$TARGET" preflight --workdir "$WORKDIR"

start_epoch=$(date +%s)
launch_epoch=$((start_epoch + DELAY_SECONDS))
echo
echo "--- [2/4] waiting until $(date -d "@$launch_epoch" -Is) ---"
while :; do
    now=$(date +%s)
    remaining=$((launch_epoch - now))
    [ "$remaining" -le 0 ] && break
    printf '    %s  %d min remaining\n' "$(date +%H:%M:%S)" $(((remaining + 59) / 60))
    if [ "$remaining" -gt 900 ]; then sleep 900; else sleep "$remaining"; fi
done

echo
echo "--- [3/4] preflight at launch time ---"
attempt=1
until bash "$TARGET" preflight --workdir "$WORKDIR"; do
    if [ "$attempt" -ge "$PREFLIGHT_RETRIES" ]; then
        echo "FATAL: preflight still failing after $attempt attempts; not starting" >&2
        exit 1
    fi
    echo "    preflight failed (attempt $attempt/$PREFLIGHT_RETRIES);" \
         "retrying in $((PREFLIGHT_RETRY_SECONDS / 60)) min -- close GPU-heavy apps"
    attempt=$((attempt + 1))
    sleep "$PREFLIGHT_RETRY_SECONDS"
done

echo
echo "--- [4/4] running the RGB-D diagnostic experiment ---"
bash "$TARGET" run --acknowledge-diagnostic-render-only --workdir "$WORKDIR"

echo
echo "=== delayed launcher finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
