#!/usr/bin/env bash
# Delayed launcher for the Rosario RGB pose-opt v3 training run.
#
# Validates the machine and the reused v2 artifacts immediately, so a broken
# setup fails in seconds instead of in two hours. Then waits DELAY_SECONDS and
# re-validates before starting the real run.
#
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_poseopt_v3_delayed.sh
#
# GPU headroom is the tight resource: the target requires 6000 MiB free and
# this machine reports ~6400-6700 MiB with nothing running, so a browser, IDE
# or video player left open will fail the launch-time preflight. That check
# therefore retries a few times rather than abandoning the night on a
# transient.
#
# Override the wait:    DELAY_SECONDS=3600 bash scripts/runs/..._delayed.sh
# Override the target:  RTK_SPLAT_WORKDIR=/path bash scripts/runs/..._delayed.sh
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/rosario_v2_sequence5_ppk_140_250_rgbd_poseopt_v3.sh"
DELAY_SECONDS="${DELAY_SECONDS:-7200}"
PREFLIGHT_RETRIES="${PREFLIGHT_RETRIES:-6}"
PREFLIGHT_RETRY_SECONDS="${PREFLIGHT_RETRY_SECONDS:-300}"

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rgbd_poseopt_v3_delayed_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }

# Hold sleep off for the wait AND the run, lid included. The guard stops
# systemd-inhibit re-entering itself, and the exported variable also stops the
# target script from opening a second, redundant inhibitor.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario RGB pose-opt v3 delayed run" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== delayed RGB pose-opt v3 launcher started $(date -Is) ==="
echo "target : $TARGET"
echo "delay  : ${DELAY_SECONDS}s"

echo
echo "--- [1/4] immediate preflight (fail fast) ---"
bash "$TARGET" --preflight-only

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
until bash "$TARGET" --preflight-only; do
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
echo "--- [4/4] running the pose-opt v3 training ---"
bash "$TARGET"

echo
echo "=== delayed launcher finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
