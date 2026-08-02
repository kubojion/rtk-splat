#!/usr/bin/env bash
# Delayed launcher for the fresh pose-prior experiment.
#
# Starts immediately, validates the environment straight away so a broken setup
# fails in seconds rather than in an hour, then waits DELAY_SECONDS before
# re-validating and running the real experiment.
#
#   bash scripts/experiments/citrusfarm_pose_prior_delayed.sh
#
# Override the wait with:  DELAY_SECONDS=1800 bash .../citrusfarm_pose_prior_delayed.sh
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/citrusfarm_pose_prior_fresh_l2.sh"
DELAY_SECONDS="${DELAY_SECONDS:-3600}"

LOG_DIR="$HOME/agromap4d_work/citrusfarm_05_13d_543_735_v1/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pose_prior_delayed_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }

# Keep the machine awake for the whole wait plus the run. The guard variable
# stops systemd-inhibit re-entering itself after the exec.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle \
        --why="citrus pose-prior delayed experiment" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== delayed pose-prior launcher started $(date -Is) ==="
echo "repo   : $REPO_ROOT"
echo "target : $TARGET"
echo "delay  : ${DELAY_SECONDS}s"

# 1. Fail fast: validate now, so a broken setup does not waste the wait.
echo
echo "--- [1/4] immediate preflight (fail fast) ---"
bash "$TARGET" preflight

start_epoch=$(date +%s)
launch_epoch=$((start_epoch + DELAY_SECONDS))
echo
echo "--- [2/4] waiting until $(date -d "@$launch_epoch" -Is) ---"
while :; do
    now=$(date +%s)
    remaining=$((launch_epoch - now))
    [ "$remaining" -le 0 ] && break
    # Report every 15 min so the log shows the wait is alive.
    printf '    %s  %d min remaining\n' "$(date +%H:%M:%S)" $(((remaining + 59) / 60))
    if [ "$remaining" -gt 900 ]; then sleep 900; else sleep "$remaining"; fi
done

# 2. Re-validate: resources may have changed during the wait.
echo
echo "--- [3/4] preflight again at launch time ---"
bash "$TARGET" preflight

echo
echo "--- [4/4] running the fresh pose-prior experiment ---"
bash "$TARGET" run

echo
echo "=== delayed launcher finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
