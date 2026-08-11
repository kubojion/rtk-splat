#!/usr/bin/env bash
# Delayed launcher for the CitrusFarm 05_13D same-corridor row-retrace run.
#
# Starts immediately, validates the setup straight away so a broken environment
# fails in seconds rather than in an hour, waits DELAY_SECONDS, re-validates,
# then runs the real experiment with diagnostic rendering enabled.
#
#   bash scripts/runs/citrusfarm_05_13d_row_retrace_delayed.sh
#
# Override the wait:  DELAY_SECONDS=1800 bash scripts/runs/..._delayed.sh
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/citrusfarm_05_13d_row_retrace.sh"
WORKDIR="${RTK_SPLAT_WORKDIR:-/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_330_410_row_retrace_v1}"
DELAY_SECONDS="${DELAY_SECONDS:-3600}"

LOG_DIR="$HOME/agromap4d_work/citrus_row_retrace_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/row_retrace_delayed_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }

# Hold sleep off for the wait AND the run. The guard stops systemd-inhibit
# re-entering itself after the exec.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle \
        --why="citrus row-retrace delayed run" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== delayed row-retrace launcher started $(date -Is) ==="
echo "target  : $TARGET"
echo "workdir : $WORKDIR"
echo "delay   : ${DELAY_SECONDS}s"

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
    printf '    %s  %d min remaining\n' "$(date +%H:%M:%S)" $(((remaining + 59) / 60))
    if [ "$remaining" -gt 900 ]; then sleep 900; else sleep "$remaining"; fi
done

echo
echo "--- [3/4] preflight again at launch time ---"
bash "$TARGET" preflight

echo
echo "--- [4/4] running the row-retrace experiment ---"
bash "$TARGET" run --workdir "$WORKDIR" --render-on-georef-failure

echo
echo "=== delayed launcher finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
