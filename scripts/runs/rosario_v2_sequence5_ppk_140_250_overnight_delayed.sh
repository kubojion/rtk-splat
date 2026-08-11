#!/usr/bin/env bash
# Delayed launcher for the Rosario v2 Sequence 5 [140,250] s overnight baseline.
#
# Validates the machine and the sealed segment immediately, so a broken setup
# fails in seconds instead of in an hour. Then waits DELAY_SECONDS, re-validates
# (state can drift while you keep using the laptop), and starts the real run.
#
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_overnight_delayed.sh
#
# Override the wait:  DELAY_SECONDS=1800 bash scripts/runs/..._delayed.sh
# Override the target workdir:  RTK_SPLAT_WORKDIR=/path bash ..._delayed.sh
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/rosario_v2_sequence5_ppk_140_250_overnight.sh"
WORKDIR="${RTK_SPLAT_WORKDIR:-/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_overnight_v1}"
DELAY_SECONDS="${DELAY_SECONDS:-3600}"

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rosario_seq5_delayed_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }
[ -e "$WORKDIR" ] && { echo "FATAL: workdir already exists: $WORKDIR" >&2; exit 1; }

# Hold sleep off for the wait AND the run, and survive a closed lid. The guard
# stops systemd-inhibit re-entering itself after the exec.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario seq5 delayed overnight run" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== delayed Rosario seq5 launcher started $(date -Is) ==="
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
echo "--- [3/4] preflight again at launch time ---"
bash "$TARGET" preflight --workdir "$WORKDIR"

echo
echo "--- [4/4] running the Rosario seq5 overnight baseline ---"
bash "$TARGET" run --workdir "$WORKDIR"

echo
echo "=== delayed launcher finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
