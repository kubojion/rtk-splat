#!/usr/bin/env bash
# RTK-Splat overnight run: field (30k) + hangar (30k) with the 2026-07-24
# improvements (MCMC annealing, gauge-fixed exposure, cc-metric, AA, cropped
# exports) + brightness diagnostics.
#
# Run from any terminal:   bash ~/agrorob_ws/src/AgroMap-4D/rtk_splat/overnight.sh
# Expected duration: ~5-7 h on the RTX 3080. Keep the laptop on AC power.
# The script inhibits system sleep for its own duration (systemd-inhibit).
# Progress: tail -f the log file it prints at start. Safe to re-run; every
# stage overwrites its own outputs deterministically.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniconda3/envs/rtk-splat/bin/python
export PYTHONNOUSERSITE=1
LOG=$HOME/agromap4d_work/overnight_$(date +%Y%m%d_%H%M).log

# re-exec under a sleep inhibitor so a lid-close/idle timer cannot kill the run
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle \
        --why="rtk_splat overnight training" bash "$0" "$@"
fi

cd "$DIR"
echo "logging to $LOG"
{
echo "=== overnight run started $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || { echo "FATAL: GPU driver not available -- reboot first"; exit 1; }

echo "=== [0/6] smoke test: 60-iteration mini-run through every new code path ==="
SMOKE=$(mktemp -d)/smoke.yaml
$PY - "$SMOKE" << 'PYEOF'
import sys
s = open('config.yaml').read()
s = s.replace('run_name: tile_e5_overnight', 'run_name: smoke_test')
s = s.replace('iterations: 30000', 'iterations: 60')
s = s.replace('eval_every: 1000', 'eval_every: 30')
s = s.replace('eval_pose_align_steps: 200', 'eval_pose_align_steps: 3')
open(sys.argv[1], 'w').write(s)
PYEOF
$PY -m rtk_splat.cli train --config "$SMOKE"
$PY -m rtk_splat.cli diagnose --config "$SMOKE"
rm -rf "$HOME/agromap4d_work/runs/smoke_test"
echo "smoke test PASSED -- committing to the full run"

echo "=== [1/6] FIELD re-extract (clears stale tilt-era poses) + cloud ==="
$PY -m rtk_splat.cli extract --config config.yaml
$PY -m rtk_splat.cli cloud   --config config.yaml

echo "=== [2/6] FIELD train 30k (tile_e5_overnight) ==="
$PY -m rtk_splat.cli train --config config.yaml

echo "=== [3/6] FIELD brightness diagnostic ==="
$PY -m rtk_splat.cli diagnose --config config.yaml

echo "=== [4/6] HANGAR train 30k (tile_hangar3) ==="
$PY -m rtk_splat.cli train --config configs/hangar.yaml

echo "=== [5/6] HANGAR brightness diagnostic ==="
$PY -m rtk_splat.cli diagnose --config configs/hangar.yaml

echo "=== [6/6] summary ==="
for m in "$HOME/agromap4d_work/runs/tile_e5_overnight/metrics.json" \
         "$HOME/agromap4d_work/hangar/runs/tile_hangar3/metrics.json"; do
    echo "--- $m (final entry) ---"
    $PY -c "import json;h=json.load(open('$m'));f=h[-1];print({k:round(v,2) if isinstance(v,float) else v for k,v in f.items() if k!='per_frame'})"
done
echo "PLYs: ~/agromap4d_work/runs/tile_e5_overnight/splat.ply"
echo "      ~/agromap4d_work/hangar/runs/tile_hangar3/splat.ply"
echo "=== overnight run finished $(date) ==="
} 2>&1 | tee "$LOG"
