#!/usr/bin/env bash
# RTK-Splat clean-baseline rerun (post evaluation-repair, 2026-07-25):
# leakage-free split manifests, WGS84 ENU + CRS metadata, GNSS quality
# recording. Produces the DEFENSIBLE baseline numbers replacing all previous
# metrics. Field + hangar, 30k each, ~5-6 h total.
#
# Run:  bash ~/agrorob_ws/src/AgroMap-4D/rtk_splat/overnight2.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniconda3/envs/rtk-splat/bin/python
export PYTHONNOUSERSITE=1
LOG=$HOME/agromap4d_work/overnight2_$(date +%Y%m%d_%H%M).log
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="rtk_splat overnight2" bash "$0" "$@"
fi
cd "$DIR"
echo "logging to $LOG"
{
echo "=== overnight2 (clean baseline) started $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader \
    || { echo "FATAL: GPU driver not available -- reboot first"; exit 1; }

echo "=== [1/7] FIELD re-extract (manifest + WGS84 ENU + gnss quality) ==="
$PY -m rtk_splat.cli extract --config config.yaml
echo "=== [2/7] FIELD cloud (TRAIN FRAMES ONLY -- leakage-free) ==="
$PY -m rtk_splat.cli cloud --config config.yaml
echo "=== [3/7] FIELD train 30k (tile_e7_clean) ==="
$PY -m rtk_splat.cli train --config config.yaml
echo "=== [4/7] HANGAR re-extract + cloud ==="
$PY -m rtk_splat.cli extract --config configs/hangar.yaml
$PY -m rtk_splat.cli cloud --config configs/hangar.yaml
echo "=== [5/7] HANGAR train 30k (tile_hangar4) ==="
$PY -m rtk_splat.cli train --config configs/hangar.yaml
echo "=== [6/7] diagnostics ==="
$PY -m rtk_splat.cli diagnose --config config.yaml || true
$PY -m rtk_splat.cli diagnose --config configs/hangar.yaml || true
echo "=== [7/7] summary (RAW metrics first -- the defensible numbers) ==="
for m in "$HOME/agromap4d_work/runs/tile_e7_clean/metrics.json" \
         "$HOME/agromap4d_work/hangar/runs/tile_hangar4/metrics.json"; do
    echo "--- $m ---"
    $PY -c "import json;h=json.load(open('$m'));f=h[-1];print({k:round(v,3) if isinstance(v,float) else v for k,v in f.items() if k!='per_frame'})"
done
echo "NOTE: these val numbers replace ALL previous metrics (old runs had init-cloud leakage)."
echo "=== overnight2 finished $(date) ==="
} 2>&1 | tee "$LOG"
