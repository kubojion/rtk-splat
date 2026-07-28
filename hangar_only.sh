#!/usr/bin/env bash
# Hangar-only recovery for overnight2 (field already completed cleanly).
# Plug in the Buffalo SSD first, then:  bash hangar_only.sh   (~3 h)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniconda3/envs/rtk-splat/bin/python
export PYTHONNOUSERSITE=1
LOG=$HOME/agromap4d_work/hangar_only_$(date +%Y%m%d_%H%M).log
[ -d "/media/jion_kubo/Buffalo SSD/0703" ] \
    || { echo "FATAL: Buffalo SSD not mounted -- plug it in first"; exit 1; }
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="rtk_splat hangar" bash "$0" "$@"
fi
cd "$DIR"
echo "logging to $LOG"
{
echo "=== hangar clean baseline started $(date) ==="
$PY -m rtk_splat.cli extract --config configs/hangar.yaml
$PY -m rtk_splat.cli cloud   --config configs/hangar.yaml
$PY -m rtk_splat.cli train   --config configs/hangar.yaml
$PY -m rtk_splat.cli diagnose --config configs/hangar.yaml || true
$PY -c "import json;h=json.load(open('$HOME/agromap4d_work/hangar/runs/tile_hangar4/metrics.json'));f=h[-1];print({k:round(v,3) if isinstance(v,float) else v for k,v in f.items() if k!='per_frame'})"
echo "=== done $(date) ==="
} 2>&1 | tee "$LOG"
