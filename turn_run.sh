#!/usr/bin/env bash
# Headland-turn experiment: row-5 end -> U-turn -> row-1 entry (t_rel 2750-3200 s).
# Tests whether directly measured dual-antenna RTK heading survives rotation --
# the regime where the AgriGS demo's VIO-chain poses collapsed.
# Pre-run check 2026-07-26: 100% RTK-fixed, heading sigma 0.41 deg, max 2.9 deg/s.
# Run:  bash turn_run.sh    (~6 h; needs >= 10 GB free disk)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniconda3/envs/rtk-splat/bin/python
export PYTHONNOUSERSITE=1
LOG=$HOME/agromap4d_work/turn_run_$(date +%Y%m%d_%H%M).log

free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 10 ] || { echo "FATAL: only ${free_gb}G free (need 10G)." \
    "Free space first, e.g.: rm -rf ~/agromap4d_work/runs/tile_e{1,2,3,4,5,6}*"; exit 1; }
$PY -c "import torch; assert torch.cuda.is_available(), \
    'CUDA unavailable -- reboot (known driver drop)'; \
    print('GPU:', torch.cuda.get_device_name(0))"

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="rtk_splat turn run" bash "$0" "$@"
fi
cd "$DIR"
echo "logging to $LOG"
{
echo "=== headland-turn run started $(date) ==="
$PY -m rtk_splat.cli select   --config configs/field_turn.yaml
$PY -m rtk_splat.cli extract  --config configs/field_turn.yaml
$PY -m rtk_splat.cli depth    --config configs/field_turn.yaml
$PY -m rtk_splat.cli cloud    --config configs/field_turn.yaml
$PY -m rtk_splat.cli train    --config configs/field_turn.yaml
$PY -m rtk_splat.cli diagnose --config configs/field_turn.yaml || true
# train-view score of the new run (fit ceiling, AgriGS-comparable convention)
$PY -m rtk_splat.cli evalonly --config configs/field_turn.yaml \
    --split train --max-frames 64 || true
$PY -c "import json,pathlib;h=json.load(open(pathlib.Path.home()/'agromap4d_work/field_turn/runs/tile_turn1/metrics.json'));f=h[-1];print({k:round(v,3) if isinstance(v,float) else v for k,v in f.items() if k!='per_frame'})"
echo "=== done $(date) ==="
} 2>&1 | tee "$LOG"
