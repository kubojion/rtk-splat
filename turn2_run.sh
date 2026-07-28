#!/usr/bin/env bash
# Budget A/B on the headland-turn segment: tile_turn2 = tile_turn1 with
# cap 1.5M -> 2.5M and 45k -> 65k iterations, nothing else changed.
# Reuses turn1's extracted segment/depth/cloud -- training only, ~3.5-4 h.
# POWERS OFF the computer when everything finishes cleanly; on failure it
# stays on with the log for inspection.
# Run:  bash turn2_run.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$HOME/miniconda3/envs/rtk-splat/bin/python
export PYTHONNOUSERSITE=1
LOG=$HOME/agromap4d_work/turn2_run_$(date +%Y%m%d_%H%M).log
SEG=$HOME/agromap4d_work/field_turn/segment

[ -f "$SEG/init_cloud.npz" ] && [ -f "$SEG/viewmats.npy" ] \
    || { echo "FATAL: turn segment incomplete at $SEG -- run turn_run.sh stages first"; exit 1; }
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 5 ] || { echo "FATAL: only ${free_gb}G free (need 5G)"; exit 1; }
$PY -c "import torch; assert torch.cuda.is_available(), \
    'CUDA unavailable -- reboot (known driver drop)'; \
    print('GPU:', torch.cuda.get_device_name(0))"

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="rtk_splat turn2" bash "$0" "$@"
fi
cd "$DIR"
echo "logging to $LOG"
{
echo "=== turn2 budget run started $(date) ==="
$PY -m rtk_splat.cli train    --config configs/field_turn2.yaml
$PY -m rtk_splat.cli diagnose --config configs/field_turn2.yaml || true
$PY -m rtk_splat.cli evalonly --config configs/field_turn2.yaml \
    --split train --max-frames 64 || true
$PY -c "import json,pathlib;h=json.load(open(pathlib.Path.home()/'agromap4d_work/field_turn/runs/tile_turn2/metrics.json'));f=h[-1];print({k:round(v,3) if isinstance(v,float) else v for k,v in f.items() if k!='per_frame'})"
echo "=== done $(date) -- powering off in 60 s (Ctrl-C to keep the machine on) ==="
} 2>&1 | tee "$LOG"
sync
sleep 60
systemctl poweroff || echo "poweroff not permitted -- shut down manually"
