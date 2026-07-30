#!/usr/bin/env bash
# Calibrated-stereo COLMAP pose A/B on the existing headland-turn segment.
# This never rewrites the root RTK poses/cloud.  The long solve, refined
# cloud, and GS run live under pose_artifacts/colmap_stereo and tile_turn3.
#
# Readiness-only (never starts the long solve):
#   bash stereo_ba_overnight.sh --check
# Full run:
#   bash stereo_ba_overnight.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/jion_kubo/miniconda3/envs/rtk-splat/bin/python
COLMAP_DEFAULT=/home/jion_kubo/miniconda3/envs/colmap-rtk/bin/colmap
CFG="$DIR/configs/field_turn2.yaml"
SEG=/home/jion_kubo/agromap4d_work/field_turn/segment
ARTIFACT=colmap_stereo
RUN=tile_turn3_stereo_ba
LOG=/home/jion_kubo/agromap4d_work/stereo_ba_$(date +%Y%m%d_%H%M).log
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export COLMAP_BIN="${COLMAP_BIN:-$COLMAP_DEFAULT}"

MODE="${1:-run}"
if [ "$MODE" != "run" ] && [ "$MODE" != "--check" ]; then
    echo "Usage: bash stereo_ba_overnight.sh [--check]"
    exit 2
fi

[ -x "$PY" ] || { echo "FATAL: Python environment is missing: $PY"; exit 1; }
[ -f "$SEG/viewmats.npy" ] && [ -f "$SEG/cam_centers.npy" ] \
    || { echo "FATAL: extracted RTK segment is incomplete: $SEG"; exit 1; }
[ -d "$SEG/depth" ] \
    || { echo "FATAL: stereo depth is missing: $SEG/depth"; exit 1; }
command -v "$COLMAP_BIN" >/dev/null 2>&1 \
    || { echo "FATAL: COLMAP not found ($COLMAP_BIN). Install 4.1.1 or set COLMAP_BIN."; exit 1; }
colmap_banner=$("$COLMAP_BIN" -h 2>&1 | head -1)
case "$colmap_banner" in
    "COLMAP 4.1.1"*"(Commit "*" with CUDA)") ;;
    *) echo "FATAL: expected COLMAP 4.1.1 with CUDA, got: $colmap_banner"; exit 1 ;;
esac
free_gb=$(df --output=avail -BG /home/jion_kubo | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 20 ] \
    || { echo "FATAL: only ${free_gb}G free; keep at least 20G for COLMAP + GS."; exit 1; }

cd "$DIR"
if [ "$MODE" = "--check" ]; then
    "$PY" -m unittest discover -s tests -v
    "$PY" -m rtk_splat.cli stereo-prepare --config "$CFG" \
        --pose-artifact "$ARTIFACT"
    n_links=$(find "$SEG/pose_artifacts/$ARTIFACT/colmap/images" \
        -type l | wc -l)
    [ "$n_links" -eq 2688 ] \
        || { echo "FATAL: expected 2688 prepared image links, found $n_links"; exit 1; }
    echo "READY: $colmap_banner; ${free_gb}G free; 1344 stereo pairs."
    echo "No feature extraction, matching, mapping, cloud build, or training was run."
    exit 0
fi

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="RTK-Splat stereo BA" \
        bash "$0" "$@"
fi

echo "Logging to $LOG"
{
echo "=== stereo BA A/B started $(date --iso-8601=seconds) ==="
"$PY" -m rtk_splat.cli stereo-prepare --config "$CFG" \
    --pose-artifact "$ARTIFACT"
"$PY" -m rtk_splat.cli stereo-solve --config "$CFG" \
    --pose-artifact "$ARTIFACT"
"$PY" -m rtk_splat.cli stereo-export --config "$CFG" \
    --pose-artifact "$ARTIFACT"
"$PY" -m rtk_splat.cli cloud --config "$CFG" \
    --pose-artifact "$ARTIFACT"
"$PY" -m rtk_splat.cli train --config "$CFG" \
    --pose-artifact "$ARTIFACT" --run "$RUN"
"$PY" -m rtk_splat.cli diagnose --config "$CFG" \
    --pose-artifact "$ARTIFACT" --run "$RUN" || true
"$PY" -m rtk_splat.cli evalonly --config "$CFG" \
    --pose-artifact "$ARTIFACT" --run "$RUN" \
    --split train --max-frames 64 || true
echo "=== stereo BA A/B complete $(date --iso-8601=seconds) ==="
echo "The machine was deliberately left on."
} 2>&1 | tee "$LOG"
