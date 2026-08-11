#!/usr/bin/env bash
# Daytime re-run of the Rosario RGB-D diagnostic colour-GS experiment after the
# depth-projection lattice fix (_fill_projection_lattice in rgbd_transfer).
#
# v1 measured the defect this rerun corrects: nearest-pixel forward scatter
# into the 1.4x-longer-focal colour camera left a 1-px lattice of unsupervised
# rows/columns (35.3% of each image supervised vs 65.0% in the IR source), GS
# training diverged, and the best checkpoint was the first evaluation.
#
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2_day.sh
#
# Runs immediately (no delay): preflight, transfer + cloud + train (~3-3.5 h),
# then prints the verdict against the v1 RGB and production IR baselines.
set -Eeuo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
HERE="$(dirname "$SCRIPT_PATH")"
REPO_ROOT="$(cd "$HERE/../.." && pwd -P)"
TARGET="$HERE/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_overnight.sh"
WORKDIR="${RTK_SPLAT_WORKDIR:-/home/jion_kubo/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2}"
PYTHON="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"

# Fixed reference results for the verdict, from completed runs on this machine:
#   IR production:  agromap4d_work/rosario_v2_sequence5_ppk_140_250_overnight_v1
#   RGB v1 (lattice defect): agromap4d_work/..._rgbd_diagnostic_v1
readonly IR_MASKED=20.799 IR_CC=21.151 IR_SSIM=0.696 IR_LPIPS_CC=0.308
readonly V1_MASKED=18.687 V1_CC=19.035 V1_SSIM=0.360 V1_LPIPS_CC=0.594

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rgbd_diagnostic_v2_day_$(date +%Y%m%d_%H%M).log"

[ -f "$TARGET" ] || { echo "FATAL: missing $TARGET" >&2; exit 1; }
[ -e "$WORKDIR" ] && { echo "FATAL: workdir already exists: $WORKDIR" >&2; exit 1; }

# Keep the machine awake for the whole run, lid included. The guard stops
# systemd-inhibit re-entering itself after the exec.
if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario RGB-D diagnostic v2 day run" bash "$SCRIPT_PATH" "$@"
fi

cd "$REPO_ROOT"
echo "logging to $LOG"
{
echo "=== RGB-D diagnostic v2 day run started $(date -Is) ==="
echo "target  : $TARGET"
echo "workdir : $WORKDIR"

echo
echo "--- [1/3] preflight ---"
bash "$TARGET" preflight --workdir "$WORKDIR"

echo
echo "--- [2/3] transfer + cloud + train ---"
bash "$TARGET" run --acknowledge-diagnostic-render-only --workdir "$WORKDIR"

echo
echo "--- [3/3] verdict against v1 RGB and production IR ---"
"$PYTHON" - "$WORKDIR" \
    "$IR_MASKED" "$IR_CC" "$IR_SSIM" "$IR_LPIPS_CC" \
    "$V1_MASKED" "$V1_CC" "$V1_SSIM" "$V1_LPIPS_CC" <<'PY'
import json
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
ir = dict(zip(("masked", "cc", "ssim", "lpips_cc"), map(float, sys.argv[2:6])))
v1 = dict(zip(("masked", "cc", "ssim", "lpips_cc"), map(float, sys.argv[6:10])))

run = workdir / "runs" / "rosario-seq5-ppk-140-250-rgb-gs-diagnostic-v1"
pose = workdir / "pose_artifacts" / "rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1"
final = json.loads((run / "metrics.json").read_text())[-1]
best = json.loads((run / "best_checkpoint.json").read_text())
transfer = json.loads((pose / "transfer_evaluation.json").read_text())
projected = transfer["projected_depth_quality"]

print(f"supervised coverage : {projected['projected_depth_coverage_fraction']:.3f} "
      f"(v1: 0.353, IR source: 0.650)")
fill = projected.get("resampling_fill", {})
print(f"lattice fill        : +{fill.get('filled_fraction_of_output', 0.0):.3f} "
      f"of every image, retention unchanged at "
      f"{projected['projected_depth_retained_fraction']:.3f}")
steps = best["completed_training_steps"]
if best["best_step"] >= steps - 2500:
    convergence = "CONVERGED - improved to the end"
else:
    convergence = "DIVERGED - training peaked early, as in v1"
print(f"best checkpoint     : step {best['best_step']:,} of {steps:,} ({convergence})")
print()
print(f"{'metric':<16}{'v2':>9}{'v1 RGB':>9}{'IR':>9}{'v2-v1':>9}")
for key, name in (("psnr_masked", "masked"), ("psnr_masked_cc", "cc"),
                  ("ssim", "ssim"), ("lpips_cc", "lpips_cc")):
    value = float(final[key])
    print(f"{key:<16}{value:>9.3f}{v1[name]:>9.3f}{ir[name]:>9.3f}"
          f"{value - v1[name]:>+9.3f}")
print()
print("note: masked metrics are computed inside each run's own supervision")
print("mask; v2's mask is ~2x larger than v1's, so equal numbers already mean")
print("a better reconstruction. SSIM/LPIPS are full-frame and comparable.")
PY

echo
echo "=== RGB-D diagnostic v2 day run finished $(date -Is) ==="
} 2>&1 | tee "$LOG"
