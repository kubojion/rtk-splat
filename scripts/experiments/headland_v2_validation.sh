#!/usr/bin/env bash
# Headland golden-reproduction check for the rebuilt contract-v2 pipeline.
#
# Question: does the current pipeline (sealed frontend, GPU SIFT, RTK-prior
# Global Mapper, ~12k pair graph) still reach the accepted headland result that
# was produced by the previous stereo-BA path (~53k pairs, no in-solve priors)?
#
#   accepted reference : psnr_masked 24.4449   psnr_masked_cc 25.8480
#   acceptance band    : within 0.30 dB of psnr_masked_cc
#
# Poses come from the completed GPU arm of headland_feature_ab_03, which passed
# every gate (1.213 px reprojection, 100% registration, held-out RTK 10.5 cm)
# and sat 0.79 cm / 0.127 deg from the CPU arm -- below the 1.03 cm / 0.121 deg
# separation that previously produced identical GS quality.
#
# Only the pose source differs from the accepted run. Gaussian budget,
# iterations, SH degree, holdout split and every learning rate are unchanged.
#
# Run:  bash scripts/experiments/headland_v2_validation.sh
# Time: ~2 min cloud + ~4 h training. Nothing existing is modified.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="$REPO/configs/reproductions/headland_stereo_ba.yaml"
SEGMENT="${RTK_SPLAT_SEGMENT:-$HOME/agromap4d_work/field_turn_contract_v2_normalized/segment}"
ARM="${RTK_SPLAT_ARM:-$HOME/agromap4d_work/headland_feature_ab_03/arms/gpu}"
POSE_NAME="feature-profile-ab-gpu-pose"
RUN_NAME="${RTK_SPLAT_RUN_NAME:-headland_v2_gpu_65k}"
PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"

REFERENCE_MASKED=24.4449
REFERENCE_CC=25.8480
TOLERANCE_DB=0.30

export PYTHONNOUSERSITE=1
LOG_DIR="$HOME/agromap4d_work/headland_feature_ab_03/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/headland_v2_validation_$(date +%Y%m%d_%H%M).log"

fail() { echo "FATAL: $*" >&2; exit 1; }

# ---- preflight: refuse to start a 4 h job on a broken setup -----------------
[ -x "$PY" ]                     || fail "python not executable: $PY"
[ -f "$CONFIG" ]                 || fail "missing config: $CONFIG"
[ -d "$SEGMENT" ]                || fail "missing segment: $SEGMENT"
[ -d "$ARM/pose_artifacts/$POSE_NAME" ] \
    || fail "missing pose artifact: $ARM/pose_artifacts/$POSE_NAME"
[ -e "$ARM/runs/$RUN_NAME" ] \
    && fail "run '$RUN_NAME' already exists; choose another RTK_SPLAT_RUN_NAME"

"$PY" - "$ARM/pose_artifacts/$POSE_NAME/quality.json" <<'PY' || exit 1
import json, sys
q = json.load(open(sys.argv[1]))
if not q.get("rtk_alignment_passed"):
    raise SystemExit("pose artifact did not pass its RTK alignment gates")
if q.get("registration_fraction") != 1.0:
    raise SystemExit("pose artifact does not cover every frame")
print(f"pose gate ok: {q['n_frames']} frames, "
      f"{q['mean_reprojection_error_px']:.3f} px, "
      f"held-out RTK {q['holdout_rtk_residual_m']['median']*100:.1f} cm")
PY

"$PY" -c "import torch,gsplat; assert torch.cuda.is_available(), 'CUDA unavailable -- reboot'; \
print('GPU:', torch.cuda.get_device_name(0))" || exit 1

free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 10 ] || fail "only ${free_gb}G free (need 10G)"

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle --why="headland v2 validation" bash "$0" "$@"
fi

cd "$REPO"
echo "logging to $LOG"
{
echo "=== headland contract-v2 validation started $(date -Is) ==="
echo "segment : $SEGMENT"
echo "poses   : $ARM/pose_artifacts/$POSE_NAME"
echo "run     : $ARM/runs/$RUN_NAME"
echo "target  : masked_cc $REFERENCE_CC +/- $TOLERANCE_DB dB (accepted: masked $REFERENCE_MASKED)"

echo "--- [1/2] init cloud from stereo-BA poses ---"
time "$PY" -m rtk_splat.workflows.cli cloud \
    --config "$CONFIG" --segment "$SEGMENT" --workdir "$ARM" \
    --pose-name "$POSE_NAME"

echo "--- [2/2] 65k Gaussian training ---"
time "$PY" -m rtk_splat.workflows.cli train \
    --config "$CONFIG" --segment "$SEGMENT" --workdir "$ARM" \
    --pose-name "$POSE_NAME" --run-name "$RUN_NAME"

echo "--- verdict ---"
"$PY" - "$ARM/runs/$RUN_NAME/metrics.json" "$REFERENCE_MASKED" "$REFERENCE_CC" "$TOLERANCE_DB" <<'PY'
import json, sys
final = json.load(open(sys.argv[1]))[-1]
ref_masked, ref_cc, tol = (float(v) for v in sys.argv[2:5])
masked, cc = final["psnr_masked"], final["psnr_masked_cc"]
print(f"masked    {masked:7.3f}  (accepted {ref_masked:7.3f}, delta {masked-ref_masked:+.3f} dB)")
print(f"masked_cc {cc:7.3f}  (accepted {ref_cc:7.3f}, delta {cc-ref_cc:+.3f} dB)")
print(f"ssim {final['ssim']:.3f}  lpips_cc {final['lpips_cc']:.3f}  "
      f"gaussians {final['n_gaussians']:,}")
if abs(cc - ref_cc) <= tol:
    print(f"\nREPRODUCED: within {tol} dB. The rebuilt pipeline preserves the "
          "accepted headland result; CitrusFarm numbers are now interpretable.")
else:
    print(f"\nNOT REPRODUCED: outside {tol} dB. Only the pose source changed, so "
          "investigate the two known differences before trusting a new dataset:\n"
          "  - pair graph is ~12,068 pairs vs ~53,058 in the accepted run\n"
          "  - RTK priors are now inserted into the solve (accepted run: none)")
PY
echo "=== done $(date -Is) ==="
} 2>&1 | tee "$LOG"
