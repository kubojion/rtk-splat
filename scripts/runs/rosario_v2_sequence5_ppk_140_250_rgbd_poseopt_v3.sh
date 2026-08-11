#!/usr/bin/env bash
# v3: colour GS training with bounded per-view pose refinement.
#
# Rationale (measured, 2026-08-05): the RGB supervision is misregistered
# against the depth-anchored geometry by 6-13 px median per frame. A clock
# sweep localises most of it to a ~+40 ms bias (residual 5.59 -> 2.29 px),
# which at 0.94 m/s is 3.8 cm of travel -- inside pose_opt's 5 cm trust
# radius. The stored factory extrinsic was also A/B'd against the official
# Kalibr alternative and is the better of the two (5.33 vs 7.41 px), so the
# extrinsic is not the lever. v1 (35% supervision) and v2 (65% after the
# lattice fix) both diverged with near-identical curves, eliminating
# supervision area. pose_opt is the intervention this evidence points to.
#
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_poseopt_v3.sh
#   bash scripts/runs/rosario_v2_sequence5_ppk_140_250_rgbd_poseopt_v3.sh --preflight-only
#
# A fresh workdir is required: the resolved-config ledger binds one workdir to
# one authored config, and v3's config differs from v2's. The transfer segment
# is read in place from the v2 workdir; only the pose artifact (444K) and the
# initial cloud (11M) are copied, so nothing is recomputed.
#
# Expected: ~3 h training plus final evaluation.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

CONFIG="$REPO/configs/reproductions/rosario_rgbd_poseopt_v3.yaml"
SOURCE_WORKDIR="${RTK_SPLAT_SOURCE_WORKDIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2}"
WORKDIR="${RTK_SPLAT_WORKDIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_poseopt_v3}"
SEGMENT="$SOURCE_WORKDIR/segment"
POSE_NAME="rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1"
# A diagnostic run must not occupy the configured production run name; the CLI
# enforces this, so the artifact carries the repo's -diagnostic-render suffix.
RUN_NAME="${RTK_SPLAT_RUN_NAME:-rosario-seq5-ppk-140-250-rgb-gs-poseopt-v3-diagnostic-render}"
TRAIN_ITERS="${RTK_SPLAT_TRAIN_ITERS:-}"
PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"

V2_RUN="$SOURCE_WORKDIR/runs/rosario-seq5-ppk-140-250-rgb-gs-diagnostic-v1"
readonly IR_MASKED=20.799 IR_CC=21.151 IR_SSIM=0.696 IR_LPIPS_CC=0.308

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rgbd_poseopt_v3_$(date +%Y%m%d_%H%M).log"

fail() { echo "FATAL: $*" >&2; exit 1; }

# --preflight-only validates everything and exits without training.
PREFLIGHT_ONLY=0
if [ "${1:-}" = "--preflight-only" ]; then PREFLIGHT_ONLY=1; shift; fi
[ $# -eq 0 ] || fail "unknown argument: $1"

# ---- preflight ---------------------------------------------------------------
[ -x "$PY" ] || fail "python not executable: $PY"
[ -f "$CONFIG" ] || fail "missing config: $CONFIG"
[ -d "$SEGMENT" ] || fail "missing v2 transfer segment: $SEGMENT (run v2 first)"
[ -d "$SOURCE_WORKDIR/pose_artifacts/$POSE_NAME" ] || fail "missing v2 pose artifact"
[ -f "$SOURCE_WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz" ] \
    || fail "missing v2 initial cloud"
[ -f "$SOURCE_WORKDIR/run_state/train.done" ] \
    || fail "v2 training has not completed; let it finish first"
[ -e "$WORKDIR/runs/$RUN_NAME" ] && fail "run '$RUN_NAME' already exists in $WORKDIR"

active="$(pgrep -af '[r]tk_splat\.(workflows|backends).*(rgbd_transfer|cloud|train)' || true)"
[ -z "$active" ] || fail "another rtk-splat compute stage is running: $active"

"$PY" - <<'PY' || exit 1
import torch, gsplat  # noqa: F401
assert torch.cuda.is_available(), "CUDA unavailable -- reboot"
free, _total = torch.cuda.mem_get_info()
assert free > 6000 * 1024 * 1024, f"only {free>>20} MiB GPU free -- close GPU apps"
print("GPU:", torch.cuda.get_device_name(0), f"({free>>20} MiB free)")
PY

free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 15 ] || fail "only ${free_gb}G free (need 15G)"

# The exact argv the real run will use, defined once so the dry-run below
# cannot drift from it.
TRAIN_ARGV=(train --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR"
    --pose-name "$POSE_NAME" --run-name "$RUN_NAME"
    --allow-failed-georeferencing-for-render)
[ -n "$TRAIN_ITERS" ] && TRAIN_ARGV+=(--train-iters "$TRAIN_ITERS")

# Dry-run the CLI's own argument and config validation: catches naming and
# option errors in milliseconds rather than after a launcher's wait.
"$PY" - "${TRAIN_ARGV[@]}" <<'PY' || fail "CLI argument validation failed"
import sys
from rtk_splat.workflows.cli import build_parser, _validate_diagnostic_render_names
from rtk_splat.workflows.configio import load_config
args = build_parser().parse_args(sys.argv[1:])
cfg = load_config(args.config, config_root=args.config_root)
_validate_diagnostic_render_names(args, cfg)
print(f"CLI validation passed: run-name {args.run_name!r} "
      f"(config production name {cfg.train.run_name!r})")
PY

# The ledger binds a workdir to one authored config, so v3 needs its own.
# Stage the two small inputs it needs; the segment is read in place.
if [ -d "$WORKDIR" ] && [ -f "$WORKDIR/.config_ledger.json" ]; then
    ledger_config="$("$PY" -c "
import json,sys
print(json.load(open(sys.argv[1])).get('authored_config_path',''))" \
        "$WORKDIR/.config_ledger.json" 2>/dev/null || true)"
    [ -z "$ledger_config" ] || [ "$ledger_config" = "$CONFIG" ] \
        || fail "workdir $WORKDIR is bound to a different config: $ledger_config"
fi
mkdir -p "$WORKDIR/pose_artifacts" "$WORKDIR/cloud_artifacts"
for artifact in "pose_artifacts/$POSE_NAME" "cloud_artifacts/$POSE_NAME"; do
    if [ ! -e "$WORKDIR/$artifact" ]; then
        cp -a "$SOURCE_WORKDIR/$artifact" "$WORKDIR/$artifact"
        echo "staged $artifact"
    fi
done

# Prove the staged copies are intact and mutually consistent.
"$PY" - "$WORKDIR" "$POSE_NAME" <<'PY' || fail "staged artifacts failed verification"
import sys
from pathlib import Path
import numpy as np
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
workdir, name = Path(sys.argv[1]), sys.argv[2]
evidence = verify_pose_georeferencing_artifact(workdir / "pose_artifacts" / name,
                                               expected_name=name)
with np.load(workdir / "cloud_artifacts" / name / "init_cloud.npz",
             allow_pickle=False) as cloud:
    for key in ("pose_quality_sha256", "pose_manifest_sha256",
                "pose_georeferencing_sha256"):
        stored = str(cloud[key])
        if stored != str(evidence[key]):
            raise SystemExit(f"cloud/pose mismatch on {key}")
    points = len(cloud["xyz"])
print(f"staged inputs verified: {points:,}-point cloud bound to pose "
      f"{name} ({evidence['artifact_class']})")
PY

((PREFLIGHT_ONLY)) && {
    echo "Preflight passed (${free_gb}G disk free). No training was started."
    exit 0
}

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario RGB pose-opt v3 training" bash "$0" "$@"
fi

cd "$REPO"
echo "logging to $LOG"
{
echo "=== RGB pose-opt v3 training started $(date -Is) ==="
echo "segment : $SEGMENT (read in place from v2)"
echo "workdir : $WORKDIR (fresh; pose + cloud staged from v2)"
echo "run     : $WORKDIR/runs/$RUN_NAME"
echo "change  : train.pose_opt.enabled=true -- everything else identical to v2"

time "$PY" -m rtk_splat.workflows.cli "${TRAIN_ARGV[@]}"

echo
echo "--- verdict: v3 (pose_opt) vs v2 (no pose_opt) vs IR production ---"
"$PY" - "$WORKDIR/runs/$RUN_NAME" "$V2_RUN" \
    "$IR_MASKED" "$IR_CC" "$IR_SSIM" "$IR_LPIPS_CC" <<'PY'
import json
import sys
from pathlib import Path

v3_run, v2_run = Path(sys.argv[1]), Path(sys.argv[2])
ir = dict(zip(("masked", "cc", "ssim", "lpips_cc"), map(float, sys.argv[3:7])))
v3 = json.loads((v3_run / "metrics.json").read_text())[-1]
v2 = json.loads((v2_run / "metrics.json").read_text())[-1]
best = json.loads((v3_run / "best_checkpoint.json").read_text())
steps = best["completed_training_steps"]
if best["best_step"] >= steps - 2500:
    convergence = "CONVERGED - misregistration was the destabilizer"
else:
    convergence = ("STILL DIVERGED - pose_opt insufficient; next suspect is the "
                   "measured ~40 ms clock bias, which needs the transfer to "
                   "separate its association clock from its pose clock")
print(f"best checkpoint : step {best['best_step']:,} of {steps:,} ({convergence})")
print()
print(f"{'metric':<16}{'v3':>9}{'v2':>9}{'IR':>9}{'v3-v2':>9}")
for key, name in (("psnr_masked", "masked"), ("psnr_masked_cc", "cc"),
                  ("ssim", "ssim"), ("lpips_cc", "lpips_cc")):
    a, b = float(v3[key]), float(v2[key])
    print(f"{key:<16}{a:>9.3f}{b:>9.3f}{ir[name]:>9.3f}{a - b:>+9.3f}")
print()
print("v2 and v3 share the same segment and 65% supervision mask, so masked")
print("metrics are directly comparable between them. SSIM/LPIPS are full-frame.")
PY
echo "=== done $(date -Is) ==="
} 2>&1 | tee "$LOG"
