#!/usr/bin/env bash
# Rosario colour GS v5: early MCMC stop + long noise-free settling tail.
#
# Starts immediately, does not power off, and reports rendered DETAIL (the
# metric that actually tracks usable output) against every previous run.
#
#   bash scripts/runs/rosario_rgbd_shortrefine_v5.sh
#
# Expected ~1 h 50 for 20,000 iterations plus the final evaluation.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Any reproduction config works; the workdir and run name default from its
# filename so a new variant needs only RTK_SPLAT_CONFIG.
CONFIG="${RTK_SPLAT_CONFIG:-$REPO/configs/reproductions/rosario_rgbd_shortrefine_v5.yaml}"
[[ "$CONFIG" == /* ]] || CONFIG="$REPO/$CONFIG"
TAG="$(basename "$CONFIG" .yaml)"
SOURCE_WORKDIR="${RTK_SPLAT_SOURCE_WORKDIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2}"
WORKDIR="${RTK_SPLAT_WORKDIR:-$HOME/agromap4d_work/$TAG}"
SEGMENT="$SOURCE_WORKDIR/segment"
POSE_NAME="rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1"
RUN_NAME="${RTK_SPLAT_RUN_NAME:-${TAG}-diagnostic-render}"
TRAIN_ITERS="${RTK_SPLAT_TRAIN_ITERS:-20000}"
PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${TAG}_$(date +%Y%m%d_%H%M).log"

fail() { echo "FATAL: $*" >&2; exit 1; }

PREFLIGHT_ONLY=0
if [ "${1:-}" = "--preflight-only" ]; then PREFLIGHT_ONLY=1; shift; fi
[ $# -eq 0 ] || fail "unknown argument: $1"

# ---- preflight ---------------------------------------------------------------
[ -x "$PY" ] || fail "python not executable: $PY"
[ -f "$CONFIG" ] || fail "missing config: $CONFIG"
[ -d "$SEGMENT" ] || fail "missing transfer segment: $SEGMENT"
[ -d "$SOURCE_WORKDIR/pose_artifacts/$POSE_NAME" ] || fail "missing pose artifact"
[ -f "$SOURCE_WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz" ] || fail "missing cloud"
[ -e "$WORKDIR/runs/$RUN_NAME" ] && fail "run '$RUN_NAME' already exists in $WORKDIR"
active="$(pgrep -af '[r]tk_splat\.(workflows|backends).*(rgbd_transfer|cloud|train)' || true)"
[ -z "$active" ] || fail "another rtk-splat compute stage is running: $active"

"$PY" - <<'PY' || exit 1
import torch, gsplat  # noqa: F401
assert torch.cuda.is_available(), "CUDA unavailable -- reboot"
free, _ = torch.cuda.mem_get_info()
assert free > 6000 * 1024 * 1024, f"only {free>>20} MiB GPU free -- close GPU apps"
print(f"GPU: {torch.cuda.get_device_name(0)} ({free>>20} MiB free)")
PY
free_gb=$(df --output=avail -BG "$HOME" | tail -1 | tr -dc '0-9')
[ "$free_gb" -ge 15 ] || fail "only ${free_gb}G free (need 15G)"

TRAIN_ARGV=(train --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR"
    --pose-name "$POSE_NAME" --run-name "$RUN_NAME"
    --train-iters "$TRAIN_ITERS" --allow-failed-georeferencing-for-render)

# Dry-run the CLI's validation, and confirm MCMC really stops after the
# Gaussian cap is reached (it fills at step 4,000 in every run so far).
"$PY" - "${TRAIN_ARGV[@]}" <<'PY' || fail "CLI/schedule validation failed"
import sys
from rtk_splat.workflows.cli import build_parser, _validate_diagnostic_render_names
from rtk_splat.workflows.configio import load_config
args = build_parser().parse_args(sys.argv[1:])
cfg = load_config(args.config, config_root=args.config_root)
_validate_diagnostic_render_names(args, cfg)
iters = args.train_iters
stop = int(iters * cfg.train.refine_stop_frac)
assert stop >= 4500, (
    f"MCMC would stop at step {stop}, before densification fills the cap at ~4,000; "
    "raise refine_stop_frac or --train-iters")
print(f"schedule: {iters:,} iterations, MCMC noise stops at step {stop:,}, "
      f"then {iters - stop:,} noise-free settling steps "
      f"(the 8.1-detail screen run had only 1,200)")
print(f"pose_opt enabled={cfg.train.pose_opt.enabled} "
      f"gauge_projection={cfg.train.pose_opt.gauge_projection}")
PY

((PREFLIGHT_ONLY)) && { echo "Preflight passed. Nothing was started."; exit 0; }

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario colour shortrefine v5" bash "$0" "$@"
fi

cd "$REPO"
echo "logging to $LOG"
{
echo "=== colour shortrefine v5 started $(date -Is) ==="
echo "workdir : $WORKDIR"
echo "change  : refine_stop_frac 0.8 -> 0.25 at ${TRAIN_ITERS} iterations"

echo
echo "--- [1/3] staging inputs ---"
mkdir -p "$WORKDIR/pose_artifacts" "$WORKDIR/cloud_artifacts"
for a in "pose_artifacts/$POSE_NAME" "cloud_artifacts/$POSE_NAME"; do
    [ -e "$WORKDIR/$a" ] || { cp -a "$SOURCE_WORKDIR/$a" "$WORKDIR/$a"; echo "staged $a"; }
done
"$PY" - "$WORKDIR" "$POSE_NAME" <<'PY' || fail "staged artifacts failed verification"
import sys
from pathlib import Path
import numpy as np
from rtk_splat.backends.pose_evidence import verify_pose_georeferencing_artifact
w, name = Path(sys.argv[1]), sys.argv[2]
ev = verify_pose_georeferencing_artifact(w / "pose_artifacts" / name, expected_name=name)
with np.load(w / "cloud_artifacts" / name / "init_cloud.npz", allow_pickle=False) as c:
    for k in ("pose_quality_sha256", "pose_manifest_sha256", "pose_georeferencing_sha256"):
        assert str(c[k]) == str(ev[k]), f"cloud/pose mismatch on {k}"
    n = len(c["xyz"])
print(f"staged inputs verified: {n:,}-point cloud, {ev['artifact_class']}")
PY

echo
echo "--- [2/3] training ---"
time "$PY" -m rtk_splat.workflows.cli "${TRAIN_ARGV[@]}"

echo
echo "--- [3/3] detail verdict against every previous run ---"
"$PY" - "$WORKDIR/runs/$RUN_NAME" <<'PY'
import glob, json, re, sys
from pathlib import Path
import cv2, numpy as np

def detail_curve(renders):
    out = []
    for f in sorted(glob.glob(f"{renders}/eval_*.jpg"),
                    key=lambda p: int(re.search(r"(\d+)", p.split('/')[-1]).group(1))):
        im = cv2.imread(f)
        h, w = im.shape[:2]
        g = cv2.cvtColor(im[:, w // 2:], cv2.COLOR_BGR2GRAY).astype(np.float32)[int(h * .45):, :]
        out.append((int(re.search(r"(\d+)", f.split('/')[-1]).group(1)),
                    float(cv2.Laplacian(g, cv2.CV_32F).var())))
    return out

run = Path(sys.argv[1])
curve = detail_curve(run / "renders")
print("v5 detail by step: " + "  ".join(f"{s//1000}k:{v:.1f}" for s, v in curve))
final = curve[-1][1] if curve else float("nan")
print()
print(f"{'run':<34}{'final detail':>13}")
for name, value in (("v1/v2/v3 long runs", 2.1),
                    ("v4 free poses, long", 0.5),
                    ("v4 screen (6k steps)", 8.1),
                    ("IR run (works)", 45.4),
                    ("v5 THIS RUN", final)):
    print(f"{name:<34}{value:>13.1f}")
m = [e for e in json.loads((run / "metrics.json").read_text())
     if e.get("selected_for_export") is not True][-1]
best = json.loads((run / "best_checkpoint.json").read_text())
print()
print(f"psnr_masked {m['psnr_masked']:.3f}  ssim {m['ssim']:.3f}  lpips_cc {m['lpips_cc']:.3f}")
print(f"best step {best['best_step']:,} of {best['completed_training_steps']:,}")
ply = run / "splat.DIAGNOSTIC_ONLY.ply"
print(f"PLY: {ply} ({ply.stat().st_size/2**20:.0f} MiB)" if ply.is_file() else "PLY MISSING")
print()
if final > 8.1:
    print("BETTER than the 6k screen -- the settling tail is productive; consider more iterations.")
elif final > 3.0:
    print("Beats the long-run baseline but not the 6k screen -- shorter may simply be better here.")
else:
    print("No better than the long-run baseline -- the early MCMC stop is not the lever.")
PY
echo "=== done $(date -Is) ==="
} 2>&1 | tee "$LOG"
