#!/usr/bin/env bash
# Rosario colour GS: scale/sharpness check, then a full run ONLY if it earns it.
#
# WHY. The colour model never develops detail. Render high-frequency energy is
# flat at 1-2 from step 2,000 to 44,300 while the IR run on the same trajectory
# climbs 6 -> 45 (ground truth 1522), and its Gaussians sit at 7.81 px projected
# radius (p10 3.21) vs IR's 2.18 px (p10 0.26). A splat cannot be sharper than
# its views agree; adjacent colour frames disagree by 3.74 px where IR frames
# disagree by 0.34. A per-view 6-dof rigid correction collapses that to 0.70 px
# -- exactly what pose_opt parameterises -- but as shipped it realises 0.05% of
# it, because project_gauge deletes the mean delta (98.6% of the correction) and
# trans_penalty caps the rest. v4 turns both off.
#
# WHAT THIS DOES.
#   1. waits DELAY_SECONDS (default 90 min)
#   2. SCREEN: 6,000 steps, then measures rendered detail (~45 min)
#   3. GATE: detail >= DETAIL_GATE (default 5, vs the control's 1-2)
#        pass -> FULL 44,300-step run, real PLY (~3 h 50)
#        fail -> stops. The mechanism is wrong; a full run would waste the night.
#   4. powers the machine off either way, after artifacts are flushed
#
#   bash scripts/runs/rosario_rgbd_freepose_v4_scalecheck.sh
#   NO_POWEROFF=1 bash ...   # run everything, leave the machine up
#
# This produces a DIAGNOSTIC_ONLY artifact by construction: without gauge
# projection the map may translate wholesale, so it cannot carry a metric
# georeferencing claim. That is intended.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

CONFIG="$REPO/configs/reproductions/rosario_rgbd_freepose_v4.yaml"
SOURCE_WORKDIR="${RTK_SPLAT_SOURCE_WORKDIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2}"
# The resolved-config ledger pins each derived value per workdir, so the
# screen (--train-iters 6000) and the full run (auto -> 44,300) cannot share
# one. They get their own, each staged with the same two small inputs.
WORKDIR="${RTK_SPLAT_WORKDIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_freepose_v4}"
FULL_WORKDIR="${RTK_SPLAT_FULL_WORKDIR:-${WORKDIR}_full}"
SKIP_SCREEN="${SKIP_SCREEN:-0}"
SEGMENT="$SOURCE_WORKDIR/segment"
POSE_NAME="rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1"
SCREEN_RUN="rosario-seq5-ppk-140-250-rgb-gs-freepose-v4-screen-diagnostic-render"
FULL_RUN="rosario-seq5-ppk-140-250-rgb-gs-freepose-v4-diagnostic-render"
PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"

DELAY_SECONDS="${DELAY_SECONDS:-5400}"
SCREEN_ITERS="${SCREEN_ITERS:-6000}"
DETAIL_GATE="${DETAIL_GATE:-5.0}"
NO_POWEROFF="${NO_POWEROFF:-0}"

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/freepose_v4_scalecheck_$(date +%Y%m%d_%H%M).log"

fail() { echo "FATAL: $*" >&2; exit 1; }

# ---- preflight (before the wait, so a broken setup fails in seconds) ---------
[ -x "$PY" ] || fail "python not executable: $PY"
[ -f "$CONFIG" ] || fail "missing config: $CONFIG"
[ -d "$SEGMENT" ] || fail "missing transfer segment: $SEGMENT"
[ -d "$SOURCE_WORKDIR/pose_artifacts/$POSE_NAME" ] || fail "missing pose artifact"
[ -f "$SOURCE_WORKDIR/cloud_artifacts/$POSE_NAME/init_cloud.npz" ] || fail "missing cloud"
((SKIP_SCREEN)) || [ ! -e "$WORKDIR/runs/$SCREEN_RUN" ] \
    || fail "screen run already exists in $WORKDIR (set SKIP_SCREEN=1 to go straight to the full run)"
[ -e "$FULL_WORKDIR/runs/$FULL_RUN" ] && fail "full run already exists in $FULL_WORKDIR"
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
[ "$free_gb" -ge 20 ] || fail "only ${free_gb}G free (need 20G for two runs)"

# Confirm the v4 knobs really resolve, and dry-run the CLI's own validation.
"$PY" - "$CONFIG" "$SCREEN_RUN" <<'PY' || fail "config validation failed"
import sys
from rtk_splat.workflows.configio import load_config
cfg = load_config(sys.argv[1])
po = cfg.train.pose_opt
assert po.enabled and not po.gauge_projection, "v4 requires pose_opt on, gauge off"
assert po.trans_penalty == 0.0 and po.rot_penalty == 0.0, "v4 requires penalties off"
assert sys.argv[2] != cfg.train.run_name, "diagnostic run name must differ"
print(f"v4 knobs: gauge_projection={po.gauge_projection} trans_penalty={po.trans_penalty} "
      f"start_iter={po.start_iter} max_trans_m={po.max_trans_m}")
PY

if [ -z "${RTK_SPLAT_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null; then
    export RTK_SPLAT_INHIBITED=1
    exec systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block \
        --why="Rosario colour scale check" bash "$0" "$@"
fi

power_off() {
    sync
    if ((NO_POWEROFF)); then
        echo "NO_POWEROFF set -- leaving the machine up."
        return 0
    fi
    echo "All work flushed. Powering off in 60 s -- Ctrl-C to cancel."
    sleep 60
    systemctl poweroff 2>/dev/null && return 0
    sudo -n systemctl poweroff 2>/dev/null && return 0
    sudo -n poweroff 2>/dev/null && return 0
    echo "WARNING: could not power off (no authorisation). The machine is idle" >&2
    echo "and all artifacts are written; shut down manually." >&2
}

cd "$REPO"
echo "logging to $LOG"
{
echo "=== colour scale check started $(date -Is) ==="
echo "workdir : $WORKDIR"
echo "delay   : ${DELAY_SECONDS}s   screen: ${SCREEN_ITERS} steps   gate: detail >= ${DETAIL_GATE}"

launch=$(( $(date +%s) + DELAY_SECONDS ))
echo
echo "--- [1/5] waiting until $(date -d "@$launch" -Is) ---"
while :; do
    remaining=$(( launch - $(date +%s) ))
    [ "$remaining" -le 0 ] && break
    printf '    %s  %d min remaining\n' "$(date +%H:%M:%S)" $(((remaining + 59) / 60))
    if [ "$remaining" -gt 900 ]; then sleep 900; else sleep "$remaining"; fi
done

stage_into() {
    local dest="$1"
    mkdir -p "$dest/pose_artifacts" "$dest/cloud_artifacts"
    for a in "pose_artifacts/$POSE_NAME" "cloud_artifacts/$POSE_NAME"; do
        [ -e "$dest/$a" ] || { cp -a "$SOURCE_WORKDIR/$a" "$dest/$a"; echo "staged $a -> $dest"; }
    done
}

echo
echo "--- [2/5] staging inputs into fresh workdirs ---"
stage_into "$WORKDIR"
stage_into "$FULL_WORKDIR"
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

if ((SKIP_SCREEN)); then
    echo
    echo "--- [3/5] SCREEN skipped (SKIP_SCREEN=1); gate already passed ---"
    gate_status=0
else
echo
echo "--- [3/5] SCREEN: ${SCREEN_ITERS} steps with free per-view poses ---"
time "$PY" -m rtk_splat.workflows.cli train \
    --config "$CONFIG" --segment "$SEGMENT" --workdir "$WORKDIR" \
    --pose-name "$POSE_NAME" --run-name "$SCREEN_RUN" \
    --train-iters "$SCREEN_ITERS" --allow-failed-georeferencing-for-render

echo
echo "--- [4/5] measuring rendered detail against the control ---"
set +e
"$PY" - "$WORKDIR/runs/$SCREEN_RUN/renders" "$DETAIL_GATE" <<'PY'
import glob, re, sys
import cv2, numpy as np
renders, gate = sys.argv[1], float(sys.argv[2])
files = sorted(glob.glob(f"{renders}/eval_*.jpg"),
               key=lambda p: int(re.search(r"(\d+)", p.split('/')[-1]).group(1)))
if not files:
    print("no eval renders produced"); raise SystemExit(2)
im = cv2.imread(files[-1]); h, w = im.shape[:2]; half = w // 2
def detail(x):
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)[int(h * 0.45):, :]
    return float(cv2.Laplacian(g, cv2.CV_32F).var())
gt, rd = detail(im[:, :half]), detail(im[:, half:])
print(f"ground truth detail {gt:.0f}   rendered detail {rd:.1f}   ratio {rd/max(gt,1e-9):.2%}")
print(f"reference: colour control 1-2 (flat to step 44,300); IR at this stage ~14-20")
if rd >= gate:
    print(f"PASS: {rd:.1f} >= {gate} -- free poses sharpened the model. Proceeding to the full run.")
    raise SystemExit(0)
print(f"FAIL: {rd:.1f} < {gate} -- unconstrained per-view poses did NOT restore detail.")
print("Registration is therefore not the binding cause; a full run would waste the night.")
raise SystemExit(1)
PY
gate_status=$?
set -e
fi

if [ "$gate_status" -eq 0 ]; then
    echo
    echo "--- [5/5] FULL run (44,300 steps) ---"
    time "$PY" -m rtk_splat.workflows.cli train \
        --config "$CONFIG" --segment "$SEGMENT" --workdir "$FULL_WORKDIR" \
        --pose-name "$POSE_NAME" --run-name "$FULL_RUN" \
        --allow-failed-georeferencing-for-render
    "$PY" - "$FULL_WORKDIR/runs/$FULL_RUN" <<'PY'
import glob, json, re, sys
import cv2, numpy as np
from pathlib import Path
run = Path(sys.argv[1])
m = [e for e in json.loads((run / "metrics.json").read_text())
     if e.get("selected_for_export") is not True][-1]
best = json.loads((run / "best_checkpoint.json").read_text())
f = sorted(glob.glob(f"{run}/renders/eval_*.jpg"),
           key=lambda p: int(re.search(r"(\d+)", p.split('/')[-1]).group(1)))[-1]
im = cv2.imread(f); h, w = im.shape[:2]; half = w // 2
d = lambda x: float(cv2.Laplacian(
    cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)[int(h*0.45):, :], cv2.CV_32F).var())
ply = run / "splat.DIAGNOSTIC_ONLY.ply"
print(f"detail: rendered {d(im[:, half:]):.1f} of ground truth {d(im[:, :half]):.0f}"
      f"   (control 1-2, IR final 45 of 165)")
print(f"psnr_masked {m['psnr_masked']:.3f}  ssim {m['ssim']:.3f}  lpips_cc {m['lpips_cc']:.3f}")
print(f"best step {best['best_step']:,} of {best['completed_training_steps']:,}")
print(f"PLY: {ply} ({ply.stat().st_size/2**20:.0f} MiB)" if ply.is_file() else "PLY MISSING")
PY
    echo "=== full run complete $(date -Is) ==="
else
    echo
    echo "--- [5/5] gate not met: stopping without the full run ---"
    echo "The screen artifacts remain at $WORKDIR/runs/$SCREEN_RUN for inspection."
fi

echo
echo "=== scale check finished $(date -Is) ==="
power_off
} 2>&1 | tee "$LOG"
