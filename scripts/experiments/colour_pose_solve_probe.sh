#!/usr/bin/env bash
# Does solving poses ON the colour images fix their inter-view disagreement?
#
# THE QUESTION. Colour poses are currently inherited from the IR stereo rig
# (COLMAP saw only left_/right_ infrared). Adjacent colour views disagree by
# 4.40 px, where headland (0.44) and Rosario IR (0.47) -- both of which had
# poses solved FROM the images being measured -- agree sub-pixel. Bundle
# adjustment absorbs systematic per-image distortion into each image's pose, so
# colour may simply never have received that treatment.
#
# The earlier content-swap test (IR content in colour geometry: 0.31 px) does
# NOT settle this: it projects IR into the colour grid with K and T and warps
# back with the same K and T, so any error in the colour camera model cancels.
# This probe removes that circularity by solving colour poses independently.
#
# METHOD. Take a contiguous window of colour frames, run monocular COLMAP on
# them alone, Sim(3)-align the result to the metric transferred poses (scale is
# arbitrary in mono SfM; only RELATIVE agreement is being measured, so this is
# sound), then run the identical adjacent-frame warp test with both pose sets.
#
#   colour-solved ~0.5 px  -> poses were the problem; get colour into the solve
#   colour-solved ~4 px    -> the disagreement is in the pixels (rolling
#                             shutter / specularity); no pose solve will help
#
#   bash scripts/experiments/colour_pose_solve_probe.sh
#   bash scripts/experiments/colour_pose_solve_probe.sh --preflight-only
#
# Strictly diagnostic: everything lands in a scratch directory, no artifact is
# published, and no pipeline file is modified. ~30-45 min.
set -Eeuo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

SEGMENT="${RTK_SPLAT_SEGMENT:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2/segment}"
POSE_DIR="${RTK_SPLAT_POSE_DIR:-$HOME/agromap4d_work/rosario_v2_sequence5_ppk_140_250_rgbd_diagnostic_v2/pose_artifacts/rosario-seq5-ppk-140-250-rgbd-transfer-diagnostic-v1}"
SCRATCH="${RTK_SPLAT_SCRATCH:-$HOME/agromap4d_work/.colour_pose_probe}"
PY="${RTK_SPLAT_PYTHON:-$HOME/miniconda3/envs/rtk-splat/bin/python}"
COLMAP="${RTK_SPLAT_COLMAP:-$HOME/miniconda3/envs/colmap-rtk/bin/colmap}"

FIRST="${PROBE_FIRST:-300}"          # first colour frame index
COUNT="${PROBE_COUNT:-250}"          # how many consecutive frames (~25 m)

LOG_DIR="$HOME/agromap4d_work/rosario_seq5_logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/colour_pose_probe_$(date +%Y%m%d_%H%M).log"

fail() { echo "FATAL: $*" >&2; exit 1; }

PREFLIGHT_ONLY=0
if [ "${1:-}" = "--preflight-only" ]; then PREFLIGHT_ONLY=1; shift; fi
[ $# -eq 0 ] || fail "unknown argument: $1"

# ---- preflight ---------------------------------------------------------------
[ -x "$PY" ] || fail "python not executable: $PY"
[ -x "$COLMAP" ] || fail "colmap not executable: $COLMAP"
[ -d "$SEGMENT/images" ] || fail "missing colour images: $SEGMENT/images"
[ -f "$POSE_DIR/viewmats.npy" ] || fail "missing transferred poses: $POSE_DIR"
[ -e "$SCRATCH" ] && fail "scratch already exists, remove it: $SCRATCH"
"$COLMAP" -h 2>&1 | grep -q 'COLMAP 4\.1\.1' || fail "expected the tested COLMAP 4.1.1 build"
active="$(pgrep -x colmap || true)"; [ -z "$active" ] || fail "another COLMAP is running"

"$PY" - "$SEGMENT" "$FIRST" "$COUNT" <<'PY' || fail "input check failed"
import json, sys
from pathlib import Path
import numpy as np
seg, first, count = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n = len(np.load(seg / "frames.npz")["frame_id"])
if first + count > n:
    raise SystemExit(f"window {first}..{first+count} exceeds {n} frames")
cam = json.load(open(seg / "calibration.json"))["cameras"]["left"]
K = np.asarray(cam["K"])
missing = [i for i in range(first, first + count)
           if not (seg / "images" / f"rgb_{i:06d}.png").is_file()]
if missing:
    raise SystemExit(f"{len(missing)} colour images missing, first {missing[0]}")
print(f"window {first}..{first+count-1} of {n} frames, {cam['width']}x{cam['height']}, "
      f"fx={K[0,0]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
PY

((PREFLIGHT_ONLY)) && { echo "Preflight passed. Nothing was started."; exit 0; }

echo "logging to $LOG"
{
echo "=== colour pose-solve probe started $(date -Is) ==="
echo "segment : $SEGMENT"
echo "window  : frames $FIRST..$((FIRST+COUNT-1))"
echo "scratch : $SCRATCH"

IMG="$SCRATCH/images"
mkdir -p "$IMG"
for i in $(seq "$FIRST" $((FIRST + COUNT - 1))); do
    n=$(printf '%06d' "$i")
    ln -s "$SEGMENT/images/rgb_$n.png" "$IMG/rgb_$n.png"
done
echo "linked $(ls "$IMG" | wc -l) colour images"

# Fixed intrinsics from the sealed calibration: this probe tests POSES, so the
# camera model must not be free to absorb the effect under test.
read -r FX FY CX CY W H < <("$PY" -c "
import json
c=json.load(open('$SEGMENT/calibration.json'))['cameras']['left']
K=c['K']; print(K[0][0], K[1][1], K[0][2], K[1][2], c['width'], c['height'])")
echo "fixed PINHOLE intrinsics: fx=$FX fy=$FY cx=$CX cy=$CY (${W}x${H})"

echo
echo "--- [1/4] features (GPU SIFT, single camera) ---"
time "$COLMAP" feature_extractor --database_path "$SCRATCH/db.db" --image_path "$IMG" \
    --ImageReader.single_camera 1 --ImageReader.camera_model PINHOLE \
    --ImageReader.camera_params "$FX,$FY,$CX,$CY" \
    --FeatureExtraction.use_gpu 1 >/dev/null

echo "--- [2/4] exhaustive matching ---"
time "$COLMAP" exhaustive_matcher --database_path "$SCRATCH/db.db" \
    --FeatureMatching.use_gpu 1 >/dev/null

echo "--- [3/4] monocular mapper (intrinsics held fixed) ---"
mkdir -p "$SCRATCH/sparse"
time "$COLMAP" mapper --database_path "$SCRATCH/db.db" --image_path "$IMG" \
    --output_path "$SCRATCH/sparse" \
    --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params 0 >/dev/null
[ -d "$SCRATCH/sparse/0" ] || fail "mapper produced no reconstruction (mono SfM failed outright -- itself an answer)"
"$COLMAP" model_converter --input_path "$SCRATCH/sparse/0" \
    --output_path "$SCRATCH/sparse/0" --output_type TXT >/dev/null

echo
echo "--- [4/4] adjacent-view agreement: colour-solved vs transferred poses ---"
"$PY" - "$SEGMENT" "$POSE_DIR" "$SCRATCH/sparse/0" "$FIRST" "$COUNT" <<'PY'
import json, sys
from pathlib import Path
import cv2, numpy as np

seg, pose_dir, sparse, first, count = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
K = np.asarray(json.load(open(seg / "calibration.json"))["cameras"]["left"]["K"])
H, W = 720, 1280
transferred = np.load(pose_dir / "viewmats.npy")

# ---- read the COLMAP reconstruction -----------------------------------------
solved = {}
for line in (sparse / "images.txt").read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    parts = line.split()
    # a pose line is exactly: ID QW QX QY QZ TX TY TZ CAM_ID NAME.png
    if len(parts) != 10 or not parts[0].isdigit() or not parts[9].endswith(".png"):
        continue                                   # skips the POINTS2D lines
    qw, qx, qy, qz = (float(v) for v in parts[1:5])
    tx, ty, tz = (float(v) for v in parts[5:8])
    idx = int(Path(parts[9]).stem.split("_")[1])
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]])
    vm = np.eye(4); vm[:3, :3] = R; vm[:3, 3] = (tx, ty, tz)
    solved[idx] = vm
reg = len(solved)
print(f"COLMAP registered {reg}/{count} colour images ({reg/count:.1%})")
if reg < 0.8 * count:
    print("WARNING: poor registration -- monocular SfM on forward motion is")
    print("near-degenerate, so treat the comparison below as weak evidence.")
if reg < 20:
    raise SystemExit("too few registered images to compare")

# ---- Sim(3): COLMAP's arbitrary scale -> the metric frame --------------------
ids = sorted(solved)
c_col = np.array([(-solved[i][:3, :3].T @ solved[i][:3, 3]) for i in ids])
c_met = np.array([(-transferred[i][:3, :3].T @ transferred[i][:3, 3]) for i in ids])
mu_c, mu_m = c_col.mean(0), c_met.mean(0)
X, Y = c_col - mu_c, c_met - mu_m
U, S, Vt = np.linalg.svd(X.T @ Y)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R = Vt.T @ D @ U.T
scale = (S * np.diag(D)).sum() / (X ** 2).sum()
rms = np.sqrt(((c_met - (scale * (R @ c_col.T).T + (mu_m - scale * (R @ mu_c)))) ** 2).sum(1).mean())
print(f"Sim(3) to metric: scale {scale:.4f}, camera-centre RMS {rms*100:.1f} cm")
aligned = {}
for i in ids:
    c2w = np.linalg.inv(solved[i]); c2w[:3, 3] = scale * (R @ c2w[:3, 3]) + (mu_m - scale * (R @ mu_c))
    c2w[:3, :3] = R @ c2w[:3, :3]
    aligned[i] = np.linalg.inv(c2w)

# ---- the identical adjacent-frame warp test ---------------------------------
han = cv2.createHanningWindow((96, 96), cv2.CV_32F)
def grad(im):
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, 3); gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, 3)
    s = np.sqrt(gx * gx + gy * gy); return (s - s.mean()) / (s.std() + 1e-6)

def disagreement(posefn, label):
    res = []
    for i in ids:
        if i + 1 not in aligned:
            continue
        with np.load(seg / "depth" / f"{i:06d}.npz") as d:
            z, v = d["depth"].astype(np.float32), np.asarray(d["valid"], bool)
        a = cv2.imread(str(seg / "images" / f"rgb_{i:06d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        b = cv2.imread(str(seg / "images" / f"rgb_{i+1:06d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        M = posefn(i + 1) @ np.linalg.inv(posefn(i))
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        P = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z,
                      z.astype(np.float64), np.ones_like(uu)], 0).reshape(4, -1)
        Q = M @ P
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = (K[0, 0] * Q[0] / Q[2] + K[0, 2]).reshape(H, W).astype(np.float32)
            my = (K[1, 1] * Q[1] / Q[2] + K[1, 2]).reshape(H, W).astype(np.float32)
        mx[~v] = -1; my[~v] = -1
        ga, gw = grad(a), grad(cv2.remap(b, mx, my, cv2.INTER_LINEAR, borderValue=0))
        sh = []
        for y0 in range(int(H * .45), H - 96, 80):
            for x0 in range(60, W - 96, 120):
                if v[y0:y0+96, x0:x0+96].mean() < 0.9:
                    continue
                (dx, dy), r = cv2.phaseCorrelate(
                    (gw[y0:y0+96, x0:x0+96] * han).astype(np.float64),
                    (ga[y0:y0+96, x0:x0+96] * han).astype(np.float64))
                if r > 0.05 and abs(dx) < 20 and abs(dy) < 20:
                    sh.append(np.hypot(dx, dy))
        if len(sh) >= 4:
            res.append(np.median(sh))
    m = float(np.median(res)) if res else float("nan")
    print(f"  {label:<34} {m:5.2f} px   (over {len(res)} frame pairs)")
    return m

print()
print("adjacent-view disagreement, identical test, identical depth and images:")
t = disagreement(lambda i: transferred[i], "transferred from IR rig")
s = disagreement(lambda i: aligned[i], "solved ON the colour images")
print()
print("reference: headland 0.44 px | Rosario IR 0.47 px | colour (full seq) 4.40 px")
print()
if not np.isfinite(s):
    print("INCONCLUSIVE: could not measure the colour-solved poses.")
elif s < 0.5 * t and s < 1.5:
    print(f"POSES WERE THE PROBLEM: {t:.2f} -> {s:.2f} px when solved on colour.")
    print("Getting colour into the pose solve (3-sensor rig, or BiNAR-style joint")
    print("optimisation) should recover the missing sharpness.")
elif s < 0.8 * t:
    print(f"PARTIAL: {t:.2f} -> {s:.2f} px. Poses carry some of it, but a large")
    print("residual remains that no pose solve can explain.")
else:
    print(f"NOT THE POSES: {t:.2f} -> {s:.2f} px, essentially unchanged. The")
    print("disagreement is in the colour pixels themselves (rolling shutter or")
    print("specular, view-dependent appearance). No pose solve will fix it, and")
    print("the bi-modal route is the only one with a mechanism behind it.")
PY

echo
echo "scratch left at $SCRATCH for inspection -- rm -rf it when done"
echo "=== done $(date -Is) ==="
} 2>&1 | tee "$LOG"
