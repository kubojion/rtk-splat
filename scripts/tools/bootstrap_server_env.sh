#!/usr/bin/env bash
# Install and verify RTK-Splat's server dependencies without modifying the
# system Python, system CUDA, base Conda environment, or another user's files.
set -Eeuo pipefail
IFS=$'\n\t'
umask 027
export PYTHONNOUSERSITE=1
unset PYTHONPATH

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd -P)"
SERVER_ROOT="${RTK_SPLAT_SERVER_ROOT:-/data/jkobo/rtk-splat}"
CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
ACTION="${1:-plan}"
if (($#)); then shift; fi

fail() {
    echo "FATAL: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage:
  bash scripts/tools/bootstrap_server_env.sh plan [--root PATH]
  bash scripts/tools/bootstrap_server_env.sh install [--root PATH]
  bash scripts/tools/bootstrap_server_env.sh verify [--root PATH]

Options:
  --root PATH   Private RTK-Splat server root

The script creates two prefix-based Conda environments below PATH/envs. It
never uses sudo, modifies base Conda, installs a driver/toolkit, or writes to
/usr. The project root must already be owned and writable by the current user.
EOF
}

while (($#)); do
    case "$1" in
        --root)
            (($# >= 2)) || fail "--root needs a path"
            SERVER_ROOT="$(readlink -m "$2")"
            shift 2
            ;;
        -h|--help)
            ACTION=help
            shift
            ;;
        *) fail "unknown argument: $1" ;;
    esac
done

case "$ACTION" in
    plan|install|verify|help) ;;
    *) fail "action must be plan, install, verify, or help" ;;
esac
if [[ "$ACTION" == help ]]; then usage; exit 0; fi

SERVER_ROOT="$(readlink -m "$SERVER_ROOT")"
[[ "$SERVER_ROOT" == /data/*/rtk-splat ]] \
    || fail "refusing broad/unexpected server root: $SERVER_ROOT"
PY_ENV="$SERVER_ROOT/envs/rtk-splat-server"
COLMAP_ENV="$SERVER_ROOT/envs/colmap-rtk"
CONDA_PACKAGES="$SERVER_ROOT/cache/conda-pkgs"
PIP_CACHE="$SERVER_ROOT/cache/pip"
PY_REQUIREMENTS="$REPO/configs/environments/rtk-splat-cu121.txt"
COLMAP_SPEC="$REPO/configs/environments/colmap-4.1.1-cuda.yml"

print_plan() {
    cat <<EOF
Isolated RTK-Splat server environment

  repository:        $REPO
  private root:      $SERVER_ROOT
  Python/GS prefix:  $PY_ENV
  COLMAP prefix:     $COLMAP_ENV
  Conda package cache: $CONDA_PACKAGES
  pip cache:         $PIP_CACHE
  Conda executable:  ${CONDA_EXE:-NOT FOUND}

No system package, CUDA driver/toolkit, base Conda package, shell startup file,
Docker configuration, or other user's directory is changed.
EOF
}

if [[ "$ACTION" == plan ]]; then print_plan; exit 0; fi

[[ "$EUID" -ne 0 ]] || fail "run this as the normal server user, never with sudo"
[[ -n "$CONDA_EXE" && -x "$CONDA_EXE" ]] || fail "conda is not available"
[[ -f "$PY_REQUIREMENTS" ]] || fail "missing requirements: $PY_REQUIREMENTS"
[[ -f "$COLMAP_SPEC" ]] || fail "missing COLMAP spec: $COLMAP_SPEC"
[[ -d "$SERVER_ROOT" ]] || fail "server root is missing: $SERVER_ROOT"
[[ ! -L "$SERVER_ROOT" && -O "$SERVER_ROOT" && -w "$SERVER_ROOT" ]] \
    || fail "server root must be a real directory owned/writable by $(id -un): $SERVER_ROOT"
[[ -z "$(git -C "$REPO" status --porcelain)" ]] \
    || fail "server bootstrap requires a clean committed Git checkout: $REPO"

export CONDA_PKGS_DIRS="$CONDA_PACKAGES"
export PIP_CACHE_DIR="$PIP_CACHE"
mkdir -p "$SERVER_ROOT/envs" "$CONDA_PACKAGES" "$PIP_CACHE"

install_environments() {
    if [[ ! -x "$PY_ENV/bin/python" ]]; then
        "$CONDA_EXE" create --yes --prefix "$PY_ENV" python=3.10 pip
    else
        echo "REUSE: Python environment exists: $PY_ENV"
    fi
    "$CONDA_EXE" run --no-capture-output --prefix "$PY_ENV" \
        python -m pip install --requirement "$PY_REQUIREMENTS"
    "$CONDA_EXE" run --no-capture-output --prefix "$PY_ENV" \
        python -m pip install --editable "$REPO" --no-deps

    if [[ ! -x "$COLMAP_ENV/bin/colmap" ]]; then
        "$CONDA_EXE" env create --yes --prefix "$COLMAP_ENV" \
            --file "$COLMAP_SPEC"
    else
        echo "REUSE: COLMAP environment exists: $COLMAP_ENV"
    fi

    # The production preflight is offline and verifies this exact cached file.
    "$CONDA_EXE" run --no-capture-output --prefix "$PY_ENV" python - <<'PY'
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
print("LPIPS VGG cache ready")
PY
}

verify_environments() {
    [[ -x "$PY_ENV/bin/python" ]] || fail "Python environment is absent: $PY_ENV"
    [[ -x "$COLMAP_ENV/bin/colmap" ]] || fail "COLMAP environment is absent: $COLMAP_ENV"
    "$PY_ENV/bin/python" - "$REPO" <<'PY'
import importlib.metadata
import pathlib
import sys

expected = {
    "torch": "2.4.1+cu121",
    "torchvision": "0.19.1+cu121",
    "gsplat": "1.5.3+pt24cu121",
    "torchmetrics": "1.9.0",
    "numpy": "2.2.6",
    "opencv-python-headless": "5.0.0.93",
    "scipy": "1.15.3",
    "pymap3d": "3.2.0",
    "PyYAML": "6.0.3",
}
actual = {name: importlib.metadata.version(name) for name in expected}
wrong = {name: (expected[name], actual[name]) for name in expected if actual[name] != expected[name]}
if wrong:
    raise SystemExit(f"package version mismatch: {wrong}")

import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA")
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
if not any(marker in name for marker in ("RTX 3090", "RTX 4090")) or total < 22 * 1024**3:
    raise SystemExit(f"unsupported GPU: {name}, {total / 1024**3:.1f} GiB")

repo = pathlib.Path(sys.argv[1]).resolve()
import rtk_splat
package_path = pathlib.Path(rtk_splat.__file__).resolve()
if repo not in package_path.parents:
    raise SystemExit(f"rtk_splat is not installed from this checkout: {package_path}")

# Exercise the installed gsplat wheel, not just torch.cuda.is_available().
from rtk_splat.backends.gsplat import render
device = "cuda"
params = {
    "means": torch.tensor([[0.0, 0.0, 3.0]], device=device),
    "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    "scales": torch.log(torch.tensor([[0.1, 0.1, 0.1]], device=device)),
    "opacities": torch.tensor([0.0], device=device),
    "sh0": torch.zeros((1, 1, 3), device=device),
    "shN": torch.zeros((1, 0, 3), device=device),
}
viewmat = torch.eye(4, device=device)
intrinsics = torch.tensor(
    [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]],
    device=device,
)
rendered, alpha, _ = render(params, viewmat, intrinsics, 16, 16, 0)
torch.cuda.synchronize()
if rendered.shape != (1, 16, 16, 4) or alpha.shape != (1, 16, 16, 1):
    raise SystemExit("gsplat CUDA rasterization smoke returned wrong shapes")
print(f"Python stack PASS: {sys.version.split()[0]}, {name}, torch CUDA {torch.version.cuda}")
print("gsplat CUDA rasterization PASS")
print(f"RTK-Splat checkout: {package_path}")
PY

    local help_text banner
    help_text="$($COLMAP_ENV/bin/colmap -h 2>&1)"
    banner="${help_text%%$'\n'*}"
    [[ "$banner" == COLMAP\ 4.1.1*"with CUDA"* ]] \
        || fail "expected COLMAP 4.1.1 with CUDA, got: $banner"
    echo "COLMAP stack PASS: $banner"

    local vgg="$HOME/.cache/torch/hub/checkpoints/vgg16-397923af.pth"
    [[ -f "$vgg" ]] || fail "LPIPS VGG cache is missing: $vgg"
    [[ "$(sha256sum "$vgg" | awk '{print $1}')" == \
        397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0 ]] \
        || fail "LPIPS VGG checksum is wrong"
    echo "LPIPS cache PASS"

    cat <<EOF

Environment verification passed. Activate/use it with:
  conda activate $PY_ENV
  export COLMAP_BIN=$COLMAP_ENV/bin/colmap
  export RTK_SPLAT_SERVER_ROOT=$SERVER_ROOT
EOF
}

if [[ "$ACTION" == install ]]; then install_environments; fi
verify_environments
