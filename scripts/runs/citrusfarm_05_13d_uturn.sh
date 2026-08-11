#!/usr/bin/env bash
# CitrusFarm 05_13D [543,735] generic-policy transfer experiment.

set -Eeuo pipefail

readonly CITRUS_PUBLIC_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly CITRUS_REPO_ROOT="$(cd "$(dirname "$CITRUS_PUBLIC_SCRIPT_PATH")/../.." && pwd -P)"
readonly CITRUS_DEFAULT_CONFIG="$CITRUS_REPO_ROOT/configs/sequences/citrusfarm_05_13d_uturn.yaml"
readonly CITRUS_DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_543_735_auto_v1"
readonly CITRUS_FRONTEND_NAME="citrus-05-13d-543-735-auto-all-gpu-v1"
readonly CITRUS_BACKEND_NAME="citrus-05-13d-543-735-auto-global-v1"
readonly CITRUS_PRODUCTION_POSE_NAME="$CITRUS_BACKEND_NAME"
readonly CITRUS_PRODUCTION_TRAIN_NAME="citrus-05-13d-543-735-auto-gs-v1"
readonly CITRUS_SOURCE_SEGMENT_NAME="citrus-05-13d-543-735-auto-rgb-v1"
readonly CITRUS_DERIVED_SEGMENT_NAME="citrus-05-13d-543-735-auto-sgbm-v1"
readonly CITRUS_EXPECTED_WINDOW_START="543.0"
readonly CITRUS_EXPECTED_WINDOW_STOP="735.0"
readonly CITRUS_MIN_OUTPUT_FREE_GIB="65"
readonly CITRUS_PLAN_TITLE="CitrusFarm 05_13D generic-auto end-to-end plan"
readonly CITRUS_PLAN_WINDOW="[543, 735] s"
readonly CITRUS_PLAN_DURATION_PATH="192 s / approximately 229 m"
readonly CITRUS_PLAN_EXPECTED_SCALE="roughly 2,288 stereo pairs"
readonly CITRUS_PLAN_TRAINING="likely 65k iterations"
readonly CITRUS_PLAN_POSE_ESTIMATE="55-90 min"
readonly CITRUS_PLAN_CLOUD_ESTIMATE=" 5-20 min"
readonly CITRUS_PLAN_GS_ESTIMATE="3-5 h"
readonly CITRUS_PLAN_TOTAL_ESTIMATE="approximately 5-8 h; allow 8-10 h"

# shellcheck source=_citrusfarm_05_13d_common.sh
source "$CITRUS_REPO_ROOT/scripts/runs/_citrusfarm_05_13d_common.sh"
