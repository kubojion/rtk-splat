#!/usr/bin/env bash
# CitrusFarm 05_13D literal same-corridor row retrace [330,410].

set -Eeuo pipefail

readonly CITRUS_PUBLIC_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly CITRUS_REPO_ROOT="$(cd "$(dirname "$CITRUS_PUBLIC_SCRIPT_PATH")/../.." && pwd -P)"
readonly CITRUS_DEFAULT_CONFIG="$CITRUS_REPO_ROOT/configs/sequences/citrusfarm_05_13d_row_retrace.yaml"
readonly CITRUS_DEFAULT_WORKDIR="/home/jion_kubo/agromap4d_work/citrusfarm_05_13d_330_410_row_retrace_v1"
readonly CITRUS_PREFIX="citrus-05-13d-330-410-row-retrace-v1"
readonly CITRUS_FRONTEND_NAME="${CITRUS_PREFIX}-frontend-all-gpu"
readonly CITRUS_BACKEND_NAME="${CITRUS_PREFIX}-global"
readonly CITRUS_PRODUCTION_POSE_NAME="${CITRUS_PREFIX}-global"
readonly CITRUS_PRODUCTION_TRAIN_NAME="${CITRUS_PREFIX}-gs"
readonly CITRUS_SOURCE_SEGMENT_NAME="${CITRUS_PREFIX}-rgb"
readonly CITRUS_DERIVED_SEGMENT_NAME="${CITRUS_PREFIX}-sgbm"
readonly CITRUS_EXPECTED_WINDOW_START="330.0"
readonly CITRUS_EXPECTED_WINDOW_STOP="410.0"
readonly CITRUS_MIN_OUTPUT_FREE_GIB="30"
readonly CITRUS_PLAN_TITLE="CitrusFarm 05_13D same-corridor row-retrace end-to-end plan"
readonly CITRUS_PLAN_WINDOW="[330, 410] s"
readonly CITRUS_PLAN_DURATION_PATH="80 s / approximately 96.7 m driven, 48 m unique corridor"
readonly CITRUS_PLAN_EXPECTED_SCALE="roughly 800 stereo pairs over about 38 m of direct retrace overlap"
readonly CITRUS_PLAN_TRAINING="likely about 35k iterations"
readonly CITRUS_PLAN_POSE_ESTIMATE="25-55 min"
readonly CITRUS_PLAN_CLOUD_ESTIMATE=" 2-10 min"
readonly CITRUS_PLAN_GS_ESTIMATE="1-2 h"
readonly CITRUS_PLAN_TOTAL_ESTIMATE="approximately 2-4 h"

# shellcheck source=_citrusfarm_05_13d_common.sh
source "$CITRUS_REPO_ROOT/scripts/runs/_citrusfarm_05_13d_common.sh"
