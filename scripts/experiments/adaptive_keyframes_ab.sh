#!/usr/bin/env bash
# Controlled adaptive-keyframe A/B/C. Features remain available for every
# image, and COLMAP image_registrator restores poses for every non-keyframe.

set -Eeuo pipefail

AB_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
AB_REPO_ROOT="$(cd "$(dirname "$AB_SCRIPT_PATH")/../.." && pwd)"
AB_KIND="adaptive-keyframes-ab"
AB_TITLE="Dense vs balanced vs sparse adaptive-keyframe A/B/C"
AB_ARMS=("dense" "balanced" "sparse")
AB_FEATURE_PROFILES=("gpu" "gpu" "gpu")
AB_KEYFRAME_PRESETS=("dense" "balanced" "sparse")
AB_REFERENCE_KEYFRAMES=("459" "347" "270")

# shellcheck source=scripts/experiments/_ab_common.sh
source "$AB_REPO_ROOT/scripts/experiments/_ab_common.sh"
ab_main "$@"
