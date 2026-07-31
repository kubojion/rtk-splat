#!/usr/bin/env bash
# Controlled all-frame SIFT profile A/B. Both arms solve and retain all 1,344
# stereo frames; only the COLMAP feature profile differs.

set -Eeuo pipefail

AB_SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
AB_REPO_ROOT="$(cd "$(dirname "$AB_SCRIPT_PATH")/../.." && pwd)"
AB_KIND="feature-profile-ab"
AB_TITLE="GPU vs CPU-reference all-frame feature A/B"
AB_ARMS=("gpu" "cpu")
AB_FEATURE_PROFILES=("gpu" "cpu_reference")
AB_KEYFRAME_PRESETS=("all" "all")
AB_REFERENCE_KEYFRAMES=("1,344" "1,344")

# shellcheck source=scripts/experiments/_ab_common.sh
source "$AB_REPO_ROOT/scripts/experiments/_ab_common.sh"
ab_main "$@"
