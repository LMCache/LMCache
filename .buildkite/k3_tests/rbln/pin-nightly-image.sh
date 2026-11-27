#!/usr/bin/env bash
# Records the rbln-serve image the nightly smoke just passed on as the RBLN
# runtime pin, so PR builds (resolve-pinned-vllm.sh) move to it. The nightly
# pipeline runs this behind a wait, so it only runs after the smoke passes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# <image_ref>@<digest>, as resolve_nightly_image.py printed it.
image="$(buildkite-agent meta-data get rbln_candidate_image)"
echo "Pinning verified rbln-serve image: ${image}"

PIN_BACKEND=rbln \
VLLM_IMAGE_TAG="${image%@*}" \
VLLM_IMAGE_REF="${image}" \
VLLM_IMAGE_DIGEST="${image#*@}" \
    exec bash "${SCRIPT_DIR}/../../../.github/scripts/pin-tested-vllm.sh"
