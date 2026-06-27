# Example script to build the LMCache integrated with vLLM container image

# Update the following variables accordingly
CUDA_VERSION=13.0
DOCKERFILE_NAME='Dockerfile'
# VLLM_VERSION resolution order:
#   1. Pre-set VLLM_VERSION env var (overrides everything).
#   2. PINNED_VLLM_VERSION resolved from the
#      `buildkite_latest_tested_vllm` branch (the most recent vLLM nightly
#      verified by the canary build). Empty when offline / first run.
#   3. Fallback to "nightly" so behaviour matches the previous default.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck source=../.buildkite/k3_harness/resolve-pinned-vllm.sh
source "${SCRIPT_DIR}/../.buildkite/k3_harness/resolve-pinned-vllm.sh"
VLLM_VERSION="${VLLM_VERSION:-${PINNED_VLLM_VERSION:-nightly}}"
DOCKER_BUILD_PATH='../' # This path should point to the LMCache root for access to 'requirements' directory
UBUNTU_VERSION=24.04

# `image-build` target will use the latest LMCache and vLLM code
# Change to 'image-release' target for using release package versions of vLLM and LMCache
BUILD_TARGET=image-build 

IMAGE_TAG='lmcache/vllm-openai:build-latest' # Name of container image to build

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
    --build-arg VLLM_VERSION=$VLLM_VERSION \
    --target $BUILD_TARGET --file $DOCKERFILE_NAME \
    --tag $IMAGE_TAG  $DOCKER_BUILD_PATH
