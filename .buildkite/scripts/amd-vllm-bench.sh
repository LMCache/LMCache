#!/usr/bin/env bash
# Run the shared vLLM benchmark in a pinned official ROCm image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Pin to a released vLLM ROCm image so LMCache regressions can be separated
# from upstream image changes. Override via VLLM_ROCM_IMAGE for explicit bumps.
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:v0.26.0}"
CONTAINER_NAME="lmcache-amd-vllm-bench-${BUILDKITE_BUILD_ID}"
DOCKER=(docker)

cd "${REPO_ROOT}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run the latest official vLLM ROCm image"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
        DOCKER=(sudo docker)
    else
        echo "docker is installed but this Buildkite agent cannot access /var/run/docker.sock"
        echo "Tried: docker info and sudo -n docker info"
        exit 1
    fi
fi

# vllm_bench compares LMCache-enabled vLLM against a baseline server, so it
# needs two devices. The host selector returns physical device IDs, which are
# passed through to the native-process launcher inside the container.
# shellcheck disable=SC1091
source .buildkite/scripts/pick-free-gpu-amd.sh 70000 2
IFS=',' read -r GPU_FOR_VLLM GPU_FOR_BASELINE <<< "${HIP_VISIBLE_DEVICES}"
if [[ -z "${GPU_FOR_VLLM}" || -z "${GPU_FOR_BASELINE}" ]]; then
    echo "Expected two AMD GPUs, got HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    exit 1
fi
export GPU_FOR_VLLM GPU_FOR_BASELINE

print_selected_amd_gpu_info() {
    local gpu_info
    gpu_info="$(
        rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null | \
            awk -v selected="${HIP_VISIBLE_DEVICES}" '
                BEGIN {
                    count = split(selected, ids, ",")
                    for (i = 1; i <= count; i++) {
                        wanted[ids[i] + 0] = 1
                    }
                }
                {
                    if (match($0, /GPU\[([0-9]+)\]/, m)) {
                        gpu_idx = m[1] + 0
                        if (gpu_idx in wanted) {
                            print
                        }
                    }
                }
            '
    )"

    echo "=== Selected AMD GPU info (host rocm-smi) ==="
    echo "GPU_FOR_VLLM=${GPU_FOR_VLLM}, GPU_FOR_BASELINE=${GPU_FOR_BASELINE}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    if [[ -n "${gpu_info}" ]]; then
        echo "${gpu_info}"
    else
        echo "rocm-smi did not return per-device product details for the selected GPUs"
    fi
    echo ""
}

print_selected_amd_gpu_info

cleanup() {
    "${DOCKER[@]}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    # The image runs as root so compiled extensions and artifacts in the
    # mounted checkout must be returned to the Buildkite agent user.
    sudo chown -R "$(id -u):$(id -g)" "${REPO_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Pulling ${VLLM_ROCM_IMAGE}"
"${DOCKER[@]}" pull "${VLLM_ROCM_IMAGE}"
"${DOCKER[@]}" image inspect "${VLLM_ROCM_IMAGE}" \
    --format 'vLLM ROCm image: {{index .RepoDigests 0}}'

"${DOCKER[@]}" run --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    --volume "${REPO_ROOT}:/workspace/LMCache" \
    --volume "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --workdir /workspace/LMCache \
    --env "BUILDKITE_BUILD_ID=${BUILDKITE_BUILD_ID}" \
    --env "GPU_FOR_VLLM=${GPU_FOR_VLLM}" \
    --env "GPU_FOR_BASELINE=${GPU_FOR_BASELINE}" \
    --env "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx942}" \
    --env ATTENTION_BACKEND=auto \
    --env BATCH_INVARIANT=0 \
    --env "MAX_SLOWDOWN_PERCENT=${MAX_SLOWDOWN_PERCENT:-10}" \
    --env "VLLM_DISABLE_PREFIX_CACHING=${VLLM_DISABLE_PREFIX_CACHING:-true}" \
    --env LMCACHE_TRACK_USAGE=false \
    --env RESULTS_DIR=/workspace/LMCache/amd-vllm-bench-results \
    --entrypoint bash \
    "${VLLM_ROCM_IMAGE}" \
    .buildkite/scripts/amd-vllm-bench-container.sh
