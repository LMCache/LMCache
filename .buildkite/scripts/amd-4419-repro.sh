#!/usr/bin/env bash
# Reproduce LMCache issue #4419 on the AMD Buildkite lane.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:latest}"
CONTAINER_NAME="lmcache-amd-4419-repro-${BUILDKITE_BUILD_ID}"
RESULTS_DIR="${REPO_ROOT}/amd-4419-repro-results"
HOST_LOG="${RESULTS_DIR}/host.log"
DOCKER=(docker)

cd "${REPO_ROOT}"
mkdir -p "${RESULTS_DIR}"
exec > >(tee -a "${HOST_LOG}") 2>&1

annotate_failure() {
    local status=$?
    if [[ "${status}" -ne 0 ]] && command -v buildkite-agent >/dev/null 2>&1; then
        {
            echo "### AMD #4419 repro failed"
            echo ""
            echo '```text'
            tail -n 250 "${HOST_LOG}" || true
            echo '```'
        } | buildkite-agent annotate --style "error" --context "amd-4419-repro" || true
    fi
    exit "${status}"
}
trap annotate_failure EXIT

echo "=== AMD #4419 repro host setup ==="
echo "repo=${REPO_ROOT}"
echo "buildkite_build_id=${BUILDKITE_BUILD_ID}"
echo "image=${VLLM_ROCM_IMAGE}"
echo "results_dir=${RESULTS_DIR}"
git rev-parse --short HEAD
git status --short

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to run the official vLLM ROCm image"
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
echo "Docker command: ${DOCKER[*]}"
"${DOCKER[@]}" version || true
echo "=== Host disk usage before Docker cleanup ==="
df -h /var/lib/docker /var/lib/buildkite-agent "${REPO_ROOT}" 2>/dev/null || df -h
"${DOCKER[@]}" system df || true

echo "=== Pruning unused Docker data before pulling large ROCm image ==="
"${DOCKER[@]}" container prune -f || true
"${DOCKER[@]}" image prune -af || true
"${DOCKER[@]}" builder prune -af || true
"${DOCKER[@]}" system df || true
echo "=== Host disk usage after Docker cleanup ==="
df -h /var/lib/docker /var/lib/buildkite-agent "${REPO_ROOT}" 2>/dev/null || df -h

# The reproducer needs one large MI300 GPU. The selector returns the host
# physical GPU id, which is passed to the container and used by the native
# LMCache/vLLM launch commands.
# shellcheck disable=SC1091
source .buildkite/scripts/pick-free-gpu-amd.sh "${MIN_FREE_MEM_MB:-70000}" 1
GPU_FOR_VLLM="${HIP_VISIBLE_DEVICES}"
export GPU_FOR_VLLM

print_selected_amd_gpu_info() {
    local gpu_info
    gpu_info="$(
        rocm-smi --showproductname --showmeminfo vram --showuse 2>/dev/null | \
            awk -v selected="${HIP_VISIBLE_DEVICES}" '
                {
                    if (match($0, /GPU\[([0-9]+)\]/, m)) {
                        gpu_idx = m[1] + 0
                        if (gpu_idx == selected + 0) {
                            print
                        }
                    }
                }
            '
    )"

    echo "=== Selected AMD GPU info (host rocm-smi) ==="
    echo "GPU_FOR_VLLM=${GPU_FOR_VLLM}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    if [[ -n "${gpu_info}" ]]; then
        echo "${gpu_info}"
    else
        echo "rocm-smi did not return per-device product details for the selected GPU"
    fi
    echo ""
}

cleanup() {
    "${DOCKER[@]}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    # The image runs as root, so compiled extensions and artifacts in the
    # mounted checkout must be returned to the Buildkite agent user.
    sudo chown -R "$(id -u):$(id -g)" "${REPO_ROOT}" 2>/dev/null || true
}
trap cleanup EXIT

print_selected_amd_gpu_info

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
    --env "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx942}" \
    --env "SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION:-0.5.4rc5.dev16}" \
    --env "LMCACHE_PORT=${LMCACHE_PORT:-16555}" \
    --env "VLLM_PORT=${VLLM_PORT:-18080}" \
    --env RESULTS_DIR=/workspace/LMCache/amd-4419-repro-results \
    --entrypoint bash \
    "${VLLM_ROCM_IMAGE}" \
    .buildkite/scripts/amd-4419-repro-container.sh
