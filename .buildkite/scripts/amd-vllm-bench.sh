#!/usr/bin/env bash
# Run the shared vLLM benchmark in a pinned official ROCm image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Pin to a released vLLM ROCm image so LMCache regressions can be separated
# from upstream image changes. Override via VLLM_ROCM_IMAGE for explicit bumps.
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:v0.26.0}"
MODE="${1:-}"
TEST_SCENARIO="${2:-vllm_bench}"
SERIALIZE_ENV=()
DOCKER=(docker)

case "${MODE}" in
    serialized)
        SERIALIZE_ENV=(--env "AMD_SERIALIZE_KERNEL=1")
        ;;
    unserialized)
        ;;
    *)
        echo "Usage: $0 <serialized|unserialized> [vllm_bench|lazy_offload_eviction_aware]" >&2
        exit 2
        ;;
esac

case "${TEST_SCENARIO}" in
    vllm_bench)
        GPU_COUNT=2
        VLLM_DISABLE_PREFIX_CACHING="${VLLM_DISABLE_PREFIX_CACHING:-true}"
        ;;
    lazy_offload_eviction_aware)
        GPU_COUNT=1
        VLLM_DISABLE_PREFIX_CACHING=false
        ;;
    *)
        echo "Unknown AMD test scenario: ${TEST_SCENARIO}" >&2
        exit 2
        ;;
esac

JOB_TAG="${BUILDKITE_JOB_ID:-local-$$}"
CONTAINER_NAME="lmcache-amd-${TEST_SCENARIO}-${MODE}-${JOB_TAG}"

cd "${REPO_ROOT}"
echo "AMD kernel mode: ${MODE}"
echo "AMD test scenario: ${TEST_SCENARIO}"

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

# The benchmark needs a second device for its baseline server; lazy offload is
# a single-server scenario. The selected physical IDs are passed through to the
# native-process launcher inside the container.
# shellcheck disable=SC1091
source .buildkite/scripts/pick-free-gpu-amd.sh 70000 "${GPU_COUNT}"
IFS=',' read -r GPU_FOR_VLLM GPU_FOR_BASELINE <<< "${HIP_VISIBLE_DEVICES}"
if [[ -z "${GPU_FOR_VLLM}" ]]; then
    echo "Expected at least one AMD GPU, got HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    exit 1
fi
if [[ "${GPU_COUNT}" -eq 2 && -z "${GPU_FOR_BASELINE}" ]]; then
    echo "Expected two AMD GPUs, got HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    exit 1
fi
export GPU_FOR_VLLM GPU_FOR_BASELINE

# These containers share the host network. Select per-job ports so concurrent
# serialized and unserialized jobs cannot attach to each other's services.
read -r LMCACHE_PORT LMCACHE_HTTP_PORT VLLM_PORT VLLM_BASELINE_PORT < <(
    python3 - <<'PY'
import socket

sockets = []
ports = []
for _ in range(4):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sockets.append(sock)
    ports.append(sock.getsockname()[1])
print(*ports)
PY
)
export LMCACHE_PORT LMCACHE_HTTP_PORT VLLM_PORT VLLM_BASELINE_PORT
echo "Ports: LMCache=${LMCACHE_PORT}, HTTP=${LMCACHE_HTTP_PORT}, vLLM=${VLLM_PORT}, baseline=${VLLM_BASELINE_PORT}"

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
    --env "AMD_KERNEL_MODE=${MODE}" \
    --env "TEST_SCENARIO=${TEST_SCENARIO}" \
    --env "LMCACHE_REQUEST_TRANSPORT=${LMCACHE_REQUEST_TRANSPORT:-zmq}" \
    --env "LMCACHE_PORT=${LMCACHE_PORT}" \
    --env "LMCACHE_HTTP_PORT=${LMCACHE_HTTP_PORT}" \
    --env "VLLM_PORT=${VLLM_PORT}" \
    --env "VLLM_BASELINE_PORT=${VLLM_BASELINE_PORT}" \
    "${SERIALIZE_ENV[@]}" \
    --env ATTENTION_BACKEND=auto \
    --env BATCH_INVARIANT=0 \
    --env "MAX_SLOWDOWN_PERCENT=${MAX_SLOWDOWN_PERCENT:-10}" \
    --env "VLLM_DISABLE_PREFIX_CACHING=${VLLM_DISABLE_PREFIX_CACHING}" \
    --env LMCACHE_TRACK_USAGE=false \
    --env "RESULTS_DIR=/workspace/LMCache/amd-vllm-bench-results/${TEST_SCENARIO}/${MODE}" \
    --entrypoint bash \
    "${VLLM_ROCM_IMAGE}" \
    .buildkite/scripts/amd-vllm-bench-container.sh
