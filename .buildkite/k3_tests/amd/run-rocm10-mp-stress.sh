#!/usr/bin/env bash
# Host side of the ROCm 10 multi-writer stress: 8 GPUs, the ROCm 10 vLLM image,
# then the container script. Prints host amdgpu dmesg lines afterwards.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:nightly-rocm100}"
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
cd "${REPO_ROOT}"
# shellcheck source=.buildkite/scripts/amd-disk-guard.sh
source .buildkite/scripts/amd-disk-guard.sh
source .buildkite/scripts/pick-free-gpu-amd.sh 70000 8
echo "=== ROCm 10 MP stress on ${VLLM_ROCM_IMAGE}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} ==="
"${DOCKER[@]}" pull "${VLLM_ROCM_IMAGE}" >/dev/null
sudo -n dmesg -C >/dev/null 2>&1 || true
"${DOCKER[@]}" run --rm --name "lmcache-rocm10-stress-${BUILDKITE_BUILD_ID:-local}" \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --device /dev/kfd --device /dev/dri \
    --volume "${REPO_ROOT}:/workspace/LMCache" \
    --volume "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    --workdir /workspace/LMCache \
    --env "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" \
    --env "STRESS_MINUTES=${STRESS_MINUTES:-20}" \
    --env "MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}" \
    --env "CHUNK_SIZE=${CHUNK_SIZE:-256}" \
    --env LMCACHE_TRACK_USAGE=false \
    --entrypoint bash "${VLLM_ROCM_IMAGE}" \
    .buildkite/k3_tests/amd/rocm10-mp-stress-container.sh
status=$?
echo "--- host dmesg (amdgpu lines since start) ---"
sudo -n dmesg 2>/dev/null | grep -iE "amdgpu|page fault|VM_L2" | tail -20 || echo "(dmesg not readable)"
exit "$status"
