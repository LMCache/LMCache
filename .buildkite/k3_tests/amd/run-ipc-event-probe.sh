#!/usr/bin/env bash
# Probe: does dropping an imported IPC event under a queued stream wait fault
# on this runtime? Runs imported_event_lifetime.py --drop and --hold inside the
# ROCm image named by VLLM_ROCM_IMAGE and reports both; never fails the build.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:nightly-rocm100}"
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
cd "${REPO_ROOT}"
source .buildkite/scripts/pick-free-gpu-amd.sh 20000 1
echo "=== IPC event lifetime probe on ${VLLM_ROCM_IMAGE}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} ==="
"${DOCKER[@]}" pull "${VLLM_ROCM_IMAGE}" >/dev/null
sudo -n dmesg -C >/dev/null 2>&1 || true
for mode in drop hold; do
    echo "--- mode: ${mode} ---"
    timeout 600 "${DOCKER[@]}" run --rm \
        --network host --ipc host --group-add video \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --device /dev/kfd --device /dev/dri \
        --volume "${REPO_ROOT}:/workspace/LMCache" --workdir /workspace/LMCache \
        --env "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" \
        --entrypoint python3 "${VLLM_ROCM_IMAGE}" \
        .buildkite/k3_tests/amd/imported_event_lifetime.py "--${mode}"
    echo "mode=${mode} exit=$?"
done
echo "--- host dmesg (amdgpu lines since probe start) ---"
sudo -n dmesg 2>/dev/null | grep -i "amdgpu\|page fault\|VM_L2" | tail -20 || echo "(dmesg not readable)"
exit 0
