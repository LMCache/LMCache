#!/usr/bin/env bash
# Probe: does dropping an imported IPC event under a queued stream wait fault
# on this runtime? Runs imported_event_lifetime.py --drop and --hold inside the
# ROCm image named by VLLM_ROCM_IMAGE.
#
# --hold is the fixed behaviour and must pass. --drop is the pre-fix behaviour;
# a crash there is the hazard reproducing on this runtime and is reported, not
# failed, because whether it faults is a property of the runtime, not of
# LMCache. The step fails if either run never reached a pending wait, since
# such a run proves nothing.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VLLM_ROCM_IMAGE="${VLLM_ROCM_IMAGE:-vllm/vllm-openai-rocm:nightly-rocm100}"
DOCKER=(docker)
docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
cd "${REPO_ROOT}"
# shellcheck source=.buildkite/scripts/amd-disk-guard.sh
source .buildkite/scripts/amd-disk-guard.sh
source .buildkite/scripts/pick-free-gpu-amd.sh 20000 1
echo "=== IPC event lifetime probe on ${VLLM_ROCM_IMAGE}, HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} ==="
"${DOCKER[@]}" pull "${VLLM_ROCM_IMAGE}" >/dev/null
sudo -n dmesg -C >/dev/null 2>&1 || true

run_mode() {
    local mode="$1"
    echo "--- mode: ${mode} ---"
    timeout 600 "${DOCKER[@]}" run --rm \
        --network host --ipc host --group-add video \
        --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
        --device /dev/kfd --device /dev/dri \
        --volume "${REPO_ROOT}:/workspace/LMCache" --workdir /workspace/LMCache \
        --env "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" \
        --entrypoint python3 "${VLLM_ROCM_IMAGE}" \
        .buildkite/k3_tests/amd/imported_event_lifetime.py "--${mode}" 2>&1 | tee "probe-${mode}.log"
    echo "mode=${mode} exit=${PIPESTATUS[0]}"
}

run_mode hold
run_mode drop

hold_result=$(grep -E '^PROBE_RESULT mode=hold' probe-hold.log || true)
drop_result=$(grep -E '^PROBE_RESULT mode=drop' probe-drop.log || true)

echo "--- host dmesg (amdgpu lines since probe start) ---"
sudo -n dmesg 2>/dev/null | grep -iE "amdgpu|page fault|VM_L2" | tail -20 || echo "(no amdgpu lines)"

echo "+++ summary"
status=0
if [[ "${hold_result}" == *"pending=True"*"result_ok=True"*"exporter_exit=0"* ]]; then
    echo "hold: PASS (${hold_result#PROBE_RESULT })"
else
    echo "hold: FAIL (${hold_result:-no result line, process crashed})"
    status=1
fi
if [[ -z "${drop_result}" ]]; then
    echo "drop: crashed before reporting; imported event freed under a pending wait faults on this runtime"
    drop_summary="drop mode crashed (hazard reproduced)"
elif [[ "${drop_result}" == *"pending=False"* ]]; then
    echo "drop: INVALID, wait was not pending at the drop point (${drop_result#PROBE_RESULT })"
    drop_summary="drop mode invalid (wait not pending)"
    status=1
else
    echo "drop: completed without fault on this runtime (${drop_result#PROBE_RESULT })"
    drop_summary="drop mode completed without fault"
fi
if command -v buildkite-agent >/dev/null 2>&1; then
    buildkite-agent annotate --style "$([[ ${status} -eq 0 ]] && echo success || echo error)" \
        --context "amd-ipc-event-probe" \
        "IPC event lifetime probe on \`${VLLM_ROCM_IMAGE}\`: hold mode $([[ ${status} -eq 0 ]] && echo passed || echo failed); ${drop_summary}." || true
fi
exit "${status}"
