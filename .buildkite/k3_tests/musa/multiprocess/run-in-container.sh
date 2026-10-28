#!/usr/bin/env bash
# Run the MUSA CI scripts in the maintainer-provisioned TorchMUSA image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODE="${1:-}"
VLLM_IMAGE="${MUSA_CI_IMAGE:-sh-harbor.mthreads.com/ai-kv/kuae-lmcache-vllm-ci:latest}"
SGLANG_IMAGE="sh-harbor.mthreads.com/ai-kv/kuae-lmcache-sglang-ci:latest"
IMAGE=""
ARTIFACT_PATH="${MUSA_CI_ARTIFACT_DIR:-musa-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi
INNER_SCRIPT=""
DOCKER_ENV_ARGS=(
    -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"
    -e "LMCACHE_TRACK_USAGE=${LMCACHE_TRACK_USAGE:-false}"
    -e "MUSA_CI_ARTIFACT_DIR=/musa-ci-artifacts"
    -e "MUSA_CI_PYTHON=${MUSA_CI_PYTHON:-python3}"
    -e "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}"
    -e "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}"
)
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
DOCKER_ENV_ARGS+=(
    -e "MUSA_CI_HOST_UID=${HOST_UID}"
    -e "MUSA_CI_HOST_GID=${HOST_GID}"
)

log() {
    echo "--- :musa: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[musa-ci] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/runner-failure.log" >&2
    exit 1
}

case "${MODE}" in
    unit)
        INNER_SCRIPT=".buildkite/k3_tests/musa/unittests/run.sh"
        IMAGE="${VLLM_IMAGE}"
        ;;
    smoke)
        INNER_SCRIPT=".buildkite/k3_tests/musa/multiprocess/run.sh"
        IMAGE="${VLLM_IMAGE}"
        ;;
    vllm-e2e)
        INNER_SCRIPT=".buildkite/k3_tests/musa/e2e/run-vllm.sh"
        IMAGE="${VLLM_IMAGE}"
        ;;
    sglang-e2e)
        INNER_SCRIPT=".buildkite/k3_tests/musa/e2e/run-sglang.sh"
        IMAGE="${SGLANG_IMAGE}"
        DOCKER_ENV_ARGS+=("-e" "LMCACHE_MUSA_HANDLE_TRANSFER=1")
        DOCKER_ENV_ARGS+=("-e" "MUSA_CI_ACTIVATE_SGLANG=1")
        ;;
    *)
        fail "usage: $0 {unit|smoke|vllm-e2e|sglang-e2e}"
        ;;
esac

mkdir -p "${ARTIFACT_DIR}"
command -v docker >/dev/null 2>&1 || fail "docker is required on the MUSA agent"

for optional_variable in \
    LMCACHE_DEVICE_BACKEND \
    TEST_SELECTOR \
    MUSA_CI_ZMQ_PORT \
    MUSA_CI_HTTP_PORT \
    MUSA_E2E_MODEL \
    MUSA_SGLANG_MODEL \
    MUSA_E2E_ENABLE_SGLANG \
    MUSA_E2E_PROMPT \
    MUSA_E2E_MAX_TOKENS \
    MUSA_E2E_TOP_K \
    MUSA_E2E_SEED \
    MUSA_E2E_TEMPERATURE \
    MUSA_E2E_HIT_PATTERN \
    MUSA_E2E_LOG_LEVEL \
    MUSA_E2E_STARTUP_TIMEOUT \
    MUSA_E2E_MAX_MODEL_LEN \
    MUSA_E2E_GPU_MEMORY_UTILIZATION \
    MUSA_E2E_CHUNK_SIZE \
    MUSA_E2E_LOCAL_CPU_SIZE_GB \
    MUSA_E2E_DAEMON_L1_SIZE_GB \
    MUSA_E2E_SERVING_MODEL \
    MUSA_E2E_VLLM_PORT \
    MUSA_E2E_VLLM_LAUNCHER \
    MUSA_SGLANG_PORT \
    MUSA_SGLANG_DEVICE \
    MUSA_SGLANG_LAUNCHER \
    MUSA_SGLANG_MAX_TOTAL_TOKENS \
    MUSA_SGLANG_MEM_FRACTION_STATIC \
    MUSA_SGLANG_LMCACHE_PORT \
    MUSA_SGLANG_LMCACHE_HTTP_PORT; do
    if [[ -n "${!optional_variable:-}" ]]; then
        DOCKER_ENV_ARGS+=(-e "${optional_variable}=${!optional_variable}")
    fi
done

if command -v mthreads-gmi >/dev/null 2>&1; then
    mthreads-gmi 2>&1 | tee "${ARTIFACT_DIR}/host-mthreads-gmi.txt" || true
fi

log "Running ${MODE} tests in ${IMAGE}"
CONTAINER_COMMAND='
set -euo pipefail
cleanup_artifacts() {
    local exit_code=$?
    chown -R "${MUSA_CI_HOST_UID}:${MUSA_CI_HOST_GID}" /musa-ci-artifacts 2>/dev/null || true
    exit "${exit_code}"
}
trap cleanup_artifacts EXIT
if [[ "${MUSA_CI_ACTIVATE_SGLANG:-0}" == "1" ]]; then
    test -f /root/.virtualenvs/sglang-default/bin/activate
    source /root/.virtualenvs/sglang-default/bin/activate
fi
workdir="$(mktemp -d /tmp/lmcache-ci.XXXXXX)"
cp -R /mnt/LMCache-src/. "${workdir}"
cd "${workdir}"
bash "$1"
'

docker run --rm \
    --pull=always \
    --ipc=host \
    --network=host \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${REPO_ROOT}:/mnt/LMCache-src:ro" \
    -v "${ARTIFACT_DIR}:/musa-ci-artifacts" \
    -w /tmp \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc \
        "${CONTAINER_COMMAND}" \
        musa-ci "${INNER_SCRIPT}" || fail "MUSA ${MODE} container failed"
