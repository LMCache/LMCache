#!/usr/bin/env bash
# Run the MUSA CI scripts in the maintainer-provisioned TorchMUSA image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODE="${1:-}"
IMAGE="${MUSA_CI_IMAGE:-sh-harbor.mthreads.com/ai-kv/kuae-lmcache-vllm:20260819-kuae-ssd-e2e-full-tests-musa-aiter-ipc}"
ARTIFACT_PATH="${MUSA_CI_ARTIFACT_DIR:-musa-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi
INNER_SCRIPT=""
DOCKER_ENV_ARGS=(
    -e "LMCACHE_TRACK_USAGE=${LMCACHE_TRACK_USAGE:-false}"
    -e "MUSA_CI_ARTIFACT_DIR=/musa-ci-artifacts"
    -e "MUSA_CI_PREPROVISIONED=1"
    -e "MUSA_CI_PYTHON=${MUSA_CI_PYTHON:-python3}"
    -e "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES:-0}"
)

log() {
    echo "--- :musa: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[musa-ci] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    exit 1
}

case "${MODE}" in
    unit)
        INNER_SCRIPT=".buildkite/k3_tests/musa/run-unit-tests.sh"
        ;;
    smoke)
        INNER_SCRIPT=".buildkite/k3_tests/musa/run.sh"
        ;;
    *)
        fail "usage: $0 {unit|smoke}"
        ;;
esac

mkdir -p "${ARTIFACT_DIR}"
command -v docker >/dev/null 2>&1 || fail "docker is required on the MUSA agent"

for optional_variable in \
    LMCACHE_DEVICE_BACKEND TEST_SELECTOR MUSA_CI_ZMQ_PORT MUSA_CI_HTTP_PORT; do
    if [[ -n "${!optional_variable:-}" ]]; then
        DOCKER_ENV_ARGS+=(-e "${optional_variable}=${!optional_variable}")
    fi
done

if command -v mthreads-gmi >/dev/null 2>&1; then
    mthreads-gmi 2>&1 | tee "${ARTIFACT_DIR}/host-mthreads-gmi.txt" || true
fi

log "Running ${MODE} tests in ${IMAGE}"
docker run --rm \
    --ipc=host \
    --network=host \
    "${DOCKER_ENV_ARGS[@]}" \
    -v "${REPO_ROOT}:/mnt/LMCache-src:ro" \
    -v "${ARTIFACT_DIR}:/musa-ci-artifacts" \
    -w /tmp \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc \
        'workdir="$(mktemp -d /tmp/lmcache-ci.XXXXXX)" && cp -R /mnt/LMCache-src/. "${workdir}" && cd "${workdir}" && bash "$1"' \
        musa-ci "${INNER_SCRIPT}" || fail "MUSA ${MODE} container failed"
