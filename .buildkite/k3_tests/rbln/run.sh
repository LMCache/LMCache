#!/usr/bin/env bash
# Run the LMCache RBLN multiprocess hardware smoke on a self-hosted RBLN agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT_PATH="${RBLN_CI_ARTIFACT_DIR:-rbln-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi
SERVER_LOG="${ARTIFACT_DIR}/lmcache-server.log"
SERVER_PID=""
PYTHON_BIN="${RBLN_CI_PYTHON:-python3}"
INSTALL_CMD=()
FREEZE_CMD=()

log() {
    echo "--- :rbln: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[rbln-ci] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    exit 1
}

check_rbln_runtime() {
    local phase="$1"
    local output_file="$2"

    log "Checking the RBLN runtime and hardware (${phase})"
    if "${PYTHON_BIN}" - "${phase}" <<'PY' 2>&1 | tee "${output_file}"; then
import importlib.metadata
import os
import sys

import torch

assert hasattr(torch, "rbln"), "torch.rbln is unavailable; is torch-rbln installed?"
assert torch.rbln.is_available(), "torch.rbln.is_available() returned False"
device_count = torch.rbln.device_count()
assert device_count > 0, "no RBLN device is visible"

print("phase=", sys.argv[1])
print("python=", sys.executable)
print("torch=", torch.__version__)
print("torch_rbln=", importlib.metadata.version("torch-rbln"))
print("RBLN_VISIBLE_DEVICES=", os.environ.get("RBLN_VISIBLE_DEVICES", "<unset>"))
print("rbln_device_count=", device_count)

probe = torch.arange(6, dtype=torch.float32).reshape(2, 3).to("rbln:0")
probe_result = (probe + probe).cpu().tolist()
assert probe_result == [[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]], probe_result
print("rbln_tensor_device=", probe.device)
print("rbln_add_result=", probe_result)
PY
        return
    fi

    fail "RBLN runtime check failed during ${phase}; see "\
"${output_file#"${REPO_ROOT}/"}"
}

wait_for_process_exit() {
    local process_id="$1"
    local timeout_seconds="$2"
    local elapsed

    for ((elapsed = 0; elapsed < timeout_seconds; elapsed++)); do
        if ! kill -0 "${process_id}" 2>/dev/null; then
            wait "${process_id}"
            return $?
        fi
        sleep 1
    done

    return 124
}

# The rbln-serve image has no curl, so probe the endpoint with Python.
server_is_healthy() {
    "${PYTHON_BIN}" - "http://127.0.0.1:${HTTP_PORT}/healthcheck" \
        <<'PY' 2>/dev/null
import sys
import urllib.request

urllib.request.urlopen(sys.argv[1], timeout=5)
PY
}

cleanup() {
    local exit_code=$?
    set +e

    if [[ -n "${SERVER_PID}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null
        wait_for_process_exit "${SERVER_PID}" 10
        if [[ $? -eq 124 ]]; then
            kill -KILL "${SERVER_PID}" 2>/dev/null
            wait "${SERVER_PID}" 2>/dev/null
        fi
    fi

    return "${exit_code}"
}

trap cleanup EXIT

mkdir -p "${ARTIFACT_DIR}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    fail "${PYTHON_BIN} is required in the RBLN environment"

cd "${REPO_ROOT}"

# The rbln-serve image ships a matched torch-rbln / rebel-compiler pair in
# its Python environment; LMCache and its test deps are layered on top, so
# no RBLN package index (or its credentials) is needed.
INSTALL_CMD=("${PYTHON_BIN}" -m pip install --no-cache-dir)
FREEZE_CMD=("${PYTHON_BIN}" -m pip freeze)

check_rbln_runtime \
    "before LMCache dependency setup" \
    "${ARTIFACT_DIR}/runtime-preflight.txt"

log "Installing current LMCache dependencies around the torch-rbln stack"
"${INSTALL_CMD[@]}" \
    -r requirements/build.txt \
    -r requirements/common.txt \
    -r requirements/test.txt

check_rbln_runtime \
    "after LMCache dependency setup" \
    "${ARTIFACT_DIR}/runtime-post-install.txt"

# --no-deps: the requirements are already installed above, and skipping
# resolution keeps the image's torch-rbln / rebel-compiler pair untouched.
log "Building LMCache from the current checkout with the RBLN profile"
BUILD_WITH_RBLN=1 \
BUILD_MOONCAKE=0 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+ci \
    "${INSTALL_CMD[@]}" --no-deps -e . --no-build-isolation

"${FREEZE_CMD[@]}" > "${ARTIFACT_DIR}/pip-freeze.txt"

log "Verifying LMCache selected the RBLN backend"
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${ARTIFACT_DIR}/lmcache-preflight.txt"
import lmcache
import lmcache.lmcache_native as lmcache_native

assert lmcache.torch_device_type == "rbln", (
    f"LMCache selected {lmcache.torch_device_type!r}, expected 'rbln'"
)
print("lmcache_version=", lmcache.__version__)
print("lmcache_device=", lmcache.torch_device_type)
print("lmcache_native=", lmcache_native.__file__)
PY

PYTEST_ARGS=(-q --maxfail=1 -rs)
if [[ -n "${TEST_SELECTOR:-}" ]]; then
    PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
fi

log "Running the RBLN device-backend and real-NPU transfer tests"
"${PYTHON_BIN}" -m pytest "${PYTEST_ARGS[@]}" \
    tests/v1/platform/devices/rbln \
    2>&1 | tee "${ARTIFACT_DIR}/pytest.log"

ZMQ_PORT="${RBLN_CI_ZMQ_PORT:-6555}"
HTTP_PORT="${RBLN_CI_HTTP_PORT:-7555}"

log "Starting the LMCache multiprocess server smoke test"
LMCACHE_DEVICE_BACKEND=rbln lmcache server \
    --host 127.0.0.1 \
    --port "${ZMQ_PORT}" \
    --http-host 127.0.0.1 \
    --http-port "${HTTP_PORT}" \
    --l1-size-gb 0.25 \
    --no-l1-use-lazy \
    --eviction-policy LRU \
    --chunk-size 128 \
    --disable-metrics \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        tail -n 200 "${SERVER_LOG}" >&2
        fail "LMCache server exited before becoming healthy"
    fi

    if server_is_healthy; then
        log "LMCache server is healthy"
        break
    fi

    sleep 1
done

if ! server_is_healthy; then
    tail -n 200 "${SERVER_LOG}" >&2
    fail "LMCache server did not become healthy within 60 seconds"
fi

kill "${SERVER_PID}"
set +e
wait_for_process_exit "${SERVER_PID}" 30
SERVER_EXIT_CODE=$?
set -e

if [[ "${SERVER_EXIT_CODE}" -eq 124 ]]; then
    tail -n 200 "${SERVER_LOG}" >&2
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    fail "LMCache server did not stop within 30 seconds"
fi

if [[ "${SERVER_EXIT_CODE}" -ne 0 && "${SERVER_EXIT_CODE}" -ne 143 ]]; then
    tail -n 200 "${SERVER_LOG}" >&2
    SERVER_PID=""
    fail "LMCache server exited with status ${SERVER_EXIT_CODE} during shutdown"
fi

if ! grep -q "LMCache HTTP server stopped" "${SERVER_LOG}"; then
    tail -n 200 "${SERVER_LOG}" >&2
    SERVER_PID=""
    fail "LMCache server did not report a clean HTTP shutdown"
fi

SERVER_PID=""
log "RBLN MP hardware smoke test finished successfully"
