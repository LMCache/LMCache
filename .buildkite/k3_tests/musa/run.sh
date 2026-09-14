#!/usr/bin/env bash
# Run LMCache unit and focused smoke suites on a self-hosted MUSA agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ARTIFACT_PATH="${MUSA_CI_ARTIFACT_DIR:-musa-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi
SERVER_LOG="${ARTIFACT_DIR}/lmcache-server.log"
VENV_DIR=""
SERVER_PID=""
PYTHON_BIN="${MUSA_CI_PYTHON:-python}"
INSTALL_CMD=()
FREEZE_CMD=()

log() {
    echo "--- :musa: $*"
}

fail() {
    mkdir -p "${ARTIFACT_DIR}" 2>/dev/null || true
    printf '[musa-ci] ERROR: %s\n' "$*" \
        | tee -a "${ARTIFACT_DIR}/failure.log" >&2
    exit 1
}

check_musa_runtime() {
    local phase="$1"
    local output_file="$2"

    log "Checking the MUSA runtime and hardware (${phase})"
    if "${PYTHON_BIN}" - "${phase}" <<'PY' 2>&1 | tee "${output_file}"; then
import ctypes
import importlib.metadata
import os
import sys

import torch
import torch_musa


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


assert hasattr(torch, "musa"), "torch.musa is unavailable after importing torch_musa"
assert torch.musa.is_available(), "torch.musa.is_available() returned False"
device_count = torch.musa.device_count()
assert device_count > 0, "no MUSA device is visible"

try:
    ctypes.CDLL("libmusart.so")
except OSError as exc:
    raise AssertionError(f"libmusart.so is not loadable: {exc}") from exc

print("phase=", sys.argv[1])
print("python=", sys.executable)
print("torch=", torch.__version__)
print("torch_musa_runtime=", getattr(torch.version, "musa", "unknown"))
print("torch_musa=", package_version("torch_musa"))
print("torch_musa_module=", getattr(torch_musa, "__version__", "unknown"))
print("MUSA_VISIBLE_DEVICES=", os.environ["MUSA_VISIBLE_DEVICES"])
print("MUSA_HOME=", os.environ.get("MUSA_HOME", "<unset>"))
print("musa_device_count=", device_count)
print("musa_current_device=", torch.musa.current_device())
for device_index in range(device_count):
    try:
        device_name = torch.musa.get_device_name(device_index)
    except Exception as exc:
        device_name = f"<unavailable: {exc}>"
    print(f"musa_device_{device_index}=", device_name)
print("libmusart.so=loadable")

probe = torch.arange(6, dtype=torch.float32, device="musa:0").reshape(2, 3)
probe_result = probe @ probe.T
torch.musa.synchronize()
probe_result_cpu = probe_result.cpu().tolist()
assert probe_result_cpu == [[5.0, 14.0], [14.0, 50.0]], probe_result_cpu
print("musa_tensor_device=", probe.device)
print("musa_matmul_result=", probe_result_cpu)
PY
        return
    fi

    fail "MUSA runtime check failed during ${phase}; see "\
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

run_pytest() {
    "${PYTHON_BIN}" - "$@" <<'PY'
import sys

import pytest
import torch_musa  # noqa: F401 - registers torch.musa before test collection

raise SystemExit(pytest.main(sys.argv[1:]))
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

    if [[ -n "${VENV_DIR}" && -d "${VENV_DIR}" ]]; then
        rm -rf -- "${VENV_DIR}"
    fi

    return "${exit_code}"
}

trap cleanup EXIT

mkdir -p "${ARTIFACT_DIR}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
    fail "${PYTHON_BIN} is required in the MUSA environment"
if [[ "${MUSA_CI_UNIT_ONLY:-0}" != "1" ]]; then
    command -v curl >/dev/null 2>&1 || \
        fail "curl is required for the server smoke test"
fi
[[ -n "${MUSA_VISIBLE_DEVICES:-}" ]] || fail \
    "MUSA_VISIBLE_DEVICES must be set by the Buildkite agent or pipeline"

cd "${REPO_ROOT}"

if [[ "${MUSA_CI_PREPROVISIONED:-0}" == "1" ]]; then
    log "Using the pre-provisioned container Python and TorchMUSA stack"
    INSTALL_CMD=("${PYTHON_BIN}" -m pip install)
    FREEZE_CMD=("${PYTHON_BIN}" -m pip freeze)
else
    VENV_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lmcache-musa-ci.XXXXXX")"
    log "Creating an isolated environment while preserving the pinned TorchMUSA stack"
    BASE_PYTHON="$(command -v "${PYTHON_BIN}")"
    if command -v uv >/dev/null 2>&1; then
        log "Using uv for environment and package management"
        uv venv --system-site-packages --python "${BASE_PYTHON}" "${VENV_DIR}" || \
            fail "uv failed to create the temporary environment"
        INSTALL_CMD=(uv pip install)
        FREEZE_CMD=(uv pip freeze)
    else
        log "uv is unavailable; using the standard venv and pip fallback"
        "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}" || \
            fail "python -m venv failed; install the Python venv module on the agent"
        INSTALL_CMD=(python -m pip install)
        FREEZE_CMD=(python -m pip freeze)
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate" || \
        fail "failed to activate the temporary environment"
    PYTHON_BIN=python
    python -m pip --version >/dev/null 2>&1 || \
        python -m ensurepip --upgrade >/dev/null || \
        fail "pip is unavailable and ensurepip could not install it"
fi

check_musa_runtime \
    "before dependency setup" \
    "${ARTIFACT_DIR}/runtime-preflight.txt"

if [[ "${MUSA_CI_PREPROVISIONED:-0}" == "1" ]]; then
    log "Installing current LMCache dependencies around the pinned TorchMUSA stack"
else
    log "Installing LMCache build and test dependencies"
fi
"${INSTALL_CMD[@]}" \
    -r requirements/build.txt \
    -r requirements/common.txt \
    -r requirements/test.txt

check_musa_runtime \
    "after dependency setup" \
    "${ARTIFACT_DIR}/runtime-post-install.txt"

log "Building LMCache from the current checkout with the MUSA profile"
BUILD_WITH_MUSA=1 \
BUILD_MOONCAKE=0 \
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0+ci \
    "${INSTALL_CMD[@]}" --no-deps -e . --no-build-isolation

"${FREEZE_CMD[@]}" > "${ARTIFACT_DIR}/pip-freeze.txt"

log "Verifying LMCache selected the MUSA backend and built native support"
"${PYTHON_BIN}" - <<'PY' 2>&1 | tee "${ARTIFACT_DIR}/lmcache-preflight.txt"
import torch_musa  # noqa: F401 - registers torch.musa for auto-detection

import lmcache
import lmcache.lmcache_native as lmcache_native

assert lmcache.torch_device_type == "musa", (
    f"LMCache selected {lmcache.torch_device_type!r}, expected 'musa'"
)
print("lmcache_version=", lmcache.__version__)
print("lmcache_device=", lmcache.torch_device_type)
print("lmcache_native=", lmcache_native.__file__)
PY

PYTEST_ARGS=(-q --maxfail=1 -rs)
if [[ -n "${TEST_SELECTOR:-}" ]]; then
    PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
fi

if [[ "${MUSA_CI_UNIT_ONLY:-0}" == "1" ]]; then
    discover_unit_tests() {
        "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

allowlist = (
    "tests/test_*.py",
    "tests/cli/**/test_*.py",
    "tests/v1/**/test_*.py",
)
excluded = {
    # These modules import optional CUDA/Triton/vLLM components during
    # collection and are covered by their platform-specific jobs.
    "tests/v1/compute/attention/test_triton_kernels.py",
    "tests/v1/test_pos_kernels.py",
    # These lanes require CUDA/NIXL-specific host services.
    "tests/v1/test_device_id_race.py",
    "tests/v1/test_nixl_batched_contains.py",
    "tests/v1/test_nixl_multipath.py",
    "tests/v1/storage_backend/test_eic.py",
}

selected: set[str] = set()
for pattern in allowlist:
    selected.update(
        path.as_posix()
        for path in Path(".").glob(pattern)
        if path.is_file()
    )

for path in sorted(selected - excluded):
    print(path)
PY
    }

    mapfile -t UNIT_TEST_FILES < <(discover_unit_tests)
    if [ "${#UNIT_TEST_FILES[@]}" -eq 0 ]; then
        fail "no MUSA unit-test files found under tests"
    fi

    log "Running ${#UNIT_TEST_FILES[@]} MUSA-compatible unit-test files"
    printf '  %s\n' "${UNIT_TEST_FILES[@]}"
    run_pytest "${PYTEST_ARGS[@]}" \
        -m "not cuda and not xpu and not sglang" \
        "${UNIT_TEST_FILES[@]}" \
        2>&1 | tee "${ARTIFACT_DIR}/pytest.log"
    log "MUSA unit tests finished successfully"
    exit 0
fi

log "Running focused MUSA connector and transfer tests"
run_pytest "${PYTEST_ARGS[@]}" \
    tests/v1/test_musa_support.py \
    tests/v1/test_musa_connector.py \
    tests/v1/test_musa_native.py \
    tests/v1/platform/musa/test_musa_pin_memory.py \
    tests/v1/platform/musa/test_musa_staging_copy.py \
    tests/v1/platform/musa/test_musa_mp_block_transfer.py::test_musa_block_transfer_device_non_mla_d2h_and_h2d \
    tests/v1/platform/musa/test_musa_mp_block_transfer.py::test_musa_block_transfer_device_mla_d2h_and_h2d \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_keeps_layout_validation_device_agnostic \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_store_uses_device_agnostic_gather \
    tests/v1/multiprocess/test_engine_driven_transfer.py::test_musa_data_context_retrieve_uses_device_agnostic_scatter \
    2>&1 | tee "${ARTIFACT_DIR}/pytest.log"

ZMQ_PORT="${MUSA_CI_ZMQ_PORT:-6555}"
HTTP_PORT="${MUSA_CI_HTTP_PORT:-7555}"

log "Starting the LMCache multiprocess server smoke test"
LMCACHE_DEVICE_BACKEND=musa lmcache server \
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

    if curl -fsS "http://127.0.0.1:${HTTP_PORT}/healthcheck" >/dev/null; then
        log "LMCache server is healthy"
        break
    fi

    sleep 1
done

if ! curl -fsS "http://127.0.0.1:${HTTP_PORT}/healthcheck" >/dev/null; then
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
log "MUSA hardware smoke test finished successfully"
