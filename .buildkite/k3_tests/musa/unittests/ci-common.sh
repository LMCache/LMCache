#!/usr/bin/env bash
# Shared bootstrap for MUSA unit, smoke, and E2E jobs inside the vendor image.
# Sourced by the unit, multiprocess, and serving E2E entrypoints.
# shellcheck shell=bash

MUSA_CI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MUSA_CI_ROOT}/../../../.." && pwd)"
ARTIFACT_PATH="${MUSA_CI_ARTIFACT_DIR:-musa-ci-artifacts}"
if [[ "${ARTIFACT_PATH}" == /* ]]; then
    ARTIFACT_DIR="${ARTIFACT_PATH}"
else
    ARTIFACT_DIR="${REPO_ROOT}/${ARTIFACT_PATH}"
fi

PYTHON_BIN="${MUSA_CI_PYTHON:-python}"
INSTALL_CMD=()
FREEZE_CMD=()
SERVER_PID=""

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
    "${PYTHON_BIN}" "${MUSA_CI_ROOT}/pytest_runner.py" "$@"
}

musa_ci_cleanup() {
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

bootstrap_musa_ci() {
    local require_curl="${1:-0}"

    mkdir -p "${ARTIFACT_DIR}"
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
        fail "${PYTHON_BIN} is required in the MUSA environment"
    if [[ "${require_curl}" == "1" ]]; then
        command -v curl >/dev/null 2>&1 || \
            fail "curl is required for the server smoke test"
    fi
    [[ -n "${MUSA_VISIBLE_DEVICES:-}" ]] || fail \
        "MUSA_VISIBLE_DEVICES must be set by the Buildkite agent or pipeline"

    cd "${REPO_ROOT}"
    trap musa_ci_cleanup EXIT

    log "Using the pre-provisioned container Python and TorchMUSA stack"
    INSTALL_CMD=("${PYTHON_BIN}" -m pip install)
    FREEZE_CMD=("${PYTHON_BIN}" -m pip freeze)

    check_musa_runtime \
        "before dependency setup" \
        "${ARTIFACT_DIR}/runtime-preflight.txt"

    log "Installing current LMCache dependencies around the pinned TorchMUSA stack"
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
}

pytest_base_args() {
    PYTEST_ARGS=(-q --maxfail=1 -rs)
    if [[ -n "${TEST_SELECTOR:-}" ]]; then
        PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
    fi
}
