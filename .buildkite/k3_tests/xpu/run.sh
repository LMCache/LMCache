#!/usr/bin/env bash
# XPU smoke test entrypoint.
# Runs directly inside the agent-stack-k8s job pod.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_XPU_DEVICE_PLUGIN_SOURCE_URL="https://github.com/opendataio/lmcache_xpu_device_with_native/archive/refs/tags/v0.1.3.tar.gz"
XPU_DEVICE_PLUGIN_SOURCE_URL="${LMCACHE_XPU_DEVICE_PLUGIN_SOURCE_URL:-${DEFAULT_XPU_DEVICE_PLUGIN_SOURCE_URL}}"

log() {
  echo "[intel-xpu-unit-tests] $*"
}

fail() {
  echo "[intel-xpu-unit-tests] ERROR: $*" >&2
  exit 1
}

assert_in_tree_xpu_removed() {
  [ ! -d "${REPO_ROOT}/lmcache/v1/platform/xpu" ] || fail "in-tree lmcache/v1/platform/xpu is still present"
  [ ! -d "${REPO_ROOT}/csrc/sycl" ] || fail "in-tree csrc/sycl is still present"
  [ ! -f "${REPO_ROOT}/setup_extensions/build_profiles/sycl.py" ] || fail "in-tree setup_extensions/build_profiles/sycl.py is still present"
  [ -f "${REPO_ROOT}/csrc/kv_transfer_types.h" ] || fail "shared header csrc/kv_transfer_types.h is missing"
  [ -f "${REPO_ROOT}/csrc/engine_kv_format.h" ] || fail "shared header csrc/engine_kv_format.h is missing"
}

build_and_install_external_xpu_device_plugin() {
  local plugin_build_dir
  local plugin_source_dir
  local wheel_path

  plugin_build_dir="$(mktemp -d)"

  log "downloading external native XPU device plugin source"
  log "source url: ${XPU_DEVICE_PLUGIN_SOURCE_URL}"
  plugin_source_dir="$(python - "${XPU_DEVICE_PLUGIN_SOURCE_URL}" "${plugin_build_dir}" <<'PY'
import pathlib
import sys
import tarfile
import urllib.request

source_url = sys.argv[1]
workdir = pathlib.Path(sys.argv[2])
archive = workdir / "plugin.tar.gz"

urllib.request.urlretrieve(source_url, archive)
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(workdir)

dirs = sorted(path for path in workdir.iterdir() if path.is_dir())
assert len(dirs) == 1, dirs
print(dirs[0])
PY
)"

  log "building external native XPU device wheel"
  (
    cd "${plugin_source_dir}"
    export LMCACHE_CSRC_INCLUDE_DIR="${REPO_ROOT}/csrc"
    uv pip install build
    python -m build --wheel --no-isolation
  )

  wheel_path="$(find "${plugin_source_dir}/dist" -maxdepth 1 -name '*.whl' | head -n 1)"
  [ -n "${wheel_path}" ] || fail "failed to build external XPU device wheel"
  log "installing built wheel: ${wheel_path}"
  uv pip install "${wheel_path}"

  python - <<'PY'
import importlib.metadata

from lmcache.v1.platform import get_device_spec, resolve_device_ops

entry_points = importlib.metadata.entry_points(group="lmcache.device_plugins")
assert any(ep.name == "xpu" for ep in entry_points), "xpu entry point not found"

spec = get_device_spec("xpu")
assert spec is not None, "xpu DeviceSpec not resolved"
assert spec.backend_name == "xpu", spec.backend_name

ops = resolve_device_ops("xpu")
assert ops.device_type == "xpu", ops.device_type
import lmcache.xpu_ops as xpu_ops
assert xpu_ops is not None

print("external native xpu device plugin resolved successfully")
PY
}

[ -f /opt/intel/oneapi/setvars.sh ] || fail "/opt/intel/oneapi/setvars.sh not found"
# shellcheck disable=SC1091
log "enable Intel XPU runtime environment"
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true

log "checking torch.xpu and xpu h/w availability"
python - <<'PY'
import torch

assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU not available in pod"
print("torch.xpu.is_available() = True")
PY

cd "${REPO_ROOT}"
assert_in_tree_xpu_removed
source "${REPO_ROOT}/.buildkite/k3_harness/setup-lmcache-only-env.sh"

log "installing job dependencies"
uv pip install -r requirements/common.txt -r requirements/test.txt
build_and_install_external_xpu_device_plugin

discover_xpu_tests() {
  python - <<'PY'
from pathlib import Path

root = Path(".")
# Temporary allowlist of XPU cases that are known to run in the current
# vLLM-based XPU image.
#
# Future plan:
#   1. Replace this allowlist with denylist + blocklist and run full test
#      discovery by default.
#   2. denylist: CUDA-only tests that should never run in XPU jobs.
#   3. blocklist: tests that are expected to fail on current XPU runtime and
#      are not ready yet.
#   4. As tests become ready, shrink blocklist until XPU jobs run the full set.
allowlist = {
  "tests/benchmarks/test_*.py",
  "tests/test_*.py",
  "tests/cli/**/test_*.py",
  "tests/disagg/test_*.py",
  "tests/v1/**/test_*.py",
}
selected: set[str] = set()

for pattern in allowlist:
    for path in root.glob(pattern):
        if path.is_file():
            selected.add(path.as_posix())

for rel in sorted(selected):
    print(rel)
PY
}

mapfile -t XPU_TEST_FILES < <(discover_xpu_tests)
if [ "${#XPU_TEST_FILES[@]}" -eq 0 ]; then
  fail "no XPU-related test files found under tests"
fi

log "discovered ${#XPU_TEST_FILES[@]} XPU-related test files"
printf '  %s\n' "${XPU_TEST_FILES[@]}"

PYTEST_ARGS=(-q --maxfail=1 -m "not cuda and not musa and not sglang" --ignore=tests/v1/gpu_connector/)
if [ -n "${TEST_SELECTOR:-}" ]; then
  PYTEST_ARGS+=(-k "${TEST_SELECTOR}")
fi

log "running XPU-related tests"
pytest "${PYTEST_ARGS[@]}" "${XPU_TEST_FILES[@]}"
log "xpu smoke test finished successfully"
