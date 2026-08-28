#!/bin/bash
# Build the LMCache ROCm/HIP wheel against the PyTorch 2.10 ABI.
#
# This script runs in an environment that already contains the target torch.
# It must not install or upgrade torch: the purpose of this variant is to bind
# LMCache's native extensions to the exact public AMD torch build selected by
# the workflow. The wheel is not ATOM-specific.
#
# Env:
#   PYTORCH_ROCM_ARCH          gfx target list      (default gfx942;gfx950)
#   EXPECTED_TORCH_VERSION    exact torch runtime version
#   EXPECTED_TORCH_GIT_VERSION exact torch source revision
#   EXPECTED_HIP_VERSION      exact HIP runtime version
#   EXPECTED_ROCM_VERSION     exact ROCm release
#   EXPECTED_PYTHON_ABI       exact CPython ABI tag
#   EXPECTED_CXX11_ABI        torch C++ ABI flag (0 or 1)
#   MANYLINUX_PLATFORM        output wheel policy   (default manylinux_2_39_x86_64)
#   SETUPTOOLS_SCM_PRETEND_VERSION wheel version    (default 0.0.0.dev0)
set -euxo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=.github/scripts/rocm_wheel_common.sh
source "${SCRIPT_DIR}/rocm_wheel_common.sh"

PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
MANYLINUX_PLATFORM="${MANYLINUX_PLATFORM:-manylinux_2_39_x86_64}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"
export MAX_JOBS="${MAX_JOBS:-2}"

PY="${PYTHON:-python}"
REPO_ROOT="${LMCACHE_REPO_ROOT:-/work/LMCache}"

git config --global --add safe.directory "${REPO_ROOT}"

"${PY}" "${SCRIPT_DIR}/validate_rocm_torch210_wheel.py" --runtime-only

"${PY}" -m pip install --no-cache-dir \
    -r "${REPO_ROOT}/requirements/build.txt" \
    pybind11
install_rocm_repair_tools "${PY}"
preflight_rocm_repair_tools "${PY}"

cd "${REPO_ROOT}"
rm -rf build dist_rocm_torch210 dist_rocm_torch210_raw csrc_hip
find csrc -name '*_hip.*' -delete 2>/dev/null || true
find csrc -name '*.hip' -delete 2>/dev/null || true

export BUILD_WITH_HIP=1
export CXX=hipcc
export PYTORCH_ROCM_ARCH
"${PY}" setup.py bdist_wheel --dist-dir=dist_rocm_torch210_raw

raw_wheels=(dist_rocm_torch210_raw/*.whl)
test "${#raw_wheels[@]}" -eq 1
repair_rocm_wheel \
    "${PY}" "${MANYLINUX_PLATFORM}" "${raw_wheels[0]}" dist_rocm_torch210

"${PY}" - <<'PY'
from pathlib import Path
import zipfile

wheels = list(Path("dist_rocm_torch210").glob("*.whl"))
assert len(wheels) == 1, f"expected one ROCm torch 2.10 wheel, found {wheels}"
with zipfile.ZipFile(wheels[0]) as archive:
    names = archive.namelist()
assert any(name.startswith("lmcache/cuda_ops") and name.endswith(".so") for name in names)
assert any(
    name.startswith("lmcache/lmcache_native") and name.endswith(".so")
    for name in names
)
if Path("lmcache/integration/atom").is_dir():
    atom_paths = (
        "lmcache/integration/atom/__init__.py",
        "lmcache/integration/atom/multi_process_adapter.py",
        "lmcache/v1/gpu_connector/kv_format/detectors/atom.py",
    )
    for path in atom_paths:
        assert path in names, f"ROCm torch 2.10 wheel is missing {path}"
    print("ATOM integration is present in the wheel")
print("ROCM TORCH 2.10 WHEEL:", wheels[0])
PY

repaired_wheels=(dist_rocm_torch210/*.whl)
test "${#repaired_wheels[@]}" -eq 1
assert_rocm_offload_arches \
    "${PY}" "${repaired_wheels[0]}" "${PYTORCH_ROCM_ARCH}"

ls -la dist_rocm_torch210/
