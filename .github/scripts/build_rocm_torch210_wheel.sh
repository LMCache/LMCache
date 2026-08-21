#!/bin/bash
# Build the LMCache ROCm/HIP wheel against the PyTorch 2.10 ABI.
#
# This script runs in an environment that already contains the target torch.
# It must not install or upgrade torch: the purpose of this variant is to bind
# LMCache's native extensions to the ROCm torch 2.10 ABI. A pinned ATOM image
# currently provides that reproducible toolchain, but the wheel is not
# ATOM-specific.
#
# Env:
#   PYTORCH_ROCM_ARCH          gfx target list      (default gfx942;gfx950)
#   TORCH_VERSION_PREFIX      expected torch prefix (default 2.10.)
#   ROCM_VERSION_PREFIX       expected ROCm prefix  (default 7.2)
#   MANYLINUX_PLATFORM        output wheel policy   (default manylinux_2_39_x86_64)
#   SETUPTOOLS_SCM_PRETEND_VERSION wheel version    (default 0.0.0.dev0)
set -euxo pipefail

PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942;gfx950}"
TORCH_VERSION_PREFIX="${TORCH_VERSION_PREFIX:-2.10.}"
ROCM_VERSION_PREFIX="${ROCM_VERSION_PREFIX:-7.2}"
MANYLINUX_PLATFORM="${MANYLINUX_PLATFORM:-manylinux_2_39_x86_64}"
export TORCH_VERSION_PREFIX
export ROCM_VERSION_PREFIX
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"
export MAX_JOBS="${MAX_JOBS:-2}"

PY="${PYTHON:-python}"
REPO_ROOT="${LMCACHE_REPO_ROOT:-/work/LMCache}"

git config --global --add safe.directory "${REPO_ROOT}"

"${PY}" - <<'PY'
import os
import torch

torch_prefix = os.environ["TORCH_VERSION_PREFIX"]
rocm_prefix = os.environ["ROCM_VERSION_PREFIX"]
assert torch.__version__.startswith(torch_prefix), (
    f"ROCm torch 2.10 wheel requires torch {torch_prefix}*, found {torch.__version__}"
)
assert torch.version.hip is not None and torch.version.hip.startswith(rocm_prefix), (
    f"ROCm torch 2.10 wheel requires ROCm {rocm_prefix}*, found {torch.version.hip}"
)
print(
    "ROCM TORCH 2.10 BUILD ABI:",
    "torch", torch.__version__,
    "hip", torch.version.hip,
    "cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI,
)
PY

"${PY}" -m pip install --no-cache-dir \
    -r "${REPO_ROOT}/requirements/build.txt" \
    pybind11 auditwheel

cd "${REPO_ROOT}"
rm -rf build dist_rocm_torch210 dist_rocm_torch210_raw csrc_hip
find csrc -name '*_hip.*' -delete 2>/dev/null || true
find csrc -name '*.hip' -delete 2>/dev/null || true

export BUILD_WITH_HIP=1
export CXX=hipcc
export PYTORCH_ROCM_ARCH
"${PY}" setup.py bdist_wheel --dist-dir=dist_rocm_torch210_raw

"${PY}" -m auditwheel repair \
    --plat "${MANYLINUX_PLATFORM}" \
    --exclude 'libtorch*.so*' \
    --exclude 'libc10*.so*' \
    --exclude 'libamdhip64.so*' \
    --exclude 'libhsa-runtime64.so*' \
    --exclude 'librocprofiler-register.so*' \
    --exclude 'libamd_comgr.so*' \
    --exclude 'librocm-core.so*' \
    --exclude 'librocblas.so*' \
    --exclude 'libhipblas.so*' \
    --exclude 'libMIOpen.so*' \
    --exclude 'libdrm.so*' \
    --exclude 'libdrm_amdgpu.so*' \
    -w dist_rocm_torch210 dist_rocm_torch210_raw/*.whl

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

check_dir=$(mktemp -d)
"${PY}" -c "import glob,zipfile; w=glob.glob('dist_rocm_torch210/*.whl')[0]; zipfile.ZipFile(w).extractall('${check_dir}')"
cuda_ops=$(find "${check_dir}" -name 'cuda_ops*.so' -print -quit)
test -n "${cuda_ops}"
/opt/rocm/llvm/bin/llvm-objdump --offloading "${cuda_ops}" 2>/dev/null \
    | grep -oE 'gfx[0-9a-z]+' | sort -u \
    | tee "${check_dir}/gpu-archs.txt"
grep -qx 'gfx942' "${check_dir}/gpu-archs.txt"
grep -qx 'gfx950' "${check_dir}/gpu-archs.txt"

ls -la dist_rocm_torch210/
