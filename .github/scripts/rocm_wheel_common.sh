#!/bin/bash
# Shared auditwheel tooling for LMCache ROCm wheels.

set -euo pipefail

install_rocm_repair_tools() {
    local python_bin=$1
    "${python_bin}" -m pip install --no-cache-dir auditwheel "patchelf>=0.17"
}

preflight_rocm_repair_tools() {
    local python_bin=$1
    "${python_bin}" - <<'PY'
from importlib.metadata import version
import shutil
import subprocess
import sys

patchelf = shutil.which("patchelf")
if patchelf is None:
    sys.exit("patchelf not found on PATH")
reported = subprocess.run(
    [patchelf, "--version"], capture_output=True, text=True, check=True
).stdout.strip()
print(f"BUILD auditwheel: {version('auditwheel')}")
print(f"BUILD patchelf:   {version('patchelf')} ({reported}) at {patchelf}")

try:
    from auditwheel.patcher import _verify_patchelf
except ImportError:
    # Internal helper moved; `repair` still enforces its own requirement.
    sys.exit(0)
try:
    _verify_patchelf("patchelf")
except TypeError:
    _verify_patchelf()  # signature before auditwheel grew patcher variants
print("BUILD patchelf accepted by auditwheel")
PY
}

repair_rocm_wheel() {
    local python_bin=$1
    local platform_tag=$2
    local input_wheel=$3
    local output_dir=$4

    # Bundle generic userspace libraries, but bind torch, ROCm, and driver
    # libraries at runtime. Globs intentionally cover versioned SONAMEs.
    "${python_bin}" -m auditwheel repair \
        --plat "${platform_tag}" \
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
        -w "${output_dir}" "${input_wheel}"
}

assert_rocm_offload_arches() {
    local python_bin=$1
    local wheel=$2
    local expected_arches=$3
    local check_dir
    local cuda_ops
    local arch

    check_dir=$(mktemp -d)
    "${python_bin}" -c \
        'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
        "${wheel}" "${check_dir}"
    cuda_ops=$(find "${check_dir}" -name 'cuda_ops*.so' -print -quit)
    test -n "${cuda_ops}"
    /opt/rocm/llvm/bin/llvm-objdump --offloading "${cuda_ops}" 2>/dev/null \
        | grep -oE 'gfx[0-9a-z]+' | sort -u \
        | tee "${check_dir}/gpu-archs.txt"

    while IFS= read -r arch; do
        test -z "${arch}" || grep -qx "${arch}" "${check_dir}/gpu-archs.txt"
    done < <(printf '%s\n' "${expected_arches}" | tr ';,' '\n')
    rm -rf "${check_dir}"
}
