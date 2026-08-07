#!/bin/bash
# Build the LMCache Intel XPU/SYCL wheel.
#
# Runs inside an Intel oneAPI build image. No GPU is required to compile.
# Produces a manylinux-tagged wheel in /work/LMCache/dist that excludes torch
# and oneAPI/SYCL runtime libs (bound at runtime by the host XPU image).
#
# Env (set by workflow, with sensible defaults for local runs):
#   TORCH_XPU_SPEC             pip torch spec      (default detected from preinstalled torch)
#   TORCH_XPU_INDEX            torch wheel index   (default xpu index)
#   SKIP_AUDITWHEEL_REPAIR     1 to skip repair    (default auto-detect)
#   SETUPTOOLS_SCM_PRETEND_VERSION wheel version   (default 0.0.0.dev0)
set -euxo pipefail

if command -v python3.12 >/dev/null 2>&1; then
    PY=python3.12
else
    PY=python3
fi

DEFAULT_TORCH_SPEC="$($PY - <<'PY'
try:
    import torch
    print(f"torch=={torch.__version__}")
except Exception:
    print("torch==2.12.0+xpu")
PY
)"

TORCH_XPU_SPEC="${TORCH_XPU_SPEC:-$DEFAULT_TORCH_SPEC}"
TORCH_XPU_INDEX="${TORCH_XPU_INDEX:-https://download.pytorch.org/whl/xpu}"

export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# The repo is bind-mounted from the CI runner (owned by a non-root UID) while
# this container runs as root, so git refuses to operate on it. setup.py's
# version/git introspection runs during the wheel build, so mark it safe.
git config --global --add safe.directory /work/LMCache

# oneAPI toolchain env for icpx/dpcpp.
if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
    export OCL_ICD_FILENAMES="${OCL_ICD_FILENAMES:-}"
    # oneAPI setvars can trip over unset optional vars in some images.
    # Load it with nounset disabled, then restore nounset for the rest.
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
    set -u
fi

$PY --version
icpx --version

# Build against XPU torch; ABI compatibility is keyed to this torch minor.
INSTALLED_TORCH="$($PY - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception:
    print("")
PY
)"
REQUESTED_TORCH="${TORCH_XPU_SPEC#torch==}"
if [[ -z "$INSTALLED_TORCH" || "$INSTALLED_TORCH" != "$REQUESTED_TORCH" ]]; then
    $PY -m pip install --no-cache-dir "${TORCH_XPU_SPEC}" --index-url "${TORCH_XPU_INDEX}"
fi

$PY -m pip install --no-cache-dir \
    ninja "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel pybind11 auditwheel patchelf
$PY -c 'import torch; print("BUILD TORCH:", torch.__version__, "xpu:", hasattr(torch, "xpu"), "cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)'

cd /work/LMCache
rm -rf build dist dist_xpu

export BUILD_WITH_SYCL=1
$PY setup.py bdist_wheel --dist-dir=dist_xpu

# Repair: exclude torch and oneAPI/SYCL runtime libs so the wheel binds to the
# host image at runtime (same policy as ROCm/CUDA variant wheels).
SKIP_AUDITWHEEL_REPAIR="${SKIP_AUDITWHEEL_REPAIR:-}"
if [[ -z "$SKIP_AUDITWHEEL_REPAIR" ]]; then
    if command -v patchelf >/dev/null 2>&1; then
        SKIP_AUDITWHEEL_REPAIR=0
    else
        SKIP_AUDITWHEEL_REPAIR=1
    fi
fi

if [[ "$SKIP_AUDITWHEEL_REPAIR" == "1" ]]; then
    echo "patchelf unavailable; skipping auditwheel repair and copying raw wheel"
    mkdir -p dist
    cp dist_xpu/*.whl dist/
else
    $PY -m auditwheel repair \
        --plat manylinux_2_35_x86_64 \
        --exclude 'libtorch*.so*' \
        --exclude 'libc10*.so*' \
        --exclude 'libsycl.so*' \
        --exclude 'libze_loader.so*' \
        --exclude 'libur_loader.so*' \
        --exclude 'libOpenCL.so*' \
        --exclude 'libiomp5.so*' \
        --exclude 'libtbb.so*' \
        --exclude 'libmkl_sycl*.so*' \
        --exclude 'libmkl_intel_ilp64.so*' \
        --exclude 'libmkl_intel_lp64.so*' \
        --exclude 'libmkl_core.so*' \
        -w dist dist_xpu/*.whl
fi

echo "=== XPU extension dynamic deps (post-repair wheel) ==="
python3 -c "import zipfile,glob; w=glob.glob('dist/*.whl')[0]; zipfile.ZipFile(w).extractall('/tmp/whcheck_xpu')"
SO=$(find /tmp/whcheck_xpu -name 'xpu_ops*.so')
readelf -d "$SO" | grep NEEDED || true

echo "=== final XPU wheel ==="
ls -la /work/LMCache/dist/
echo "Install hint (runtime image): docker run --rm --shm-size=4g -v /path/to/LMCache:/work/LMCache --entrypoint bash <xpu-runtime-image> -lc 'python3 -m pip install --no-deps /work/LMCache/dist/*.whl'"