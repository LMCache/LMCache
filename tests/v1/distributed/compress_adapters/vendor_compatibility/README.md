# Portable Record Vendor Compatibility Probe

This standalone probe submits LMCache's frozen version-1 raw-Deflate and Gzip records to a real nvCOMP or hipCOMP installation. It copies the complete record to device memory and uses each record-relative payload offset directly, so a passing run verifies framing compatibility and the no-repacking alignment contract.

This probe is intentionally separate from the LMCache build. The production package does not yet depend on either vendor library. See the [portable record design](../../../../../docs/design/v1/distributed/compress_adapters/portable-record-format.md) for the wire layout and validation contract.

## NVIDIA nvCOMP 5.3

Run on a Linux host with an NVIDIA GPU, compatible driver, CUDA Toolkit 12, a C++17 compiler, and CMake 3.22 or newer. Install NVIDIA's pinned C/C++ wheel in a dedicated virtual environment:

```bash
PROBE_WORK_ROOT=/tmp/lmcache-vendor-probe
mkdir -p "$PROBE_WORK_ROOT"

python3 -m venv "$PROBE_WORK_ROOT/nvcomp-venv"
source "$PROBE_WORK_ROOT/nvcomp-venv/bin/activate"
python -m pip install nvidia-libnvcomp-cu12==5.3.0.16

NVCOMP_PREFIX=$(python -c \
  'from pathlib import Path; import nvidia.libnvcomp; print(Path(nvidia.libnvcomp.__file__).parent)')

cmake \
  -S tests/v1/distributed/compress_adapters/vendor_compatibility \
  -B "$PROBE_WORK_ROOT/nvcomp-record-probe" \
  -DLMCACHE_COMPAT_VENDOR=NVIDIA \
  -Dnvcomp_DIR="$NVCOMP_PREFIX/lib64/cmake/nvcomp"
cmake --build "$PROBE_WORK_ROOT/nvcomp-record-probe" --parallel
ctest \
  --test-dir "$PROBE_WORK_ROOT/nvcomp-record-probe" \
  --output-on-failure
```

For CUDA 13, use `nvidia-libnvcomp-cu13==5.3.0.16`. NVIDIA also publishes distribution packages; the `nvcomp-cuda-12` or `nvcomp-cuda-13` metapackage installs both headers and runtime libraries into system search paths.

Recorded compatibility result from the source branch (not a fresh run for this PR):

- nvCOMP 5.3.0.16, CUDA 12.9.86, driver 595.71.05, and an RTX 4060 passed both frozen raw-Deflate and Gzip records on 2026-08-20.

## AMD hipCOMP 2.3

Run on a Linux host with a supported AMD GPU and ROCm installed. Build the reviewed hipCOMP commit against that same ROCm installation:

```bash
PROBE_WORK_ROOT=/tmp/lmcache-vendor-probe
mkdir -p "$PROBE_WORK_ROOT"

git clone \
  https://github.com/ROCm/hipCOMP-core.git \
  "$PROBE_WORK_ROOT/hipcomp-src"
git -C "$PROBE_WORK_ROOT/hipcomp-src" checkout \
  22cc762f54fba7cdfca74a4c50c00f2aac4ace7a

cmake \
  -S "$PROBE_WORK_ROOT/hipcomp-src" \
  -B "$PROBE_WORK_ROOT/hipcomp-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm/lib/cmake \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_INSTALL_PREFIX="$PROBE_WORK_ROOT/hipcomp-install" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build "$PROBE_WORK_ROOT/hipcomp-build" --parallel
cmake --install "$PROBE_WORK_ROOT/hipcomp-build"

cmake \
  -S tests/v1/distributed/compress_adapters/vendor_compatibility \
  -B "$PROBE_WORK_ROOT/hipcomp-record-probe" \
  -DLMCACHE_COMPAT_VENDOR=AMD \
  -DCMAKE_PREFIX_PATH="/opt/rocm/lib/cmake;$PROBE_WORK_ROOT/hipcomp-install"
cmake --build "$PROBE_WORK_ROOT/hipcomp-record-probe" --parallel
ctest \
  --test-dir "$PROBE_WORK_ROOT/hipcomp-record-probe" \
  --output-on-failure
```

Replace `gfx942` with the host GPU architecture. hipCOMP 2.3 reports successful testing on `gfx1030`, `gfx1100`, `gfx90a`, `gfx942`, and `gfx950`. It is an early-access preview; Deflate and Gzip support is experimental and decompression-only, so a passing probe establishes compatibility rather than a production-support commitment.

An AMD hardware run of these fixtures is still pending.

## Expected result

The executable prints one success line per framing and CTest reports one passing test. A failure means the portable wire contract must be reconsidered before the shared backend interface or production integration is implemented.
