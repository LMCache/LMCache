#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the LMCache wheel for Rebellions RBLN NPUs.
#
# RBLN has no in-repo native kernels (KV moves with torch ops on torch.rbln),
# so the wheel carries only the common C++ extensions. Those link libtorch,
# so they are compiled against the exact torch that torch-rbln pins: the
# script installs torch-rbln and lets pip resolve that torch from the PyTorch
# CPU index. The script runs in a manylinux_2_34 image, the same glibc floor
# as torch-rbln's own wheels, so the wheel installs wherever torch-rbln does.
#
# Environment variables (the workflow sets the validated values):
#   TORCH_RBLN_VERSION             torch-rbln release that fixes the torch ABI
#   RBLN_PYTHON_VERSION            CPython version to build for (default 3.12)
#   RBLN_PYTHON                    Python executable (default: the manylinux
#                                  interpreter for RBLN_PYTHON_VERSION)
#   RBLN_MANYLINUX_PLATFORM        auditwheel policy (default manylinux_2_34_x86_64)
#   SETUPTOOLS_SCM_PRETEND_VERSION wheel version (default 0.0.0.dev0+rbln)

set -euo pipefail
shopt -s nullglob

PROJECT_DIR="${RBLN_PROJECT_DIR:-/work/LMCache}"
: "${TORCH_RBLN_VERSION:?TORCH_RBLN_VERSION must name the torch-rbln release to build against}"
RBLN_PYTHON_VERSION="${RBLN_PYTHON_VERSION:-3.12}"
python_tag="cp${RBLN_PYTHON_VERSION//./}"
RBLN_PYTHON="${RBLN_PYTHON:-/opt/python/${python_tag}-${python_tag}/bin/python}"
RBLN_MANYLINUX_PLATFORM="${RBLN_MANYLINUX_PLATFORM:-manylinux_2_34_x86_64}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0+rbln}"
export MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
# torch-rbln registers a torch device-backend entry point that loads the RBLN
# runtime from rebel-compiler, which the public PyPI index does not serve.
# The build needs only torch's headers and libraries.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

# auditwheel looks up patchelf on PATH; prefer the copy installed below over
# an older system one.
export PATH="$(dirname "${RBLN_PYTHON}"):${PATH}"

"${RBLN_PYTHON}" --version
git config --global --add safe.directory "${PROJECT_DIR}"
cd "${PROJECT_DIR}"

"${RBLN_PYTHON}" -m pip install --no-cache-dir \
    -r requirements/build.txt \
    build \
    auditwheel \
    patchelf
"${RBLN_PYTHON}" -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch-rbln==${TORCH_RBLN_VERSION}"
"${RBLN_PYTHON}" -c 'import torch; print("BUILD TORCH:", torch.__version__)'

rm -rf build dist dist_rbln

# BUILD_WITH_RBLN selects the RBLN profile explicitly instead of relying on
# torch_rbln detection. --skip-dependency-check: pyproject.toml pins the
# CUDA build's torch, while this build deliberately uses torch-rbln's torch.
BUILD_WITH_RBLN=1 "${RBLN_PYTHON}" -m build --wheel --no-isolation \
    --skip-dependency-check --outdir dist_rbln

raw_wheels=(dist_rbln/*.whl)
if [[ "${#raw_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one raw RBLN wheel, found ${#raw_wheels[@]}" >&2
    exit 1
fi

# torch's libraries must resolve to the torch that torch-rbln installed on
# the host; a bundled copy would load a second libtorch.
"${RBLN_PYTHON}" -m auditwheel repair \
    --plat "${RBLN_MANYLINUX_PLATFORM}" \
    --exclude 'libtorch*.so*' \
    --exclude 'libc10*.so*' \
    -w dist "${raw_wheels[0]}"

repaired_wheels=(dist/*.whl)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one repaired RBLN wheel, found ${#repaired_wheels[@]}" >&2
    exit 1
fi

"${RBLN_PYTHON}" - "${repaired_wheels[0]}" "${SETUPTOOLS_SCM_PRETEND_VERSION}" "${RBLN_MANYLINUX_PLATFORM}" <<'PY'
import re
import sys
import zipfile
from email.parser import Parser

from packaging.version import Version

from setup_extensions.common_cpp import COMMON_EXTENSIONS

wheel, expected_version, expected_platform = sys.argv[1:]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    dist_info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
    metadata = Parser().parsestr(archive.read(f"{dist_info}/METADATA").decode())
    wheel_metadata = archive.read(f"{dist_info}/WHEEL").decode()

version = metadata["Version"]
if Version(version) != Version(expected_version):
    raise SystemExit(f"wheel version mismatch: expected {expected_version!r}, got {version!r}")
if not version.endswith("+rbln"):
    raise SystemExit(f"RBLN wheel is missing the +rbln local version: {version}")
if expected_platform not in wheel_metadata:
    raise SystemExit(f"wheel is missing the platform tag {expected_platform!r}")

# RBLN builds no backend extension, so the compiled modules must be exactly
# the common ones: anything else means another profile leaked into the build.
built = {
    match.group(1)
    for name in names
    if (match := re.fullmatch(r"lmcache/(\w+)\.cpython-[^/]+\.so", name))
}
expected = {spec.name for spec in COMMON_EXTENSIONS}
if built != expected:
    raise SystemExit(f"extension modules {sorted(built)} != common {sorted(expected)}")

bundled_torch = [n for n in names if re.search(r"\.libs/lib(torch|c10)", n)]
if bundled_torch:
    raise SystemExit("torch libraries were bundled: " + ", ".join(bundled_torch))
print("RBLN wheel:", version, sorted(built))
PY

echo "=== final RBLN wheel ==="
ls -la dist/
