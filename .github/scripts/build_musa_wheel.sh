#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the LMCache MUSA-compatible wheel.
#
# The build runs in a vendor-provided TorchMUSA image.  LMCache's current MUSA
# profile deliberately has no in-tree GPU extension (the device transfer fast
# path is supplied by the optional musa_aiter package), so this wheel contains
# the common native extensions and the complete Python MUSA integration.  The
# script still requires the MUSA SDK/compiler at build time. TorchMUSA is
# validated separately on the device-backed Buildkite lane.
#
# Environment variables (the workflow sets the validated values):
#   MUSA_HOME                    MUSA SDK root (default /usr/local/musa)
#   MUSA_PYTHON                  Python executable (default auto-detected)
#   TORCH_MUSA_PACKAGES          Optional whitespace-separated torch packages
#   TORCH_MUSA_INDEX             Optional package index for those packages
#   MUSA_MANYLINUX_PLATFORM      auditwheel policy (default manylinux_2_35_x86_64)
#   SKIP_AUDITWHEEL_REPAIR       1 to intentionally keep the raw wheel
#   SETUPTOOLS_SCM_PRETEND_VERSION wheel version (default 0.0.0.dev0+musa)

set -euo pipefail
shopt -s nullglob

PROJECT_DIR="${MUSA_PROJECT_DIR:-/work/LMCache}"
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
MUSA_PYTHON="${MUSA_PYTHON:-}"
MUSA_MANYLINUX_PLATFORM="${MUSA_MANYLINUX_PLATFORM:-manylinux_2_35_x86_64}"
SKIP_AUDITWHEEL_REPAIR="${SKIP_AUDITWHEEL_REPAIR:-0}"

if [[ -z "${MUSA_PYTHON}" ]]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            MUSA_PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ -z "${MUSA_PYTHON}" ]]; then
    echo "No Python interpreter was found in the MUSA build image" >&2
    exit 1
fi

# The SDK ships an env script in different locations across MUSA releases.
# Source it only when mcc is not already available; sourcing twice can prepend
# duplicate library paths and makes diagnostics needlessly noisy.
export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib:${MUSA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
if ! command -v mcc >/dev/null 2>&1; then
    for env_script in \
        "${MUSA_HOME}/env.sh" \
        "${MUSA_HOME}/setvars.sh" \
        /etc/profile.d/musa.sh; do
        if [[ -f "${env_script}" ]]; then
            # shellcheck disable=SC1090
            set +u
            source "${env_script}" >/dev/null 2>&1 || true
            set -u
            break
        fi
    done
fi

command -v mcc >/dev/null 2>&1 || {
    echo "MUSA compiler mcc was not found; check MUSA_HOME and the builder image" >&2
    exit 1
}

export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0+musa}"
export MAX_JOBS="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"

"${MUSA_PYTHON}" --version
mcc --version
git config --global --add safe.directory "${PROJECT_DIR}"

if [[ -n "${TORCH_MUSA_PACKAGES:-}" ]]; then
    # Deliberately use an array so package names containing version operators
    # are passed to pip without shell evaluation or accidental concatenation.
    read -r -a torch_packages <<< "${TORCH_MUSA_PACKAGES}"
    pip_args=(--no-cache-dir)
    if [[ -n "${TORCH_MUSA_INDEX:-}" ]]; then
        pip_args+=(--index-url "${TORCH_MUSA_INDEX}")
    fi
    "${MUSA_PYTHON}" -m pip install "${pip_args[@]}" "${torch_packages[@]}"
fi

"${MUSA_PYTHON}" -c 'import torch; print("BUILD TORCH:", torch.__version__)'

"${MUSA_PYTHON}" -m pip install --no-cache-dir \
    ninja "setuptools>=77.0.3,<81.0.0" setuptools_scm wheel pybind11 auditwheel patchelf

cd "${PROJECT_DIR}"
rm -rf build dist dist_musa

# BUILD_WITH_MUSA selects the MUSA profile even when the image also exposes a
# CUDA-compatible compiler.  The profile currently builds common C++ modules;
# MUSA-specific kernels remain an explicit future extension of that profile.
export BUILD_WITH_MUSA=1
"${MUSA_PYTHON}" setup.py bdist_wheel --dist-dir=dist_musa

raw_wheels=(dist_musa/*.whl)
if [[ "${#raw_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one raw MUSA wheel, found ${#raw_wheels[@]}" >&2
    exit 1
fi

# Keep vendor-owned torch/MUSA userspace libraries out of the artifact.  They
# are intentionally resolved from the matching TorchMUSA runtime image.
if [[ "${SKIP_AUDITWHEEL_REPAIR}" == "1" ]]; then
    echo "SKIP_AUDITWHEEL_REPAIR=1; copying raw wheel"
    mkdir -p dist
    cp "${raw_wheels[0]}" dist/
else
    "${MUSA_PYTHON}" -m auditwheel repair \
        --plat "${MUSA_MANYLINUX_PLATFORM}" \
        --exclude 'libtorch*.so*' \
        --exclude 'libc10*.so*' \
        --exclude 'libmusa*.so*' \
        --exclude 'libmusart*.so*' \
        --exclude 'libmudnn*.so*' \
        --exclude 'libmccl*.so*' \
        --exclude 'libmcc*.so*' \
        --exclude 'libmusa_python*.so*' \
        -w dist "${raw_wheels[0]}"
fi

repaired_wheels=(dist/*.whl)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
    echo "Expected exactly one repaired MUSA wheel, found ${#repaired_wheels[@]}" >&2
    exit 1
fi

check_dir="$(mktemp -d)"
cleanup() {
    rm -rf -- "${check_dir}"
}
trap cleanup EXIT

"${MUSA_PYTHON}" - "${repaired_wheels[0]}" "${check_dir}" "${SETUPTOOLS_SCM_PRETEND_VERSION}" "${MUSA_MANYLINUX_PLATFORM}" <<'PY'
import sys
import zipfile
from pathlib import Path

wheel, destination, expected_version, expected_platform = sys.argv[1:]
with zipfile.ZipFile(wheel) as archive:
    archive.extractall(destination)
metadata = next(Path(destination).glob("*.dist-info/METADATA"))
wheel_metadata = next(Path(destination).glob("*.dist-info/WHEEL"))
version_line = next(
    line for line in metadata.read_text().splitlines() if line.startswith("Version: ")
)
actual_version = version_line.partition(": ")[2]
if actual_version != expected_version:
    raise SystemExit(
        f"wheel version mismatch: expected {expected_version!r}, got {actual_version!r}"
    )
if "+musa" not in actual_version:
    raise SystemExit(f"MUSA wheel is missing the +musa local version: {actual_version}")
if expected_platform not in wheel_metadata.read_text():
    raise SystemExit(
        f"wheel is missing the requested platform tag {expected_platform!r}"
    )
print("MUSA wheel metadata:", actual_version)
PY

echo "=== bundled MUSA/torch libraries (must be empty) ==="
"${MUSA_PYTHON}" - "${repaired_wheels[0]}" <<'PY'
import re
import sys
import zipfile

wheel = sys.argv[1]
pattern = re.compile(r"lib(?:torch|c10|musa|musart|mudnn|mccl|mcc).*\.so", re.I)
with zipfile.ZipFile(wheel) as archive:
    leaked = [name for name in archive.namelist() if pattern.search(name)]
if leaked:
    raise SystemExit("vendor TorchMUSA libraries were bundled: " + ", ".join(leaked))
PY

echo "=== final MUSA wheel ==="
ls -la dist/
