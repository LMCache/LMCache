#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROCM_VERSION="${ROCM_VERSION:-7.0}"
BASE_IMAGE="${BASE_IMAGE:-rocm/dev-ubuntu-24.04:${ROCM_VERSION}-complete}"
ROCM_PLATFORM="${ROCM_PLATFORM:-linux/amd64}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942,gfx950}"
PYTORCH_ROCM_INDEX_URL="${PYTORCH_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm${ROCM_VERSION}}"
MAX_JOBS="${MAX_JOBS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dist/rocm}"
SCM_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-}"

if [[ -z "${SCM_VERSION}" && -f "${REPO_ROOT}/.git" ]]; then
    SCM_VERSION="0.0.0.dev0"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to build the ROCm wheel" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cat <<EOF
Building LMCache ROCm wheel
  base image:              ${BASE_IMAGE}
  target platform:         ${ROCM_PLATFORM}
  ROCm version:            ${ROCM_VERSION}
  PyTorch ROCm index:      ${PYTORCH_ROCM_INDEX_URL}
  PYTORCH_ROCM_ARCH:       ${PYTORCH_ROCM_ARCH}
  MAX_JOBS:                ${MAX_JOBS}
  output directory:        ${OUTPUT_DIR}
EOF

build_args=(
    --build-arg "ROCM_VERSION=${ROCM_VERSION}"
    --build-arg "BASE_IMAGE=${BASE_IMAGE}"
    --build-arg "ROCM_PLATFORM=${ROCM_PLATFORM}"
    --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    --build-arg "PYTORCH_ROCM_INDEX_URL=${PYTORCH_ROCM_INDEX_URL}"
    --build-arg "MAX_JOBS=${MAX_JOBS}"
)

if [[ -n "${SCM_VERSION}" ]]; then
    echo "  setuptools-scm version: ${SCM_VERSION}"
    build_args+=(--build-arg "SETUPTOOLS_SCM_PRETEND_VERSION=${SCM_VERSION}")
fi

DOCKER_BUILDKIT=1 docker build \
    --file "${REPO_ROOT}/docker/Dockerfile.rocm-wheel" \
    --target wheel-export \
    --output "type=local,dest=${OUTPUT_DIR}" \
    "${build_args[@]}" \
    "${REPO_ROOT}"

echo
echo "Built ROCm wheel artifact(s):"
ls -lh "${OUTPUT_DIR}"/lmcache-*.whl
