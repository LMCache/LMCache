#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# Print the failing command and line number on any error.
trap 'echo "ERROR: setup-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

echo "--- :python: Installing vLLM nightly"
# Resolve the latest nightly wheel URL directly from the nightly index.
# PEP 440 ranks stable releases (0.17.0) above pre-release nightlies
# (0.17.0rc1.devN), so pip/uv always picks the stable version when both
# indexes are available. We work around this by parsing the nightly index
# page and installing the wheel by URL.
ARCH=$(uname -m)  # x86_64 or aarch64
VLLM_NIGHTLY_INDEX="https://wheels.vllm.ai/nightly/vllm/"
INDEX_HTML=$(curl -sfL "$VLLM_NIGHTLY_INDEX" 2>&1) || true
VLLM_NIGHTLY_URL=$(echo "$INDEX_HTML" \
    | grep -oP 'href="\K[^"]+'"${ARCH}"'\.whl' \
    | head -1) || true
if [[ -z "$VLLM_NIGHTLY_URL" ]]; then
    echo "WARNING: Could not find vLLM nightly wheel for ${ARCH} — falling back to latest stable" >&2
    uv pip install "vllm[runai,tensorizer,flashinfer]"
else
    # href is relative (../../<commit>/vllm-....whl), resolve to absolute URL
    VLLM_WHEEL_URL="https://wheels.vllm.ai/nightly/vllm/${VLLM_NIGHTLY_URL}"
    echo "Resolved nightly wheel: $VLLM_WHEEL_URL"
    uv pip install --prerelease=allow \
        "${VLLM_WHEEL_URL}[runai,tensorizer,flashinfer]" \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-best-match
fi

# vLLM nightlies periodically add eager imports of packages that aren't in
# their declared deps (e.g. `pandas` from vllm/_aiter_ops.py). Probe-import
# vllm's CLI entry point and auto-install any ModuleNotFoundError modules
# so the job keeps going. Capped to avoid infinite loops; every auto-install
# is logged so the drift is visible in the build output.
MAX_AUTO_INSTALL=5
for i in $(seq 1 "$MAX_AUTO_INSTALL"); do
    if err=$(python -c "from vllm.entrypoints.cli.main import main" 2>&1); then
        break
    fi
    mod=$(printf '%s\n' "$err" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
    if [[ -z "$mod" ]]; then
        echo "vLLM import failed with a non-ModuleNotFoundError:" >&2
        echo "$err" >&2
        exit 1
    fi
    if [[ "$i" == "$MAX_AUTO_INSTALL" ]]; then
        echo "Hit $MAX_AUTO_INSTALL auto-install retries; last missing module: $mod" >&2
        echo "$err" >&2
        exit 1
    fi
    echo "Auto-installing missing vLLM runtime dep: $mod"
    uv pip install "$mod"
done

echo "--- :wrench: Aligning nvcc with torch's reported CUDA version"
# The base image's nvcc major version and vLLM nightly's torch CUDA major
# version have drifted apart before (image bumped to CUDA 13 while torch
# shipped cu128, then torch rolled to cu130). If they mismatch,
# torch.utils.cpp_extension._check_cuda_version refuses to compile
# LMCache's CUDAExtension. Detect the mismatch and install the matching
# `cuda-compiler-<major>-<minor>` package via NVIDIA's apt repo so the
# build gets a nvcc whose major version lines up with torch. No-op when
# they already agree.
read -r TORCH_CUDA TORCH_CUDA_MAJOR TORCH_CUDA_MINOR < <(python -c "
import torch
v = torch.version.cuda or ''
parts = v.split('.') + ['0']
print(v, parts[0], parts[1])
")
SYS_NVCC_MAJOR=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\).*/\1/p' | head -1)
echo "torch.version.cuda=${TORCH_CUDA}; system nvcc major=${SYS_NVCC_MAJOR:-none}"

if [[ -z "$TORCH_CUDA" ]]; then
    echo "torch has no CUDA version (CPU-only build?); skipping nvcc alignment"
    CUDA_HOME_BUILD=""
elif [[ "$TORCH_CUDA_MAJOR" == "$SYS_NVCC_MAJOR" ]]; then
    echo "System nvcc major matches torch; using system nvcc for LMCache build"
    CUDA_HOME_BUILD=""
else
    APT_PKG="cuda-compiler-${TORCH_CUDA_MAJOR}-${TORCH_CUDA_MINOR}"
    CUDA_HOME_BUILD="/usr/local/cuda-${TORCH_CUDA_MAJOR}.${TORCH_CUDA_MINOR}"
    echo "Major version mismatch; installing ${APT_PKG} to get matching nvcc at ${CUDA_HOME_BUILD}"
    if [[ ! -x "${CUDA_HOME_BUILD}/bin/nvcc" ]]; then
        apt-get update -y
        apt-get install -y --no-install-recommends "$APT_PKG"
    fi
    if [[ ! -x "${CUDA_HOME_BUILD}/bin/nvcc" ]]; then
        echo "ERROR: nvcc still missing at ${CUDA_HOME_BUILD}/bin/nvcc after apt install" >&2
        ls /usr/local/ >&2 || true
        exit 1
    fi
    "${CUDA_HOME_BUILD}/bin/nvcc" --version
fi

echo "--- :python: Installing LMCache from source"
if [[ -n "$CUDA_HOME_BUILD" ]]; then
    CUDA_HOME="$CUDA_HOME_BUILD" uv pip install -e . --no-build-isolation
else
    uv pip install -e . --no-build-isolation
fi

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
