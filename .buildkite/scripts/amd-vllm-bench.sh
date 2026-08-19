#!/usr/bin/env bash
# Run the shared vLLM benchmark end to end on a bare-metal ROCm agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

uv venv --python 3.12 ".venv-${BUILDKITE_BUILD_ID}"
# shellcheck disable=SC1090
source ".venv-${BUILDKITE_BUILD_ID}/bin/activate"
uv pip install --upgrade pip setuptools wheel
uv pip install -r requirements/build.txt

# vLLM's ROCm wheel provides the matching torch build. Install it before
# building LMCache so the HIP extensions link against the same torch runtime.
# The AMD agents run ROCm 7.0, whose newest archived vLLM release is 0.18.1.
VLLM_VERSION="${VLLM_VERSION:-0.18.1}"
VLLM_ROCM_VARIANT="${VLLM_ROCM_VARIANT:-rocm700}"
uv pip install "vllm[runai,tensorizer]==${VLLM_VERSION}+${VLLM_ROCM_VARIANT}" \
    --extra-index-url \
    "https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}" \
    --index-strategy unsafe-best-match

export AMD_SERIALIZE_KERNEL=1
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
export TORCH_DONT_CHECK_COMPILER_ABI=1
export CXX=hipcc
export BUILD_WITH_HIP=1

uv pip install -r requirements/rocm_core.txt
uv pip install -e . --no-build-isolation
uv pip install openai pandas matplotlib
uv pip freeze

# vllm_bench compares LMCache-enabled vLLM against a baseline server, so it
# needs two devices. Convert the selector's host-visible list into the physical
# device IDs expected by the shared native-process launcher.
# shellcheck disable=SC1091
source .buildkite/scripts/pick-free-gpu-amd.sh 70000 2
IFS=',' read -r GPU_FOR_VLLM GPU_FOR_BASELINE <<< "${HIP_VISIBLE_DEVICES}"
if [[ -z "${GPU_FOR_VLLM}" || -z "${GPU_FOR_BASELINE}" ]]; then
    echo "Expected two AMD GPUs, got HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
    exit 1
fi
export GPU_FOR_VLLM GPU_FOR_BASELINE
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

# Let vLLM select the ROCm attention backend instead of the CUDA FLASH_ATTN
# default used by the K3 NVIDIA jobs.
export ATTENTION_BACKEND=auto
export BATCH_INVARIANT=0
export LMCACHE_TRACK_USAGE=false
export RESULTS_DIR="${REPO_ROOT}/amd-vllm-bench-results"

exec .buildkite/k3_tests/multiprocess/scripts/run-single-test.sh vllm_bench
