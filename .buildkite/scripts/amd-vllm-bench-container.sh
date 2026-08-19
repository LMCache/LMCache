#!/usr/bin/env bash
# Container-side setup for the AMD vLLM + LMCache end-to-end benchmark.
set -euo pipefail

REPO_ROOT="/workspace/LMCache"
cd "${REPO_ROOT}"

# The official image defaults to `vllm serve`. The host runner overrides its
# entrypoint with bash so the PR checkout can be built and tested first.
if ! command -v git >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends git
    rm -rf /var/lib/apt/lists/*
fi
git config --global --add safe.directory "${REPO_ROOT}"

export CXX=hipcc
export BUILD_WITH_HIP=1
export TORCH_DONT_CHECK_COMPILER_ABI=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"

uv pip install --system --no-cache -r requirements/build.txt
uv pip install --system --no-cache --no-build-isolation -e .
uv pip install --system --no-cache openai pandas matplotlib

# Fail during setup with the real import error instead of waiting for server
# health checks to time out. Also record the exact versions tested by CI.
vllm --help >/dev/null
python3 - <<'PY'
import lmcache
import os
import torch
import vllm

print(f"vLLM={vllm.__version__}")
print(f"torch={torch.__version__}, HIP={torch.version.hip}")
print(f"LMCache={lmcache.__version__}")
print(
    "Selected host AMD GPUs: "
    f"LMCache vLLM={os.getenv('GPU_FOR_VLLM', 'unset')}, "
    f"baseline={os.getenv('GPU_FOR_BASELINE', 'unset')}"
)
print(f"Visible torch CUDA/HIP devices: {torch.cuda.device_count()}")
for device_idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(device_idx)
    total_memory_gib = props.total_memory / (1024 ** 3)
    print(
        f"torch device {device_idx}: "
        f"name={props.name}, total_memory_gib={total_memory_gib:.1f}"
    )
PY

exec .buildkite/k3_tests/multiprocess/scripts/run-single-test.sh vllm_bench
