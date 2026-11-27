#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../../.."

# Do not inherit the moving vLLM nightly pin from the general CI lanes.
export PINNED_VLLM_VERSION=0.25.1
unset PINNED_VLLM_ARCHIVE_INDEX_URL PINNED_VLLM_FULL_SHA
export VLLM_CUDA_VARIANT=cu130
source .buildkite/k3_harness/setup-env.sh
uv pip install -r requirements/test.txt

python - <<'PY'
import torch
import vllm
from lmcache.v1.platform import torch_device_type

assert vllm.__version__.split("+", 1)[0] == "0.25.1", vllm.__version__
assert torch.cuda.is_available() and torch_device_type == "cuda", "CUDA required"
torch.ones(1, device="cuda").add_(1)
torch.cuda.synchronize()
print("CUDA device:", torch.cuda.get_device_name())
PY

# The worker tests land with the worker implementation. The GPU view tests
# are already in dev and must run even before the worker is introduced.
tests=(tests/v1/test_token_drop_gpu_views.py)
if [[ -f lmcache/integration/vllm/experimental/token_drop_worker.py || -f tests/v1/test_token_drop_worker.py ]]; then
    tests+=(tests/v1/test_token_drop_worker.py)
fi

rm -f junit-token-dropping-gpu.xml
PYTEST_ADDOPTS= python -m pytest -o addopts= -q -rs \
    --junitxml=junit-token-dropping-gpu.xml "${tests[@]}"
python "${SCRIPT_DIR}/verify-results.py" junit-token-dropping-gpu.xml "${tests[@]}"
