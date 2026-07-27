#!/bin/bash
# Smoke test the prebuilt LMCache ROCm wheel inside the upstream
# vllm/vllm-openai-rocm image, on a real AMD Instinct GPU.
#
# Proves three things end to end:
#   1. ABI load: the wheel (built against public torch 2.11 ROCm) imports
#      against the image's fork torch 2.11.
#   2. HIP runtime: a device query round-trips to the GPU.
#   3. Kernel execution: the real KV-transfer kernels run on this gfx target.
#
# Run with: --device /dev/kfd --device /dev/dri --group-add video
set -euxo pipefail

cd /work/LMCache
WHEEL=$(ls dist/*.whl | head -1)

# Install just the wheel (no deps: torch already present in the image).
pip install --no-deps --force-reinstall "${WHEEL}"

# (1) ABI load + (2) HIP runtime device query.
python3 - <<'PY'
import torch
import lmcache.c_ops as c
print("c_ops loaded; symbol count:", len([x for x in dir(c) if not x.startswith("__")]))
print("gpu0 pci bus id:", c.get_gpu_pci_bus_id(0))
assert torch.cuda.is_available(), "torch HIP runtime does not see a device"
PY

# (3) Kernel execution: run the real mem-kernel suite on the GPU. --noconftest
# avoids pulling the full engine dep tree; we install only what these tests
# import.
pip install --no-deps pytest sortedcontainers
python3 -m pytest tests/v1/test_mem_kernels.py -q -p no:cacheprovider --noconftest

echo "ROCm wheel smoke test PASSED"
