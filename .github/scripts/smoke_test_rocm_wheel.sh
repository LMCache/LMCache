#!/bin/bash
# Smoke test the prebuilt LMCache ROCm wheel inside the upstream
# vllm/vllm-openai-rocm image, on a real AMD Instinct GPU.
#
# Proves three things end to end:
#   1. ABI load: the wheel (built against public torch 2.11 ROCm) imports
#      against the image's fork torch 2.11, and its *compiled* c_ops loads
#      (not the torch-baseline fallback).
#   2. HIP runtime: a device query round-trips to the GPU.
#   3. Kernel execution: the real KV-transfer kernels run on this gfx target.
#
# Run with: --device /dev/kfd --device /dev/dri --group-add video --group-add render
set -euxo pipefail

cd /work/LMCache
WHEEL=$(ls dist/*.whl | head -1)

# Install the wheel. Deps already live in the vLLM image; --no-deps avoids
# pulling a second torch/cupy.
pip install --no-deps --force-reinstall "${WHEEL}"

# (1) ABI load + (2) HIP runtime. Run from a neutral CWD so the checked-out
# source `lmcache/` package does not shadow the installed wheel. The attr-count
# assertion guards against silently falling back to the torch baseline
# (CudaDeviceOps) when the compiled extension fails to load.
( cd /tmp && python3 - <<'PY'
import torch
import lmcache.c_ops as c
attrs = [x for x in dir(c) if not x.startswith("__")]
assert len(attrs) >= 50, (
    f"compiled c_ops not loaded (only {len(attrs)} attrs) -- "
    "lmcache fell back to the torch baseline"
)
print("c_ops loaded; attrs:", len(attrs), "| gpu0 pci:", c.get_gpu_pci_bus_id(0))
assert torch.cuda.is_available(), "torch HIP runtime does not see a device"
print("device arch:", torch.cuda.get_device_properties(0).gcnArchName)
PY
)

# (3) Kernel execution on-device. Stage the wheel's compiled extensions into
# the checked-out tree and run the kernel suite from there, so the freshly
# built .so is exercised against its matching source. Running pytest against
# the installed wheel from within the repo would double-import lmcache
# (source + site-packages) and segfault, so we consolidate on source + wheel .so.
python3 - <<'PY'
import glob, zipfile
whl = glob.glob("dist/*.whl")[0]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if n.startswith("lmcache/") and n.endswith(".so"):
            z.extract(n, ".")
print("staged compiled extensions into source tree")
PY

# Minimal runtime deps the kernel tests import (image already has torch/numpy).
# cufile-python is CUDA-only; drop it on ROCm.
grep -v cufile-python requirements/common.txt > /tmp/common.txt
pip install --no-deps pytest -r /tmp/common.txt
python3 -m pytest tests/v1/test_mem_kernels.py -q -p no:cacheprovider --noconftest

echo "ROCm wheel smoke test PASSED"
