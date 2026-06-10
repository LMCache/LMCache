#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash HMA registration check against a live LMCache MP server.

Run via ``run-single-test.sh hma_dsv4_flash``, which starts the LMCache MP
server through the standard ``launch-processes.sh`` flow (with
``LAUNCH_VLLM=false``); this script drives its own in-process vLLM engine.

Dummy-loads a 4-layer DeepSeek-V4-Flash through vLLM with the
``LMCacheMPConnector``, then asserts (via the LMCache server's HTTP
``/status``) that every KV cache group registered with the per-group
geometry the vLLM specs declare:

- the 256-token MLA / indexer-k group derives compress ratios 4, 4 and 128
  from ``tokens_per_block`` (spec ``block_size``) over the physical
  ``slots_per_block`` detected from the registered tensors;
- the per-layer 64-token sliding-window-MLA groups and the 4-/8-token
  compressor state groups derive compress ratio 1.

Model warmup and the profiling forward are skipped: registration happens
during engine init and does not require runnable sparse-MLA kernels, so
this check also runs on GPUs FlashMLA-Sparse does not support (the
capability check is bypassed there).
"""

# Standard
import json
import os
import sys
import urllib.request

# Must be set before vLLM/torch are imported: the hardware-independence
# patches below only apply in-process, and the engine must share the test's
# GPU selection (the same GPU launch-processes.sh gave the LMCache server).
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("GPU_FOR_VLLM", "0"))

LMCACHE_PORT = int(os.environ.get("LMCACHE_PORT", "6555"))
LMCACHE_HTTP_PORT = int(os.environ.get("LMCACHE_HTTP_PORT", "8080"))

GiB = 1024**3

# ---- Hardware-independence patches (registration-only check) ------------

# Third Party
from vllm.v1.attention.backends.mla.flashmla_sparse import (  # noqa: E402
    FlashMLASparseBackend,
)
import torch  # noqa: E402

# Let the backend selector pick FLASHMLA_SPARSE even on GPUs that cannot
# run it: registration is pure metadata + tensor allocation.
cap_major = torch.cuda.get_device_capability()[0]
if cap_major not in (9, 10):
    print(f"NOTE: forcing FlashMLA-Sparse capability on sm major {cap_major}")
    FlashMLASparseBackend.supports_compute_capability = classmethod(lambda cls, c: True)

# A vLLM build whose C extension predates the V4 clamped-silu op raises
# AttributeError at op registration; fall back to the native forward (the
# op never runs in this check).
try:
    # Third Party
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp

    _orig_clamp_init = SiluAndMulWithClamp.__init__

    def _patched_clamp_init(self, swiglu_limit, *, compile_native=True):
        try:
            _orig_clamp_init(self, swiglu_limit, compile_native=compile_native)
        except AttributeError:
            self.swiglu_limit = float(swiglu_limit)
            self._forward_method = self.forward_native

    SiluAndMulWithClamp.__init__ = _patched_clamp_init
except ImportError:
    pass

# Skip the memory-profiling forward and warmup (both would launch kernels).
# Third Party
from vllm.v1.executor.abstract import Executor  # noqa: E402
from vllm.v1.worker.gpu_worker import Worker  # noqa: E402
from vllm.v1.worker.worker_base import CompilationTimes  # noqa: E402

Executor.determine_available_memory = lambda self: [8 * GiB]
Worker.compile_or_warm_up_model = lambda self: CompilationTimes(
    language_model=0.0, encoder=0.0
)

# ---- Engine init (registration happens here) -----------------------------

# Third Party
from vllm import LLM  # noqa: E402

LLM(
    model="deepseek-ai/DeepSeek-V4-Flash",
    kv_transfer_config={
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.port": LMCACHE_PORT,
            "lmcache.mp.mq_timeout": 30,
        },
    },
    load_format="dummy",
    hf_overrides={"num_hidden_layers": 4},
    enforce_eager=True,
    kv_cache_dtype="fp8_ds_mla",
    max_model_len=4096,
    gpu_memory_utilization=0.4,
)

# ---- Verify the server-side registration ----------------------------------

with urllib.request.urlopen(
    f"http://127.0.0.1:{LMCACHE_HTTP_PORT}/status", timeout=30
) as resp:
    status = json.load(resp)


def find_kernel_group_lists(node: object) -> list[list[dict]]:
    """Recursively find every "kernel_groups" list in the status JSON.

    The same registered context may be reported under more than one key;
    callers deduplicate identical lists.
    """
    found: list[list[dict]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "kernel_groups" and isinstance(value, list):
                found.append(value)
            else:
                found.extend(find_kernel_group_lists(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(find_kernel_group_lists(item))
    return found


group_lists = find_kernel_group_lists(status)
unique_lists = {json.dumps(gl, sort_keys=True) for gl in group_lists}
if len(unique_lists) != 1:
    print("DSV4_HMA_CHECK: FAIL")
    print(f" - expected one registered context, got {len(unique_lists)} distinct")
    sys.exit(1)
groups = group_lists[0]

print(f"Registered kernel groups: {len(groups)}")
for g in groups:
    print(
        f"  kg{g['kernel_group_idx']}: engine_group={g['engine_group_idx']} "
        f"layers={g['num_layers']} tokens_per_block={g['tokens_per_block']} "
        f"slots_per_block={g['slots_per_block']} "
        f"compress_ratio={g['compress_ratio']} dtype={g['dtype']}"
    )

failures: list[str] = []

# Per-group (tokens_per_block, compress_ratio) pairs the 4-layer
# DeepSeek-V4-Flash kv_cache_config declares:
# - MLA / indexer-k group (256-token blocks): ratios 4, 4, 128
# - per-layer SWA groups (64-token blocks): ratio 1
# - compressor state groups (4- and 8-token blocks): ratio 1
expected_pairs = sorted(
    [
        (256, 4),
        (256, 4),
        (256, 128),
        (64, 1),
        (64, 1),
        (64, 1),
        (64, 1),
        (4, 1),
        (4, 1),
        (8, 1),
    ]
)
actual_pairs = sorted((g["tokens_per_block"], g["compress_ratio"]) for g in groups)
if actual_pairs != expected_pairs:
    failures.append(
        f"per-group (tokens_per_block, compress_ratio) mismatch:\n"
        f"  expected {expected_pairs}\n"
        f"  actual   {actual_pairs}"
    )

for g in groups:
    if g["tokens_per_block"] != g["slots_per_block"] * g["compress_ratio"]:
        failures.append(
            f"kernel group {g['kernel_group_idx']}: tokens_per_block "
            f"{g['tokens_per_block']} != slots_per_block "
            f"{g['slots_per_block']} * compress_ratio {g['compress_ratio']}"
        )

if failures:
    print("DSV4_HMA_CHECK: FAIL")
    for failure in failures:
        print(" -", failure)
    sys.exit(1)

print("DSV4_HMA_CHECK: PASS")
