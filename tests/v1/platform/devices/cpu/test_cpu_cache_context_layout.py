# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.v1.common.custom_types import EngineType
from lmcache.v1.platform.devices.cpu.cache_context import CPUCacheContext
from lmcache.v1.platform.devices.cpu.shm import migrate_to_shm_and_wrap
from lmcache.v1.platform.posix_shm import shm_unlink
import lmcache.lmcache_native as lmcache_native

F = lmcache_native.EngineKVFormat


@pytest.mark.parametrize(
    ("hints", "expected"),
    [
        (None, F.NL_X_TWO_NB_NH_BS_HS),
        ({}, F.NL_X_TWO_NB_NH_BS_HS),
        ({"kv_layout": "NHD"}, F.NL_X_TWO_NB_BS_NH_HS),
    ],
    ids=["none", "empty", "nhd"],
)
def test_vllm_layout_defaults_to_hnd_without_a_hint(hints, expected):
    tensors = [torch.zeros((2, 4, 16, 2, 8), dtype=torch.float16) for _ in range(2)]
    wrappers = [migrate_to_shm_and_wrap(t) for t in tensors]
    try:
        ctx = CPUCacheContext(
            wrappers,
            lmcache_tokens_per_chunk=16,
            layout_hints=hints,
            engine_type=EngineType.VLLM,
        )
        try:
            assert ctx.engine_kv_format_per_layer() == [expected, expected]
        finally:
            ctx.close()
    finally:
        for wrapper in wrappers:
            shm_unlink(wrapper.shm_name)
