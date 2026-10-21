# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest


def test_pinned_vllm_apc_guards_preserve_normal_cache():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("APC guard requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    import torch

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_apc_guard,
    )

    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
    )
    normal = SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args=None),
        block_hashes=[BlockHash(b"0" * 32)],
        num_tokens=33,
        skip_reading_prefix_cache=False,
    )
    td = SimpleNamespace(
        sampling_params=SimpleNamespace(
            extra_args={
                "kv_transfer_params": {
                    "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
                }
            }
        ),
        block_hashes=normal.block_hashes,
        num_tokens=33,
        skip_reading_prefix_cache=False,
    )
    original_publish = BlockPool.cache_full_blocks
    try:
        install_token_drop_apc_guard()
        td.skip_reading_prefix_cache = True
        assert manager.get_computed_blocks(td) == (manager.empty_kv_cache_blocks, 0)

        block = manager.block_pool.get_new_blocks(1)[0]
        manager.block_pool.cache_full_blocks(td, [block], 0, 1, 16, 0)
        assert block.block_hash is None

        manager.block_pool.cache_full_blocks(normal, [block], 0, 1, 16, 0)
        assert block.block_hash is not None
        _, matched = manager.get_computed_blocks(normal)
        assert matched == 16
        assert manager.get_computed_blocks(td) == (manager.empty_kv_cache_blocks, 0)
    finally:
        BlockPool.cache_full_blocks = original_publish
