# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.token_drop_apc_adapter import _cache_private_or_normal


def _request(spec=None):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"kv_transfer_params": spec})
    )


def test_normal_request_calls_original_without_changes():
    seen = []

    def original(*args):
        seen.append(args)

    pool = object()
    request = _request({"lmcache.max_offload_tokens": 32})
    blocks = [object()]
    _cache_private_or_normal(original, pool, request, blocks, 0, 1, 16, 0)
    assert seen == [(pool, request, blocks, 0, 1, 16, 0, None)]


def test_td_request_never_publishes_blocks():
    def original(*args):
        pytest.fail("TD block must not be published")

    _cache_private_or_normal(
        original,
        object(),
        _request({"lmcache.token_drop": {"algorithm": "rkv", "config": {}}}),
        [object()],
        0,
        1,
        16,
        0,
    )


def test_invalid_td_config_fails_closed():
    with pytest.raises(ValueError):
        _cache_private_or_normal(
            lambda *args: None,
            object(),
            _request({"lmcache.token_drop": {}}),
            [],
            0,
            1,
            16,
            0,
        )


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
    from lmcache.integration.vllm.token_drop_apc_adapter import (
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
    original_get = KVCacheManager.get_computed_blocks
    original_publish = BlockPool.cache_full_blocks
    try:
        install_token_drop_apc_guard()
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
        KVCacheManager.get_computed_blocks = original_get
        BlockPool.cache_full_blocks = original_publish
