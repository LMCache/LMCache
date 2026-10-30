# SPDX-License-Identifier: Apache-2.0
"""Verify prefix-cache publication isolation for token-dropping requests."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest


@pytest.fixture
def installed_publication_guard():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Prefix-cache publication guard requires pinned vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    original_cache_full_blocks = BlockPool.cache_full_blocks
    try:
        install_token_drop_prefix_caching_publication_guard()
        yield
    finally:
        BlockPool.cache_full_blocks = original_cache_full_blocks


def _make_kv_cache_manager():
    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    import torch

    return KVCacheManager(
        kv_cache_config=KVCacheConfig(
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
        ),
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
    )


def _make_request(request_id, *, token_dropping=False, hash_byte=b"0"):
    # Third Party
    from vllm.v1.core.kv_cache_utils import BlockHash

    extra_args = (
        {
            "kv_transfer_params": {
                "lmcache.token_drop": {"algorithm": "rkv", "config": {}}
            }
        }
        if token_dropping
        else None
    )
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=33,
        sampling_params=SimpleNamespace(extra_args=extra_args),
        block_hashes=[BlockHash(hash_byte * 32)],
        num_tokens=33,
        skip_reading_prefix_cache=False,
    )


def _allocate_and_publish_block(manager, request):
    # Exercise the real allocation and publication path used by KVCacheManager.
    (blocks,) = manager.coordinator.allocate_new_blocks(
        request_id=request.request_id,
        num_tokens=16,
        num_tokens_main_model=16,
    )
    manager.cache_blocks(request, num_computed_tokens=16)
    return blocks[0]


def test_token_dropping_blocks_are_not_published_but_normal_blocks_are(
    installed_publication_guard, monkeypatch
):
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    def unexpected_partial_publication(*args, **kwargs):
        pytest.fail("Unexpected partial prefix-cache publication")

    monkeypatch.setattr(
        BlockPool, "cache_partial_block", unexpected_partial_publication
    )

    manager = _make_kv_cache_manager()
    token_drop_request = _make_request(
        "token-drop", token_dropping=True, hash_byte=b"a"
    )
    normal_request = _make_request("normal", hash_byte=b"b")

    token_drop_block = _allocate_and_publish_block(manager, token_drop_request)
    normal_block = _allocate_and_publish_block(manager, normal_request)

    assert token_drop_block.block_hash is None
    assert normal_block.block_hash is not None
    assert manager.get_computed_blocks(token_drop_request)[1] == 0
    assert manager.get_computed_blocks(normal_request)[1] == 16


def test_publication_guard_rejects_unsupported_vllm_version():
    version = pytest.importorskip("vllm.version")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    original_cache_full_blocks = BlockPool.cache_full_blocks
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(version, "__version__", "0.26.0")
        with pytest.raises(RuntimeError, match="requires vLLM 0.25.1"):
            install_token_drop_prefix_caching_publication_guard()
    assert BlockPool.cache_full_blocks is original_cache_full_blocks


def test_publication_guard_installation_is_idempotent(
    installed_publication_guard,
):
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    wrapped_cache_full_blocks = BlockPool.cache_full_blocks
    install_token_drop_prefix_caching_publication_guard()

    assert BlockPool.cache_full_blocks is wrapped_cache_full_blocks
