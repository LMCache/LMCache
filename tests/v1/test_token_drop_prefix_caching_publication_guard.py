# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache publication isolation against pinned vLLM."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest


@pytest.fixture
def guarded_block_pool():
    vllm = pytest.importorskip("vllm")
    if vllm.__version__.split("+", 1)[0] != "0.25.1":
        pytest.skip("Publication guard requires vLLM 0.25.1")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    original = BlockPool.cache_full_blocks
    try:
        install_token_drop_prefix_caching_publication_guard()
        yield BlockPool
    finally:
        BlockPool.cache_full_blocks = original


def _make_kv_cache_manager():
    # Third Party
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
    )
    import torch

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
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=16,
        hash_block_size=16,
    )


def _make_request(request_id, *, token_dropping):
    # Third Party
    from vllm.v1.core.kv_cache_utils import BlockHash

    config = (
        {"kv_transfer_params": {"lmcache.token_drop": {"algorithm": "rkv"}}}
        if token_dropping
        else None
    )
    return SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=33,
        sampling_params=SimpleNamespace(extra_args=config),
        block_hashes=[BlockHash(request_id.encode().ljust(32, b"0"))],
        num_tokens=33,
        skip_reading_prefix_cache=False,
    )


def _cache_one_block(manager, request):
    (blocks,) = manager.coordinator.allocate_new_blocks(
        request_id=request.request_id,
        num_tokens=16,
        num_tokens_main_model=16,
    )
    manager.cache_blocks(request, num_computed_tokens=16)
    return blocks[0]


@pytest.mark.parametrize(
    "first_request_kind",
    [
        pytest.param("token_dropping", id="token-dropping-first"),
        pytest.param("normal", id="normal-first"),
    ],
)
def test_token_dropping_blocks_stay_private_and_normal_blocks_are_cached(
    guarded_block_pool, monkeypatch, first_request_kind
):
    # An unsupported partial-publication path must not silently be exercised.
    monkeypatch.setattr(
        guarded_block_pool,
        "cache_partial_block",
        lambda *_args, **_kwargs: pytest.fail("Unexpected partial publication"),
    )
    manager = _make_kv_cache_manager()
    requests = {
        "token_dropping": _make_request("token-dropping", token_dropping=True),
        "normal": _make_request("normal", token_dropping=False),
    }
    second_request_kind = (
        "normal" if first_request_kind == "token_dropping" else "token_dropping"
    )
    blocks = {
        kind: _cache_one_block(manager, requests[kind])
        for kind in (first_request_kind, second_request_kind)
    }

    assert blocks["token_dropping"].block_hash is None
    assert manager.get_computed_blocks(requests["token_dropping"])[1] == 0
    assert blocks["normal"].block_hash is not None
    assert manager.get_computed_blocks(requests["normal"])[1] == 16


def test_guard_rejects_unsupported_vllm_before_patching():
    version = pytest.importorskip("vllm.version")

    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    original = BlockPool.cache_full_blocks
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(version, "__version__", "0.26.0")
        with pytest.raises(RuntimeError, match="requires vLLM 0.25.1"):
            install_token_drop_prefix_caching_publication_guard()
    assert BlockPool.cache_full_blocks is original


def test_repeated_guard_installation_does_not_stack_wrappers(guarded_block_pool):
    # First Party
    from lmcache.integration.vllm.experimental.token_drop_adapter import (
        install_token_drop_prefix_caching_publication_guard,
    )

    installed = guarded_block_pool.cache_full_blocks
    install_token_drop_prefix_caching_publication_guard()
    assert guarded_block_pool.cache_full_blocks is installed
