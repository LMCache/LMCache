# SPDX-License-Identifier: Apache-2.0
"""Kernel-paged vLLM groups round-trip through the unchanged MP transfer path.

When the attention backend cannot page at the configured block size, vLLM
pages the tensor at a smaller kernel block size, so block ID ``b`` covers
pages ``b*k .. b*k+k-1``. The vLLM connector describes such a group per page
(``create_engine_group_infos_from_vllm``) and expands its block IDs to page IDs
(``expand_block_ids_to_kernel_pages``). These tests feed exactly that to a real
worker transfer context and server and check that every page of each block
moves, which ``dev`` got wrong by reading the extra pages as slot compression.

Tensors are ``[2, NB, X, X, HS]`` with the two middle axes equal, so the block
size is the same whichever of NHD / HND the detector resolves.
"""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    create_engine_group_infos_from_vllm,
    expand_block_ids_to_kernel_pages,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

CHUNK_TOKENS = 32
BLOCK_TOKENS = 16  # tokens one vLLM block ID covers
PAGE_TOKENS = 8  # kernel page size of the kernel-paged group
PAGES_PER_BLOCK = BLOCK_TOKENS // PAGE_TOKENS
NUM_BLOCKS = 6  # vLLM block IDs; block 0 is the null block
NUM_TOKENS = 64  # two chunks, four vLLM blocks
HEAD_SIZE = 4


@dataclass
class FullAttentionSpec:
    """vLLM spec double; the connector reads specs by class name."""

    block_size: int


@dataclass
class SlidingWindowSpec:
    """vLLM spec double; the connector reads specs by class name."""

    block_size: int
    sliding_window: int


def _cache(num_pages: int, page_tokens: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(2, num_pages, page_tokens, page_tokens, HEAD_SIZE)


def _pages(block_id: int) -> slice:
    return slice(block_id * PAGES_PER_BLOCK, (block_id + 1) * PAGES_PER_BLOCK)


def _round_trip(
    port: int,
    use_shm: bool,
    kv_cache_config: SimpleNamespace,
    source: dict[str, torch.Tensor],
    store_ids: list[list[int]],
    retrieve_ids: list[list[int]],
) -> dict[str, torch.Tensor]:
    """Register ``source`` the way the vLLM connector does, store it from
    ``store_ids``, then retrieve into zeroed tensors at ``retrieve_ids``.

    Block IDs are vLLM block IDs, one list per vLLM group; they are expanded
    to page IDs as the connector's worker adapter does.
    """
    infos, pages_per_block = create_engine_group_infos_from_vllm(
        kv_cache_config, source
    )

    def to_pages(block_ids: list[list[int]]) -> list[list[int]]:
        return [
            expand_block_ids_to_kernel_pages(block_ids[info.engine_group_id], k)
            for info, k in zip(infos, pages_per_block, strict=True)
        ]

    # The adapter counts a single group's chunk in the IDs it sends.
    blocks_in_chunk = CHUNK_TOKENS // infos[0].tokens_per_block
    shm_name = f"lmcache_l1_pool_test_kernel_paged_{uuid4().hex}" if use_shm else ""
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=8 * 1024 * 1024, use_lazy=False, shm_name=shm_name
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    engine = MPCacheServerContext(config, chunk_size=CHUNK_TOKENS)
    module = EngineDrivenTransferModule(engine)
    lookup = LookupModule(engine)
    server = create_request_server(
        [module, lookup], MPServerConfig(host="127.0.0.1", port=port)
    )
    server.start()
    client = RequestClientFactory.create(f"tcp://127.0.0.1:{port}")
    worker = EngineDrivenTransferContext(1, client)
    destination = {name: torch.zeros_like(t) for name, t in source.items()}
    key = IPCCacheServerKey.from_token_ids(
        "kp", 1, 0, [1] * NUM_TOKENS, start=0, end=NUM_TOKENS, request_id="kp"
    )
    try:
        worker.register(
            source,
            "kp",
            1,
            blocks_in_chunk,
            5.0,
            engine_group_infos=infos,
            lmcache_chunk_tokens=CHUNK_TOKENS,
        )
        assert worker.submit_store(
            key.request_id, key, source, to_pages(store_ids), None, blocks_in_chunk
        ).result(timeout=5)
        client.lookup(key, 1).result(timeout=5)
        assert client.wait_prefetch_status(key.request_id, 5).result(timeout=10) == 2
        assert worker.submit_retrieve(
            key.request_id,
            key,
            destination,
            to_pages(retrieve_ids),
            None,
            blocks_in_chunk,
        ).result(timeout=5)
        client.end_session(key.request_id).result(timeout=5)
        unregistration = worker.unregister()
        assert unregistration is not None
        unregistration.result(timeout=5)
    finally:
        worker.close()
        client.close()
        server.close()
        lookup.close()
        module.close()
        engine.close()
    return destination


@pytest.mark.parametrize("use_shm", [False, True], ids=["inline", "shm"])
def test_single_kernel_paged_group_moves_every_page(
    use_shm: bool, unused_tcp_port: int
) -> None:
    """Retrieving into other vLLM blocks restores both pages of each block,
    and pages outside the request stay untouched."""
    source = {
        "layer.0": _cache(NUM_BLOCKS * PAGES_PER_BLOCK, PAGE_TOKENS, seed=1),
    }
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer.0"],
                kv_cache_spec=FullAttentionSpec(block_size=BLOCK_TOKENS),
            )
        ],
        num_blocks=NUM_BLOCKS,
    )
    store_ids, retrieve_ids = [1, 2, 3, 4], [5, 4, 3, 2]

    destination = _round_trip(
        unused_tcp_port,
        use_shm,
        kv_cache_config,
        source,
        [store_ids],
        [retrieve_ids],
    )

    expected = torch.zeros_like(source["layer.0"])
    for s, d in zip(store_ids, retrieve_ids, strict=True):
        expected[:, _pages(d)] = source["layer.0"][:, _pages(s)]
    assert torch.equal(destination["layer.0"], expected)


@pytest.mark.parametrize("use_shm", [False, True], ids=["inline", "shm"])
def test_kernel_paged_sliding_window_group_beside_full_attention(
    use_shm: bool, unused_tcp_port: int
) -> None:
    """A full-attention group paged at its block size and a kernel-paged
    sliding-window group round-trip together. The window group's chunk of
    null blocks is never stored, and only its last block (the one-block
    window of the last chunk) comes back, page by page."""
    source = {
        "full.0": _cache(NUM_BLOCKS, BLOCK_TOKENS, seed=2),
        "swa.0": _cache(NUM_BLOCKS * PAGES_PER_BLOCK, PAGE_TOKENS, seed=3),
    }
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["full.0"],
                kv_cache_spec=FullAttentionSpec(block_size=BLOCK_TOKENS),
            ),
            SimpleNamespace(
                layer_names=["swa.0"],
                kv_cache_spec=SlidingWindowSpec(
                    block_size=BLOCK_TOKENS, sliding_window=BLOCK_TOKENS
                ),
            ),
        ],
        num_blocks=NUM_BLOCKS,
    )
    # vLLM nulls the window group's out-of-window blocks.
    full_store, full_retrieve = [1, 2, 3, 4], [5, 4, 3, 2]
    swa_store, swa_retrieve = [0, 0, 2, 3], [0, 0, 4, 5]

    destination = _round_trip(
        unused_tcp_port,
        use_shm,
        kv_cache_config,
        source,
        [full_store, swa_store],
        [full_retrieve, swa_retrieve],
    )

    expected_full = torch.zeros_like(source["full.0"])
    for s, d in zip(full_store, full_retrieve, strict=True):
        expected_full[:, d] = source["full.0"][:, s]
    assert torch.equal(destination["full.0"], expected_full)
    expected_swa = torch.zeros_like(source["swa.0"])
    expected_swa[:, _pages(5)] = source["swa.0"][:, _pages(3)]
    assert torch.equal(destination["swa.0"], expected_swa)
