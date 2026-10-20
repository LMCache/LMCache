# SPDX-License-Identifier: Apache-2.0
"""Real LMCache backend tests with synthetic DeepInfer-shaped byte buffers."""

# Standard
from collections.abc import Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.deepinfer.snapshot import DeepInferSnapshotCache
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@pytest.fixture
def backend() -> Iterator[LocalCPUBackend]:
    config = LMCacheEngineConfig.from_defaults(local_cpu=True)
    metadata = LMCacheMetadata(
        "test-deepinfer", 1, 1, 0, 0, torch.uint8, (1, 1, 4, 1, 8)
    )
    allocator = TensorMemoryAllocator(
        torch.empty(4096, dtype=torch.uint8), align_bytes=512
    )
    result = LocalCPUBackend(config, metadata, memory_allocator=allocator)
    yield result
    result.close()
    assert allocator.total_allocated_size == 0


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_restores_relocated_blocks_and_full_state(
    backend: LocalCPUBackend, device: str
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    blocks = torch.arange(80, dtype=torch.uint8, device=device).reshape(10, 8)
    states = torch.arange(48, dtype=torch.uint8, device=device).reshape(4, 12)
    cache = DeepInferSnapshotCache(backend, blocks, states, 4, "model@revision/layout")
    expected_blocks = blocks[[7, 2]].clone()
    expected_state = states[1].clone()
    assert cache.store(list(range(8)), [7, 2, 9], 1)
    # Release/reuse the original physical locations, as the scheduler would.
    blocks.zero_()
    states.zero_()
    assert cache.retrieve(list(range(9)), [4, 6, 8], 2) == 8
    torch.testing.assert_close(blocks[[4, 6]], expected_blocks)
    torch.testing.assert_close(states[2], expected_state)
    assert not blocks[8].any()
    assert not states[1].any()


def test_longest_prefix_and_full_hit_recomputes_last_token(
    backend: LocalCPUBackend,
) -> None:
    blocks = torch.full((10, 8), 11, dtype=torch.uint8)
    states = torch.full((4, 12), 22, dtype=torch.uint8)
    cache = DeepInferSnapshotCache(backend, blocks, states, 4, "model")
    assert cache.store(list(range(4)), [1], 1)
    states[1].fill_(33)
    assert cache.store(list(range(8)), [1, 2], 1)
    assert cache.retrieve(list(range(9)), [4, 5], 2) == 8
    assert states[2].tolist() == [33] * 12
    assert cache.retrieve(list(range(8)), [4, 5], 2) == 4
    assert states[2].tolist() == [22] * 12
    assert cache.retrieve(list(range(4)), [4], 2) == 0


def test_miss_and_namespace_isolation_do_not_modify_destination(
    backend: LocalCPUBackend,
) -> None:
    blocks = torch.ones((10, 8), dtype=torch.uint8)
    states = torch.ones((4, 12), dtype=torch.uint8)
    cache = DeepInferSnapshotCache(backend, blocks, states, 4, "revision-a")
    cache.store([1, 2, 3, 4], [1], 1)
    for namespace, worker_id, tokens in [
        ("revision-b", 0, [1, 2, 3, 4, 5]),
        ("revision-a", 1, [1, 2, 3, 4, 5]),
        ("revision-a", 0, [9, 2, 3, 4, 5]),
    ]:
        reader = DeepInferSnapshotCache(
            backend, blocks, states, 4, namespace, worker_id
        )
        assert reader.retrieve(tokens, [2], 2) == 0
        assert torch.all(blocks == 1)
        assert torch.all(states == 1)


@pytest.mark.parametrize("ids,state_id", [([0], 1), ([10], 1), ([], 1), ([1], -1)])
def test_invalid_mapping_rejected_before_writes(
    backend: LocalCPUBackend, ids: list[int], state_id: int
) -> None:
    blocks = torch.ones((10, 8), dtype=torch.uint8)
    states = torch.ones((4, 12), dtype=torch.uint8)
    cache = DeepInferSnapshotCache(backend, blocks, states, 4, "model")
    cache.store([1, 2, 3, 4], [1], 1)
    with pytest.raises(ValueError):
        cache.retrieve([1, 2, 3, 4, 5], ids, state_id)
    assert torch.all(blocks == 1)
    assert torch.all(states == 1)


def test_only_exact_boundaries_can_be_stored(backend: LocalCPUBackend) -> None:
    cache = DeepInferSnapshotCache(
        backend,
        torch.zeros((4, 8), dtype=torch.uint8),
        torch.zeros((4, 12), dtype=torch.uint8),
        4,
        "model",
    )
    for tokens in ([], [1, 2, 3], [1, 2, 3, 2**31]):
        with pytest.raises(ValueError):
            cache.store(tokens, [1], 1)
    with pytest.raises(ValueError):
        cache.store(list(range(8)), [1, 1], 1)


def test_eviction_and_oversized_snapshot(backend: LocalCPUBackend) -> None:
    blocks = torch.ones((10, 1024), dtype=torch.uint8)
    states = torch.ones((4, 12), dtype=torch.uint8)
    cache = DeepInferSnapshotCache(backend, blocks, states, 4, "model")
    assert cache.store([1] * 4, [1], 1)
    assert cache.store([2] * 4, [1], 1)
    assert cache.store([3] * 4, [1], 1)
    assert cache.retrieve([1] * 5, [2], 2) == 0
    assert cache.retrieve([3] * 5, [2], 2) == 4
    assert not cache.store(list(range(20)), [1, 2, 3, 4, 5], 1)
    assert cache.retrieve(list(range(21)), [1, 2, 3, 4, 5], 1) == 0


def test_no_state_layout(backend: LocalCPUBackend) -> None:
    blocks = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    cache = DeepInferSnapshotCache(
        backend, blocks, torch.empty((0, 0), dtype=torch.uint8), 4, "no-state"
    )
    expected = blocks[1].clone()
    assert cache.store([1, 2, 3, 4], [1], -1)
    assert cache.retrieve([1, 2, 3, 4, 5], [2], -1) == 4
    torch.testing.assert_close(blocks[2], expected)
