# SPDX-License-Identifier: Apache-2.0
"""Contracts for worker-local KV allocation, block access, and ownership."""

# Standard
import gc
import hashlib
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench.server_bench.buffers import KVCacheBuffers
from lmcache.v1.kv_layer_groups import parse_kvcache_shape_spec
from lmcache.v1.multiprocess.group_view import EngineGroupInfo


@pytest.fixture
def buffers() -> KVCacheBuffers:
    """Allocate two multi-plane layers and one single-plane layer on CPU."""
    return KVCacheBuffers(
        parse_kvcache_shape_spec("(2,6,2,2,4):float32:2;(1,6,2,1,8):bfloat16:1"),
        device="cpu",
    )


def test_registration_order_and_groups(buffers: KVCacheBuffers) -> None:
    assert list(buffers.kv_caches) == ["layer.0", "layer.1", "layer.2"]
    assert buffers.block_size == 2
    assert buffers.engine_group_infos == [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=2),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2,), tokens_per_block=2),
    ]


def test_registration_preserves_effective_cli_block_size() -> None:
    # The client resolves the legacy fallback; allocation still uses the spec.
    buffers = KVCacheBuffers(
        parse_kvcache_shape_spec("(2,6,0,1,8):float32:1"),
        device="cpu",
        block_size=2,
    )
    assert buffers.kv_caches["layer.0"].shape == (2, 6, 0, 1, 8)
    assert buffers.engine_group_infos[0].tokens_per_block == 2
    assert buffers.checksums(0, 0, chunk_size=4) == []


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "uint8"])
def test_seed_and_allocation_order_match_baseline(dtype: str) -> None:
    groups = parse_kvcache_shape_spec(f"(2,3,2,1,4):{dtype}:2;(1,3,2,1,8):{dtype}:1")
    buffers = KVCacheBuffers(groups, device="cpu")
    torch.random.manual_seed(42)
    expected = []
    for shape in [(2, 3, 2, 1, 4), (2, 3, 2, 1, 4), (3, 2, 8)]:
        if dtype == "uint8":
            expected.append(torch.randint(0, 256, shape, dtype=torch.uint8))
        else:
            expected.append(torch.randn(shape, dtype=getattr(torch, dtype)))
    for actual, reference in zip(buffers.kv_caches.values(), expected, strict=True):
        assert torch.equal(actual, reference)


def test_views_write_through_on_each_physical_block_axis(
    buffers: KVCacheBuffers,
) -> None:
    originals = [tensor.clone() for tensor in buffers.kv_caches.values()]
    for view in buffers.block_views(2, 2):
        view.fill_(7)
    for index, tensor in enumerate(buffers.kv_caches.values()):
        expected = originals[index]
        if index < 2:
            expected[:, 2:4] = 7
        else:
            expected[2:4] = 7
        assert torch.equal(tensor, expected)


def test_zero_only_changes_selected_blocks(buffers: KVCacheBuffers) -> None:
    originals = [tensor.clone() for tensor in buffers.kv_caches.values()]
    buffers.zero_blocks(1, 3)
    for index, tensor in enumerate(buffers.kv_caches.values()):
        expected = originals[index]
        if index < 2:
            expected[:, 1:4] = 0
        else:
            expected[1:4] = 0
        assert torch.equal(tensor, expected)


def test_checksum_chunk_and_layer_order(buffers: KVCacheBuffers) -> None:
    # Include both physical block axes and bfloat16's byte representation.
    expected = []
    for start in (1, 3):
        digest = hashlib.md5()
        for name in ("layer.0", "layer.1"):
            selected = buffers.kv_caches[name][:, start : start + 2]
            digest.update(selected.contiguous().view(torch.uint8).numpy().tobytes())
        selected = buffers.kv_caches["layer.2"][start : start + 2]
        digest.update(selected.contiguous().view(torch.uint8).numpy().tobytes())
        expected.append(digest.hexdigest())
    assert buffers.checksums(1, 4, chunk_size=4) == expected
    # A block outside the requested range must not affect the digest.
    buffers.zero_blocks(0, 1)
    assert buffers.checksums(1, 4, chunk_size=4) == expected
    buffers.zero_blocks(1, 1)
    actual = buffers.checksums(1, 4, chunk_size=4)
    assert actual[0] != expected[0]
    assert actual[1] == expected[1]


def test_empty_range_and_incomplete_chunk(buffers: KVCacheBuffers) -> None:
    before = buffers.checksums(0, 6, chunk_size=4)
    buffers.zero_blocks(6, 0)
    assert buffers.checksums(6, 0, chunk_size=4) == []
    assert buffers.checksums(0, 1, chunk_size=4) == []
    assert buffers.checksums(0, 5, chunk_size=4) == before[:2]
    assert buffers.checksums(0, 6, chunk_size=4) == before


def test_checksum_requires_chunk_multiple(buffers: KVCacheBuffers) -> None:
    with pytest.raises(ValueError, match="multiple of block_size"):
        buffers.checksums(0, 2, chunk_size=3)


def test_block_range_cannot_exceed_allocation(buffers: KVCacheBuffers) -> None:
    before = buffers.checksums(0, 6, chunk_size=4)
    with pytest.raises(RuntimeError):
        buffers.zero_blocks(5, 2)
    assert buffers.checksums(0, 6, chunk_size=4) == before


def test_workers_have_independent_storage() -> None:
    groups = parse_kvcache_shape_spec("(2,3,2,1,4):float32:1")
    first = KVCacheBuffers(groups, device="cpu")
    second = KVCacheBuffers(groups, device="cpu")
    expected = second.checksums(0, 3, chunk_size=2)
    first.zero_blocks(0, 3)
    assert first.checksums(0, 3, chunk_size=2) != expected
    assert second.checksums(0, 3, chunk_size=2) == expected
    first.release()
    assert second.checksums(0, 3, chunk_size=2) == expected


def test_release_drops_ownership_and_preserves_external_views(
    buffers: KVCacheBuffers,
) -> None:
    refs = [weakref.ref(tensor) for tensor in buffers.kv_caches.values()]
    view = next(buffers.block_views(0, 1))
    expected = view.clone()
    buffers.release()
    buffers.release()
    assert buffers.kv_caches == {}
    assert torch.equal(view, expected)
    del view
    gc.collect()
    assert all(ref() is None for ref in refs)
