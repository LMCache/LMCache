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


def _blocks(start: int, count: int, groups: int = 2) -> tuple[tuple[int, ...], ...]:
    return (tuple(range(start, start + count)),) * groups


@pytest.fixture
def buffers() -> KVCacheBuffers:
    """Allocate two multi-plane tensors and one single-plane tensor on CPU."""
    return KVCacheBuffers(
        parse_kvcache_shape_spec("(2,6,2,2,4):float32:2;(1,6,2,1,8):bfloat16:1"),
        device="cpu",
    )


def test_registration_order_and_groups(buffers: KVCacheBuffers) -> None:
    assert list(buffers.kv_caches) == ["tensor.0", "tensor.1", "tensor.2"]
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
    assert buffers.kv_caches["tensor.0"].shape == (2, 6, 0, 1, 8)
    assert buffers.engine_group_infos[0].tokens_per_block == 2
    assert buffers.checksum(_blocks(0, 0, 1)) == hashlib.md5(b"").hexdigest()


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
    for view in buffers.block_views(_blocks(2, 2)):
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
    buffers.set_blocks_to_zero(_blocks(1, 3))
    for index, tensor in enumerate(buffers.kv_caches.values()):
        expected = originals[index]
        if index < 2:
            expected[:, 1:4] = 0
        else:
            expected[1:4] = 0
        assert torch.equal(tensor, expected)


@pytest.mark.parametrize(
    "blocks",
    [
        *(_blocks(1, count) for count in (1, 3, 4)),
        ((4, 1), (0, 5)),
    ],
)
def test_checksum_follows_logical_block_and_tensor_order(
    buffers: KVCacheBuffers,
    blocks: tuple[tuple[int, ...], ...],
) -> None:
    selected = [
        buffers.kv_caches[name][:, list(blocks[0])] for name in ("tensor.0", "tensor.1")
    ]
    selected.append(buffers.kv_caches["tensor.2"][list(blocks[1])].float())
    expected = hashlib.md5(
        b"".join(
            value.contiguous().view(torch.uint8).numpy().tobytes() for value in selected
        )
    ).hexdigest()
    assert buffers.checksum(blocks) == expected
    if len(blocks[0]) > 1:
        assert buffers.checksum(tuple(ids[::-1] for ids in blocks)) != expected
    outside = tuple(tuple(set(range(6)) - set(ids)) for ids in blocks)
    buffers.set_blocks_to_zero(outside)
    assert buffers.checksum(blocks) == expected
    buffers.set_blocks_to_zero(tuple(ids[:1] for ids in blocks))
    assert buffers.checksum(blocks) != expected


def test_empty_selection_hashes_empty_bytes(buffers: KVCacheBuffers) -> None:
    before = buffers.checksum(_blocks(0, 6))
    buffers.set_blocks_to_zero(_blocks(6, 0))
    assert buffers.checksum(_blocks(6, 0)) == hashlib.md5(b"").hexdigest()
    assert buffers.checksum(_blocks(0, 6)) == before


def test_block_views_cannot_exceed_allocation(buffers: KVCacheBuffers) -> None:
    before = buffers.checksum(_blocks(0, 6))
    with pytest.raises(RuntimeError):
        list(buffers.block_views(_blocks(5, 2)))
    assert buffers.checksum(_blocks(0, 6)) == before


def test_workers_have_independent_storage() -> None:
    groups = parse_kvcache_shape_spec("(2,3,2,1,4):float32:1")
    first = KVCacheBuffers(groups, device="cpu")
    second = KVCacheBuffers(groups, device="cpu")
    expected = second.checksum(_blocks(0, 3, 1))
    first.set_blocks_to_zero(_blocks(0, 3, 1))
    assert first.checksum(_blocks(0, 3, 1)) != expected
    assert second.checksum(_blocks(0, 3, 1)) == expected
    first.release()
    assert second.checksum(_blocks(0, 3, 1)) == expected


def test_release_drops_ownership_and_preserves_external_views(
    buffers: KVCacheBuffers,
) -> None:
    refs = [weakref.ref(tensor) for tensor in buffers.kv_caches.values()]
    view = next(buffers.block_views(_blocks(0, 1)))
    expected = view.clone()
    buffers.release()
    buffers.release()
    assert buffers.kv_caches == {}
    assert torch.equal(view, expected)
    del view
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "uint8"])
def test_fill_depends_on_logical_identity_not_physical_location(dtype: str) -> None:
    buffers = KVCacheBuffers(
        parse_kvcache_shape_spec(f"(2,8,2,2,4):{dtype}:2;(1,8,2,1,8):{dtype}:1"),
        device="cpu",
    )
    source = ((2, 0), (1, 4))
    target = ((7, 5), (6, 3))
    tokens = (10, 11, 12, 13)
    outside = ((1, 3, 4, 5, 6, 7), (0, 2, 3, 5, 6, 7))
    untouched = [view.clone() for view in buffers.block_views(outside)]
    buffers.fill_deterministic_kv(source, tokens, worker_id=0)
    torch.testing.assert_close(
        list(buffers.block_views(outside)), untouched, rtol=0, atol=0
    )
    expected = buffers.checksum(source)
    buffers.fill_deterministic_kv(target, tokens, worker_id=0)
    assert buffers.checksum(target) == expected
    buffers.fill_deterministic_kv(target, tokens, worker_id=1)
    assert buffers.checksum(target) != expected
    buffers.fill_deterministic_kv(target, (10, 11, 12, 14), worker_id=0)
    assert buffers.checksum(target) != expected


def test_fill_uses_registered_tensor_identity_across_traversal_orders() -> None:
    """Reordering tensor traversal preserves each registered tensor's contents."""
    groups = parse_kvcache_shape_spec("(2,4,2,1,4):float32:2")
    original = KVCacheBuffers(groups, device="cpu")
    groups[0].layer_indices.reverse()
    reordered = KVCacheBuffers(groups, device="cpu")
    original.fill_deterministic_kv(((0, 1),), (10, 11, 12, 13), worker_id=0)
    reordered.fill_deterministic_kv(((3, 2),), (10, 11, 12, 13), worker_id=0)
    for name in ("tensor.0", "tensor.1"):
        torch.testing.assert_close(
            original.kv_caches[name][:, [0, 1]],
            reordered.kv_caches[name][:, [3, 2]],
            rtol=0,
            atol=0,
        )
