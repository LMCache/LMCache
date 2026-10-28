# SPDX-License-Identifier: Apache-2.0
"""CPU contract experiment; no vLLM API or accelerator kernels are mocked."""

# Standard
from itertools import product

# Third Party
import pytest
import torch

# First Party
from examples.kv_cache_spec_slicing.spec_slicing import (
    PackedContentSpec,
    PackedHeadSpec,
    SliceSpec,
    load_views,
    store_views,
)
from lmcache.v1.memory_management import MemoryFormat


# Unique values make incorrect K/V, head, block, state, and layer selections visible.
def _values(heads: int = 4) -> torch.Tensor:
    return torch.arange(2 * 2 * 2 * heads * 4 * 3).view(2, 2, 2, heads, 4, 3)


def _register(values: torch.Tensor, packing: str) -> tuple[SliceSpec, torch.Tensor]:
    layers, blocks, _, heads, states, width = values.shape
    if packing == "heads":
        return PackedHeadSpec(heads), values.view(
            layers, blocks, 2 * heads, states, width
        )
    return PackedContentSpec(heads), values.permute(0, 1, 3, 4, 2, 5).contiguous().view(
        layers, blocks, heads, states, 2 * width
    )


def _components(spec: SliceSpec, tensor: torch.Tensor) -> list[torch.Tensor]:
    return spec.slice_for_tp_transfer(tensor, 1, 0, spec, 1, 0)


@pytest.mark.parametrize("packing", ["heads", "content"])
def test_spec_views_share_storage_and_preserve_components(packing: str) -> None:
    """The producer spec, not the consumer, determines K/V meaning."""
    expected = _values()
    spec, registered = _register(expected, packing)
    for index, view in enumerate(_components(spec, registered)):
        assert (
            view.untyped_storage().data_ptr() == registered.untyped_storage().data_ptr()
        )
        torch.testing.assert_close(view, expected[:, :, index])


@pytest.mark.parametrize(
    "source_packing,destination_packing", list(product(["heads", "content"], repeat=2))
)
@pytest.mark.parametrize("source_tp,destination_tp", [(1, 1), (1, 2), (2, 1), (2, 4)])
def test_tp_transfer_pairs_spec_views(
    source_packing: str,
    destination_packing: str,
    source_tp: int,
    destination_tp: int,
) -> None:
    """Separate K/V slices support different TP sizes and storage encodings."""
    expected = _values()
    source_heads = 4 // source_tp
    destination_heads = 4 // destination_tp
    destinations = []
    for rank in range(destination_tp):
        local = expected[
            :, :, :, rank * destination_heads : (rank + 1) * destination_heads
        ]
        spec, tensor = _register(torch.full_like(local, -1), destination_packing)
        destinations.append((spec, tensor))
    for source_rank in range(source_tp):
        local = expected[
            :, :, :, source_rank * source_heads : (source_rank + 1) * source_heads
        ]
        source_spec, source = _register(local.contiguous(), source_packing)
        for destination_rank, (destination_spec, destination) in enumerate(
            destinations
        ):
            source_views = source_spec.slice_for_tp_transfer(
                source,
                source_tp,
                source_rank,
                destination_spec,
                destination_tp,
                destination_rank,
            )
            destination_views = destination_spec.slice_for_tp_transfer(
                destination,
                destination_tp,
                destination_rank,
                source_spec,
                source_tp,
                source_rank,
            )
            for source_view, destination_view in zip(
                source_views, destination_views, strict=True
            ):
                assert source_view.shape == destination_view.shape
                destination_view.copy_(source_view)
    for rank, (spec, tensor) in enumerate(destinations):
        for index, view in enumerate(_components(spec, tensor)):
            torch.testing.assert_close(
                view,
                expected[
                    :,
                    :,
                    index,
                    rank * destination_heads : (rank + 1) * destination_heads,
                ],
            )


def test_naive_head_split_loses_matching_values() -> None:
    """Reproduce the thread's TP=1 -> TP=2 counterexample, with exact values."""
    spec, tensor = _register(_values(), "heads")
    naive = tensor[:, :, :4]
    correct = spec.slice_for_tp_transfer(tensor, 1, 0, PackedHeadSpec(2), 2, 0)
    assert not torch.equal(naive[:, :, 2:], correct[1])
    torch.testing.assert_close(correct[0], tensor[:, :, :2])
    torch.testing.assert_close(correct[1], tensor[:, :, 4:6])


@pytest.mark.parametrize(
    "source_packing,destination_packing", list(product(["heads", "content"], repeat=2))
)
@pytest.mark.parametrize("start,end", [(0, 8), (1, 7), (3, 6), (4, 5)])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float16])
def test_token_range_roundtrip_through_lmcache_memory(
    source_packing: str,
    destination_packing: str,
    start: int,
    end: int,
    dtype: torch.dtype,
) -> None:
    """Store/load crosses blocks, remaps pages, and preserves untouched states."""
    expected = _values().to(dtype=dtype)
    source_spec, source = _register(expected, source_packing)
    destination_spec, destination = _register(
        torch.full_like(expected, -1), destination_packing
    )
    source_views = _components(source_spec, source)
    destination_views = _components(destination_spec, destination)
    source_before = source.clone()
    memory = store_views(source_views, [1, 0], start, end)
    assert memory.get_memory_format() == MemoryFormat.KV_2LTD
    assert memory.get_num_tokens() == end - start
    assert memory.get_shape() == torch.Size((2, 2, end - start, 12))
    load_views(memory, destination_views, [0, 1], start, end)
    torch.testing.assert_close(source, source_before)
    for component, view in enumerate(destination_views):
        for token in range(8):
            actual = view[:, token // 4, :, token % 4]
            if start <= token < end:
                torch.testing.assert_close(
                    actual, expected[:, 1 - token // 4, component, :, token % 4]
                )
            else:
                assert torch.all(actual == -1)


@pytest.mark.parametrize("block_count", [2, 3])
def test_rbln_native_registration_copy_boundary(block_count: int) -> None:
    """BH packing aliases blocks-first; KV-first needs more than a view."""
    heads, states, width = 3, 4, 5
    blocks_first = torch.arange(block_count * 2 * heads * states * width).view(
        block_count, 2, heads, 1, states, width
    )
    packed = blocks_first.squeeze(3).view(block_count, 2 * heads, states, width)
    assert (
        packed.untyped_storage().data_ptr() == blocks_first.untyped_storage().data_ptr()
    )
    kv_first = torch.arange(blocks_first.numel()).view(
        2, block_count, heads, 1, states, width
    )
    reordered = kv_first.squeeze(3).permute(1, 0, 2, 3, 4)
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        reordered.view(block_count, 2 * heads, states, width)
    copied = reordered.reshape(block_count, 2 * heads, states, width)
    assert copied.untyped_storage().data_ptr() != kv_first.untyped_storage().data_ptr()
    for component, view in enumerate(_components(PackedHeadSpec(heads), packed[None])):
        torch.testing.assert_close(view[0], blocks_first[:, component, :, 0])
    for component, view in enumerate(_components(PackedHeadSpec(heads), copied[None])):
        torch.testing.assert_close(view[0], kv_first[component, :, :, 0])


def test_rbln_kv_first_component_views_are_zero_copy() -> None:
    """Separate components are feasible views, but their registration is unresolved."""
    native = torch.arange(2 * 2 * 3 * 4 * 5).view(2, 2, 3, 1, 4, 5)
    for component in range(2):
        view = native[component, :, :, 0][None]
        assert view.shape == (1, 2, 3, 4, 5)
        assert view.untyped_storage().data_ptr() == native.untyped_storage().data_ptr()
        torch.testing.assert_close(view[0], native[component, :, :, 0])


def test_rbln_kv_first_components_roundtrip_without_registration_repack() -> None:
    """Validate the component route without claiming upstream accepts that input."""
    source = torch.arange(2 * 2 * 3 * 4 * 5).view(2, 2, 3, 1, 4, 5)
    destination = torch.full_like(source, -1)
    source_views = [source[index, :, :, 0][None] for index in range(2)]
    destination_views = [destination[index, :, :, 0][None] for index in range(2)]
    memory = store_views(source_views, [1, 0], 1, 7)
    load_views(memory, destination_views, [0, 1], 1, 7)
    for token in range(8):
        actual = destination[:, token // 4, :, 0, token % 4]
        if 1 <= token < 7:
            torch.testing.assert_close(
                actual, source[:, 1 - token // 4, :, 0, token % 4]
            )
        else:
            assert torch.all(actual == -1)
