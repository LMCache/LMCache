# SPDX-License-Identifier: Apache-2.0
"""Property tests for the engine-neutral hybrid geometry reference model."""

# Standard
from math import lcm
from typing import Any, cast

# Third Party
from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import torch

# First Party
from benchmarks.microbenchmark.hybrid_geometry_reference import (
    GroupGeometry,
    reference_block_ids_for_token_range,
    reference_num_physical_slots,
    reference_windowed_block_ids,
)
from lmcache.v1.kv_layer_groups import KernelGroupInfo
from lmcache.v1.multiprocess.group_view import slice_block_ids_per_group
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    downsample_and_stage_block_ids,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

_BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@st.composite
def _aligned_geometry_cases(draw):
    tokens_per_block = draw(
        st.lists(
            st.sampled_from(_BLOCK_SIZES),
            min_size=1,
            max_size=6,
        )
    )
    alignment = lcm(*tokens_per_block)
    start = draw(st.integers(min_value=0, max_value=8)) * alignment
    end = start + draw(st.integers(min_value=0, max_value=8)) * alignment
    extra_blocks = draw(st.integers(min_value=0, max_value=4))
    allocated = {
        group_id: [
            group_id * 1_000_000 + offset
            for offset in range(end // block_size + extra_blocks)
        ]
        for group_id, block_size in enumerate(tokens_per_block)
    }
    return allocated, tokens_per_block, start, end


@st.composite
def _window_geometry_cases(draw):
    logical_chunk_tokens = 256
    tokens_per_block = draw(
        st.lists(
            st.sampled_from(_BLOCK_SIZES),
            min_size=1,
            max_size=6,
        )
    )
    windows = [
        draw(
            st.one_of(
                st.none(),
                st.sampled_from(
                    list(
                        range(
                            block_size,
                            logical_chunk_tokens + block_size,
                            block_size,
                        )
                    )
                ),
            )
        )
        for block_size in tokens_per_block
    ]
    num_chunks = draw(st.integers(min_value=0, max_value=16))
    geometries = [
        GroupGeometry(
            group_id=group_id,
            tokens_per_block=block_size,
            physical_slots_per_block=block_size,
            window_tokens=windows[group_id],
        )
        for group_id, block_size in enumerate(tokens_per_block)
    ]
    block_ids = {
        group_id: [
            group_id * 1_000_000 + offset
            for offset in range(num_chunks * logical_chunk_tokens // block_size)
        ]
        for group_id, block_size in enumerate(tokens_per_block)
    }
    return logical_chunk_tokens, geometries, block_ids


class _ReferenceManager:
    def __init__(self, geometries: list[GroupGeometry], chunk_tokens: int):
        self.geometries = geometries
        self.chunk_tokens = chunk_tokens
        self.num_kernel_groups = len(geometries)

    def get_subchunk_sw_size_tokens(self, group_id: int) -> int:
        window = self.geometries[group_id].window_tokens
        return self.chunk_tokens if window is None else window


class _ReferenceContext:
    def __init__(self, geometries: list[GroupGeometry], chunk_tokens: int):
        self.geometries = geometries
        self.lmcache_tokens_per_chunk = chunk_tokens
        self.kv_layer_groups_manager = _ReferenceManager(geometries, chunk_tokens)

    def calculate_num_blocks(self, num_tokens: int, group_id: int) -> int:
        return num_tokens // self.geometries[group_id].tokens_per_block

    def stage_block_ids(self, block_ids: list[list[int]]) -> list[list[int]]:
        return block_ids


@given(_aligned_geometry_cases())
@settings(max_examples=300, deadline=None, derandomize=True)
def test_reference_matches_runtime_group_slicing(case) -> None:
    """Runtime slicing matches an independently enumerated reference map."""
    allocated, tokens_per_block, start, end = case

    actual = slice_block_ids_per_group(
        allocated,
        tokens_per_block,
        start,
        end,
    )
    expected = reference_block_ids_for_token_range(
        allocated,
        [
            GroupGeometry(
                group_id=group_id,
                tokens_per_block=block_size,
                physical_slots_per_block=block_size,
            )
            for group_id, block_size in enumerate(tokens_per_block)
        ],
        start,
        end,
    )

    assert actual == [expected[group_id] for group_id in range(len(expected))]
    assert [len(ids) for ids in actual] == [
        (end - start) // block_size for block_size in tokens_per_block
    ]


@given(_window_geometry_cases())
@settings(max_examples=300, deadline=None, derandomize=True)
def test_reference_matches_runtime_window_downsampling(case) -> None:
    """Runtime SWA tail selection matches the independent reference model."""
    logical_chunk_tokens, geometries, block_ids = case
    context = _ReferenceContext(geometries, logical_chunk_tokens)

    actual = downsample_and_stage_block_ids(
        cast(Any, context),
        [list(block_ids[group_id]) for group_id in range(len(geometries))],
    )
    expected = reference_windowed_block_ids(
        block_ids,
        geometries,
        logical_chunk_tokens,
    )

    assert actual == [expected[group_id] for group_id in range(len(expected))]


@given(
    tokens_per_block=st.sampled_from(_BLOCK_SIZES),
    compression_ratio=st.sampled_from(_BLOCK_SIZES),
    block_count=st.integers(min_value=0, max_value=512),
)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_physical_slot_reference_respects_compression(
    tokens_per_block: int,
    compression_ratio: int,
    block_count: int,
) -> None:
    """Logical tokens map to the exact physical-slot count for each group."""
    physical_slots_per_block = max(1, tokens_per_block // compression_ratio)
    if tokens_per_block % physical_slots_per_block:
        return
    geometry = GroupGeometry(
        group_id=0,
        tokens_per_block=tokens_per_block,
        physical_slots_per_block=physical_slots_per_block,
    )
    logical_tokens = block_count * tokens_per_block
    shape_desc = PageBufferShapeDesc()
    shape_desc.bs = physical_slots_per_block
    runtime_group = KernelGroupInfo(
        layer_indices=[0],
        shape_desc=shape_desc,
        dtype=torch.float16,
        tokens_per_block=tokens_per_block,
    )

    expected = reference_num_physical_slots(logical_tokens, geometry)
    assert expected == block_count * physical_slots_per_block
    assert runtime_group.calculate_slots(logical_tokens) == expected


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-16, 16, "non-negative"),
        (32, 16, "not precede"),
        (1, 16, "align"),
        (0, 15, "align"),
    ],
)
def test_reference_rejects_invalid_token_ranges(
    start: int,
    end: int,
    message: str,
) -> None:
    geometry = GroupGeometry(0, tokens_per_block=16, physical_slots_per_block=8)
    with pytest.raises(ValueError, match=message):
        reference_block_ids_for_token_range(
            {0: [10, 11]},
            [geometry],
            start,
            end,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_id": -1, "tokens_per_block": 16, "physical_slots_per_block": 16},
        {"group_id": 0, "tokens_per_block": 0, "physical_slots_per_block": 16},
        {"group_id": 0, "tokens_per_block": 16, "physical_slots_per_block": 0},
        {"group_id": 0, "tokens_per_block": 16, "physical_slots_per_block": 6},
        {
            "group_id": 0,
            "tokens_per_block": 16,
            "physical_slots_per_block": 8,
            "window_tokens": 12,
        },
    ],
)
def test_reference_rejects_invalid_group_geometry(kwargs) -> None:
    with pytest.raises(ValueError):
        GroupGeometry(**kwargs)


def test_reference_rejects_duplicate_and_missing_groups() -> None:
    duplicate = [GroupGeometry(0, 16, 16), GroupGeometry(0, 16, 16)]
    missing = [GroupGeometry(0, 16, 16), GroupGeometry(2, 16, 16)]

    with pytest.raises(ValueError, match="unique"):
        reference_block_ids_for_token_range({0: [1]}, duplicate, 0, 16)
    with pytest.raises(ValueError, match="dense"):
        reference_block_ids_for_token_range({0: [1], 2: [2]}, missing, 0, 16)


def test_reference_rejects_short_and_malformed_block_tables() -> None:
    geometries = [GroupGeometry(0, 16, 8), GroupGeometry(1, 32, 32)]

    with pytest.raises(ValueError, match="missing block table"):
        reference_block_ids_for_token_range({0: [10, 11]}, geometries, 0, 32)
    with pytest.raises(ValueError, match="too short"):
        reference_block_ids_for_token_range({0: [10], 1: [20]}, geometries, 0, 32)
    with pytest.raises(ValueError, match="non-negative integers"):
        reference_block_ids_for_token_range({0: [10, -1], 1: [20]}, geometries, 0, 32)


def test_deepseek_style_mixed_geometry_reference() -> None:
    """One range maps independently across dense, SWA, and state groups."""
    geometries = [
        GroupGeometry(0, 256, 64),
        GroupGeometry(1, 64, 64, window_tokens=256),
        GroupGeometry(2, 4, 4, window_tokens=4),
    ]
    allocated = {
        0: [10, 11],
        1: list(range(100, 108)),
        2: list(range(1_000, 1_128)),
    }

    mapped = reference_block_ids_for_token_range(
        allocated,
        geometries,
        token_start=256,
        token_end=512,
    )

    assert mapped == {
        0: [11],
        1: [104, 105, 106, 107],
        2: list(range(1_064, 1_128)),
    }
    assert [reference_num_physical_slots(256, item) for item in geometries] == [
        64,
        256,
        256,
    ]
