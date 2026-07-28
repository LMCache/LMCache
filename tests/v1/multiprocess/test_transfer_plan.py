# SPDX-License-Identifier: Apache-2.0
"""Unit tests for path-agnostic multiprocess transfer-plan helpers."""

# Standard
from dataclasses import dataclass, replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.multiprocess.transfer_plan import (
    batched_iteration_with_skip,
    build_kernel_group_layout,
    build_object_group_layout_desc,
    compute_num_objects_to_skip,
    downsample_block_ids,
    export_kv_transfer_metadata,
    has_sufficient_block_ids,
    recalculate_blocks_to_skip,
    select_block_ids_for_window,
)


@dataclass(frozen=True)
class _FakeShapeDesc:
    """Minimal shape descriptor test double for transfer metadata export."""

    kv_size: int
    """Number of KV planes in this fake group shape."""
    bs: int
    """Slots-per-block axis used by fake block-geometry math."""


@dataclass(frozen=True)
class _FakeKernelGroup:
    """Kernel-group metadata test double for transfer metadata export."""

    layer_indices: list[int]
    """Layer indices assigned to this fake kernel group."""
    shape_desc: _FakeShapeDesc
    """Shape descriptor used by fake block/layout calculations."""
    dtype: torch.dtype
    """Tensor dtype associated with this fake group."""
    engine_kv_format: object
    """Engine KV format enum value (or None in invalid tests)."""
    tokens_per_block: int
    """Logical tokens represented by one engine block in this group."""
    engine_group_idx: int
    """Engine group ID providing block IDs for this group."""
    num_layers: int
    """Layer count for this group."""
    hidden_dim_size: int
    """Hidden dimension width for this group."""
    slots_per_block: int
    """Physical slots represented by one engine block in this group."""


@dataclass(frozen=True)
class _FakeObjectGroup:
    """Object-group metadata test double for transfer metadata export."""

    kernel_group_indices: list[int]
    """Kernel-group indices packed into this fake object group."""


class _FakeManager:
    """Minimal KVLayerGroupsManager test double for metadata export."""

    def __init__(
        self,
        kernel_groups: list[_FakeKernelGroup],
        object_groups: list[_FakeObjectGroup],
        attn_desc: AttnWindowDesc,
        subchunk_tokens: dict[int, int],
    ) -> None:
        self.kernel_groups = kernel_groups
        self.object_groups = object_groups
        self._attn_desc = attn_desc
        self._subchunk_tokens = subchunk_tokens

    def get_attn_desc(self) -> AttnWindowDesc:
        return self._attn_desc

    def get_subchunk_sw_size_tokens(self, kernel_group_idx: int) -> int:
        return self._subchunk_tokens[kernel_group_idx]

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        group = self.kernel_groups[kernel_group_idx]
        sw_tokens = self._subchunk_tokens[kernel_group_idx]
        return sw_tokens * group.slots_per_block // group.tokens_per_block

    def calculate_num_blocks(self, kernel_group_idx: int, num_tokens: int) -> int:
        group = self.kernel_groups[kernel_group_idx]
        return (
            num_tokens
            * group.slots_per_block
            // group.tokens_per_block
            // group.shape_desc.bs
        )


def _fake_manager() -> _FakeManager:
    """Create a deterministic fake manager with two kernel/object groups."""

    # First Party
    import lmcache.c_ops as lmc_ops

    return _FakeManager(
        kernel_groups=[
            _FakeKernelGroup(
                layer_indices=[0, 2],
                shape_desc=_FakeShapeDesc(kv_size=2, bs=2),
                dtype=torch.float16,
                engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
                tokens_per_block=4,
                engine_group_idx=2,
                num_layers=2,
                hidden_dim_size=64,
                slots_per_block=2,
            ),
            _FakeKernelGroup(
                layer_indices=[1],
                shape_desc=_FakeShapeDesc(kv_size=1, bs=4),
                dtype=torch.bfloat16,
                engine_kv_format=lmc_ops.EngineKVFormat.NL_X_NB_BS_HS,
                tokens_per_block=8,
                engine_group_idx=7,
                num_layers=1,
                hidden_dim_size=128,
                slots_per_block=4,
            ),
        ],
        object_groups=[
            _FakeObjectGroup(kernel_group_indices=[1, 0]),
            _FakeObjectGroup(kernel_group_indices=[0]),
        ],
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[2, -1]),
        subchunk_tokens={0: 16, 1: 8},
    )


def test_has_sufficient_block_ids_variants() -> None:
    """Sufficient, insufficient, and extra block-ID coverage is detected."""
    blocks_per_chunk = [2, 3]
    num_chunks = 2

    assert has_sufficient_block_ids(
        [[1, 2, 3, 4], [9, 8, 7, 6, 5, 4]], blocks_per_chunk, num_chunks
    )
    assert not has_sufficient_block_ids(
        [[1, 2, 3], [9, 8, 7, 6, 5, 4]], blocks_per_chunk, num_chunks
    )
    assert has_sufficient_block_ids(
        [[1, 2, 3, 4, 10], [9, 8, 7, 6, 5, 4, 3]], blocks_per_chunk, num_chunks
    )
    assert has_sufficient_block_ids([[], []], blocks_per_chunk, num_chunks=0)


def test_has_sufficient_block_ids_group_mismatch_raises() -> None:
    """Mismatched group counts are rejected via strict zip semantics."""
    with pytest.raises(ValueError):
        has_sufficient_block_ids([[1, 2]], [2, 3], num_chunks=1)
    with pytest.raises(ValueError, match="at least one"):
        has_sufficient_block_ids([[1, 2]], [0], num_chunks=1)


@pytest.mark.parametrize(
    ("block_ids", "total_blocks_per_chunk", "keep_blocks_per_chunk", "expected"),
    [
        ([1, 2, 3, 4, 5, 6], 3, 3, [1, 2, 3, 4, 5, 6]),
        ([1, 2, 3, 4, 5, 6], 3, 2, [2, 3, 5, 6]),
    ],
)
def test_select_block_ids_for_window(
    block_ids: list[int],
    total_blocks_per_chunk: int,
    keep_blocks_per_chunk: int,
    expected: list[int],
) -> None:
    """Selection keeps either full windows or each chunk's trailing window."""
    original = list(block_ids)
    assert (
        select_block_ids_for_window(
            block_ids, total_blocks_per_chunk, keep_blocks_per_chunk
        )
        == expected
    )
    assert block_ids == original


def test_select_block_ids_for_window_invalid_geometry() -> None:
    """Invalid geometry raises ValueError."""
    with pytest.raises(ValueError, match="at least one"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=0, keep_blocks_per_chunk=1
        )
    with pytest.raises(ValueError, match="at least one"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=2, keep_blocks_per_chunk=0
        )
    with pytest.raises(ValueError, match="less than or equal"):
        select_block_ids_for_window(
            [1, 2], total_blocks_per_chunk=1, keep_blocks_per_chunk=2
        )


def test_select_block_ids_for_window_incomplete_chunks() -> None:
    """Incomplete chunk data raises ValueError."""
    with pytest.raises(ValueError, match="multiple"):
        select_block_ids_for_window(
            [1, 2, 3], total_blocks_per_chunk=2, keep_blocks_per_chunk=1
        )


def test_downsample_block_ids_multi_group_and_no_input_mutation() -> None:
    """Downsampling preserves group ordering and leaves input untouched."""
    source = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ]
    source_snapshot = [list(group) for group in source]

    downsampled = downsample_block_ids(
        source,
        blocks_per_chunk=[4, 2],
        blocks_per_window=[4, 1],
    )

    assert downsampled == [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [12, 14, 16, 18],
    ]
    assert source == source_snapshot


def test_downsample_block_ids_invalid_inputs_raise() -> None:
    """Invalid group counts and geometry raise ValueError."""
    with pytest.raises(ValueError, match="zip"):
        downsample_block_ids(
            [[1, 2, 3, 4]],
            blocks_per_chunk=[2],
            blocks_per_window=[1, 1],
        )
    with pytest.raises(ValueError, match="multiple"):
        downsample_block_ids([[1, 2, 3]], blocks_per_chunk=[2], blocks_per_window=[1])


def test_export_kv_transfer_metadata_preserves_order_and_geometry() -> None:
    """Export keeps deterministic kernel/object order and window geometry."""
    # Standard
    from typing import cast

    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    metadata = export_kv_transfer_metadata(
        cast(KVLayerGroupsManager, _fake_manager()),
        tokens_per_chunk=16,
    )

    assert metadata.tokens_per_chunk == 16
    assert metadata.num_chunks_in_sw == (2, -1)
    assert [group.kernel_group_id for group in metadata.kernel_groups] == [0, 1]
    assert [group.engine_group_id for group in metadata.kernel_groups] == [2, 7]
    assert metadata.kernel_groups[0].layer_indices == (0, 2)
    assert metadata.kernel_groups[0].blocks_per_chunk == 4
    assert metadata.kernel_groups[0].blocks_per_window == 4
    assert metadata.kernel_groups[1].blocks_per_chunk == 2
    assert metadata.kernel_groups[1].blocks_per_window == 1
    assert metadata.object_groups[0].kernel_group_ids == (1, 0)
    assert metadata.object_groups[0].sw_size_chunks == 2
    assert metadata.object_groups[1].kernel_group_ids == (0,)
    assert metadata.object_groups[1].sw_size_chunks == -1


def test_export_kv_transfer_metadata_windows_snapshot_is_immutable() -> None:
    """Exported window metadata is immutable and detached from manager state."""
    # Standard
    from typing import cast

    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    manager = _fake_manager()
    metadata = export_kv_transfer_metadata(
        cast(KVLayerGroupsManager, manager),
        tokens_per_chunk=16,
    )

    manager.get_attn_desc().num_chunks_in_sw[0] = 99
    assert metadata.num_chunks_in_sw == (2, -1)

    exported_attn_desc = metadata.build_attn_desc()
    exported_attn_desc.num_chunks_in_sw[0] = 88
    assert metadata.num_chunks_in_sw == (2, -1)


def test_build_object_group_layout_desc_preserves_kernel_group_order() -> None:
    """Object-group layout keeps kernel-group order, shapes, and dtypes."""
    # Standard
    from typing import cast

    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    metadata = export_kv_transfer_metadata(
        cast(KVLayerGroupsManager, _fake_manager()),
        tokens_per_chunk=16,
    )
    layout_desc = build_object_group_layout_desc(
        metadata, num_tokens=16, object_group_id=0
    )

    assert layout_desc.shapes == [
        torch.Size([1, 1, 8, 128]),
        torch.Size([2, 2, 8, 64]),
    ]
    assert layout_desc.dtypes == [torch.bfloat16, torch.float16]


def test_build_kernel_group_layout_invalid_alignment_raises() -> None:
    """Token counts must align to per-group compress ratio."""
    # Standard
    from typing import cast

    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    metadata = export_kv_transfer_metadata(
        cast(KVLayerGroupsManager, _fake_manager()),
        tokens_per_chunk=16,
    )
    with pytest.raises(ValueError, match="multiple of compress_ratio"):
        build_kernel_group_layout(metadata, num_tokens=3, kernel_group_id=0)


def test_export_kv_transfer_metadata_invalid_inputs_raise() -> None:
    """Invalid chunk size, IDs, formats, and attention metadata are rejected."""
    # Standard
    from typing import cast

    # First Party
    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    manager = _fake_manager()

    with pytest.raises(ValueError, match="at least one"):
        export_kv_transfer_metadata(
            cast(KVLayerGroupsManager, manager), tokens_per_chunk=0
        )

    manager_bad_format = _fake_manager()
    manager_bad_format.kernel_groups[0] = replace(
        manager_bad_format.kernel_groups[0],
        engine_kv_format=None,
    )
    with pytest.raises(ValueError, match="no engine_kv_format"):
        export_kv_transfer_metadata(
            cast(KVLayerGroupsManager, manager_bad_format),
            tokens_per_chunk=16,
        )

    manager_bad_ids = _fake_manager()
    manager_bad_ids.object_groups[0] = _FakeObjectGroup(kernel_group_indices=[3])
    with pytest.raises(ValueError, match="references invalid kernel group"):
        export_kv_transfer_metadata(
            cast(KVLayerGroupsManager, manager_bad_ids),
            tokens_per_chunk=16,
        )

    manager_bad_attn = _fake_manager()
    manager_bad_attn._attn_desc = AttnWindowDesc(num_chunks_in_sw=[-1])
    with pytest.raises(ValueError, match="does not match object-group count"):
        export_kv_transfer_metadata(
            cast(KVLayerGroupsManager, manager_bad_attn),
            tokens_per_chunk=16,
        )


def test_compute_num_objects_to_skip_cases() -> None:
    """Skip count covers store/full-attention/sliding-window retrieve cases."""
    assert (
        compute_num_objects_to_skip(sw_size_chunks=4, num_objects=7, is_retrieve=False)
        == 0
    )
    assert (
        compute_num_objects_to_skip(sw_size_chunks=-1, num_objects=7, is_retrieve=True)
        == 0
    )
    assert (
        compute_num_objects_to_skip(sw_size_chunks=3, num_objects=7, is_retrieve=True)
        == 4
    )
    assert (
        compute_num_objects_to_skip(sw_size_chunks=1, num_objects=7, is_retrieve=True)
        == 6
    )


@pytest.mark.parametrize("sw_size_chunks", [0, -2])
def test_compute_num_objects_to_skip_invalid_sw_size_chunks_raise(
    sw_size_chunks: int,
) -> None:
    """Invalid attention-window values are rejected."""
    with pytest.raises(ValueError, match="must be -1 \\(full\\) or at least one"):
        compute_num_objects_to_skip(
            sw_size_chunks=sw_size_chunks,
            num_objects=7,
            is_retrieve=True,
        )


def test_batched_iteration_with_skip_preserves_original_indices() -> None:
    """Batch start indices remain in original sequence coordinate space."""
    result = list(
        batched_iteration_with_skip(
            list(range(10)),
            batch_size=3,
            skip_count=4,
        )
    )
    assert result == [
        (4, (4, 5, 6)),
        (7, (7, 8, 9)),
    ]


def test_batched_iteration_with_skip_overlong_skip_yields_nothing() -> None:
    """Skipping past the tail exhausts iteration without yielding batches."""
    assert (
        list(
            batched_iteration_with_skip(
                list(range(10)),
                batch_size=3,
                skip_count=15,
            )
        )
        == []
    )


@pytest.mark.parametrize(
    ("batch_size", "skip_count", "message"),
    [
        (0, 0, "batch size must be at least one"),
        (1, -1, "skip_count must be non-negative"),
    ],
)
def test_batched_iteration_with_skip_invalid_arguments(
    batch_size: int, skip_count: int, message: str
) -> None:
    """Public argument validation raises ValueError."""
    with pytest.raises(ValueError, match=message):
        list(batched_iteration_with_skip([1, 2, 3], batch_size, skip_count))


def test_recalculate_blocks_to_skip_cases() -> None:
    """Identity, prefix drop, window retain, full chunk, and multi-chunk cases."""
    assert recalculate_blocks_to_skip(4, 4, 3) == 3
    assert recalculate_blocks_to_skip(4, 2, 1) == 0
    assert recalculate_blocks_to_skip(4, 2, 3) == 1
    assert recalculate_blocks_to_skip(4, 2, 4) == 2
    assert recalculate_blocks_to_skip(4, 2, 9) == 4


def test_recalculate_blocks_to_skip_invalid_inputs_raise() -> None:
    """Invalid geometry and negative skip values raise ValueError."""
    with pytest.raises(ValueError, match="less than or equal"):
        recalculate_blocks_to_skip(2, 3, 0)
    with pytest.raises(ValueError, match="non-negative"):
        recalculate_blocks_to_skip(4, 2, -1)
