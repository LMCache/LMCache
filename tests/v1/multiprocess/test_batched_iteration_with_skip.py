# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``batched_iteration_with_skip``."""

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    batched_iteration_with_skip,
)


def test_basic_batching_with_skip():
    """Skipped items are dropped and reported indices stay in original space."""
    data = list(range(10))
    result = list(batched_iteration_with_skip(data, batch_size=3, skip_count=2))

    assert result == [
        (2, (2, 3, 4)),
        (5, (5, 6, 7)),
        (8, (8, 9)),
    ]


def test_skip_count_zero_matches_plain_batching():
    """With skip_count=0 every item is yielded, indexed from 0."""
    data = list(range(7))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=0))

    assert result == [
        (0, (0, 1)),
        (2, (2, 3)),
        (4, (4, 5)),
        (6, (6,)),
    ]
    # The concatenation of all batches equals the unskipped tail of the list.
    flattened = [item for _, batch in result for item in batch]
    assert flattened == data


def test_batch_start_indices_are_original_indices():
    """Reported start index is the original list index, accounting for skip."""
    data = list(range(20))
    result = list(batched_iteration_with_skip(data, batch_size=5, skip_count=10))

    start_indices = [start for start, _ in result]
    assert start_indices == [10, 15]
    # The docstring example: skip_count=10, batch_size=5 -> first start idx 10.
    assert result[0] == (10, (10, 11, 12, 13, 14))


def test_partial_final_batch():
    """The final short batch still reports the correct start index."""
    data = list(range(8))
    result = list(batched_iteration_with_skip(data, batch_size=3, skip_count=1))

    assert result == [
        (1, (1, 2, 3)),
        (4, (4, 5, 6)),
        (7, (7,)),
    ]


def test_skip_equal_to_length_yields_nothing():
    """Skipping the entire list yields no batches."""
    data = list(range(5))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=5))
    assert result == []


def test_skip_larger_than_length_yields_nothing():
    """Skipping past the end of the list yields no batches and does not raise."""
    data = list(range(5))
    result = list(batched_iteration_with_skip(data, batch_size=2, skip_count=100))
    assert result == []


def test_empty_list():
    """An empty input yields no batches regardless of skip_count."""
    assert list(batched_iteration_with_skip([], batch_size=4, skip_count=0)) == []
    assert list(batched_iteration_with_skip([], batch_size=4, skip_count=3)) == []


def test_batch_size_larger_than_remaining():
    """A batch_size exceeding the remaining items yields one full-remainder batch."""
    data = list(range(6))
    result = list(batched_iteration_with_skip(data, batch_size=100, skip_count=2))
    assert result == [(2, (2, 3, 4, 5))]


@pytest.mark.parametrize("batch_size", [0, -1, -10])
def test_invalid_batch_size_raises(batch_size):
    """A batch_size below 1 raises ValueError."""
    with pytest.raises(ValueError, match="batch size must be at least one"):
        list(batched_iteration_with_skip([1, 2, 3], batch_size, skip_count=0))


@pytest.mark.parametrize("skip_count", [-1, -5])
def test_negative_skip_count_raises(skip_count):
    """A negative skip_count raises ValueError."""
    with pytest.raises(ValueError, match="skip_count must be non-negative"):
        list(
            batched_iteration_with_skip([1, 2, 3], batch_size=2, skip_count=skip_count)
        )


def test_returns_tuples_not_lists():
    """Each yielded batch is a tuple, mirroring batched_iteration."""
    _, batch = next(
        batched_iteration_with_skip([1, 2, 3, 4], batch_size=2, skip_count=0)
    )
    assert isinstance(batch, tuple)


# ---------------------------------------------------------------------------
# Retrieve scatter lanes: lane staging plumbed through the transfer helpers.
# ---------------------------------------------------------------------------


def test_downsample_and_stage_forwards_lane_buffer():
    """downsample_and_stage_block_ids passes the lane's private buffer to
    stage_block_ids; None (lane 0) keeps the stock call."""
    # Standard
    from unittest.mock import MagicMock

    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
        downsample_and_stage_block_ids,
    )

    ctx = MagicMock()
    ctx.kv_layer_groups_manager.num_kernel_groups = 1
    ctx.kv_layer_groups_manager.get_subchunk_sw_size_tokens.return_value = 128
    ctx.lmcache_tokens_per_chunk = 128
    ctx.calculate_num_blocks.return_value = 2

    lane_buf = torch.zeros(8, dtype=torch.long)
    downsample_and_stage_block_ids(ctx, [[1, 2]], out=lane_buf)
    assert ctx.stage_block_ids.call_args.kwargs["out"] is lane_buf

    downsample_and_stage_block_ids(ctx, [[1, 2]])
    assert ctx.stage_block_ids.call_args.kwargs["out"] is None


def test_transfer_fallback_uses_lane_buffers():
    """The Python fallback of transfer_kv_per_object_group stages H2D into
    the provided lane buffers (not the context's); buffers=None (lane 0) is
    the stock context-buffer path."""
    # Standard
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    # Third Party
    import torch

    # First Party
    from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod

    ctx = MagicMock()
    ctx.lmcache_tokens_per_chunk = 4
    ctx.kv_layer_groups_manager.object_groups = [
        SimpleNamespace(kernel_group_indices=[0])
    ]
    attn = ctx.kv_layer_groups_manager.get_attn_desc.return_value
    attn.is_full_attention.return_value = True
    ctx.calculate_num_blocks.return_value = 1
    ctx.kv_layer_groups_manager.get_subchunk_sw_size_tokens.return_value = 4

    lane_bufs = MagicMock()
    memory_objs = [MagicMock()]
    block_ids_gpu = [torch.tensor([0], dtype=torch.long)]

    with (
        patch.object(mod, "_HAS_NATIVE_OBJECT_GROUP_TRANSFER", False),
        patch.object(mod, "lmcache_memcpy_async_h2d") as h2d,
        patch.object(mod, "lmc_ops") as ops,
    ):
        ops.TransferDirection.H2D = "H2D"
        mod.transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            memory_objs,
            object_group_id=0,
            batch_size=2,
            skip_first_n_tokens=0,
            direction="H2D",
            buffers=lane_bufs,
        )
        # Staging + kernel temp buffers come from the lane, not the context.
        lane_bufs.get_temp_object_group_buffer.assert_called_once_with(0, 0)
        lane_bufs.get_temp_kernel_group_buffer.assert_called_once_with(0, 0)
        ctx.get_temp_object_group_buffer.assert_not_called()
        ctx.get_temp_kernel_group_buffer.assert_not_called()
        assert h2d.call_count == 1

        # Lane 0 (buffers=None): stock context buffers.
        mod.transfer_kv_per_object_group(
            ctx,
            block_ids_gpu,
            memory_objs,
            object_group_id=0,
            batch_size=2,
            skip_first_n_tokens=0,
            direction="H2D",
        )
        ctx.get_temp_object_group_buffer.assert_called_once_with(0, 0)
        ctx.get_temp_kernel_group_buffer.assert_called_once_with(0, 0)
