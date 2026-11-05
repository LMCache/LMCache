# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.compute.attention.metadata import LMCFlashInferSparseMetadata


@pytest.mark.parametrize(
    "selected_count, expected_row_sizes, visible_columns",
    [
        (1, [1], [2]),
        (15, [15], [2]),
        (31, [31], [2]),
        (32, [32], [2]),
        (33, [33], [2]),
        (64, [32, 32], [2, 3]),
        (100, [32, 32, 36], [2, 3, 4]),
    ],
)
def test_plan_covers_short_and_full_selections(
    selected_count: int,
    expected_row_sizes: list[int],
    visible_columns: list[int],
) -> None:
    """Plan every selected query, including a selection shorter than one block."""
    wrapper = Mock(spec=["plan"])
    metadata = LMCFlashInferSparseMetadata(
        wrapper=wrapper,
        seq_len=256,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        block_col_sizes=torch.full((8,), 32, dtype=torch.int64),
    )
    positions = torch.arange(64, 64 + selected_count)

    metadata.update_from_top_indices(positions)

    wrapper.plan.assert_called_once()
    mask, row_sizes, col_sizes, qo_heads, kv_heads, head_dim = (
        wrapper.plan.call_args.args
    )
    torch.testing.assert_close(
        row_sizes, torch.tensor(expected_row_sizes).expand(2, -1)
    )
    expected_mask = torch.tensor(
        [[column < count for column in range(8)] for count in visible_columns]
    )
    assert torch.equal(mask, expected_mask.expand(2, -1, -1))
    torch.testing.assert_close(col_sizes, torch.full((2, 8), 32, dtype=torch.int64))
    assert (qo_heads, kv_heads, head_dim) == (4, 2, 64)
    assert wrapper.plan.call_args.kwargs == {"q_data_type": torch.bfloat16}
