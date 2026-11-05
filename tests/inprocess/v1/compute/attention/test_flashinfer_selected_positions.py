# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.v1.compute.attention.metadata import LMCFlashInferSparseMetadata


@pytest.mark.parametrize("selected_count", [32, 64, 100])
@pytest.mark.parametrize("stride", [1, 2])
def test_planning_preserves_selected_positions(
    selected_count: int, stride: int
) -> None:
    """Planning must preserve the token positions reused for cached KV writes."""
    wrapper = Mock(spec=["plan"])
    metadata = LMCFlashInferSparseMetadata(
        wrapper=wrapper,
        seq_len=256,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=64,
        block_col_sizes=torch.full((8,), 32, dtype=torch.int64),
    )
    storage = torch.arange(3, 3 + selected_count * stride)
    positions = storage[::stride]
    original_storage = storage.clone()
    original_positions = positions.clone()

    metadata.update_from_top_indices(positions)

    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)
    torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)
    wrapper.plan.assert_called_once()
    first_mask = wrapper.plan.call_args.args[0].clone()

    metadata.update_from_top_indices(positions)

    assert wrapper.plan.call_count == 2
    assert torch.equal(wrapper.plan.call_args.args[0], first_mask)
    torch.testing.assert_close(storage, original_storage, rtol=0, atol=0)
