# SPDX-License-Identifier: Apache-2.0
"""Block ID slicing for streaming windows with different attention layouts."""

# Standard
from types import SimpleNamespace
from typing import cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.object_group_transfer import (
    slice_staged_block_ids_for_window,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext


def test_window_slices_downsampled_groups_independently() -> None:
    manager = SimpleNamespace(
        num_kernel_groups=2,
        get_subchunk_sw_size_tokens=lambda group: [128, 32][group],
    )
    context = cast(
        BaseCacheContext,
        SimpleNamespace(
            kv_layer_groups_manager=manager,
            lmcache_tokens_per_chunk=128,
            calculate_num_blocks=lambda tokens, group: tokens // 16,
        ),
    )
    host = [list(range(32)), list(range(100, 108))]
    gpu = [torch.tensor(group) for group in host]

    sliced_gpu, sliced_host = slice_staged_block_ids_for_window(
        context, gpu, host, start_chunk=1, num_chunks=2
    )

    assert sliced_host == [list(range(8, 24)), list(range(102, 106))]
    assert [group.tolist() for group in sliced_gpu] == sliced_host
    with pytest.raises(ValueError, match="exceeds"):
        slice_staged_block_ids_for_window(context, gpu, host, 3, 2)
