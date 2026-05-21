# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.gpu_context import PagedGPUCacheBlendContext


def _uninitialized_context() -> PagedGPUCacheBlendContext:
    return PagedGPUCacheBlendContext.__new__(PagedGPUCacheBlendContext)


def test_paged_cacheblend_accepts_vllm_block_kv_slot_layout():
    """Live vLLM V1 paged KV can arrive as [B,2,S,H,D]."""
    ctx = _uninitialized_context()
    tensor = torch.arange(3 * 2 * 4 * 2 * 5).reshape(3, 2, 4, 2, 5)

    view = ctx._layer_token_view(tensor)

    assert tuple(view.shape) == (2, 12, 10)
    assert torch.equal(view[0, 0], tensor[0, 0, 0].reshape(-1))
    assert torch.equal(view[1, 7], tensor[1, 1, 3].reshape(-1))


def test_paged_cacheblend_writable_slice_updates_vllm_block_kv_slot_layout():
    """Writable CB slices must mutate [B,2,S,H,D], not a reshaped copy."""
    ctx = _uninitialized_context()
    tensor = torch.zeros(3, 2, 4, 2, 5)
    ctx._kv_caches = [tensor]
    ctx._num_layers = 1
    ctx._num_tokens = 12
    ctx._hidden_dim_size = 10

    src = torch.arange(2 * 1 * 6 * 10, dtype=tensor.dtype).reshape(2, 1, 6, 10)

    ctx.writable_slice_on_tokens(3, 9).copy_(src)

    # Token 3 is block 0, offset 3.
    assert torch.equal(tensor[0, 0, 3], src[0, 0, 0].reshape(2, 5))
    assert torch.equal(tensor[0, 1, 3], src[1, 0, 0].reshape(2, 5))
    # Token 4 crosses into block 1, offset 0.
    assert torch.equal(tensor[1, 0, 0], src[0, 0, 1].reshape(2, 5))
    assert torch.equal(tensor[1, 1, 0], src[1, 0, 1].reshape(2, 5))
    # Token 8 is block 2, offset 0.
    assert torch.equal(tensor[2, 0, 0], src[0, 0, 5].reshape(2, 5))
    assert torch.equal(tensor[2, 1, 0], src[1, 0, 5].reshape(2, 5))
    # Neighboring positions outside the slice remain untouched.
    assert torch.count_nonzero(tensor[0, :, :3]) == 0
    assert torch.count_nonzero(tensor[2, :, 1:]) == 0
