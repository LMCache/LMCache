# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_kv import PagedKVSequenceView

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

KV_CASES = [
    pytest.param("cpu", torch.float32, id="cpu-fp32"),
    pytest.param("cuda", torch.float16, marks=CUDA, id="cuda-fp16"),
    pytest.param("cuda", torch.bfloat16, marks=CUDA, id="cuda-bf16"),
]


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_gathers_keys_and_values_in_request_order(device, dtype):
    # [blocks, K/V, block_size, kv_heads, head_dim]
    cache = torch.arange(5 * 2 * 4 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(5, 2, 4, 2, 3).to(device=device, dtype=dtype)
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)

    view = PagedKVSequenceView(cache, slots)
    for kv_index, actual in ((0, view.get_keys()), (1, view.get_values())):
        expected = original[:, kv_index][slots // 4, slots % 4]
        expected = expected.contiguous()
        assert actual.shape == (8, 2, 3)
        assert actual.device == cache.device
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)

    # Independent spot checks for block order, KV head, and K/V index.
    assert view.get_keys()[0, 0, 0].item() == 192
    assert view.get_keys()[4, 1, 2].item() == 53
    assert view.get_values()[7, 1, 2].item() == 95
    assert torch.equal(cache, original)


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_gathers_from_strided_cache_into_contiguous_output(device, dtype):
    cache = torch.arange(5 * 2 * 4 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(5, 2, 4, 2, 3).to(device=device, dtype=dtype)
    cache = torch.stack((cache, torch.zeros_like(cache)), dim=-1)[..., 0]
    assert not cache.is_contiguous()
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)

    view = PagedKVSequenceView(cache, slots)
    for kv_index, actual in ((0, view.get_keys()), (1, view.get_values())):
        expected = original[:, kv_index][slots // 4, slots % 4]
        expected = expected.contiguous()
        assert actual.shape == (8, 2, 3)
        assert actual.device == cache.device
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)
    assert torch.equal(cache, original)


def test_repeated_reads_reflect_cache_updates():
    cache = torch.zeros(5, 2, 4, 2, 3)
    view = PagedKVSequenceView(cache, torch.tensor([16, 4]))
    old_keys = view.get_keys()
    old_values = view.get_values()

    cache[4, 0, 0, 0, 0] = -1
    cache[4, 1, 0, 0, 0] = -2

    assert old_keys[0, 0, 0].item() == 0
    assert old_values[0, 0, 0].item() == 0
    assert view.get_keys()[0, 0, 0].item() == -1
    assert view.get_values()[0, 0, 0].item() == -2
