# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.experimental.token_drop_kv import PagedKVSequenceView
from lmcache.v1.platform import torch_device_type

pytestmark = pytest.mark.skipif(torch_device_type != "cuda", reason="CUDA required")

KV_CASES = [
    pytest.param("cuda", torch.float16, id="cuda-fp16"),
    pytest.param("cuda", torch.bfloat16, id="cuda-bf16"),
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
        assert actual.shape == (8, 2, 3)
        assert actual.device == cache.device
        assert torch.equal(actual, expected)

    # Independent spot checks for block order, KV head, and K/V index.
    assert view.get_keys()[0, 0, 0].item() == 192
    assert view.get_keys()[4, 1, 2].item() == 53
    assert view.get_values()[7, 1, 2].item() == 95
    assert torch.equal(cache, original)


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_gathers_from_strided_cache(device, dtype):
    cache = torch.arange(5 * 2 * 4 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(5, 2, 4, 2, 3).to(device=device, dtype=dtype)
    cache = torch.stack((cache, torch.zeros_like(cache)), dim=-1)[..., 0]
    assert not cache.is_contiguous()
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)

    view = PagedKVSequenceView(cache, slots)
    for kv_index, actual in ((0, view.get_keys()), (1, view.get_values())):
        expected = original[:, kv_index][slots // 4, slots % 4]
        assert actual.shape == (8, 2, 3)
        assert actual.device == cache.device
        assert torch.equal(actual, expected)
    assert torch.equal(cache, original)


def test_repeated_reads_reflect_cache_updates():
    cache = torch.zeros(5, 2, 4, 2, 3, device="cuda")
    view = PagedKVSequenceView(cache, torch.tensor([16, 4], device="cuda"))
    old_keys = view.get_keys()
    old_values = view.get_values()

    cache[4, 0, 0, 0, 0] = -1
    cache[4, 1, 0, 0, 0] = -2

    assert old_keys[0, 0, 0].item() == 0
    assert old_values[0, 0, 0].item() == 0
    assert view.get_keys()[0, 0, 0].item() == -1
    assert view.get_values()[0, 0, 0].item() == -2


@pytest.mark.parametrize("device,dtype", KV_CASES)
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32], ids=["i64", "i32"])
def test_compacts_per_head_positions_without_mutating_other_blocks(
    device, dtype, index_dtype
):
    cache = torch.arange(5 * 2 * 4 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(5, 2, 4, 2, 3).to(device=device, dtype=dtype)
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)
    kept = torch.tensor(
        [[7, 0], [1, 4], [4, 7], [0, 2]], device=device, dtype=index_dtype
    )

    assert PagedKVSequenceView(cache, slots).compact_kv_in_place(kept) == 4
    for head in (0, 1):
        for dest, source in enumerate(kept[:, head].tolist()):
            dst_slot = slots[dest].item()
            src_slot = slots[source].item()
            for plane in (0, 1):
                assert torch.equal(
                    cache[dst_slot // 4, plane, dst_slot % 4, head],
                    original[src_slot // 4, plane, src_slot % 4, head],
                )
    for block in (0, 2, 3):
        assert torch.equal(cache[block], original[block])


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_repeated_compaction_uses_current_physical_sequence(device, dtype):
    cache = torch.arange(5 * 2 * 4 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(5, 2, 4, 2, 3).to(device=device, dtype=dtype)
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)
    first = torch.tensor([[1, 0], [3, 2], [4, 5], [7, 6]], device=device)
    second = torch.tensor([[1, 2], [3, 0]], device=device)

    assert PagedKVSequenceView(cache, slots).compact_kv_in_place(first) == 4
    assert PagedKVSequenceView(cache, slots[:4]).compact_kv_in_place(second) == 2
    for head in (0, 1):
        for dest, first_index in enumerate(second[:, head].tolist()):
            src_slot = slots[first[first_index, head]].item()
            dst_slot = slots[dest].item()
            for plane in (0, 1):
                assert torch.equal(
                    cache[dst_slot // 4, plane, dst_slot % 4, head],
                    original[src_slot // 4, plane, src_slot % 4, head],
                )


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_drop_first_across_shuffled_block_boundaries(device, dtype):
    cache = torch.arange(10 * 2 * 3 * 2 * 3, dtype=torch.float32)
    cache = cache.reshape(10, 2, 3, 2, 3).to(device=device, dtype=dtype)
    original = cache.clone()
    slots = torch.tensor(
        [block * 3 + offset for block in (6, 2, 9, 4) for offset in range(3)],
        device=device,
    )
    kept = torch.arange(1, len(slots), device=device)[:, None].expand(-1, 2)
    assert PagedKVSequenceView(cache, slots).compact_kv_in_place(kept) == 11
    for dest, source in enumerate(range(1, len(slots))):
        dst, src = slots[dest].item(), slots[source].item()
        assert torch.equal(cache[dst // 3, :, dst % 3], original[src // 3, :, src % 3])
    for block in (0, 1, 3, 5, 7, 8):
        assert torch.equal(cache[block], original[block])


@pytest.mark.parametrize("device,dtype", KV_CASES)
def test_compaction_updates_permuted_backing_storage(device, dtype):
    backing = torch.arange(5 * 4 * 2 * 2 * 3, dtype=torch.float32)
    backing = backing.reshape(5, 4, 2, 2, 3).to(device=device, dtype=dtype)
    cache = backing.permute(0, 2, 1, 3, 4)
    assert not cache.is_contiguous()
    original = cache.clone()
    slots = torch.tensor([16, 17, 18, 19, 4, 5, 6, 7], device=device)
    kept = torch.tensor([[7, 0], [1, 4], [4, 7], [0, 2]], device=device)

    assert PagedKVSequenceView(cache, slots).compact_kv_in_place(kept) == 4
    expected = original.clone()
    for head in range(2):
        for dest, src_pos in enumerate(kept[:, head].tolist()):
            dst_slot, src_slot = slots[dest].item(), slots[src_pos].item()
            expected[dst_slot // 4, :, dst_slot % 4, head] = original[
                src_slot // 4, :, src_slot % 4, head
            ]
    assert torch.equal(backing, expected.permute(0, 2, 1, 3, 4))


def test_invalid_positions_fail_before_mutation():
    cache = torch.randn(3, 2, 4, 2, 3, device="cuda", dtype=torch.float16)
    original = cache.clone()
    view = PagedKVSequenceView(cache, torch.tensor([8, 9, 10, 11, 0, 1], device="cuda"))
    with pytest.raises(ValueError, match="Expected"):
        view.compact_kv_in_place(torch.tensor([[0, 1, 3], [0, 1, 3]], device="cuda"))
    with pytest.raises(ValueError, match="Invalid"):
        view.compact_kv_in_place(torch.empty((0, 2), dtype=torch.int64, device="cuda"))
    with pytest.raises(ValueError, match="Invalid"):
        view.compact_kv_in_place(torch.zeros((7, 2), dtype=torch.int64, device="cuda"))
    assert torch.equal(cache, original)
