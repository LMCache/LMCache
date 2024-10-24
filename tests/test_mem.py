import random

import lmc_ops
import torch


# TODO(Jiayi): add more dtypes
def test_fast_mem_load():
    blk_size = 16
    num_blk = 1000
    num_tok = num_blk * blk_size
    num_load = 100
    num_head = 8
    head_size = 128
    kv_dtype = torch.bfloat16
    # [num_blk, blk_size, num_head, head_size]
    kv_shape = [num_blk, blk_size, num_head, head_size]
    slot_mapping = random.sample(range(0, num_tok + 1), num_load)
    slot_mapping = torch.tensor(slot_mapping, device='cuda')
    kv_cache = [
        torch.rand(kv_shape, dtype=kv_dtype, device='cuda'),
        torch.rand(kv_shape, dtype=kv_dtype, device='cuda')
    ]

    key_cuda, value_cuda = lmc_ops.load_and_reshape_flash(
        kv_cache[0], kv_cache[1], slot_mapping, "auto", 1.0, 1.0)

    key_cache = kv_cache[0].reshape(-1, num_head, head_size)
    value_cache = kv_cache[1].reshape(-1, num_head, head_size)
    key = key_cache[slot_mapping]
    value = value_cache[slot_mapping]

    assert torch.allclose(key, key_cuda, atol=1e-5)
    assert torch.allclose(value, value_cuda, atol=1e-5)
