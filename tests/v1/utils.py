# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# First Party
import lmcache.lmcache_native as lmcache_native


def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size=16,
    dtype=torch.bfloat16,
    num_layers=32,
    head_size=128,
    # default vllm non-MLA flash attention
    engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
):
    """
    Instead of Tuple[Tuple[Tensor, Tensor]], return List[Tensor]
    where KV are in the same tensor
    """
    ret = []
    # only support vllm MLA format for now
    use_mla = engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
    num_heads = 1 if use_mla else 8
    if use_mla:
        shape = [num_blocks, block_size, head_size]
    else:
        if engine_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
            shape = [2, num_blocks, block_size, num_heads, head_size]
        elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
            shape = [num_blocks, 2, block_size, num_heads, head_size]
        elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
            shape = [2, num_blocks, num_heads, block_size, head_size]
        elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS:
            shape = [num_blocks, 2, num_heads, block_size, head_size]
        elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_NH_BS_TWO_HS:
            # blocks-first, K/V fused into the trailing dim
            shape = [num_blocks, num_heads, block_size, 2, head_size]
        else:
            raise ValueError(f"Unsupported engine_kv_format: {engine_kv_format}")

    for i in range(num_layers):
        # TODO(chunxiaozheng): support more dtypes
        if dtype == torch.uint8:
            kv = torch.randint(0, 256, shape, dtype=dtype, device=device)
        else:
            kv = torch.rand(shape, dtype=dtype, device=device)
        ret.append(kv)

    return ret


def generate_kv_cache_paged(num_blocks, device, block_size=16, dtype=torch.bfloat16):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_blocks, block_size, num_heads, head_size]

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def check_mem_obj_equal(left, right, use_mla: bool = False):
    """
    check whether two memory objects are the same
    """
    for left_mem_obj, right_mem_obj in zip(left, right, strict=False):
        left_tensor_size = left_mem_obj.tensor.size()
        right_tensor_size = right_mem_obj.tensor.size()
        if use_mla:
            assert left_tensor_size[0] == 1
            assert right_tensor_size[0] == 1

            left_kv, right_kv = left_mem_obj.tensor[0], right_mem_obj.tensor[0]
            right_kv = right_kv.to(left_kv.device)

            assert len(left_kv.shape) == 3
            assert len(right_kv.shape) == 3

            assert (left_kv[:, :, :] == right_kv[:, :, :]).all()
        else:
            assert left_tensor_size[0] == 2
            assert right_tensor_size[0] == 2

            left_kv, right_kv = left_mem_obj.tensor, right_mem_obj.tensor
            left_k, left_v = left_kv[0], left_kv[1]
            right_k, right_v = right_kv[0], right_kv[1]
            right_k = right_k.to(left_k.device)
            right_v = right_v.to(left_v.device)

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert (left_k[:, :, :] == right_k[:, :, :]).all()
            assert (left_v[:, :, :] == right_v[:, :, :]).all()


# default checks for vllm non-MLA flash attention


def check_paged_kv_cache_equal(
    left,
    right,
    slot_mapping,
    num_heads=8,
    head_size=128,
    engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
):
    """
    check whether two paged kv caches are the same at slot_mapping
    """

    if engine_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS:
        token_dim = 0
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = left_kv_layer[0].reshape(-1, num_heads, head_size)
            left_v = left_kv_layer[1].reshape(-1, num_heads, head_size)
            right_k = right_kv_layer[0].reshape(-1, num_heads, head_size)
            right_v = right_kv_layer[1].reshape(-1, num_heads, head_size)

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert left_k.shape[token_dim] >= num_tokens
            assert left_v.shape[token_dim] >= num_tokens
            assert right_k.shape[token_dim] >= num_tokens
            assert right_v.shape[token_dim] >= num_tokens

            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
        token_dim = 0
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = left_kv_layer[:, 0].contiguous().reshape(-1, num_heads, head_size)
            left_v = left_kv_layer[:, 1].contiguous().reshape(-1, num_heads, head_size)
            right_k = (
                right_kv_layer[:, 0].contiguous().reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[:, 1].contiguous().reshape(-1, num_heads, head_size)
            )

            assert len(left_k.shape) == 3
            assert len(left_v.shape) == 3
            assert len(right_k.shape) == 3
            assert len(right_v.shape) == 3

            assert left_k.shape[token_dim] >= num_tokens
            assert left_v.shape[token_dim] >= num_tokens
            assert right_k.shape[token_dim] >= num_tokens
            assert right_v.shape[token_dim] >= num_tokens

            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS:
        # HND flash attention: [2, num_blocks, num_heads, block_size, head_size]
        # Flatten [num_blocks, num_heads, block_size, head_size] ->
        #   swap to [num_blocks, block_size, num_heads, head_size] ->
        #   reshape to [num_blocks*block_size, num_heads, head_size]
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = (
                left_kv_layer[0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            left_v = (
                left_kv_layer[1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_k = (
                right_kv_layer[0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )

            assert left_k.shape[0] >= num_tokens
            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()

    elif engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS:
        # HND flash infer: [num_blocks, 2, num_heads, block_size, head_size]
        # left_kv_layer[:, 0] -> [num_blocks, num_heads, block_size, head_size]
        num_tokens = slot_mapping.shape[0]
        for left_kv_layer, right_kv_layer in zip(left, right, strict=False):
            left_k = (
                left_kv_layer[:, 0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            left_v = (
                left_kv_layer[:, 1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_k = (
                right_kv_layer[:, 0]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )
            right_v = (
                right_kv_layer[:, 1]
                .permute(0, 2, 1, 3)
                .contiguous()
                .reshape(-1, num_heads, head_size)
            )

            assert left_k.shape[0] >= num_tokens
            assert (left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
            assert (left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()
