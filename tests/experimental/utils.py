import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.gpu_connector import (VLLMNestedTupleGPUConnector,
                                                VLLMPagedMemGPUConnector)


def dumb_metadata(fmt="vllm", kv_shape=(32, 2, 256, 8, 128)):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt, torch.bfloat16,
                                 kv_shape)


def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = ([num_tokens, num_heads, head_size]
             if fmt == "vllm" else [num_heads, num_tokens, head_size])
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def generate_kv_cache_paged(num_blocks,
                            device,
                            block_size=16,
                            dtype=torch.bfloat16):
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


def generate_tokens(num_tokens, device):
    return torch.randint(0, 10000, size=[num_tokens]).to(device)


def concatenate_kv_caches(kv_chunks, fmt):
    dim = 1 if fmt == "huggingface" else 0
    ret = []
    for kv_layer in zip(*kv_chunks):
        klist, vlist = zip(*kv_layer)
        klayer = torch.cat(klist, dim=dim)
        vlayer = torch.cat(vlist, dim=dim)
        ret.append((klayer, vlayer))
    return tuple(ret)


def check_kv_cache_equal(left, right, num_tokens, fmt, offset=0):
    """
    check if the first num_tokens of left and right kv cache are the same
    """
    dim = 0 if fmt == "vllm" else 1
    for left_kv, right_kv in zip(left, right):
        left_k, left_v = left_kv
        right_k, right_v = right_kv
        right_k = right_k.to(left_k.device)
        right_v = right_v.to(left_v.device)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        st = offset
        ed = offset + num_tokens
        assert left_k.shape[dim] >= ed
        assert left_v.shape[dim] >= ed
        assert right_k.shape[dim] >= ed
        assert right_v.shape[dim] >= ed

        match fmt:
            case "huggingface":
                assert (left_k[:, st:ed, :] == right_k[:, st:ed, :]).all()
                assert (left_v[:, st:ed, :] == right_v[:, offset:ed, :]).all()
            case "vllm":
                assert (left_k[st:ed, :, :] == right_k[st:ed, :, :]).all()
                assert (left_v[st:ed, :, :] == right_v[st:ed, :, :]).all()


def check_mem_obj_equal(left, right, offset=0):
    """
    check whether two memory objects are the same
    """
    for left_mem_obj, right_mem_obj in zip(left, right):
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


def check_paged_kv_cache_equal(left,
                               right,
                               num_tokens,
                               slot_mapping,
                               num_heads=8,
                               head_size=128):
    """
    check whether two paged kv caches are the same at slot_mapping
    """
    token_dim = 0
    for left_kv, right_kv in zip(left, right):
        left_k = left_kv[0].reshape(-1, num_heads, head_size)
        left_v = left_kv[1].reshape(-1, num_heads, head_size)
        right_k = right_kv[0].reshape(-1, num_heads, head_size)
        right_v = right_kv[1].reshape(-1, num_heads, head_size)

        assert len(left_k.shape) == 3
        assert len(left_v.shape) == 3
        assert len(right_k.shape) == 3
        assert len(right_v.shape) == 3

        assert left_k.shape[token_dim] >= num_tokens
        assert left_v.shape[token_dim] >= num_tokens
        assert right_k.shape[token_dim] >= num_tokens
        assert right_v.shape[token_dim] >= num_tokens

        assert (
            left_k[slot_mapping, :, :] == right_k[slot_mapping, :, :]).all()
        assert (
            left_v[slot_mapping, :, :] == right_v[slot_mapping, :, :]).all()


def check_kv_cache_device(kvs, device):
    for kv in kvs:
        k, v = kv
        assert k.device == torch.device(device)
        assert v.device == torch.device(device)


def create_gpu_connector(hidden_dim, num_layers, paged=False):
    if paged:
        return VLLMPagedMemGPUConnector(hidden_dim, num_layers)
    else:
        return VLLMNestedTupleGPUConnector(hidden_dim, num_layers)
