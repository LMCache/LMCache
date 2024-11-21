import torch
import pytest

from lmcache.experimental.gpu_connector import VLLMNestedTupleGPUConnector
from lmcache.experimental.memory_management import HostMemoryAllocator
from lmcache.experimental.memory_management import MemoryFormat

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

def check_kv_cache_equal(left, right, num_tokens, fmt):
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

        assert left_k.shape[dim] >= num_tokens
        assert left_v.shape[dim] >= num_tokens
        assert right_k.shape[dim] >= num_tokens
        assert right_v.shape[dim] >= num_tokens

        match fmt:
            case "huggingface":
                assert (left_k[:, :num_tokens, :] == right_k[:, :num_tokens, :]
                        ).all()
                assert (left_v[:, :num_tokens, :] == right_v[:, :num_tokens, :]
                        ).all()
            case "vllm":
                assert (left_k[:num_tokens, :, :] == right_k[:num_tokens, :, :]
                        ).all()
                assert (left_v[:num_tokens, :, :] == right_v[:num_tokens, :, :]
                        ).all()


def test_vllm_nested_gpu_connector():
    num_layers = 32
    num_heads = 8
    head_size = 128
    hidden_dim = num_heads * head_size
    connector = VLLMNestedTupleGPUConnector(hidden_dim, num_layers)
    allocator = HostMemoryAllocator(1024 * 1024 * 1024)

    assert connector.get_shape(10) == (2, num_layers, 10, hidden_dim)

    num_tokens = 512
    gpu_kv_src = generate_kv_cache(num_tokens, "vllm", "cuda")
    gpu_kv_dst = generate_kv_cache(num_tokens, "vllm", "cuda")

    # Check the gpu_kv is not the same before copying
    with pytest.raises(AssertionError):
        check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, 512, "vllm")

    slices = 4
    num_slice_tokens = num_tokens // slices
    for slice_id in range(slices):
        print("Here", slice_id)
        st, ed = slice_id * num_slice_tokens, (slice_id + 1) * num_slice_tokens
        shape = connector.get_shape(num_slice_tokens)
        memory_obj = allocator.allocate(shape, gpu_kv_src[0][0].dtype)
        connector.from_gpu(memory_obj, st, ed, kvcaches=gpu_kv_src)
        assert memory_obj.metadata.fmt == MemoryFormat.KV_BLOB
        connector.to_gpu(memory_obj, st, ed, kvcaches=gpu_kv_dst)
        allocator.free(memory_obj)
        assert allocator.memcheck()

    # Check gpu_kv becomes the same
    check_kv_cache_equal(gpu_kv_src, gpu_kv_dst, 512, "vllm")
