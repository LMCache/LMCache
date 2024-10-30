import pytest
import torch

from lmcache.cache_engine import LMCacheEngine
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata


def dumb_metadata(fmt="vllm"):
    return LMCacheEngineMetadata("test_model", 3, 123, fmt, "half")


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


def generate_tokens(num_tokens, device):
    return torch.randint(0, 10000, size=[num_tokens]).to(device)


def get_tensor_size(tensor):
    num_elements = tensor.numel()
    element_size = tensor.element_size()
    size_in_bytes = num_elements * element_size
    size_in_gb = size_in_bytes / (1024**3)
    return size_in_gb


@pytest.mark.skip(
    reason="Temporarily disabling the evictor as stated in PR 185")
@pytest.mark.parametrize("dst_device", ["cuda:0"])
@pytest.mark.parametrize("backend", ["cuda", "cpu", "file://local_disk/"])
def test_evict(backend, dst_device, autorelease):
    fmt = "vllm"
    num_tokens = 256
    src_device = "cuda:0"
    """ initialize the engine """
    tokens_1 = generate_tokens(num_tokens, src_device)
    kv_cache_1 = generate_kv_cache(num_tokens, fmt, src_device)
    tokens_2 = generate_tokens(num_tokens, src_device)
    kv_cache_2 = generate_kv_cache(num_tokens, fmt, src_device)
    tokens_3 = generate_tokens(num_tokens, src_device)
    kv_cache_3 = generate_kv_cache(num_tokens, fmt, src_device)
    cfg = LMCacheEngineConfig.from_legacy(chunk_size=256, backend=backend)
    engine = autorelease(LMCacheEngine(cfg, dumb_metadata(fmt)))

    # can store upto two chunks
    max_token = 513
    max_size = get_tensor_size(kv_cache_1[0][0]) * 32 * 2 / 256 * max_token
    engine.engine_.evictor.MAX_CACHE_SIZE = max_size

    # store kv_cache_1 and kv_cache_2
    engine.store(tokens_1, kv_cache_1)
    engine.store(tokens_2, kv_cache_2)

    # retrieve (hit) kv_cache_1
    retrieved_cache, ret_mask = engine.retrieve(tokens_1)
    assert retrieved_cache[0][0].shape[0] == 256
    check_kv_cache_equal(retrieved_cache, kv_cache_1, num_tokens, fmt)

    # store kv_cache_3 -> kv_cache_2 should be evicted
    engine.store(tokens_3, kv_cache_3)

    # retrieve kv_cache_1, should be in cache
    retrieved_cache, ret_mask = engine.retrieve(tokens_1)
    assert retrieved_cache[0][0].shape[0] == 256
    check_kv_cache_equal(retrieved_cache, kv_cache_1, num_tokens, fmt)

    # retrieve kv_cache_2, should be evicted
    retrieved_cache, ret_mask = engine.retrieve(tokens_2)
    assert retrieved_cache == ()

    # retrieve kv_cache_3, should be in cache
    retrieved_cache, ret_mask = engine.retrieve(tokens_3)
    assert retrieved_cache[0][0].shape[0] == 256
    check_kv_cache_equal(retrieved_cache, kv_cache_3, num_tokens, fmt)
