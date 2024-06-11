import pytest
import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer

def generate_kv_cache(num_tokens, fmt, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size] if fmt == "vllm" else [num_heads, num_tokens, head_size]
    dtype = torch.bfloat16 if fmt == "vllm" else torch.float16

    for i in range(32):
        k = torch.rand(shape, dtype = dtype, device = device)
        v = torch.rand(shape, dtype = dtype, device = device)
        ret.append((k, v))

    return tuple(ret)

def to_blob(kv_tuples):
    return torch.stack([torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0)

def test_encoder():
    fmt = "vllm"
    fmt2 = "huggingface"
    config = LMCacheEngineConfig.from_defaults()
    metadata = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt)
    metadata2 = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt2)
    serializer = CacheGenSerializer(config, metadata)
    serializer2 = CacheGenSerializer(config, metadata2)

    kv = to_blob(generate_kv_cache(256, fmt, "cuda"))
    output = serializer.to_bytes(kv)
    kv2 = kv.permute([0, 1, 3, 2, 4])
    output2 = serializer2.to_bytes(kv2)

    assert len(output) == len(output2)
