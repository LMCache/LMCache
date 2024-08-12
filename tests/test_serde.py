import pytest
import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_basics import CacheGenEncoderOutput
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer

#from lmcache.storage_backend.serde.safe_serde import SafeSerializer, SafeDeserializer

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

@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_encoder(chunk_size):
    fmt = "vllm"
    fmt2 = "huggingface"
    config = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    metadata = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt)
    metadata2 = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt2)
    serializer = CacheGenSerializer(config, metadata)
    serializer2 = CacheGenSerializer(config, metadata2)

    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    output = serializer.to_bytes(kv)
    kv2 = kv.permute([0, 1, 3, 2, 4])
    output2 = serializer2.to_bytes(kv2)

    assert abs(len(output) - len(output2)) < 10
    output_dict = CacheGenEncoderOutput.from_bytes(output)
    assert output_dict.num_heads == 8
    assert output_dict.head_size == 128

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_decoder(fmt, chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    metadata = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt)
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata)

    kv = to_blob(generate_kv_cache(chunk_size, fmt, "cuda"))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0

@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_cachegen_unmatched_size(fmt):
    chunk_size = 256
    fmt = "vllm"
    config = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
    metadata = LMCacheEngineMetadata(model_name = "mistralai/Mistral-7B-Instruct-v0.2", world_size = 1, worker_id = 0, fmt = fmt)
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata)

    kv = to_blob(generate_kv_cache(chunk_size - 20, fmt, "cuda"))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0
