# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache.config import LMCacheMetadata
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.v1.config import LMCacheEngineConfig


def generate_kv_cache(num_tokens, use_case, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = (
        [num_tokens, num_heads, head_size]
        if use_case == "vllm"
        else [num_heads, num_tokens, head_size]
    )
    dtype = torch.bfloat16 if use_case == "vllm" else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def to_blob(kv_tuples):
    return torch.stack(
        [torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0
    )


# @pytest.mark.parametrize("chunk_size", [64, 256, 768])
# @pytest.mark.parametrize("use_case", ["vllm", "huggingface"])
# def test_cachegen_encoder_bench(benchmark, chunk_size, use_case):
#    use_case = "vllm"
#    config = LMCacheEngineConfig.from_defaults(chunk_size = chunk_size)
#    metadata = LMCacheMetadata(
# model_name = "mistralai/Mistral-7B-Instruct-v0.2",
# world_size = 1, worker_id = 0, use_case = use_case)
#    serializer = CacheGenSerializer(config, metadata)
#
#    kv = to_blob(generate_kv_cache(chunk_size, use_case, "cuda"))
#
#    benchmark(serializer.to_bytes, kv)


@pytest.mark.benchmark(group="cachegen")
@pytest.mark.parametrize("use_case", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [64, 256, 768])
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO: Add non-CUDA implementation to CacheGenSerializer/Deserializer",
)
def test_cachegen_decoder_bench(benchmark, use_case, chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        worker_id=0,
        use_case=use_case,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)

    kv = to_blob(generate_kv_cache(chunk_size, use_case, "cuda"))
    output = serializer.to_bytes(kv)

    benchmark(deserializer.from_bytes, output)
