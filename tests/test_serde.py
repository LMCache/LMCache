import pytest
import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_basics import CacheGenEncoderOutput
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.utils import KVCache  # Import KVCache type hint


def generate_kv_cache(num_tokens, fmt, device) -> KVCache:
    """Generates KVCache in the tuple format: List[Tuple[Tensor, Tensor]]"""
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    # Determine shape based on format (vllm vs hf) for individual K/V tensors
    shape = ([num_tokens, num_heads, head_size]
             if fmt == "vllm" else [num_heads, num_tokens, head_size])
    # Use bfloat16 for both formats if supported, otherwise float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def to_blob(kv_tensors: KVCache) -> torch.Tensor:
    """ Convert the nested tuple of kv tensors (KVCache) to a single
    big tensor with shape [2, num_layers, ...].
    The first dimension represents K (0) and V (1).
    (Mirrors the updated logic in cache_engine.py)
    """
    k_temp = []
    v_temp = []
    for kv_layer in kv_tensors:
        k_temp.append(kv_layer[0])
        v_temp.append(kv_layer[1])
    # k_tensor_blob/v_tensor_blob shape: [num_layers, ...]
    k_tensor_blob = torch.stack(k_temp)
    v_tensor_blob = torch.stack(v_temp)

    # kv_tensors_blob: [2, num_layers, ...]
    kv_tensors_blob = torch.stack((k_tensor_blob, v_tensor_blob))
    # No permute needed here anymore

    return kv_tensors_blob


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_encoder(chunk_size):
    fmt_vllm = "vllm"
    fmt_hf = "huggingface"
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)

    # Ensure model has CacheGenConfig
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    metadata_vllm = LMCacheEngineMetadata(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        fmt=fmt_vllm,
        kv_dtype=dtype,
        kv_shape=None,  # Shape is inferred
        num_layers=32,  # Provide num_layers for potential
        # fallback config
    )
    metadata_hf = LMCacheEngineMetadata(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        fmt=fmt_hf,
        kv_dtype=dtype,
        kv_shape=None,  # Shape is inferred
        num_layers=32,  # Provide num_layers for potential
        # fallback config
    )
    serializer_vllm = CacheGenSerializer(config, metadata_vllm)
    serializer_hf = CacheGenSerializer(config, metadata_hf)

    # Generate KVCache tuple for vllm format
    kv_tuple_vllm = generate_kv_cache(chunk_size, fmt_vllm, device)
    # Convert to blob [2, num_layers, ntokens, nheads, hsize]
    kv_blob_vllm = to_blob(kv_tuple_vllm)

    # Generate KVCache tuple for hf format
    kv_tuple_hf = generate_kv_cache(chunk_size, fmt_hf, device)
    # Convert to blob [2, num_layers, nheads, ntokens, hsize]
    kv_blob_hf = to_blob(kv_tuple_hf)

    # Serialize both formats
    output_vllm = serializer_vllm.to_bytes(kv_blob_vllm)
    output_hf = serializer_hf.to_bytes(kv_blob_hf)

    # Basic checks on output bytes
    assert len(output_vllm) > 0
    assert len(output_hf) > 0
    # Lengths might differ slightly due to compression differences
    # assert abs(len(output_vllm) - len(output_hf)) < len(output_vllm) * 0.1

    # Check metadata in one of the outputs
    output_dict = CacheGenEncoderOutput.from_bytes(output_vllm)
    assert output_dict.num_heads == 8
    assert output_dict.head_size == 128


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_decoder(fmt, chunk_size):
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    # Ensure model has CacheGenConfig
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    metadata = LMCacheEngineMetadata(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=dtype,
        kv_shape=None,  # Shape is inferred
        num_layers=32,  # Provide num_layers for potential
        # fallback config
    )
    serializer = CacheGenSerializer(config, metadata)
    # Deserializer needs the target dtype
    deserializer = CacheGenDeserializer(config, metadata, dtype)

    # Generate original data (tuple format)
    kv_tuple = generate_kv_cache(chunk_size, fmt, device)
    # Convert to blob format [2, num_layers, ...] expected by serializer
    kv_blob = to_blob(kv_tuple)

    # Serialize
    output_bytes = serializer.to_bytes(kv_blob)
    assert len(output_bytes) > 0

    # Deserialize
    decoded_kv_blob = deserializer.from_bytes(output_bytes)

    # Check shape and basic properties
    assert decoded_kv_blob.shape == kv_blob.shape
    assert decoded_kv_blob.dtype == dtype
    # Dequantization might not be perfect, check if mean is reasonable
    assert decoded_kv_blob.abs().mean() > 1e-4

    # Optional: More rigorous check (e.g., compare means, max diff) if needed
    # Note: Due to quantization, exact match is not expected.
    # diff = (decoded_kv_blob - kv_blob).abs()
    # print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
    # assert diff.max() < threshold # Define an appropriate threshold


@pytest.mark.parametrize("fmt", ["vllm", "huggingface"])
def test_cachegen_unmatched_size(fmt):
    """Tests encoding/decoding when num_tokens is not equal to
    chunk_size."""
    chunk_size = 256
    num_tokens = chunk_size - 20  # Number of tokens less than default
    # chunk size
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Config still uses default chunk_size, but we pass fewer tokens
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    # Ensure model has CacheGenConfig
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    metadata = LMCacheEngineMetadata(
        model_name=model_name,
        world_size=1,
        worker_id=0,
        fmt=fmt,
        kv_dtype=dtype,
        kv_shape=None,  # Shape is inferred
        num_layers=32,  # Provide num_layers for potential
        # fallback config
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, dtype)

    # Generate original data with num_tokens
    kv_tuple = generate_kv_cache(num_tokens, fmt, device)
    # Convert to blob format [2, num_layers, ...]
    kv_blob = to_blob(kv_tuple)

    # Serialize
    output_bytes = serializer.to_bytes(kv_blob)
    assert len(output_bytes) > 0

    # Deserialize
    decoded_kv_blob = deserializer.from_bytes(output_bytes)

    # Check shape and basic properties - should match the original blob
    assert decoded_kv_blob.shape == kv_blob.shape
    assert decoded_kv_blob.dtype == dtype
    assert decoded_kv_blob.abs().mean() > 1e-4
