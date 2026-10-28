# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.legacy_serde.cachegen_basics import (
    CacheGenEncoderOutput,
)
from lmcache.inprocess.v1.storage_backend.legacy_serde.cachegen_decoder import (
    CacheGenDeserializer,
    do_dequantize,
)
from lmcache.inprocess.v1.storage_backend.legacy_serde.cachegen_encoder import (
    CacheGenSerializer,
    _split_kv,
    torch_quant_vectorized,
)

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


def generate_kv_cache(num_tokens, device):
    ret = []
    num_layers = 32
    num_heads = 8
    head_size = 128
    shape = [num_tokens, num_heads, head_size]
    dtype = torch.bfloat16

    for i in range(num_layers):
        k = torch.rand(shape, dtype=dtype, device=device)
        v = torch.rand(shape, dtype=dtype, device=device)
        ret.append((k, v))

    return tuple(ret)


def to_blob(kv_tuples):
    return torch.stack(
        [torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tuples], dim=0
    )


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_encoder(chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    # metadata2 = LMCacheMetadata(
    #     model_name="mistralai/Mistral-7B-Instruct-v0.2",
    #     world_size=1,
    #     local_world_size=1,
    #     worker_id=0,
    #     local_worker_id=0,
    #     kv_dtype=torch.bfloat16,
    #     kv_shape=None,
    # )
    serializer = CacheGenSerializer(config, metadata)
    # serializer2 = CacheGenSerializer(config, metadata2)

    kv = to_blob(generate_kv_cache(chunk_size, torch_device_type))
    output = serializer.to_bytes(kv)
    # huggingface:
    # kv2 = kv.permute([0, 1, 3, 2, 4])
    # output2 = serializer2.to_bytes(kv2)

    # assert abs(len(output) - len(output2)) < 10
    output_dict = CacheGenEncoderOutput.from_bytes(output)
    assert output_dict.num_heads == 8
    assert output_dict.head_size == 128


@pytest.mark.parametrize("chunk_size", [16, 128, 256])
def test_cachegen_decoder(chunk_size):
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)

    kv = to_blob(generate_kv_cache(chunk_size, torch_device_type))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0


def test_cachegen_unmatched_size():
    chunk_size = 256
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)

    kv = to_blob(generate_kv_cache(chunk_size - 20, torch_device_type))
    output = serializer.to_bytes(kv)

    decoded_kv = deserializer.from_bytes(output)
    assert decoded_kv.shape == kv.shape
    assert decoded_kv.mean() != 0


@pytest.mark.parametrize("key_max_bins, value_max_bins", [(32, 12), (12, 32), (16, 16)])
def test_cachegen_different_key_value_max_bins(key_max_bins, value_max_bins):
    """Keys and values may use different maximum bin counts.

    The encoder stores the key and value CDFs in one tensor; it used to build
    each at its own maximum and then fail in torch.cat when the two differed.
    The round trip must also stay lossless: arithmetic coding adds no error, so
    the decoder returns exactly the quantize -> dequantize reconstruction.
    """
    chunk_size = 256
    config = LMCacheEngineConfig.from_defaults(chunk_size=chunk_size)
    metadata = LMCacheMetadata(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=None,
    )
    serializer = CacheGenSerializer(config, metadata)
    deserializer = CacheGenDeserializer(config, metadata, torch.bfloat16)
    nlayers = serializer.key_bins.shape[0]
    for coder in (serializer, deserializer):
        coder.key_bins = torch.full(
            (nlayers,), float(key_max_bins), device=torch_device_type
        )
        coder.value_bins = torch.full(
            (nlayers,), float(value_max_bins), device=torch_device_type
        )

    kv = to_blob(generate_kv_cache(chunk_size, torch_device_type))
    decoded_kv = deserializer.from_bytes(serializer.to_bytes(kv))

    expected = []
    bins_kv = (serializer.key_bins, serializer.value_bins)
    for x, bins in zip(_split_kv(kv), bins_kv, strict=True):
        xq, max1 = torch_quant_vectorized(bins, x)
        expected.append(do_dequantize(xq.float(), bins, max1))
    expected_kv = (
        torch.stack(expected)
        .reshape(kv.shape[1], kv.shape[0], *kv.shape[2:])
        .permute((1, 0, 2, 3, 4))
        .to(decoded_kv.dtype)
    )
    assert decoded_kv.shape == kv.shape
    assert torch.equal(decoded_kv, expected_kv)
