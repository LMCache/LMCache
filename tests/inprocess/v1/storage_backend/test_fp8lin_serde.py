# SPDX-License-Identifier: Apache-2.0
"""Tests for ``remote_serde: fp8lin`` (lossless predictive coding of FP8 KV).

The coder used here has random weights: prediction quality only changes the size,
never correctness, so every round trip must be exact.
"""

# Standard
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev
from lmcache.inprocess.v1.storage_backend.naive_serde import CreateSerde
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata

# Test helpers
from tests.v1.fp8lin_testutil import write_coder

if not torch_dev.is_available():
    pytest.skip("fp8lin needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

L, H, D = 3, 2, 128
C = H * D
CHUNK = 256


def _serde(tmp_path: Path, kv_dtype: torch.dtype):
    write_coder(tmp_path, L, H, D)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=CHUNK, extra_config={"fp8lin.coder_path": str(tmp_path)}
    )
    metadata = LMCacheMetadata(
        model_name="test/fp8lin",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=kv_dtype,
        kv_shape=(L, 2, CHUNK, H, D),
    )
    return CreateSerde("fp8lin", metadata, config)


def _obj(kv: torch.Tensor, positions: torch.Tensor | None = None) -> TensorMemoryObj:
    meta = MemoryObjMetadata(
        shape=kv.shape,
        dtype=kv.dtype,
        address=-1,
        phy_size=kv.numel() * kv.element_size(),
        ref_count=-1,
        fmt=MemoryFormat.KV_2LTD,
        cached_positions=positions,
    )
    return TensorMemoryObj(raw_data=kv, metadata=meta, parent_allocator=None)


def _fp8_kv(T: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, L, T, C, generator=g) * 4
    return x.to(torch.float8_e4m3fn).view(torch.uint8)  # vLLM stores fp8 as uint8


def test_fp8_kv_roundtrip_is_lossless(tmp_path):
    s, d = _serde(tmp_path, torch.uint8)
    kv = _fp8_kv(CHUNK, 1)
    out = d.deserialize(s.serialize(_obj(kv, torch.arange(512, 512 + CHUNK))))
    assert out is not None
    assert torch.equal(out.tensor.cpu(), kv)


def test_bf16_kv_roundtrip_equals_e4m3_rounding(tmp_path):
    s, d = _serde(tmp_path, torch.bfloat16)
    kv = torch.randn(2, L, CHUNK, C, generator=torch.Generator().manual_seed(2)).to(
        torch.bfloat16
    )
    scale = torch.tensor([[0.05] * L, [0.02] * L])[..., None, None]
    ref = (kv.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    out = d.deserialize(s.serialize(_obj(kv)))
    assert out is not None
    assert out.tensor.dtype == torch.bfloat16
    assert torch.equal(out.tensor.cpu(), ref.to(torch.bfloat16))


def test_batched_roundtrip_with_partial_chunk(tmp_path):
    s, d = _serde(tmp_path, torch.uint8)
    kvs = [_fp8_kv(CHUNK, 10 + i) for i in range(3)] + [_fp8_kv(100, 20)]
    blobs = s.batched_serialize([_obj(kv) for kv in kvs])
    outs = d.batched_deserialize(blobs)
    for kv, out in zip(kvs, outs, strict=True):
        assert out is not None
        assert torch.equal(out.tensor.cpu(), kv)
    single = d.deserialize(s.serialize(_obj(kvs[1])))
    assert single is not None and torch.equal(single.tensor.cpu(), kvs[1])


def test_decode_independent_of_batching(tmp_path):
    s, d = _serde(tmp_path, torch.uint8)
    kvs = [_fp8_kv(CHUNK, 30 + i) for i in range(4)]
    blobs = s.batched_serialize([_obj(kv) for kv in kvs])
    for order in ([2], [3, 0], [1, 2, 3, 0]):
        outs = d.batched_deserialize([blobs[i] for i in order])
        for i, out in zip(order, outs, strict=True):
            assert out is not None and torch.equal(out.tensor.cpu(), kvs[i])


def test_smaller_than_fp8(tmp_path):
    s, _ = _serde(tmp_path, torch.uint8)
    kv = _fp8_kv(CHUNK, 3)
    blob = s.serialize(_obj(kv))
    assert len(bytes(blob.byte_array)) < kv.numel()


def test_nan_codes_fall_back_to_raw(tmp_path):
    s, d = _serde(tmp_path, torch.uint8)
    kv = _fp8_kv(CHUNK, 4)
    kv[1, 2, 3, 4] = 0x7F  # an E4M3 NaN
    out = d.deserialize(s.serialize(_obj(kv)))
    assert out is not None
    assert torch.equal(out.tensor.cpu(), kv)


def test_corrupt_blob_is_a_miss(tmp_path):
    s, d = _serde(tmp_path, torch.uint8)
    blob = bytearray(bytes(s.serialize(_obj(_fp8_kv(CHUNK, 5))).byte_array))
    blob[len(blob) // 2] ^= 0x5A
    # First Party
    from lmcache.v1.memory_management import BytesBufferMemoryObj

    try:
        out = d.deserialize(BytesBufferMemoryObj(bytes(blob)))
    except Exception:  # a malformed stream may also be rejected outright
        out = None
    assert out is None


def test_missing_coder_path_raises():
    config = LMCacheEngineConfig.from_defaults(chunk_size=CHUNK)
    metadata = LMCacheMetadata(
        model_name="test/fp8lin",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(L, 2, CHUNK, H, D),
    )
    with pytest.raises(ValueError, match="fp8lin.coder_path"):
        CreateSerde("fp8lin", metadata, config)
