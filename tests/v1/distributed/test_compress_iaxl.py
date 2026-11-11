# SPDX-License-Identifier: Apache-2.0
"""Tests for the iaxl accel_kv_compress backend."""

# Standard
import importlib.util
import os
import sys
import time
import types
import zlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.compress_adapters import _create_backend
from lmcache.v1.distributed.compress_adapters.iaxl_backend import IaxlBackend
from lmcache.v1.distributed.compress_adapters.serde import (
    AccelCompressDeserializer,
    AccelCompressSerializer,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.dram_l2_adapter import DramL2AdapterConfig
from lmcache.v1.distributed.serde import SerdeConfig
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

_KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="m", kv_rank=0)


class FakeTorchExt:
    """zlib stand-in for iaxl.torch_ext with the same buffer API."""

    def __init__(self) -> None:
        self.compress_calls: list[tuple[int, torch.dtype]] = []

    def zip_max_block_size(self, size: int) -> int:
        return 8 + size + size // 1000 + 64

    def zip_compress_into(self, srcs, dsts, dtype):
        if len(srcs) != len(dsts):
            raise ValueError("count mismatch")
        self.compress_calls.append((len(srcs), dtype))
        sizes = []
        for src, dst in zip(srcs, dsts, strict=True):
            if len(dst) < self.zip_max_block_size(len(src)):
                raise ValueError("destination too small")
            block = len(src).to_bytes(8, "little") + zlib.compress(bytes(src))
            dst[: len(block)] = block
            sizes.append(len(block))
        return sizes

    def zip_decompress_into(self, srcs, dsts):
        sizes = []
        for src, dst in zip(srcs, dsts, strict=True):
            raw = zlib.decompress(bytes(src[8:]))
            dst[: len(raw)] = raw
            sizes.append(len(raw))
        return sizes


@pytest.fixture
def fake_ext(monkeypatch):
    ext = FakeTorchExt()
    pkg = types.ModuleType("iaxl")
    pkg.torch_ext = ext  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "iaxl", pkg)
    monkeypatch.setitem(sys.modules, "iaxl.torch_ext", ext)
    return ext


@pytest.fixture
def ext(request):
    """The fake iaxl.torch_ext, or the installed one for "real"."""
    if request.param == "fake":
        return request.getfixturevalue("fake_ext")
    return None


def _wrap(tensor: torch.Tensor) -> TensorMemoryObj:
    metadata = MemoryObjMetadata(
        shape=torch.Size([tensor.numel()]),
        dtype=torch.uint8,
        address=0,
        phy_size=tensor.numel(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(tensor, metadata, parent_allocator=None)


def _kv_bytes(numel: int, dtype: torch.dtype = torch.bfloat16) -> bytes:
    gen = torch.Generator().manual_seed(0)
    values = (torch.randn(numel, generator=gen) * 0.01).to(dtype)
    return bytes(values.view(torch.uint8).numpy())


def _roundtrip(backend: IaxlBackend, data: bytes, num_groups: int) -> int:
    serializer = AccelCompressSerializer(backend=backend, num_groups=num_groups)
    deserializer = AccelCompressDeserializer(backend=backend)
    src = _wrap(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    dst_tensor = torch.zeros(len(data) * 2 + 8192, dtype=torch.uint8)
    written = serializer.serialize(src, _wrap(dst_tensor), _KEY)
    assert bytes(src.byte_array) == data, "serialize must not modify the source"

    packed = _wrap(dst_tensor[:written].clone())
    out = _wrap(torch.zeros(len(data), dtype=torch.uint8))
    deserializer.deserialize(packed, out, _KEY)
    assert bytes(out.byte_array) == data
    return written


class TestFactory:
    def test_creates_iaxl_backend_with_dtype(self, fake_ext):
        backend = _create_backend({"backend": "iaxl", "dtype": "float16"})
        assert isinstance(backend, IaxlBackend)
        _roundtrip(backend, _kv_bytes(4096, torch.float16), num_groups=1)
        assert fake_ext.compress_calls == [(1, torch.float16)]

    def test_dtype_defaults_to_bfloat16(self, fake_ext):
        backend = _create_backend({"backend": "iaxl"})
        _roundtrip(backend, _kv_bytes(4096), num_groups=1)
        assert fake_ext.compress_calls[0][1] == torch.bfloat16

    @pytest.mark.parametrize("dtype", ["nope", "Tensor"])
    def test_rejects_unknown_dtype(self, fake_ext, dtype):
        with pytest.raises(ValueError, match="dtype"):
            _create_backend({"backend": "iaxl", "dtype": dtype})

    def test_missing_iaxl_raises_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "iaxl", None)
        with pytest.raises(ImportError, match="iaxl"):
            _create_backend({"backend": "iaxl"})


class TestRoundtrip:
    @pytest.mark.parametrize("num_groups", [1, 3])
    def test_roundtrip_batches_all_blocks_in_one_call(self, fake_ext, num_groups):
        _roundtrip(IaxlBackend(), _kv_bytes(1 << 16), num_groups)
        assert fake_ext.compress_calls == [(num_groups, torch.bfloat16)]

    def test_estimate_uses_iaxl_bound(self, fake_ext):
        backend = IaxlBackend()
        assert backend.max_compressed_length(1000) == fake_ext.zip_max_block_size(1000)
        assert backend.max_block_size is None


_needs_real_iaxl = pytest.mark.skipif(
    importlib.util.find_spec("iaxl") is None
    or os.environ.get("IAXL_KV_LOSSY_TRUNC", "0") != "0",
    reason="needs iaxl with lossless compression",
)


@_needs_real_iaxl
def test_real_iaxl_roundtrip():
    data = _kv_bytes(1 << 15)
    assert _roundtrip(IaxlBackend(), data, num_groups=1) < len(data) + 4096


def _wait_for(predicate, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return None


@pytest.mark.parametrize(
    "ext", ["fake", pytest.param("real", marks=_needs_real_iaxl)], indirect=True
)
def test_storage_manager_store_then_prefetch(ext):
    """Store compresses into DramL2 and drops L1; prefetch decompresses."""
    adapter_cfg = DramL2AdapterConfig(max_size_gb=0.1)
    adapter_cfg.serde_config = SerdeConfig(
        type="accel_kv_compress",
        kwargs={"backend": "iaxl", "dtype": "bfloat16", "max_workers": 2},
    )
    sm = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 << 20, use_lazy=True, init_size_in_bytes=32 << 20
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[adapter_cfg]),
            store_policy="skip_l1",
        )
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 16, 256])], dtypes=[torch.bfloat16]
    )
    keys = [
        ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)
        for i in range(3)
    ]
    try:
        reserved = sm.reserve_write(keys, layout)
        originals = {}
        for i, key in enumerate(keys):
            gen = torch.Generator().manual_seed(i)
            data = torch.randn(reserved[key].tensor.shape, generator=gen)
            reserved[key].tensor.copy_(data.to(torch.bfloat16))
            originals[key] = reserved[key].tensor.clone()
        sm.finish_write(list(reserved), l1_owners_of(reserved))

        assert _wait_for(
            lambda: sm.report_status()["l1_manager"]["total_object_count"] == 0
        ), sm.report_status()

        handle = sm.submit_prefetch_task(
            PrefetchTaskSpec(
                key_groups=[
                    GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=layout)
                ]
            )
        )
        result = _wait_for(lambda: sm.query_prefetch_status(handle))
        assert result is not None
        assert result.l2_hit_cells[0].count_leading_ones() == len(keys)
        with sm.read_prefetched_results(keys) as objs:
            for key, obj in zip(keys, objs, strict=True):
                assert torch.equal(obj.tensor, originals[key])
        sm.finish_read_prefetched(keys)
    finally:
        sm.close()
