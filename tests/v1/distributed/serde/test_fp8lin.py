# SPDX-License-Identifier: Apache-2.0
"""
Tests for the fp8lin serde (lossless predictive coding of FP8 KV) in
multiprocess mode: the pure transform, and a full disk round trip through the
StorageManager with the filesystem L2 adapter (variable-size objects).
"""

# Standard
from dataclasses import dataclass
from typing import Optional
import os
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev

if not torch_dev.is_available():
    pytest.skip("fp8lin needs CUDA", allow_module_level=True)
pytest.importorskip("triton")

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.config import (  # noqa: E402
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig  # noqa: E402
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (  # noqa: E402
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.serde import (  # noqa: E402
    SerdeConfig,
    create_serde_processor,
)
from lmcache.v1.distributed.serde.fp8lin import (  # noqa: E402
    Fp8LinDeserializer,
    Fp8LinSerializer,
    SharedCodec,
)
from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: E402
from lmcache.v1.platform import current_device_spec  # noqa: E402

# Test helpers
from tests.v1.distributed.utils import single_row_spec  # noqa: E402
from tests.v1.fp8lin_testutil import write_coder  # noqa: E402

L, H, D = 3, 2, 128
C = H * D
T = 256
KEY = ObjectKey(chunk_hash=b"\x00" * 32, model_name="test", kv_rank=0)


@dataclass
class _Obj:
    """Minimal stand-in exposing ``.tensor``."""

    tensor: Optional[torch.Tensor]


@pytest.fixture
def coder(tmp_path):
    write_coder(tmp_path, L, H, D)
    return str(tmp_path)


def _fp8(seed: int, tokens: int = T) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (
        (torch.randn(2, L, tokens, C, generator=g) * 4)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )


def _pair(coder: str):
    shared = SharedCodec(coder, "cuda", T)
    return Fp8LinSerializer(shared), Fp8LinDeserializer(shared)


def test_estimate_bounds_raw_codes(coder):
    s, _ = _pair(coder)
    layout = MemoryLayoutDesc(shapes=[torch.Size([2, L, T, C])], dtypes=[torch.uint8])
    assert s.estimate_serialized_size(layout) >= 2 * L * T * C


def test_fp8_roundtrip_lossless_and_smaller(coder):
    s, d = _pair(coder)
    kv = _fp8(1)
    layout = MemoryLayoutDesc(shapes=[kv.shape], dtypes=[kv.dtype])
    buf = torch.zeros(s.estimate_serialized_size(layout), dtype=torch.uint8)
    n = s.serialize(_Obj(kv), _Obj(buf), KEY)
    assert 0 < n < kv.numel()
    out = torch.zeros_like(kv)
    d.deserialize(_Obj(buf), _Obj(out), KEY)  # the buffer may be longer than n
    assert torch.equal(out, kv)


@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.bfloat16])
def test_fused_layout_matches_split(coder, kv_dtype):
    """[1, L, T, H*2*D] (per-head K and V adjacent) codes like [2, L, T, C]."""
    s, d = _pair(coder)
    split = _fp8(4)
    if kv_dtype != torch.uint8:
        sc = torch.tensor([[0.05] * L, [0.02] * L])[..., None, None]
        split = (split.view(torch.float8_e4m3fn).float() * sc).to(kv_dtype)
    fused = (
        split.view(2, L, T, H, D)
        .permute(1, 2, 3, 0, 4)
        .reshape(1, L, T, 2 * C)
        .contiguous()
    )
    bufs = [torch.zeros(L * T * C * 2 + 4096, dtype=torch.uint8) for _ in range(2)]
    n_split = s.serialize(_Obj(split), _Obj(bufs[0]), KEY)
    n_fused = s.serialize(_Obj(fused), _Obj(bufs[1]), KEY)
    assert n_split == n_fused and torch.equal(bufs[0][:n_split], bufs[1][:n_fused])
    out = torch.zeros_like(fused)
    d.deserialize(_Obj(bufs[1]), _Obj(out), KEY)
    assert torch.equal(out, fused)


def test_bf16_roundtrip_equals_e4m3_rounding(coder):
    s, d = _pair(coder)
    g = torch.Generator().manual_seed(2)
    kv = torch.randn(2, L, T, C, generator=g).to(torch.bfloat16)
    scale = torch.tensor([[0.05] * L, [0.02] * L])[..., None, None]
    ref = (kv.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    buf = torch.zeros(L * T * C * 2 + 4096, dtype=torch.uint8)
    s.serialize(_Obj(kv), _Obj(buf), KEY)
    out = torch.zeros_like(kv)
    d.deserialize(_Obj(buf), _Obj(out), KEY)
    assert torch.equal(out, ref.to(torch.bfloat16))


def test_batch_and_order_independent(coder):
    s, d = _pair(coder)
    kvs = [_fp8(10 + i) for i in range(4)]
    bufs = [torch.zeros(2 * L * T * C + 4096, dtype=torch.uint8) for _ in kvs]
    s.serialize_batch([_Obj(k) for k in kvs], [_Obj(b) for b in bufs], [KEY] * 4)
    for order in ([2], [3, 0], [1, 2, 3, 0]):
        outs = [torch.zeros_like(kvs[i]) for i in order]
        d.deserialize_batch(
            [_Obj(bufs[i]) for i in order], [_Obj(o) for o in outs], [KEY] * len(order)
        )
        for i, o in zip(order, outs, strict=True):
            assert torch.equal(o, kvs[i])


def test_corrupt_blob_raises(coder):
    s, d = _pair(coder)
    kv = _fp8(5)
    buf = torch.zeros(2 * L * T * C + 4096, dtype=torch.uint8)
    n = s.serialize(_Obj(kv), _Obj(buf), KEY)
    buf[n // 2] ^= 0x5A
    with pytest.raises(ValueError):
        d.deserialize(_Obj(buf), _Obj(torch.zeros_like(kv)), KEY)


def test_factory(coder):
    proc = create_serde_processor(
        SerdeConfig(type="fp8lin", kwargs={"coder_path": coder})
    )
    try:
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, L, T, C])], dtypes=[torch.uint8]
        )
        assert proc.estimate_serialized_size(layout) >= 2 * L * T * C
    finally:
        proc.close()


def _wait(pred, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("kv_dtype", [torch.uint8, torch.bfloat16])
def test_fs_roundtrip_variable_size(coder, kv_dtype):
    """L1 write -> fp8lin serialize -> disk (exact bytes) -> clear L1 -> load ->
    deserialize -> identical KV (FP8) / E4M3-rounded KV (bf16)."""
    disk = tempfile.mkdtemp(prefix="lmcache_fp8lin_fs_")
    try:
        fs_cfg = FSL2AdapterConfig(
            base_path=disk,
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_cfg.serde_config = SerdeConfig(type="fp8lin", kwargs={"coder_path": coder})
        sm = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=1 << 30,
                        use_lazy=current_device_spec.is_pin_supported,
                        init_size_in_bytes=1 << 28,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
        )
        shape = torch.Size([2, L, T, C])
        layout = MemoryLayoutDesc(shapes=[shape], dtypes=[kv_dtype])
        keys = [
            ObjectKey(chunk_hash=b"\x00" * 31 + bytes([i]), model_name="m", kv_rank=0)
            for i in (1, 2)
        ]
        if kv_dtype == torch.uint8:
            originals = [_fp8(20 + i) for i in range(2)]
            expected = originals
        else:
            # values on the coder's E4M3 grid, so they code below raw size
            sc = torch.tensor([[0.05] * L, [0.02] * L])[..., None, None]
            originals = [
                (_fp8(20 + i).view(torch.float8_e4m3fn).float() * sc).to(kv_dtype)
                for i in range(2)
            ]
            expected = [
                (
                    (o.float() / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float()
                    * sc
                ).to(kv_dtype)
                for o in originals
            ]
        reserved = sm.reserve_write(keys, layout)
        for k, o in zip(keys, originals, strict=True):
            reserved[k].tensor.view(shape).view(kv_dtype).copy_(o)
        sm.finish_write(keys)

        def files() -> list[os.DirEntry]:
            return [
                e
                for e in os.scandir(disk)
                if e.is_file() and not e.name.endswith(".tmp")
            ]

        assert _wait(lambda: len(files()) >= len(keys))
        assert _wait(
            lambda: sm.report_status()["store_controller"]["in_flight_task_count"] == 0
            and sm.report_status()["store_controller"]["pending_keys_count"] == 0
        )
        raw = 2 * L * T * C
        assert all(e.stat().st_size < raw for e in files()), (
            "stored objects not smaller"
        )

        sm.clear(force=True)
        handle = sm.submit_prefetch_task(single_row_spec(keys, layout))
        hits = None
        end = time.monotonic() + 20
        while hits is None and time.monotonic() < end:
            r = sm.query_prefetch_status(handle)
            hits = None if r is None else r.hit_cells[0].count_leading_ones()
            time.sleep(0.1)
        assert hits == len(keys)
        with sm.read_prefetched_results(keys) as objs:
            for e, o in zip(expected, objs, strict=True):
                assert torch.equal(o.tensor.view(shape).view(kv_dtype).cpu(), e)
        sm.finish_read_prefetched(keys)
        sm.close()
    finally:
        shutil.rmtree(disk, ignore_errors=True)
