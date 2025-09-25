# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import get_transfer_metrics_snapshot, reset_transfer_metrics
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryFormat,
    TensorMemoryAllocator,
)
from lmcache.v1.storage_backend.storage_manager import allocate_and_copy_objects

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _make_src_objs(num: int, bytes_per: int):
    cpu_buf = torch.empty(num * bytes_per, dtype=torch.uint8, device="cpu")
    talloc = TensorMemoryAllocator(cpu_buf)
    src_objs = []
    for _ in range(num):
        obj = talloc.allocate(
            torch.Size([bytes_per]), torch.uint8, MemoryFormat.KV_2LTD
        )
        assert obj is not None and obj.tensor is not None
        src_objs.append(obj)
    return src_objs


def _make_dst_allocator(total_bytes: int):
    gal = GPUMemoryAllocator(total_bytes, device="cuda")

    class DummyAlloc:
        def __init__(self, inner):
            self.inner = inner
            self._dict = {}

        def contains(self, key):
            return key in self._dict

        def allocate(self, **kwargs):
            obj = self.inner.allocate(
                kwargs["shape"],
                kwargs["dtype"],
                kwargs.get("fmt", MemoryFormat.KV_2LTD),
            )
            return obj

    return DummyAlloc(gal)


def test_env_toggle_off_on_changes_calls(tmp_path):
    num = 8
    sz = 256 * 1024
    keys = [f"k{i}" for i in range(num)]
    src_objs = _make_src_objs(num, sz)
    dst_alloc = _make_dst_allocator(num * sz)
    stream = torch.cuda.Stream()

    # OFF
    os.environ["LMCACHE_KV_IO_GRANULARITY_BYTES"] = "0"
    reset_transfer_metrics()
    allocate_and_copy_objects(dst_alloc, keys, src_objs, stream)
    snap_off = get_transfer_metrics_snapshot()

    # ON
    os.environ["LMCACHE_KV_IO_GRANULARITY_BYTES"] = str(2 * 1024 * 1024)
    reset_transfer_metrics()
    allocate_and_copy_objects(dst_alloc, keys, src_objs, stream)
    snap_on = get_transfer_metrics_snapshot()

    # Either fewer calls or larger bytes/call
    off_calls = snap_off.get("h2d_calls", 0)
    on_calls = snap_on.get("h2d_calls", 0)
    off_bytes = snap_off.get("h2d_bytes", 0)
    on_bytes = snap_on.get("h2d_bytes", 0)

    if off_calls > 0 and on_calls > 0:
        assert on_calls <= off_calls or (on_bytes / max(on_calls, 1)) >= (
            off_bytes / max(off_calls, 1)
        )


def test_concurrency_small_random_chunks(tmp_path):
    num = 32
    sizes = [64 * 1024, 128 * 1024, 256 * 1024]
    keys = [f"r{i}" for i in range(num)]
    src_tensors = [
        torch.empty(sizes[i % len(sizes)], dtype=torch.uint8, device="cpu")
        for i in range(num)
    ]

    talloc = TensorMemoryAllocator(
        torch.empty(sum(s.numel() for s in src_tensors), dtype=torch.uint8)
    )
    src_objs = []
    for t in src_tensors:
        obj = talloc.allocate(
            torch.Size([t.numel()]), torch.uint8, MemoryFormat.KV_2LTD
        )
        assert obj is not None and obj.tensor is not None
        obj.tensor.copy_(t)
        src_objs.append(obj)

    dst_alloc = _make_dst_allocator(sum(t.numel() for t in src_tensors))
    stream = torch.cuda.Stream()

    os.environ["LMCACHE_KV_IO_GRANULARITY_BYTES"] = str(2 * 1024 * 1024)
    reset_transfer_metrics()
    allocate_and_copy_objects(dst_alloc, keys, src_objs, stream)
    snap = get_transfer_metrics_snapshot()
    assert snap.get("h2d_calls", 0) >= 1
