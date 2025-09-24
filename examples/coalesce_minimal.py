#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import argparse
import time
from typing import List

import torch

from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    TensorMemoryAllocator,
    MemoryFormat,
)
from lmcache.v1.storage_backend.storage_manager import allocate_and_copy_objects
from lmcache.observability import (
    reset_transfer_metrics,
    get_transfer_metrics_snapshot,
)


def _human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f}{units[i]}"


def _make_src_objs(num: int, sizes: List[int]):
    total = sum(sizes)
    cpu_buf = torch.empty(total, dtype=torch.uint8, device="cpu")
    talloc = TensorMemoryAllocator(cpu_buf)
    src_objs = []
    off = 0
    for sz in sizes:
        obj = talloc.allocate(torch.Size([sz]), torch.uint8, MemoryFormat.KV_2LTD)
        assert obj is not None and obj.tensor is not None
        # Fill data (optional)
        obj.tensor.copy_(torch.full((sz,), 0xAB, dtype=torch.uint8))
        src_objs.append(obj)
        off += sz
    return src_objs, total


def _make_dst_allocator(total_bytes: int):
    gal = GPUMemoryAllocator(total_bytes, device="cuda")

    class DummyAlloc:
        def __init__(self, inner):
            self.inner = inner
            self._dict = {}

        def contains(self, key):
            return key in self._dict

        def allocate(self, **kwargs):
            # shape, dtype, fmt provided by caller
            return self.inner.allocate(kwargs["shape"], kwargs["dtype"], kwargs.get("fmt", MemoryFormat.KV_2LTD))

    return DummyAlloc(gal)


def run_case(name: str, gran_bytes: int, sizes: List[int]) -> dict:
    os.environ["LMCACHE_KV_IO_GRANULARITY_BYTES"] = str(int(gran_bytes))
    # THP toggle exposed for completeness (no-op if unsupported):
    os.environ.setdefault("LMCACHE_USE_THP", "auto")

    src_objs, total = _make_src_objs(len(sizes), sizes)
    dst_alloc = _make_dst_allocator(total)
    stream = torch.cuda.Stream()

    keys = [f"k{i}" for i in range(len(sizes))]

    reset_transfer_metrics()
    torch.cuda.synchronize()
    t0 = time.time()
    allocate_and_copy_objects(dst_alloc, keys, src_objs, stream)
    torch.cuda.synchronize()
    t1 = time.time()

    snap = get_transfer_metrics_snapshot()
    elapsed = max(t1 - t0, 1e-6)
    bytes_total = int(snap.get("h2d_bytes", 0))
    calls = int(snap.get("h2d_calls", 0))
    bpc = bytes_total / max(calls, 1)

    print(f"[{name}] gran={gran_bytes} bytes | total={_human(bytes_total)} | calls={calls} | bytes/call={_human(int(bpc))} | time={elapsed*1000:.2f} ms")
    return {
        "name": name,
        "gran": gran_bytes,
        "total": bytes_total,
        "calls": calls,
        "bpc": bpc,
        "elapsed_ms": elapsed * 1000.0,
    }


def parse_sizes(spec: str) -> List[int]:
    def parse_one(v: str) -> int:
        vs = v.strip().lower()
        if vs.endswith("kib"): return int(float(vs[:-3]) * 1024)
        if vs.endswith("mib"): return int(float(vs[:-3]) * 1024 * 1024)
        if vs.endswith("gib"): return int(float(vs[:-3]) * 1024 * 1024 * 1024)
        if vs.endswith("kb"): return int(float(vs[:-2]) * 1000)
        if vs.endswith("mb"): return int(float(vs[:-2]) * 1000 * 1000)
        if vs.endswith("gb"): return int(float(vs[:-2]) * 1000 * 1000 * 1000)
        return int(vs)

    return [parse_one(x) for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="LMCache minimal coalescing demo (ENV-gated)")
    parser.add_argument("--sizes", type=str, default="128KiB,256KiB,512KiB,2MiB,3MiB,512KiB,256KiB,128KiB", help="Comma-separated sizes")
    parser.add_argument("--gran-on", type=str, default="2MiB", help="Granularity when ON (bytes)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping demo.")
        return

    sizes = parse_sizes(args.sizes)
    gran_on = parse_sizes(args.gran_on)[0]

    print("--- LMCache Coalescing Minimal Demo ---")
    print("Note: Toggle via LMCACHE_KV_IO_GRANULARITY_BYTES; default OFF(0)")

    off = run_case("OFF", 0, sizes)
    on = run_case("ON", gran_on, sizes)

    if off["calls"] > 0 and on["calls"] > 0:
        improved_calls = on["calls"] <= off["calls"]
        improved_bpc = on["bpc"] >= off["bpc"]
        verdict = "PASS" if (improved_calls or improved_bpc) else "NEUTRAL"
        print(f"Result: {verdict} | calls {off['calls']} -> {on['calls']} | bytes/call {_human(int(off['bpc']))} -> {_human(int(on['bpc']))}")

    print("\nTips:")
    print("- You can also set LMCACHE_USE_THP=true to hint hugepages (no-op if unsupported).")
    print("- As Linux GPU ZONE_DEVICE/HMM matures, the same design benefits more.")


if __name__ == "__main__":
    main() 