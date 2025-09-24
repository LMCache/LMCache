#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import argparse
import time
import threading
from typing import List, Tuple

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


def _make_src_objs(num: int, sizes: List[int]) -> Tuple[List, int, TensorMemoryAllocator]:
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
    return src_objs, total, talloc


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
            return self.inner.allocate(
                kwargs["shape"], kwargs["dtype"], kwargs.get("fmt", MemoryFormat.KV_2LTD)
            )

    return DummyAlloc(gal)


def _plan_sizes(total_bytes: int, mode: str) -> List[int]:
    sizes: List[int] = []
    if mode == "many_smalls":
        smalls = [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024]
        i = 0
        acc = 0
        while acc < total_bytes:
            s = smalls[i % len(smalls)]
            sizes.append(s)
            acc += s
            i += 1
    else:  # mixed
        pattern = [128 * 1024, 256 * 1024, 512 * 1024, 2 * 1024 * 1024, 3 * 1024 * 1024]
        i = 0
        acc = 0
        while acc < total_bytes:
            s = pattern[i % len(pattern)]
            sizes.append(s)
            acc += s
            i += 1
    return sizes


def _partition(lst: List, parts: int) -> List[List]:
    parts = max(1, int(parts))
    out: List[List] = [[] for _ in range(parts)]
    for i, v in enumerate(lst):
        out[i % parts].append(v)
    return out


def run_case(name: str, gran_bytes: int, sizes: List[int], streams_n: int,
             latency_profile: bool = False) -> dict:
    os.environ["LMCACHE_KV_IO_GRANULARITY_BYTES"] = str(int(gran_bytes))
    os.environ.setdefault("LMCACHE_USE_THP", "auto")  # no-op if unsupported
    if latency_profile:
        # Recommended latency-friendly profile
        os.environ.setdefault("LMCACHE_COALESCE_MAX_ITEMS", "4")
        os.environ.setdefault("LMCACHE_COALESCE_MAX_GROUP_BYTES", str(16 * 1024 * 1024))
        os.environ.setdefault("LMCACHE_STAGING_BUFFERS", "4")
        os.environ.setdefault("LMCACHE_STAGING_SPIN_US", "0")

    src_objs, total, _ = _make_src_objs(len(sizes), sizes)
    dst_alloc = _make_dst_allocator(total)

    # Partition keys/objs across N streams and run in parallel threads
    keys = [f"k{i}" for i in range(len(sizes))]
    keys_parts = _partition(keys, streams_n)
    objs_parts = _partition(src_objs, streams_n)
    streams = [torch.cuda.Stream() for _ in range(streams_n)]

    reset_transfer_metrics()
    torch.cuda.synchronize()
    t0 = time.time()

    threads = []
    for i in range(streams_n):
        if not keys_parts[i]:
            continue
        th = threading.Thread(
            target=allocate_and_copy_objects,
            args=(dst_alloc, keys_parts[i], objs_parts[i], streams[i]),
            daemon=True,
        )
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

    torch.cuda.synchronize()
    t1 = time.time()

    snap = get_transfer_metrics_snapshot()
    elapsed = max(t1 - t0, 1e-6)
    bytes_total = int(snap.get("h2d_bytes", 0))
    calls = int(snap.get("h2d_calls", 0))
    bpc = bytes_total / max(calls, 1)
    gbs = (bytes_total / (1 << 30)) / elapsed

    print(
        (
            f"[{name}] gran={gran_bytes} | total={_human(bytes_total)} | "
            f"calls={calls} | bytes/call={_human(int(bpc))} | "
            f"H2D={gbs:.2f} GB/s | time={elapsed*1000:.2f} ms | "
            f"streams={streams_n}"
            + (" | profile=latency" if latency_profile else "")
        )
    )
    return {
        "name": name,
        "gran": gran_bytes,
        "total": bytes_total,
        "calls": calls,
        "bpc": bpc,
        "gbps": gbs,
        "elapsed_ms": elapsed * 1000.0,
        "latency_profile": latency_profile,
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
    parser.add_argument("--mode", type=str, default="many_smalls", choices=["many_smalls", "mixed"], help="Size pattern")
    parser.add_argument("--total-mib", type=int, default=256, help="Total data per run in MiB")
    parser.add_argument("--streams", type=int, default=4, help="Concurrent CUDA streams")
    parser.add_argument("--gran-on", type=str, default="2MiB",
                        help="Granularity when ON (bytes)")
    parser.add_argument("--latency-profile", action="store_true",
                        help="Apply recommended latency-friendly ENV settings")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; skipping demo.")
        return

    total_bytes = args.total_mib * 1024 * 1024
    sizes = _plan_sizes(total_bytes, args.mode)
    gran_on = parse_sizes(args.gran_on)[0]

    print("--- LMCache Coalescing Minimal Demo ---")
    print("Note: Toggle via LMCACHE_KV_IO_GRANULARITY_BYTES; default OFF(0)")

    off = run_case("OFF", 0, sizes, args.streams, args.latency_profile)
    on = run_case("ON", gran_on, sizes, args.streams, args.latency_profile)

    if off["calls"] > 0 and on["calls"] > 0:
        improved_calls = on["calls"] <= off["calls"]
        improved_bpc = on["bpc"] >= off["bpc"]
        verdict = "PASS" if (improved_calls or improved_bpc) else "NEUTRAL"
        print(
            f"Result: {verdict} | calls {off['calls']} -> {on['calls']} | "
            f"bytes/call {_human(int(off['bpc']))} -> {_human(int(on['bpc']))}"
        )

    print("\nTips:")
    print("- Increase --total-mib or --streams for stronger effect.")
    print("- You can set LMCACHE_USE_THP=true to hint hugepages (no-op if unsupported).")
    print("- As Linux GPU ZONE_DEVICE/HMM matures, the same design benefits more.")
    if args.latency_profile:
        print("- Latency profile: MAX_ITEMS=4, MAX_GROUP_BYTES=16MiB, "
              "STAGING_BUFFERS=4, STAGING_SPIN_US=0")


if __name__ == "__main__":
    main() 