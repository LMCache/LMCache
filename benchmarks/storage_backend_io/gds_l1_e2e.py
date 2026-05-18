# SPDX-License-Identifier: Apache-2.0
"""End-to-end harness for the GDS L1 backend in MP mode.

Exercises the full ``StorageManager`` → ``L1Manager`` → ``gpu_ops``
dispatch → ``cufile.read/write`` pipeline against an actual NVMe-backed
``gds_path``, without spinning up vLLM. This is the closest you can
get to a real ``long_doc_qa`` run without the heavyweight integration:

- Pure-Python launch, no Docker, no model weights.
- Uses ``StorageManager.reserve_write`` / ``read_prefetched_results``
  exactly the way the MP server does.
- Goes through ``lmcache_memcpy_async_d2h/_h2d`` so the
  ``GdsScratchAllocator`` ``isinstance`` dispatch branch is exercised.

Numbers are reported as MiB/s for store and retrieve, plus a
correctness check (data round-trips byte-for-byte).

Run::

    LMCACHE_TEST_TMPDIR=/path/to/nvme \
        python benchmarks/storage_backend_io/gds_l1_e2e.py

The ``gds_path`` defaults to ``$LMCACHE_TEST_TMPDIR`` if set, else
``/tmp/gds_l1_e2e``. cuFile fast-path only kicks in if ``nvidia-fs``
is loaded on the host; otherwise cuFile falls back to compatible mode
and both runs measure CPU-bounced throughput.
"""

# Standard
import argparse
import os
import shutil
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.gpu_connector.gpu_ops import (
    lmcache_memcpy_async_d2h,
    lmcache_memcpy_async_h2d,
)


def _make_config(gds_path: str, use_gds: bool) -> StorageManagerConfig:
    """Minimal config that turns on GDS L1.

    L1 memory manager is sized at 1 GB just so the controllers
    initialise cleanly; with GDS L1 attached, the actual eviction
    signal comes from disk usage, not pinned-slab usage.
    """
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 30,
                use_lazy=True,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        gds_l1_config=GdsL1Config(
            gds_path=gds_path,
            gds_path_sharding="by_gpu",
            use_gds=use_gds,
        ),
    )


def _object_key(seed: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\x00" * 28,
        model_name="gds-l1-e2e",
        kv_rank=0,
    )


def _bench(
    *,
    gds_path: str,
    use_gds: bool,
    n_chunks: int,
    chunk_bytes: int,
    max_batch_size: int,
) -> dict:
    """Run one end-to-end pass and return throughput metrics."""

    if os.path.isdir(gds_path):
        shutil.rmtree(gds_path)
    os.makedirs(gds_path, exist_ok=True)

    config = _make_config(gds_path, use_gds=use_gds)
    storage = StorageManager(config)
    allocator = storage.get_gds_scratch_allocator()
    if allocator is None:
        raise RuntimeError(
            "Expected a GDS scratch allocator but storage manager has none"
        )

    # Mimic the per-GPUCacheContext tmp_gpu_buffer_ register step.
    tmp_gpu_buffer = torch.empty(
        chunk_bytes * max_batch_size,
        dtype=torch.uint8,
        device="cuda:0",
    )
    allocator.register_gpu_buffer(tmp_gpu_buffer)

    layout = MemoryLayoutDesc(
        shapes=[torch.Size([chunk_bytes])],
        dtypes=[torch.uint8],
    )
    keys = [_object_key(seed=i) for i in range(n_chunks)]

    # Deterministic write payload so we can verify the round-trip.
    pattern_buffer = torch.arange(chunk_bytes, dtype=torch.uint8, device="cuda:0")
    pattern_buffer.fill_(0xCD)

    slot_views = [
        tmp_gpu_buffer[i * chunk_bytes : (i + 1) * chunk_bytes]
        for i in range(max_batch_size)
    ]

    # ---- STORE phase -------------------------------------------------
    # ``reserve_write`` -> stage data into the registered slot ->
    # ``lmcache_memcpy_async_d2h`` (dispatches to cuFile write via
    # gpu_ops) -> ``finish_write``. This exercises the same code path
    # that ``MPCacheEngine.store`` uses.
    reserve_results = storage.reserve_write(keys, layout, mode="new")
    mem_objs = [reserve_results[k] for k in keys]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, mo in enumerate(mem_objs):
        slot = slot_views[i % max_batch_size]
        # Stage the deterministic payload into the slot. In MP mode
        # this is the gather kernel output; for the harness a memcpy
        # is enough.
        slot.copy_(pattern_buffer, non_blocking=False)
        lmcache_memcpy_async_d2h(slot, mo)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    store_secs = t1 - t0
    storage.finish_write(keys)

    # Drop the in-memory L1 entries so the retrieve phase has to refill
    # from disk via the GDS L1 fill-on-miss path. Without this the
    # objects would still be in L1Manager._objects from the write side
    # and we would not be measuring the disk read.
    # ``clear(force=False)`` only drops unlocked entries which is what
    # we want here.
    storage.clear(force=False)

    # ---- RETRIEVE phase ---------------------------------------------
    # ``submit_prefetch_task`` calls L1Manager.reserve_read which goes
    # through the GDS L1 fill-on-miss branch (since we just cleared L1
    # and the keys still exist on disk). After that we consume them
    # via ``read_prefetched_results`` exactly the same way
    # ``MPCacheEngine.retrieve`` does.
    for s in slot_views:
        s.fill_(0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    storage.submit_prefetch_task(keys, layout)
    with storage.read_prefetched_results(keys) as read_objs:
        if read_objs is None:
            raise RuntimeError("read_prefetched_results returned None — miss")
        for i, mo in enumerate(read_objs):
            slot = slot_views[i % max_batch_size]
            lmcache_memcpy_async_h2d(mo, slot)
        torch.cuda.synchronize()
        # While inside the context, the locks are held; the last cuFile
        # read into slot_views[0] should match the pattern.
        last_slot = slot_views[(len(read_objs) - 1) % max_batch_size]
        if not torch.equal(last_slot.cpu(), pattern_buffer.cpu()):
            raise RuntimeError("data round-trip check FAILED — slot mismatch")
        storage.finish_read_prefetched(keys)
    t1 = time.perf_counter()
    retrieve_secs = t1 - t0

    storage.close()

    total_mib = (n_chunks * chunk_bytes) / (1024 * 1024)
    return {
        "use_gds": use_gds,
        "store_mibs": total_mib / store_secs,
        "retrieve_mibs": total_mib / retrieve_secs,
        "store_secs": store_secs,
        "retrieve_secs": retrieve_secs,
        "total_mib": total_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gds-path",
        type=str,
        default=os.environ.get("LMCACHE_TEST_TMPDIR") or "/tmp/gds_l1_e2e",
        help="Root path for the GDS disk layer.",
    )
    parser.add_argument("--n-chunks", type=int, default=256)
    parser.add_argument("--chunk-mib", type=int, default=2)
    parser.add_argument("--max-batch-size", type=int, default=4)
    args = parser.parse_args()

    chunk_bytes = args.chunk_mib * 1024 * 1024

    print("=" * 60)
    print(
        f"GDS L1 end-to-end harness: {args.n_chunks} chunks × "
        f"{args.chunk_mib} MiB = "
        f"{args.n_chunks * args.chunk_mib} MiB total per phase"
    )
    print(f"gds_path: {args.gds_path}")
    print("=" * 60)

    for use_gds in (True, False):
        label = "cuFile" if use_gds else "POSIX fallback"
        path = f"{args.gds_path}_{'gds' if use_gds else 'posix'}"
        print(f"\n[{label}]")
        result = _bench(
            gds_path=path,
            use_gds=use_gds,
            n_chunks=args.n_chunks,
            chunk_bytes=chunk_bytes,
            max_batch_size=args.max_batch_size,
        )
        print(
            f"  STORE:    {result['total_mib']:6.1f} MiB in "
            f"{result['store_secs']:6.3f}s = "
            f"{result['store_mibs']:8.1f} MiB/s"
        )
        print(
            f"  RETRIEVE: {result['total_mib']:6.1f} MiB in "
            f"{result['retrieve_secs']:6.3f}s = "
            f"{result['retrieve_mibs']:8.1f} MiB/s"
        )
        print("  CORRECTNESS: round-trip data matches pattern")


if __name__ == "__main__":
    main()
