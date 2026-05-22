# SPDX-License-Identifier: Apache-2.0
"""Probe + benchmark implementation for ``lmcache tool gds-check``.

Three phases, each independently skippable from the CLI:

- ``info``: read-only host inspection — fstype under ``gds_path``,
  whether ``nvidia-fs`` is loaded, kvikio's compat-mode preference,
  cuFile 4 KiB alignment compatibility for the requested chunk size.
  Catches setup mistakes before any I/O happens.
- ``verify``: write a known pattern through ``GdsScratchAllocator.
  cufile_write_from``, read it back through ``cufile_read_into``,
  byte-compare the result. Same code path as the production data
  path; if this passes, the GDS L1 backend works on this host.
- ``bench``: store + retrieve N × chunk MiB and report MiB/s for
  each direction. Useful for comparing hardware (the absolute numbers
  matter much less than the delta between a compat-mode host and an
  ``nvidia-fs``-enabled one).
"""

# Standard
from dataclasses import dataclass
import asyncio
import os
import shutil
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.gds_l1 import GdsL1Backend, key_to_disk_path
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata

logger = init_logger(__name__)


@dataclass
class HostInfo:
    """Snapshot of the host's GDS-relevant state, returned by :func:`probe_host`."""

    gds_path: str
    fstype: str
    nvidia_fs_loaded: bool
    kvikio_compat_mode_preferred: bool
    chunk_bytes: int
    chunk_aligned_4kib: bool


@dataclass
class BenchResult:
    """One direction (store or retrieve) of the throughput benchmark."""

    direction: str
    total_mib: float
    seconds: float

    @property
    def mibs(self) -> float:
        return self.total_mib / self.seconds if self.seconds > 0 else float("nan")


def _start_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """Spin up an asyncio loop on a background daemon thread.

    ``GdsL1Backend`` schedules its startup metadata scan on this loop.
    The thread is daemon so an unclean exit doesn't block the
    interpreter; :func:`_stop_loop` joins it cleanly on success.
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=loop.run_forever, name="gds-check-loop", daemon=True
    )
    thread.start()
    return loop, thread


def _stop_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Stop the background loop and join the thread."""
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    loop.close()


def _object_key(seed: int) -> ObjectKey:
    """Deterministic ObjectKey for repeatable benchmark runs."""
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\x00" * 28,
        model_name="lmcache-tool-gds-check",
        kv_rank=0,
    )


def probe_host(gds_path: str, chunk_bytes: int) -> HostInfo:
    """Inspect the host's GDS readiness without touching disk.

    Args:
        gds_path: Directory the GDS L1 backend would use.
        chunk_bytes: Per-chunk size the caller plans to use; used
            only to compute the 4 KiB alignment check that cuFile
            requires.

    Returns:
        A :class:`HostInfo` snapshot.
    """
    # First Party
    from lmcache.v1.distributed.gds_l1 import get_fstype

    os.makedirs(gds_path, exist_ok=True)
    fstype = get_fstype(gds_path)

    nvidia_fs_loaded = os.path.exists("/proc/driver/nvidia-fs/stats")

    try:
        # Third Party
        import kvikio.defaults

        compat = bool(kvikio.defaults.is_compat_mode_preferred())
    except Exception:
        # kvikio not importable on this host. The backend would have
        # failed to load anyway; we surface that as "compat-mode
        # preferred" because that's the practical outcome.
        compat = True

    return HostInfo(
        gds_path=gds_path,
        fstype=fstype,
        nvidia_fs_loaded=nvidia_fs_loaded,
        kvikio_compat_mode_preferred=compat,
        chunk_bytes=chunk_bytes,
        chunk_aligned_4kib=(chunk_bytes % 4096 == 0),
    )


def verify_round_trip(
    backend: GdsL1Backend, chunk_bytes: int, pattern_val: int = 0xCD
) -> None:
    """Write a deterministic pattern, read it back, fail loudly on mismatch.

    Uses the same ``GdsScratchAllocator`` code path as production.

    Args:
        backend: An already-constructed :class:`GdsL1Backend`.
        chunk_bytes: Size of the test chunk in bytes (must be 4 KiB
            aligned).
        pattern_val: Byte value to fill the test chunk with.

    Raises:
        RuntimeError: If the bytes read back do not match what was
            written.
    """
    allocator = backend.scratch_allocator
    buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0")
    allocator.register_gpu_buffer(buf)
    try:
        buf.fill_(pattern_val)
        torch.cuda.synchronize()

        key = _object_key(seed=0xCAFE)
        layout_path, _, _, _ = key_to_disk_path(
            key, backend.gds_path, backend.data_suffix
        )
        meta = MemoryObjMetadata(
            shape=torch.Size([chunk_bytes]),
            dtype=torch.uint8,
            address=0,
            phy_size=chunk_bytes,
            ref_count=0,
            pin_count=0,
            fmt=MemoryFormat.KV_2LTD,
            shapes=[torch.Size([chunk_bytes])],
            dtypes=[torch.uint8],
        )
        # First Party
        from lmcache.v1.distributed.gds_l1 import _METADATA_MAX_SIZE, GdsMemoryObj

        mo = GdsMemoryObj(
            key=key,
            disk_path=layout_path,
            file_offset=_METADATA_MAX_SIZE,
            metadata=meta,
            parent_allocator=allocator,
        )
        allocator.cufile_write_from(mo, buf)

        buf.zero_()
        torch.cuda.synchronize()
        allocator.cufile_read_into(mo, buf)
        torch.cuda.synchronize()

        if not torch.equal(
            buf.cpu(), torch.full((chunk_bytes,), pattern_val, dtype=torch.uint8)
        ):
            raise RuntimeError(
                "GDS round-trip mismatch: bytes read back do not match the "
                f"written pattern (0x{pattern_val:02x}). Most likely a "
                "cuFile compat-mode / nvidia-fs setup issue."
            )
    finally:
        allocator.deregister_gpu_buffer()


def benchmark(
    backend: GdsL1Backend, num_chunks: int, chunk_bytes: int, max_batch_size: int = 4
) -> tuple[BenchResult, BenchResult]:
    """Store then retrieve ``num_chunks × chunk_bytes`` and report timings.

    The store / retrieve loop mirrors ``benchmarks/storage_backend_io/
    gds_l1_e2e.py`` — same dispatch shape as the MP server.

    Args:
        backend: An already-constructed :class:`GdsL1Backend`.
        num_chunks: Number of distinct chunks to write / read.
        chunk_bytes: Bytes per chunk (4 KiB aligned).
        max_batch_size: Number of staging slots in the registered
            buffer. Defaults to 4, matching
            ``GPUCacheContext.max_batch_size``.

    Returns:
        ``(store_result, retrieve_result)``.
    """
    allocator = backend.scratch_allocator

    tmp_buf = torch.empty(
        max_batch_size * chunk_bytes, dtype=torch.uint8, device="cuda:0"
    )
    allocator.register_gpu_buffer(tmp_buf)
    try:
        # Deterministic payload — content doesn't matter, just needs
        # to be on the GPU so cuFile sees a registered source.
        pattern = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8, device="cuda:0")

        slot_views = [
            tmp_buf[i * chunk_bytes : (i + 1) * chunk_bytes]
            for i in range(max_batch_size)
        ]
        # First Party
        from lmcache.v1.distributed.gds_l1 import _METADATA_MAX_SIZE, GdsMemoryObj

        mem_objs: list[GdsMemoryObj] = []
        for i in range(num_chunks):
            key = _object_key(seed=i)
            path, _, _, _ = key_to_disk_path(key, backend.gds_path, backend.data_suffix)
            meta = MemoryObjMetadata(
                shape=torch.Size([chunk_bytes]),
                dtype=torch.uint8,
                address=0,
                phy_size=chunk_bytes,
                ref_count=0,
                pin_count=0,
                fmt=MemoryFormat.KV_2LTD,
                shapes=[torch.Size([chunk_bytes])],
                dtypes=[torch.uint8],
            )
            mem_objs.append(
                GdsMemoryObj(
                    key=key,
                    disk_path=path,
                    file_offset=_METADATA_MAX_SIZE,
                    metadata=meta,
                    parent_allocator=allocator,
                )
            )

        # --- STORE phase ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, mo in enumerate(mem_objs):
            slot = slot_views[i % max_batch_size]
            slot.copy_(pattern, non_blocking=False)
            allocator.cufile_write_from(mo, slot)
        torch.cuda.synchronize()
        store_secs = time.perf_counter() - t0

        # --- RETRIEVE phase ---
        for s in slot_views:
            s.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, mo in enumerate(mem_objs):
            slot = slot_views[i % max_batch_size]
            allocator.cufile_read_into(mo, slot)
        torch.cuda.synchronize()
        retrieve_secs = time.perf_counter() - t0
    finally:
        allocator.deregister_gpu_buffer()

    total_mib = (num_chunks * chunk_bytes) / (1024 * 1024)
    return (
        BenchResult("store", total_mib, store_secs),
        BenchResult("retrieve", total_mib, retrieve_secs),
    )


def run_gds_check(
    *,
    gds_path: str,
    small_num_chunks: int,
    small_chunk_bytes: int,
    large_num_chunks: int,
    large_chunk_bytes: int,
    use_gds: bool,
    skip_verify: bool,
    skip_bench: bool,
) -> None:
    """Top-level entry point invoked by the CLI wrapper.

    Performs the host-info probe unconditionally, then optionally the
    verify and bench phases. The bench phase runs two workload shapes
    back-to-back:

    - **small chunks** (default 64 × 2 MiB): per-call-overhead-dominated.
      Tells you how much time is spent in kvikio / Python dispatch
      relative to the actual I/O — relevant for KV-cache-shaped
      access patterns.
    - **large chunks** (default 8 × 256 MiB): bandwidth-dominated.
      Tells you how close the path gets to peak NVMe / cuFile
      throughput once per-call overhead is amortized.

    The verify phase uses the small chunk size (faster) and is the
    same byte-for-byte round-trip check as before.
    """
    # Probe with the small chunk size — alignment check only needs
    # one chunk_bytes value, and both phases use 4 KiB multiples so
    # either would do.
    info = probe_host(gds_path, small_chunk_bytes)

    print("=" * 60)
    print("LMCache GDS L1 platform check")
    print("=" * 60)
    print(f"  gds_path             : {info.gds_path}")
    print(f"  fstype               : {info.fstype}")
    print(f"  nvidia-fs loaded     : {info.nvidia_fs_loaded}")
    print(f"  kvikio compat mode   : {info.kvikio_compat_mode_preferred}")
    print(f"  chunk size           : {info.chunk_bytes} bytes")
    print(f"  4 KiB-aligned chunk  : {info.chunk_aligned_4kib}")
    if not info.nvidia_fs_loaded:
        print()
        print("  NOTE: nvidia-fs not loaded, so cuFile will run in compat mode.")
        print("        Reads/writes go through CPU-bounced libcufile, not the")
        print("        true GDS DMA path. Throughput numbers below will be")
        print("        significantly lower than what a GDS-enabled host shows.")
    if not info.chunk_aligned_4kib:
        print()
        print("  ERROR: chunk size is not a 4 KiB multiple. cuFile requires")
        print("         4 KiB alignment; GDS L1 will hard-error at registration.")
        raise SystemExit(2)

    if skip_verify and skip_bench:
        return

    # Wipe the gds_path so verify/bench start fresh (avoid stale files
    # from a prior run inflating disk-cache hits).
    if os.path.isdir(gds_path):
        shutil.rmtree(gds_path)
    os.makedirs(gds_path, exist_ok=True)

    config = GdsL1Config(
        gds_path=gds_path,
        gds_path_sharding="by_gpu",
        use_gds=use_gds,
        use_direct_io=False,
    )
    loop, thread = _start_loop()
    backend = GdsL1Backend(config=config, loop=loop, dst_device="cuda:0")
    try:
        backend.wait_for_scan(timeout=10.0)

        if not skip_verify:
            print()
            print("--- VERIFY round-trip ---")
            verify_round_trip(backend, chunk_bytes=small_chunk_bytes)
            print("  PASS: bytes read back match the pattern.")

        if not skip_bench:
            # Small chunks: per-call overhead dominates. Use the
            # production max_batch_size=4 so the scratch buffer
            # matches what GPUCacheContext registers.
            _run_bench_phase(
                backend,
                label="small chunks (overhead-dominated)",
                num_chunks=small_num_chunks,
                chunk_bytes=small_chunk_bytes,
                max_batch_size=4,
            )
            # Large chunks: bandwidth-dominated. max_batch_size=1
            # keeps the scratch buffer at one chunk, which matters
            # when chunks are GB-sized (4 × 2 GiB = 8 GiB VRAM is
            # often impractical).
            _run_bench_phase(
                backend,
                label="large chunks (bandwidth-dominated)",
                num_chunks=large_num_chunks,
                chunk_bytes=large_chunk_bytes,
                max_batch_size=1,
            )
    finally:
        backend.close()
        _stop_loop(loop, thread)


def _run_bench_phase(
    backend: GdsL1Backend,
    *,
    label: str,
    num_chunks: int,
    chunk_bytes: int,
    max_batch_size: int,
) -> None:
    """Run one bench phase and print its results."""
    chunk_mib = chunk_bytes / 1024 / 1024
    chunk_label = (
        f"{chunk_mib:.0f} MiB"
        if chunk_mib < 1024
        else f"{chunk_mib / 1024:.1f} GiB"
    )
    print()
    print(f"--- BENCH {label}: {num_chunks} × {chunk_label} ---")
    store_r, retrieve_r = benchmark(
        backend,
        num_chunks=num_chunks,
        chunk_bytes=chunk_bytes,
        max_batch_size=max_batch_size,
    )
    print(
        f"  STORE   : {store_r.total_mib:8.1f} MiB in {store_r.seconds:6.3f}s "
        f"= {store_r.mibs:8.1f} MiB/s"
    )
    r_mib = retrieve_r.total_mib
    r_sec = retrieve_r.seconds
    r_rate = retrieve_r.mibs
    print(
        f"  RETRIEVE: {r_mib:8.1f} MiB in {r_sec:6.3f}s = {r_rate:8.1f} MiB/s"
    )
