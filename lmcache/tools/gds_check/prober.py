# SPDX-License-Identifier: Apache-2.0
"""Probe + benchmark implementation for ``lmcache tool gds-check``.

Three phases, each independently skippable from the CLI:

- ``info``: read-only host inspection — fstype under ``gds_path``,
  whether ``nvidia-fs`` is loaded, cuFile 4 KiB alignment
  compatibility for the requested chunk size. Catches setup mistakes
  before any I/O happens.
- ``verify``: write a known pattern through
  ``GdsSlabAllocator.cufile_write_from``, read it back through
  ``cufile_read_into``, byte-compare the result. Same code path as
  the production data path; if this passes, the GDS L1 backend works
  on this host.
- ``bench``: store + retrieve N × chunk MiB and report MiB/s for
  each direction. Useful for comparing hardware (the absolute numbers
  matter much less than the delta between a compat-mode host and an
  ``nvidia-fs``-enabled one).

All chunk allocations route through the backend's single slab file,
so the bench measures the production path exactly: one cuFile handle
register at backend startup, per-chunk reads/writes as offset I/O
inside that slab.
"""

# Standard
from dataclasses import dataclass
import os
import shutil
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.distributed.gds_l1 import GdsSlabAllocator

logger = init_logger(__name__)


@dataclass
class HostInfo:
    """Snapshot of the host's GDS-relevant state, returned by :func:`probe_host`."""

    gds_path: str
    fstype: str
    nvidia_fs_loaded: bool
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


def _object_key(seed: int) -> ObjectKey:
    """Deterministic ObjectKey for repeatable benchmark runs."""
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\x00" * 28,
        model_name="lmcache-tool-gds-check",
        kv_rank=0,
    )


def _layout_for(chunk_bytes: int) -> MemoryLayoutDesc:
    """Build a 1-D uint8 layout descriptor of ``chunk_bytes`` bytes."""
    shape = torch.Size([chunk_bytes])
    return MemoryLayoutDesc(shapes=[shape], dtypes=[torch.uint8])


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
    from lmcache.v1.storage_backend.gds_backend import get_fstype

    os.makedirs(gds_path, exist_ok=True)
    fstype = get_fstype(gds_path)
    nvidia_fs_loaded = os.path.exists("/proc/driver/nvidia-fs/stats")

    return HostInfo(
        gds_path=gds_path,
        fstype=fstype,
        nvidia_fs_loaded=nvidia_fs_loaded,
        chunk_bytes=chunk_bytes,
        chunk_aligned_4kib=(chunk_bytes % 4096 == 0),
    )


def verify_round_trip(
    backend: GdsSlabAllocator, chunk_bytes: int, pattern_val: int = 0xCD
) -> None:
    """Write a deterministic pattern, read it back, fail loudly on mismatch.

    Uses the same ``GdsSlabAllocator`` code path as production —
    ``create_memory_obj`` reserves a slab region, ``cufile_write_from``
    DMAs into it, ``cufile_read_into`` DMAs back.

    Args:
        backend: An already-constructed :class:`GdsSlabAllocator`.
        chunk_bytes: Size of the test chunk in bytes (must be 4 KiB
            aligned).
        pattern_val: Byte value to fill the test chunk with.

    Raises:
        RuntimeError: If the bytes read back do not match what was
            written, or if the slab is too small for ``chunk_bytes``.
    """
    buf = torch.empty(chunk_bytes, dtype=torch.uint8, device="cuda:0")
    backend.cufile_io.register_gpu_buffer(buf)
    try:
        buf.fill_(pattern_val)
        torch_dev.synchronize()

        key = _object_key(seed=0xCAFE)
        mo = backend.create_memory_obj(
            key=key,
            layout_desc=_layout_for(chunk_bytes),
        )
        if mo is None:
            raise RuntimeError(
                f"GDS round-trip verify: slab too small to allocate {chunk_bytes} "
                "bytes. Increase --gds-l1-slab-size-gb."
            )
        backend.cufile_write_from(mo, buf)

        buf.zero_()
        torch_dev.synchronize()
        backend.cufile_read_into(mo, buf)
        torch_dev.synchronize()

        expected = torch.full((chunk_bytes,), pattern_val, dtype=torch.uint8)
        if not torch.equal(buf.cpu(), expected):
            raise RuntimeError(
                "GDS round-trip mismatch: bytes read back do not match the "
                f"written pattern (0x{pattern_val:02x}). Most likely a "
                "cuFile compat-mode / nvidia-fs setup issue."
            )
    finally:
        backend.cufile_io.deregister_gpu_buffer()


def benchmark(
    backend: GdsSlabAllocator,
    num_chunks: int,
    chunk_bytes: int,
    max_batch_size: int = 4,
) -> tuple[BenchResult, BenchResult]:
    """Store then retrieve ``num_chunks × chunk_bytes`` and report timings.

    The store / retrieve loop mirrors the MP server's dispatch shape:
    a small registered staging buffer with ``max_batch_size`` slots,
    one ``cufile_write_from`` / ``cufile_read_into`` per chunk, single
    ``cudaStreamSynchronize`` at the end.

    Args:
        backend: An already-constructed :class:`GdsSlabAllocator`.
        num_chunks: Number of distinct chunks to write / read.
        chunk_bytes: Bytes per chunk (4 KiB aligned).
        max_batch_size: Number of staging slots in the registered
            buffer. Defaults to 4, matching
            ``GPUCacheContext.max_batch_size``.

    Returns:
        ``(store_result, retrieve_result)``.
    """
    tmp_buf = torch.empty(
        max_batch_size * chunk_bytes, dtype=torch.uint8, device="cuda:0"
    )
    backend.cufile_io.register_gpu_buffer(tmp_buf)
    try:
        pattern = torch.full((chunk_bytes,), 0xAB, dtype=torch.uint8, device="cuda:0")

        slot_views = [
            tmp_buf[i * chunk_bytes : (i + 1) * chunk_bytes]
            for i in range(max_batch_size)
        ]
        layout = _layout_for(chunk_bytes)
        mem_objs = []
        for i in range(num_chunks):
            key = _object_key(seed=i)
            mo = backend.create_memory_obj(key=key, layout_desc=layout)
            if mo is None:
                raise RuntimeError(
                    f"GDS bench: slab exhausted after {i} of {num_chunks} chunks. "
                    f"Increase --gds-l1-slab-size-gb or reduce the workload."
                )
            mem_objs.append(mo)

        torch_dev.synchronize()
        t0 = time.perf_counter()
        for i, mo in enumerate(mem_objs):
            slot = slot_views[i % max_batch_size]
            slot.copy_(pattern, non_blocking=False)
            backend.cufile_write_from(mo, slot)
        torch_dev.synchronize()
        store_secs = time.perf_counter() - t0

        for s in slot_views:
            s.zero_()
        torch_dev.synchronize()
        t0 = time.perf_counter()
        for i, mo in enumerate(mem_objs):
            slot = slot_views[i % max_batch_size]
            backend.cufile_read_into(mo, slot)
        torch_dev.synchronize()
        retrieve_secs = time.perf_counter() - t0
    finally:
        backend.cufile_io.deregister_gpu_buffer()

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
    use_direct_io: bool,
    skip_verify: bool,
    skip_bench: bool,
) -> None:
    """Top-level entry point invoked by the CLI wrapper.

    Performs the host-info probe unconditionally, then optionally the
    verify and bench phases. The bench phase runs two workload shapes
    back-to-back:

    - **small chunks** (default 64 × 2 MiB): per-call-overhead-dominated.
      Tells you how much time is spent in Python dispatch relative to
      the actual I/O — relevant for KV-cache-shaped access patterns.
    - **large chunks** (default 8 × 256 MiB): bandwidth-dominated.
      Tells you how close the path gets to peak NVMe / cuFile
      throughput once per-call overhead is amortized.

    The verify phase uses the small chunk size (faster) and is the
    byte-for-byte round-trip check in :func:`verify_round_trip`.

    Args:
        gds_path: Directory to use as the slab root. Wiped fresh.
        small_num_chunks: Chunk count for the small phase.
        small_chunk_bytes: Per-chunk size for the small phase (4 KiB aligned).
        large_num_chunks: Chunk count for the large phase.
        large_chunk_bytes: Per-chunk size for the large phase (4 KiB aligned).
        use_gds: If ``False``, force the POSIX fallback.
        use_direct_io: If ``True``, open the slab with ``O_DIRECT``.
        skip_verify: Skip the round-trip correctness check.
        skip_bench: Skip the throughput benchmark.
    """
    info = probe_host(gds_path, small_chunk_bytes)

    print("=" * 60)
    print("LMCache GDS L1 platform check")
    print("=" * 60)
    print(f"  gds_path             : {info.gds_path}")
    print(f"  fstype               : {info.fstype}")
    print(f"  nvidia-fs loaded     : {info.nvidia_fs_loaded}")
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

    # Wipe gds_path.
    if os.path.isdir(gds_path):
        shutil.rmtree(gds_path)
    os.makedirs(gds_path, exist_ok=True)

    # Size the slab for the largest phase that will run.
    needed_bytes = max(
        small_num_chunks * small_chunk_bytes if not skip_bench else 0,
        large_num_chunks * large_chunk_bytes if not skip_bench else 0,
        small_chunk_bytes if not skip_verify else 0,
    )
    # Round up to GiB with 10% headroom.
    slab_size_gb = max(1.0, (needed_bytes * 1.1) / (1 << 30))

    config = GdsL1Config(
        gds_path=gds_path,
        gds_path_sharding="by_gpu",
        use_gds=use_gds,
        use_direct_io=use_direct_io,
        slab_size_gb=slab_size_gb,
    )
    backend = GdsSlabAllocator(config=config, dst_device="cuda:0")
    try:
        if not skip_verify:
            print()
            print("--- VERIFY round-trip ---")
            verify_round_trip(backend, chunk_bytes=small_chunk_bytes)
            print("  PASS: bytes read back match the pattern.")
            # Free the verify chunk back to the slab pool.
            for k in list(backend._index.keys()):  # noqa: SLF001
                mo = backend.create_memory_obj_from_index(k)
                if mo is not None:
                    backend.free(mo)

        if not skip_bench:
            _run_bench_phase(
                backend,
                label="small chunks (overhead-dominated)",
                num_chunks=small_num_chunks,
                chunk_bytes=small_chunk_bytes,
                max_batch_size=4,
            )
            # Free the small phase's chunks back to the slab pool.
            for k in list(backend._index.keys()):  # noqa: SLF001
                mo = backend.create_memory_obj_from_index(k)
                if mo is not None:
                    backend.free(mo)
            _run_bench_phase(
                backend,
                label="large chunks (bandwidth-dominated)",
                num_chunks=large_num_chunks,
                chunk_bytes=large_chunk_bytes,
                max_batch_size=1,
            )
    finally:
        backend.close()


def _run_bench_phase(
    backend: GdsSlabAllocator,
    *,
    label: str,
    num_chunks: int,
    chunk_bytes: int,
    max_batch_size: int,
) -> None:
    """Run one bench phase and print its results.

    Catches ``ValueError`` raised by ``register_gpu_buffer`` when the
    requested staging buffer exceeds the nvidia-fs slab cap (16 MiB
    on the reference host). The phase is skipped with a printed note
    rather than aborting the whole run — the small-chunks phase
    still produces a useful number on the same invocation.
    """
    chunk_mib = chunk_bytes / 1024 / 1024
    chunk_label = (
        f"{chunk_mib:.0f} MiB" if chunk_mib < 1024 else f"{chunk_mib / 1024:.1f} GiB"
    )
    print()
    print(f"--- BENCH {label}: {num_chunks} × {chunk_label} ---")
    try:
        store_r, retrieve_r = benchmark(
            backend,
            num_chunks=num_chunks,
            chunk_bytes=chunk_bytes,
            max_batch_size=max_batch_size,
        )
    except ValueError as e:
        print(f"  SKIPPED: {e}")
        return
    print(
        f"  STORE   : {store_r.total_mib:8.1f} MiB in {store_r.seconds:6.3f}s "
        f"= {store_r.mibs:8.1f} MiB/s"
    )
    r_mib = retrieve_r.total_mib
    r_sec = retrieve_r.seconds
    r_rate = retrieve_r.mibs
    print(f"  RETRIEVE: {r_mib:8.1f} MiB in {r_sec:6.3f}s = {r_rate:8.1f} MiB/s")
