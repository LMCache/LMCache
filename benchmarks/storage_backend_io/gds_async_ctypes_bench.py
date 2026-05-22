# SPDX-License-Identifier: Apache-2.0
"""Direct-ctypes async cuFile benchmark.

The existing GdsL1Backend uses kvikio's ``pread`` / ``pwrite``
(IOFuture, kvikio's internal thread pool). That preserves the
sync-helper contract but in compat-mode it ran slower than the bare
``cufile`` package's sync API + a Python thread pool — kvikio's
per-call dispatch overhead is non-trivial when the work itself is
just a CPU-bounce memcpy.

This script measures a third option: declare the C-level
``cuFileReadAsync`` / ``cuFileWriteAsync`` symbols ourselves via
ctypes on top of the ``libcufile.so`` that the ``cufile`` Python
package already loads, then issue N async reads / writes and stream-
sync once at the end. That removes:

- kvikio's Python dispatch + future bookkeeping (about 60 µs/op in
  compat mode benchmarks),
- the per-call CUDA stream sync that ``raw_read_async +
  check_bytes_done`` does,

leaving only the libcufile dispatch + CUDA stream serialization.

This is **an experiment**, not the production code path. We keep
the pointer-to-host-value parameters (``size_p``, ``file_offset_p``,
``bufPtr_offset_p``, ``bytes_done_p``) alive in a Python list for the
duration of the stream operation so the values they point to remain
valid until the read actually executes on the GPU.

Run::

    python benchmarks/storage_backend_io/gds_async_ctypes_bench.py

By default it writes 256 × 2 MiB files to ``$HOME/gds_async_bench/``
via kvikio, then reads them three different ways:

1. kvikio.CuFile.pread + IOFuture.get   (current production path)
2. kvikio.CuFile.raw_read_async + per-call check_bytes_done
3. Our ctypes ``cuFileReadAsync`` + single stream sync at the end
"""

# Standard
import argparse
import ctypes
import os
import shutil
import time

# Third Party
from cufile.bindings import (
    CUfileError,
    cuFileHandleDeregister,
    cuFileHandleRegister,
    libcufile,
)
import kvikio
import torch

# --- cuFile async API ctypes declarations ----------------------------
#
# CUfileError_t cuFileReadAsync(
#     CUfileHandle_t fh,
#     void *bufPtr_base,
#     size_t *size_p,
#     off_t  *file_offset_p,
#     off_t  *bufPtr_offset_p,
#     ssize_t *bytes_read_p,
#     CUstream stream
# );

# libcufile already loads. Declare argtypes/restype on the raw symbols.
libcufile.cuFileReadAsync.argtypes = [
    ctypes.c_void_p,  # fh (CUfileHandle_t)
    ctypes.c_void_p,  # bufPtr_base
    ctypes.POINTER(ctypes.c_size_t),  # size_p
    ctypes.POINTER(ctypes.c_int64),  # file_offset_p (off_t)
    ctypes.POINTER(ctypes.c_int64),  # bufPtr_offset_p
    ctypes.POINTER(ctypes.c_int64),  # bytes_read_p (ssize_t)
    ctypes.c_void_p,  # stream (CUstream)
]
libcufile.cuFileReadAsync.restype = CUfileError

libcufile.cuFileWriteAsync.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_void_p,
]
libcufile.cuFileWriteAsync.restype = CUfileError

libcufile.cuFileStreamRegister.argtypes = [
    ctypes.c_void_p,  # stream
    ctypes.c_uint,  # flags
]
libcufile.cuFileStreamRegister.restype = CUfileError

libcufile.cuFileStreamDeregister.argtypes = [ctypes.c_void_p]
libcufile.cuFileStreamDeregister.restype = CUfileError


def _check(err: CUfileError, op: str) -> None:
    if err.err != 0:
        raise RuntimeError(
            f"{op} failed: cuFileError(err={err.err}, cu_err={err.cu_err})"
        )


# --- File handle helpers ---------------------------------------------


def open_cufile_handle(path: str, write: bool = False) -> tuple[int, object]:
    """Open the file via POSIX and register it with cuFile.

    Uses the ``cufile.bindings.cuFileHandleRegister`` wrapper, which
    builds the ``CUfileDescr`` from a raw fd and returns a
    ``CUfileHandle_t``. Caller must close via
    :func:`close_cufile_handle`.
    """
    flags = os.O_RDWR if write else os.O_RDONLY
    fd = os.open(path, flags)
    try:
        handle = cuFileHandleRegister(fd)
    except Exception:
        os.close(fd)
        raise
    return fd, handle


def close_cufile_handle(fd: int, handle: object) -> None:
    cuFileHandleDeregister(handle)
    os.close(fd)


# --- Benchmark drivers -----------------------------------------------


def _make_test_files(
    root: str, n: int, chunk_bytes: int, pattern_val: int = 0xAB
) -> list[str]:
    """Write N files via kvikio.pwrite so the read-path benchmark has
    real data to read. Returns the list of file paths in stable order.
    """
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    pattern = torch.full(
        (chunk_bytes,), pattern_val, dtype=torch.uint8, device="cuda:0"
    )
    kvikio.memory_register(pattern)
    paths = []
    for i in range(n):
        p = os.path.join(root, f"chunk_{i:06d}.bin")
        with kvikio.CuFile(p, "w") as cf:
            fut = cf.pwrite(pattern, chunk_bytes, 0)
            fut.get()
        paths.append(p)
    kvikio.memory_deregister(pattern)
    return paths


def bench_kvikio_pread(paths: list[str], buf: torch.Tensor, chunk_bytes: int) -> float:
    """Read all paths through kvikio.pread + IOFuture.get (per call)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        with kvikio.CuFile(p, "r") as cf:
            view = buf.view(torch.uint8)[i * chunk_bytes : (i + 1) * chunk_bytes]
            fut = cf.pread(view, chunk_bytes, 0)
            fut.get()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_kvikio_raw_async_per_call(
    paths: list[str], buf: torch.Tensor, chunk_bytes: int
) -> float:
    """kvikio.raw_read_async + check_bytes_done per call (current code)."""
    stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        with kvikio.CuFile(p, "r") as cf:
            fut = cf.raw_read_async(buf, stream, chunk_bytes, 0, i * chunk_bytes)
            fut.check_bytes_done()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_ctypes_async(paths: list[str], buf: torch.Tensor, chunk_bytes: int) -> float:
    """Issue all reads via cuFileReadAsync then stream-sync once.

    Opens handles inside the timed loop (matching the kvikio-pread
    baseline, which opens fresh CuFile per call). The pointer-to-host
    parameters are kept in lists so they outlive the stream op.
    """
    # Register the stream once. Same CUDA stream is used throughout.
    cuda_stream_handle = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _check(
        libcufile.cuFileStreamRegister(cuda_stream_handle, 0),
        "cuFileStreamRegister",
    )

    sizes = [ctypes.c_size_t(chunk_bytes) for _ in paths]
    file_offsets = [ctypes.c_int64(0) for _ in paths]
    dev_offsets = [ctypes.c_int64(i * chunk_bytes) for i in range(len(paths))]
    bytes_read = [ctypes.c_int64(0) for _ in paths]

    base = ctypes.c_void_p(buf.data_ptr())

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    handles: list[tuple[int, ctypes.c_void_p]] = []
    try:
        for i, p in enumerate(paths):
            fd, h = open_cufile_handle(p)
            handles.append((fd, h))
            _check(
                libcufile.cuFileReadAsync(
                    h,
                    base,
                    ctypes.byref(sizes[i]),
                    ctypes.byref(file_offsets[i]),
                    ctypes.byref(dev_offsets[i]),
                    ctypes.byref(bytes_read[i]),
                    cuda_stream_handle,
                ),
                "cuFileReadAsync",
            )
        # Single stream sync for the whole batch.
        torch.cuda.current_stream().synchronize()
    finally:
        elapsed = time.perf_counter() - t0
        for fd, h in handles:
            close_cufile_handle(fd, h)
        libcufile.cuFileStreamDeregister(cuda_stream_handle)
    # Sanity check (out of timed region): expected nbytes per read.
    # In cuFile *compat mode* (nvidia-fs not loaded) the async API
    # returns ``-EIO`` (-5) per chunk — the bytes never landed. The
    # call site returns 0 from cuFileReadAsync but bytes_read_p gets
    # the negative status. Surface this as a clear message rather than
    # a confusing "expected vs got" mismatch.
    failed = [
        (i, br.value) for i, br in enumerate(bytes_read) if br.value != chunk_bytes
    ]
    if failed:
        n_eio = sum(1 for _, v in failed if v == -5)
        raise RuntimeError(
            f"cuFileReadAsync produced {len(failed)} failed reads "
            f"(of {len(bytes_read)}): "
            f"e.g. chunk {failed[0][0]} bytes_read={failed[0][1]}. "
            f"{n_eio} of them are -EIO — this is the expected failure "
            f"when nvidia-fs is not loaded; the cuFile async API "
            f"requires the real GDS driver. Compat-mode users must use "
            f"kvikio.pread/pwrite or the sync cuFile API instead."
        )
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.expanduser("~/gds_async_bench"))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--chunk-mib", type=int, default=2)
    args = parser.parse_args()

    chunk_bytes = args.chunk_mib * 1024 * 1024
    total_mib = args.n * args.chunk_mib

    print(
        f"GDS async ctypes bench: {args.n} chunks × {args.chunk_mib} MiB "
        f"= {total_mib} MiB total per pass"
    )
    print(f"root: {args.root}")
    print()
    print("Writing seed files via kvikio.pwrite ...")
    paths = _make_test_files(args.root, args.n, chunk_bytes)

    # Read buffer: one slot per chunk so all reads target distinct
    # device offsets — that's what a real batched pipeline would do.
    # Use kvikio.memory_register so the buffer is visible to both
    # kvikio's tracking (for the kvikio benches) and to libcufile (for
    # the ctypes bench — they share the same underlying registration).
    buf = torch.empty(args.n * chunk_bytes, dtype=torch.uint8, device="cuda:0")
    kvikio.memory_register(buf)
    try:
        # Warm up page cache + cuFile state.
        bench_kvikio_pread(paths, buf, chunk_bytes)

        for name, fn in [
            ("kvikio.pread + IOFuture.get (per call)", bench_kvikio_pread),
            (
                "kvikio.raw_read_async + check_bytes_done (per call)",
                bench_kvikio_raw_async_per_call,
            ),
            ("ctypes cuFileReadAsync + 1 stream sync", bench_ctypes_async),
        ]:
            buf.zero_()
            try:
                secs = fn(paths, buf, chunk_bytes)
            except RuntimeError as e:
                print(f"  {name}")
                print(f"    SKIPPED: {e}")
                continue
            print(f"  {name}")
            print(f"    {total_mib} MiB in {secs:.3f}s = {total_mib / secs:.1f} MiB/s")
    finally:
        kvikio.memory_deregister(buf)


if __name__ == "__main__":
    main()
