# SPDX-License-Identifier: Apache-2.0
"""The torch-baseline copy on a real device, with the compiled ops absent.

The MP store path stages KV with a gather kernel on a non-blocking stream and
then asks ``lmcache_memcpy_async`` for the host copy. On the torch baseline that
call used to issue a synchronous ``cudaMemcpy`` on the legacy default stream,
which does not wait for a non-blocking stream, so the copy could read the
staging buffer before the kernel filled it. It now issues ``cudaMemcpyAsync`` on
the caller's stream and splits the copy at the host pin boundaries.

These tests exercise the real copy path on a device; ``-m "not cuda"`` keeps
them out of the CPU-only CI job.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.torch_ops import mem_kernels

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("requires a CUDA device", allow_module_level=True)

_MIB = 1024 * 1024
_PIN_CHUNK = 64 * _MIB  # LazyMemoryAllocator.PIN_CHUNK_SIZE


def _run_on_side_stream(fn, *_args) -> None:
    """Run ``fn`` inside a non-blocking stream, as the transfer context does."""
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fn(*_args)
    torch.cuda.synchronize()


def test_copy_waits_for_a_kernel_on_the_same_stream() -> None:
    """The copy must observe the fill queued before it on its stream."""
    size = 1 * _MIB
    device = torch.zeros(size, dtype=torch.uint8, device="cuda")
    host = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize()  # the zero-fill is on the default stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        device.fill_(0xAB)
        mem_kernels.lmcache_memcpy_async(
            host.data_ptr(),
            device.data_ptr(),
            size,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=_PIN_CHUNK,
        )
    torch.cuda.synchronize()

    assert int(host.min()) == 0xAB, "the copy ran before the fill completed"


def test_copy_across_pin_chunks_is_correct() -> None:
    """A copy spanning several 64 MiB pin chunks must not fail."""
    size = 3 * _PIN_CHUNK + 4 * _MIB
    device = torch.full((size,), 0x5A, dtype=torch.uint8, device="cuda")
    host = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize()  # the zero-fill is on the default stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        mem_kernels.lmcache_memcpy_async(
            host.data_ptr(),
            device.data_ptr(),
            size,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=0,
            host_buffer_alignments=_PIN_CHUNK,
        )
    torch.cuda.synchronize()

    assert int(host.min()) == 0x5A
    assert int(host.max()) == 0x5A


def test_offset_inside_a_pin_chunk_is_respected() -> None:
    """The first copy ends at the next boundary after ``host_buffer_offset``."""
    size = 2 * _MIB
    offset = 4 * _MIB
    device = torch.full((size,), 0x33, dtype=torch.uint8, device="cuda")
    host = torch.zeros(size, dtype=torch.uint8, pin_memory=True)
    torch.cuda.synchronize()  # the zero-fill is on the default stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        mem_kernels.lmcache_memcpy_async(
            host.data_ptr(),
            device.data_ptr(),
            size,
            mem_kernels.TransferDirection.D2H,
            host_buffer_offset=offset,
            host_buffer_alignments=_PIN_CHUNK,
        )
    torch.cuda.synchronize()

    assert int(host.min()) == 0x33


def test_h2d_follows_the_same_contract() -> None:
    """The symmetric direction: a fill on the stream, then the copy to device."""
    size = 1 * _MIB
    host = torch.full((size,), 0x7E, dtype=torch.uint8, pin_memory=True)
    device = torch.zeros(size, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()  # the zero-fill is on the default stream

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        mem_kernels.lmcache_memcpy_async(
            device.data_ptr(),
            host.data_ptr(),
            size,
            mem_kernels.TransferDirection.H2D,
            host_buffer_offset=0,
            host_buffer_alignments=_PIN_CHUNK,
        )
    torch.cuda.synchronize()

    assert int(device.min()) == 0x7E
