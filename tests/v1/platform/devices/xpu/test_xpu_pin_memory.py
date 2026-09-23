# SPDX-License-Identifier: Apache-2.0
"""Hardware tests for XPU SYCL host-memory registration."""

# Standard
import mmap

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform.devices.xpu import XpuDeviceSpec
from lmcache.v1.platform.devices.xpu.pin_memory import XpuPinMemoryBackend

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

pytestmark = pytest.mark.xpu

_PAGE_SIZE = mmap.PAGESIZE
_TRANSFER_CHUNK_SIZE = 64 << 20
_LARGE_RANGE_SIZE = (10 << 30) + 99


def _assert_xpu_round_trip(host_tensor: torch.Tensor) -> None:
    """Copy a host tensor to XPU and back, then verify its contents."""
    device_tensor = host_tensor.to(torch_device_type)
    round_trip = device_tensor.cpu()
    torch_dev.synchronize()

    if not torch.equal(round_trip, host_tensor):
        raise AssertionError("XPU round trip changed registered host-memory data")


@pytest.mark.parametrize(
    ("ptr_offset", "size"),
    [
        pytest.param(0, _PAGE_SIZE, id="aligned-ptr-aligned-size"),
        pytest.param(0, _PAGE_SIZE - 1, id="aligned-ptr-unaligned-size"),
        pytest.param(1, _PAGE_SIZE, id="unaligned-ptr-aligned-size"),
        pytest.param(1, _PAGE_SIZE - 1, id="unaligned-ptr-unaligned-size"),
    ],
)
def test_xpu_pin_memory_round_trip_with_alignment_combinations(
    ptr_offset: int,
    size: int,
) -> None:
    """Round-trip host ranges covering all pointer and size alignments."""
    host_mapping = mmap.mmap(-1, size + ptr_offset)
    host_tensor = torch.frombuffer(
        host_mapping,
        dtype=torch.uint8,
        count=size,
        offset=ptr_offset,
    )
    host_tensor.copy_(
        torch.arange(size, dtype=torch.int64).remainder_(251).to(torch.uint8)
    )
    spec = XpuDeviceSpec()

    try:
        assert (host_tensor.data_ptr() % _PAGE_SIZE == 0) == (ptr_offset == 0)
        assert (host_tensor.nbytes % _PAGE_SIZE == 0) == (size == _PAGE_SIZE)
        assert spec.pin_memory_backend is XpuPinMemoryBackend
        assert spec.is_pin_supported is True
        assert spec.pin_memory(host_tensor.data_ptr(), host_tensor.nbytes) is True

        try:
            _assert_xpu_round_trip(host_tensor)
        finally:
            assert spec.unpin_memory(host_tensor.data_ptr()) is True
    finally:
        del host_tensor
        host_mapping.close()


def test_xpu_pin_memory_round_trip_with_large_unaligned_range() -> None:
    """Round-trip a 10 GiB host range with unaligned pointer and size."""
    ptr_offset = 1
    host_mapping = mmap.mmap(-1, _LARGE_RANGE_SIZE + ptr_offset)
    host_tensor = torch.frombuffer(
        host_mapping,
        dtype=torch.uint8,
        count=_LARGE_RANGE_SIZE,
        offset=ptr_offset,
    )
    spec = XpuDeviceSpec()

    try:
        assert host_tensor.data_ptr() % _PAGE_SIZE != 0
        assert host_tensor.nbytes % _PAGE_SIZE != 0
        assert spec.pin_memory_backend is XpuPinMemoryBackend
        assert spec.is_pin_supported is True
        assert spec.pin_memory(host_tensor.data_ptr(), host_tensor.nbytes) is True

        try:
            for offset in range(0, host_tensor.nbytes, _TRANSFER_CHUNK_SIZE):
                chunk_size = min(
                    _TRANSFER_CHUNK_SIZE,
                    host_tensor.nbytes - offset,
                )
                host_chunk = host_tensor[offset : offset + chunk_size]
                host_chunk.fill_((offset // _TRANSFER_CHUNK_SIZE) % 251)

                _assert_xpu_round_trip(host_chunk)
                del host_chunk
                host_mapping.madvise(
                    mmap.MADV_DONTNEED,
                    offset,
                    chunk_size,
                )
        finally:
            assert spec.unpin_memory(host_tensor.data_ptr()) is True
    finally:
        del host_tensor
        host_mapping.close()
