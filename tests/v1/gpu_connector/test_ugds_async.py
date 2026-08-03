# SPDX-License-Identifier: Apache-2.0
"""Tests for the uGDS async backend (_ugds_async.py).

Verifies that _ugds_async exports the same API surface as _cufile_async
and that the uGDS DMA path works end-to-end on hardware.

The API surface tests need no hardware. The roundtrip test writes to a raw
uGDS device and so is opt-in via ``UGDS_TEST_DEVICE``; it skips when unset.
Point it only at a scratch device, since its first blocks are overwritten.

Run standalone:
    PYTHONPATH=. python -m pytest \
        tests/v1/gpu_connector/test_ugds_async.py --noconftest -v

    UGDS_TEST_DEVICE=/dev/ugds_drv0 PYTHONPATH=. python -m pytest \
        tests/v1/gpu_connector/test_ugds_async.py --noconftest -v
"""

# Standard
import ctypes
import importlib.util
import os

# Third Party
import pytest
import torch

# Direct import of the module under test, avoiding lmcache's heavy import chain.
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "lmcache",
    "v1",
    "gpu_connector",
    "_ugds_async.py",
)
_MODULE_PATH = os.path.normpath(_MODULE_PATH)


def _load_ugds_async():
    spec = importlib.util.spec_from_file_location("_ugds_async", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ugds_test_device() -> str:
    """Return the opt-in uGDS test device, or an empty string when unset.

    The roundtrip test writes to the raw device, destroying whatever occupies
    the first blocks. It is therefore opt-in via ``UGDS_TEST_DEVICE`` rather
    than probing for any ``/dev/ugds_drv*``: on a host with several bound
    devices, probing could overwrite one the caller did not intend to sacrifice.
    """
    device = os.environ.get("UGDS_TEST_DEVICE", "")
    if not device or not os.path.exists(device):
        return ""
    if not torch.cuda.is_available():
        return ""
    try:
        ctypes.CDLL("libugds.so")
    except OSError:
        return ""
    return device


requires_ugds = pytest.mark.skipif(
    not _ugds_test_device(),
    reason=(
        "set UGDS_TEST_DEVICE to a scratch uGDS device (e.g. /dev/ugds_drv0); "
        "also needs CUDA and libugds.so. The test overwrites the device."
    ),
)


class TestApiSurface:
    """_ugds_async must export the same names as _cufile_async."""

    def test_exports_all_required_names(self):
        ua = _load_ugds_async()
        required = [
            "AsyncHandle",
            "Submission",
            "close_driver",
            "register_handle",
            "deregister_handle",
            "register_buffer",
            "deregister_buffer",
            "register_stream",
            "deregister_stream",
        ]
        for name in required:
            assert hasattr(ua, name), f"_ugds_async missing '{name}'"

    def test_submission_has_bytes_done(self):
        ua = _load_ugds_async()
        sub = ua.Submission(size=4096, file_offset=0, buf_offset=0)
        assert hasattr(sub, "bytes_done")
        assert sub.bytes_done == 0

    def test_async_handle_has_read_write_close(self):
        ua = _load_ugds_async()
        assert callable(getattr(ua.AsyncHandle, "read_async", None))
        assert callable(getattr(ua.AsyncHandle, "write_async", None))
        assert callable(getattr(ua.AsyncHandle, "close", None))


@requires_ugds
class TestUgdsRoundtrip:
    """End-to-end DMA through uGDS: write a pattern, read it back."""

    def test_write_read_4kb(self):
        ua = _load_ugds_async()
        device_path = _ugds_test_device()
        fd = os.open(device_path, os.O_RDWR)
        try:
            ugds_handle = ua.register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        handle = ua.AsyncHandle.from_fd(fd, ugds_handle, device_path, writable=True)
        try:
            size = 4096
            buf = torch.empty(size, dtype=torch.uint8, device="cuda")
            ua.register_buffer(buf)

            stream = torch.cuda.current_stream()
            raw_stream = stream.cuda_stream
            ua.register_stream(raw_stream)

            # Write pattern
            buf.fill_(0xAB)
            torch.cuda.synchronize()
            handle.write_async(
                buf.data_ptr(),
                size,
                file_offset=0,
                buf_offset=0,
                raw_stream=raw_stream,
            )
            torch.cuda.synchronize()

            # Read back
            buf.zero_()
            torch.cuda.synchronize()
            sub_r = handle.read_async(
                buf.data_ptr(),
                size,
                file_offset=0,
                buf_offset=0,
                raw_stream=raw_stream,
            )
            torch.cuda.synchronize()

            expected = torch.full((size,), 0xAB, dtype=torch.uint8)
            assert torch.equal(buf.cpu(), expected), "4KB roundtrip data mismatch"
            assert sub_r.bytes_done == size

            ua.deregister_stream(raw_stream)
            ua.deregister_buffer(buf)
        finally:
            handle.close()
