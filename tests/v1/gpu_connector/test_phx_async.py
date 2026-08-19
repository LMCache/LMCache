# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Phoenix phxfs async wrapper (``_phx_async``).

These tests are pure: ``libphoenix.so`` is never loaded and no phxfs device
is opened. A fake library exercises the Python ctypes wrapper, including
device lifecycle, buffer registration (page-size alignment, where the page
size is whatever the library's ``phxfs_get_page_size`` reports -- never a
hardcoded constant; per-GPU device open, duplicate handling), the no-op
stream-registration surface, stream-ordered IO argument marshalling (byref
ctypes storage + raw stream handle) and error propagation, the fail-fast taken
when the library predates the stream API, handle lifecycle, and
``close_driver`` cleanup.

The real phxfs ABI and end-to-end DMA path are exercised on Phoenix
hardware via the GDS L1 tier (``--gds-l1-backend phx``).
"""

# Standard
from types import SimpleNamespace
from typing import Any, Optional

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector import _phx_async as pa


class _FakeLib:
    """Stand-in for ``libphoenix.so`` that records all symbol calls."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}
        self.events: list[tuple[str, ...]] = []
        # Default mirrors NVIDIA's 64 KiB; individual tests override it to
        # prove the wrapper derives alignment from the library, not a const.
        self.page_size = 64 * 1024
        self.map_mode = 0
        # When False, stream symbols raise AttributeError like a CDLL
        # without them (loading must then fail fast: no sync fallback).
        self.has_stream_api = True
        self.find_dev_results: dict[int, int] = {}
        self.open_rc = 0
        self.regmem_rc = 0
        self.deregmem_rc = 0
        self.read_stream_rc = 0
        self.write_stream_rc = 0
        # Optional override for the value written into bytes_done by the
        # fake stream call (None -> succeed with nbytes).
        self.read_stream_bd: Optional[int] = None
        self.write_stream_bd: Optional[int] = None

    def phxfs_get_page_size(self) -> int:
        self.calls.setdefault("phxfs_get_page_size", []).append(())
        return self.page_size

    def __getattr__(self, name: str) -> Any:
        if not self.has_stream_api and name in (
            "phxfs_read_stream",
            "phxfs_write_stream",
        ):
            raise AttributeError(name)

        def _record(*args: Any) -> Any:
            self.calls.setdefault(name, []).append(args)
            self.events.append((name,))
            if name == "phxfs_find_dev":
                return self.find_dev_results.get(args[0], 0)
            if name == "phxfs_open":
                return self.open_rc
            if name == "phxfs_regmem":
                args[3]._obj.value = 0xFFFF0000  # target_addr out-param
                return self.regmem_rc
            if name == "phxfs_deregmem":
                return self.deregmem_rc
            if name == "phxfs_get_map_mode":
                return self.map_mode
            if name == "phxfs_read_stream":
                # (fd, buf, nb_p, bo_p, fo_p, bd_p, stream)
                bd = self.read_stream_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.read_stream_rc
            if name == "phxfs_write_stream":
                bd = self.write_stream_bd
                args[5]._obj.value = bd if bd is not None else args[2]._obj.value
                return self.write_stream_rc
            return 0

        return _record


@pytest.fixture(autouse=True)
def _fake_lib(monkeypatch: pytest.MonkeyPatch) -> _FakeLib:
    """Replace the lazy-loaded CDLL and reset process-global phx state."""
    lib = _FakeLib()
    monkeypatch.setattr(pa, "_lib", lib)
    monkeypatch.setattr(pa, "_page_size", 0)
    monkeypatch.setattr(pa, "_devices", {})
    monkeypatch.setattr(pa, "_reg_bases", [])
    monkeypatch.setattr(pa, "_reg_entries", [])
    # register_buffer falls back to the current CUDA ordinal when the
    # tensor's device index is unset; CI may have no GPU.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return lib


def _gpu_tensor(
    ptr: int = 0x100000, nbytes: int = 64 * 1024, cuda_index: int = 0
) -> SimpleNamespace:
    """Return a GPU-tensor stand-in accepted by the wrapper."""
    return SimpleNamespace(
        is_cuda=True,
        data_ptr=lambda: ptr,
        numel=lambda: nbytes,
        element_size=lambda: 1,
        device=SimpleNamespace(index=cuda_index),
    )


class TestApiSurface:
    def test_exports_required_names(self) -> None:
        required = (
            "AsyncHandle",
            "Submission",
            "close_driver",
            "register_handle",
            "deregister_handle",
            "register_buffer",
            "deregister_buffer",
            "register_stream",
            "deregister_stream",
        )
        for name in required:
            assert hasattr(pa, name), f"_phx_async missing {name!r}"


class TestLibLoading:
    def test_get_lib_raises_when_library_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(pa.ctypes.util, "find_library", lambda _: None)

        def _raise(path: str) -> Any:
            raise OSError(f"cannot open {path}")

        monkeypatch.setattr(pa.ctypes, "CDLL", _raise)
        with pytest.raises(OSError, match="libphoenix"):
            pa._get_lib()

    def test_get_lib_prefers_find_library_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(
            pa.ctypes.util, "find_library", lambda _: "/opt/lib/libphoenix.so.1"
        )

        class _Lib:
            def __getattr__(self, name: str) -> Any:
                return lambda *args: 0

        def _load(path: str) -> Any:
            loaded.append(path)
            return _Lib()

        monkeypatch.setattr(pa.ctypes, "CDLL", _load)
        pa._get_lib()
        assert loaded == ["/opt/lib/libphoenix.so.1"]


class TestBufferRegistration:
    def test_rejects_non_gpu_tensor(self) -> None:
        with pytest.raises(ValueError, match="CUDA or ROCm"):
            pa.register_buffer(SimpleNamespace(is_cuda=False))

    def test_rejects_empty_tensor(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            pa.register_buffer(_gpu_tensor(nbytes=0))

    def test_opens_device_and_queries_page_size_from_library(
        self, _fake_lib: _FakeLib
    ) -> None:
        pa.register_buffer(_gpu_tensor(ptr=0x200000, nbytes=4096, cuda_index=3))
        assert _fake_lib.calls["phxfs_find_dev"] == [(3,)]
        assert _fake_lib.calls["phxfs_open"] == [(0,)]
        # The alignment granularity must come from the library interface.
        assert "phxfs_get_page_size" in _fake_lib.calls
        device, addr, length, _target = _fake_lib.calls["phxfs_regmem"][0]
        assert device == 0
        assert addr == 0x200000
        assert length == _fake_lib.page_size

    @pytest.mark.parametrize(
        "page_size",
        [4 * 1024, 64 * 1024, 128 * 1024, 2 * 1024 * 1024],
    )
    def test_registration_alignment_follows_library_page_size(
        self, _fake_lib: _FakeLib, page_size: int
    ) -> None:
        """Alignment is derived from ``phxfs_get_page_size``, not a constant."""
        _fake_lib.page_size = page_size
        cases = [
            (1, page_size),  # sub-page rounds up to one page
            (page_size, page_size),  # exact single page stays
            (2 * page_size, 2 * page_size),  # exact multiple stays
            (page_size + 1, 2 * page_size),  # one page + 1 byte rounds up
        ]
        for i, (nbytes, expected_len) in enumerate(cases):
            # Fresh base pointer per case: re-registering the same base with
            # a different length is (correctly) rejected as a duplicate.
            ptr = 0x200000 + i * 0x100000
            _fake_lib.calls.clear()
            pa.register_buffer(_gpu_tensor(ptr=ptr, nbytes=nbytes))
            _device, _addr, length, _target = _fake_lib.calls["phxfs_regmem"][0]
            assert length == expected_len, f"page_size={page_size}, nbytes={nbytes}"

    def test_find_dev_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.find_dev_results = {0: -19}
        with pytest.raises(RuntimeError, match="phxfs_find_dev"):
            pa.register_buffer(_gpu_tensor())

    def test_open_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.open_rc = -16
        with pytest.raises(RuntimeError, match="phxfs_open"):
            pa.register_buffer(_gpu_tensor())

    def test_regmem_failure_raises(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.regmem_rc = -12
        with pytest.raises(RuntimeError, match="phxfs_regmem"):
            pa.register_buffer(_gpu_tensor())
        # A failed registration must not leave a table entry behind.
        assert pa._reg_bases == []

    def test_device_opened_once_per_gpu(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.find_dev_results = {0: 0, 1: 5}
        pa.register_buffer(_gpu_tensor(ptr=0x100000, cuda_index=0))
        pa.register_buffer(_gpu_tensor(ptr=0x200000, cuda_index=0))
        pa.register_buffer(_gpu_tensor(ptr=0x300000, cuda_index=1))
        assert _fake_lib.calls["phxfs_find_dev"] == [(0,), (1,)]
        assert _fake_lib.calls["phxfs_open"] == [(0,), (5,)]

    def test_duplicate_register_same_base_and_len_passes_through(
        self, _fake_lib: _FakeLib
    ) -> None:
        tensor = _gpu_tensor(ptr=0x200000)
        pa.register_buffer(tensor)
        pa.register_buffer(tensor)
        # phxfs reference-counts the exact duplicate; the wrapper records
        # every successful call so register/deregister stay symmetric.
        assert len(_fake_lib.calls["phxfs_regmem"]) == 2
        assert pa._reg_bases == [0x200000, 0x200000]

    def test_deregister_calls_deregmem_with_aligned_length(
        self, _fake_lib: _FakeLib
    ) -> None:
        pa.register_buffer(_gpu_tensor(ptr=0x200000, nbytes=4096))
        pa.deregister_buffer(_gpu_tensor(ptr=0x200000))
        device, addr, length = _fake_lib.calls["phxfs_deregmem"][0]
        assert device == 0
        assert addr == 0x200000
        # Deregistration must pass back the same library-aligned length.
        assert length == _fake_lib.page_size
        assert pa._reg_bases == []

    def test_deregister_unregistered_buffer_is_noop(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_buffer(_gpu_tensor(ptr=0x999000))
        assert "phxfs_deregmem" not in _fake_lib.calls

    def test_deregister_failure_raises_and_keeps_entry(
        self, _fake_lib: _FakeLib
    ) -> None:
        pa.register_buffer(_gpu_tensor(ptr=0x200000))
        _fake_lib.deregmem_rc = -22
        with pytest.raises(RuntimeError, match="phxfs_deregmem"):
            pa.deregister_buffer(_gpu_tensor(ptr=0x200000))
        # The entry is kept so a retry can attempt the deregmem again.
        assert pa._reg_bases == [0x200000]


class TestStreamRegistration:
    """phxfs has no stream registration; the surface functions are no-ops."""

    def test_register_stream_is_noop(self, _fake_lib: _FakeLib) -> None:
        pa.register_stream(0xABC)
        assert _fake_lib.calls == {}

    def test_deregister_stream_is_noop(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_stream(0xABC)
        assert _fake_lib.calls == {}

    def test_missing_stream_symbols_fail_fast(
        self, _fake_lib: _FakeLib, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _fake_lib.has_stream_api = False
        monkeypatch.setattr(pa, "_lib", None)
        monkeypatch.setattr(pa.ctypes.util, "find_library", lambda _: "libphoenix.so")

        def _load(path: str) -> Any:
            return _fake_lib

        monkeypatch.setattr(pa.ctypes, "CDLL", _load)
        with pytest.raises(RuntimeError, match="stream-ordered API"):
            pa._get_lib()


class TestHandleRegistration:
    def test_register_handle_returns_fd(self, _fake_lib: _FakeLib) -> None:
        assert pa.register_handle(42) == 42

    def test_deregister_handle_is_noop(self, _fake_lib: _FakeLib) -> None:
        pa.deregister_handle(42)
        assert _fake_lib.calls == {}


class TestSubmission:
    def test_stores_arguments_and_result(self) -> None:
        submission = pa.Submission(size=4096, file_offset=8192, buf_offset=512)
        assert submission._size.value == 4096
        assert submission._file_offset.value == 8192
        assert submission._buf_offset.value == 512
        assert submission.bytes_done == 0
        submission._bytes_done.value = 4096
        assert submission.bytes_done == 4096


class TestAsyncHandleIO:
    """Stream-ordered submissions (default path with a stream-capable lib)."""

    def _handle(self) -> pa.AsyncHandle:
        return pa.AsyncHandle.from_fd(
            fd=5,
            handle=5,  # phx handle == fd
            path="/mnt/nvme/lmcache_gds_slab.bin",
            writable=True,
        )

    @staticmethod
    def _registered(raw_stream: int = 0x9) -> None:
        del raw_stream  # phxfs needs no stream registration
        pa.register_buffer(_gpu_tensor(ptr=0x300000, nbytes=2 * 64 * 1024))

    def test_read_async_submits_stream_ordered(self, _fake_lib: _FakeLib) -> None:
        self._registered()
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=8192,
            buf_offset=512,
            raw_stream=0x9,
        )
        # One stream-ordered submission, no synchronous fallback call.
        assert len(_fake_lib.calls["phxfs_read_stream"]) == 1
        assert "phxfs_read" not in _fake_lib.calls
        fd, buf, nb_p, bo_p, fo_p, bd_p, stream = _fake_lib.calls["phxfs_read_stream"][
            0
        ]
        assert fd == 5
        assert buf == 0x300000
        # The submission's ctypes storage is handed in by reference.
        assert nb_p._obj is submission._size
        assert bo_p._obj is submission._buf_offset
        assert fo_p._obj is submission._file_offset
        assert bd_p._obj is submission._bytes_done
        assert nb_p._obj.value == 4096
        assert bo_p._obj.value == 512
        assert fo_p._obj.value == 8192
        assert stream.value == 0x9
        # The fake completed the transfer synchronously.
        assert submission.bytes_done == 4096

    def test_read_async_submits_across_multiple_devices(
        self, _fake_lib: _FakeLib
    ) -> None:
        # The device is resolved inside the library from the buffer; the
        # wrapper no longer maps buf -> device (two devices registered,
        # both buffers submit fine through the same path).
        _fake_lib.find_dev_results = {0: 0, 1: 5}
        pa.register_buffer(_gpu_tensor(ptr=0x100000, cuda_index=0))
        pa.register_buffer(_gpu_tensor(ptr=0x300000, cuda_index=1))
        self._handle().read_async(
            buf_base=0x100000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
        )
        self._handle().read_async(
            buf_base=0x300000, size=4096, file_offset=0, buf_offset=0, raw_stream=0x9
        )
        bufs = [call[1] for call in _fake_lib.calls["phxfs_read_stream"]]
        assert bufs == [0x100000, 0x300000]

    def test_read_async_submission_error_raises(self, _fake_lib: _FakeLib) -> None:
        self._registered()
        _fake_lib.read_stream_rc = -22  # submission-level failure
        with pytest.raises(RuntimeError, match="phxfs_read_stream"):
            self._handle().read_async(
                buf_base=0x300000,
                size=4096,
                file_offset=0,
                buf_offset=0,
                raw_stream=0x9,
            )

    def test_read_async_defers_dma_error_to_bytes_done(
        self, _fake_lib: _FakeLib
    ) -> None:
        self._registered()
        _fake_lib.read_stream_bd = -28  # ENOSPC during the transfer
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        # cuFile contract: the submission is accepted; the failure surfaces
        # in bytes_done after the stream sync -- no synchronous raise.
        assert submission.bytes_done == -28

    def test_read_async_unregistered_buffer_passes_through(
        self, _fake_lib: _FakeLib
    ) -> None:
        # The device resolution lives in the library now: an unregistered
        # buffer is submitted as-is (the library treats a registration
        # miss as a plain CPU address, like the synchronous API).
        submission = self._handle().read_async(
            buf_base=0x400000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert len(_fake_lib.calls["phxfs_read_stream"]) == 1
        assert submission.bytes_done == 4096

    def test_io_needs_no_stream_registration(self, _fake_lib: _FakeLib) -> None:
        # No register_stream call anywhere: an unregistered stream is
        # submitted on first use (no-registration model, like cuFile).
        pa.register_buffer(_gpu_tensor(ptr=0x300000))
        submission = self._handle().read_async(
            buf_base=0x300000,
            size=4096,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert len(_fake_lib.calls["phxfs_read_stream"]) == 1
        assert submission.bytes_done == 4096

    def test_write_async_submits_stream_ordered(self, _fake_lib: _FakeLib) -> None:
        self._registered()
        submission = self._handle().write_async(
            buf_base=0x300000,
            size=2048,
            file_offset=512,
            buf_offset=128,
            raw_stream=0x9,
        )
        # Stream-ordered write: no host-side stream synchronization.
        assert len(_fake_lib.calls["phxfs_write_stream"]) == 1
        assert "phxfs_write" not in _fake_lib.calls
        fd, buf, nb_p, bo_p, fo_p, bd_p, stream = _fake_lib.calls["phxfs_write_stream"][
            0
        ]
        assert fd == 5
        assert buf == 0x300000
        assert nb_p._obj.value == 2048
        assert bo_p._obj.value == 128
        assert fo_p._obj.value == 512
        assert stream.value == 0x9
        assert submission.bytes_done == 2048

    def test_write_async_defers_dma_error_to_bytes_done(
        self, _fake_lib: _FakeLib
    ) -> None:
        self._registered()
        _fake_lib.write_stream_bd = -5  # EIO during the transfer
        submission = self._handle().write_async(
            buf_base=0x300000,
            size=2048,
            file_offset=0,
            buf_offset=0,
            raw_stream=0x9,
        )
        assert submission.bytes_done == -5


class TestAsyncHandleLifecycle:
    def test_constructor_opens_and_registers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[tuple[str, int]] = []
        registered: list[int] = []

        def open_device(path: str, flags: int) -> int:
            opened.append((path, flags))
            return 33

        def register_device(fd: int) -> int:
            registered.append(fd)
            return fd

        monkeypatch.setattr(pa.os, "open", open_device)
        monkeypatch.setattr(pa, "register_handle", register_device)
        handle = pa.AsyncHandle("/mnt/nvme/lmcache_gds_slab.bin")
        assert opened == [("/mnt/nvme/lmcache_gds_slab.bin", pa.os.O_RDWR)]
        assert registered == [33]
        assert handle.fd == 33
        assert handle.path == "/mnt/nvme/lmcache_gds_slab.bin"
        assert handle.writable is True

    def test_constructor_closes_fd_on_registration_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed: list[int] = []
        monkeypatch.setattr(pa.os, "open", lambda path, flags: 33)
        monkeypatch.setattr(
            pa,
            "register_handle",
            lambda fd: (_ for _ in ()).throw(RuntimeError("register failed")),
        )
        monkeypatch.setattr(pa.os, "close", closed.append)
        with pytest.raises(RuntimeError, match="register failed"):
            pa.AsyncHandle("/mnt/nvme/lmcache_gds_slab.bin")
        assert closed == [33]

    def test_close_deregisters_handle_and_closes_fd_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deregistered: list[int] = []
        closed: list[int] = []
        monkeypatch.setattr(pa, "deregister_handle", deregistered.append)
        monkeypatch.setattr(pa.os, "close", closed.append)
        handle = pa.AsyncHandle.from_fd(
            fd=5,
            handle=5,
            path="/mnt/nvme/lmcache_gds_slab.bin",
            writable=True,
        )
        handle.close()
        handle.close()
        assert deregistered == [5]
        assert closed == [5]
        assert handle.fd == -1


class TestCloseDriver:
    def test_releases_registrations_and_devices(self, _fake_lib: _FakeLib) -> None:
        _fake_lib.find_dev_results = {0: 0, 1: 5}
        pa.register_buffer(_gpu_tensor(ptr=0x100000, cuda_index=0))
        pa.register_buffer(_gpu_tensor(ptr=0x300000, cuda_index=1))
        pa.close_driver()
        assert len(_fake_lib.calls["phxfs_deregmem"]) == 2
        closed = [call[0] for call in _fake_lib.calls["phxfs_close"]]
        assert closed == [0, 5]
        assert pa._reg_bases == []
        assert pa._devices == {}

    def test_without_state_is_noop(self, _fake_lib: _FakeLib) -> None:
        pa.close_driver()
        assert "phxfs_deregmem" not in _fake_lib.calls
        assert "phxfs_close" not in _fake_lib.calls
