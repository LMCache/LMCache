# SPDX-License-Identifier: Apache-2.0
"""OpenEulerNDSLibrary tests with a fake ``libnds.so`` CDLL."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
import ctypes
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gds_backends import _nds_libnds as nl
from lmcache.v1.gpu_connector.gds_backends._nds_libnds import (
    NDS_API_VERSION,
    OpenEulerNDSLibrary,
    _IoSegment,
)


class _FakeCDLL:
    """Stand-in for ``libnds.so`` recording symbol calls."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {}
        self.submit_rc = 1
        self.getevents_rc = 1
        self.completion_res = 0
        self._allocated_ctxs: list[nl._NdsIoCtx] = []
        self.getevents_hook: Callable[..., int] | None = None

    def __getattr__(self, name: str) -> Any:
        def _call(*args: Any) -> int:
            self.calls.setdefault(name, []).append(args)
            if name == "nds_init":
                args[0]._obj.version = NDS_API_VERSION
                return 0
            if name == "nds_io_new_ctx":
                ctx_obj = nl._NdsIoCtx()
                self._allocated_ctxs.append(ctx_obj)
                ptr = ctypes.pointer(ctx_obj)
                ctypes.memmove(
                    ctypes.byref(args[1]._obj),
                    ctypes.byref(ptr),
                    ctypes.sizeof(ptr),
                )
                return 0
            if name == "nds_io_submit":
                return self.submit_rc
            if name == "nds_io_getevents":
                if self.getevents_hook is not None:
                    return self.getevents_hook(*args)
                args[3][0].res = self.completion_res
                return self.getevents_rc
            return 0

        return _call


@pytest.fixture
def fake_cdll(monkeypatch: pytest.MonkeyPatch) -> _FakeCDLL:
    cdll = _FakeCDLL()
    monkeypatch.setattr(nl.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(nl.ctypes, "CDLL", lambda _path, mode=0: cdll)
    return cdll


@pytest.fixture
def lib(fake_cdll: _FakeCDLL) -> OpenEulerNDSLibrary:
    del fake_cdll
    instance = OpenEulerNDSLibrary(library_path="/fake/libnds.so")
    instance.driver_open()
    return instance


def _mock_current_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream_id: int = 7,
    launch_fn: Callable[[Callable[[], None]], None] | None = None,
) -> SimpleNamespace:
    """Point the current stream at ``stream_id`` and stub ``aclrtLaunchHostFunc``.

    ``launch_fn`` still receives the zero-arg host callback the tests invoke.
    ``None`` makes the launch return an error.
    """
    nl._pending_callbacks.clear()
    stream = SimpleNamespace(npu_stream=stream_id)
    npu = SimpleNamespace(
        is_available=lambda: True,
        current_stream=lambda: stream,
        launched_streams=[],
    )

    def launch(
        stream_ptr: ctypes.c_void_p,
        trampoline: Callable[[int], None],
        user_data: ctypes.c_void_p,
    ) -> int:
        npu.launched_streams.append(stream_ptr.value)
        if launch_fn is None:
            return 1
        payload_address = user_data.value
        if payload_address is None:
            raise AssertionError("host callback payload must be non-null")
        launch_fn(lambda: trampoline(payload_address))
        return 0

    monkeypatch.setattr(nl, "_aclrt_launch_host_func", lambda: launch)
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    return npu


class TestContextLifecycle:
    def test_register_stream_creates_one_context_and_deregisters_once(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        lib.register_stream(7)
        lib.register_stream(7)
        assert len(fake_cdll.calls["nds_io_new_ctx"]) == 1
        lib.deregister_stream(7)
        lib.deregister_stream(7)
        assert len(fake_cdll.calls["nds_io_destroy_ctx"]) == 1

    def test_register_stream_requires_open_driver(self, fake_cdll: _FakeCDLL) -> None:
        del fake_cdll
        closed_lib = OpenEulerNDSLibrary(library_path="/fake/libnds.so")
        with pytest.raises(RuntimeError, match="driver is not open"):
            closed_lib.register_stream(7)

    def test_driver_close_destroys_remaining_contexts(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        lib.register_stream(1)
        lib.register_stream(2)
        lib.driver_close()
        assert len(fake_cdll.calls["nds_io_destroy_ctx"]) == 2
        assert len(fake_cdll.calls["nds_exit"]) == 1


class TestBlockingIO:
    def test_io_submits_and_reaps_the_same_context(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        lib.register_stream(7)
        ctx = lib._stream_ctxs[7]
        assert lib._io(ctx, 3, 0x1000, 4096, 0x8000, 128, False) == 4096
        submit_ctx, count, cb_ref = fake_cdll.calls["nds_io_submit"][0]
        assert ctypes.addressof(submit_ctx.contents) == ctypes.addressof(ctx.contents)
        assert count == 1
        assert cb_ref._obj.offset == 0x8000
        assert cb_ref._obj.obj.u.fd == 3
        reap_ctx, min_nr, max_nr, _events, timeout = fake_cdll.calls[
            "nds_io_getevents"
        ][0]
        assert ctypes.addressof(reap_ctx.contents) == ctypes.addressof(ctx.contents)
        assert (min_nr, max_nr, timeout) == (1, 1, None)

    def test_io_raises_on_negative_completion(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        fake_cdll.completion_res = -5
        lib.register_stream(7)
        with pytest.raises(OSError):
            lib._io(lib._stream_ctxs[7], 3, 0x1000, 4096, 0, 0, False)

    def test_submit_failure_does_not_call_getevents(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        fake_cdll.submit_rc = 0
        lib.register_stream(7)
        with pytest.raises(OSError):
            lib._io(lib._stream_ctxs[7], 3, 0x1000, 4096, 0, 0, False)
        assert "nds_io_getevents" not in fake_cdll.calls

    def test_two_streams_block_on_separate_contexts(
        self, lib: OpenEulerNDSLibrary, fake_cdll: _FakeCDLL
    ) -> None:
        lib.register_stream(1)
        lib.register_stream(2)
        ctx_a, ctx_b = lib._stream_ctxs[1], lib._stream_ctxs[2]
        addr_a = ctypes.addressof(ctx_a.contents)
        addr_b = ctypes.addressof(ctx_b.contents)
        release = threading.Event()
        entered = {addr_a: threading.Event(), addr_b: threading.Event()}

        def wait_both(
            ctx: Any, _min_nr: int, _max_nr: int, events_out: Any, _timeout: Any
        ) -> int:
            entered[ctypes.addressof(ctx.contents)].set()
            assert release.wait(timeout=2.0)
            events_out[0].res = 0
            return 1

        fake_cdll.getevents_hook = wait_both
        results: dict[int, int] = {}

        def run(stream_id: int, ctx: Any) -> None:
            results[stream_id] = lib._io(ctx, 3, 0x1000, 4096, 0, 0, False)

        threads = [
            threading.Thread(target=run, args=(1, ctx_a)),
            threading.Thread(target=run, args=(2, ctx_b)),
        ]
        for thread in threads:
            thread.start()
        assert entered[addr_a].wait(timeout=2.0)
        assert entered[addr_b].wait(timeout=2.0)
        release.set()
        for thread in threads:
            thread.join(timeout=2.0)
        assert results == {1: 4096, 2: 4096}


class TestStreamOrderedAsyncIO:
    def test_read_async_defers_result_until_host_callback(
        self,
        lib: OpenEulerNDSLibrary,
        fake_cdll: _FakeCDLL,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: list[Callable[[], None]] = []
        npu = _mock_current_stream(monkeypatch, stream_id=7, launch_fn=captured.append)
        lib.register_stream(7)
        result = ctypes.c_int64(0)
        segments = [_IoSegment(device_offset=0x8000, length=4096, buf_offset=0)]
        lib.read_async(3, 0x1000, segments, result, 7)
        lib.read_async(3, 0x1000, segments, result, 7)
        assert npu.launched_streams == [7, 7]
        assert result.value == 0
        assert "nds_io_submit" not in fake_cdll.calls
        captured[0]()
        captured[1]()
        assert result.value == 4096
        assert len(fake_cdll.calls["nds_io_submit"]) == 2

    def test_host_callbacks_preserve_issue_order_across_interleaved_work(
        self,
        lib: OpenEulerNDSLibrary,
        fake_cdll: _FakeCDLL,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stream runs each callback after the work issued before it.

        Stream 7 is issued as kernel, read, kernel, two-segment write, kernel,
        then a read on stream 11, then another read and kernel on stream 7.
        Draining stream 11 first must not run any stream 7 callback.
        """
        nl._pending_callbacks.clear()
        stream = SimpleNamespace(npu_stream=7)
        queues: dict[int, list[Callable[[], None]]] = {7: [], 11: []}

        def launch(
            stream_ptr: ctypes.c_void_p,
            trampoline: Callable[[int], None],
            user_data: ctypes.c_void_p,
        ) -> int:
            payload_address = user_data.value
            if payload_address is None:
                raise AssertionError("host callback payload must be non-null")
            stream_id = stream_ptr.value
            if stream_id is None:
                raise AssertionError("NPU stream must be non-null")
            queues[stream_id].append(lambda: trampoline(payload_address))
            return 0

        monkeypatch.setattr(nl, "_aclrt_launch_host_func", lambda: launch)
        monkeypatch.setattr(
            torch,
            "npu",
            SimpleNamespace(is_available=lambda: True, current_stream=lambda: stream),
            raising=False,
        )
        lib.register_stream(7)
        lib.register_stream(11)
        trace: list[object] = []

        def issue(stream_id: int, operation: Callable[[], None]) -> None:
            stream.npu_stream = stream_id
            operation()

        def marker(stream_id: int, name: str) -> None:
            def mark() -> None:
                trace.append(name)

            queues[stream_id].append(mark)

        def drain(stream_id: int, count: int) -> None:
            for step in queues[stream_id][:count]:
                before = len(fake_cdll.calls.get("nds_io_submit", []))
                step()
                submits = fake_cdll.calls.get("nds_io_submit", [])
                for args in submits[before:]:
                    cb = args[2]._obj
                    kind = "write" if cb.opcode == nl.NDS_IO_OP_PWRITE else "read"
                    trace.append(
                        (
                            kind,
                            cb.obj.u.fd,
                            cb.offset,
                            cb.iov.contents.buf_len,
                            cb.iov.contents.buf_addr,
                        )
                    )
            del queues[stream_id][:count]

        read_a = ctypes.c_int64(0)
        write_b = ctypes.c_int64(0)
        read_other = ctypes.c_int64(0)
        read_c = ctypes.c_int64(0)
        marker(7, "kernel-a")
        issue(
            7,
            lambda: lib.read_async(3, 0x1000, [_IoSegment(0x1000, 4096, 0)], read_a, 7),
        )
        marker(7, "kernel-b")
        issue(
            7,
            lambda: lib.write_async(
                4,
                0x2000,
                [
                    _IoSegment(0x2000, 4096, 128),
                    _IoSegment(0x3000, 8192, 4224),
                ],
                write_b,
                7,
            ),
        )
        marker(7, "kernel-c")
        issue(
            11,
            lambda: lib.read_async(
                5, 0x4000, [_IoSegment(0x4000, 4096, 0)], read_other, 11
            ),
        )
        issue(
            7,
            lambda: lib.read_async(
                3, 0x5000, [_IoSegment(0x5000, 8192, 64)], read_c, 7
            ),
        )
        marker(7, "kernel-d")

        assert "nds_io_submit" not in fake_cdll.calls
        assert (read_a.value, write_b.value, read_other.value, read_c.value) == (
            0,
            0,
            0,
            0,
        )

        drain(11, 1)
        assert trace == [("read", 5, 0x4000, 4096, 0x4000)]
        assert read_other.value == 4096
        assert (read_a.value, write_b.value, read_c.value) == (0, 0, 0)

        drain(7, 3)
        assert trace == [
            ("read", 5, 0x4000, 4096, 0x4000),
            "kernel-a",
            ("read", 3, 0x1000, 4096, 0x1000),
            "kernel-b",
        ]
        assert read_a.value == 4096
        assert write_b.value == 0

        drain(7, len(queues[7]))
        assert trace == [
            ("read", 5, 0x4000, 4096, 0x4000),
            "kernel-a",
            ("read", 3, 0x1000, 4096, 0x1000),
            "kernel-b",
            ("write", 4, 0x2000, 4096, 0x2000 + 128),
            ("write", 4, 0x3000, 8192, 0x2000 + 4224),
            "kernel-c",
            ("read", 3, 0x5000, 8192, 0x5000 + 64),
            "kernel-d",
        ]
        assert (read_a.value, write_b.value, read_other.value, read_c.value) == (
            4096,
            12288,
            4096,
            8192,
        )
        assert queues == {7: [], 11: []}
        assert nl._pending_callbacks == {}

    @pytest.mark.parametrize(
        ("operation", "expected_opcode"),
        [("read", nl.NDS_IO_OP_PREAD), ("write", nl.NDS_IO_OP_PWRITE)],
    )
    def test_async_methods_dispatch_every_segment_symmetrically(
        self,
        lib: OpenEulerNDSLibrary,
        fake_cdll: _FakeCDLL,
        monkeypatch: pytest.MonkeyPatch,
        operation: str,
        expected_opcode: int,
    ) -> None:
        _mock_current_stream(monkeypatch, stream_id=7, launch_fn=lambda cb: cb())
        lib.register_stream(7)
        result = ctypes.c_int64(0)
        segments = [
            _IoSegment(device_offset=0xA000, length=4096, buf_offset=128),
            _IoSegment(device_offset=0xB000, length=8192, buf_offset=4224),
        ]
        submit = lib.read_async if operation == "read" else lib.write_async
        submit(9, 0x1000, segments, result, 7)
        assert result.value == 12288
        submits = fake_cdll.calls["nds_io_submit"]
        assert len(submits) == 2
        cb0, cb1 = submits[0][2]._obj, submits[1][2]._obj
        assert cb0.opcode == expected_opcode
        assert (cb0.obj.u.fd, cb0.offset, cb0.iov.contents.buf_len) == (9, 0xA000, 4096)
        assert cb0.iov.contents.buf_addr == 0x1000 + 128
        assert cb1.opcode == expected_opcode
        assert (cb1.obj.u.fd, cb1.offset, cb1.iov.contents.buf_len) == (9, 0xB000, 8192)
        assert cb1.iov.contents.buf_addr == 0x1000 + 4224

    def test_async_rejects_wrong_current_stream(
        self, lib: OpenEulerNDSLibrary, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mock_current_stream(monkeypatch, stream_id=8, launch_fn=lambda cb: cb())
        result = ctypes.c_int64(0)
        segments = [_IoSegment(device_offset=0, length=4096, buf_offset=0)]
        with pytest.raises(RuntimeError, match="current NPU stream"):
            lib.read_async(3, 0x1000, segments, result, 7)

    def test_async_requires_torch_stream_host_callback(
        self,
        lib: OpenEulerNDSLibrary,
        fake_cdll: _FakeCDLL,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_current_stream(monkeypatch, stream_id=7, launch_fn=None)
        lib.register_stream(7)
        result = ctypes.c_int64(0)
        segments = [_IoSegment(device_offset=0, length=4096, buf_offset=0)]
        with pytest.raises(RuntimeError, match="aclrtLaunchHostFunc"):
            lib.read_async(3, 0x1000, segments, result, 7)
        assert result.value == 0
        assert nl._pending_callbacks == {}
        assert "nds_io_submit" not in fake_cdll.calls

    def test_async_maps_missing_context_and_io_failure_to_negative_result(
        self,
        lib: OpenEulerNDSLibrary,
        fake_cdll: _FakeCDLL,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _mock_current_stream(monkeypatch, stream_id=7, launch_fn=lambda cb: cb())
        segments = [_IoSegment(device_offset=0x8000, length=4096, buf_offset=0)]

        missing_ctx_result = ctypes.c_int64(0)
        lib.read_async(3, 0x1000, segments, missing_ctx_result, 7)
        assert missing_ctx_result.value == -1

        lib.register_stream(7)
        fake_cdll.completion_res = -5
        io_err_result = ctypes.c_int64(0)
        lib.read_async(3, 0x1000, segments, io_err_result, 7)
        assert io_err_result.value == -5


class TestStructLayout:
    """Guard the ``libnds`` ctypes structures against ``nds_api.h`` (LP64)."""

    def test_io_vec_and_event_sizes(self) -> None:
        assert ctypes.sizeof(nl._NdsIoVec) == 16
        assert ctypes.sizeof(nl._NdsIoObj) == 16
        assert ctypes.sizeof(nl._NdsIoCb) == 72
        assert ctypes.sizeof(nl._NdsIoEvent) == 48
        assert nl._NdsIoEvent.res.offset == 8
