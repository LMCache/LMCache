# SPDX-License-Identifier: Apache-2.0

"""Tests for the C++ CompletionRecorder and CompletionDispatcher."""

# Standard
import pickle
import threading
import time

# Third Party
import pytest

torch = pytest.importorskip("torch", reason="torch required")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

lmc_ops = pytest.importorskip("lmcache.c_ops", reason="lmcache.c_ops not built")
if not hasattr(lmc_ops, "record_completion_on_stream"):
    pytest.skip(
        "record_completion_on_stream not available", allow_module_level=True
    )

# Third Party
import cupy  # noqa: E402

# First Party
from lmcache.v1.multiprocess.native_completion import (  # noqa: E402
    CompletionDispatcher,
    is_native_available,
    record_on_stream,
)


@pytest.fixture()
def stream():
    s = cupy.cuda.Stream()
    yield s
    s.synchronize()


@pytest.fixture()
def dispatcher():
    d = CompletionDispatcher(drain_interval_seconds=0.001)
    d.start()
    yield d
    d.stop()


class TestRecordAndDrain:
    """Low-level tests on lmc_ops.record_completion_on_stream / drain."""

    def test_native_available(self):
        assert is_native_available()

    def test_drain_empty(self):
        completions = lmc_ops.drain_recorded_completions()
        assert completions == []

    def test_single_completion(self, stream):
        encoded = [pickle.dumps(b"key-1"), pickle.dumps(b"key-2")]
        lmc_ops.record_completion_on_stream(stream.ptr, "finish_write", encoded)
        stream.synchronize()

        completions = lmc_ops.drain_recorded_completions()
        assert len(completions) == 1
        kind, payload = completions[0]
        assert kind == "finish_write"
        assert [pickle.loads(p) for p in payload] == [b"key-1", b"key-2"]

    def test_many_completions_in_order(self, stream):
        for i in range(50):
            lmc_ops.record_completion_on_stream(
                stream.ptr, "finish_write", [pickle.dumps(i)]
            )
        stream.synchronize()

        completions = lmc_ops.drain_recorded_completions()
        assert len(completions) == 50
        for idx, (kind, payload) in enumerate(completions):
            assert kind == "finish_write"
            assert pickle.loads(payload[0]) == idx


class TestDispatcher:
    """Integration tests for CompletionDispatcher (drain + handler dispatch)."""

    def test_dispatch_to_registered_handler(self, dispatcher, stream):
        seen: list[list] = []
        dispatcher.register("finish_write", seen.append)

        record_on_stream(
            stream, "finish_write", [b"k0", b"k1", b"k2"]
        )
        stream.synchronize()

        # Wait for drain thread to dispatch
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.01)
        assert seen == [[b"k0", b"k1", b"k2"]]

    def test_unknown_kind_drops_payload(self, dispatcher, stream, caplog):
        # No handler registered for this kind
        record_on_stream(stream, "finish_unknown", [b"x"])
        stream.synchronize()
        time.sleep(0.1)
        assert dispatcher.dispatched_count() == 0

    def test_handler_exception_does_not_kill_thread(self, dispatcher, stream):
        calls: list[list] = []

        def handler(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise RuntimeError("boom")

        dispatcher.register("finish_write", handler)
        record_on_stream(stream, "finish_write", [b"a"])
        record_on_stream(stream, "finish_write", [b"b"])
        stream.synchronize()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(calls) < 2:
            time.sleep(0.01)
        assert calls == [[b"a"], [b"b"]]
        assert dispatcher.handler_exception_counts().get("finish_write", 0) == 1


class TestDeadlockRegression:
    """Reproduces the deadlock shape PR fix addresses.

    The original ``cupy_stream.launch_host_func(python_fn, ...)`` path
    deadlocked when the Python callback tried to acquire the GIL while
    the calling thread held both the GIL and the CUDA driver lock. With
    the C++ host callback the driver thread never touches the GIL, so
    the test below should complete well within the timeout.
    """

    def test_many_concurrent_records_no_deadlock(self, dispatcher, stream):
        received: list[list] = []
        ready = threading.Event()

        def handler(payload):
            received.append(payload)
            if len(received) >= 200:
                ready.set()

        dispatcher.register("finish_write", handler)

        for i in range(200):
            record_on_stream(stream, "finish_write", [pickle.dumps(i)])
        stream.synchronize()

        assert ready.wait(timeout=10.0), (
            f"deadlock or drop: only {len(received)} of 200 dispatched"
        )
        assert len(received) == 200
