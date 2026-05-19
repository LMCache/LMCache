# SPDX-License-Identifier: Apache-2.0

"""Tests for the C++ CompletionRecorder and CompletionDispatcher."""

# Standard
import pickle
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.native_completion import (
    CompletionDispatcher,
    is_native_available,
    record_on_stream,
)

# Native path needs CUDA + c_ops with record_completion_on_stream.
# Fallback path tests run anywhere.
try:
    # Third Party
    import torch  # noqa: F401

    _has_cuda = torch.cuda.is_available()
except ImportError:
    _has_cuda = False

try:
    # First Party
    import lmcache.c_ops as lmc_ops

    _has_native_op = hasattr(lmc_ops, "record_completion_on_stream")
except ImportError:
    lmc_ops = None
    _has_native_op = False

native_only = pytest.mark.skipif(
    not (_has_cuda and _has_native_op),
    reason="requires CUDA and native record_completion_on_stream",
)

if _has_cuda and _has_native_op:
    import cupy  # noqa: E402


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


@native_only
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


@native_only
class TestDispatcher:
    """Integration tests for CompletionDispatcher (drain + handler dispatch)."""

    def test_dispatch_to_registered_handler(self, dispatcher, stream):
        seen: list[list] = []
        dispatcher.register("finish_write", seen.append)

        record_on_stream(stream, "finish_write", [b"k0", b"k1", b"k2"])
        stream.synchronize()

        # Wait for drain thread to dispatch
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.01)
        assert seen == [[b"k0", b"k1", b"k2"]]

    def test_unknown_kind_drops_payload(self, dispatcher, stream):
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


@native_only
class TestDeadlockRegression:
    """Regression for the GIL deadlock: the legacy
    ``stream.launch_host_func(python_fn, ...)`` path stalls; the C++
    callback runs without the GIL and must finish within the timeout."""

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

        assert ready.wait(
            timeout=10.0
        ), f"deadlock or drop: only {len(received)} of 200 dispatched"
        assert len(received) == 200


class TestFallbackPath:
    """``record_on_stream`` must delegate to
    ``stream.launch_host_func(fallback_handler, payload_list)`` when the
    native recorder isn't compiled in."""

    def test_falls_back_to_launch_host_func(self, monkeypatch):
        # First Party
        from lmcache.v1.multiprocess import native_completion

        monkeypatch.setattr(native_completion, "_has_native", False)

        recorded: list = []

        class FakeStream:
            def launch_host_func(self, fn, payload):
                recorded.append((fn, payload))

        def fallback(payload):
            pass

        stream = FakeStream()
        native_completion.record_on_stream(
            stream, "finish_write", [b"k0", b"k1"], fallback_handler=fallback
        )

        assert len(recorded) == 1
        fn, payload = recorded[0]
        assert fn is fallback
        assert payload == [b"k0", b"k1"]

    def test_raises_when_no_fallback_and_no_native(self, monkeypatch):
        # First Party
        from lmcache.v1.multiprocess import native_completion

        monkeypatch.setattr(native_completion, "_has_native", False)

        class FakeStream:
            def launch_host_func(self, fn, payload):
                raise AssertionError("should not be called when fallback is None")

        with pytest.raises(RuntimeError, match="native recorder unavailable"):
            native_completion.record_on_stream(
                FakeStream(), "finish_write", [b"k"], fallback_handler=None
            )
