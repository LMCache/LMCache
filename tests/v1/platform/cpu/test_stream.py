# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.platform.cpu.stream.MockExternalStream``.

These tests exercise the pure-Python fallback without requiring CUDA
or cupy.
"""

# Standard
import gc
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.platform import cpu as _cpu_pkg  # re-export anchor
from lmcache.v1.platform.cpu import stream as _stream_mod
from lmcache.v1.platform.cpu.stream import MockExternalStream
from lmcache.v1.platform.stream import ExternalStreamLike

# Silence "imported but unused" — we only need the import to prove the
# sub-package exists as a public surface callers can rely on.
_ = _cpu_pkg


class TestMockExternalStream:
    """Behavior of the pure-Python mock stream."""

    def test_preserves_valid_stream_pointer(self):
        """A non-zero caller pointer is kept verbatim on ``ptr``."""
        stream = MockExternalStream(0xCAFEBABE)
        try:
            assert stream.ptr == 0xCAFEBABE
        finally:
            stream._shutdown()

    def test_fallback_ptr_is_non_zero_when_no_handle(self):
        """A zero pointer is replaced with a unique non-zero id."""
        stream = MockExternalStream(0)
        try:
            assert stream.ptr != 0
            assert stream.ptr == id(stream)
        finally:
            stream._shutdown()

    def test_conforms_to_external_stream_protocol(self):
        """Mock stream satisfies the ``ExternalStreamLike`` protocol."""
        stream = MockExternalStream(1)
        try:
            # Runtime protocol check: ``Protocol`` without
            # ``runtime_checkable`` cannot be used in ``isinstance``,
            # so fall back to attribute presence.
            assert hasattr(stream, "ptr")
            assert callable(stream.launch_host_func)
            # Structural typing sanity — a function accepting the
            # protocol accepts the mock.

            def _accept(s: ExternalStreamLike) -> int:
                return s.ptr

            assert _accept(stream) == stream.ptr
        finally:
            stream._shutdown()

    def test_launch_host_func_executes_callback(self):
        """Enqueued callbacks are invoked exactly once on the worker."""
        stream = MockExternalStream(0)
        try:
            done = threading.Event()
            seen = []

            def cb(arg):
                seen.append(arg)
                done.set()

            stream.launch_host_func(cb, 42)
            assert done.wait(timeout=2.0)
            assert seen == [42]
        finally:
            stream._shutdown()

    def test_fifo_order_is_preserved(self):
        """Callbacks run serially in submission order."""
        stream = MockExternalStream(0)
        try:
            order = []
            done = threading.Event()
            n = 32

            def make_cb(i):
                def cb(_):
                    order.append(i)
                    if i == n - 1:
                        done.set()

                return cb

            for i in range(n):
                stream.launch_host_func(make_cb(i), None)
            assert done.wait(timeout=2.0)
            assert order == list(range(n))
        finally:
            stream._shutdown()

    def test_callback_exception_is_swallowed(self):
        """Worker survives a throwing callback and keeps draining."""
        stream = MockExternalStream(0)
        try:
            done = threading.Event()

            def bad(_):
                raise RuntimeError("boom")

            def good(_):
                done.set()

            stream.launch_host_func(bad, None)
            stream.launch_host_func(good, None)
            assert done.wait(timeout=2.0)
        finally:
            stream._shutdown()

    def test_shutdown_is_idempotent(self):
        """Calling ``_shutdown`` twice does not raise."""
        stream = MockExternalStream(0)
        stream._shutdown()
        stream._shutdown()

    def test_post_shutdown_fifo_then_sync(self):
        """After shutdown, pending tasks drain before sync fallback runs."""
        stream = MockExternalStream(0)
        order = []
        gate = threading.Event()

        def slow(_):
            # Block the worker so the task is still in-flight when
            # shutdown is invoked.
            gate.wait(timeout=2.0)
            order.append("async")

        stream.launch_host_func(slow, None)
        # Kick off shutdown in a side thread; _shutdown joins the
        # worker, so we must release the gate shortly afterwards.
        shutdown_thread = threading.Thread(target=stream._shutdown, daemon=True)
        shutdown_thread.start()
        # Release the worker so it can finish its pending task.
        time.sleep(0.05)
        gate.set()
        shutdown_thread.join(timeout=2.0)
        assert not shutdown_thread.is_alive()

        # New callbacks after shutdown run synchronously on the caller,
        # but only after the worker has fully drained.
        stream.launch_host_func(lambda _: order.append("sync"), None)
        assert order == ["async", "sync"]

    def test_weakref_finalizer_no_leak_on_gc(self):
        """Dropping the last reference lets GC reclaim the instance.

        Regression test: registering a bound method with ``atexit``
        used to pin every stream for the whole process lifetime.
        """
        stream = MockExternalStream(0)
        ref = _stream_mod.weakref.ref(stream)
        del stream
        gc.collect()
        assert ref() is None, "mock stream should be garbage-collected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
