# SPDX-License-Identifier: Apache-2.0

"""Tests for the C++ CompletionRecorder and DeviceHostFuncDispatcher."""

# Standard
from unittest.mock import MagicMock
import threading
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)

pytestmark = pytest.mark.cuda


if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip(
        "requires available CUDA runtime",
        allow_module_level=True,
    )

try:
    # First Party
    import lmcache.cuda_ops as cuda_ops
except ImportError:
    pytest.skip(
        "requires lmcache.cuda_ops",
        allow_module_level=True,
    )

if not hasattr(cuda_ops, "record_completion_on_stream"):
    pytest.skip(
        "requires native record_completion_on_stream",
        allow_module_level=True,
    )

cupy = pytest.importorskip("cupy", reason="requires cupy")


@pytest.fixture()
def stream():
    s = cupy.cuda.Stream()
    yield s
    s.synchronize()


@pytest.fixture()
def dispatcher():
    d = DeviceHostFuncDispatcher(drain_interval_seconds=0.001)
    d.start()
    yield d
    d.stop()


class TestRecordAndDrain:
    """Low-level tests on cuda_ops.record_completion_on_stream / drain."""

    def test_drain_empty(self):
        completions = cuda_ops.drain_recorded_completions()
        assert completions == []

    def test_single_completion_payload_is_bytes(self, stream):
        # drain must hand back py::bytes, not utf-8-decoded str — msgpack
        # output is arbitrary bytes including invalid utf-8.
        encoded = msgspec.msgpack.encode([b"key-1", b"key-2"])
        cuda_ops.record_completion_on_stream(stream.ptr, "finish_write", encoded)
        stream.synchronize()

        completions = cuda_ops.drain_recorded_completions()
        assert len(completions) == 1
        kind, payload = completions[0]
        assert kind == "finish_write"
        assert isinstance(payload, bytes)
        assert msgspec.msgpack.decode(payload, type=list[bytes]) == [
            b"key-1",
            b"key-2",
        ]

    def test_many_completions_in_order(self, stream):
        for i in range(50):
            cuda_ops.record_completion_on_stream(
                stream.ptr, "finish_write", msgspec.msgpack.encode(i)
            )
        stream.synchronize()

        completions = cuda_ops.drain_recorded_completions()
        assert len(completions) == 50
        for idx, (kind, payload) in enumerate(completions):
            assert kind == "finish_write"
            assert msgspec.msgpack.decode(payload, type=int) == idx


class TestDispatcher:
    """Integration tests for DeviceHostFuncDispatcher (drain + dispatch)."""

    def test_dispatch_to_registered_handler(self, dispatcher, stream):
        seen: list[list[bytes]] = []
        dispatcher.register("finish_write", seen.append, payload_type=list[bytes])

        submit_callback_to_stream(stream, "finish_write", [b"k0", b"k1", b"k2"])
        stream.synchronize()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.01)
        assert seen == [[b"k0", b"k1", b"k2"]]

    def test_unknown_kind_drops_payload(self, dispatcher, stream):
        submit_callback_to_stream(stream, "finish_unknown", [b"x"])
        stream.synchronize()
        time.sleep(0.1)
        assert dispatcher.dispatched_count() == 0

    def test_handler_exception_does_not_kill_thread(self, dispatcher, stream):
        calls: list[list[bytes]] = []

        def handler(payload):
            calls.append(payload)
            if len(calls) == 1:
                raise RuntimeError("boom")

        dispatcher.register("finish_write", handler, payload_type=list[bytes])
        submit_callback_to_stream(stream, "finish_write", [b"a"])
        submit_callback_to_stream(stream, "finish_write", [b"b"])
        stream.synchronize()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(calls) < 2:
            time.sleep(0.01)
        assert calls == [[b"a"], [b"b"]]
        assert dispatcher.handler_exception_counts().get("finish_write", 0) == 1

    def test_tuple_payload_for_multiple_arguments(self, dispatcher, stream):
        # Multi-arg handlers: wrap args in a tuple, register tuple decode type.
        seen: list[tuple[int, str]] = []
        dispatcher.register("finish_multi", seen.append, payload_type=tuple[int, str])

        submit_callback_to_stream(stream, "finish_multi", (42, "hello"))
        stream.synchronize()

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not seen:
            time.sleep(0.01)
        assert seen == [(42, "hello")]


class TestDeadlockRegression:
    """Regression for the GIL deadlock: the legacy
    ``stream.launch_host_func(python_fn, ...)`` path stalls; the C++
    callback runs without the GIL and must finish within the timeout."""

    def test_many_concurrent_records_no_deadlock(self, dispatcher, stream):
        received: list[int] = []
        ready = threading.Event()

        def handler(payload):
            received.append(payload)
            if len(received) >= 200:
                ready.set()

        dispatcher.register("finish_write", handler, payload_type=int)

        for i in range(200):
            submit_callback_to_stream(stream, "finish_write", i)
        stream.synchronize()

        assert ready.wait(timeout=10.0), (
            f"deadlock or drop: only {len(received)} of 200 dispatched"
        )
        assert len(received) == 200


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_close_drains_pending_cuda_transfers(
    monkeypatch: pytest.MonkeyPatch, direction: str
) -> None:
    """Closing with pending copies delivers native callbacks before context release."""
    delay = cupy.RawKernel(
        r"""extern "C" __global__ void delay() {
            unsigned long long start, now;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
            do {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
            } while (now - start < 250000000ULL);
        }""",
        "delay",
    )
    delay.compile()
    streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(2)]
    hosts = [
        torch.full((4096,), 37, dtype=torch.uint8, pin_memory=True) for _ in streams
    ]
    devices = [cupy.full(4096, 53, dtype=cupy.uint8) for _ in streams]
    torch_dev.synchronize()
    contexts = []
    for stream in streams:
        context = MagicMock()
        context.device = torch.device("cuda", 0)
        context.kv_layer_groups_manager.num_object_groups = 1
        context.stream = stream
        context.close.side_effect = stream.synchronize
        contexts.append(context)
    monkeypatch.setattr(
        lmcache_driven_transfer, "create_cache_context", MagicMock(side_effect=contexts)
    )
    monkeypatch.setattr(lmcache_driven_transfer, "get_layout_desc", MagicMock())
    module = lmcache_driven_transfer.LMCacheDrivenTransferModule(MagicMock())
    delivered: list[int] = []

    def completed(instance_id: int) -> None:
        assert module.get_and_touch_context_entry(instance_id) is not None
        delivered.append(instance_id)

    module.register_host_func("close_transfer", completed, int)
    try:
        for instance_id in range(2):
            module.register_kv_cache(
                instance_id, [], "close-cuda", 1, EngineType.VLLM, {}, []
            )
        for instance_id, (stream, host, device) in enumerate(
            zip(streams, hosts, devices, strict=True)
        ):
            delay((1,), (1,), (), stream=stream)
            dst, src = device.data.ptr, host.data_ptr()
            kind = cupy.cuda.runtime.memcpyHostToDevice
            if direction == "d2h":
                dst, src = src, dst
                kind = cupy.cuda.runtime.memcpyDeviceToHost
            cupy.cuda.runtime.memcpyAsync(dst, src, host.numel(), kind, stream.ptr)
            submit_callback_to_stream(stream, "close_transfer", instance_id)
            submit_callback_to_stream(
                stream, "release_imported_event", (instance_id, 7)
            )
        assert all(not stream.done for stream in streams)
        module.close()
        assert sorted(delivered) == [0, 1]
        assert cuda_ops.drain_recorded_completions() == []
        assert module.tracked_instance_count() == 0
        for context, host, device in zip(contexts, hosts, devices, strict=True):
            context.release_imported_event.assert_called_once_with(7)
            context.close.assert_called_once_with()
            values = device.get().tolist() if direction == "h2d" else host.tolist()
            assert values == [37 if direction == "h2d" else 53] * 4096
    finally:
        for stream in streams:
            stream.synchronize()
        module.close()
