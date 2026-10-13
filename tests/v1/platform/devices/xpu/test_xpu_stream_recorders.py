# SPDX-License-Identifier: Apache-2.0
"""Stream ordering of the XPU completion and event recorders.

L1 commits stored objects and releases read locks from these completions,
so a completion must never become visible while the transfer that
produced it is still running on the queue.
"""

# Standard
from typing import Any
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.native_completion import (
    DeviceHostFuncDispatcher,
    submit_callback_to_stream,
)
from lmcache.v1.platform.devices.xpu.cache_context import _XpuHostCallbackStream
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("requires an XPU device", allow_module_level=True)

try:
    # First Party
    import lmcache.xpu_ops as xpu_ops
except ImportError:
    xpu_ops = None

# Enough 4096^2 bf16 matmuls that the queue is still busy when the
# recorder is called and the drain runs right after.
_BUSY_MATMULS = 40


def _python_fallback() -> XpuDeviceOps:
    """An XpuDeviceOps with native ops left unbound."""
    return XpuDeviceOps()


def _native() -> Any:
    if xpu_ops is None:
        pytest.skip("requires lmcache.xpu_ops")
    return xpu_ops


@pytest.fixture(params=["native", "python_fallback"])
def recorder(request: pytest.FixtureRequest) -> Any:
    ops = _native() if request.param == "native" else _python_fallback()
    ops.drain_recorded_completions()
    ops.drain_recorded_events()
    return ops


@pytest.fixture()
def stream() -> torch.xpu.Stream:
    s = torch.xpu.Stream()
    yield s
    s.synchronize()


def _make_busy(stream: torch.xpu.Stream) -> None:
    a = torch.randn(4096, 4096, device="xpu", dtype=torch.bfloat16)
    stream.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(stream):
        for _ in range(_BUSY_MATMULS):
            a = a @ a
    # Keep the precondition explicit: if the device outruns the workload,
    # an unordered recorder would pass vacuously.
    assert not stream.query(), "workload finished too early to test ordering"


def _assert_ordered(record, drain, stream: torch.xpu.Stream) -> list:
    """Record after busy work; nothing may be drained while still busy."""
    _make_busy(stream)
    record()
    early = drain()
    idle_after_drain = stream.query()
    assert not early or idle_after_drain, (
        "record became visible while the queue was still running prior work"
    )
    stream.synchronize()
    return early + drain()


def test_completion_waits_for_prior_queue_work(recorder, stream) -> None:
    items = _assert_ordered(
        lambda: recorder.record_completion_on_stream(
            stream.sycl_queue, "finish_write_by_owner", b"\xff\x00payload"
        ),
        recorder.drain_recorded_completions,
        stream,
    )
    assert items == [("finish_write_by_owner", b"\xff\x00payload")]


def test_event_waits_for_prior_queue_work(recorder, stream) -> None:
    before = time.time()
    items = _assert_ordered(
        lambda: recorder.record_event_on_stream(
            stream.sycl_queue, "mp.store.end", "sid", {"k": "v"}, {"n": 3}
        ),
        recorder.drain_recorded_events,
        stream,
    )
    assert len(items) == 1
    event_type, session_id, timestamp, str_metadata, int_metadata = items[0]
    assert (event_type, session_id) == ("mp.store.end", "sid")
    assert (str_metadata, int_metadata) == ({"k": "v"}, {"n": 3})
    assert before <= timestamp <= time.time()


def test_completions_drain_in_submission_order(recorder, stream) -> None:
    for i in range(50):
        recorder.record_completion_on_stream(
            stream.sycl_queue, "finish_read_by_owner", msgspec.msgpack.encode(i)
        )
    stream.synchronize()

    items = recorder.drain_recorded_completions()
    assert [msgspec.msgpack.decode(p, type=int) for _, p in items] == list(range(50))


def test_native_rejects_null_queue() -> None:
    native = _native()
    with pytest.raises(RuntimeError, match="sycl::queue"):
        native.record_completion_on_stream(0, "finish_write_by_owner", b"")


def test_bound_device_ops_use_native_record_and_drain() -> None:
    """Record and drain share one buffer, so both must come from xpu_ops."""
    native = _native()
    ops = XpuDeviceOps()
    ops.ensure_native()
    for name in (
        "record_completion_on_stream",
        "drain_recorded_completions",
        "record_event_on_stream",
        "drain_recorded_events",
    ):
        assert getattr(ops, name) is getattr(native, name)


def test_dispatcher_runs_handler_after_queue_work(stream) -> None:
    """End to end: the cache-context stream adapter and dispatcher."""
    _native()
    seen: list[list[bytes]] = []
    dispatcher = DeviceHostFuncDispatcher(drain_interval_seconds=0.001)
    dispatcher.register("finish_write_by_owner", seen.append, list[bytes])
    dispatcher.start()
    try:
        _make_busy(stream)
        submit_callback_to_stream(
            _XpuHostCallbackStream(stream), "finish_write_by_owner", [b"k0"]
        )
        while not stream.query():
            assert not seen, "handler ran while the queue was still busy"
            time.sleep(0.001)
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            time.sleep(0.001)
        assert seen == [[b"k0"]]
    finally:
        dispatcher.stop()
