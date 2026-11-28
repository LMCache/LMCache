# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm event IPC backend and ``wait_producer_event``."""

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    ProducerEventWaitError,
    wait_producer_event,
)
from lmcache.v1.platform.devices.cuda.timeline_semaphore_event_ipc import (
    TimelineSemaphoreEventIPCBackend,
)
from lmcache.v1.platform.devices.rocm import RocmDeviceSpec
from lmcache.v1.platform.devices.rocm.event_ipc import RocmEventIPCBackend
from lmcache.v1.platform.ipc_policy import is_isolated_ipc, set_isolated_ipc
import lmcache.v1.platform.base.event_ipc as event_ipc


class _Event:
    """An imported event that completes after ``pending`` queries."""

    def __init__(self, pending: int = 0) -> None:
        self.pending = pending
        self.log: list[tuple] = []

    @classmethod
    def from_ipc_handle(cls, device, handle):
        return cls()

    def query(self) -> bool:
        self.log.append(("query",))
        if self.pending:
            self.pending -= 1
            return False
        return True

    def wait(self, stream=None) -> None:
        self.log.append(("wait", stream))


class _EventModule:
    Event = _Event


class _Clock:
    """Fake monotonic clock that ``pause`` advances."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pauses: list[float] = []

    def __call__(self) -> float:
        return self.now

    def pause(self, seconds: float) -> None:
        self.pauses.append(seconds)
        self.now += seconds


@pytest.fixture
def restore_isolated_ipc():
    """Restore the process-global isolated-IPC switch after the test."""
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


def test_rocm_backend_polls_the_producer_event_before_the_stream_wait() -> None:
    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Event(pending=3)
    clock = _Clock()
    wait_producer_event(backend, event, "stream", clock=clock, pause=clock.pause)
    # The stream wait is queued only after the host saw the event complete.
    assert event.log == [("query",)] * 4 + [("wait", "stream")]
    assert len(clock.pauses) == 3


def test_rocm_backend_completed_event_is_not_delayed() -> None:
    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Event()
    clock = _Clock()
    wait_producer_event(backend, event, "stream", clock=clock, pause=clock.pause)
    assert event.log == [("query",), ("wait", "stream")]
    assert clock.pauses == []


def test_default_backend_queues_the_wait_without_polling() -> None:
    backend = DefaultEventIPCBackend(event_module=_EventModule, device_type="cuda")
    event = _Event(pending=5)
    wait_producer_event(
        backend,
        event,
        "stream",
        is_alive=lambda: pytest.fail("is_alive consulted"),
        timeout_s=0.0,
        pause=pytest.fail,
    )
    assert event.log == [("wait", "stream")]


def test_poll_slows_down_and_warns_before_the_deadline(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        event_ipc.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    backend = RocmEventIPCBackend(event_module=_EventModule)
    # Pending for longer than the first warning (10 s at 2 ms polls).
    event = _Event(pending=6000)
    clock = _Clock()
    wait_producer_event(backend, event, "stream", clock=clock, pause=clock.pause)
    assert event.log[-1] == ("wait", "stream")
    assert clock.pauses[0] == pytest.approx(0.0005)
    assert clock.pauses[-1] == pytest.approx(0.002)
    assert clock.now > 10.0
    assert len(warnings) == 1 and "not complete after 10 s" in warnings[0]


def test_poll_gives_up_at_the_deadline_without_queueing_the_wait() -> None:
    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Event(pending=10**9)
    clock = _Clock()
    with pytest.raises(ProducerEventWaitError, match="limit 30 s"):
        wait_producer_event(
            backend, event, "stream", timeout_s=30.0, clock=clock, pause=clock.pause
        )
    assert ("wait", "stream") not in event.log
    assert 30.0 <= clock.now < 30.01


def test_poll_gives_up_when_the_engine_is_gone() -> None:
    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Event(pending=10**9)
    clock = _Clock()
    with pytest.raises(ProducerEventWaitError, match="no longer registered"):
        wait_producer_event(
            backend,
            event,
            "stream",
            is_alive=lambda: clock.now < 1.0,
            clock=clock,
            pause=clock.pause,
        )
    assert ("wait", "stream") not in event.log
    assert 1.0 <= clock.now < 1.01


def test_engine_liveness_is_not_checked_while_the_poll_is_fast() -> None:
    """A fence that completes within the fast polls never consults is_alive."""
    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Event(pending=3)
    clock = _Clock()
    wait_producer_event(
        backend,
        event,
        "stream",
        is_alive=lambda: pytest.fail("is_alive consulted"),
        clock=clock,
        pause=clock.pause,
    )
    assert event.log[-1] == ("wait", "stream")


def test_poll_propagates_query_errors() -> None:
    class _Broken(_Event):
        def query(self) -> bool:
            raise RuntimeError("hipEventQuery failed")

    backend = RocmEventIPCBackend(event_module=_EventModule)
    event = _Broken()
    with pytest.raises(RuntimeError, match="hipEventQuery failed"):
        wait_producer_event(backend, event, "stream")
    assert ("wait", "stream") not in event.log


def test_rocm_device_spec_selects_rocm_backend(restore_isolated_ipc) -> None:
    set_isolated_ipc(False)
    spec = RocmDeviceSpec()
    backend = spec.event_ipc_backend
    assert isinstance(backend, RocmEventIPCBackend)
    assert backend.host_ready_before_imported_wait
    assert backend is spec.event_ipc_backend


def test_rocm_device_spec_keeps_timeline_backend_when_isolated(
    restore_isolated_ipc,
) -> None:
    set_isolated_ipc(True)
    spec = RocmDeviceSpec()
    backend = spec.event_ipc_backend
    assert isinstance(backend, TimelineSemaphoreEventIPCBackend)
    assert not getattr(backend, "host_ready_before_imported_wait", False)
