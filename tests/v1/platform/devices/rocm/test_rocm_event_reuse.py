# SPDX-License-Identifier: Apache-2.0
"""Tests for interprocess event reuse on ROCm (event pool, per-peer imports)."""

# Standard
from typing import Any
import gc

# Third Party
import pytest

# First Party
from lmcache.v1.platform.base.event_ipc import (
    DefaultEventIPCBackend,
    peer_event_ipc_backend,
)
from lmcache.v1.platform.devices.rocm import RocmDeviceSpec
from lmcache.v1.platform.devices.rocm.event_reuse import (
    _POOL_PROBES,
    IMPORTED_EVENT_CACHE_SIZE,
    PeerEventIPCBackend,
    ReusingEventIPCBackend,
)
from lmcache.v1.platform.ipc_policy import is_isolated_ipc, set_isolated_ipc


class _Runtime:
    """Fake device runtime: which handle bytes it issues and what is open."""

    def __init__(self) -> None:
        self.created = 0
        self.opened: list[bytes] = []
        # Work behind the event a handle names right now; a restarted peer
        # that issues the same bytes replaces the entry.
        self.pending: dict[bytes, bool] = {}


class _Event:
    """Fake interprocess event whose completion the test controls."""

    runtime: _Runtime

    def __init__(self, interprocess: bool = False) -> None:
        self.runtime.created += 1
        self.handle = b"h%d" % self.runtime.created
        self.done = True  # never recorded: complete, like a device event
        self.records: list[object] = []

    def record(self, stream: object) -> None:
        self.records.append(stream)
        self.done = False

    def query(self) -> bool:
        return self.done

    def ipc_handle(self) -> bytes:
        return self.handle

    @classmethod
    def from_ipc_handle(cls, device: object, handle: bytes) -> "_Imported":
        cls.runtime.opened.append(handle)
        return _Imported(cls.runtime.pending.get(handle, False))


class _Imported:
    """An opened handle: it keeps showing the event it was opened on."""

    def __init__(self, pending: bool) -> None:
        self.pending = pending

    def query(self) -> bool:
        return not self.pending


@pytest.fixture
def runtime() -> _Runtime:
    return _Runtime()


@pytest.fixture
def backend(runtime: _Runtime) -> ReusingEventIPCBackend:
    event_cls = type("Event", (_Event,), {"runtime": runtime})
    module = type("Module", (), {"Event": event_cls})()
    return ReusingEventIPCBackend(event_module=module, device_type="cuda")


def _create(backend: ReusingEventIPCBackend, device: str = "d0") -> Any:
    """``create_event``, typed as the fake event the tests inspect."""
    return backend.create_event(device)


def _transfer(backend: ReusingEventIPCBackend, device: str = "d0") -> Any:
    """Create and record an event, as one transfer does."""
    event = _create(backend, device)
    backend.record_event(event, "stream")
    return event


# ---------------------------------------------------------------- event pool


def test_finished_event_is_reused_for_the_next_transfer(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    first = _transfer(backend)
    first.done = True
    assert _create(backend, "d0") is first
    assert runtime.created == 1


def test_event_handed_out_and_not_recorded_is_never_handed_out_again(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    """Review case: ``a = create_event`` and 256 more creates must not give ``a``."""
    pool = [_transfer(backend) for _ in range(256)]  # 256 events in flight
    for event in pool:
        event.done = True
    a = _create(backend, "d0")  # e.g. a handler between create and record
    assert any(a is event for event in pool)
    others = []
    for _ in range(256):
        event = _create(backend, "d0")
        others.append(event)
        backend.record_event(event, "stream")
        event.done = True
    assert all(event is not a for event in others)
    assert a.records == ["stream"]  # only its previous transfer recorded it
    assert runtime.created == 256  # the other 255 events served the 256 creates


def test_two_transfers_created_before_either_records_get_different_events(
    backend: ReusingEventIPCBackend,
) -> None:
    for _ in range(4):
        _transfer(backend).done = True
    a = _create(backend, "d0")
    b = _create(backend, "d0")
    assert a is not b
    backend.record_event(a, "slow-stream")
    backend.record_event(b, "fast-stream")
    assert a.records[-1] == "slow-stream" and b.records[-1] == "fast-stream"


def test_recorded_event_returns_to_the_pool_once(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    event = _create(backend, "d0")
    backend.record_event(event, "stream")
    backend.record_event(event, "stream")  # e.g. a retry path records again
    event.done = True
    assert _create(backend, "d0") is event
    assert _create(backend, "d0") is not event
    assert runtime.created == 2


def test_event_in_flight_is_kept_and_reused_once_it_finishes(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    busy = _transfer(backend)
    other = _transfer(backend)
    assert other is not busy  # a consumer of ``busy`` must not see this record
    other.done = True
    assert _create(backend, "d0") is other
    busy.done = True
    assert _create(backend, "d0") is busy  # not replaced while in flight
    assert runtime.created == 2


def test_create_queries_a_bounded_number_of_events_in_flight(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    for _ in range(_POOL_PROBES + 1):
        _transfer(backend)
    queries = 0

    def counting_query(self: _Event) -> bool:
        nonlocal queries
        queries += 1
        return self.done

    type(_create(backend, "d0")).query = counting_query  # type: ignore[method-assign]
    queries = 0
    _create(backend, "d0")
    assert queries == _POOL_PROBES


def test_event_that_is_never_recorded_does_not_stay_in_the_pool(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    _transfer(backend).done = True
    abandoned = _create(backend, "d0")  # the transfer fails before record
    del abandoned
    gc.collect()
    assert len(backend._unrecorded) == 0
    assert not backend._pools["d0"]
    assert _transfer(backend).handle == b"h2"  # a new event, not a leaked slot


def test_pools_are_per_device(backend: ReusingEventIPCBackend) -> None:
    first = _transfer(backend, "d0")
    first.done = True
    assert _create(backend, "d1") is not first
    assert _create(backend, "d0") is first


# ---------------------------------------------------------- per-peer imports


def test_peer_view_opens_each_handle_once(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    peer = backend.for_peer()
    a = peer.import_event(b"a", "d0")
    assert peer.import_event(b"a", "d0") is a
    assert peer.import_event(b"a", "d1") is not a
    assert peer.import_event(b"b", "d0") is not a
    assert runtime.opened == [b"a", b"a", b"b"]


def test_shared_backend_does_not_keep_imports(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    assert backend.import_event(b"a", "d0") is not backend.import_event(b"a", "d0")
    assert runtime.opened == [b"a", b"a"]


def test_restarted_peer_with_the_same_handle_bytes_is_opened_again(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    """Review case: a new registration never gets the dead peer's event."""
    old_peer = backend.for_peer()
    stale = old_peer.import_event(b"h1", "d0")
    assert backend.query_event(stale) is True  # the dead engine's work is done

    # The engine restarts, re-registers and issues the same handle bytes for
    # an event whose work is still running.
    runtime.pending[b"h1"] = True
    new_peer = backend.for_peer()
    fresh = new_peer.import_event(b"h1", "d0")
    assert fresh is not stale
    assert backend.query_event(fresh) is False
    assert runtime.opened == [b"h1", b"h1"]


def test_two_peers_with_the_same_handle_bytes_do_not_share_an_import(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    first, second = backend.for_peer(), backend.for_peer()
    assert first.import_event(b"h1", "d0") is not second.import_event(b"h1", "d0")


def test_peer_view_is_bounded(
    backend: ReusingEventIPCBackend, runtime: _Runtime
) -> None:
    peer = backend.for_peer()
    a = peer.import_event(b"a", "d0")
    for i in range(IMPORTED_EVENT_CACHE_SIZE):
        peer.import_event(b"x%d" % i, "d0")
    assert len(peer._imports) == IMPORTED_EVENT_CACHE_SIZE
    assert peer.import_event(b"a", "d0") is not a  # least recently used: evicted


def test_peer_view_delegates_everything_else(
    backend: ReusingEventIPCBackend,
) -> None:
    peer = backend.for_peer()
    event: Any = peer.create_event("d0")
    peer.record_event(event, "stream")
    assert peer.export_event(event, "d0") == event.handle
    assert peer.query_event(event) is False
    assert peer.device_type == "cuda"
    event.done = True
    assert peer.create_event("d0") is event  # the pool is the shared backend's


# -------------------------------------------------------------------- wiring


def test_peer_backend_helper_scopes_only_backends_that_ask_for_it(
    backend: ReusingEventIPCBackend,
) -> None:
    plain = DefaultEventIPCBackend(event_module=object(), device_type="fake")
    assert peer_event_ipc_backend(plain) is plain
    scoped = peer_event_ipc_backend(backend)
    assert isinstance(scoped, PeerEventIPCBackend)
    assert peer_event_ipc_backend(backend) is not scoped


def test_default_backend_creates_an_event_per_transfer(runtime: _Runtime) -> None:
    """Reuse is ROCm-only: the backend other devices use is unchanged."""
    event_cls = type("Event", (_Event,), {"runtime": runtime})
    plain = DefaultEventIPCBackend(
        event_module=type("Module", (), {"Event": event_cls})(), device_type="cuda"
    )
    first: Any = plain.create_event("d0")
    plain.record_event(first, "stream")
    first.done = True
    assert plain.create_event("d0") is not first
    assert plain.import_event(b"a", "d0") is not plain.import_event(b"a", "d0")


@pytest.fixture
def restore_isolated_ipc():
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


def test_rocm_device_spec_exposes_the_reusing_backend(restore_isolated_ipc) -> None:
    set_isolated_ipc(False)
    spec = RocmDeviceSpec()
    backend = spec.event_ipc_backend
    assert isinstance(backend, ReusingEventIPCBackend)
    assert spec.event_ipc_backend is backend


def test_rocm_device_spec_keeps_the_cuda_backend_when_isolated(
    restore_isolated_ipc,
) -> None:
    set_isolated_ipc(True)
    assert not isinstance(RocmDeviceSpec().event_ipc_backend, ReusingEventIPCBackend)
