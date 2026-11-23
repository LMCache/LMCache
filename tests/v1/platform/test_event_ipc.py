# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import cast
import gc
import inspect
import threading
import time
import weakref

# Third Party
import pytest

# First Party
from lmcache import torch_device_type
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.base.event_ipc import (
    _EXPORTED_EVENT_RING_SIZE,
    CachedImportEventBackend,
    DefaultEventIPCBackend,
    EventIPCBackend,
    EventRecordPool,
    _BackgroundReleaser,
    get_event_ipc_backend,
    supports_event_reuse,
)
from lmcache.v1.platform.devices.cpu import CpuDeviceSpec
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec
from lmcache.v1.platform.devices.cuda.timeline_semaphore_event_ipc import (
    TimelineSemaphoreEventIPCBackend,
)
from lmcache.v1.platform.devices.rocm import RocmDeviceSpec
from lmcache.v1.platform.ipc_policy import is_isolated_ipc, set_isolated_ipc
import lmcache.v1.platform as platform

pytestmark = pytest.mark.skipif(
    torch_device_type == "xpu",
    reason="Event IPC tests are not supported on XPU",
)


class _FakeEvent:
    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess
        self.calls: list[tuple] = []

    def ipc_handle(self) -> bytes:
        return b"handle-bytes"

    @classmethod
    def from_ipc_handle(cls, device, handle):
        ev = cls(interprocess=True)
        ev.calls.append(("from_ipc_handle", device, handle))
        return ev

    def record(self, stream=None):
        self.calls.append(("record", stream))

    def wait(self, stream=None):
        self.calls.append(("wait", stream))

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.calls.append(("synchronize",))


class _FakeEventModule:
    Event = _FakeEvent


class _NoInterprocessEvent:
    def __init__(self) -> None:  # no interprocess parameter
        pass

    @classmethod
    def from_ipc_handle(cls, device, handle):
        return cls()


class _NoInterprocessModule:
    Event = _NoInterprocessEvent


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


class _FakeDeviceSpec:
    def __init__(self, backend: EventIPCBackend | None) -> None:
        self.event_ipc_backend = backend


def test_default_backend_create_export_import_delegate():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    event = backend.create_event(_Device("fake"))
    assert isinstance(event, _FakeEvent) and event.interprocess is True
    assert backend.export_event(event, _Device("fake")) == b"handle-bytes"
    imported = backend.import_event(b"h", _Device("fake"))
    assert (
        "from_ipc_handle",
        _is_device := imported.calls[0][1],
        b"h",
    ) == imported.calls[0]


def test_export_event_retains_event_while_handle_is_usable():
    """An IPC handle dangles if its event dies, so export must retain it.

    The exporter previously dropped its only reference on return, so whether a
    peer could still import the handle depended on refcount/GC timing.
    """
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    event = backend.create_event(_Device("fake"))
    alive = weakref.ref(event)

    backend.export_event(event, _Device("fake"))
    del event
    gc.collect()

    assert alive() is not None


def test_exported_event_retention_is_bounded():
    """Retention must not grow without limit on a long-lived backend."""
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    for _ in range(_EXPORTED_EVENT_RING_SIZE + 64):
        backend.export_event(backend.create_event(_Device("fake")), _Device("fake"))

    assert len(backend._exported_events) == _EXPORTED_EVENT_RING_SIZE


def test_default_backend_record_wait_query_synchronize_delegate():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    event = backend.create_event(_Device("fake"))
    backend.record_event(event, "STREAM")
    backend.wait_event(event, "STREAM")
    assert backend.query_event(event) is True
    backend.synchronize_event(event, _Device("fake"))
    assert ("record", "STREAM") in event.calls
    assert ("wait", "STREAM") in event.calls
    assert ("synchronize",) in event.calls


def test_check_event_support_raises_with_device_name_when_no_interprocess():
    backend = DefaultEventIPCBackend(
        event_module=_NoInterprocessModule(), device_type="weirddev"
    )
    with pytest.raises(RuntimeError, match="weirddev"):
        backend.check_event_support(_Device("weirddev"))


def test_check_event_support_passes_for_capable_module():
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    backend.check_event_support(_Device("fake"))  # must not raise


def test_get_event_ipc_backend_resolves_cpu_to_default():
    backend = get_event_ipc_backend(_Device("cpu"))
    assert isinstance(backend, EventIPCBackend)
    assert isinstance(backend, DefaultEventIPCBackend)


def test_get_event_ipc_backend_requires_device() -> None:
    signature = inspect.signature(get_event_ipc_backend)
    assert signature.parameters["device"].default is inspect.Parameter.empty


def test_get_event_ipc_backend_accepts_string_device_type():
    assert isinstance(get_event_ipc_backend("cpu"), DefaultEventIPCBackend)


@pytest.mark.parametrize(
    ("device", "expected_device_type"),
    [
        pytest.param(0, torch_device_type, id="integer-index"),
        pytest.param(2, torch_device_type, id="nonzero-integer-index"),
        pytest.param(
            f"{torch_device_type}:0",
            torch_device_type,
            id="indexed-active-device-string",
        ),
        pytest.param("musa:3", "musa", id="indexed-musa-string"),
    ],
)
def test_get_event_ipc_backend_normalizes_indexed_devices(
    monkeypatch: pytest.MonkeyPatch,
    device: object,
    expected_device_type: str,
) -> None:
    """Backend lookup uses the base type for integer and indexed devices."""
    backend = DefaultEventIPCBackend(
        event_module=_FakeEventModule(), device_type="fake"
    )
    resolved_device_types: list[str] = []

    def fake_get_device_spec(device_type: str) -> _FakeDeviceSpec:
        resolved_device_types.append(device_type)
        return _FakeDeviceSpec(backend)

    monkeypatch.setattr(platform, "get_device_spec", fake_get_device_spec)

    assert get_event_ipc_backend(device) is backend
    assert resolved_device_types == [expected_device_type]


def test_get_event_ipc_backend_rejects_unregistered_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered device is a platform configuration error."""
    monkeypatch.setattr(platform, "get_device_spec", lambda device_type: None)

    with pytest.raises(
        RuntimeError,
        match="No DeviceSpec registered for device type 'unregistered'",
    ):
        get_event_ipc_backend("unregistered:0")


def test_get_event_ipc_backend_rejects_unsupported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered device must explicitly provide event IPC support."""
    monkeypatch.setattr(
        platform,
        "get_device_spec",
        lambda device_type: _FakeDeviceSpec(None),
    )

    with pytest.raises(
        RuntimeError,
        match="Device type 'unsupported' does not support event IPC",
    ):
        get_event_ipc_backend("unsupported")


def test_get_event_ipc_backend_cpu_uses_stub_when_accelerator_is_active() -> None:
    """Explicit CPU lookups do not use the active accelerator event module."""
    backend = get_event_ipc_backend("cpu")
    backend.check_event_support("cpu")
    event = backend.create_event("cpu")

    assert backend.export_event(event, "cpu") == b"stub_ipc_handle"


def test_device_spec_does_not_expose_event_backend() -> None:
    spec = DeviceSpec()
    assert spec.event_ipc_backend is None


def test_cpu_device_spec_exposes_cached_default_event_backend() -> None:
    spec = CpuDeviceSpec()
    assert isinstance(spec.event_ipc_backend, DefaultEventIPCBackend)
    assert spec.event_ipc_backend is spec.event_ipc_backend


@pytest.fixture
def restore_isolated_ipc():
    """Restore the process-global isolated-IPC switch after the test."""
    previous = is_isolated_ipc()
    yield
    set_isolated_ipc(previous)


def test_cuda_device_spec_exposes_timeline_backend_when_isolated(
    restore_isolated_ipc,
) -> None:
    set_isolated_ipc(True)
    spec = CudaDeviceSpec()
    assert isinstance(spec.event_ipc_backend, TimelineSemaphoreEventIPCBackend)
    assert spec.event_ipc_backend.device_type == "cuda"
    assert spec.event_ipc_backend is spec.event_ipc_backend


def test_cuda_device_spec_exposes_default_backend_by_default(
    restore_isolated_ipc,
) -> None:
    """Isolated IPC defaults to off, keeping CUDA interprocess event handles."""
    set_isolated_ipc(False)
    spec = CudaDeviceSpec()
    assert isinstance(spec.event_ipc_backend, DefaultEventIPCBackend)
    assert spec.event_ipc_backend.device_type == "cuda"
    assert spec.event_ipc_backend is spec.event_ipc_backend


def test_cuda_device_spec_backend_cache_ignores_later_switch_changes(
    restore_isolated_ipc,
) -> None:
    """The backend is cached on first read; later switch flips are ignored."""
    set_isolated_ipc(True)
    spec = CudaDeviceSpec()
    backend = spec.event_ipc_backend
    set_isolated_ipc(False)
    assert spec.event_ipc_backend is backend


def test_default_backend_stub_end_to_end():
    # Uses the real active torch_dev (StubCPUDevice/StubEvent) via the default.
    backend = get_event_ipc_backend("cpu")
    backend.check_event_support("cpu")
    event = backend.create_event("cpu")
    handle = backend.export_event(event, "cpu")
    assert isinstance(handle, bytes)
    remote = backend.import_event(handle, "cpu")
    backend.record_event(event, None)
    assert backend.query_event(remote) is True
    backend.synchronize_event(remote, "cpu")


def test_protocol_signatures_match_default_backend():
    proto_methods = {
        "check_event_support",
        "create_event",
        "export_event",
        "import_event",
        "record_event",
        "wait_event",
        "query_event",
        "synchronize_event",
    }
    for name in proto_methods:
        assert callable(getattr(DefaultEventIPCBackend, name))
        assert inspect.signature(getattr(DefaultEventIPCBackend, name)) is not None


class _CountingEvent(_FakeEvent):
    """Fake event with a distinct handle per event and counted exports."""

    _count = 0

    def __init__(self, interprocess: bool = False) -> None:
        super().__init__(interprocess)
        _CountingEvent._count += 1
        self.id = _CountingEvent._count
        self.exports = 0
        self.records = 0

    def ipc_handle(self) -> bytes:
        self.exports += 1
        return b"handle-%d" % self.id

    def record(self, stream=None):
        self.records += 1
        super().record(stream)


class _CountingEventModule:
    Event = _CountingEvent


def _reusing_backend(max_event_records: int | None) -> DefaultEventIPCBackend:
    return DefaultEventIPCBackend(
        event_module=_CountingEventModule(),
        device_type="fake",
        max_event_records=max_event_records,
    )


def _record(pool: EventRecordPool, stream: str) -> _CountingEvent:
    event = cast(_CountingEvent, pool.acquire(stream))
    event.record(stream)
    return event


def test_record_pool_rotates_per_stream_and_caps_records():
    """Events rotate per stream and are replaced after max_event_records."""
    pool = EventRecordPool(_reusing_backend(2), "dev", slots=2)

    first = [_record(pool, "s") for _ in range(4)]
    other = _record(pool, "t")
    replaced = _record(pool, "s")

    assert first[0] is first[2] and first[1] is first[3]
    assert first[0] is not first[1]
    assert other not in first
    assert replaced not in first
    assert [event.records for event in first[:2]] == [2, 2]
    assert [call[1] for call in first[0].calls] == ["s", "s"]


def test_record_pool_exports_each_pooled_event_once():
    pool = EventRecordPool(_reusing_backend(None), "dev", slots=1)
    event = _record(pool, "s")
    foreign = _CountingEvent(interprocess=True)

    assert pool.export(event) == pool.export(_record(pool, "s"))
    assert event.exports == 1
    pool.export(foreign)
    pool.export(foreign)
    assert foreign.exports == 2


def test_record_pool_without_reuse_creates_an_event_per_record():
    """With max_event_records == 1 the pool behaves as if absent."""
    pool = EventRecordPool(_reusing_backend(1), "dev", slots=4)
    events = [_record(pool, "s") for _ in range(3)]

    assert len({id(event) for event in events}) == 3
    assert [event.records for event in events] == [1, 1, 1]
    pool.export(events[0])
    pool.export(events[0])
    assert events[0].exports == 2


def test_record_pool_creates_an_event_per_record_unless_the_peer_agreed():
    """reuse=False (the importing peer did not agree) disables the pool."""
    pool = EventRecordPool(_reusing_backend(None), "dev", slots=4, reuse=False)
    events = [_record(pool, "s") for _ in range(3)]

    assert len({id(event) for event in events}) == 3
    assert len({pool.export(event) for event in events}) == 3


def test_supports_event_reuse_follows_max_event_records():
    assert supports_event_reuse(_reusing_backend(None))
    assert supports_event_reuse(_reusing_backend(31))
    assert not supports_event_reuse(_reusing_backend(1))
    assert not supports_event_reuse(object())


def test_record_pool_close_drops_its_events():
    pool = EventRecordPool(_reusing_backend(None), "dev", slots=1)
    alive = weakref.ref(pool.acquire("s"))
    pool.close()
    gc.collect()

    assert alive() is None
    assert _record(pool, "s") is not None


def test_cached_import_keeps_recent_imports_and_releases_evicted_ones():
    """Imports are kept per handle, least recently used out first."""
    released: list[object] = []
    backend = CachedImportEventBackend(_reusing_backend(None), 2, released.append)

    a = backend.import_event(b"a", "dev")
    b = backend.import_event(b"b", "dev")
    assert backend.import_event(b"a", "dev") is a
    backend.import_event(b"c", "dev")

    assert released == [b]
    assert backend.import_event(b"a", "dev") is a
    assert backend.import_event(b"b", "dev") is not b


def test_cached_import_serves_a_new_handle_per_operation():
    """A peer that sends a new handle per operation keeps working, bounded."""
    released: list[object] = []
    backend = CachedImportEventBackend(_reusing_backend(None), 4, released.append)

    imported = [backend.import_event(b"op-%d" % i, "dev") for i in range(10)]

    assert [event.calls[0][2] for event in imported] == [
        b"op-%d" % i for i in range(10)
    ]
    assert released == imported[:6]


def test_cached_import_passes_through_without_reuse():
    released: list[object] = []
    backend = CachedImportEventBackend(_reusing_backend(1), 4, released.append)

    assert backend.import_event(b"a", "dev") is not backend.import_event(b"a", "dev")
    assert released == []


def test_cached_import_close_releases_every_import():
    released: list[object] = []
    backend = CachedImportEventBackend(_reusing_backend(None), 4, released.append)
    imported = [backend.import_event(h, "dev") for h in (b"a", b"b")]

    backend.close()

    assert released == imported
    assert backend.import_event(b"a", "dev") is not imported[0]


def test_cached_import_drops_evicted_imports_off_the_calling_thread():
    """Destroying an import can block, so the last reference goes elsewhere."""
    dropped_on: list[str] = []

    class _TrackedEvent(_FakeEvent):
        def __del__(self) -> None:
            dropped_on.append(threading.current_thread().name)

    class _TrackedModule:
        Event = _TrackedEvent

    backend = CachedImportEventBackend(
        DefaultEventIPCBackend(
            event_module=_TrackedModule(), device_type="fake", max_event_records=None
        ),
        1,
    )
    backend.import_event(b"a", "dev")
    backend.import_event(b"b", "dev")

    deadline = time.monotonic() + 5
    while not dropped_on and time.monotonic() < deadline:
        time.sleep(0.01)
    assert dropped_on == ["lmcache-event-release"]


def test_release_thread_drops_everything_when_stopped():
    """At exit nothing is left to be destroyed on the release thread."""
    dropped_on: list[str] = []

    class _TrackedEvent(_FakeEvent):
        def __del__(self) -> None:
            dropped_on.append(threading.current_thread().name)

    releaser = _BackgroundReleaser()
    releaser.release(_TrackedEvent())
    releaser.stop()
    assert dropped_on == ["lmcache-event-release"]

    releaser.release(_TrackedEvent())
    assert dropped_on == ["lmcache-event-release", threading.current_thread().name]


def test_cached_import_backend_satisfies_protocol():
    backend = CachedImportEventBackend(_reusing_backend(None), 1)
    assert isinstance(backend, EventIPCBackend)
    assert backend.device_type == "fake"
    assert backend.max_event_records is None


def test_device_specs_declare_event_reuse(restore_isolated_ipc) -> None:
    """ROCm reuses events within its record limit; CUDA and snapshots do not."""
    set_isolated_ipc(False)
    assert RocmDeviceSpec().event_ipc_backend.max_event_records == 31
    assert CudaDeviceSpec().event_ipc_backend.max_event_records == 1
    assert TimelineSemaphoreEventIPCBackend.max_event_records == 1
    assert CpuDeviceSpec().event_ipc_backend.max_event_records == 1
