# SPDX-License-Identifier: Apache-2.0
"""The XPU event IPC backend pools exported events and caches imports.

Two backend instances stand in for the exporting and importing processes,
over a fake ``torch.xpu`` event module, so no accelerator is needed.
"""

# Future
from __future__ import annotations

# Third Party
import pytest

# First Party
from lmcache.v1.platform.devices.xpu.event_ipc import (
    ENV_XPU_EVENT_POOL,
    ENV_XPU_EVENT_POOL_REUSE_DELAY_S,
    XpuEventIPCBackend,
)

DELAY = 1.0


class _FakeEvent:
    def __init__(self, interprocess: bool = False, handle: bytes = b"") -> None:
        self.interprocess = interprocess
        self.handle = handle
        self.done = True
        self.calls: list[str] = []

    def ipc_handle(self) -> bytes:
        return self.handle

    def record(self, stream: object = None) -> None:
        self.calls.append("record")

    def wait(self, stream: object = None) -> None:
        self.calls.append("wait")

    def query(self) -> bool:
        self.calls.append("query")
        return self.done

    def synchronize(self) -> None:
        self.calls.append("synchronize")


class _FakeXpu:
    """Fake ``torch.xpu``: numbered handles, and a log of every import."""

    def __init__(self, same_handle: bytes | None = None) -> None:
        self.created = 0
        self.imports: list[bytes] = []
        self._same_handle = same_handle
        module = self

        class Event(_FakeEvent):
            def __init__(self, interprocess: bool = False) -> None:
                module.created += 1
                handle = module._same_handle or b"h%d" % module.created
                super().__init__(interprocess, handle)

            @classmethod
            def from_ipc_handle(cls, device: object, handle: bytes) -> _FakeEvent:
                module.imports.append(handle)
                return _FakeEvent(True, handle)

        self.Event = Event


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _backend(
    module: _FakeXpu | None = None, clock: _Clock | None = None, pool: bool = True
) -> XpuEventIPCBackend:
    return XpuEventIPCBackend(
        event_module=module or _FakeXpu(),
        pool_enabled=pool,
        reuse_delay_s=DELAY,
        clock=clock or _Clock(),
    )


def _raw(event: object) -> _FakeEvent:
    return event.event  # type: ignore[attr-defined]


# Exporter: reuse


def test_dropped_event_is_reused_only_after_the_delay() -> None:
    clock = _Clock()
    backend = _backend(clock=clock)
    event = backend.create_event("xpu:0")
    raw = _raw(event)
    del event

    held = [backend.create_event("xpu:0")]  # sees the drop at t=0
    clock.now = DELAY / 2
    held.append(backend.create_event("xpu:0"))
    assert all(_raw(e) is not raw for e in held)

    clock.now = DELAY
    assert _raw(backend.create_event("xpu:0")) is raw


def test_held_event_is_not_reused() -> None:
    clock = _Clock()
    backend = _backend(clock=clock)
    event = backend.create_event("xpu:0")
    backend.create_event("xpu:0")
    clock.now = 10 * DELAY
    assert _raw(backend.create_event("xpu:0")) is not _raw(event)


def test_event_still_running_is_not_reused() -> None:
    clock = _Clock()
    backend = _backend(clock=clock)
    event = backend.create_event("xpu:0")
    raw = _raw(event)
    raw.done = False
    del event

    held = [backend.create_event("xpu:0")]  # sees the drop at t=0
    clock.now = 10 * DELAY
    held.append(backend.create_event("xpu:0"))
    assert all(_raw(e) is not raw for e in held)

    raw.done = True
    held.append(backend.create_event("xpu:0"))  # frees it now
    assert _raw(held[-1]) is not raw
    clock.now += DELAY
    assert _raw(backend.create_event("xpu:0")) is raw


def test_oldest_free_event_is_reused_first() -> None:
    clock = _Clock()
    backend = _backend(clock=clock)
    first = backend.create_event("xpu:0")
    second = backend.create_event("xpu:0")
    raws = [_raw(first), _raw(second)]
    del first
    del second

    held = [backend.create_event("xpu:0")]
    clock.now = DELAY
    held.append(backend.create_event("xpu:0"))
    held.append(backend.create_event("xpu:0"))
    assert [_raw(e) for e in held[1:]] == raws


def test_pools_are_per_device() -> None:
    clock = _Clock()
    backend = _backend(clock=clock)
    event = backend.create_event("xpu:0")
    raw = _raw(event)
    del event

    held = [backend.create_event("xpu:1")]
    clock.now = DELAY
    held.append(backend.create_event("xpu:1"))
    assert all(_raw(e) is not raw for e in held)
    assert _raw(backend.create_event("xpu:0")) is raw


def test_disabled_pool_creates_plain_events() -> None:
    module = _FakeXpu()
    backend = _backend(module, pool=False)
    event = backend.create_event("xpu:0")
    assert isinstance(event, module.Event)
    assert backend.export_event(event, "xpu:0") == b"h1"


def test_pooled_event_forwards_direct_calls() -> None:
    backend = _backend()
    event = backend.create_event("xpu:0")
    backend.record_event(event, "stream")
    event.wait()  # type: ignore[attr-defined]
    assert _raw(event).calls == ["record", "wait"]


# Importer: cache


def test_pooled_handle_is_imported_once() -> None:
    module = _FakeXpu()
    exporter, importer = _backend(), _backend(module)
    event = exporter.create_event("xpu:0")
    handle = exporter.export_event(event, "xpu:0")

    first = importer.import_event(handle, "xpu:0")
    second = importer.import_event(handle, "xpu:0")
    assert _raw(first) is _raw(second)
    assert module.imports == [b"h1"]


def test_pooled_handle_is_imported_per_device() -> None:
    module = _FakeXpu()
    exporter, importer = _backend(), _backend(module)
    handle = exporter.export_event(exporter.create_event("xpu:0"), "xpu:0")
    importer.import_event(handle, "xpu:0")
    importer.import_event(handle, "xpu:1")
    assert len(module.imports) == 2


def test_equal_raw_handles_of_different_exporters_are_not_shared() -> None:
    # A restarted exporter that reuses a dead one's PID yields equal bytes.
    module = _FakeXpu()
    importer = _backend(module)
    handles = []
    for _ in range(2):
        exporter = _backend(_FakeXpu(same_handle=b"pid-reused"))
        handles.append(exporter.export_event(exporter.create_event("xpu:0"), "xpu:0"))
    first = importer.import_event(handles[0], "xpu:0")
    second = importer.import_event(handles[1], "xpu:0")
    assert _raw(first) is not _raw(second)
    assert module.imports == [b"pid-reused", b"pid-reused"]


def test_unpooled_handle_is_imported_every_time() -> None:
    module = _FakeXpu()
    exporter, importer = _backend(pool=False), _backend(module)
    handle = exporter.export_event(exporter.create_event("xpu:0"), "xpu:0")
    importer.import_event(handle, "xpu:0")
    importer.import_event(handle, "xpu:0")
    assert module.imports == [b"h1", b"h1"]


def test_import_remembers_completion() -> None:
    exporter, importer = _backend(), _backend()
    handle = exporter.export_event(exporter.create_event("xpu:0"), "xpu:0")
    view = importer.import_event(handle, "xpu:0")
    raw = _raw(view)

    raw.done = False
    assert not importer.query_event(view)
    raw.done = True
    assert importer.query_event(view)

    raw.done = False  # the exporter re-recorded it for a later transfer
    raw.calls.clear()
    assert importer.query_event(view)
    importer.synchronize_event(view, "xpu:0")
    importer.wait_event(view, "stream")
    assert raw.calls == []


def test_synchronize_marks_import_complete() -> None:
    exporter, importer = _backend(), _backend()
    handle = exporter.export_event(exporter.create_event("xpu:0"), "xpu:0")
    view = importer.import_event(handle, "xpu:0")
    raw = _raw(view)
    importer.synchronize_event(view, "xpu:0")
    raw.done = False
    assert importer.query_event(view)
    assert raw.calls == ["synchronize"]


def test_early_reuse_is_counted() -> None:
    exporter, importer = _backend(), _backend()
    handle = exporter.export_event(exporter.create_event("xpu:0"), "xpu:0")

    unchecked = importer.import_event(handle, "xpu:0")
    checked = importer.import_event(handle, "xpu:0")
    assert importer.early_reuse_count == 1

    assert importer.query_event(checked)
    waited = importer.import_event(handle, "xpu:0")
    assert importer.early_reuse_count == 1

    importer.wait_event(waited, "stream")
    importer.import_event(handle, "xpu:0")
    assert importer.early_reuse_count == 1
    del unchecked


# Configuration


@pytest.mark.parametrize("value", [None, "0", "off", "abc"])
def test_pool_is_off_unless_enabled(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(ENV_XPU_EVENT_POOL, raising=False)
    else:
        monkeypatch.setenv(ENV_XPU_EVENT_POOL, value)
    module = _FakeXpu()
    backend = XpuEventIPCBackend(event_module=module)
    assert isinstance(backend.create_event("xpu:0"), module.Event)


@pytest.mark.parametrize("value", ["1", "true", "ON"])
def test_env_enables_pool(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENV_XPU_EVENT_POOL, value)
    module = _FakeXpu()
    backend = XpuEventIPCBackend(event_module=module)
    assert not isinstance(backend.create_event("xpu:0"), module.Event)


@pytest.mark.parametrize("value", ["abc", "-1", "nan"])
def test_invalid_env_delay_falls_back(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(ENV_XPU_EVENT_POOL, "1")
    monkeypatch.setenv(ENV_XPU_EVENT_POOL_REUSE_DELAY_S, value)
    clock = _Clock()
    backend = XpuEventIPCBackend(event_module=_FakeXpu(), clock=clock)
    event = backend.create_event("xpu:0")
    raw = _raw(event)
    del event
    held = [backend.create_event("xpu:0")]
    clock.now = 0.5
    held.append(backend.create_event("xpu:0"))
    assert all(_raw(e) is not raw for e in held)
    clock.now = 1.0
    assert _raw(backend.create_event("xpu:0")) is raw


def test_env_delay_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_XPU_EVENT_POOL, "1")
    monkeypatch.setenv(ENV_XPU_EVENT_POOL_REUSE_DELAY_S, "0")
    backend = XpuEventIPCBackend(event_module=_FakeXpu(), clock=_Clock())
    event = backend.create_event("xpu:0")
    raw = _raw(event)
    del event
    assert _raw(backend.create_event("xpu:0")) is raw
