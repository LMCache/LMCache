# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QStoreModule: the server side of query-tensor transfer.
Covers Q ring registration/liveness/reaping, IPC release ordering, handler and
protocol wiring, store_q's fail-closed guards, and composition through
``--server-module`` (the module is an opt-in plugin, never auto-discovered).
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import contextlib
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.custom_types import KV_EVENT_CAPABILITY, TRANSFER_QUERY
from lmcache.v1.multiprocess.engine_module import discover_modules
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    load_server_module_classes,
)
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import ContextEntry
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.plugin_modules import qstore as qstore_mod
from lmcache.v1.multiprocess.plugin_modules.qstore import QStoreModule

REGISTER_ARGS = ("model##query", 2)
QSTORE_SPEC = ExtServerModuleSpec("lmcache.v1.multiprocess.plugin_modules.qstore")


def _ctx() -> MagicMock:
    """Creates a context stub to test QStoreModule."""
    ctx = MagicMock(name="ctx")
    ctx.chunk_size = 256
    ctx.separate_object_groups = False
    ctx.full_sw_kv = False
    return ctx


def _module(ctx: MagicMock | None = None) -> QStoreModule:
    """Creates a QStoreModule with a stub context."""
    return QStoreModule(ctx or _ctx())


def _stub_registration(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Creates a stub for the create_cache_context call to test
    register_q_cache()."""
    create = MagicMock(return_value=MagicMock(num_layers=2))
    event_backend = MagicMock(name="event_backend")
    monkeypatch.setattr(qstore_mod, "create_cache_context", create)
    monkeypatch.setattr(qstore_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        qstore_mod, "get_event_ipc_backend", lambda device: event_backend
    )
    return create


def _register(module: QStoreModule, instance_id: int = 1) -> None:
    """Registers a Q ring with the module, using the stubbed
    create_cache_context."""
    module.register_q_cache(
        instance_id, MagicMock(), *REGISTER_ARGS, MagicMock(), MagicMock(), []
    )


def test_register_inserts_unlatched_entry(monkeypatch) -> None:
    """Checks that successful registered entries inserts an entry to be
    tracked."""
    _stub_registration(monkeypatch)
    module = _module()

    _register(module)

    assert module.tracked_instance_count() == 1
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None


def test_duplicate_register_refreshes_without_rebuilding(monkeypatch) -> None:
    """Checks that a duplicate registration refreshes the last_seen timestamp and
    does not create a second context or layout descriptor."""
    create = _stub_registration(monkeypatch)
    module = _module()
    _register(module)
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None
    entry.last_seen = 0.0

    _register(module)

    assert create.call_count == 1
    assert module.tracked_instance_count() == 1
    assert entry.last_seen > 0.0


def test_reap_uses_two_tier_windows() -> None:
    """Check that the reaper uses two time windows to decide which entries to drop.
    This is a verbatim copy from tests/v1/multiprocess/test_worker_liveness.py"""
    module = _module()
    old = time.monotonic() - 1000.0
    module._q_contexts.update(
        {
            1: ContextEntry(MagicMock(), *REGISTER_ARGS, old, True),
            2: ContextEntry(MagicMock(), *REGISTER_ARGS, old, False),
            3: ContextEntry(MagicMock(), *REGISTER_ARGS, time.monotonic(), True),
        }
    )

    assert module.reap_stale_instances(120.0, 3600.0) == [1]
    assert module.tracked_instance_count() == 2
    cast(
        MagicMock, module.context.layout_desc_registry.unregister
    ).assert_called_once_with(*REGISTER_ARGS)


def test_unregister_releases_context_and_layout(monkeypatch) -> None:
    """Check that unregister_q_cache releases the context and layout descriptor, and
    that it tolerates repeated calls for the same instance."""
    calls: list[str] = []
    monkeypatch.setattr(
        qstore_mod,
        "torch_dev",
        SimpleNamespace(
            empty_cache=lambda: calls.append("empty_cache"),
            ipc_collect=lambda: calls.append("ipc_collect"),
        ),
    )
    module = _module()
    context = MagicMock()
    context.close.side_effect = lambda: calls.append("close")
    module._q_contexts[1] = ContextEntry(context, *REGISTER_ARGS, time.monotonic())

    module.unregister_q_cache(1)

    assert module.tracked_instance_count() == 0
    # Reclaim must come after close so the last IPC tensor reference is gone
    # before ipc_collect tries to unmap the segment (LMCache#4014).
    assert calls == ["close", "empty_cache", "ipc_collect"]
    cast(
        MagicMock, module.context.layout_desc_registry.unregister
    ).assert_called_once_with(*REGISTER_ARGS)

    module.unregister_q_cache(1)  # already gone -> no exception


def test_release_tolerates_backends_without_ipc_collect(monkeypatch) -> None:
    """xpu/musa device modules expose no ipc_collect; release must not fail."""
    monkeypatch.setattr(
        qstore_mod, "torch_dev", SimpleNamespace(empty_cache=lambda: None)
    )
    module = _module()
    module._q_contexts[1] = ContextEntry(MagicMock(), *REGISTER_ARGS, time.monotonic())

    module.unregister_q_cache(1)

    assert module.tracked_instance_count() == 0


class _FakeEvent:
    """Device event double exposing the record/ipc_handle surface store_q uses."""

    created: list["_FakeEvent"] = []

    def __init__(self, interprocess: bool = False) -> None:
        self.records = 0
        _FakeEvent.created.append(self)

    def record(self, stream=None) -> None:
        self.records += 1

    def ipc_handle(self) -> bytes:
        return b"event-handle"

    def wait(self, stream=None) -> None:
        return None

    @classmethod
    def from_ipc_handle(cls, device, handle: bytes) -> "_FakeEvent":
        return cls()


class _FakeEventBackend:
    def __init__(self) -> None:
        self.created: list[_FakeEvent] = []

    def create_event(self, device) -> _FakeEvent:
        event = _FakeEvent()
        self.created.append(event)
        return event

    def check_event_support(self, device) -> None:
        return None

    def record_event(self, event: _FakeEvent, stream=None) -> None:
        event.record(stream)

    def export_event(self, event: _FakeEvent, device) -> bytes:
        return event.ipc_handle()

    def import_event(self, handle: bytes, device) -> _FakeEvent:
        return _FakeEvent.from_ipc_handle(device, handle)

    def wait_event(self, event: _FakeEvent, stream=None) -> None:
        event.wait(stream)


@pytest.fixture
def stub_device(monkeypatch):
    """Replace the device module so store_q runs on CPU."""

    @contextlib.contextmanager
    def _null(*args, **kwargs):
        yield

    _FakeEvent.created.clear()
    backend = _FakeEventBackend()
    monkeypatch.setattr(
        qstore_mod,
        "torch_dev",
        SimpleNamespace(device=_null, stream=_null, Event=_FakeEvent),
    )
    monkeypatch.setattr(qstore_mod, "get_event_ipc_backend", lambda device: backend)
    return backend


def test_store_q_unregistered_instance_raises() -> None:
    """A store for an unknown instance is a protocol error."""
    with pytest.raises(ValueError, match="No Q ring registered"):
        _module().store_q(MagicMock(), 42, [[0]], b"handle")


def test_store_q_block_id_underflow_fails_closed(stub_device) -> None:
    """Short block-id lists would drive the transfer kernel out of bounds, so
    the whole store is skipped: nothing reserved, failure reported, event still
    recorded so the waiting worker is released."""
    ctx = _ctx()
    ctx.resolve_obj_keys.return_value = [["obj-0", "obj-1"]]  # 2 chunks
    module = _module(ctx)
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1
    cache_context.calculate_num_blocks.return_value = 2  # 2 chunks * 2 = 4 needed
    module._q_contexts[1] = ContextEntry(
        cache_context,
        *REGISTER_ARGS,
        time.monotonic(),
        event_backend=stub_device,
    )

    handle, ok = module.store_q(MagicMock(), 1, [[0, 1, 2]], b"peer-handle")

    assert ok is False
    assert handle == b"event-handle"
    assert stub_device.created[0].records == 1
    cast(MagicMock, ctx.storage_manager.reserve_write).assert_not_called()
    cast(MagicMock, ctx.event_bus.publish).assert_not_called()


def _build(**config: Any) -> list:
    config.setdefault("kv_event_log_size", 0)
    return server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(worker_reap_timeout_seconds=0, **config),
        CoordinatorConfig(),
    )


def test_qstore_is_not_auto_discovered() -> None:
    """QStore is an opt-in plugin: absent from the built-in scan, and loading
    its module path yields exactly that one class (its imported dependency
    LMCacheDrivenTransferModule is not re-registered)."""
    discovered = discover_modules()

    assert QStoreModule not in discovered
    assert not any(
        cls.__module__.startswith("lmcache.v1.multiprocess.plugin_modules")
        for cls in discovered
    )
    assert load_server_module_classes([QSTORE_SPEC]) == [(QStoreModule, {})]


def test_server_requires_lmcache_driven_transfer(stub_module_resources) -> None:
    """Check that the server rejects the qstore module when the transfer mode is
    not lmcache-driven."""
    with pytest.raises(ValueError, match="lmcache_driven"):
        _build(
            server_modules=[QSTORE_SPEC],
            supported_transfer_mode="engine_driven",
        )


def test_server_builds_q_store_module(
    stub_module_resources, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check that --server-module builds the Q store module after its transfer
    dependency, advertises transfer_query, and wires it to the reaper."""
    modules = _build(
        server_modules=[QSTORE_SPEC],
        supported_transfer_mode="lmcache_driven",
    )

    qstore = next(m for m in modules if isinstance(m, QStoreModule))
    management = next(m for m in modules if isinstance(m, ManagementModule))
    assert management.get_experimental() == [TRANSFER_QUERY]
    touch = MagicMock()
    monkeypatch.setattr(qstore, "touch_instance", touch)
    management.ping(123)
    touch.assert_called_once_with(123)


def test_server_omits_qstore_without_server_module(
    stub_module_resources,
) -> None:
    """Check that QStore and its advertised feature are absent by default."""
    modules = _build()

    assert not any(isinstance(m, QStoreModule) for m in modules)
    management = next(m for m in modules if isinstance(m, ManagementModule))
    assert management.get_experimental() == []


@pytest.mark.parametrize(
    ("log_size", "bus_enabled", "enabled"),
    [(0, True, False), (3, False, False), (3, True, True)],
)
def test_server_wires_event_module_liveness(
    stub_module_resources,
    monkeypatch: pytest.MonkeyPatch,
    log_size: int,
    bus_enabled: bool,
    enabled: bool,
) -> None:
    ctx = MagicMock(name="ctx")
    ctx.event_bus.enabled = bus_enabled
    modules = server_mod._build_modules(
        ctx,
        MPServerConfig(worker_reap_timeout_seconds=0, kv_event_log_size=log_size),
        CoordinatorConfig(),
    )
    events = next(module for module in modules if isinstance(module, KVEventModule))
    management = next(m for m in modules if isinstance(m, ManagementModule))
    assert events.enabled is enabled
    assert management.get_experimental() == ([KV_EVENT_CAPABILITY] if enabled else [])
    touch = MagicMock()
    monkeypatch.setattr(events, "touch_instance", touch)
    management.ping(123)
    touch.assert_called_once_with(123)
