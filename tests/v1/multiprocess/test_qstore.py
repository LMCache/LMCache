# SPDX-License-Identifier: Apache-2.0
"""Unit tests for QStoreModule: the server side of query-tensor transfer.
Covers Q ring registration/liveness/reaping, IPC release ordering, handler and
protocol wiring, store_q's fail-closed guards, and composition through
``--server-module`` (the module is an opt-in plugin, never auto-discovered).
"""

# Standard
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch
import contextlib
import time

# Third Party
import pytest

# First Party
from lmcache.v1.common.custom_types import TRANSFER_QUERY
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
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


def test_unified_drop_releases_q_state_idempotently(monkeypatch) -> None:
    """Main instance unregister can release optional Q state exactly once."""
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
    context = MagicMock(num_layers=2)
    context.close.side_effect = lambda: calls.append("close")
    monkeypatch.setattr(
        qstore_mod, "create_cache_context", MagicMock(return_value=context)
    )
    monkeypatch.setattr(qstore_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    _register(module)

    module.drop_instance_state(1)
    module.drop_instance_state(1)

    assert calls == ["close", "empty_cache", "ipc_collect"]
    assert module.tracked_instance_count() == 0


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
    assert management.get_experimental() == []
    touch = MagicMock()
    monkeypatch.setattr(events, "touch_instance", touch)
    management.ping(123)
    touch.assert_called_once_with(123)


# Q ring state for SDK transfers through the ring.


class _Done:
    """Completion event double: ``query()`` reports whether copies finished."""

    def __init__(self, done: bool) -> None:
        self.done = done

    def query(self) -> bool:
        return self.done


def _ring_state(
    module: QStoreModule,
    instance_id: int = 1,
    model_name: str = "model##query",
    world_size: int = 2,
) -> qstore_mod._QRingState:
    """Track a ring for ``instance_id`` as register_q_cache does."""
    state = qstore_mod._QRingState(
        q_caches=[MagicMock(name=f"wrapper-{instance_id}")],
        model_name=model_name,
        world_size=world_size,
    )
    module._q_rings[instance_id] = state
    return state


def test_register_tracks_an_unused_ring(monkeypatch) -> None:
    _stub_registration(monkeypatch)
    module = _module()

    _register(module)

    state = module._q_rings[1]
    assert state.used_blocks == set()
    assert (state.num_alloc, state.worker_id) == (0, -1)
    assert (state.model_name, state.world_size) == REGISTER_ARGS


def test_report_replaces_the_used_view_and_ignores_older_reports() -> None:
    module = _module()
    state = _ring_state(module)

    module.report_q_ring_state(1, 0, [0, 1], 5)
    module.report_q_ring_state(1, 0, [0], 4)  # arrived late

    assert (state.used_blocks, state.num_alloc, state.worker_id) == ({0, 1}, 5, 0)


def test_report_for_an_unknown_ring_is_ignored() -> None:
    _module().report_q_ring_state(42, 0, [0], 1)


def test_completed_store_blocks_leave_the_used_view() -> None:
    """A store finishing after vLLM's last step: vLLM frees its blocks only on
    its next step, so the server frees them itself."""
    module = _module()
    state = _ring_state(module)
    module.report_q_ring_state(1, 0, [0, 1, 2], 1)
    state.pending_stores = [(_Done(True), [0, 1]), (_Done(False), [2])]

    (ring,) = module.get_q_ring(*REGISTER_ARGS)

    assert ring.used_blocks == [2]
    assert [blocks for _, blocks in state.pending_stores] == [[2]]


def test_a_newer_report_wins_over_a_completed_store() -> None:
    """vLLM frees a stored block only after the store completed; a later
    report that shows it taken again (reallocated) must win."""
    module = _module()
    state = _ring_state(module)
    state.pending_stores = [(_Done(True), [0])]

    module.report_q_ring_state(1, 0, [0], 1)
    (ring,) = module.get_q_ring(*REGISTER_ARGS)

    assert ring.used_blocks == [0]


def test_get_q_ring_filters_by_model_and_sorts_by_rank() -> None:
    module = _module()
    _ring_state(module, instance_id=7)
    _ring_state(module, instance_id=3)
    _ring_state(module, instance_id=9, model_name="other##query")
    module.report_q_ring_state(7, 0, [0], 1)
    module.report_q_ring_state(3, 1, [0], 1)

    rings = module.get_q_ring(*REGISTER_ARGS)

    assert [(r.instance_id, r.worker_id) for r in rings] == [(7, 0), (3, 1)]
    assert rings[0].q_caches == module._q_rings[7].q_caches


def test_store_q_tracks_its_blocks_until_completed(stub_device, monkeypatch) -> None:
    ctx = _ctx()
    ctx.resolve_obj_keys.return_value = [["obj-0"]]
    ctx.storage_manager.reserve_write.return_value = {}
    monkeypatch.setattr(
        qstore_mod, "downsample_and_stage_block_ids", lambda _c, ids: ids
    )
    monkeypatch.setattr(qstore_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    transfers: list[int] = []
    monkeypatch.setattr(
        qstore_mod,
        "transfer_kv_per_object_group",
        lambda *a, object_group_id, **kw: transfers.append(object_group_id),
    )
    module = _module(ctx)
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1
    cache_context.calculate_num_blocks.return_value = 2
    module._q_contexts[1] = ContextEntry(
        cache_context, *REGISTER_ARGS, time.monotonic(), event_backend=stub_device
    )
    state = _ring_state(module)

    _, ok = module.store_q(MagicMock(), 1, [[0, 1]], b"peer-handle")

    assert ok
    assert transfers == [0]
    ctx.storage_manager.reserve_write.assert_called_once()
    assert [blocks for _, blocks in state.pending_stores] == [[0, 1]]


def test_unregister_drops_the_ring_state(monkeypatch) -> None:
    _stub_registration(monkeypatch)
    module = _module()
    _register(module)

    module.unregister_q_cache(1)

    assert 1 not in module._q_rings


# SDK transfers: the engine's L1 objects through the Q ring, as bytes.

# Ring of 2 layers x 4 blocks; a block is [2, 4] float32 per layer: 32-byte
# pieces. A dense model's attention object (group 0) is 64 bytes (2 pieces).
_SDK_LAYERS = 2
_SDK_BLOCKS = 4
_SDK_SIZE = 16  # float32 elements per object


class _Obj:
    """L1 memory object double over a flat float32 tensor."""

    def __init__(self, values: Any) -> None:
        # Third Party
        import torch

        self.data = torch.as_tensor(values, dtype=torch.float32)

    @property
    def raw_tensor(self):
        return self.data.view(-1)

    def get_size(self) -> int:
        return self.data.numel() * 4

    def parent(self) -> None:
        return None


def _sdk_module(stub_device, monkeypatch, attn_desc: Any = None):
    """A module with a Q ring (instance 1) and the engine's KV registration of
    a dense model."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.common.attn_window import AttnWindowDesc

    ctx = _ctx()
    ctx.chunk_size = 4
    keys = {0: ["a0", "a1", "a2"]}
    ctx.resolve_obj_keys.side_effect = lambda key, groups: [keys[g] for g in groups]
    kv_context = MagicMock(name="kv_context")
    kv_context.kv_layer_groups_manager.num_object_groups = 1
    kv_context.kv_layer_groups_manager.get_attn_desc.return_value = (
        attn_desc or AttnWindowDesc(num_chunks_in_sw=[-1], group_kinds=("attention",))
    )
    kv_module = MagicMock(name="kv_module")
    kv_module.context_entries_snapshot.return_value = {
        9: ContextEntry(kv_context, "m", 1, time.monotonic())
    }
    module = QStoreModule(ctx, kv_module)
    ring_context = MagicMock(name="ring_context")
    ring_context.kv_tensors = [
        torch.zeros(_SDK_BLOCKS, 2, 4) for _ in range(_SDK_LAYERS)
    ]
    module._q_contexts[1] = ContextEntry(
        ring_context, *REGISTER_ARGS, time.monotonic(), event_backend=stub_device
    )
    _ring_state(module)
    monkeypatch.setattr(
        qstore_mod,
        "get_layout_desc",
        lambda _c, _n, object_group_id: SimpleNamespace(
            shapes=[torch.Size([_SDK_SIZE])], dtypes=[torch.float32]
        ),
    )
    callbacks: list = []
    monkeypatch.setattr(
        qstore_mod,
        "submit_callback_to_stream",
        lambda _stream, kind, payload: callbacks.append((kind, payload)),
    )
    return module, ctx, ring_context, callbacks


def _sdk_key() -> Any:
    return SimpleNamespace(model_name="m", world_size=1, request_id="req")


def _piece(ring_context, piece_id: int):
    block, layer = divmod(piece_id, _SDK_LAYERS)
    return ring_context.kv_tensors[layer][block].reshape(-1)


def test_ring_imports_are_held_on_the_workers_kv_registration(stub_device) -> None:
    """The transfer module's release looks the import up by instance ID, which
    finds the worker's KV registration, so the ring holds it there."""
    kv_module = MagicMock(name="kv_module")
    module = QStoreModule(_ctx(), kv_module)
    ring_context = MagicMock(name="ring_context")
    ring = ContextEntry(
        ring_context, *REGISTER_ARGS, time.monotonic(), event_backend=stub_device
    )
    kv_context = kv_module.get_and_touch_context_entry.return_value.cache_context
    kv_context.hold_imported_event.return_value = 7

    with patch.object(qstore_mod, "submit_callback_to_stream") as submit:
        module._wait_for_sdk_gpu_copy(ring, 1, b"peer")

    kv_module.get_and_touch_context_entry.assert_called_once_with(1)
    ring_context.hold_imported_event.assert_not_called()
    assert submit.call_args.args[1:] == ("release_imported_event", (1, 7))


def test_ring_imports_without_kv_registration_synchronize(stub_device) -> None:
    module = QStoreModule(_ctx(), None)
    ring_context = MagicMock(name="ring_context")
    ring = ContextEntry(
        ring_context, *REGISTER_ARGS, time.monotonic(), event_backend=stub_device
    )

    with patch.object(qstore_mod, "submit_callback_to_stream") as submit:
        module._wait_for_sdk_gpu_copy(ring, 1, b"peer")

    ring_context.stream.synchronize.assert_called_once_with()
    submit.assert_not_called()


def test_sdk_retrieve_copies_objects_into_pieces_in_chunk_order(
    stub_device, monkeypatch
) -> None:
    # Third Party
    import torch

    module, ctx, ring_context, callbacks = _sdk_module(stub_device, monkeypatch)
    objects = {f"a{i}": _Obj(torch.arange(16.0) + 100 * i) for i in range(3)}
    ctx.storage_manager.unsafe_read.side_effect = lambda keys, l1_owners: (
        keys,
        [objects[k] for k in keys],
    )
    pieces = [7, 0, 1, 2, 3, 4]

    handle, ok = module.sdk_retrieve(_sdk_key(), 1, [0], pieces, b"peer")

    assert (handle, ok) == (b"event-handle", True)
    for chunk in range(3):
        got = torch.cat(
            [_piece(ring_context, p) for p in pieces[2 * chunk : 2 * chunk + 2]]
        )
        assert torch.equal(got, torch.arange(16.0) + 100 * chunk)
    assert ctx.storage_manager.prepare_read_completion.call_args.args[0] == [
        "a0",
        "a1",
        "a2",
    ]
    assert [kind for kind, _ in callbacks] == [
        "release_imported_event",
        "finish_read_by_owner",
    ]


@pytest.mark.parametrize(
    "pieces",
    [
        [0, 1, 2, 3, 4, 6],  # block 3 (pieces 6, 7) is used by vLLM
        [0, 1, 2, 3, 4],  # one piece short
        [0, 0, 1, 2, 3, 4],  # repeated
    ],
)
def test_sdk_retrieve_rejects_bad_pieces_and_releases_the_locks(
    stub_device, monkeypatch, pieces
) -> None:
    module, ctx, _, _ = _sdk_module(stub_device, monkeypatch)
    module._q_rings[1].used_blocks = {3}
    storage = ctx.storage_manager

    assert module.sdk_retrieve(_sdk_key(), 1, [0], pieces, b"h") == (b"", False)
    assert storage.prepare_read_completion.call_args.args[0] == ["a0", "a1", "a2"]
    storage.finish_read_by_owner.assert_called_once_with(
        storage.prepare_read_completion.return_value
    )
    storage.unsafe_read.assert_not_called()


def test_sdk_retrieve_without_the_kv_registration_is_rejected(
    stub_device, monkeypatch
) -> None:
    """Without a registration the keys cannot be resolved; the locks expire
    with the read-lock TTL."""
    module, ctx, _, _ = _sdk_module(stub_device, monkeypatch)
    key = SimpleNamespace(model_name="other", world_size=1, request_id="req")

    assert module.sdk_retrieve(key, 1, [0], [0, 1, 2, 3, 4, 5], b"h") == (
        b"",
        False,
    )
    ctx.storage_manager.finish_read_by_owner.assert_not_called()


@pytest.mark.parametrize(
    ("window", "kind"), [([2], "attention"), ([-1], "recurrent"), ([-1], "aux")]
)
def test_sdk_transfers_reject_groups_that_are_not_full_attention(
    stub_device, monkeypatch, window, kind
) -> None:
    # First Party
    from lmcache.v1.common.attn_window import AttnWindowDesc

    attn_desc = AttnWindowDesc(num_chunks_in_sw=window, group_kinds=(kind,))
    module, ctx, _, _ = _sdk_module(stub_device, monkeypatch, attn_desc)

    assert module.sdk_retrieve(_sdk_key(), 1, [0], [0, 1, 2, 3, 4, 5], b"h") == (
        b"",
        False,
    )
    ctx.storage_manager.unsafe_read.assert_not_called()


def test_sdk_store_writes_pieces_into_reserved_objects_and_commits(
    stub_device, monkeypatch
) -> None:
    # Third Party
    import torch

    module, ctx, ring_context, callbacks = _sdk_module(stub_device, monkeypatch)
    reserved: dict[str, _Obj] = {}

    def reserve(keys, layout):
        objs = {k: _Obj(torch.zeros(layout.shapes[0])) for k in keys}
        reserved.update(objs)
        return objs

    ctx.storage_manager.reserve_write.side_effect = reserve
    monkeypatch.setattr(qstore_mod, "l1_owners_of", lambda d: {k: 0 for k in d})
    pieces = [6, 0, 1, 2, 3, 4]
    for index, piece_id in enumerate(pieces):
        _piece(ring_context, piece_id)[:] = index

    handle, ok = module.sdk_store(_sdk_key(), 1, [0], pieces, b"peer")

    assert (handle, ok) == (b"event-handle", True)
    assert sorted(reserved) == ["a0", "a1", "a2"]
    for chunk in range(3):
        expected = torch.cat(
            [torch.full((8,), 2.0 * chunk), torch.full((8,), 2.0 * chunk + 1)]
        )
        assert torch.equal(reserved[f"a{chunk}"].data, expected)
    assert callbacks[-1] == ("finish_write", {"a0": 0, "a1": 0, "a2": 0})


def test_sdk_store_rejects_pieces_in_blocks_vllm_holds(
    stub_device, monkeypatch
) -> None:
    module, ctx, _, _ = _sdk_module(stub_device, monkeypatch)
    module._q_rings[1].used_blocks = {0}

    assert module.sdk_store(_sdk_key(), 1, [0], [0, 2, 3, 4, 5, 6], b"h") == (
        b"",
        False,
    )
    ctx.storage_manager.reserve_write.assert_not_called()


@pytest.mark.parametrize("to_ring", [True, False])
def test_lazy_l1_objects_copy_through_the_boundary_splitting_helper(
    monkeypatch, to_ring
) -> None:
    """The lazy allocator pins L1 in separately registered chunks; a plain
    cudaMemcpy spanning two fails, so each piece goes through the native copy
    with its offset in the allocator."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_allocators.lazy_memory_allocator import (
        LazyMemoryAllocator,
    )

    calls: list[tuple] = []
    monkeypatch.setattr(
        qstore_mod,
        "device_ops",
        SimpleNamespace(lmcache_memcpy_async=lambda *args: calls.append(args)),
    )
    pieces = [torch.zeros(2, 32, dtype=torch.uint8) for _ in range(_SDK_LAYERS)]
    obj = MagicMock()
    obj.get_size.return_value = 48  # one full piece and half of another
    obj.parent.return_value = MagicMock(spec=LazyMemoryAllocator)
    obj.raw_tensor = torch.zeros(48, dtype=torch.uint8)
    obj.data_ptr = 1000
    obj.meta.address = 5000

    qstore_mod._copy_object(pieces, [3, 0], obj, to_ring=to_ring)

    block1_layer1, block0_layer0 = pieces[1][1], pieces[0][0]
    hosts = [(1000, 5000, 32), (1032, 5032, 16)]
    devices = [block1_layer1.data_ptr(), block0_layer0.data_ptr()]
    for call, (host, offset, nbytes), dev in zip(calls, hosts, devices, strict=True):
        dest, src, size, _direction, host_offset, alignment = call
        assert (dest, src) == ((dev, host) if to_ring else (host, dev))
        assert (size, host_offset) == (nbytes, offset)
        assert alignment == LazyMemoryAllocator.PIN_CHUNK_SIZE


def test_sdk_retrieve_reads_query_objects_with_the_rings_own_layout(
    stub_device, monkeypatch
) -> None:
    """Query objects were stored by STORE_Q with the Q ring's registration, so
    a raw retrieve of ``<model>##query`` takes its layout from it."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.common.attn_window import AttnWindowDesc

    module, ctx, ring_context, callbacks = _sdk_module(stub_device, monkeypatch)
    ring_context.kv_layer_groups_manager.num_object_groups = 1
    ring_context.kv_layer_groups_manager.get_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], group_kinds=("attention",)
    )
    layouts: list = []

    def layout_desc(context, _n, object_group_id):
        layouts.append(context)
        return SimpleNamespace(shapes=[torch.Size([8])], dtypes=[torch.float32])

    monkeypatch.setattr(qstore_mod, "get_layout_desc", layout_desc)
    ctx.resolve_obj_keys.side_effect = lambda key, groups: [["q0"]]
    objects = {"q0": _Obj(torch.full((8,), 7.0))}
    ctx.storage_manager.unsafe_read.side_effect = lambda keys, l1_owners: (
        keys,
        [objects[k] for k in keys],
    )
    key = SimpleNamespace(model_name="model##query", world_size=2, request_id="r")

    assert module.sdk_retrieve(key, 1, [0], [5], b"peer") == (b"event-handle", True)
    assert layouts == [ring_context]
    assert torch.equal(_piece(ring_context, 5), torch.full((8,), 7.0))
    assert ctx.storage_manager.prepare_read_completion.call_args.args[0] == ["q0"]


def test_status_reports_object_layouts_from_the_layout_registry() -> None:
    """An SDK decodes raw objects with the layouts the server stores them with:
    the engine's KV model and the Q ring."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.common.attn_window import AttnWindowDesc
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry

    ctx = _ctx()
    ctx.layout_desc_registry = LayoutDescRegistry()
    attention = MemoryLayoutDesc([torch.Size([2, 36, 256, 1024])], [torch.bfloat16])
    ctx.layout_desc_registry.register(
        "m",
        1,
        attention,
        AttnWindowDesc(num_chunks_in_sw=[-1], group_kinds=("attention",)),
    )
    query = MemoryLayoutDesc([torch.Size([1, 36, 256, 4096])], [torch.bfloat16])
    ctx.layout_desc_registry.register(
        "m##query",
        1,
        query,
        AttnWindowDesc(num_chunks_in_sw=[-1], group_kinds=("attention",)),
    )
    kv_module = MagicMock(name="kv_module")
    kv_module.context_entries_snapshot.return_value = {
        9: ContextEntry(MagicMock(), "m", 1, time.monotonic())
    }
    module = QStoreModule(ctx, kv_module)
    ring = MagicMock()
    ring.report_status.return_value = {}
    module._q_contexts[1] = ContextEntry(ring, "m##query", 1, time.monotonic())

    layouts = module.report_status()["object_layouts"]

    assert layouts == [
        {
            "model_name": name,
            "world_size": 1,
            "object_groups": [
                {
                    "object_group_idx": 0,
                    "kind": "attention",
                    "num_chunks_in_sw": -1,
                    "chunk_object_shapes": [shape],
                    "chunk_object_dtypes": ["torch.bfloat16"],
                }
            ],
        }
        for name, shape in [
            ("m", [2, 36, 256, 1024]),
            ("m##query", [1, 36, 256, 4096]),
        ]
    ]
