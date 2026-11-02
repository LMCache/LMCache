# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server management: liveness and CPU KV event subscriptions.

Cover the public liveness interfaces of the transfer modules, the
management reaper wiring, the blend reap listener, and config validation.
No GPU or live server is required; module-level construction dependencies
are stubbed.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Literal, cast
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch
import zmq

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.engine_module import ModuleBuildContext
from lmcache.v1.multiprocess.modules import engine_driven_transfer as non_gpu_mod
from lmcache.v1.multiprocess.modules import kv_events as kv_events_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as gpu_mod
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from lmcache.v1.periodic_thread import PeriodicThreadRegistry
from tests.v1.multiprocess.transport_test_utils import requires_pygrpc_native


@pytest.mark.parametrize("mode", ["lmcache_driven", "engine_driven", "auto"])
def test_transfer_factories_gate_modules(
    mode: Literal["lmcache_driven", "engine_driven", "auto"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_ctx = ModuleBuildContext(
        MagicMock(), MPServerConfig(supported_transfer_mode=mode), CoordinatorConfig()
    )
    for module_type, supported in (
        (LMCacheDrivenTransferModule, ("lmcache_driven", "auto")),
        (EngineDrivenTransferModule, ("engine_driven", "auto")),
    ):
        monkeypatch.setattr(module_type, "__init__", MagicMock(return_value=None))
        assert isinstance(module_type.create(build_ctx), module_type) == (
            mode in supported
        )


def _event_module(
    size: int = 3,
    bus: EventBus | None = None,
    world_size: int = 1,
    groups: int = 1,
) -> tuple[KVEventModule, EventBus]:
    bus = bus or EventBus(EventBusConfig())
    layouts = LayoutDescRegistry()
    for model in ("model", "other"):
        layouts.register(
            model,
            world_size,
            MemoryLayoutDesc([torch.Size([1])], [torch.float16]),
            AttnWindowDesc(num_chunks_in_sw=[-1] * groups),
        )
    ctx = MagicMock(event_bus=bus, layout_desc_registry=layouts)
    return KVEventModule(ctx, kv_event_log_size=size), bus


def _publish_kv(bus: EventBus, event_type: EventType, **metadata: Any) -> None:
    if event_type != EventType.MP_TOKENS:
        metadata.setdefault("l1_tag", "_default")
    bus.publish(Event(event_type, metadata=metadata))
    bus.stop()  # Public shutdown flushes queued callbacks synchronously.


def _event_key(
    index: int, model: str = "model", rank: int = 0, world_size: int = 1
) -> ObjectKey:
    return ObjectKey(
        bytes([index]) * 32,
        model,
        ObjectKey.ComputeKVRank(world_size, rank, world_size, rank),
    )


def _bind_kv(bus: EventBus, keys: list[ObjectKey]) -> None:
    hashes = [key.chunk_hash for key in keys]
    _publish_kv(
        bus,
        EventType.MP_TOKENS,
        chunk_hashes=hashes,
        token_chunks=[[i] for i in range(len(keys))],
        token_offsets=list(range(len(keys))),
        parent_hashes=[None] + hashes[:-1],
    )


def _bare_gpu_module() -> LMCacheDrivenTransferModule:
    """A LMCacheDrivenTransferModule with only the liveness state initialized.

    Bypasses __init__ (which starts a CUDA host-func dispatcher) so the
    liveness methods can be exercised without GPU hardware.
    """
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._cache_contexts = {}
    module._lock = threading.Lock()
    return module


def _bare_non_gpu_module() -> EngineDrivenTransferModule:
    """Create an engine-driven module without hardware dependencies."""
    return EngineDrivenTransferModule(MagicMock(name="ctx"))


def _stub_gpu_registration_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub Event IPC lookup for liveness tests unrelated to event handling."""
    monkeypatch.setattr(
        gpu_mod,
        "get_event_ipc_backend",
        MagicMock(return_value=MagicMock(name="event_backend")),
    )


def test_gpu_register_inserts_unlatched_entry(monkeypatch) -> None:
    """register_kv_cache inserts an entry that is not yet ping-proven."""
    _stub_gpu_registration_backend(monkeypatch)
    monkeypatch.setattr(
        gpu_mod,
        "create_cache_context",
        lambda *a, **kw: MagicMock(
            num_layers=2, **{"kv_layer_groups_manager.num_object_groups": 1}
        ),
    )
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_gpu_module()

    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])

    assert module.tracked_instance_count() == 1
    entry = module.get_and_touch_context_entry(1)
    assert entry is not None and entry.has_liveness_signal is False


def test_gpu_noop_register_refreshes_without_latching(monkeypatch) -> None:
    """Re-registering a known instance refreshes last_seen but does not
    rebuild the context or latch the ping-proven flag."""
    _stub_gpu_registration_backend(monkeypatch)
    create = MagicMock(
        return_value=MagicMock(
            num_layers=2, **{"kv_layer_groups_manager.num_object_groups": 1}
        )
    )
    monkeypatch.setattr(gpu_mod, "create_cache_context", create)
    monkeypatch.setattr(gpu_mod, "get_layout_desc", lambda *a, **kw: MagicMock())
    module = _bare_gpu_module()
    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])
    module._cache_contexts[1].last_seen = 0.0

    module.register_kv_cache(1, MagicMock(), "model", 1, MagicMock(), MagicMock(), [])

    assert create.call_count == 1  # not rebuilt
    assert module._cache_contexts[1].last_seen > 0.0  # refreshed
    assert module._cache_contexts[1].has_liveness_signal is False


def test_gpu_touch_latches_get_does_not() -> None:
    """touch_instance marks ping-proven; get_and_touch_context_entry only refreshes."""
    module = _bare_gpu_module()
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, last_seen=0.0)

    module.get_and_touch_context_entry(1)
    assert module._cache_contexts[1].last_seen > 0.0
    assert module._cache_contexts[1].has_liveness_signal is False

    module.touch_instance(1)
    assert module._cache_contexts[1].has_liveness_signal is True
    module.touch_instance(999)  # absent -> no error


def test_gpu_reap_two_tier_windows() -> None:
    """Ping-proven entries reap at the timeout; never-pinged ones survive
    until the larger registration grace."""
    module = _bare_gpu_module()
    old = time.monotonic() - 1000.0
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, old, True)
    module._cache_contexts[2] = ContextEntry(MagicMock(), "m", 1, old, False)
    module._cache_contexts[3] = ContextEntry(
        MagicMock(), "m", 1, time.monotonic(), True
    )

    reaped = module.reap_stale_instances(120.0, 3600.0)

    assert reaped == [1]  # only the ping-proven stale entry
    assert module.tracked_instance_count() == 2
    cast(
        MagicMock, module._ctx.layout_desc_registry.unregister
    ).assert_called_once_with("m", 1)


def test_gpu_unregister_cleans_up() -> None:
    """unregister_kv_cache pops and releases; missing id is a no-op."""
    module = _bare_gpu_module()
    module._cache_contexts[1] = ContextEntry(MagicMock(), "m", 1, time.monotonic())

    module.unregister_kv_cache(1)
    assert module.tracked_instance_count() == 0
    cast(
        MagicMock, module._ctx.layout_desc_registry.unregister
    ).assert_called_once_with("m", 1)

    module.unregister_kv_cache(1)  # already gone -> no exception


def test_non_gpu_reap_pops_strategy_as_pair() -> None:
    """Reaping a non-GPU entry pops its strategy in the same scan, keeping
    'strategy present iff entry present'."""
    module = _bare_non_gpu_module()
    old = time.monotonic() - 1000.0
    module._engine_driven_contexts[1] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, old, True
    )
    module._strategies[1] = MagicMock()
    module._engine_driven_contexts[2] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, old, False
    )
    module._strategies[2] = MagicMock()

    reaped = module.reap_stale_instances(120.0, 3600.0)

    assert reaped == [1]
    assert 1 not in module._strategies  # strategy popped with the entry
    assert 2 in module._strategies  # never-pinged survives on grace


def test_non_gpu_resolve_for_transfer_refreshes_and_raises() -> None:
    """_resolve_for_transfer returns (entry, strategy) and refreshes
    last_seen; an unknown id raises ValueError."""
    module = _bare_non_gpu_module()
    module._engine_driven_contexts[1] = non_gpu_mod.EngineDrivenContextEntry(
        MagicMock(), "m", 1, 0.0, False
    )
    strategy = MagicMock()
    module._strategies[1] = strategy

    entry, resolved = module._resolve_for_transfer(1)
    assert resolved is strategy
    assert entry.last_seen > 0.0
    assert entry.has_liveness_signal is False  # traffic does not latch

    with pytest.raises(ValueError, match="not registered"):
        module._resolve_for_transfer(999)


class _FakeTarget:
    """Liveness target double recording touches/drops and scripted reaps."""

    def __init__(self) -> None:
        self.touched: list[int] = []
        self.to_reap: list[int] = []
        self.dropped: list[int] = []
        self.count = 0

    def touch_instance(self, instance_id: int) -> None:
        self.touched.append(instance_id)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        reaped = self.to_reap[:]
        self.to_reap.clear()
        return reaped

    def tracked_instance_count(self) -> int:
        return self.count

    def drop_instance_state(self, instance_id: int) -> None:
        self.dropped.append(instance_id)


@pytest.fixture(autouse=True)
def _reset_periodic_registry():
    """Keep the reaper out of the global registry across tests."""
    PeriodicThreadRegistry.reset()
    yield
    PeriodicThreadRegistry.reset()


def test_management_ping_touches_targets() -> None:
    """ping refreshes every target for a real id; None is ignored."""
    target = _FakeTarget()
    mgmt = ManagementModule(MagicMock(), liveness_targets=[target])

    assert mgmt.ping(42) is True
    assert mgmt.ping(None) is True
    assert target.touched == [42]


def test_management_clear_defaults_to_non_force() -> None:
    ctx = MagicMock()
    mgmt = ManagementModule(ctx)

    mgmt.clear()

    ctx.storage_manager.clear.assert_called_once_with(force=False)


def test_management_clear_accepts_force() -> None:
    ctx = MagicMock()
    mgmt = ManagementModule(ctx)

    mgmt.clear(force=True)

    ctx.storage_manager.clear.assert_called_once_with(force=True)


def test_management_reaper_reaps_and_drops() -> None:
    """The reaper scans targets and calls drop_instance_state for reaped ids."""
    target = _FakeTarget()
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[target],
        worker_reap_timeout_seconds=0.4,
        worker_registration_grace_seconds=0.8,
    )
    try:
        target.to_reap = [7]
        deadline = time.monotonic() + 2.0
        while target.dropped != [7] and time.monotonic() < deadline:
            time.sleep(0.02)
        assert target.dropped == [7]
    finally:
        mgmt.close()


def test_management_reaper_disabled_when_timeout_zero() -> None:
    """timeout == 0 starts no reaper thread."""
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[_FakeTarget()],
        worker_reap_timeout_seconds=0.0,
    )
    assert mgmt._reaper is None
    assert mgmt.ping(1) is True


def test_management_report_status_summarizes_liveness() -> None:
    """report_status reports a worker_liveness summary when targets exist."""
    target = _FakeTarget()
    target.count = 3
    mgmt = ManagementModule(
        MagicMock(),
        liveness_targets=[target],
        worker_reap_timeout_seconds=120.0,
        worker_registration_grace_seconds=3600.0,
    )
    try:
        status = mgmt.report_status()["worker_liveness"]
        assert status["enabled"] is True
        assert status["tracked_instances"] == 3
        assert status["reap_timeout_seconds"] == 120.0
    finally:
        mgmt.close()

    assert "worker_liveness" not in ManagementModule(MagicMock()).report_status()


def test_blend_drop_instance_state_drops_rope_state() -> None:
    """drop_instance_state pops the reaped instance's CB rope state.

    The GPU context is no longer mirrored in BlendModule (reaping the GPU
    entry frees it directly), so only the rope state is dropped here.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.blend import BlendModule

    module = BlendModule.__new__(BlendModule)
    module._cb_rope_state = {5: MagicMock()}

    module.drop_instance_state(5)

    assert 5 not in module._cb_rope_state
    module.drop_instance_state(999)  # nothing held -> no error


def test_config_rejects_bad_reap_timeouts() -> None:
    """Validation rejects sub-floor reap timeouts and undersized grace."""
    with pytest.raises(ValueError, match="reap timeout"):
        MPServerConfig(worker_reap_timeout_seconds=10.0)
    with pytest.raises(ValueError, match="registration grace"):
        MPServerConfig(
            worker_reap_timeout_seconds=120.0,
            worker_registration_grace_seconds=60.0,
        )


def test_config_accepts_disabled_and_defaults() -> None:
    """0 disables reaping; defaults satisfy the grace >= timeout invariant."""
    MPServerConfig(
        worker_reap_timeout_seconds=0.0, worker_registration_grace_seconds=0.0
    )
    default = MPServerConfig()
    assert default.worker_reap_timeout_seconds == 120.0
    assert default.worker_registration_grace_seconds == 3600.0


def test_cpu_events_preserve_tokens_parents_and_deduplicate_ranks() -> None:
    module, bus = _event_module(world_size=2)
    keys = [_event_key(1, world_size=2), _event_key(2, world_size=2)]
    _bind_kv(bus, keys)
    _publish_kv(
        bus,
        EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
        keys=[*keys, *[_event_key(i, rank=1, world_size=2) for i in (1, 2)]],
    )
    result = module.read_kv_events("model", 0, 8)
    assert [
        (r.kind, r.block_hashes, r.token_ids, r.parent_block_hash)
        for r in result.events
    ] == [
        ("stored", [keys[0].chunk_hash], [0], None),
        ("stored", [keys[1].chunk_hash], [1], keys[0].chunk_hash),
    ]
    _publish_kv(
        bus,
        EventType.L1_KEYS_EVICTED,
        keys=[keys[0], _event_key(1, rank=1, world_size=2)],
    )
    removed = module.read_kv_events("model", result.next_cursor, 8)
    assert [(r.kind, r.medium, r.block_hashes) for r in removed.events] == [
        ("removed", "CPU", [keys[0].chunk_hash]),
    ]
    assert removed.incarnation == result.incarnation and not removed.lost
    assert module.read_kv_events("model", removed.next_cursor, 8).events == []


@pytest.mark.parametrize("groups", [1, 2])
def test_cpu_events_require_every_committed_rank_and_group(groups: int) -> None:
    """A token binding and all but one committed object cannot advertise a chunk."""
    module, bus = _event_module(world_size=2, groups=groups)
    keys = [
        replace(_event_key(1, rank=rank, world_size=2), object_group_id=group)
        for rank in range(2)
        for group in range(groups)
    ]
    _bind_kv(bus, [keys[0]])
    assert module.read_kv_events("model", 0, 8).events == []
    for key in keys[:-1]:
        _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
        assert module.read_kv_events("model", 0, 8).events == []
    _publish_kv(bus, EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, keys=[keys[-1]])
    first = module.read_kv_events("model", 0, 8)
    assert [record.kind for record in first.events] == ["stored"]
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=keys)
    assert module.read_kv_events("model", first.next_cursor, 8).events == []


@pytest.mark.parametrize("world_size", [1, 2])
def test_cpu_events_keep_a_chunk_until_its_required_object_loses_all_copies(
    world_size: int,
) -> None:
    """Redundant L1 copies cannot mask a missing shard or cause early removal."""
    module, bus = _event_module(size=16, world_size=world_size)
    keys = [
        _event_key(1, rank=rank, world_size=world_size) for rank in range(world_size)
    ]
    _bind_kv(bus, keys[:1])
    for tag in ("_default", "second"):
        _publish_kv(bus, EventType.L1_WRITE_FINISHED, l1_tag=tag, keys=keys)
    first = module.read_kv_events("model", 0, 8)
    assert [record.kind for record in first.events] == ["stored"]
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=keys)
    assert module.read_kv_events("model", first.next_cursor, 8).events == []
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, l1_tag="second", keys=keys[:1])
    removed = module.read_kv_events("model", first.next_cursor, 8)
    assert [record.kind for record in removed.events] == ["removed"]
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, l1_tag="second", keys=keys[:1])
    assert module.read_kv_events("model", removed.next_cursor, 8).events == []
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=keys[:1])
    assert [
        record.kind
        for record in module.read_kv_events("model", removed.next_cursor, 8).events
    ] == ["stored"]


def test_cpu_events_refresh_a_committed_chunk_after_its_record_ages_out() -> None:
    """Replay loss must not permanently hide a chunk when it is stored again."""
    module, bus = _event_module(size=2)
    keys = [_event_key(i) for i in range(1, 4)]
    _bind_kv(bus, keys)
    for key in keys:
        _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
    lagging = module.read_kv_events("model", 0, 8)
    assert lagging.lost
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, l1_tag="second", keys=keys[:1])
    replay = module.read_kv_events("model", lagging.next_cursor, 8)
    assert [record.block_hashes for record in replay.events] == [[keys[0].chunk_hash]]


def test_cpu_events_follow_real_l1_commit_and_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real L1 reservations stay invisible until both rank objects are committed."""
    module, bus = _event_module(size=16, world_size=2)
    monkeypatch.setattr("lmcache.v1.distributed.l1_manager.get_event_bus", lambda: bus)
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            align_bytes=64,
            shm_name="",
        )
    )
    managers = [L1Manager(replace(config, tag=tag)) for tag in ("first", "second")]
    keys = [_event_key(1, rank=rank, world_size=2) for rank in range(2)]
    layout = module.context.layout_desc_registry.find("model", 2)
    assert layout is not None
    _bind_kv(bus, keys[:1])
    try:
        for manager in managers:
            reserved = manager.reserve_write(keys, [False, False], layout)
            for _, memory in reserved.values():
                assert memory is not None
                tensor = memory.get_tensor(0)
                assert tensor is not None
                tensor.fill_(1)
        bus.stop()
        assert not module.read_kv_events("model", 0, 8).events
        managers[0].finish_write(keys[:1])
        bus.stop()
        assert not module.read_kv_events("model", 0, 8).events
        managers[0].finish_write(keys[1:])
        bus.stop()
        stored = module.read_kv_events("model", 0, 8)
        assert [r.kind for r in stored.events] == ["stored"]
        managers[1].finish_write(keys)
        managers[0].delete(keys)
        bus.stop()
        assert not module.read_kv_events("model", stored.next_cursor, 8).events
        managers[1].delete(keys[:1])
        bus.stop()
        assert [
            r.kind for r in module.read_kv_events("model", stored.next_cursor, 8).events
        ] == ["removed"]
    finally:
        for manager in managers:
            manager.close()
        bus.stop()
        module.close()


def test_event_log_pages_filters_models_and_reports_overflow() -> None:
    module, bus = _event_module(size=3)
    for index, model in enumerate(("model", "other", "model")):
        key = _event_key(index, model)
        _bind_kv(bus, [key])
        _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
    first = module.read_kv_events("model", 0, 1)
    assert [r.seq for r in first.events] == [1] and first.next_cursor == 1
    second = module.read_kv_events("model", first.next_cursor, 1)
    assert [r.seq for r in second.events] == [3] and second.next_cursor == 3
    assert not second.lost
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[_event_key(0)])
    assert module.read_kv_events("model", 0, 8).lost
    assert not module.read_kv_events("model", 1, 8).lost
    stale = module.read_kv_events("model", 99, 8)
    assert stale.lost and stale.next_cursor == 4 and stale.events == []


def test_bus_loss_is_visible_without_later_bus_traffic() -> None:
    module, bus = _event_module(bus=EventBus(EventBusConfig(max_queue_size=1)))
    key = _event_key(1)
    _bind_kv(bus, [key])
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
    event = Event(
        EventType.L1_KEYS_EVICTED, metadata={"l1_tag": "_default", "keys": [key]}
    )
    bus.publish(event)
    bus.publish(event)
    bus.stop()
    result = module.read_kv_events("model", 0, 8)
    assert result.lost and result.events == [] and result.next_cursor == 2
    assert not module.read_kv_events("model", result.next_cursor, 8).lost
    assert module.report_status()["kv_events"]["lost_markers"] == 1
    bus.stop()


def test_bus_loss_discards_queued_commits_and_requires_all_ranks_again() -> None:
    """A queued pre-loss completion cannot combine with a fresh shard."""
    module, bus = _event_module(
        bus=EventBus(EventBusConfig(max_queue_size=1)),
        world_size=2,
    )
    keys = [_event_key(1, rank=rank, world_size=2) for rank in range(2)]
    _bind_kv(bus, keys[:1])
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=keys)
    bus.publish(
        Event(
            EventType.L1_WRITE_FINISHED,
            metadata={"keys": keys[:1], "l1_tag": "_default"},
        )
    )
    bus.publish(
        Event(
            EventType.L1_KEYS_EVICTED, metadata={"keys": keys[:1], "l1_tag": "_default"}
        )
    )
    bus.stop()
    lost = module.read_kv_events("model", 0, 8)
    assert lost.lost and not lost.events
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=keys[1:])
    assert not module.read_kv_events("model", lost.next_cursor, 8).events
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=keys[:1])
    assert [
        r.kind for r in module.read_kv_events("model", lost.next_cursor, 8).events
    ] == ["stored"]


def test_bus_loss_during_notification_cannot_replay_an_evicted_completion() -> None:
    """A producer can refill the queue before the module's loss callback runs."""
    bus = EventBus(EventBusConfig(max_queue_size=1))
    key = _event_key(1)
    write = Event(
        EventType.L1_WRITE_FINISHED, metadata={"keys": [key], "l1_tag": "_default"}
    )
    eviction = Event(
        EventType.L1_KEYS_EVICTED, metadata={"keys": [key], "l1_tag": "_default"}
    )
    refilled = False

    def refill_during_loss() -> None:
        nonlocal refilled
        if not refilled:
            refilled = True
            bus.publish(write)
            bus.publish(eviction)

    bus.subscribe_drops(refill_during_loss)
    module, _ = _event_module(bus=bus)
    _bind_kv(bus, [key])
    bus.publish(write)
    bus.publish(eviction)
    bus.stop()
    bus.stop()
    lost = module.read_kv_events("model", 0, 8)
    assert lost.lost and lost.events == []
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
    assert [
        r.kind for r in module.read_kv_events("model", lost.next_cursor, 8).events
    ] == ["stored"]


@pytest.mark.parametrize("other", ["model", "salt", "world"])
def test_cpu_events_do_not_combine_objects_from_other_namespaces(other: str) -> None:
    """Equal chunk hashes in another namespace cannot complete the missing rank."""
    module, bus = _event_module(world_size=2)
    first, second = [_event_key(1, rank=rank, world_size=2) for rank in range(2)]
    foreign = {
        "model": replace(second, model_name="other"),
        "salt": replace(second, cache_salt="other"),
        "world": _event_key(1, rank=1, world_size=4),
    }[other]
    _bind_kv(bus, [first])
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[first, foreign])
    assert not module.read_kv_events("model", 0, 8).events
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[second])
    assert [r.kind for r in module.read_kv_events("model", 0, 8).events] == ["stored"]


@pytest.mark.parametrize("groups", [None, {}])
def test_cpu_events_require_registered_nonempty_geometry(
    groups: dict[int, MemoryLayoutDesc] | None,
) -> None:
    """A missing layout must not turn an empty required-object set into readiness."""
    module, bus = _event_module()
    registry = module.context.layout_desc_registry
    registry.unregister("model", 1)
    if groups is not None:
        registry.register(
            "model", 1, MemoryLayoutDesc([], []), group_layout_descs=groups
        )
    key = _event_key(1)
    _bind_kv(bus, [key])
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
    assert not module.read_kv_events("model", 0, 8).events


@pytest.mark.parametrize(
    "bus_enabled, size",
    [
        (False, 3),
        (True, 0),
        (True, 3),
    ],
)
def test_event_channel_enabled_and_disabled_state(
    bus_enabled: bool,
    size: int,
) -> None:
    module, _ = _event_module(
        size,
        EventBus(EventBusConfig(enabled=bus_enabled)),
    )
    enabled = bus_enabled and size > 0
    result = module.read_kv_events("model", 0, 8)
    assert result.enabled == enabled and result.events == []
    assert module.enabled == enabled
    assert (
        _event_module()[0].read_kv_events("model", 0, 8).incarnation
        != result.incarnation
    )


def test_event_log_validates_arguments() -> None:
    with pytest.raises(ValueError, match="kv_event_log_size"):
        _event_module(size=-1)
    module, _ = _event_module()
    for cursor, size in ((-1, 1), (0, 0)):
        with pytest.raises(ValueError, match="cursor|max_events"):
            module.read_kv_events("model", cursor, size)


def test_unknown_and_expired_token_bindings_skip_stores(monkeypatch) -> None:
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.kv_events._TOKEN_BINDING_CACHE_SIZE",
        4,
    )
    module, bus = _event_module()
    keys = [_event_key(i) for i in range(5)]
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[keys[0]])
    _bind_kv(bus, keys)
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[keys[0], keys[-1]])
    result = module.read_kv_events("model", 0, 8)
    assert [r.block_hashes for r in result.events] == [[keys[-1].chunk_hash]]
    assert module.report_status()["kv_events"]["unbound_stores"] == 2


@pytest.mark.parametrize(
    ("transport", "grpc_server_mode"),
    [
        ("zmq", "sync"),
        ("grpc", "sync"),
        ("grpc", "aio"),
        pytest.param("grpc", "native", marks=requires_pygrpc_native),
    ],
)
def test_event_subscription_pushes_without_polling_and_keeps_requests_live(
    transport: Literal["zmq", "grpc"],
    grpc_server_mode: Literal["sync", "aio", "native"],
) -> None:
    """Real sockets: one subscription pushes store/remove and wakes on shutdown."""
    module, bus = _event_module()
    config = MPServerConfig(
        transport=transport,
        host="127.0.0.1",
        port=0,
        max_cpu_workers=1,
        grpc_server_workers=1,
        grpc_server_mode=grpc_server_mode,
    )
    management = ManagementModule(
        module.context,
        liveness_targets=[module],
    )
    server: Any = create_request_server([management, module], config)
    if transport == "grpc":
        url = f"grpc://127.0.0.1:{server.bound_port}"
    else:
        url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    server.start()
    client = RequestClientFactory.create(url, context=zmq.Context.instance())
    streams = []
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        # More idle streams than unary workers must not starve ping/lookup.
        for instance_id in range(3):
            stream = client.subscribe_kv_events(instance_id, "model", 0, 8)
            streams.append(stream)
            assert pool.submit(next, stream).result(timeout=5).events == []
        waiting_readers = [pool.submit(next, stream) for stream in streams]
        assert client.ping(0).result(timeout=5)
        assert all(not future.done() for future in waiting_readers)

        key = _event_key(1)
        _bind_kv(bus, [key])
        _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[key])
        for future in waiting_readers:
            batch = future.result(timeout=5)
            assert [(r.kind, r.medium, r.token_ids) for r in batch.events] == [
                ("stored", "CPU", [0])
            ]
        waiting = pool.submit(next, streams[0])
        _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[key])
        assert waiting.result(timeout=5).events[0].kind == "removed"
        waiting = pool.submit(next, streams[0])
        old_incarnation = batch.incarnation
        server.close()
        with pytest.raises((StopIteration, ConnectionError)):
            waiting.result(timeout=5)
        module.close()
        # Reuse the same client/channel after a server restart.
        module, bus = _event_module()
        config.port = int(url.rsplit(":", 1)[1])
        management = ManagementModule(module.context, liveness_targets=[module])
        server = create_request_server([management, module], config)
        server.start()
        resumed = client.subscribe_kv_events(0, "model", 2, 8)
        streams.append(resumed)
        batch = pool.submit(next, resumed).result(timeout=5)
        assert batch.incarnation != old_incarnation and batch.lost
        waiting = pool.submit(next, resumed)
        resumed.close()
        with pytest.raises((StopIteration, ConnectionError)):
            waiting.result(timeout=5)
    finally:
        for stream in streams:
            stream.close()
        client.close()
        server.close()
        module.close()
        pool.shutdown(wait=True)


def test_subscription_wakes_for_final_dropped_event_and_close() -> None:
    module, bus = _event_module(bus=EventBus(EventBusConfig(max_queue_size=0)))
    stream = module.subscribe_kv_events(1, "model", 0, 8)
    assert next(stream).events == []
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, stream)
        try:
            _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[_event_key(1)])
            assert pending.result(timeout=5).lost
        finally:
            stream.close()


@pytest.mark.parametrize("log_size", [0, 3])
def test_subscription_liveness_uses_management_ping_without_reaping_workers(
    monkeypatch: pytest.MonkeyPatch,
    log_size: int,
) -> None:
    now = 0.0
    monkeypatch.setattr(kv_events_mod.time, "monotonic", lambda: now)
    module, _ = _event_module(size=log_size)
    management = ManagementModule(module.context, liveness_targets=[module])
    streams = [module.subscribe_kv_events(i, "model", 0, 8) for i in (1, 2)]
    try:
        now = 10.0
        management.ping(1)
        now = 15.0
        assert module.reap_stale_instances(10.0, 120.0) == []
        assert module.report_status()["kv_events"]["subscribers"] == 1
        with pytest.raises(StopIteration):
            next(streams[1])
        # A subscription is not another worker registration in status counts.
        assert management.report_status()["worker_liveness"]["tracked_instances"] == 0
        module.drop_instance_state(1)
        with pytest.raises(StopIteration):
            next(streams[0])
    finally:
        management.close()
        module.close()


def test_management_reaper_closes_subscription_for_reaped_worker() -> None:
    module, _ = _event_module()
    target = _FakeTarget()
    management = ManagementModule(
        module.context,
        liveness_targets=[target, module],
        worker_reap_timeout_seconds=0.4,
        worker_registration_grace_seconds=0.8,
    )
    stream = module.subscribe_kv_events(7, "model", 0, 8)
    assert next(stream).events == []
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(next, stream)
        try:
            target.to_reap = [7]
            with pytest.raises(StopIteration):
                waiting.result(timeout=5)
            assert 7 in target.dropped
        finally:
            management.close()
            module.close()
