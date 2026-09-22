# SPDX-License-Identifier: Apache-2.0
"""Unit tests for server management: liveness and CPU KV event polling.

Cover the public liveness interfaces of the transfer modules, the
management reaper wiring, the blend reap listener, and config validation.
No GPU or live server is required; module-level construction dependencies
are stubbed.
"""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import KV_EVENT_CAPABILITY
from lmcache.v1.multiprocess.modules import engine_driven_transfer as non_gpu_mod
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as gpu_mod
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.periodic_thread import PeriodicThreadRegistry


def _event_module(
    size: int = 3,
    bus: EventBus | None = None,
    advertise: bool = True,
) -> tuple[ManagementModule, EventBus]:
    bus = bus or EventBus(EventBusConfig())
    ctx = MagicMock(event_bus=bus)
    return ManagementModule(
        ctx,
        kv_event_log_size=size,
        advertise_kv_events=advertise,
        experimental_transfer=("transfer_query",),
    ), bus


def _publish_kv(bus: EventBus, event_type: EventType, **metadata: Any) -> None:
    bus.publish(Event(event_type, metadata=metadata))
    bus.stop()  # Public shutdown flushes queued callbacks synchronously.


def _event_key(index: int, model: str = "model", rank: int = 0) -> ObjectKey:
    return ObjectKey(bytes([index]) * 32, model, rank)


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
    """A EngineDrivenTransferModule with only the liveness state initialized."""
    module = EngineDrivenTransferModule.__new__(EngineDrivenTransferModule)
    module._ctx = MagicMock(name="ctx")
    module._engine_driven_contexts = {}
    module._strategies = {}
    module._lock = threading.Lock()
    module._pending_shm_writes = {}
    module._pending_shm_reads = {}
    module._pending_shm_lock = threading.Lock()
    return module


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
    module, bus = _event_module()
    keys = [_event_key(1), _event_key(2)]
    _bind_kv(bus, keys)
    _publish_kv(
        bus,
        EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
        keys=[*keys, _event_key(1, rank=1)],
    )
    result = module.poll_kv_events("model", 0, 8)
    assert [
        (r.kind, r.block_hashes, r.token_ids, r.parent_block_hash)
        for r in result.events
    ] == [
        ("stored", [keys[0].chunk_hash], [0], None),
        ("stored", [keys[1].chunk_hash], [1], keys[0].chunk_hash),
    ]
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[keys[0], _event_key(1, rank=1)])
    removed = module.poll_kv_events("model", result.next_cursor, 8)
    assert [(r.kind, r.medium, r.block_hashes) for r in removed.events] == [
        ("removed", "CPU", [keys[0].chunk_hash]),
    ]
    assert removed.incarnation == result.incarnation and not removed.lost
    assert module.poll_kv_events("model", removed.next_cursor, 8).events == []


def test_event_polling_pages_filters_models_and_reports_overflow() -> None:
    module, bus = _event_module(size=3)
    for index, model in enumerate(("model", "other", "model")):
        _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[_event_key(index, model)])
    first = module.poll_kv_events("model", 0, 1)
    assert [r.seq for r in first.events] == [1] and first.next_cursor == 1
    second = module.poll_kv_events("model", first.next_cursor, 1)
    assert [r.seq for r in second.events] == [3] and second.next_cursor == 3
    assert not second.lost
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[_event_key(3)])
    assert module.poll_kv_events("model", 0, 8).lost
    assert not module.poll_kv_events("model", 1, 8).lost
    stale = module.poll_kv_events("model", 99, 8)
    assert stale.lost and stale.next_cursor == 4 and stale.events == []


def test_bus_loss_is_visible_without_later_bus_traffic() -> None:
    module, bus = _event_module(bus=EventBus(EventBusConfig(max_queue_size=1)))
    _publish_kv(bus, EventType.L1_KEYS_EVICTED, keys=[_event_key(1)])
    event = Event(EventType.L1_KEYS_EVICTED, metadata={"keys": [_event_key(2)]})
    bus.publish(event)
    bus.publish(event)
    result = module.poll_kv_events("model", 0, 8)
    assert result.lost and result.events == [] and result.next_cursor == 2
    assert not module.poll_kv_events("model", result.next_cursor, 8).lost
    assert module.report_status()["kv_events"]["lost_markers"] == 1
    bus.stop()


@pytest.mark.parametrize(
    "bus_enabled, size, advertise",
    [
        (False, 3, True),
        (True, 0, True),
        (True, 3, False),
        (True, 3, True),
    ],
)
def test_event_channel_capability_and_disabled_state(
    bus_enabled: bool,
    size: int,
    advertise: bool,
) -> None:
    module, _ = _event_module(
        size,
        EventBus(EventBusConfig(enabled=bus_enabled)),
        advertise,
    )
    enabled = bus_enabled and size > 0
    result = module.poll_kv_events("model", 0, 8)
    assert result.enabled == enabled and result.events == []
    assert module.get_experimental() == (
        ["transfer_query", KV_EVENT_CAPABILITY]
        if enabled and advertise
        else ["transfer_query"]
    )
    assert (
        _event_module()[0].poll_kv_events("model", 0, 8).incarnation
        != result.incarnation
    )


def test_event_polling_validates_arguments() -> None:
    with pytest.raises(ValueError, match="kv_event_log_size"):
        _event_module(size=-1)
    module, _ = _event_module()
    for cursor, size in ((-1, 1), (0, 0)):
        with pytest.raises(ValueError, match="cursor|max_events"):
            module.poll_kv_events("model", cursor, size)


def test_unknown_and_expired_token_bindings_skip_stores(monkeypatch) -> None:
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.modules.management._TOKEN_BINDING_CACHE_SIZE",
        4,
    )
    module, bus = _event_module()
    keys = [_event_key(i) for i in range(5)]
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[keys[0]])
    _bind_kv(bus, keys)
    _publish_kv(bus, EventType.L1_WRITE_FINISHED, keys=[keys[0], keys[-1]])
    result = module.poll_kv_events("model", 0, 8)
    assert [r.block_hashes for r in result.events] == [[keys[-1].chunk_hash]]
    assert module.report_status()["kv_events"]["unbound_stores"] == 2
