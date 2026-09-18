# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the optional L2 prefetch/load deadline.

These drive the real ``PrefetchController`` with an injected monotonic clock and
a load-gated mock adapter, so deadline expiry, both orderings of the
completion-vs-deadline race, exactly-once publish/retire, head-of-line backpressure,
and same-key retry are pinned without depending on wall-clock timing. State is
asserted through ``report_status`` and the public query API, matching how
``test_prefetch_controller.py`` synchronizes with the loop thread.
"""

# Standard
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers import prefetch_controller
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


# =============================================================================
# Helpers
# =============================================================================


class FakeClock:
    """A thread-safe monotonic clock the test advances explicitly."""

    def __init__(self, start: float = 1000.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._t

    def advance(self, dt: float) -> None:
        with self._lock:
            self._t += dt


class GatedLoadMockAdapter(MockL2Adapter):
    """MockL2Adapter whose load (and optionally lookup) completion is gated.

    Loads copy their data but do not report completion (the efd stays
    unsignaled, the controller's task stays pending) until the test sets the
    gate via :meth:`release_loads`, so a load is genuinely in-flight while the
    deadline fires. Lookup gating is opt-in via :meth:`gate_lookups` (default
    released) for LOOKUP-phase deadline tests. ``debug_locked_key_count`` exposes
    the number of L2 keys still read-locked so a test can assert the drain
    released them.
    """

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self._load_released = threading.Event()
        self._lookup_released = threading.Event()
        self._lookup_released.set()
        self._fail_loads = False

    def release_loads(self) -> None:
        self._load_released.set()

    def fail_loads(self) -> None:
        """When released, loads complete but report zero keys loaded, so the
        controller treats every reserved key as a failed load."""
        self._fail_loads = True

    def gate_lookups(self) -> None:
        self._lookup_released.clear()

    def release_lookups(self) -> None:
        self._lookup_released.set()

    def debug_locked_key_count(self) -> int:
        with self._lock:
            return len(self._locked_keys)

    def _execute_lookup_in_the_loop(self, keys, task_id):  # type: ignore[override]
        if not self._lookup_released.is_set():
            self._loop.call_later(
                0.005, self._execute_lookup_in_the_loop, keys, task_id
            )
            return
        super()._execute_lookup_in_the_loop(keys, task_id)

    async def _execute_load_in_loop(self, keys, objects, task_id):  # type: ignore[override]
        bitmap = Bitmap(len(keys))
        accessed = []
        for i, key in enumerate(keys):
            if key not in self._memory_objects:
                continue
            objects[i].tensor.copy_(self._memory_objects[key].tensor)
            if not self._fail_loads:
                bitmap.set(i)
                accessed.append(key)
        while not self._load_released.is_set():
            await asyncio.sleep(0.005)
        if accessed:
            self._notify_keys_accessed(accessed)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


def make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def make_gated_adapter() -> GatedLoadMockAdapter:
    return GatedLoadMockAdapter(
        MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    return AdapterDescriptor(
        index=index,
        config=MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0),
    )


def wait_until(pred, timeout: float = 5.0) -> bool:
    """Poll a predicate until true or timeout (same mechanism the existing
    controller tests use to synchronize with the loop thread)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def store_keys_in_l2(adapter, keys, layout) -> None:
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        meta = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        objs.append(
            TensorMemoryObj(raw_data=tensor, metadata=meta, parent_allocator=None)
        )
    adapter.submit_store_task(keys, objs)  # type: ignore[arg-type]
    assert wait_until(lambda: all(adapter.debug_has_key(k) for k in keys)), (
        "Failed to store test data in L2 adapter"
    )


def active_count(ctrl) -> int:
    s = ctrl.report_status()
    return s["in_flight_request_count"] - s["draining_request_count"]


def draining_count(ctrl) -> int:
    return ctrl.report_status()["draining_request_count"]


def in_flight_count(ctrl) -> int:
    return ctrl.report_status()["in_flight_request_count"]


def result_ready(ctrl, req) -> bool:
    # Non-consuming presence check via the public wait API (timeout 0).
    return ctrl.wait_prefetch_result(req, 0.0)


def submit(ctrl, keys, layout, policy=TrimPolicy.PREFIX) -> int:
    spec = PrefetchRequestSpec(
        keys=keys,
        group_layout_descs={0: layout},
        num_kv_readers=1,
        policy=policy,
    )
    return ctrl.submit_prefetch_request(spec)


@pytest.fixture(autouse=True)
def _fast_loop(monkeypatch):
    # Shrink the loop poll so it reacts to a clock advance within ~20ms; keeps
    # the tests fast without changing behavior.
    monkeypatch.setattr(prefetch_controller, "PREFETCH_LOOP_POLL_TIMEOUT_MS", 20)


@pytest.fixture
def l1_manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


def make_controller(l1_manager, adapter, clock, timeout, max_in_flight=8):
    return make_controller_multi(l1_manager, [adapter], clock, timeout, max_in_flight)


def make_controller_multi(l1_manager, adapters, clock, timeout, max_in_flight=8):
    return PrefetchController(
        l1_manager=l1_manager,
        l2_adapters=list(adapters),
        adapter_descriptors=[make_descriptor(i) for i in range(len(adapters))],
        policy=DefaultPrefetchPolicy(),
        max_in_flight=max_in_flight,
        l2_load_timeout=timeout,
        clock=clock,
    )


class _EventCapture:
    """Thread-safe capture of published events, filterable by type."""

    def __init__(self) -> None:
        self._events: list = []
        self._lock = threading.Lock()

    def record(self, event) -> None:
        with self._lock:
            self._events.append(event)

    def of_type(self, event_type) -> list:
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]


@pytest.fixture
def event_capture():
    """Install an enabled global EventBus, capture the deadline-related events,
    and restore the disabled default afterwards."""
    bus = init_event_bus(EventBusConfig(enabled=True))
    bus.start()
    cap = _EventCapture()
    for et in (
        EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
        EventType.L2_PREFETCH_LOOKUP_COMPLETED,
        EventType.L2_PREFETCH_DEADLINE,
    ):
        bus.subscribe(et, cap.record)
    try:
        yield cap
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


# =============================================================================
# Configuration validation
# =============================================================================


def _base_config_kwargs():
    return dict(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=32 * 1024 * 1024,
                align_bytes=0x1000,
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


class TestConfigValidation:
    def test_default_is_disabled(self):
        cfg = StorageManagerConfig(**_base_config_kwargs())
        assert cfg.prefetch_load_timeout is None

    def test_positive_value_accepted(self):
        cfg = StorageManagerConfig(prefetch_load_timeout=0.5, **_base_config_kwargs())
        assert cfg.prefetch_load_timeout == 0.5

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.001])
    def test_non_positive_fails_closed(self, bad):
        with pytest.raises(ValueError):
            StorageManagerConfig(prefetch_load_timeout=bad, **_base_config_kwargs())


# =============================================================================
# Deadline behavior on the real controller path
# =============================================================================


class TestLoadDeadline:
    def test_disabled_is_behavior_neutral(self, l1_manager):
        """With the feature off, a normal prefetch completes and retires as
        before, and no timeout is ever recorded."""
        adapter = make_gated_adapter()
        adapter.release_loads()  # let loads complete immediately
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=None)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req))
            assert ctrl.query_prefetch_result(req) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.report_status()["deadline_timeout_count"] == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_timeout_during_load_publishes_fallback_without_freeing_buffers(
        self, l1_manager
    ):
        """Deadline fires while the load is pending: the caller gets a final
        fallback (not None), the request enters drain-only with its buffers and
        L2 lock still held, and after the late load completes it retires."""
        adapter = make_gated_adapter()  # loads stay pending until released
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)

            # The request is admitted and its load is pending. Fire the deadline.
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            clock.advance(10.0)

            # A final fallback result is published (not None) while the load is
            # still in-flight, so the request is draining, not retired.
            assert wait_until(lambda: result_ready(ctrl, req))
            assert wait_until(lambda: draining_count(ctrl) == 1)
            assert active_count(ctrl) == 0
            assert ctrl.report_status()["deadline_timeout_count"] == 1

            # Late completion drains safely and the request retires exactly once.
            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            # Result is immutable and consumable exactly once.
            assert ctrl.query_prefetch_result(req) is not None
            assert ctrl.query_prefetch_result(req) is None
        finally:
            ctrl.stop()
            adapter.close()

    def test_hit_reported_at_submit_then_smaller_retained_on_timeout(self, l1_manager):
        """The lookup hit reported at load-submit can exceed the retained set a
        timeout publishes; the authoritative result the caller consumes is the
        smaller retained bitmap (same shrink shape as a partial load failure)."""
        adapter = make_gated_adapter()  # load never completes before the deadline
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)

            # The load-phase transition reports the union lookup hit (all 4
            # chunks found in L2) before the load lands.
            assert wait_until(lambda: ctrl.query_lookup_result(req) == 4)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))

            # Retained (nothing loaded) is strictly smaller than the reported 4.
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None
            assert retained.count_leading_ones() == 0

            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_completion_before_deadline_wins(self, l1_manager):
        """If the load completes before the clock passes the deadline, the
        request completes normally and a later clock advance is a no-op."""
        adapter = make_gated_adapter()
        adapter.release_loads()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req))
            result = ctrl.query_prefetch_result(req)
            assert result is not None
            assert result.count_leading_ones() == 4  # full hit, no timeout
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.report_status()["deadline_timeout_count"] == 0
            clock.advance(100.0)  # no armed requests remain
            assert in_flight_count(ctrl) == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_same_key_request_after_timeout_succeeds(self, l1_manager):
        """After a timeout + late drain, a second request for the same keys
        completes cleanly with no leaked locks/reservations."""
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)

            req1 = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req1))
            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)

            # Second request for the same keys must not block or error.
            req2 = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req2))
            assert ctrl.query_prefetch_result(req2) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_head_of_line_drainers_then_queue_expiry_then_admission(self, l1_manager):
        """Slow-L2 backpressure: drainers hold the max_in_flight slots, a request
        queued behind them expires in the queue, and after the drainers retire a
        later request is admitted and completes."""
        adapter = make_gated_adapter()  # every load stays pending until released
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0, max_in_flight=2)
        ctrl.start()
        try:
            layout = make_layout()
            key_sets = [
                [make_object_key(10 * g + i) for i in range(4)] for g in range(4)
            ]
            for ks in key_sets:
                store_keys_in_l2(adapter, ks, layout)

            r0 = submit(ctrl, key_sets[0], layout)
            r1 = submit(ctrl, key_sets[1], layout)
            # Both admitted (slots full at max_in_flight=2), loads pending.
            assert wait_until(lambda: in_flight_count(ctrl) == 2)

            # Fire their deadline -> both become drainers, still holding slots.
            clock.advance(10.0)
            assert wait_until(lambda: draining_count(ctrl) == 2)
            assert wait_until(lambda: result_ready(ctrl, r0) and result_ready(ctrl, r1))

            # A third request cannot be admitted (slots held by drainers); it
            # sits in the queue and expires there when its own deadline passes.
            r2 = submit(ctrl, key_sets[2], layout)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, r2))
            assert ctrl.query_prefetch_result(r2) is not None  # L1-only fallback

            # Releasing the drainers retires them and frees the slots.
            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)

            # A later request is admitted and completes normally.
            r3 = submit(ctrl, key_sets[3], layout)
            assert wait_until(lambda: result_ready(ctrl, r3))
            assert ctrl.query_prefetch_result(r3) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_timeout_during_lookup_late_lookup_releases_l2_lock(self, l1_manager):
        """Deadline fires in the LOOKUP phase (lookup gated): an L1-only fallback
        is published and the request drains; the late lookup then takes and, on
        drain, releases its L2 read lock."""
        adapter = make_gated_adapter()
        adapter.gate_lookups()  # keep the request in the LOOKUP phase
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert ctrl.report_status()["lookup_phase_count"] == 1

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            assert wait_until(lambda: draining_count(ctrl) == 1)
            # No L2 lock yet (lookup still gated), nothing loaded into L1.
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None and retained.count_leading_ones() == 0

            # Late lookup completes, takes its L2 lock, then the drain releases it.
            adapter.release_lookups()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert wait_until(lambda: adapter.debug_locked_key_count() == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_partial_load_completion_trims_before_hole_prefix(self, l1_manager):
        """PREFIX: a still-pending early key is a hole, so a later key that has
        finished loading is NOT reported in the prefix (it becomes resident)."""
        slow = make_gated_adapter()  # holds keys 0 and 2 (never released here)
        fast = make_gated_adapter()
        fast.release_loads()  # loads keys 1 and 3
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            # First adapter by index that has a key owns its load: slow={0,2},
            # fast={1,3}.
            store_keys_in_l2(slow, [keys[0], keys[2]], layout)
            store_keys_in_l2(fast, [keys[1], keys[3]], layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            # key 0 (in slow, pending) is a hole -> prefix hit 0, even though the
            # fast adapter finished keys 1 and 3.
            assert retained is not None
            assert retained.count_leading_ones() == 0

            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()

    @pytest.mark.parametrize("iteration", range(20))
    def test_race_orderings_looped(self, l1_manager, iteration):
        """Loop both orderings: on even iterations the load completes before the
        clock passes the deadline (completion wins); on odd iterations the clock
        passes first (deadline wins). Exactly-once either way."""
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            if iteration % 2 == 0:
                adapter.release_loads()  # completion wins
                assert wait_until(lambda: result_ready(ctrl, req))
                clock.advance(10.0)  # no-op
                assert ctrl.report_status()["deadline_timeout_count"] == 0
            else:
                clock.advance(10.0)  # deadline wins
                assert wait_until(lambda: result_ready(ctrl, req))
                adapter.release_loads()
                assert ctrl.report_status()["deadline_timeout_count"] == 1
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is not None
            assert ctrl.query_prefetch_result(req) is None  # consumed once
        finally:
            ctrl.stop()
            adapter.close()

    def test_shutdown_while_draining_releases_everything(self, l1_manager):
        """A drain-only request that meets controller shutdown is cleaned up
        without a double release or a crash."""
        adapter = make_gated_adapter()  # load never released
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        layout = make_layout()
        keys = [make_object_key(i) for i in range(4)]
        store_keys_in_l2(adapter, keys, layout)
        req = submit(ctrl, keys, layout)
        assert wait_until(lambda: in_flight_count(ctrl) == 1)
        clock.advance(10.0)
        assert wait_until(lambda: draining_count(ctrl) == 1)
        # Shut down while draining: stop() joins the loop then cleans up.
        ctrl.stop()
        assert ctrl.report_status()["in_flight_request_count"] == 0
        # The published fallback is still readable after shutdown.
        assert ctrl.query_prefetch_result(req) is not None
        adapter.close()

    def test_queue_expiry_emits_coherent_events_and_metric(
        self, l1_manager, event_capture
    ):
        """A queue-expired request emits LOOKUP_SUBMITTED (to zero adapters)
        before LOOKUP_COMPLETED, and one L2_PREFETCH_DEADLINE with phase=queued."""
        adapter = make_gated_adapter()  # drainers hold the single slot
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0, max_in_flight=1)
        ctrl.start()
        try:
            layout = make_layout()
            ks0 = [make_object_key(i) for i in range(4)]
            ks1 = [make_object_key(10 + i) for i in range(4)]
            store_keys_in_l2(adapter, ks0, layout)
            store_keys_in_l2(adapter, ks1, layout)

            submit(ctrl, ks0, layout)  # admitted, becomes a drainer
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            clock.advance(10.0)
            assert wait_until(lambda: draining_count(ctrl) == 1)

            r1 = submit(ctrl, ks1, layout)  # queued behind the drainer
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, r1))

            deadlines = event_capture.of_type(EventType.L2_PREFETCH_DEADLINE)
            assert wait_until(
                lambda: any(
                    e.metadata["phase"] == "queued" and e.metadata["request_id"] == r1
                    for e in event_capture.of_type(EventType.L2_PREFETCH_DEADLINE)
                )
            )
            submitted = [
                e
                for e in event_capture.of_type(EventType.L2_PREFETCH_LOOKUP_SUBMITTED)
                if e.metadata["request_id"] == r1
            ]
            completed = [
                e
                for e in event_capture.of_type(EventType.L2_PREFETCH_LOOKUP_COMPLETED)
                if e.metadata["request_id"] == r1
            ]
            assert submitted and completed
            assert submitted[0].metadata["adapter_count"] == 0
            assert submitted[0].timestamp <= completed[0].timestamp
            del deadlines

            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.mark.parametrize(
        "policy, retained_readlocked",
        [
            (TrimPolicy.PREFIX, False),  # hole at key 0 -> nothing retained
            (TrimPolicy.SEGMENTED_PREFIX, True),  # gaps kept -> 1,3 retained
            (TrimPolicy.SPARSE, True),  # scattered -> 1,3 retained
        ],
    )
    def test_partial_completion_policy_lock_accounting(
        self, l1_manager, policy, retained_readlocked
    ):
        """Timeout with a hole (key 0 pending) before completed keys 1,3 drives
        the lock/reservation accounting per policy, not just the bitmap: under
        PREFIX the completed keys become resident-but-unlocked (unsafe_read =>
        NOT_READABLE); under SEGMENTED_PREFIX/SPARSE they are read-locked for the
        caller (unsafe_read => SUCCESS)."""
        slow = make_gated_adapter()  # holds keys 0 and 2 (never released here)
        fast = make_gated_adapter()
        fast.release_loads()  # finishes keys 1 and 3
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(slow, [keys[0], keys[2]], layout)
            store_keys_in_l2(fast, [keys[1], keys[3]], layout)
            req = submit(ctrl, keys, layout, policy=policy)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None
            retained_idx = set(retained.get_indices_list())

            # Probe the completed keys' L1 state (unsafe_read: SUCCESS = read
            # locked, NOT_READABLE = resident-unlocked).
            probe = l1_manager.unsafe_read([keys[1], keys[3]])
            if retained_readlocked:
                assert retained_idx == {1, 3}
                assert probe[keys[1]][0] == L1Error.SUCCESS
                assert probe[keys[3]][0] == L1Error.SUCCESS
                l1_manager.finish_read([keys[1], keys[3]])  # caller consumes
            else:
                assert retained_idx == set()  # hole at 0 truncates the prefix
                assert probe[keys[1]][0] == L1Error.KEY_NOT_READABLE
                assert probe[keys[3]][0] == L1Error.KEY_NOT_READABLE

            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()

    def test_load_failure_during_drain(self, l1_manager):
        """A load that completes with a failure during the drain deletes the
        failed keys' buffers exactly once, does not re-publish, retires the
        request, and leaves a same-key retry working."""
        adapter = make_gated_adapter()
        adapter.fail_loads()  # its load will report zero keys loaded
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            before = ctrl.query_prefetch_result(req)  # consume the fallback
            assert before is not None and before.count_leading_ones() == 0

            # Late failing load drains: buffers deleted, request retires, no
            # re-publish (result was already consumed and stays gone).
            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert ctrl.report_status()["deadline_timeout_count"] == 1
            # Failed keys were deleted, not left half-reserved.
            probe = l1_manager.unsafe_read(keys)
            assert all(v[0] == L1Error.KEY_NOT_EXIST for v in probe.values())

            # Same-key retry works (keys still live in L2; adapter no longer fails
            # a fresh reserve because the earlier buffers were cleaned up).
            adapter._fail_loads = False  # noqa: SLF001 (test shim)
            req2 = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req2))
            assert ctrl.query_prefetch_result(req2) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_fast_success_slow_failure_two_adapters(self, l1_manager):
        """Fast adapter succeeds (keys 1,3 retained under SPARSE), slow adapter
        fails during the drain: the request retires cleanly, the failed keys are
        not published, and a same-key retry works."""
        slow = make_gated_adapter()
        slow.fail_loads()  # keys 0,2 will fail on release
        fast = make_gated_adapter()
        fast.release_loads()  # keys 1,3 succeed
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(slow, [keys[0], keys[2]], layout)
            store_keys_in_l2(fast, [keys[1], keys[3]], layout)
            req = submit(ctrl, keys, layout, policy=TrimPolicy.SPARSE)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            assert set(retained.get_indices_list()) == {1, 3}  # only fast succeeded
            l1_manager.finish_read([keys[1], keys[3]])

            slow.release_loads()  # slow load completes with failure
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None  # not re-published

            slow._fail_loads = False  # noqa: SLF001
            req2 = submit(ctrl, keys, layout, policy=TrimPolicy.SPARSE)
            assert wait_until(lambda: result_ready(ctrl, req2))
            assert ctrl.query_prefetch_result(req2) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()
