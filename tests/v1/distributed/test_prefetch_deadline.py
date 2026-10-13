# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch deadline tests using the native L1 manager and gated I/O."""

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
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
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
    RetainPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from tests.v1.distributed.test_prefetch_controller import make_l1_config
from tests.v1.distributed.utils import should_use_lazy_alloc, single_row_spec

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )


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
    """Gate lookup and buffer writes; signal load entry and result consumption."""

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self._load_released = threading.Event()
        self._lookup_released = threading.Event()
        self._lookup_released.set()
        self._fail_loads = False
        self.load_entered = threading.Event()
        self.load_result_consumed = threading.Event()

    def release_loads(self) -> None:
        self._load_released.set()

    def fail_loads(self, enabled: bool = True) -> None:
        """Report all loaded keys as failed."""
        self._fail_loads = enabled

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

    def query_load_result(self, task_id):  # type: ignore[override]
        result = super().query_load_result(task_id)
        if result is not None:
            self.load_result_consumed.set()
        return result

    async def _execute_load_in_loop(self, keys, objects, task_id):  # type: ignore[override]
        self.load_entered.set()
        while not self._load_released.is_set():
            await asyncio.sleep(0.005)
        bitmap = Bitmap(len(keys))
        accessed = []
        for i, key in enumerate(keys):
            if key not in self._memory_objects:
                continue
            objects[i].tensor.copy_(self._memory_objects[key].tensor)
            if not self._fail_loads:
                bitmap.set(i)
                accessed.append(key)
        if accessed:
            self._notify_keys_accessed(accessed)
        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._signal_load_event()


def make_object_key(chunk_id: int, kv_rank: int = 0) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=kv_rank,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def make_gated_adapter() -> GatedLoadMockAdapter:
    return GatedLoadMockAdapter(
        MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
    )


def make_descriptor(index: int) -> L2AdapterDescriptor:
    return L2AdapterDescriptor(
        index=index,
        config=MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0),
    )


def wait_until(pred, timeout: float = 5.0) -> bool:
    """Wait for a controller-thread predicate."""
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
    return ctrl.wait_prefetch_result(req, 0.0)


def submit(ctrl, keys, layout, policy="prefix") -> int:
    spec = single_row_spec(keys, layout, num_kv_readers=1, fetching_policy=policy)
    return ctrl.submit_prefetch_request(spec)


@pytest.fixture(autouse=True)
def _fast_loop(monkeypatch):
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


def make_controller(l1_manager, adapter, clock, timeout, max_in_flight=8, policy=None):
    return make_controller_multi(
        l1_manager, [adapter], clock, timeout, max_in_flight, policy
    )


def make_controller_multi(
    l1_manager, adapters, clock, timeout, max_in_flight=8, policy=None
):
    return PrefetchController(
        l1_managers=[l1_manager],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=make_l1_config())],
        l2_adapters=list(adapters),
        adapter_descriptors=[make_descriptor(i) for i in range(len(adapters))],
        policy=DefaultPrefetchPolicy() if policy is None else policy,
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
    """Capture deadline events and restore the disabled bus after the test."""
    bus = init_event_bus(EventBusConfig(enabled=True))
    bus.start()
    cap = _EventCapture()
    for et in (
        EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
        EventType.L2_PREFETCH_DEADLINE,
    ):
        bus.subscribe(et, cap.record)
    try:
        yield cap
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


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
    @pytest.mark.parametrize("timeout", [None, 0.5, 1e18])
    def test_valid_timeout(self, timeout):
        cfg = StorageManagerConfig(
            prefetch_load_timeout=timeout, **_base_config_kwargs()
        )
        assert cfg.prefetch_load_timeout == timeout

    @pytest.mark.parametrize(
        "bad",
        [
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
        ],
    )
    def test_invalid_timeout(self, bad, l1_manager):
        with pytest.raises(ValueError, match="finite positive"):
            StorageManagerConfig(prefetch_load_timeout=bad, **_base_config_kwargs())

        adapter = make_gated_adapter()
        try:
            with pytest.raises(ValueError, match="finite positive"):
                make_controller(l1_manager, adapter, FakeClock(), timeout=bad)
        finally:
            adapter.close()

    def test_direct_construction_accepts_large_finite(self, l1_manager):
        """Large finite budgets must not overflow poll timeout conversion."""
        adapter = make_gated_adapter()
        ctrl = make_controller(l1_manager, adapter, FakeClock(), timeout=1e18)
        try:
            assert ctrl._next_poll_timeout_ms() == (  # noqa: SLF001
                prefetch_controller.PREFETCH_LOOP_POLL_TIMEOUT_MS
            )
            ctrl.start()
            layout = make_layout()
            keys = [make_object_key(700 + i) for i in range(2)]
            store_keys_in_l2(adapter, keys, layout)
            adapter.release_loads()
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, req))
            assert ctrl.report_status()["deadline_timeout_count"] == 0
        finally:
            ctrl.stop()
            adapter.close()


class TestLoadDeadline:
    def test_drain_error_preserves_published_read_locks(self, l1_manager, monkeypatch):
        """A late adapter error cannot release the timeout caller's cache hit."""
        slow = make_gated_adapter()
        fast = make_gated_adapter()
        fast.release_loads()
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        keys = [make_object_key(i) for i in range(2)]
        layout = make_layout()
        ctrl.start()
        try:
            store_keys_in_l2(fast, keys[:1], layout)
            store_keys_in_l2(slow, keys[1:], layout)
            req = submit(ctrl, keys, layout)
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            result = ctrl.query_prefetch_result(req)
            assert result is not None
            assert result.hit_cells[0].get_indices_list() == [0]

            original_query = slow.query_load_result

            def fail_after_load(task_id):
                result = original_query(task_id)
                if result is not None:
                    raise RuntimeError("late adapter result error")
                return result

            monkeypatch.setattr(slow, "query_load_result", fail_after_load)
            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert l1_manager.unsafe_read(keys[:1])[keys[0]][0] == L1Error.SUCCESS
            l1_manager.finish_read(keys[:1])
        finally:
            ctrl.stop()
            slow.close()
            fast.close()

    @pytest.mark.parametrize(
        "policy, expected",
        [("prefix", [{0}, {0}]), ("full", [{0}, {0}])],
    )
    def test_grouped_deadline_preserves_rank_rows(self, l1_manager, policy, expected):
        """A partially loaded rank must not borrow another rank's ready cells."""
        slow = make_gated_adapter()
        fast = make_gated_adapter()
        fast.release_loads()
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        rows = [
            [make_object_key(i, kv_rank=rank) for i in range(3)] for rank in range(2)
        ]
        layout = make_layout()
        ctrl.start()
        try:
            store_keys_in_l2(fast, rows[0] + rows[1][:1], layout)
            store_keys_in_l2(slow, rows[1][1:], layout)
            req = ctrl.submit_prefetch_request(
                PrefetchTaskSpec(
                    key_groups=[
                        GroupedObjectKeys(
                            keys=row, object_group_id=0, layout_desc=layout
                        )
                        for row in rows
                    ],
                    fetching_policy=policy,
                )
            )
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            result = ctrl.query_prefetch_result(req)
            assert result is not None
            assert [set(row.get_indices_list()) for row in result.hit_cells] == expected
            assert result.l1_hit_count == 0
            assert result.l2_hit_count == sum(len(row) for row in expected)
            if policy == "full":
                assert result.found_cells is not None
                assert [set(row.get_indices_list()) for row in result.found_cells] == [
                    {0, 1, 2},
                    {0, 1, 2},
                ]
            for keys, kept in zip(rows, expected, strict=True):
                l1_manager.finish_read([keys[i] for i in sorted(kept)])
            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
        finally:
            ctrl.stop()
            slow.close()
            fast.close()

    @pytest.mark.parametrize("fail_load", [False, True])
    def test_timeout_during_load_preserves_buffers_and_allows_retry(
        self, l1_manager, fail_load
    ):
        """Late I/O releases locks without changing the timeout result."""
        adapter = make_gated_adapter()
        if fail_load:
            adapter.fail_loads()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)

            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert adapter.load_entered.wait(10.0)
            clock.advance(10.0)

            assert wait_until(lambda: result_ready(ctrl, req))
            assert wait_until(lambda: draining_count(ctrl) == 1)
            assert active_count(ctrl) == 0
            assert ctrl.report_status()["deadline_timeout_count"] == 1
            fallback = ctrl.query_prefetch_result(req)
            assert fallback is not None
            assert fallback.hit_cells[0].get_indices_list() == []
            assert l1_manager.report_status()["write_locked_count"] == len(keys)
            assert adapter.debug_locked_key_count() == len(keys)

            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert l1_manager.report_status()["write_locked_count"] == 0
            assert wait_until(lambda: adapter.debug_locked_key_count() == 0)
            assert l1_manager.report_status()["read_locked_count"] == 0
            if fail_load:
                assert all(
                    state[0] == L1Error.KEY_NOT_EXIST
                    for state in l1_manager.unsafe_read(keys).values()
                )

            adapter.fail_loads(False)
            retry = submit(ctrl, keys, layout)
            assert wait_until(lambda: result_ready(ctrl, retry))
            result = ctrl.query_prefetch_result(retry)
            assert result is not None
            assert result.hit_cells[0].get_indices_list() == list(range(len(keys)))
            l1_manager.finish_read(keys)
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.mark.parametrize("timed_out", [False, True])
    def test_retained_prefetch_does_not_enqueue_l2_store(self, l1_manager, timed_out):
        """Retained loads stay in L1 without being queued for another L2 store."""
        listener = StoreListener()
        l1_manager.register_listener(listener)
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(
            l1_manager, adapter, clock, timeout=5.0, policy=RetainPrefetchPolicy()
        )
        keys = [make_object_key(i) for i in range(2)]
        layout = make_layout()
        ctrl.start()
        try:
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert adapter.load_entered.wait(10.0)
            if timed_out:
                clock.advance(10.0)
                assert wait_until(lambda: result_ready(ctrl, req))
                fallback = ctrl.query_prefetch_result(req)
                assert fallback is not None
                assert fallback.hit_cells[0].get_indices_list() == []
                assert wait_until(lambda: draining_count(ctrl) == 1)
            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            if timed_out:
                assert ctrl.query_prefetch_result(req) is None
            else:
                result = ctrl.query_prefetch_result(req)
                assert result is not None
                assert result.hit_cells[0].get_indices_list() == [0, 1]
                l1_manager.finish_read(keys)
            assert l1_manager.report_status()["read_locked_count"] == 0
            assert listener.pop_pending_keys() == []
            for key in keys:
                state = l1_manager.get_object_state(key)
                assert state is not None
                assert not state.is_temporary
            assert all(
                result[0] == L1Error.SUCCESS
                for result in l1_manager.reserve_read(keys).values()
            )
            l1_manager.finish_read(keys)
            ordinary_key = make_object_key(10)
            reserved = l1_manager.reserve_write([ordinary_key], [False], layout)
            assert reserved[ordinary_key][0] == L1Error.SUCCESS
            l1_manager.finish_write([ordinary_key])
            assert listener.pop_pending_keys() == [ordinary_key]
        finally:
            adapter.release_loads()
            ctrl.stop()
            adapter.close()
            listener.close()

    @pytest.mark.parametrize("at_deadline", [False, True])
    def test_completion_before_deadline_wins(
        self, l1_manager, monkeypatch, at_deadline
    ):
        """Completed loads win at the deadline and cannot expire afterwards."""
        adapter = make_gated_adapter()
        adapter.release_loads()
        clock = FakeClock()
        if at_deadline:
            original_query = adapter.query_load_result

            def advance_on_completion(task_id):
                result = original_query(task_id)
                if result is not None:
                    clock.advance(5.0)
                return result

            monkeypatch.setattr(adapter, "query_load_result", advance_on_completion)
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
            assert result.hit_cells[0].count_leading_ones() == 4
            l1_manager.finish_read(keys)
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.report_status()["deadline_timeout_count"] == 0
            clock.advance(100.0)
            assert in_flight_count(ctrl) == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_head_of_line_drainers_then_queue_expiry_then_admission(self, l1_manager):
        """Drainers occupy admission slots until their I/O completes."""
        adapter = make_gated_adapter()
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
            assert wait_until(lambda: in_flight_count(ctrl) == 2)
            assert adapter.load_entered.wait(10.0)

            clock.advance(10.0)
            assert wait_until(lambda: draining_count(ctrl) == 2)
            assert wait_until(lambda: result_ready(ctrl, r0) and result_ready(ctrl, r1))

            r2 = submit(ctrl, key_sets[2], layout)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, r2))
            assert ctrl.query_prefetch_result(r2) is not None

            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)

            r3 = submit(ctrl, key_sets[3], layout)
            assert wait_until(lambda: result_ready(ctrl, r3))
            assert ctrl.query_prefetch_result(r3) is not None
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.mark.parametrize("at_completion", [False, True])
    def test_timeout_during_lookup_late_lookup_releases_l2_lock(
        self, l1_manager, monkeypatch, at_completion
    ):
        """Late lookup releases its locks without starting a load."""
        adapter = make_gated_adapter()
        adapter.gate_lookups()
        clock = FakeClock()
        if at_completion:
            original_query = adapter.query_lookup_and_lock_result

            def advance_on_completion(task_id):
                result = original_query(task_id)
                if result is not None:
                    clock.advance(5.0)
                return result

            monkeypatch.setattr(
                adapter, "query_lookup_and_lock_result", advance_on_completion
            )
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            store_keys_in_l2(adapter, keys, layout)
            req = submit(ctrl, keys, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert ctrl.report_status()["lookup_phase_count"] == 1

            if at_completion:
                adapter.release_lookups()
            else:
                clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            if not at_completion:
                assert wait_until(lambda: draining_count(ctrl) == 1)
            retained = ctrl.query_prefetch_result(req)
            assert (
                retained is not None and retained.hit_cells[0].count_leading_ones() == 0
            )

            adapter.release_lookups()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert wait_until(lambda: adapter.debug_locked_key_count() == 0)
            assert not adapter.load_entered.is_set()
        finally:
            ctrl.stop()
            adapter.close()

    def test_queue_expiry_emits_coherent_events_and_metric(
        self, l1_manager, event_capture
    ):
        """Queued fallback emits lookup submission before its deadline event."""
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0, max_in_flight=1)
        ctrl.start()
        try:
            layout = make_layout()
            ks0 = [make_object_key(i) for i in range(4)]
            ks1 = [make_object_key(10 + i) for i in range(4)]
            store_keys_in_l2(adapter, ks0, layout)
            store_keys_in_l2(adapter, ks1, layout)

            submit(ctrl, ks0, layout)
            assert wait_until(lambda: in_flight_count(ctrl) == 1)
            assert adapter.load_entered.wait(10.0)
            clock.advance(10.0)
            assert wait_until(lambda: draining_count(ctrl) == 1)

            r1 = submit(ctrl, ks1, layout)
            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, r1))

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
                for e in event_capture.of_type(EventType.L2_PREFETCH_DEADLINE)
                if e.metadata["request_id"] == r1
            ]
            assert submitted and completed
            assert submitted[0].metadata["adapter_count"] == 0
            assert submitted[0].timestamp <= completed[0].timestamp

            adapter.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.mark.parametrize(
        "policy, retained_readlocked",
        [
            ("prefix", False),
            ("full", True),
        ],
    )
    @pytest.mark.parametrize("fail_load", [False, True])
    def test_partial_completion_policy_lock_accounting(
        self, l1_manager, policy, retained_readlocked, fail_load
    ):
        """Only policy-selected completed cells keep caller read locks."""
        slow = make_gated_adapter()
        if fail_load:
            slow.fail_loads()
        fast = make_gated_adapter()
        fast.release_loads()
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
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)

            clock.advance(10.0)
            assert wait_until(lambda: result_ready(ctrl, req))
            retained = ctrl.query_prefetch_result(req)
            assert retained is not None
            retained_idx = set(retained.hit_cells[0].get_indices_list())

            probe = l1_manager.unsafe_read([keys[1], keys[3]])
            if retained_readlocked:
                assert retained_idx == {1, 3}
                assert probe[keys[1]][0] == L1Error.SUCCESS
                assert probe[keys[3]][0] == L1Error.SUCCESS
                l1_manager.finish_read([keys[1], keys[3]])
            else:
                assert retained_idx == set()
                assert probe[keys[1]][0] == L1Error.KEY_NOT_EXIST
                assert probe[keys[3]][0] == L1Error.KEY_NOT_EXIST

            slow.release_loads()
            assert wait_until(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert l1_manager.report_status()["write_locked_count"] == 0
            assert l1_manager.report_status()["read_locked_count"] == 0
            assert wait_until(lambda: slow.debug_locked_key_count() == 0)
            assert wait_until(lambda: fast.debug_locked_key_count() == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()
