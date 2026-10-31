# SPDX-License-Identifier: Apache-2.0
"""L2 prefetch deadline tests using the native L1 manager and gated I/O."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from unittest.mock import Mock
import asyncio
import json
import threading
import uuid

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import PrefetchLockMode
from lmcache.v1.distributed.config import (
    EvictionConfig,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
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
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import EventBusConfig, init_event_bus
from tests.v1.distributed.test_prefetch_controller import (
    CXL_TAGS,
    DRAM_TAGS,
    RetaggedL1Manager,
    make_descriptor,
    make_group,
    make_l1_config,
    make_layout,
    make_object_key,
    make_spec,
    single_row_spec,
    store_keys_in_l2,
    wait_for_condition,
    write_keys_to_l1,
)

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


def make_gated_adapter() -> GatedLoadMockAdapter:
    return GatedLoadMockAdapter(
        MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
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


def submit(ctrl, keys, policy="prefix") -> int:
    spec = single_row_spec(keys, num_kv_readers=1, fetching_policy=policy)
    return ctrl.submit_prefetch_request(spec)


@pytest.fixture(autouse=True)
def _fast_loop(monkeypatch):
    monkeypatch.setattr(prefetch_controller, "PREFETCH_LOOP_POLL_TIMEOUT_MS", 20)


@pytest.fixture
def l1_manager():
    mgr = DramL1Manager(make_l1_config())
    yield mgr
    mgr.close()


def make_controller(l1_manager, adapter, clock, timeout, inflight_watermark=0.5):
    return make_controller_multi(
        l1_manager, [adapter], clock, timeout, inflight_watermark
    )


def make_controller_multi(l1_manager, adapters, clock, timeout, inflight_watermark=0.5):
    return PrefetchController(
        l1_managers=[l1_manager],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=make_l1_config())],
        l2_adapters=list(adapters),
        adapter_descriptors=[make_descriptor(i) for i in range(len(adapters))],
        policy=DefaultPrefetchPolicy(),
        inflight_watermark=inflight_watermark,
        l2_load_timeout=timeout,
        clock=clock,
    )


@pytest.fixture
def events():
    bus = init_event_bus(EventBusConfig(enabled=True))
    captured = []
    for event_type in (
        EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
        EventType.L2_PREFETCH_DEADLINE,
    ):
        bus.subscribe(event_type, captured.append)
    bus.start()
    try:
        yield captured
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


def _base_config_kwargs():
    return dict(
        l1_manager_config=make_l1_config(),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


class TestConfigValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
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


class TestLoadDeadline:
    def test_reused_spec_gets_independent_submission_deadlines(self, l1_manager):
        adapter = make_gated_adapter()
        adapter.gate_lookups()
        clock = FakeClock()
        ctrl = make_controller(l1_manager, adapter, clock, timeout=5.0)
        spec = single_row_spec([make_object_key(0)])
        ctrl.start()
        try:
            first = ctrl.submit_prefetch_request(spec)
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 1)
            clock.advance(2.0)
            second = ctrl.submit_prefetch_request(spec)
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 2)
            assert spec.deadline_at is None

            clock.advance(4.0)
            assert wait_for_condition(lambda: result_ready(ctrl, first))
            assert not result_ready(ctrl, second)
            clock.advance(2.0)
            assert wait_for_condition(lambda: result_ready(ctrl, second))
            for request_id in (first, second):
                result = ctrl.query_prefetch_result(request_id)
                assert result is not None and result.hit_cells[0].popcount() == 0
        finally:
            adapter.release_lookups()
            adapter.release_loads()
            ctrl.stop()
            adapter.close()

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
            req = submit(ctrl, keys)
            assert slow.load_entered.wait(10.0)
            assert fast.load_result_consumed.wait(10.0)
            clock.advance(10.0)
            assert wait_for_condition(lambda: result_ready(ctrl, req))
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
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert l1_manager.unsafe_read(keys[:1])[keys[0]][0] == L1Error.SUCCESS
            l1_manager.finish_read(keys[:1])
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
            warm_key = make_object_key(10)
            store_keys_in_l2(adapter, keys + [warm_key], layout)
            req = submit(ctrl, keys)
            warm_req = ctrl.submit_prefetch_request(
                single_row_spec([warm_key], lock_mode=PrefetchLockMode.NO_LOCK)
            )
            assert wait_for_condition(
                lambda: l1_manager.report_status()["write_locked_count"] == 5
            )
            clock.advance(10.0)

            assert wait_for_condition(lambda: result_ready(ctrl, req))
            assert wait_for_condition(lambda: draining_count(ctrl) == 1)
            assert active_count(ctrl) == 1
            assert not result_ready(ctrl, warm_req)
            assert ctrl.report_status()["deadline_timeout_count"] == 1
            fallback = ctrl.query_prefetch_result(req)
            assert fallback is not None
            assert fallback.hit_cells[0].get_indices_list() == []
            assert l1_manager.report_status()["write_locked_count"] == 5
            assert adapter.debug_get_locked_key_count() == 5

            adapter.release_loads()
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert ctrl.query_prefetch_result(warm_req) is not None
            assert ctrl.report_status()["deadline_timeout_count"] == 1
            assert l1_manager.report_status()["write_locked_count"] == 0
            assert wait_for_condition(lambda: adapter.debug_get_locked_key_count() == 0)
            assert l1_manager.report_status()["read_locked_count"] == 0
            if fail_load:
                assert all(
                    state[0] == L1Error.KEY_NOT_EXIST
                    for state in l1_manager.unsafe_read(keys).values()
                )

            adapter.fail_loads(False)
            retry = submit(ctrl, keys)
            assert wait_for_condition(lambda: result_ready(ctrl, retry))
            result = ctrl.query_prefetch_result(retry)
            assert result is not None
            assert result.hit_cells[0].get_indices_list() == list(range(len(keys)))
            l1_manager.finish_read(keys)
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
        finally:
            ctrl.stop()
            adapter.close()

    def test_completion_before_deadline_wins(self, l1_manager, monkeypatch):
        """Completed loads win at the deadline and cannot expire afterwards."""
        adapter = make_gated_adapter()
        adapter.release_loads()
        clock = FakeClock()
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
            req = submit(ctrl, keys)
            assert wait_for_condition(lambda: result_ready(ctrl, req))
            result = ctrl.query_prefetch_result(req)
            assert result is not None
            assert result.hit_cells[0].count_leading_ones() == 4
            l1_manager.finish_read(keys)
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.report_status()["deadline_timeout_count"] == 0
            clock.advance(100.0)
            assert in_flight_count(ctrl) == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_queue_deadline_behind_warmup_drains_then_resumes(
        self, l1_manager, monkeypatch, events
    ):
        poller = Mock(wraps=prefetch_controller.select.poll())
        monkeypatch.setattr(prefetch_controller.select, "poll", lambda: poller)
        adapter = make_gated_adapter()
        clock = FakeClock()
        ctrl = make_controller(
            l1_manager, adapter, clock, timeout=0.005, inflight_watermark=1e-6
        )
        layout = make_layout()
        busy, warm, queued = [make_object_key(i) for i in range(3)]
        store_keys_in_l2(adapter, [busy, warm], layout)
        write_keys_to_l1(l1_manager, [queued], layout)
        touch = Mock(wraps=l1_manager.touch_keys)
        monkeypatch.setattr(l1_manager, "touch_keys", touch)
        ctrl.start()
        try:
            busy_req = submit(ctrl, [busy])
            assert adapter.load_entered.wait(10)
            clock.advance(1)
            assert wait_for_condition(lambda: draining_count(ctrl) == 1)
            assert ctrl.query_prefetch_result(busy_req) is not None
            warm_req = ctrl.submit_prefetch_request(
                single_row_spec([warm], lock_mode=PrefetchLockMode.NO_LOCK)
            )
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_queue_size"] == 1
            )
            poller.poll.reset_mock()
            queued_req = submit(ctrl, [queued])
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_queue_size"] == 2
            )
            assert wait_for_condition(
                lambda: any(
                    0 < call.args[0] < 20 for call in poller.poll.call_args_list
                )
            )
            clock.advance(1)
            assert wait_for_condition(lambda: result_ready(ctrl, queued_req))
            result = ctrl.query_prefetch_result(queued_req)
            assert result is not None and result.l1_hit_count == 1
            assert result.l1_owners == {queued: l1_manager.l1_manager_id}
            touch.assert_any_call([queued])
            assert not result_ready(ctrl, warm_req)
            assert ctrl.report_status()["deadline_timeout_count"] == 2
            assert wait_for_condition(
                lambda: any(
                    e.event_type == EventType.L2_PREFETCH_DEADLINE
                    and e.metadata["request_id"] == queued_req
                    for e in events
                )
            )
            request_events = [
                e for e in events if e.metadata["request_id"] == queued_req
            ]
            assert [e.event_type for e in request_events] == [
                EventType.L2_PREFETCH_LOOKUP_SUBMITTED,
                EventType.L2_PREFETCH_DEADLINE,
            ]
            assert request_events[0].metadata["adapter_count"] == 0
            assert request_events[1].metadata["phase"] == "queued"
            assert request_events[1].metadata["retained_chunks"] == 1
            l1_manager.finish_read([queued])
            adapter.release_loads()
            assert wait_for_condition(lambda: result_ready(ctrl, warm_req))
            assert ctrl.query_prefetch_result(warm_req) is not None
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            retry = submit(ctrl, [busy])
            assert wait_for_condition(lambda: result_ready(ctrl, retry))
            result = ctrl.query_prefetch_result(retry)
            assert result is not None and result.hit_cells[0].get_indices_list() == [0]
            l1_manager.finish_read([busy])
        finally:
            adapter.release_loads()
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
            req = submit(ctrl, keys)
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 1)
            assert ctrl.report_status()["lookup_phase_count"] == 1

            if at_completion:
                adapter.release_lookups()
            else:
                clock.advance(10.0)
            assert wait_for_condition(lambda: result_ready(ctrl, req))
            if not at_completion:
                assert wait_for_condition(lambda: draining_count(ctrl) == 1)
            retained = ctrl.query_prefetch_result(req)
            assert (
                retained is not None and retained.hit_cells[0].count_leading_ones() == 0
            )

            adapter.release_lookups()
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            assert wait_for_condition(lambda: adapter.debug_get_locked_key_count() == 0)
            assert not adapter.load_entered.is_set()
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.mark.parametrize("policy, kept", [("prefix", []), ("full", [1, 3])])
    def test_partial_completion_policy_lock_accounting(
        self, l1_manager, policy, kept, events
    ):
        slow, fast = make_gated_adapter(), make_gated_adapter()
        fast.release_loads()
        clock = FakeClock()
        ctrl = make_controller_multi(l1_manager, [slow, fast], clock, timeout=5.0)
        rows = [
            [make_object_key(i, kv_rank=rank) for i in range(4)] for rank in range(2)
        ]
        completed = [row[i] for row in rows for i in (1, 3)]
        layout = make_layout()
        ctrl.start()
        try:
            store_keys_in_l2(slow, [row[i] for row in rows for i in (0, 2)], layout)
            store_keys_in_l2(fast, completed, layout)
            req = ctrl.submit_prefetch_request(
                make_spec([make_group(row) for row in rows], fetching_policy=policy)
            )
            assert slow.load_entered.wait(10)
            assert fast.load_result_consumed.wait(10)
            clock.advance(10)
            assert wait_for_condition(lambda: result_ready(ctrl, req))
            result = ctrl.query_prefetch_result(req)
            assert result is not None
            assert [row.get_indices_list() for row in result.hit_cells] == [kept, kept]
            assert result.l2_hit_count == 2 * len(kept)
            if policy == "full":
                assert result.found_cells is not None
                assert [row.get_indices_list() for row in result.found_cells] == [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3],
                ]
            probe = l1_manager.unsafe_read(completed)
            expected = L1Error.SUCCESS if kept else L1Error.KEY_NOT_EXIST
            assert all(state[0] == expected for state in probe.values())
            if kept:
                l1_manager.finish_read(completed)
            assert wait_for_condition(
                lambda: any(
                    e.event_type == EventType.L2_PREFETCH_DEADLINE
                    and e.metadata["request_id"] == req
                    for e in events
                )
            )
            deadline = next(
                e
                for e in events
                if e.event_type == EventType.L2_PREFETCH_DEADLINE
                and e.metadata["request_id"] == req
            )
            # Count chunks, not rank cells, including non-prefix full-policy hits.
            assert deadline.metadata["retained_chunks"] == len(kept)
            assert deadline.metadata["missed_chunks"] == 4 - len(kept)
            slow.release_loads()
            assert wait_for_condition(lambda: in_flight_count(ctrl) == 0)
            assert ctrl.query_prefetch_result(req) is None
            assert l1_manager.report_status()["write_locked_count"] == 0
            assert l1_manager.report_status()["read_locked_count"] == 0
            assert wait_for_condition(lambda: slow.debug_get_locked_key_count() == 0)
            assert wait_for_condition(lambda: fast.debug_get_locked_key_count() == 0)
        finally:
            ctrl.stop()
            slow.close()
            fast.close()


@pytest.fixture
def managers() -> Iterator[list[DramL1Manager]]:
    """Use distinct real L1 pools for the default and archive affinities."""
    config = make_l1_config(8 * 1024 * 1024)
    pools = [
        DramL1Manager(
            replace(
                config,
                tag=tag,
                memory_config=replace(
                    config.memory_config,
                    shm_name=uuid.uuid4().hex[:12],
                ),
            )
        )
        for tag in ("_default", "archive")
    ]
    try:
        yield pools
    finally:
        for manager in pools:
            manager.close()


def _cli_args() -> list[str]:
    return [
        "--eviction-policy",
        "noop",
        *[
            value
            for tag in ("_default", "archive")
            for value in (
                "--l1-manager",
                json.dumps({"type": "DRAM", "tag": tag, "size_gb": 1}),
            )
        ],
    ]


@pytest.mark.parametrize("timeout", [None, 0.25])
def test_multi_l1_cli_preserves_deadline(timeout: float | None) -> None:
    args = _cli_args()
    if timeout is not None:
        args += ["--l2-prefetch-load-timeout", str(timeout)]
    config = parse_args(args)
    assert config.prefetch_load_timeout == timeout
    assert [manager.tag for manager in config.l1_manager_configs] == [
        "_default",
        "archive",
    ]


def test_partial_deadline_keeps_affinity_and_drains_without_store(
    managers: list[DramL1Manager],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, kept = "full", [0, 2]
    configs = [MockL2AdapterConfig(0.05, 10.0) for _ in managers]
    for config, tags in zip(configs, (DRAM_TAGS, CXL_TAGS), strict=True):
        monkeypatch.setattr(
            config, "get_wanted_l1_memory_tags", lambda tags=tags: (tags,)
        )
    adapters = [GatedLoadMockAdapter(config) for config in configs]
    slow, fast = adapters
    descriptors = [
        L2AdapterDescriptor(index=index, config=config)
        for index, config in enumerate(configs)
    ]
    stores = [
        StoreController(manager, [adapter], [descriptor], DefaultStorePolicy())
        for manager, adapter, descriptor in zip(
            managers, adapters, descriptors, strict=True
        )
    ]
    clock = FakeClock()
    controller = PrefetchController(
        l1_managers=[managers[0], RetaggedL1Manager(managers[1], CXL_TAGS)],  # type: ignore[list-item]
        l1_manager_descriptors=[
            L1ManagerDescriptor(index=manager.l1_manager_id, config=manager.config)
            for manager in managers
        ],
        l2_adapters=list(adapters),
        adapter_descriptors=descriptors,
        policy=RetainPrefetchPolicy(),
        l2_load_timeout=0.5,
        clock=clock,
    )
    for store in stores:
        store.start()
    controller.start()
    try:
        layout = make_layout()
        keys = [make_object_key(index) for index in range(3)]
        store_keys_in_l2(slow, [keys[1]], layout)
        store_keys_in_l2(fast, [keys[0], keys[2]], layout)
        store_calls = []
        for adapter in adapters:
            recorder = Mock(wraps=adapter.submit_store_task)
            monkeypatch.setattr(adapter, "submit_store_task", recorder)
            store_calls.append(recorder)
        fast.release_loads()
        request_id = submit(controller, keys, policy)
        assert slow.load_entered.wait(10)
        assert fast.load_result_consumed.wait(10)
        clock.advance(1)
        assert wait_for_condition(lambda: result_ready(controller, request_id))
        result = controller.query_prefetch_result(request_id)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == kept
        assert result.l1_owners == {
            keys[index]: managers[1].l1_manager_id for index in kept
        }
        assert managers[0].report_status()["staging_object_count"] == 1
        assert managers[1].report_status()["read_locked_count"] == len(kept)
        assert all(managers[0].get_object_state(keys[index]) is None for index in kept)

        # Drain cleanup must preserve both callers' read locks.
        managers[1].reserve_read([keys[0]])
        slow.release_loads()
        assert wait_for_condition(lambda: in_flight_count(controller) == 0)
        assert controller.query_prefetch_result(request_id) is None
        assert controller.report_status()["deadline_timeout_count"] == 1
        assert all(adapter.debug_get_locked_key_count() == 0 for adapter in adapters)
        assert all(
            manager.report_status()["staging_object_count"] == 0 for manager in managers
        )
        assert managers[0].get_object_state(keys[1]) is not None
        assert managers[0].report_status()["read_locked_count"] == 0
        assert managers[1].report_status()["read_locked_count"] == len(kept)
        managers[1].finish_read([keys[index] for index in kept])
        assert managers[1].report_status()["read_locked_count"] == 1
        managers[1].finish_read([keys[0]])
        assert managers[1].report_status()["read_locked_count"] == 0
        for recorder, store in zip(store_calls, stores, strict=True):
            recorder.assert_not_called()
            assert store.report_status()["pending_keys_count"] == 0
        assert all(manager.memcheck() for manager in managers)
    finally:
        slow.release_loads()
        fast.release_loads()
        controller.stop()
        for store in stores:
            store.stop()
        for adapter in adapters:
            adapter.close()
