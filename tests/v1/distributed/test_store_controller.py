# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for StoreController.

Tests verify the end-to-end flow: L1 write -> listener notification ->
L2 store submission -> completion handling -> L1 read lock release.

Uses a real DramL1Manager and MockL2Adapter (with debug methods) to exercise
the full integration without mocking internals.
"""

# Standard
from typing import NoReturn
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction_policy.noop import (
    NoOpEvictionPolicy,
)
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers import store_controller
from lmcache.v1.distributed.storage_controllers.store_controller import (
    STORE_SLICE_MAX_KEYS,
    StoreController,
    StoreListener,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    BufferOnlyStorePolicy,
    DefaultStorePolicy,
    StorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import Event, EventType
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


def make_object_key(chunk_id: int, kv_rank: int = 0) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=kv_rank,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc for testing."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    """
    Poll until a predicate returns True or timeout.

    Args:
        predicate: Callable returning bool.
        timeout: Max wait time in seconds.
        poll_interval: Time between polls in seconds.

    Returns:
        True if predicate was satisfied, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def write_keys_to_l1(
    l1_manager: DramL1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> list[ObjectKey]:
    """
    Write keys to L1 (reserve_write + finish_write).

    Args:
        l1_manager: The L1 manager to write to.
        keys: Object keys to write.
        layout: Memory layout description.

    Returns:
        List of keys that were successfully written.
    """
    results = l1_manager.reserve_write(
        keys=keys,
        is_temporary=[False] * len(keys),
        layout_desc=layout,
    )
    written = [k for k, (e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)
    return written


def budget_config(budget_bytes: float, total_bytes: int) -> EvictionConfig:
    """LRU settings whose store budget, ``trigger_watermark *
    (1 - eviction_ratio)`` of L1, is ``budget_bytes``."""
    return EvictionConfig(
        eviction_policy="LRU",
        trigger_watermark=0.5,
        eviction_ratio=1 - 2 * budget_bytes / total_bytes,
    )


# =============================================================================
# Fixtures
# =============================================================================


def make_l1_manager(read_ttl_seconds: int = 300) -> DramL1Manager:
    """Create an DramL1Manager with a reasonable memory config."""
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=read_ttl_seconds,
    )
    return DramL1Manager(config)


@pytest.fixture
def l1_manager():
    mgr = make_l1_manager()
    yield mgr
    mgr.close()


@pytest.fixture
def short_ttl_l1_manager():
    """An L1 whose read locks lapse after 1 s."""
    mgr = make_l1_manager(read_ttl_seconds=1)
    yield mgr
    mgr.close()


@pytest.fixture
def read_failures():
    """Enable the event bus and collect L1_READ_FAILED events. Controllers
    pick up the bus when built, so build them after this fixture."""
    bus = init_event_bus(EventBusConfig(enabled=True))
    events: list[Event] = []
    bus.subscribe(EventType.L1_READ_FAILED, events.append)
    bus.start()
    yield events
    bus.stop()
    init_event_bus(EventBusConfig(enabled=False))


def make_adapter() -> MockL2Adapter:
    """Create a MockL2Adapter with fast bandwidth."""
    config = MockL2AdapterConfig(
        max_size_gb=0.01,
        mock_bandwidth_gb=10.0,
    )
    return MockL2Adapter(config)


def make_descriptor(index: int) -> L2AdapterDescriptor:
    """Create an L2AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)
    return L2AdapterDescriptor(index=index, config=config)


# =============================================================================
# StoreListener Tests
# =============================================================================


class TestStoreListener:
    """Test StoreListener as a standalone component."""

    def test_pop_pending_changes_returns_empty_initially(self):
        """A fresh listener should have no pending changes."""
        listener = StoreListener()
        assert listener.pop_pending_changes() == ([], [])
        listener.close()

    def test_on_l1_keys_write_finished_enqueues_keys(self):
        """Keys passed to on_l1_keys_write_finished should be retrievable via pop."""
        listener = StoreListener()
        keys = [make_object_key(i) for i in range(3)]

        listener.on_l1_keys_write_finished(keys)
        written, _ = listener.pop_pending_changes()

        assert written == keys
        listener.close()

    def test_pop_pending_changes_clears_queue(self):
        """pop_pending_changes should drain the queue."""
        listener = StoreListener()
        listener.on_l1_keys_write_finished([make_object_key(0)])
        listener.on_l1_keys_deleted_by_manager([make_object_key(1)])
        listener.pop_pending_changes()

        assert listener.pop_pending_changes() == ([], [])
        listener.close()

    def test_deletion_drops_earlier_writes_of_the_key(self):
        """A key deleted before the pop is not reported as written, unless
        it is written again after the deletion."""
        listener = StoreListener()
        k0, k1 = make_object_key(0), make_object_key(1)
        listener.on_l1_keys_write_finished([k0, k1])
        listener.on_l1_keys_deleted_by_manager([k0, k1])
        listener.on_l1_keys_write_finished([k1])

        assert listener.pop_pending_changes() == ([k1], [k0, k1])
        listener.close()

    def test_on_l1_keys_write_finished_signals_eventfd(self):
        """The eventfd should become readable after on_l1_keys_write_finished."""
        listener = StoreListener()
        efd = listener.get_event_fd()
        poller = select.poll()
        poller.register(efd, select.POLLIN)

        listener.on_l1_keys_write_finished([make_object_key(0)])

        events = poller.poll(1000)
        assert len(events) > 0
        listener.close()

    def test_multiple_writes_accumulate(self):
        """Multiple on_l1_keys_write_finished calls should accumulate keys."""
        listener = StoreListener()
        listener.on_l1_keys_write_finished([make_object_key(0)])
        listener.on_l1_keys_write_finished([make_object_key(1), make_object_key(2)])

        written, _ = listener.pop_pending_changes()
        assert len(written) == 3
        listener.close()

    def test_finish_write_and_reserve_read_does_not_enqueue(self):
        """on_l1_keys_finish_write_and_reserve_read should not enqueue keys."""
        listener = StoreListener()
        keys = [make_object_key(i) for i in range(3)]

        listener.on_l1_keys_finish_write_and_reserve_read(keys)

        assert listener.pop_pending_changes() == ([], [])
        assert listener.pending_count() == 0
        listener.close()


# =============================================================================
# StoreController Integration Tests
# =============================================================================


class TestStoreControllerLifecycle:
    """Test StoreController start/stop behavior."""

    def test_start_stop(self, l1_manager):
        """StoreController should start and stop cleanly."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()
        ctrl.stop()
        adapter.close()

    def test_start_stop_no_adapters(self, l1_manager):
        """Controller should start and stop cleanly with no adapters."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()
        ctrl.stop()


class TestStoreControllerSingleAdapter:
    """Test StoreController with one MockL2Adapter."""

    def test_temporary_l1_write_does_not_trigger_l2_store(self, l1_manager):
        """Temporary staging objects should remain internal to L1."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        result = l1_manager.reserve_write(
            keys=keys,
            is_temporary=[True],
            layout_desc=layout,
        )
        assert result[keys[0]][1] is not None

        l1_manager.finish_write(keys)
        time.sleep(0.3)

        assert adapter.debug_get_stored_object_count() == 0
        assert l1_manager.delete(keys)[keys[0]] == L1Error.SUCCESS

        ctrl.stop()
        adapter.close()

    def test_submit_failure_releases_l1_read_lock(self, l1_manager, monkeypatch):
        """An adapter submission failure should not leak L1 read locks."""
        adapter = make_adapter()
        submission_attempted = threading.Event()

        def fail_submit(
            _keys: list[ObjectKey],
            _memory_objs: list[MemoryObj],
        ) -> NoReturn:
            submission_attempted.set()
            raise RuntimeError("injected store submission failure")

        monkeypatch.setattr(adapter, "submit_store_task", fail_submit)
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        assert submission_attempted.wait(timeout=5.0)
        ok = wait_for_condition(
            lambda: l1_manager.report_status()["read_locked_count"] == 0,
            timeout=5.0,
        )
        assert ok, "Submission failure should release the acquired L1 read lock"

        ctrl.stop()
        adapter.close()

    def test_l1_write_triggers_l2_store(self, l1_manager):
        """Writing to L1 should cause the object to appear in L2."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for the object to appear in L2
        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 1,
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2 after L1 write"
        assert adapter.debug_has_key(keys[0])

        ctrl.stop()
        adapter.close()

    def test_multiple_keys_stored_to_l2(self, l1_manager):
        """Multiple L1 writes should result in all keys in L2."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 5,
            timeout=5.0,
        )
        assert ok, "All 5 objects should be stored in L2"
        for key in keys:
            assert adapter.debug_has_key(key)

        ctrl.stop()
        adapter.close()

    def test_read_lock_released_after_store(self, l1_manager):
        """After L2 store completes, L1 read locks should be released."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store to complete
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok

        # Verify read lock is released: the key should be evictable
        ok = wait_for_condition(
            lambda: (l1_manager.is_key_evictable(keys[0])),
            timeout=5.0,
        )
        assert ok, "Key should be evictable after store controller releases read lock"

        ctrl.stop()
        adapter.close()

    def test_multiple_concurrent_requests_same_adapter(self, l1_manager):
        """Each L1-write batch produces an independent in-flight store request.

        When two batches target the same adapter, the adapter's single
        eventfd may batch-signal both completions. All in-flight requests
        should be advanced and their L1 read locks released.
        """
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        batch_a = [make_object_key(i) for i in range(3)]
        batch_b = [make_object_key(i) for i in range(3, 6)]

        write_keys_to_l1(l1_manager, batch_a, layout)
        write_keys_to_l1(l1_manager, batch_b, layout)

        all_keys = batch_a + batch_b
        ok = wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in all_keys),
            timeout=5.0,
        )
        assert ok, "Both batches should reach L2"

        # Every key must be evictable: read locks from both requests released.
        def all_keys_evictable() -> bool:
            return all(l1_manager.is_key_evictable(k) for k in all_keys)

        ok = wait_for_condition(all_keys_evictable, timeout=5.0)
        assert ok, "Read locks must be released for every concurrent request"

        ctrl.stop()
        adapter.close()


class TestStoreBudget:
    def test_groups_past_budget_wait_unlocked(self, l1_manager, read_failures):
        """With two groups in flight the third waits unlocked; evicting it
        while it waits drops it from the queue, so it never reaches L2 and
        counts as an L1 read failure."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        # Lazy allocation grows L1 in the background; wait for its full size.
        total_bytes = l1_manager.config.memory_config.size_in_bytes
        assert wait_for_condition(
            lambda: l1_manager.get_memory_usage()[1] == total_bytes
        )
        eviction = budget_config(1.5 * group_bytes, total_bytes)
        # About 3 s per group, so the first two stay in flight below.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / 3 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=eviction,
        )
        ctrl.start()

        groups = [[make_object_key(g * 4 + i) for i in range(4)] for g in range(3)]
        for keys in groups:
            write_keys_to_l1(l1_manager, keys, make_layout())
            # One listener drain per write, so each write is its own group.
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )
        assert wait_for_condition(
            lambda: ctrl.report_status()["queued_keys_count"] == 4
            and ctrl.report_status()["in_flight_task_count"] == 2
        )

        deleted = l1_manager.delete(groups[2])
        assert all(err == L1Error.SUCCESS for err in deleted.values())
        assert wait_for_condition(
            lambda: ctrl.report_status()["queued_keys_count"] == 0
            and ctrl.report_status()["in_flight_task_count"] == 2
        )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 0, timeout=10.0
        )
        assert all(adapter.debug_has_key(k) for k in groups[0] + groups[1])
        assert not any(adapter.debug_has_key(k) for k in groups[2])
        assert wait_for_condition(lambda: len(read_failures) == 1)
        meta = read_failures[0].metadata
        assert (meta["during"], meta["reason"]) == ("l2_store", "not_found")
        assert set(meta["keys"]) == set(groups[2])

        ctrl.stop()
        adapter.close()

    def test_key_deleted_and_rewritten_is_still_stored(self, l1_manager):
        """A key deleted and written again before the controller sees either
        change is queued as the new write and reaches L2."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=EvictionConfig(
                "LRU", trigger_watermark=0.5, eviction_ratio=1.0
            ),
        )
        first = [make_object_key(i) for i in range(4)]
        # Another kv_rank, so a separate group that waits behind ``first``.
        rewritten = [make_object_key(i, kv_rank=1) for i in range(4)]
        write_keys_to_l1(l1_manager, first + rewritten, make_layout())
        l1_manager.delete(rewritten)
        write_keys_to_l1(l1_manager, rewritten, make_layout())
        ctrl.start()

        assert wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in first + rewritten)
        )

        ctrl.stop()
        adapter.close()

    def test_group_larger_than_budget_goes_alone(self, l1_manager):
        """A group larger than the budget goes out when nothing is in flight,
        and later groups wait for it, so the overshoot never builds up."""
        object_bytes = make_layout().shapes[0].numel() * 2  # bf16
        total_bytes = l1_manager.config.memory_config.size_in_bytes
        assert wait_for_condition(
            lambda: l1_manager.get_memory_usage()[1] == total_bytes
        )
        eviction = budget_config(6 * object_bytes, total_bytes)
        # About 2 s for the large group, so it stays in flight below.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=5 * object_bytes / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=eviction,
        )
        ctrl.start()

        large = [make_object_key(i) for i in range(10)]
        small = [make_object_key(i) for i in range(10, 14)]
        for keys in (large, small):
            write_keys_to_l1(l1_manager, keys, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 1
            and ctrl.report_status()["queued_keys_count"] == 4
        )
        assert wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for k in large + small),
            timeout=10.0,
        )

        ctrl.stop()
        adapter.close()

    def test_key_stored_to_two_adapters_counts_once(self, l1_manager):
        """A key stored to two adapters is one L1 object, so it uses the
        budget once, until the slower adapter finishes."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        total_bytes = l1_manager.config.memory_config.size_in_bytes
        assert wait_for_condition(
            lambda: l1_manager.get_memory_usage()[1] == total_bytes
        )
        eviction = budget_config(1.5 * group_bytes, total_bytes)
        # About 1 s and 3 s per group.
        adapters = [
            MockL2Adapter(
                MockL2AdapterConfig(
                    max_size_gb=0.01, mock_bandwidth_gb=group_bytes / s / (1 << 30)
                )
            )
            for s in (1, 3)
        ]
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=[make_descriptor(i) for i in range(2)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=eviction,
        )
        ctrl.start()

        groups = [[make_object_key(g * 4 + i) for i in range(4)] for g in range(3)]
        for keys in groups:
            write_keys_to_l1(l1_manager, keys, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )

        def tasks_and_queued() -> tuple[int, int]:
            status = ctrl.report_status()
            return status["in_flight_task_count"], status["queued_keys_count"]

        # Two groups go to both adapters and the third waits, also after the
        # fast adapter is done: the slow one still locks both groups.
        assert wait_for_condition(lambda: tasks_and_queued() == (4, 4))
        assert wait_for_condition(lambda: tasks_and_queued() == (2, 4))
        assert wait_for_condition(lambda: tasks_and_queued() == (0, 0), timeout=10.0)

        ctrl.stop()
        for a in adapters:
            a.close()

    @pytest.mark.parametrize(
        "eviction",
        [
            EvictionConfig("LRU", trigger_watermark=0.5, eviction_ratio=0.6),
            EvictionConfig("noop", trigger_watermark=0.5, eviction_ratio=1.0),
        ],
        ids=["ratio_above_watermark", "noop"],
    )
    def test_small_groups_go_out_together(self, l1_manager, eviction):
        """A ratio above the watermark still leaves a budget (20% of L1 here),
        and noop has none, so three small groups are in flight at once."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        # About 2 s per group, so all three stay in flight below.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / 2 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=eviction,
        )
        ctrl.start()

        groups = [[make_object_key(g * 4 + i) for i in range(4)] for g in range(3)]
        for keys in groups:
            write_keys_to_l1(l1_manager, keys, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 3
        )
        assert wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for g in groups for k in g),
            timeout=10.0,
        )

        ctrl.stop()
        adapter.close()

    def test_ratio_one_sends_one_group_at_a_time(self, l1_manager):
        """With eviction_ratio 1 the budget is zero, so stores go one group
        at a time and all of them still reach L2."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        # About 1 s per group, so the first one stays in flight below.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=EvictionConfig(
                "LRU", trigger_watermark=0.5, eviction_ratio=1.0
            ),
        )
        ctrl.start()

        groups = [[make_object_key(g * 4 + i) for i in range(4)] for g in range(3)]
        for keys in groups:
            write_keys_to_l1(l1_manager, keys, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 1
            and ctrl.report_status()["queued_keys_count"] == 8
        )
        assert wait_for_condition(
            lambda: all(adapter.debug_has_key(k) for g in groups for k in g),
            timeout=10.0,
        )

        ctrl.stop()
        adapter.close()

    def test_stop_clears_queue_and_in_flight_status(self, l1_manager):
        """stop() drops the queued keys and the unfinished stores, and
        releases their read locks."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        # About 3 s per group, so the first one is still in flight at stop().
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / 3 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            eviction_config=EvictionConfig(
                "LRU", trigger_watermark=0.5, eviction_ratio=1.0
            ),
        )
        ctrl.start()

        groups = [[make_object_key(g * 4 + i) for i in range(4)] for g in range(3)]
        for keys in groups:
            write_keys_to_l1(l1_manager, keys, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["pending_keys_count"] == 0
            )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 1
            and ctrl.report_status()["queued_keys_count"] == 8
        )

        ctrl.stop()
        status = ctrl.report_status()
        assert status["in_flight_task_count"] == 0
        assert status["queued_keys_count"] == 0
        assert l1_manager.report_status()["read_locked_count"] == 0
        adapter.close()

    @pytest.mark.parametrize(
        "has_inflight_loads",
        [None, lambda _adapter_ids: False],
        ids=["whole_groups", "load_tracking"],
    )
    def test_store_stops_holding_budget_when_its_locks_lapse(
        self, short_ttl_l1_manager, has_inflight_loads
    ):
        """A store slower than the read lock TTL stops holding back the queue
        once its locks lapse, so the next store goes out while it still
        runs."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / 20 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=short_ttl_l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([short_ttl_l1_manager]),
            eviction_config=EvictionConfig(
                "LRU", trigger_watermark=0.5, eviction_ratio=1.0
            ),
            has_inflight_loads=has_inflight_loads,
        )
        ctrl.start()

        keys = [make_object_key(i, kv_rank=r) for r in range(2) for i in range(4)]
        write_keys_to_l1(short_ttl_l1_manager, keys, make_layout())
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 1
            and ctrl.report_status()["queued_keys_count"] == 4
        )
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 2
        )

        ctrl.stop()
        adapter.close()

    def test_store_finishing_after_its_locks_lapse_frees_budget_once(
        self, short_ttl_l1_manager
    ):
        """A store that finishes after its read locks lapsed doesn't free its
        budget again, so later groups still wait for each other."""
        group_bytes = 4 * make_layout().shapes[0].numel() * 2  # 4 bf16 objects
        # About 3 s per group, so the locks lapse well before it finishes.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=group_bytes / 3 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=short_ttl_l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([short_ttl_l1_manager]),
            eviction_config=EvictionConfig(
                "LRU", trigger_watermark=0.5, eviction_ratio=1.0
            ),
        )
        ctrl.start()

        first = [make_object_key(i) for i in range(4)]
        write_keys_to_l1(short_ttl_l1_manager, first, make_layout())
        assert wait_for_condition(lambda: adapter.debug_has_key(first[0]))
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 0
        )

        later = [make_object_key(i, kv_rank=r) for r in (1, 2) for i in range(4)]
        write_keys_to_l1(short_ttl_l1_manager, later, make_layout())
        assert wait_for_condition(
            lambda: ctrl.report_status()["in_flight_task_count"] == 1
            and ctrl.report_status()["queued_keys_count"] == 4
        )

        ctrl.stop()
        adapter.close()


class TestStoreControllerMultipleAdapters:
    """Test StoreController with multiple L2 adapters."""

    def test_stores_to_all_adapters(self, l1_manager):
        """DefaultStorePolicy sends keys to all adapters."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for both adapters to have the object
        ok = wait_for_condition(
            lambda: all(a.debug_has_key(keys[0]) for a in adapters),
            timeout=5.0,
        )
        assert ok, "Object should be stored in both L2 adapters"

        ctrl.stop()
        for a in adapters:
            a.close()

    def test_read_lock_released_after_all_adapters_complete(self, l1_manager):
        """Read locks should be released after all adapter stores complete."""
        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for stores to both adapters, then check updatability
        ok = wait_for_condition(
            lambda: all(a.debug_has_key(keys[0]) for a in adapters),
            timeout=5.0,
        )
        assert ok

        ok = wait_for_condition(
            lambda: (l1_manager.is_key_evictable(keys[0])),
            timeout=5.0,
        )
        assert ok, "Key should be evictable after all adapter stores complete"

        ctrl.stop()
        for a in adapters:
            a.close()


class TestStoreControllerNoAdapters:
    """Test StoreController with no L2 adapters."""

    def test_write_with_no_adapters_no_read_lock(self, l1_manager):
        """With no adapters, no read locks should be held after L1 write."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Give the controller time to process the listener event
        time.sleep(0.3)

        # Key should be evictable (no read locks)
        assert l1_manager.is_key_evictable(keys[0])

        ctrl.stop()


class TestStoreControllerCustomPolicy:
    """Test StoreController with custom StorePolicy implementations."""

    def test_policy_that_skips_all_adapters(self, l1_manager):
        """A policy returning empty targets should not trigger L2 stores."""

        class SkipAllPolicy(StorePolicy):
            """Never store to L2."""

            def select_store_targets(self, keys, adapters):
                return {}

            def select_l1_deletions(self, keys, adapter):
                return []

        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=SkipAllPolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Give the controller time to process
        time.sleep(0.5)

        assert adapter.debug_get_stored_object_count() == 0, (
            "No objects should be stored when policy skips all adapters"
        )

        ctrl.stop()
        adapter.close()

    def test_policy_that_deletes_from_l1(self, l1_manager):
        """A policy that requests L1 deletion should result in deleted keys."""

        class DeleteAfterStorePolicy(StorePolicy):
            """Delete all keys from L1 after storing to L2."""

            def select_store_targets(self, keys, adapters):
                return {ad.index: list(keys) for ad in adapters}

            def select_l1_deletions(self, keys, adapter):
                return list(keys)

        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DeleteAfterStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store and L1 deletion
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2"

        # After deletion, reserve_write should succeed
        ok = wait_for_condition(
            lambda: (
                l1_manager.reserve_write(
                    keys=keys,
                    is_temporary=[False],
                    layout_desc=layout,
                )[keys[0]][1]
                is not None
            ),
            timeout=5.0,
        )
        assert ok, "Key should be re-creatable after L1 deletion by policy"

        ctrl.stop()
        adapter.close()

    def test_l1_deletion_decision_is_told_which_adapter_stored(self, l1_manager):
        """The policy sees the adapter that finished, so it can decide per
        adapter whether the L1 copy goes away."""
        consulted: list[int] = []

        class DeleteAfterSecondAdapterPolicy(StorePolicy):
            def select_store_targets(self, keys, adapters):
                return {ad.index: list(keys) for ad in adapters}

            def select_l1_deletions(self, keys, adapter):
                consulted.append(adapter.index)
                return list(keys) if adapter.index == 1 else []

        adapters = [make_adapter(), make_adapter()]
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=[make_descriptor(0), make_descriptor(1)],
            policy=DeleteAfterSecondAdapterPolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(lambda: sorted(consulted) == [0, 1], timeout=5.0)
        assert ok, "The policy should be consulted once per finished adapter"
        ok = wait_for_condition(
            lambda: (
                l1_manager.reserve_write(
                    keys=keys,
                    is_temporary=[False],
                    layout_desc=layout,
                )[keys[0]][1]
                is not None
            ),
            timeout=5.0,
        )
        assert ok, "The second adapter's store should delete the key from L1"

        ctrl.stop()
        for adapter in adapters:
            adapter.close()

    def test_policy_selects_subset_of_adapters(self, l1_manager):
        """A policy that targets only some adapters should skip others."""

        class FirstAdapterOnlyPolicy(StorePolicy):
            """Only store to adapter index 0."""

            def select_store_targets(self, keys, adapters):
                for ad in adapters:
                    if ad.index == 0:
                        return {0: list(keys)}
                return {}

            def select_l1_deletions(self, keys, adapter):
                return []

        adapters = [make_adapter(), make_adapter()]
        descriptors = [make_descriptor(i) for i in range(2)]

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=adapters,
            adapter_descriptors=descriptors,
            policy=FirstAdapterOnlyPolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapters[0].debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in first adapter"

        # Give some time and verify second adapter is empty
        time.sleep(0.5)
        assert adapters[1].debug_get_stored_object_count() == 0, (
            "Second adapter should have no objects"
        )

        ctrl.stop()
        for a in adapters:
            a.close()


class TestBufferOnlyMode:
    """Test buffer-only mode: BufferOnlyStorePolicy + NoOpEvictionPolicy."""

    def test_l1_cleaned_after_l2_store(self, l1_manager):
        """L1 data should be deleted after successful L2 store."""
        adapter = make_adapter()

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=BufferOnlyStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, layout)

        # Wait for L2 store
        ok = wait_for_condition(
            lambda: adapter.debug_has_key(keys[0]),
            timeout=5.0,
        )
        assert ok, "Object should be stored in L2"

        # L1 should be cleaned: check key is gone
        ok = wait_for_condition(
            lambda: l1_manager.get_object_state(keys[0]) is None,
            timeout=5.0,
        )
        assert ok, "Key should be deleted from L1 in buffer-only mode"

        ctrl.stop()
        adapter.close()

    def test_read_locked_key_deleted_after_reader_finishes(
        self, l1_manager, monkeypatch
    ):
        """A key still being read when its L2 store finishes leaves L1 once
        the reader is done."""
        delete_called = threading.Event()
        delete = l1_manager.delete

        def watched_delete(*args, **kwargs):
            result = delete(*args, **kwargs)
            delete_called.set()
            return result

        monkeypatch.setattr(l1_manager, "delete", watched_delete)
        # Slow enough that the read lock below is taken before the store ends.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=0.001)
        )

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=BufferOnlyStorePolicy([l1_manager]),
        )
        ctrl.start()

        keys = [make_object_key(0)]
        write_keys_to_l1(l1_manager, keys, make_layout())
        assert l1_manager.reserve_read(keys)[keys[0]][0] == L1Error.SUCCESS

        assert delete_called.wait(5.0), "L2 store should finish and try L1 delete"
        assert l1_manager.get_object_state(keys[0]) is not None

        l1_manager.finish_read(keys)
        ok = wait_for_condition(
            lambda: l1_manager.get_object_state(keys[0]) is None,
            timeout=5.0,
        )
        assert ok, "Key should be deleted from L1 after the reader finishes"

        ctrl.stop()
        adapter.close()

    def test_noop_eviction_never_evicts(self):
        """NoOpEvictionPolicy should never return eviction actions."""
        noop = NoOpEvictionPolicy()
        noop.on_keys_created([make_object_key(i) for i in range(10)])
        assert noop.get_eviction_actions(1.0) == []

    def test_buffer_only_multiple_keys(self, l1_manager):
        """Multiple keys should all flow to L2 and be removed
        from L1."""
        adapter = make_adapter()

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=BufferOnlyStorePolicy([l1_manager]),
        )
        ctrl.start()

        layout = make_layout()
        keys = [make_object_key(i) for i in range(5)]
        write_keys_to_l1(l1_manager, keys, layout)

        ok = wait_for_condition(
            lambda: adapter.debug_get_stored_object_count() == 5,
            timeout=5.0,
        )
        assert ok, "All 5 objects should be stored in L2"

        # All keys should be gone from L1
        ok = wait_for_condition(
            lambda: all(l1_manager.get_object_state(k) is None for k in keys),
            timeout=5.0,
        )
        assert ok, "All keys should be deleted from L1 after buffer-only cleanup"

        ctrl.stop()
        adapter.close()


class TestStoreSlicingAndPrefetchYield:
    """Tests for store slicing and yielding to prefetch loads."""

    def test_large_batch_dispatches_one_slice_at_a_time(
        self, l1_manager, monkeypatch
    ) -> None:
        """With load tracking wired, a group larger than STORE_SLICE_MAX_KEYS
        goes out one slice at a time while the rest wait unlocked."""
        object_bytes = make_layout().shapes[0].numel() * 2
        slice_bytes = STORE_SLICE_MAX_KEYS * object_bytes
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01,
                mock_bandwidth_gb=slice_bytes / 1.5 / (1 << 30),
            )
        )
        submitted_sizes: list[int] = []
        orig_submit = adapter.submit_store_task

        def record_submit(keys: list[ObjectKey], objs: list[MemoryObj]):
            submitted_sizes.append(len(keys))
            return orig_submit(keys, objs)

        monkeypatch.setattr(adapter, "submit_store_task", record_submit)
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            has_inflight_loads=lambda _adapter_ids: False,
        )
        ctrl.start()
        try:
            total = 2 * STORE_SLICE_MAX_KEYS + 2
            keys = [make_object_key(i) for i in range(total)]
            write_keys_to_l1(l1_manager, keys, make_layout())

            assert wait_for_condition(
                lambda: (
                    ctrl.report_status()["in_flight_task_count"] == 1
                    and ctrl.report_status()["queued_keys_count"]
                    == total - STORE_SLICE_MAX_KEYS
                )
            )
            assert l1_manager.is_key_evictable(keys[-1])
            assert wait_for_condition(
                lambda: all(adapter.debug_has_key(k) for k in keys),
                timeout=10.0,
            )
            assert submitted_sizes == [STORE_SLICE_MAX_KEYS, STORE_SLICE_MAX_KEYS, 2]
        finally:
            ctrl.stop()
            adapter.close()

    def test_yields_to_inflight_loads_and_resumes_on_wake(
        self, l1_manager, monkeypatch
    ) -> None:
        """Queued slices wait while loads run on the adapter, and wake() alone
        sends them out."""
        monkeypatch.setattr(store_controller, "STORE_LOOP_POLL_TIMEOUT_MS", 60_000)
        adapter = make_adapter()
        loads_active = threading.Event()
        loads_active.set()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            has_inflight_loads=lambda adapter_ids: (
                0 in set(adapter_ids) and loads_active.is_set()
            ),
        )
        ctrl.start()
        try:
            keys = [make_object_key(i) for i in range(STORE_SLICE_MAX_KEYS)]
            write_keys_to_l1(l1_manager, keys, make_layout())

            assert wait_for_condition(
                lambda: ctrl.report_status()["queued_keys_count"] == len(keys)
            )
            time.sleep(0.2)
            assert adapter.debug_get_stored_object_count() == 0

            loads_active.clear()
            ctrl.wake()
            assert wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == len(keys)
            )
        finally:
            ctrl.stop()
            adapter.close()

    @pytest.fixture
    def small_limits(self, monkeypatch) -> None:
        """Yield limit 16 keys, catch-up stores of 8 keys, and no extra
        stores in flight unless a test lowers the step."""
        monkeypatch.setattr(store_controller, "STORE_YIELD_MAX_QUEUED_KEYS", 16)
        monkeypatch.setattr(store_controller, "STORE_CATCH_UP_MAX_KEYS", 8)
        monkeypatch.setattr(store_controller, "STORE_CATCH_UP_STEP_KEYS", 1000)

    def test_backlog_past_yield_limit_goes_out_in_catch_up_stores(
        self, l1_manager, monkeypatch, small_limits
    ) -> None:
        """Past the yield limit, stores go out while loads run, one at a time,
        each at most STORE_CATCH_UP_MAX_KEYS keys of one shape, until the
        queue is back at the limit."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            has_inflight_loads=lambda _adapter_ids: True,
        )
        submits: list[tuple[int, int]] = []
        orig_submit = adapter.submit_store_task

        def record_submit(keys: list[ObjectKey], objs: list[MemoryObj]):
            submits.append((len(keys), ctrl.report_status()["in_flight_task_count"]))
            return orig_submit(keys, objs)

        monkeypatch.setattr(adapter, "submit_store_task", record_submit)
        ctrl.start()
        try:
            first = [make_object_key(i) for i in range(6)]
            write_keys_to_l1(l1_manager, first, make_layout())
            assert wait_for_condition(
                lambda: ctrl.report_status()["queued_keys_count"] == len(first)
            )
            same_shape = [make_object_key(6 + i) for i in range(6)]
            other_shape = [make_object_key(12 + i, kv_rank=1) for i in range(20)]
            write_keys_to_l1(l1_manager, same_shape + other_shape, make_layout())

            assert wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == 20
            )
            assert ctrl.report_status()["queued_keys_count"] == 12
            # 6 + 2 merged, the other 4 of that shape, then 8 of the next.
            assert submits == [(8, 0), (4, 0), (8, 0)]
        finally:
            ctrl.stop()
            adapter.close()

    def test_growing_backlog_allows_more_stores_in_flight(
        self, l1_manager, monkeypatch, small_limits
    ) -> None:
        """Every STORE_CATCH_UP_STEP_KEYS queued keys past the yield limit
        allow one more store in flight, and no more than that."""
        monkeypatch.setattr(store_controller, "STORE_CATCH_UP_STEP_KEYS", 4)
        monkeypatch.setattr(store_controller, "STORE_CATCH_UP_MAX_KEYS", 4)
        slice_bytes = 4 * make_layout().shapes[0].numel() * 2
        # About 20 s per store, so stores stay in flight.
        adapter = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=slice_bytes / 20 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy([l1_manager]),
            has_inflight_loads=lambda _adapter_ids: True,
        )
        ctrl.start()
        try:
            keys = [make_object_key(i) for i in range(28)]
            write_keys_to_l1(l1_manager, keys, make_layout())

            # 28 queued allow 4; after two stores, 20 queued allow 2.
            assert wait_for_condition(
                lambda: ctrl.report_status()["in_flight_task_count"] == 2
                and ctrl.report_status()["queued_keys_count"] == 20
            )
            time.sleep(0.3)
            assert ctrl.report_status()["in_flight_task_count"] == 2
        finally:
            ctrl.stop()
            adapter.close()

    def test_slow_adapter_does_not_hold_back_the_fast_one(self, l1_manager) -> None:
        """With two adapters, stores keep going to the fast one while the slow
        one is still writing its first slice."""
        slice_bytes = STORE_SLICE_MAX_KEYS * make_layout().shapes[0].numel() * 2
        fast = make_adapter()
        # About 20 s per slice.
        slow = MockL2Adapter(
            MockL2AdapterConfig(
                max_size_gb=0.01, mock_bandwidth_gb=slice_bytes / 20 / (1 << 30)
            )
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[fast, slow],
            adapter_descriptors=[make_descriptor(0), make_descriptor(1)],
            policy=DefaultStorePolicy([l1_manager]),
            has_inflight_loads=lambda _adapter_ids: False,
        )
        ctrl.start()
        try:
            keys = [make_object_key(i) for i in range(3 * STORE_SLICE_MAX_KEYS)]
            write_keys_to_l1(l1_manager, keys, make_layout())

            # The fast adapter finished all three slices; all three are still
            # in flight on the slow one.
            assert wait_for_condition(
                lambda: all(fast.debug_has_key(k) for k in keys)
                and ctrl.report_status()["in_flight_task_count"] == 3
            )
        finally:
            ctrl.stop()
            fast.close()
            slow.close()

    def test_storage_manager_pauses_store_until_prefetch_load_finishes(
        self, monkeypatch
    ) -> None:
        """Through StorageManager, new L1 writes wait while an L2 prefetch load
        runs, and the load finishing sends them out."""
        monkeypatch.setattr(store_controller, "STORE_LOOP_POLL_TIMEOUT_MS", 60_000)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([256, 2, 128])],
            dtypes=[torch.float32],
        )
        obj_bytes = layout.shapes[0].numel() * 4
        sm = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=64 * 1024 * 1024,
                        use_lazy=should_use_lazy_alloc(),
                        init_size_in_bytes=64 * 1024 * 1024,
                        align_bytes=0x1000,
                    )
                ),
                eviction_config=EvictionConfig("LRU"),
                l2_adapter_config=L2AdaptersConfig(
                    [
                        MockL2AdapterConfig(
                            max_size_gb=0.01,
                            mock_bandwidth_gb=4 * obj_bytes / 1.1 / (1 << 30),
                        )
                    ]
                ),
            )
        )
        try:
            seed = [make_object_key(i) for i in range(4)]
            reserved = sm.reserve_write(seed, layout)
            sm.finish_write(list(reserved), l1_owners_of(reserved))
            assert wait_for_condition(
                lambda: (
                    sm.report_status()["l2_adapters"][0]["stored_object_count"] == 4
                    and sm.report_status()["store_controller"]["in_flight_task_count"]
                    == 0
                )
            )
            sm.clear(force=True)

            handle = sm.submit_prefetch_task(
                PrefetchTaskSpec(
                    key_groups=[
                        GroupedObjectKeys(
                            keys=seed,
                            object_group_id=0,
                            layout_desc=layout,
                            sliding_window_size=FULL_ATTENTION_WINDOW_CHUNKS,
                            num_kv_readers=1,
                        )
                    ],
                    fetching_policy="prefix",
                    lock_mode=PrefetchLockMode.NO_LOCK,
                )
            )
            assert wait_for_condition(
                lambda: (
                    sm.report_status()["prefetch_controller"]["load_phase_count"] == 1
                    and sm.report_status()["l1_manager"]["staging_object_count"] == 4
                ),
                poll_interval=0.005,
            )

            fresh = [make_object_key(100 + i) for i in range(2)]
            fresh_res = sm.reserve_write(fresh, layout)
            sm.finish_write(list(fresh_res), l1_owners_of(fresh_res))
            assert wait_for_condition(
                lambda: sm.report_status()["store_controller"]["queued_keys_count"]
                == 2,
                poll_interval=0.005,
            )
            assert sm.report_status()["store_controller"]["in_flight_task_count"] == 0

            assert sm.wait_prefetch_status(handle, timeout=5.0)
            assert sm.query_prefetch_status(handle) is not None
            assert wait_for_condition(
                lambda: sm.report_status()["store_controller"]["queued_keys_count"] == 0
            )
            assert wait_for_condition(
                lambda: sm.report_status()["l2_adapters"][0]["stored_object_count"] == 6
            )
        finally:
            sm.close()
