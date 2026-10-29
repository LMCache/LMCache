# SPDX-License-Identifier: Apache-2.0
"""Prefetch ownership and publication regression tests."""

# Standard
from collections.abc import Iterator
from threading import Event, Thread
import asyncio

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import FetchingPolicy, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.test_prefetch_controller import (
    assert_absent,
    assert_l2_unlocked,
    assert_read_locked,
    make_adapter,
    make_controller,
    make_group,
    make_l1_config,
    make_layout,
    make_object_key,
    make_spec,
    row_bits,
    single_row_spec,
    store_keys_in_l2,
    wait_for_condition,
    wait_for_result,
)


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    """Provide a real L1 manager and release its pool after the test."""
    manager = L1Manager(make_l1_config())
    try:
        yield manager
    finally:
        manager.close()


def start_shutdown(
    controller: PrefetchController,
) -> tuple[Thread, Event, list[Exception]]:
    """Start shutdown without blocking the test's adapter-completion gate.

    Args:
        controller: The live prefetch controller to stop.

    Returns:
        The shutdown thread, its completion signal, and any raised exceptions.
    """
    stopped = Event()
    errors: list[Exception] = []

    def stop() -> None:
        try:
            controller.stop()
        except Exception as exc:
            errors.append(exc)
        finally:
            stopped.set()

    thread = Thread(target=stop, daemon=True)
    thread.start()
    return thread, stopped, errors


class GatedLoadAdapter(MockL2Adapter):
    """Hold a real asynchronous load before its first destination write."""

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10))
        self.load_started = Event()
        self.allow_load = Event()
        self.pending_objects: list[MemoryObj] = []

    async def _execute_load_in_loop(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        self.pending_objects = objects
        self.load_started.set()
        await asyncio.to_thread(self.allow_load.wait)
        await super()._execute_load_in_loop(keys, objects, task_id)


class GatedLookupAdapter(MockL2Adapter):
    """Hold a lookup before it acquires its L2 locks."""

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10))
        self.lookup_started = Event()
        self.allow_lookup = Event()

    def _execute_lookup_in_the_loop(
        self, keys: list[ObjectKey], task_id: L2TaskId
    ) -> None:
        self.lookup_started.set()
        self.allow_lookup.wait()
        super()._execute_lookup_in_the_loop(keys, task_id)


class FailingLoadAdapter(MockL2Adapter):
    """Reject a load before starting any I/O against the provided objects."""

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Inject a failure before the adapter borrows any destination buffer."""
        raise RuntimeError("injected load submission failure")


class InvalidatingAdmissionL1Manager(L1Manager):
    """Interleave a force clear after the load, before admitting its objects."""

    def __init__(self, config: L1ManagerConfig, invalidated_key: ObjectKey) -> None:
        super().__init__(config)
        self.invalidated_key = invalidated_key

    def finish_write_and_reserve_read(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        tag: str = "",
        read_generations: dict[ObjectKey, int] | None = None,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Invalidate one staging object before attempting the real admission."""
        if self.invalidated_key in keys:
            self.delete([self.invalidated_key], force=True)
        return super().finish_write_and_reserve_read(
            keys, read_locks, tag, read_generations=read_generations
        )


def test_shutdown_keeps_load_buffers_until_adapter_completion(
    l1_manager: L1Manager,
) -> None:
    """Shutdown must not invalidate buffers still borrowed by an L2 load."""
    adapter = GatedLoadAdapter()
    keys = [make_object_key(i) for i in range(3)]
    store_keys_in_l2(adapter, keys, make_layout())
    controller = make_controller(l1_manager, [adapter])
    controller.start()
    request = controller.submit_prefetch_request(single_row_spec(keys))
    assert adapter.load_started.wait(5)

    thread, stopped, errors = start_shutdown(controller)
    try:
        assert not stopped.wait(0.2), "shutdown returned while L2 still owns buffers"
        assert all(obj.is_valid() for obj in adapter.pending_objects)
        assert l1_manager.get_staging_memory_usage() > 0
    finally:
        adapter.allow_load.set()
        thread.join(5)
        adapter.close()

    assert stopped.is_set(), "shutdown did not drain the completed load"
    assert not errors
    assert l1_manager.get_staging_memory_usage() == 0
    assert controller.wait_prefetch_result(request, timeout=0)
    assert row_bits(controller.query_prefetch_result(request)) == []


def test_shutdown_returns_locks_acquired_by_a_pending_lookup(
    l1_manager: L1Manager,
) -> None:
    """Late lookup completion must not strand its L2 locks after shutdown."""
    adapter = GatedLookupAdapter()
    keys = [make_object_key(i) for i in range(3)]
    store_keys_in_l2(adapter, keys, make_layout())
    controller = make_controller(l1_manager, [adapter])
    controller.start()
    request = controller.submit_prefetch_request(single_row_spec(keys))
    assert adapter.lookup_started.wait(5)

    thread, stopped, errors = start_shutdown(controller)
    try:
        assert not stopped.wait(0.2), "shutdown abandoned an unfinished lookup"
    finally:
        adapter.allow_lookup.set()
        thread.join(5)

    try:
        assert stopped.is_set()
        assert not errors
        assert_l2_unlocked(adapter)
        assert controller.wait_prefetch_result(request, timeout=0)
        assert row_bits(controller.query_prefetch_result(request)) == []
    finally:
        adapter.close()


def test_failed_submission_drains_other_adapters_before_freeing_buffers(
    l1_manager: L1Manager,
) -> None:
    """One adapter's failure must not free another adapter's active buffers."""
    slow = GatedLoadAdapter()
    failed = FailingLoadAdapter(
        MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10)
    )
    keys = [make_object_key(i) for i in range(2)]
    store_keys_in_l2(slow, keys[:1], make_layout())
    store_keys_in_l2(failed, keys[1:], make_layout())
    controller = make_controller(l1_manager, [slow, failed])
    controller.start()
    try:
        request = controller.submit_prefetch_request(single_row_spec(keys))
        assert slow.load_started.wait(5)
        assert controller.wait_prefetch_result(request, timeout=5)
        assert row_bits(controller.query_prefetch_result(request)) == []
        assert all(obj.is_valid() for obj in slow.pending_objects)

        slow.allow_load.set()
        assert wait_for_condition(
            lambda: controller.report_status()["in_flight_request_count"] == 0
        )
        assert_l2_unlocked(slow)
        assert_l2_unlocked(failed)
        assert l1_manager.get_staging_memory_usage() == 0
        assert controller.query_prefetch_result(request) is None
    finally:
        slow.allow_load.set()
        controller.stop()
        slow.close()
        failed.close()


@pytest.mark.parametrize(
    "fetching_policy,expected", [("prefix", [0]), ("full", [0, 2])]
)
def test_prefetch_reports_only_successfully_admitted_columns(
    fetching_policy: FetchingPolicy, expected: list[int]
) -> None:
    """L2 success does not imply the staging object was admitted and locked."""
    rows = [
        make_group([make_object_key(i, gid=gid) for i in range(3)], gid=gid)
        for gid in range(2)
    ]
    manager = InvalidatingAdmissionL1Manager(make_l1_config(), rows[1].keys[1])
    adapter = make_adapter()
    all_keys = rows[0].keys + rows[1].keys
    store_keys_in_l2(adapter, all_keys, make_layout())
    controller = make_controller(manager, [adapter])
    controller.start()
    try:
        request = controller.submit_prefetch_request(
            make_spec(rows, fetching_policy=fetching_policy)
        )
        result = wait_for_result(controller, request)
        assert [row_bits(result, i) for i in range(2)] == [expected, expected]
        held = [row.keys[i] for row in rows for i in expected]
        assert_read_locked(manager, held)
        assert_absent(manager, [key for key in all_keys if key not in held])
        assert manager.get_staging_memory_usage() == 0
        assert_l2_unlocked(adapter)
        manager.finish_read(held)
    finally:
        controller.stop()
        adapter.close()
        manager.close()


def test_discard_reclaims_expired_staging_without_removing_other_writers() -> None:
    """A completed writer can discard its expired reservation by its own tag."""
    config = make_l1_config()
    config.write_ttl_seconds = 1
    manager = L1Manager(config)
    key = make_object_key(1)
    layout = make_layout()
    try:
        first = manager.reserve_write([key], [False], layout, tag="expired")[key]
        assert first[0] == L1Error.SUCCESS
        assert wait_for_condition(lambda: manager.is_key_evictable(key), timeout=3)
        second = manager.reserve_write([key], [False], layout, tag="live")[key]
        assert second[0] == L1Error.SUCCESS

        assert manager.finish_write_and_delete([key], tag="expired")[key] == (
            L1Error.SUCCESS
        )
        assert first[1] is not None and not first[1].is_valid()
        assert second[1] is not None and second[1].is_valid()
        assert manager.finish_write([key], tag="live")[key] == L1Error.SUCCESS
        assert manager.get_staging_memory_usage() == 0
    finally:
        manager.close()


def test_active_load_remains_protected_after_write_ttl_expires() -> None:
    """An active adapter's borrowed buffer must outlive its write TTL."""
    config = make_l1_config()
    config.write_ttl_seconds = 1
    manager = L1Manager(config)
    adapter = GatedLoadAdapter()
    key = make_object_key(0)
    store_keys_in_l2(adapter, [key], make_layout())
    controller = make_controller(manager, [adapter])
    controller.start()
    try:
        request = controller.submit_prefetch_request(single_row_spec([key]))
        assert adapter.load_started.wait(5)
        assert not adapter.allow_load.wait(1.2)
        assert not manager.is_key_evictable(key), "TTL expired under active L2 I/O"
        assert manager.delete([key])[key] == L1Error.KEY_IS_LOCKED
        manager.clear()
        assert manager.get_staging_memory_usage() > 0
        assert all(obj.is_valid() for obj in adapter.pending_objects)

        adapter.allow_load.set()
        result = wait_for_result(controller, request)
        assert row_bits(result) == []
        assert manager.get_staging_memory_usage() == 0
        assert_l2_unlocked(adapter)
    finally:
        adapter.allow_load.set()
        controller.stop()
        adapter.close()
        manager.close()


def test_shutdown_settles_queued_requests_and_rejects_new_submissions(
    l1_manager: L1Manager,
) -> None:
    """Queued requests receive misses while the active load is still draining."""
    adapter = GatedLoadAdapter()
    keys = [make_object_key(i) for i in range(2)]
    store_keys_in_l2(adapter, keys, make_layout())
    controller = make_controller(l1_manager, [adapter], inflight_watermark=0.001)
    controller.start()
    active = controller.submit_prefetch_request(single_row_spec(keys))
    assert adapter.load_started.wait(5)
    queued = controller.submit_prefetch_request(single_row_spec(keys))
    thread, stopped, errors = start_shutdown(controller)
    try:
        for request in (active, queued):
            assert controller.wait_prefetch_result(request, timeout=5)
            assert row_bits(controller.query_prefetch_result(request)) == []
        assert not stopped.is_set()
        with pytest.raises(RuntimeError, match="shutdown"):
            controller.submit_prefetch_request(single_row_spec(keys))
        assert all(obj.is_valid() for obj in adapter.pending_objects)
    finally:
        adapter.allow_load.set()
        thread.join(5)
        adapter.close()
    assert stopped.is_set()
    assert not errors
    assert controller.query_prefetch_result(active) is None
    assert controller.query_prefetch_result(queued) is None


@pytest.mark.parametrize("pin_count", [1, 3])
def test_all_io_borrows_must_finish_before_admission(
    l1_manager: L1Manager, pin_count: int
) -> None:
    """A writer cannot publish or discard a buffer still borrowed by I/O."""
    key = make_object_key(0)
    layout = make_layout()
    assert l1_manager.reserve_write([key], [False], layout, tag="writer")[key][0] == (
        L1Error.SUCCESS
    )
    for _ in range(pin_count):
        assert l1_manager.pin_write_buffers([key], tag="writer")[key] == L1Error.SUCCESS
    for _ in range(pin_count):
        assert (
            l1_manager.finish_write([key], tag="writer")[key] == L1Error.KEY_IS_LOCKED
        )
        assert l1_manager.finish_write_and_delete([key], tag="writer")[key] == (
            L1Error.KEY_IS_LOCKED
        )
        assert (
            l1_manager.unpin_write_buffers([key], tag="writer")[key] == L1Error.SUCCESS
        )
    assert l1_manager.unpin_write_buffers([key], tag="writer")[key] == (
        L1Error.KEY_IN_WRONG_STATE
    )
    assert l1_manager.finish_write([key], tag="writer")[key] == L1Error.SUCCESS
    assert l1_manager.get_staging_memory_usage() == 0


def test_writer_pins_do_not_release_another_tag(l1_manager: L1Manager) -> None:
    """A stale writer's unpin cannot release another writer's staging lease."""
    key = make_object_key(0)
    layout = make_layout()
    for tag in ("old", "new"):
        assert l1_manager.reserve_write([key], [False], layout, tag=tag)[key][0] == (
            L1Error.SUCCESS
        )
        assert l1_manager.pin_write_buffers([key], tag=tag)[key] == L1Error.SUCCESS
    assert l1_manager.unpin_write_buffers([key], tag="old")[key] == L1Error.SUCCESS
    assert l1_manager.finish_write_and_delete([key], tag="old")[key] == L1Error.SUCCESS
    assert (
        l1_manager.unpin_write_buffers([key], tag="old")[key] == L1Error.KEY_NOT_EXIST
    )
    assert l1_manager.finish_write_and_delete([key], tag="new")[key] == (
        L1Error.KEY_IS_LOCKED
    )
    assert l1_manager.unpin_write_buffers([key], tag="new")[key] == L1Error.SUCCESS
    assert l1_manager.finish_write([key], tag="new")[key] == L1Error.SUCCESS
    assert l1_manager.get_staging_memory_usage() == 0
