# SPDX-License-Identifier: Apache-2.0
"""Unit tests for a store that skips the chunks the cache already holds.

A store's range can include chunks a tier holds or is writing, so the
server must skip those and write the ones that no tier keeps:

- ``L1Manager.split_by_residency`` sorts keys into resident, staged by a
  live write, and absent;
- ``StorageManager.select_keys_to_write`` drops the chunks a store need not
  copy and re-offers the L1-resident ones to their store controller;
- the store controller stores a re-offered key only when no adapter holds
  it and no store of it is under way, so a store that is in flight counts as
  held and a store that failed is repaired by the next re-offer.
"""

# Standard
from collections import Counter
from typing import Any, cast
import asyncio
import threading
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l1_manager import DramL1Manager as L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    DefaultStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj
from tests.v1.distributed.utils import should_use_lazy_alloc


def make_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([16, 2, 64])], dtypes=[torch.bfloat16])


def make_l1(write_ttl_seconds: int = 600) -> L1Manager:
    return L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=32 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=16 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=write_ttl_seconds,
            read_ttl_seconds=300,
        )
    )


def write(l1: L1Manager, keys: list[ObjectKey]) -> None:
    results = l1.reserve_write(
        keys=keys, is_temporary=[False] * len(keys), layout_desc=make_layout()
    )
    assert all(obj is not None for _, obj in results.values())
    l1.finish_write(keys)


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class GatedAdapter(MockL2Adapter):
    """A mock adapter that reports held keys and can hold or fail a store.

    ``gate`` holds every store until it is set. ``fail_next`` fails the next
    submitted store after the gate opens, without writing anything.
    """

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0))
        self.gate = threading.Event()
        self.gate.set()
        self.fail_next = False
        self.store_calls: list[list[ObjectKey]] = []
        self._writing: set[ObjectKey] = set()
        self._writing_lock = threading.Lock()

    def held_keys(self, keys: list[ObjectKey]) -> list[bool] | None:
        with self._writing_lock:
            writing = set(self._writing)
        return [key in writing or self.debug_has_key(key) for key in keys]

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()
        fail, self.fail_next = self.fail_next, False
        self.store_calls.append(list(keys))
        with self._writing_lock:
            self._writing.update(keys)
        threading.Thread(
            target=self._run, args=(keys, objects, task_id, fail), daemon=True
        ).start()
        return task_id

    def _run(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
        fail: bool,
    ) -> None:
        self.gate.wait(10)
        if fail:
            with self._lock:
                self._completed_store_tasks[task_id] = L2StoreResult(False, 0)
        else:
            asyncio.run_coroutine_threadsafe(
                self._execute_store_in_the_loop(keys, objects, task_id), self._loop
            ).result(10)
        with self._writing_lock:
            self._writing.difference_update(keys)
        self._signal_store_event()


def make_controller(l1: L1Manager, adapter: MockL2Adapter) -> StoreController:
    descriptor = L2AdapterDescriptor(
        index=0, config=MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
    )
    controller = StoreController(l1, [adapter], [descriptor], DefaultStorePolicy())
    controller.start()
    return controller


def idle(controller: StoreController) -> bool:
    status = controller.report_status()
    return status["pending_keys_count"] == 0 and status["in_flight_task_count"] == 0


# -- L1 residency


def test_split_by_residency_sorts_resident_staged_and_absent():
    l1 = make_l1()
    try:
        resident, staged, absent = make_key(1), make_key(2), make_key(3)
        write(l1, [resident])
        l1.reserve_write(keys=[staged], is_temporary=[False], layout_desc=make_layout())
        assert l1.split_by_residency([resident, staged, absent]) == (
            [resident],
            [staged],
            [absent],
        )
    finally:
        l1.close()


def test_split_by_residency_counts_an_expired_reservation_as_absent():
    """reserve_write hands an expired staging object to the next writer."""
    l1 = make_l1(write_ttl_seconds=1)
    try:
        key = make_key(4)
        l1.reserve_write(keys=[key], is_temporary=[False], layout_desc=make_layout())
        assert l1.split_by_residency([key]) == ([], [key], [])
        time.sleep(1.2)
        assert l1.split_by_residency([key]) == ([], [], [key])
    finally:
        l1.close()


# -- the store controller's re-offer


def test_reoffer_skips_a_key_the_adapter_holds():
    l1, adapter = make_l1(), GatedAdapter()
    controller = make_controller(l1, adapter)
    try:
        key = make_key(10)
        write(l1, [key])
        assert wait_for(lambda: adapter.debug_has_key(key) and idle(controller))
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("held") == 1
        )
        assert len(adapter.store_calls) == 1
    finally:
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_repairs_a_failed_store_from_l1():
    l1, adapter = make_l1(), GatedAdapter()
    adapter.fail_next = True
    controller = make_controller(l1, adapter)
    try:
        key = make_key(11)
        write(l1, [key])
        assert wait_for(lambda: len(adapter.store_calls) == 1 and idle(controller))
        assert not adapter.debug_has_key(key)
        controller.reoffer([key])
        assert wait_for(lambda: adapter.debug_has_key(key))
        assert controller.report_status()["reoffered_keys"]["submitted"] == 1
        assert len(adapter.store_calls) == 2
    finally:
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_counts_a_store_in_flight_as_held_and_repairs_it_later():
    """A store under way is not repeated; if it fails, the next re-offer
    stores the key."""
    l1, adapter = make_l1(), GatedAdapter()
    adapter.gate.clear()
    adapter.fail_next = True
    controller = make_controller(l1, adapter)
    try:
        key = make_key(12)
        write(l1, [key])
        assert wait_for(lambda: len(adapter.store_calls) == 1)
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("under_way") == 1
        )
        assert len(adapter.store_calls) == 1
        adapter.gate.set()
        assert wait_for(lambda: idle(controller))
        assert not adapter.debug_has_key(key)
        controller.reoffer([key])
        assert wait_for(lambda: adapter.debug_has_key(key))
        assert len(adapter.store_calls) == 2
    finally:
        adapter.gate.set()
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_of_a_key_still_pending_submits_it_once():
    l1, adapter = make_l1(), GatedAdapter()
    descriptor = L2AdapterDescriptor(
        index=0, config=MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)
    )
    # Not started: the test drives the loop's step by hand.
    controller = StoreController(l1, [adapter], [descriptor], DefaultStorePolicy())
    try:
        key = make_key(13)
        write(l1, [key])
        controller.reoffer([key])
        reoffered, pending = controller._listener.pop_queued_keys()
        assert reoffered == [key] and pending == [key]
        controller._process_reoffered_keys(reoffered, pending)
        assert adapter.store_calls == []
        controller._process_new_keys(pending)
        assert adapter.store_calls == [[key]]
        assert controller.report_status()["reoffered_keys"]["under_way"] == 1
    finally:
        adapter.gate.set()
        # Let the loop settle the submitted task before stopping.
        controller.start()
        assert wait_for(lambda: idle(controller))
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_keeps_the_old_behaviour_for_an_adapter_that_cannot_tell():
    l1, adapter = (
        make_l1(),
        MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0)),
    )
    controller = make_controller(l1, adapter)
    try:
        key = make_key(14)
        write(l1, [key])
        assert wait_for(lambda: adapter.debug_has_key(key) and idle(controller))
        assert controller.drive_holds([key]) is None
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("unknown") == 1
        )
    finally:
        controller.stop()
        adapter.close()
        l1.close()


# -- the storage manager's store filter


class _Controller:
    def __init__(self, held: dict[ObjectKey, bool] | None) -> None:
        self.held = held
        self.reoffered: list[ObjectKey] = []

    def reoffer(self, keys: list[ObjectKey]) -> None:
        self.reoffered.extend(keys)

    def drive_holds(self, keys: list[ObjectKey]) -> list[bool] | None:
        if self.held is None:
            return None
        return [self.held.get(key, False) for key in keys]


def bare_storage_manager(l1: L1Manager, controller: _Controller) -> StorageManager:
    manager = StorageManager.__new__(StorageManager)
    manager._l1_managers_by_id = {l1.l1_manager_id: l1}
    manager._store_controllers = cast(Any, {l1.l1_manager_id: controller})
    manager._store_filter_lock = threading.Lock()
    manager._store_filter_counts = Counter()
    return manager


def test_select_keys_to_write_copies_only_what_no_tier_holds():
    l1 = make_l1()
    try:
        resident, staged, on_drive, new = (make_key(i) for i in (20, 21, 22, 23))
        write(l1, [resident])
        l1.reserve_write(keys=[staged], is_temporary=[False], layout_desc=make_layout())
        controller = _Controller({on_drive: True})
        manager = bare_storage_manager(l1, controller)

        keys = [resident, staged, on_drive, new]
        assert manager.select_keys_to_write(keys) == [new]
        assert controller.reoffered == [resident]
        assert manager._store_filter_snapshot() == {
            "requested": 4,
            "l1_resident": 1,
            "l1_staged": 1,
            "l2_held": 1,
            "to_write": 1,
        }
    finally:
        l1.close()


def test_select_keys_to_write_copies_l1_misses_when_an_adapter_cannot_tell():
    l1 = make_l1()
    try:
        resident, absent = make_key(24), make_key(25)
        write(l1, [resident])
        manager = bare_storage_manager(l1, _Controller(None))
        assert manager.select_keys_to_write([absent, resident]) == [absent]
        assert manager._store_filter_snapshot()["l2_unknown"] == 1
    finally:
        l1.close()
