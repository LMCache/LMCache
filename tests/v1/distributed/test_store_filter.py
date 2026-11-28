# SPDX-License-Identifier: Apache-2.0
"""Unit tests for a store that skips the chunks the cache already holds.

A store's range can include chunks a tier holds or is writing, so the
server must skip those and write the ones that no tier keeps:

- ``L1Manager.split_by_residency`` sorts keys into resident, staged by a
  live write of the same writer, and absent;
- ``StorageManager.select_keys_to_write`` drops the chunks a store need not
  copy, deciding each key across every adapter, and re-offers the
  L1-resident ones to their store controller;
- the store controller stores a re-offered key when an adapter that can
  tell lacks it, no adapter holds it, and no store of it is queued or under
  way, so a store that is in flight counts as held and a store that failed
  is repaired by the next re-offer.
"""

# Standard
from collections.abc import Iterator
from typing import Optional
import asyncio
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2StoreResult
from lmcache.v1.distributed.l1_manager import DramL1Manager as L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
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
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
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


def make_l1_config(write_ttl_seconds: int = 600) -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=32 * 1024 * 1024,
            use_lazy=should_use_lazy_alloc(),
            init_size_in_bytes=16 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=write_ttl_seconds,
        read_ttl_seconds=300,
    )


def make_l1(write_ttl_seconds: int = 600) -> L1Manager:
    return L1Manager(make_l1_config(write_ttl_seconds))


def write(l1: L1Manager, keys: list[ObjectKey], temporary: bool = False) -> None:
    results = l1.reserve_write(
        keys=keys, is_temporary=[temporary] * len(keys), layout_desc=make_layout()
    )
    assert all(obj is not None for _, obj in results.values())
    l1.finish_write(keys)


def blind_adapter() -> MockL2Adapter:
    """A mock adapter whose ``held_keys`` cannot tell."""
    return MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0))


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


def make_controller(
    l1: L1Manager,
    adapters: list[MockL2Adapter],
    eviction_config: EvictionConfig | None = None,
) -> StoreController:
    descriptors = [
        L2AdapterDescriptor(
            index=index,
            config=MockL2AdapterConfig(max_size_gb=0.05, mock_bandwidth_gb=10.0),
        )
        for index in range(len(adapters))
    ]
    controller = StoreController(
        l1, list(adapters), descriptors, DefaultStorePolicy([l1]), eviction_config
    )
    controller.start()
    return controller


def idle(controller: StoreController) -> bool:
    status = controller.report_status()
    return (
        status["pending_keys_count"] == 0
        and status["queued_keys_count"] == 0
        and status["in_flight_task_count"] == 0
    )


# -- L1 residency


def test_split_by_residency_sorts_resident_staged_and_absent():
    """Only a durable object, or a live write by the same writer, is held:
    reserve_write under that writer's tag accepts every other key. L1 drops
    a temporary object after its read and never offloads it."""
    l1 = make_l1()
    try:
        resident, staged, other_writer, temporary, absent = (
            make_key(i) for i in range(1, 6)
        )
        write(l1, [resident])
        write(l1, [temporary], temporary=True)
        l1.reserve_write(
            keys=[staged], is_temporary=[False], layout_desc=make_layout(), tag="w"
        )
        l1.reserve_write(
            keys=[other_writer],
            is_temporary=[False],
            layout_desc=make_layout(),
            tag="prefetch",
        )
        keys = [resident, staged, other_writer, temporary, absent]
        assert l1.split_by_residency(keys, tag="w") == (
            [resident],
            [staged],
            [other_writer, temporary, absent],
        )
    finally:
        l1.close()


def test_split_by_residency_counts_an_expired_reservation_as_absent():
    """reserve_write hands an expired staging object to the next writer."""
    l1 = make_l1(write_ttl_seconds=1)
    try:
        key = make_key(6)
        l1.reserve_write(
            keys=[key], is_temporary=[False], layout_desc=make_layout(), tag="w"
        )
        assert l1.split_by_residency([key], tag="w") == ([], [key], [])
        time.sleep(1.2)
        assert l1.split_by_residency([key], tag="w") == ([], [], [key])
    finally:
        l1.close()


# -- the store controller's re-offer


def test_reoffer_skips_a_key_the_adapter_holds():
    l1, adapter = make_l1(), GatedAdapter()
    controller = make_controller(l1, [adapter])
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
    controller = make_controller(l1, [adapter])
    try:
        key = make_key(11)
        write(l1, [key])
        assert wait_for(lambda: len(adapter.store_calls) == 1 and idle(controller))
        assert not adapter.debug_has_key(key)
        controller.reoffer([key])
        assert wait_for(lambda: adapter.debug_has_key(key))
        assert controller.report_status()["reoffered_keys"]["queued"] == 1
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
    controller = make_controller(l1, [adapter])
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
    controller = StoreController(l1, [adapter], [descriptor], DefaultStorePolicy([l1]))
    try:
        key = make_key(13)
        write(l1, [key])
        controller.reoffer([key])
        # One step pops the written key and the re-offer together; the
        # written key is stored once and the re-offer finds it under way.
        controller._process_l1_changes()
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


def test_reoffer_counts_a_key_waiting_for_the_budget_as_under_way():
    """A key queued behind the in-flight budget is not queued twice."""
    l1, adapter = make_l1(), GatedAdapter()
    adapter.gate.clear()
    # eviction_ratio 1 leaves no budget: one store group at a time.
    controller = make_controller(
        l1, [adapter], EvictionConfig(eviction_policy="LRU", eviction_ratio=1.0)
    )
    try:
        first, second = make_key(15), make_key(16)
        write(l1, [first])
        assert wait_for(lambda: adapter.store_calls == [[first]])
        write(l1, [second])
        assert wait_for(lambda: controller.report_status()["queued_keys_count"] == 1)
        controller.reoffer([second])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("under_way") == 1
        )
        adapter.gate.set()
        assert wait_for(lambda: adapter.debug_has_key(second) and idle(controller))
        assert adapter.store_calls == [[first], [second]]
    finally:
        adapter.gate.set()
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_keeps_the_old_behaviour_when_no_adapter_can_tell():
    l1, adapter = make_l1(), blind_adapter()
    controller = make_controller(l1, [adapter])
    try:
        key = make_key(14)
        write(l1, [key])
        assert wait_for(lambda: adapter.debug_has_key(key) and idle(controller))
        assert controller.l2_held_keys([key]) is None
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("unknown") == 1
        )
    finally:
        controller.stop()
        adapter.close()
        l1.close()


def test_reoffer_repairs_a_key_one_adapter_lacks_while_another_cannot_tell():
    """An adapter that cannot tell does not hide a hole in one that can."""
    l1, holder, blind = make_l1(), GatedAdapter(), blind_adapter()
    holder.fail_next = True
    controller = make_controller(l1, [holder, blind])
    try:
        key = make_key(17)
        write(l1, [key])
        assert wait_for(
            lambda: len(holder.store_calls) == 1
            and blind.debug_has_key(key)
            and idle(controller)
        )
        assert controller.l2_held_keys([key]) == [False]
        controller.reoffer([key])
        assert wait_for(lambda: holder.debug_has_key(key))
        assert controller.report_status()["reoffered_keys"]["queued"] == 1
    finally:
        controller.stop()
        blind.close()
        holder.close()
        l1.close()


def test_reoffer_skips_a_key_one_adapter_holds_while_another_cannot_tell():
    l1, holder, blind = make_l1(), GatedAdapter(), blind_adapter()
    controller = make_controller(l1, [holder, blind])
    try:
        key = make_key(18)
        write(l1, [key])
        assert wait_for(lambda: holder.debug_has_key(key) and idle(controller))
        assert controller.l2_held_keys([key]) == [True]
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("held") == 1
        )
        assert holder.store_calls == [[key]]
    finally:
        controller.stop()
        blind.close()
        holder.close()
        l1.close()


def test_reoffer_never_stores_a_temporary_object():
    """The write path never offers a temporary object to L2; a re-offer of
    its key does not either."""
    l1, adapter = make_l1(), GatedAdapter()
    controller = make_controller(l1, [adapter])
    try:
        key = make_key(19)
        write(l1, [key], temporary=True)
        controller.reoffer([key])
        assert wait_for(
            lambda: controller.report_status()["reoffered_keys"].get("gone_from_l1")
            == 1
        )
        assert adapter.store_calls == []
    finally:
        controller.stop()
        adapter.close()
        l1.close()


# -- the storage manager's store filter


class FilterTestAdapterConfig(L2AdapterConfigBase):
    """Config of a test adapter; ``answers`` says whether ``held_keys`` can
    tell. The factory records the adapter it builds on ``adapter``."""

    def __init__(self, answers: bool = True) -> None:
        self.answers = answers
        self.adapter: Optional[MockL2Adapter] = None

    @classmethod
    def from_dict(cls, d: dict) -> "FilterTestAdapterConfig":
        return cls(answers=bool(d.get("answers", True)))

    @classmethod
    def help(cls) -> str:
        return "Store filter test adapter: answers (bool, optional)"


def _create_filter_test_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional[L1MemoryDesc] = None,
) -> L2AdapterInterface:
    assert isinstance(config, FilterTestAdapterConfig)
    config.adapter = GatedAdapter() if config.answers else blind_adapter()
    return config.adapter


register_l2_adapter_type("store_filter_test", FilterTestAdapterConfig)
register_l2_adapter_factory("store_filter_test", _create_filter_test_adapter)


@pytest.fixture
def storage() -> Iterator[StorageManager]:
    config = make_l1_config()
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=config,
            eviction_config=EvictionConfig("noop"),
            l1_manager_configs=[config],
        )
    )
    yield manager
    manager.close()


def add_adapter(storage: StorageManager, answers: bool = True) -> MockL2Adapter:
    config = FilterTestAdapterConfig(answers)
    storage.add_l2_adapter(config)
    assert config.adapter is not None
    return config.adapter


def store(
    storage: StorageManager, keys: list[ObjectKey], temporary: bool = False
) -> None:
    reserved = storage.reserve_write(keys, make_layout(), is_temporary=temporary)
    assert list(reserved) == keys
    storage.finish_write(keys, l1_owners=l1_owners_of(reserved))


def store_to_l2_only(
    storage: StorageManager, adapters: list[MockL2Adapter], key: ObjectKey
) -> None:
    store(storage, [key])
    assert wait_for(
        lambda: all(adapter.debug_has_key(key) for adapter in adapters)
        and storage.report_status()["store_controller"]["in_flight_task_count"] == 0
    )
    assert storage.delete_l1_keys([key]) == (1, 0)


def store_filter(storage: StorageManager) -> dict[str, int]:
    return storage.report_status()["store_filter"]


def reoffered(storage: StorageManager) -> dict[str, int]:
    return storage.report_status()["store_controller"]["reoffered_keys"]


def test_select_keys_to_write_copies_only_what_no_tier_holds(storage):
    adapter = add_adapter(storage)
    resident, staged, in_l2, new = (make_key(i) for i in (20, 21, 22, 23))
    store_to_l2_only(storage, [adapter], in_l2)
    store(storage, [resident])
    storage.reserve_write([staged], make_layout())

    keys = [resident, staged, in_l2, new]
    assert storage.select_keys_to_write(keys) == [new]
    assert store_filter(storage) == {
        "requested": 4,
        "l1_resident": 1,
        "l1_staged": 1,
        "l2_held": 1,
        "to_write": 1,
    }
    assert wait_for(lambda: reoffered(storage).get("offered") == 1)


def test_select_keys_to_write_decides_each_key_across_adapters(storage):
    """An adapter that cannot tell does not hide what another holds."""
    holder = add_adapter(storage)
    blind = add_adapter(storage, answers=False)
    in_l2, new = make_key(24), make_key(25)
    store_to_l2_only(storage, [holder, blind], in_l2)

    assert storage.select_keys_to_write([in_l2, new]) == [new]
    counts = store_filter(storage)
    assert counts["l2_held"] == 1
    assert counts["to_write"] == 1


def test_select_keys_to_write_copies_l1_misses_when_no_adapter_can_tell(storage):
    add_adapter(storage, answers=False)
    resident, absent = make_key(26), make_key(27)
    store(storage, [resident])
    assert storage.select_keys_to_write([absent, resident]) == [absent]
    assert store_filter(storage)["l2_unknown"] == 1


def test_select_keys_to_write_neither_skips_nor_reoffers_a_temporary_object(
    storage,
):
    """L1 drops a temporary object after its read and never offloads it, so
    the store decides for it as for an absent chunk."""
    add_adapter(storage)
    key = make_key(28)
    store(storage, [key], temporary=True)
    assert storage.select_keys_to_write([key]) == [key]
    assert store_filter(storage) == {
        "requested": 1,
        "l1_resident": 0,
        "l1_staged": 0,
        "to_write": 1,
    }
    assert reoffered(storage) == {}
