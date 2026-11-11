# SPDX-License-Identifier: Apache-2.0
"""
Tests for runtime add/remove of L2 adapters.

Covers both the controller-level primitives (StoreController /
PrefetchController add_adapter / request_remove_adapter with graceful
drain) and the StorageManager orchestration (add_l2_adapter /
delete_l2_adapter), including stable-id semantics and the
active/draining observability surface.
"""

# Standard
from dataclasses import replace
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.common.object_key import ObjectKey
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    ShadowL1ManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import (
    DefaultStorePolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import L2AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from tests.v1.distributed.borrowing_mock import (
    BorrowingMockL2Adapter,
    BorrowingMockL2AdapterConfig,
)
from tests.v1.distributed.utils import should_use_lazy_alloc, single_row_spec

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([100, 2, 512])],
        dtypes=[torch.bfloat16],
    )


def wait_for_condition(predicate, timeout: float = 5.0, poll: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def make_mock_config() -> MockL2AdapterConfig:
    return MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)


def make_adapter() -> MockL2Adapter:
    return MockL2Adapter(make_mock_config())


def make_descriptor(index: int) -> L2AdapterDescriptor:
    return L2AdapterDescriptor(index=index, config=make_mock_config())


def adapter_by_id(sm: StorageManager, adapter_id: int) -> MockL2Adapter:
    """Fetch an active adapter by stable id via the public ``l2_adapters()``
    API (no private-member access)."""
    for desc, adapter in sm.l2_adapters():
        if desc.index == adapter_id:
            return adapter  # type: ignore[return-value]
    raise AssertionError(f"no active L2 adapter with id {adapter_id}")


def active_adapter_ids(sm: StorageManager) -> set[int]:
    """Set of active adapter ids exposed by the public ``l2_adapters()`` API."""
    return {desc.index for desc, _ in sm.l2_adapters()}


def write_keys_to_l1(
    l1_manager: DramL1Manager,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
) -> None:
    results = l1_manager.reserve_write(
        keys=keys,
        is_temporary=[False] * len(keys),
        layout_desc=layout,
    )
    written = [k for k, (e, m) in results.items() if m is not None]
    if written:
        l1_manager.finish_write(written)


def lend_keys(
    sm: StorageManager, adapter_id: int, keys: list[ObjectKey]
) -> tuple[BorrowingMockL2Adapter, dict[ObjectKey, int]]:
    """Store ``keys`` at a borrowing mock's fake owner and borrow them through a
    LOCK prefetch; return the borrower and the read owners the caller must
    release with ``finish_read_prefetched``."""
    layout = make_layout()
    adapter = adapter_by_id(sm, adapter_id)
    assert isinstance(adapter, BorrowingMockL2Adapter)
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
        objs.append(TensorMemoryObj(tensor, meta, parent_allocator=None))
    adapter.submit_store_task(keys, objs)
    assert wait_for_condition(lambda: all(adapter.debug_has_key(k) for k in keys))
    handle = sm.submit_prefetch_task(single_row_spec(keys, layout))
    assert sm.wait_prefetch_status(handle, timeout=5.0)
    result = sm.query_prefetch_status(handle)
    assert result is not None
    assert result.hit_cells[0].count_leading_ones() == len(keys)
    assert adapter.get_local_borrow_count() == len(keys)
    assert result.l1_owners is not None
    return adapter, result.l1_owners


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
    mgr = DramL1Manager(config)
    yield mgr
    mgr.close()


@pytest.fixture
def empty_storage_manager_config():
    """A StorageManagerConfig with no L2 adapters configured."""
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
        l2_adapter_config=L2AdaptersConfig(adapters=[]),
    )


def with_shadow(
    config: StorageManagerConfig, read_ttl_seconds: int = 300
) -> StorageManagerConfig:
    """``config`` plus a DRAM shadow L1 after its owned L1."""
    shadow = ShadowL1ManagerConfig(tag="shadow", read_ttl_seconds=read_ttl_seconds)
    return replace(config, l1_manager_configs=[config.l1_manager_config, shadow])


@pytest.fixture
def shadowed_config(empty_storage_manager_config):
    """A StorageManagerConfig with an owned DRAM L1 and a DRAM shadow L1."""
    return with_shadow(empty_storage_manager_config)


# =============================================================================
# StoreController-level tests
# =============================================================================


class TestStoreControllerRuntimeAdapters:
    def test_add_adapter_routes_new_stores(self, l1_manager):
        """A key written after add_adapter is stored to the new adapter."""
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            adapter = make_adapter()
            ctrl.add_adapter(0, adapter, make_descriptor(0))

            assert ctrl.report_status()["num_active_adapters"] == 1
            assert ctrl.report_status()["num_draining_adapters"] == 0

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)

            ok = wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == 3
            )
            assert ok, "Keys should be stored to the runtime-added adapter"
        finally:
            ctrl.stop()
            adapter.close()

    def test_remove_adapter_drains_and_detaches(self, l1_manager):
        """request_remove_adapter drains in-flight work then detaches."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)
            assert wait_for_condition(
                lambda: adapter.debug_get_stored_object_count() == 3
            )

            done = ctrl.request_remove_adapter(0)
            assert done.wait(timeout=5.0), "Adapter should drain and detach"

            status = ctrl.report_status()
            assert status["num_l2_adapters"] == 0
            assert status["num_active_adapters"] == 0
            assert status["num_draining_adapters"] == 0
        finally:
            ctrl.stop()
            adapter.close()

    def test_remove_stops_new_routing(self, l1_manager):
        """After removal, new writes are not routed to the removed adapter."""
        keep = make_adapter()
        drop = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[keep, drop],
            adapter_descriptors=[make_descriptor(0), make_descriptor(1)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            done = ctrl.request_remove_adapter(1)
            assert done.wait(timeout=5.0)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            write_keys_to_l1(l1_manager, keys, layout)

            assert wait_for_condition(lambda: keep.debug_get_stored_object_count() == 3)
            # The dropped adapter must not receive the new keys.
            time.sleep(0.2)
            assert drop.debug_get_stored_object_count() == 0
        finally:
            ctrl.stop()
            keep.close()
            drop.close()

    def test_double_remove_is_safe(self, l1_manager):
        """Removing an already-removed adapter signals immediately."""
        adapter = make_adapter()
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[make_descriptor(0)],
            policy=DefaultStorePolicy(),
        )
        ctrl.start()
        try:
            assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
            # Second removal of the same (now-detached) id must not hang.
            assert ctrl.request_remove_adapter(0).wait(timeout=5.0)
        finally:
            ctrl.stop()
            adapter.close()


# =============================================================================
# StorageManager-level tests
# =============================================================================


class TestStorageManagerRuntimeAdapters:
    def test_add_to_empty_manager(self, empty_storage_manager_config):
        """add_l2_adapter wires a new adapter into all controllers."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            assert sm.report_status()["num_l2_adapters"] == 0

            adapter_id = sm.add_l2_adapter(make_mock_config())
            assert adapter_id == 0

            status = sm.report_status()
            assert status["num_l2_adapters"] == 1
            assert status["store_controller"]["num_active_adapters"] == 1
            assert status["prefetch_controller"]["num_active_adapters"] == 1
        finally:
            sm.close()

    def test_add_then_store(self, empty_storage_manager_config):
        """Keys written after add_l2_adapter land in the new adapter."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            adapter_id = sm.add_l2_adapter(make_mock_config())
            adapter = adapter_by_id(sm, adapter_id)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(4)]
            ret = sm.reserve_write(keys, layout)
            sm.finish_write(list(ret), l1_owners_of(ret))

            assert wait_for_condition(
                lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
                timeout=10.0,
            )
        finally:
            sm.close()

    def test_delete_detaches_everywhere(self, empty_storage_manager_config):
        """delete_l2_adapter drains and removes the adapter from all controllers."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            adapter_id = sm.add_l2_adapter(make_mock_config())
            assert sm.report_status()["num_l2_adapters"] == 1

            sm.delete_l2_adapter(adapter_id, timeout=10.0)

            status = sm.report_status()
            assert status["num_l2_adapters"] == 0
            assert status["store_controller"]["num_l2_adapters"] == 0
            assert status["prefetch_controller"]["num_l2_adapters"] == 0
        finally:
            sm.close()

    def test_delete_keeps_adapter_when_close_fails(
        self, empty_storage_manager_config, monkeypatch
    ):
        """A failed close leaves the adapter registered so delete can retry."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            adapter_id = sm.add_l2_adapter(make_mock_config())
            adapter = adapter_by_id(sm, adapter_id)
            original_close = adapter.close
            failures = [RuntimeError("teardown failed")]

            def flaky_close() -> None:
                if failures:
                    raise failures.pop()
                original_close()

            monkeypatch.setattr(adapter, "close", flaky_close)
            touched: list[list[ObjectKey]] = []
            monkeypatch.setattr(adapter, "submit_touch", touched.append)

            with pytest.raises(RuntimeError, match="teardown failed"):
                sm.delete_l2_adapter(adapter_id, timeout=10.0)
            # Detached but not closed: no longer active anywhere a caller can
            # observe, yet still registered, so the retry below finds it.
            assert sm.l2_adapters() == []
            assert sm.report_status()["num_l2_adapters"] == 0
            sm.touch_cached_keys([make_object_key(1)])
            assert touched == []

            sm.delete_l2_adapter(adapter_id, timeout=10.0)
            assert active_adapter_ids(sm) == set()
        finally:
            sm.close()

    def test_close_continues_past_a_failing_adapter(
        self, empty_storage_manager_config, monkeypatch
    ):
        """One adapter that fails to close must not keep the others from
        closing, and L1 stays open because that adapter may still use it."""
        sm = StorageManager(empty_storage_manager_config)
        first = adapter_by_id(sm, sm.add_l2_adapter(make_mock_config()))
        second = adapter_by_id(sm, sm.add_l2_adapter(make_mock_config()))
        closed: list[str] = []
        monkeypatch.setattr(
            first, "close", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        original_second_close = second.close
        monkeypatch.setattr(
            second, "close", lambda: (closed.append("second"), original_second_close())
        )
        (l1_manager,) = sm.l1_managers()
        original_l1_close = l1_manager.close
        monkeypatch.setattr(
            l1_manager, "close", lambda: (closed.append("l1"), original_l1_close())
        )

        sm.close()

        assert closed == ["second"]
        original_l1_close()

    def test_delete_unknown_id_raises(self, empty_storage_manager_config):
        sm = StorageManager(empty_storage_manager_config)
        try:
            with pytest.raises(ValueError):
                sm.delete_l2_adapter(999)
        finally:
            sm.close()

    def test_stable_ids_across_add_delete(self, empty_storage_manager_config):
        """Ids are monotonic and never reused; delete does not shift others."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            id0 = sm.add_l2_adapter(make_mock_config())
            id1 = sm.add_l2_adapter(make_mock_config())
            assert (id0, id1) == (0, 1)

            sm.delete_l2_adapter(id0, timeout=10.0)

            id2 = sm.add_l2_adapter(make_mock_config())
            assert id2 == 2, "Ids must not be reused after a delete"

            # The surviving adapter keeps its original id.
            assert active_adapter_ids(sm) == {1, 2}
            assert sm.report_status()["num_l2_adapters"] == 2
        finally:
            sm.close()

    def test_add_after_delete_routes_to_new_adapter(self, empty_storage_manager_config):
        """A fresh adapter added after a delete receives new stores."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            first = sm.add_l2_adapter(make_mock_config())
            sm.delete_l2_adapter(first, timeout=10.0)
            second = sm.add_l2_adapter(make_mock_config())
            adapter = adapter_by_id(sm, second)

            layout = make_layout()
            keys = [make_object_key(i) for i in range(3)]
            ret = sm.reserve_write(keys, layout)
            sm.finish_write(list(ret), l1_owners_of(ret))

            assert wait_for_condition(
                lambda: all(adapter.debug_has_key(k) for k in keys),  # type: ignore
                timeout=10.0,
            )
        finally:
            sm.close()

    def test_l2_adapters_reflects_runtime_changes(self, empty_storage_manager_config):
        """l2_adapters() tracks runtime add/delete in stable-id order."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            assert sm.l2_adapters() == []

            id0 = sm.add_l2_adapter(make_mock_config())
            id1 = sm.add_l2_adapter(make_mock_config())
            pairs = sm.l2_adapters()
            assert [d.index for d, _ in pairs] == [id0, id1]
            a0, a1 = pairs[0][1], pairs[1][1]
            assert a0 is not a1

            # Deleting one leaves the survivor (same instance); order is by id.
            sm.delete_l2_adapter(id0, timeout=10.0)
            pairs = sm.l2_adapters()
            assert len(pairs) == 1
            assert pairs[0][0].index == id1
            assert pairs[0][1] is a1
        finally:
            sm.close()

    def test_a_configured_shadow_l1_outlives_its_borrowers(self, shadowed_config):
        """A shadow L1 is configured like any L1 and owns no memory: it is
        listed with inert store and eviction controllers, and borrowers come
        and go without changing the L1 set."""
        sm = StorageManager(shadowed_config)
        try:
            status = sm.report_status()
            assert set(status["l1_managers"]) == {"_default", "shadow"}
            assert set(status["store_controllers"]) == {"_default", "shadow"}
            assert set(status["l1_eviction_controllers"]) == {"_default", "shadow"}
            assert status["is_healthy"]
            assert (
                sm.get_l1_manager_by_tags(
                    MemoryTags(MemorySegment.DRAM, MemoryResidency.SHADOW)
                ).config.tag
                == "shadow"
            )
            adapter_id = sm.add_l2_adapter(BorrowingMockL2AdapterConfig())
            assert set(sm.report_status()["l1_managers"]) == {"_default", "shadow"}
            sm.delete_l2_adapter(adapter_id)
            status = sm.report_status()
            assert set(status["l1_managers"]) == {"_default", "shadow"}
            assert status["num_l2_adapters"] == 0
        finally:
            sm.close()

    def test_a_borrower_without_a_shadow_l1_is_refused(
        self, empty_storage_manager_config
    ):
        """With no shadow L1 of its segment configured, a borrower is refused
        before anything is registered."""
        sm = StorageManager(empty_storage_manager_config)
        try:
            with pytest.raises(ValueError, match="No L1 has"):
                sm.add_l2_adapter(BorrowingMockL2AdapterConfig())
            assert sm.report_status()["num_l2_adapters"] == 0
        finally:
            sm.close()

    def test_borrower_removal_is_retried_once_its_views_are_released(
        self, shadowed_config
    ):
        """Readers hold borrowed views past their prefetch; a borrower refuses
        to close while they do, and the storage manager keeps it, detached,
        for a retry that succeeds once they finish and returns the owner's
        locks."""
        sm = StorageManager(shadowed_config)
        try:
            adapter_id = sm.add_l2_adapter(BorrowingMockL2AdapterConfig())
            keys = [make_object_key(i) for i in range(2)]
            adapter, owners = lend_keys(sm, adapter_id, keys)
            with pytest.raises(RuntimeError, match="live borrowed"):
                sm.delete_l2_adapter(adapter_id, timeout=5.0)
            assert adapter_id not in active_adapter_ids(sm)
            assert adapter.get_local_borrow_count() == 2
            sm.finish_read_prefetched(keys, l1_owners=owners)
            sm.delete_l2_adapter(adapter_id, timeout=5.0)
            assert adapter.get_local_borrow_count() == 0
            assert wait_for_condition(lambda: adapter.debug_get_locked_key_count() == 0)
            assert sm.report_status()["num_l2_adapters"] == 0
        finally:
            sm.close()
