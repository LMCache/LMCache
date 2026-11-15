# SPDX-License-Identifier: Apache-2.0
"""Tests for how the store and prefetch controllers treat an unavailable adapter.

``adapter_available`` is the controllers' only view of the health monitor: a
callable that says whether an adapter's backend is up. These tests flip it by
hand and check the controllers' observable behavior.
"""

# Standard
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    BufferOnlyPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import (
    Bitmap2D,
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def _l1_config() -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=False,
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )


def _descriptor(index: int) -> L2AdapterDescriptor:
    return L2AdapterDescriptor(
        index=index,
        config=MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
    )


def _spec(keys: list[ObjectKey]) -> PrefetchTaskSpec:
    group = GroupedObjectKeys(
        keys=keys,
        object_group_id=0,
        layout_desc=_layout(),
        sliding_window_size=FULL_ATTENTION_WINDOW_CHUNKS,
        num_kv_readers=1,
    )
    return PrefetchTaskSpec(
        key_groups=[group],
        fetching_policy="prefix",
        lock_mode=PrefetchLockMode.LOCK,
        allowed_l1_memory_tags=None,
        allowed_l2_memory_tags=None,
    )


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _wait_for_result(
    controller: PrefetchController, request_id: int, timeout: float = 5.0
) -> PrefetchResult | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = controller.query_prefetch_result(request_id)
        if result is not None:
            return result
        time.sleep(0.05)
    return None


def _hit_counts(result: PrefetchResult | None) -> tuple[int, int]:
    """(L1 hit cells, L2 hit cells) of a finished request."""
    assert result is not None, "prefetch did not complete"
    return (
        Bitmap2D(result.l1_hit_cells).popcount(),
        Bitmap2D(result.l2_hit_cells).popcount(),
    )


def _store_in_l2(adapter: MockL2Adapter, keys: list[ObjectKey]) -> None:
    layout = _layout()
    objs = []
    for _ in keys:
        tensor = torch.randn(layout.shapes[0], dtype=layout.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=layout.shapes[0],
            dtype=layout.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        objs.append(TensorMemoryObj(tensor, metadata, parent_allocator=None))
    adapter.submit_store_task(keys, objs)  # type: ignore[arg-type]
    assert _wait_for(lambda: all(adapter.debug_has_key(k) for k in keys))


def _write_to_l1(l1_manager: DramL1Manager, keys: list[ObjectKey]) -> None:
    written = l1_manager.reserve_write(
        keys, is_temporary=[False] * len(keys), layout_desc=_layout()
    )
    for key in keys:
        assert written[key][0] == L1Error.SUCCESS
    l1_manager.finish_write(keys)


class _Availability:
    """The ``adapter_available`` callable, with a switch per adapter."""

    def __init__(self, num_adapters: int) -> None:
        self._up = {i: True for i in range(num_adapters)}

    def __call__(self, adapter_id: int) -> bool:
        return self._up[adapter_id]

    def set(self, adapter_id: int, up: bool) -> None:
        self._up[adapter_id] = up


class _StallingLookupAdapter(MockL2Adapter):
    """A mock adapter whose lookup results can be held back, like the lookups of
    an adapter whose backend does not answer.

    While stalled, the adapter does the lookup (and pins the keys it finds) but
    does not hand the result to the caller.
    """

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))
        self.stalled = True

    def query_lookup_and_lock_result(self, task_id: L2TaskId):
        if self.stalled:
            return None
        return super().query_lookup_and_lock_result(task_id)


def _adapter() -> MockL2Adapter:
    return MockL2Adapter(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))


@pytest.fixture
def l1_manager():
    manager = DramL1Manager(_l1_config())
    yield manager
    manager.close()


class TestStoreRouting:
    def test_unavailable_adapter_gets_no_new_stores_until_it_is_back(self, l1_manager):
        adapter = _adapter()
        availability = _Availability(1)
        availability.set(0, False)
        controller = StoreController(
            l1_manager,
            [adapter],
            [_descriptor(0)],
            DefaultStorePolicy(),
            adapter_available=availability,
        )
        controller.start()
        try:
            skipped = [_key(i) for i in range(2)]
            _write_to_l1(l1_manager, skipped)
            assert not _wait_for(
                lambda: adapter.debug_get_stored_object_count() > 0, timeout=0.5
            )

            availability.set(0, True)
            stored = [_key(10 + i) for i in range(2)]
            _write_to_l1(l1_manager, stored)

            assert _wait_for(lambda: all(adapter.debug_has_key(k) for k in stored))
            assert not any(adapter.debug_has_key(k) for k in skipped)
        finally:
            controller.stop()
            adapter.close()

    def test_available_adapters_still_get_the_keys(self, l1_manager):
        down, up = _adapter(), _adapter()
        availability = _Availability(2)
        availability.set(0, False)
        controller = StoreController(
            l1_manager,
            [down, up],
            [_descriptor(0), _descriptor(1)],
            DefaultStorePolicy(),
            adapter_available=availability,
        )
        controller.start()
        try:
            keys = [_key(i) for i in range(3)]
            _write_to_l1(l1_manager, keys)

            assert _wait_for(lambda: all(up.debug_has_key(k) for k in keys))
            assert down.debug_get_stored_object_count() == 0
        finally:
            controller.stop()
            down.close()
            up.close()


def _prefetch_controller(
    l1_manager: DramL1Manager,
    adapters: list,
    availability: _Availability,
    inflight_watermark: float = 0.5,
) -> PrefetchController:
    return PrefetchController(
        l1_managers=[l1_manager],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=_l1_config())],
        l2_adapters=adapters,
        adapter_descriptors=[_descriptor(i) for i in range(len(adapters))],
        policy=BufferOnlyPrefetchPolicy(),
        inflight_watermark=inflight_watermark,
        adapter_available=availability,
    )


class TestPrefetchRouting:
    def test_unavailable_adapter_is_not_asked(self, l1_manager):
        adapter = _adapter()
        keys = [_key(i) for i in range(3)]
        _store_in_l2(adapter, keys)
        availability = _Availability(1)
        availability.set(0, False)
        controller = _prefetch_controller(l1_manager, [adapter], availability)
        controller.start()
        try:
            request_id = controller.submit_prefetch_request(_spec(keys))
            assert _hit_counts(_wait_for_result(controller, request_id)) == (0, 0)

            availability.set(0, True)
            request_id = controller.submit_prefetch_request(_spec(keys))
            assert _hit_counts(_wait_for_result(controller, request_id)) == (0, 3)
        finally:
            controller.stop()
            adapter.close()

    def test_other_adapters_serve_while_one_is_down(self, l1_manager):
        down = _StallingLookupAdapter()
        up = _adapter()
        keys = [_key(i) for i in range(2)]
        _store_in_l2(up, keys)
        availability = _Availability(2)
        availability.set(0, False)
        controller = _prefetch_controller(l1_manager, [down, up], availability)
        controller.start()
        try:
            request_id = controller.submit_prefetch_request(_spec(keys))

            assert _hit_counts(_wait_for_result(controller, request_id)) == (0, 2)
        finally:
            controller.stop()
            down.close()
            up.close()


class TestAbandonedLookups:
    def test_a_lookup_waiting_on_an_adapter_that_goes_down_is_given_up(
        self, l1_manager
    ):
        adapter = _StallingLookupAdapter()
        keys = [_key(i) for i in range(3)]
        _store_in_l2(adapter, keys)
        availability = _Availability(1)
        controller = _prefetch_controller(l1_manager, [adapter], availability)
        controller.start()
        try:
            request_id = controller.submit_prefetch_request(_spec(keys))
            # The adapter holds its answer back: the request is stuck.
            assert _wait_for_result(controller, request_id, timeout=1.0) is None
            assert controller.report_status()["lookup_phase_count"] == 1

            availability.set(0, False)

            # The request finishes with no hits instead of waiting for ever.
            assert _hit_counts(_wait_for_result(controller, request_id)) == (0, 0)
            assert controller.report_status()["in_flight_request_count"] == 0
        finally:
            controller.stop()
            adapter.close()

    def test_a_late_lookup_result_is_unlocked_once_the_adapter_is_back(
        self, l1_manager
    ):
        adapter = _StallingLookupAdapter()
        keys = [_key(i) for i in range(3)]
        _store_in_l2(adapter, keys)
        availability = _Availability(1)
        controller = _prefetch_controller(l1_manager, [adapter], availability)
        controller.start()
        try:
            request_id = controller.submit_prefetch_request(_spec(keys))
            assert _wait_for(
                lambda: controller.report_status()["lookup_phase_count"] == 1
            )

            availability.set(0, False)
            _wait_for_result(controller, request_id)
            # The abandoned lookup did pin the keys on the adapter.
            assert adapter.debug_get_locked_key_count() == len(keys)

            adapter.stalled = False
            availability.set(0, True)

            assert _wait_for(
                lambda: adapter.debug_get_locked_key_count() == 0, timeout=5.0
            )
        finally:
            controller.stop()
            adapter.close()

    def test_requests_held_back_by_stuck_lookups_proceed(self, l1_manager):
        adapter = _StallingLookupAdapter()
        availability = _Availability(1)
        # A tiny watermark: a few stuck requests fill it, so later ones wait.
        controller = _prefetch_controller(
            l1_manager, [adapter], availability, inflight_watermark=0.01
        )
        controller.start()
        try:
            request_ids = [
                controller.submit_prefetch_request(
                    _spec([_key(100 * n + i) for i in range(3)])
                )
                for n in range(6)
            ]
            assert _wait_for(
                lambda: controller.report_status()["pending_queue_size"] >= 1
            )

            availability.set(0, False)

            for request_id in request_ids:
                assert _hit_counts(_wait_for_result(controller, request_id)) == (0, 0)
            assert controller.report_status()["pending_queue_size"] == 0
        finally:
            controller.stop()
            adapter.close()


def test_l1_only_hits_are_unaffected_by_an_unavailable_adapter(l1_manager):
    adapter = _adapter()
    keys = [_key(i) for i in range(2)]
    _write_to_l1(l1_manager, keys)
    availability = _Availability(1)
    availability.set(0, False)
    controller = _prefetch_controller(l1_manager, [adapter], availability)
    controller.start()
    try:
        request_id = controller.submit_prefetch_request(_spec(keys))

        assert _hit_counts(_wait_for_result(controller, request_id)) == (2, 0)
    finally:
        controller.stop()
        adapter.close()
