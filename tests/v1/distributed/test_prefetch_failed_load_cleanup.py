# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the cleanup of L2 loads that fail or are cancelled.

The L1 destination of an L2 load is write-locked from its reservation until the
load is confirmed. When the load fails (the adapter reports the key as not
loaded) or the controller stops while the load is still in flight, the
destination holds data that was never confirmed: it may be incomplete or
uninitialized. PrefetchController must remove such a reservation while it is
still write-locked. Releasing the write lock first would make the destination
readable and announce the key to the write-finished listeners; the store
controller is such a listener and may reserve a read and store the key to L2
before the delete runs, which then fails with KEY_IS_LOCKED.

Uses a real L1Manager, MockL2Adapter and FaultInjectL2Adapter; the assertions
go through the public L1Manager and L2 adapter interfaces only.
"""

# Standard
from collections.abc import Callable, Iterator
from contextlib import ExitStack
import itertools
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
)
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.fault_inject_l2_adapter import (
    FaultInjectL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    DefaultPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.store_controller import (
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from tests.v1.distributed.utils import should_use_lazy_alloc

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime",
        allow_module_level=True,
    )

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_layout() -> MemoryLayoutDesc:
    """Create a small MemoryLayoutDesc (32 KiB per object)."""
    return MemoryLayoutDesc(
        shapes=[torch.Size([64, 2, 128])],
        dtypes=[torch.bfloat16],
    )


def make_config() -> MockL2AdapterConfig:
    """Create the MockL2AdapterConfig used by every adapter of these tests."""
    return MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    return AdapterDescriptor(index=index, config=make_config())


def wait_for_condition(
    predicate: Callable[[], bool],
    timeout: float = 5.0,
    poll_interval: float = 0.01,
) -> bool:
    """Poll until ``predicate`` returns True or ``timeout`` seconds have passed.

    Args:
        predicate: Called once per poll; the wait ends when it returns True.
        timeout: Upper bound on the wait, in seconds.
        poll_interval: Pause between two polls, in seconds.

    Returns:
        True if the predicate held before the deadline, False otherwise. The
        predicate is evaluated one last time at the deadline.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return predicate()


def wait_for_prefetch_result(
    ctrl: PrefetchController,
    req_id: int,
    timeout: float = 5.0,
    poll_interval: float = 0.01,
) -> Bitmap | None:
    """Poll ``query_prefetch_result`` until it returns the retained bitmap.

    The query pops the result, so it is called once per poll and its value kept.

    Args:
        ctrl: The controller the request was submitted to.
        req_id: The request id returned by ``submit_prefetch_request``.
        timeout: Upper bound on the wait, in seconds.
        poll_interval: Pause between two polls, in seconds.

    Returns:
        The retained bitmap, or None if no result arrived before the deadline.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = ctrl.query_prefetch_result(req_id)
        if result is not None or time.monotonic() >= deadline:
            return result
        time.sleep(poll_interval)


def store_keys_in_l2(
    adapter: MockL2Adapter,
    keys: list[ObjectKey],
    layout: MemoryLayoutDesc,
    timeout: float = 10.0,
) -> None:
    """Store random data for ``keys`` directly in the adapter and wait for it."""
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
        obj = TensorMemoryObj(raw_data=tensor, metadata=metadata, parent_allocator=None)
        objs.append(obj)
    adapter.submit_store_task(keys, objs)  # type: ignore
    ok = wait_for_condition(
        lambda: all(adapter.debug_has_key(k) for k in keys),
        timeout=timeout,
    )
    assert ok, "Failed to store test data in L2 adapter"


def read_error(l1_manager: L1Manager, key: ObjectKey) -> L1Error:
    """Return the L1Error of a ``reserve_read`` on ``key``, releasing any lock taken."""
    err, _ = l1_manager.reserve_read([key])[key]
    if err == L1Error.SUCCESS:
        l1_manager.finish_read([key])
    return err


def write_key_to_l1(
    l1_manager: L1Manager, key: ObjectKey, layout: MemoryLayoutDesc
) -> None:
    """Reserve, fill and finish a write of ``key`` in L1 through the public API."""
    err, obj = l1_manager.reserve_write(
        keys=[key], is_temporary=[False], layout_desc=layout, mode="new"
    )[key]
    assert err == L1Error.SUCCESS and obj is not None
    assert obj.tensor is not None
    obj.tensor.zero_()
    assert l1_manager.finish_write([key])[key] == L1Error.SUCCESS


class RecordingListener(L1ManagerListener):
    """Records the keys L1Manager announces as write-finished.

    The other callbacks are intentionally no-ops: the tests only ask which
    keys were published as completed writes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._write_finished: list[ObjectKey] = []

    def write_finished_keys(self) -> list[ObjectKey]:
        """Return a copy of the keys announced as write-finished so far."""
        with self._lock:
            return list(self._write_finished)

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        """Record ``keys``; called inside L1Manager's lock, so it only appends."""
        with self._lock:
            self._write_finished.extend(keys)

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        """Not recorded."""


class PendingLoadAdapter(MockL2Adapter):
    """A MockL2Adapter whose loads never complete.

    Lookup and store behave as in MockL2Adapter. ``submit_load_task`` records
    the call and sets ``load_submitted`` without touching the destinations;
    ``query_load_result`` always answers None, so a load stays in flight until
    the controller is stopped.
    """

    def __init__(self, config: MockL2AdapterConfig) -> None:
        super().__init__(config)
        self.load_submitted = threading.Event()
        self._load_task_ids = itertools.count(1)

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Record the submission and leave the load pending forever."""
        task_id = next(self._load_task_ids)
        self.load_submitted.set()
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Never complete a load."""
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    """Create an L1Manager with a reasonable memory config."""
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


# =============================================================================
# Tests
# =============================================================================


class TestFailedLoadCleanup:
    """A failed or cancelled load never publishes its L1 destination."""

    def test_failed_load_reservation_is_never_published(
        self, l1_manager: L1Manager
    ) -> None:
        """A load that fails leaves no readable, announced or stored key behind.

        The source adapter reports the key present at lookup and fails its
        load (FaultInjectL2Adapter, gap at task position 0). A store controller
        with an empty sink adapter listens to L1. A control key written to L1
        after the prefetch shows that the store controller is running and has
        drained everything announced before it; the failed key must not be
        among the objects the sink received.
        """
        layout = make_layout()
        key = make_object_key(1)
        control = make_object_key(2)

        with ExitStack() as stack:
            source = MockL2Adapter(make_config())
            stack.callback(source.close)
            store_keys_in_l2(source, [key], layout)
            faulty = FaultInjectL2Adapter(source, rate=0.0, seed=0, gap_indices=(0,))
            sink = MockL2Adapter(make_config())
            stack.callback(sink.close)

            listener = RecordingListener()
            l1_manager.register_listener(listener)
            store_ctrl = StoreController(
                l1_manager=l1_manager,
                l2_adapters=[sink],
                adapter_descriptors=[make_descriptor(1)],
                policy=DefaultStorePolicy(),
            )
            store_ctrl.start()
            stack.callback(store_ctrl.stop)
            ctrl = PrefetchController(
                l1_manager=l1_manager,
                l2_adapters=[faulty],
                adapter_descriptors=[make_descriptor(0)],
                policy=DefaultPrefetchPolicy(),
            )
            ctrl.start()
            stack.callback(ctrl.stop)

            req_id = ctrl.submit_prefetch_request(
                PrefetchRequestSpec([key], {0: layout})
            )
            retained = wait_for_prefetch_result(ctrl, req_id)
            assert retained is not None, "prefetch did not complete"
            assert retained.popcount() == 0

            # The destination is gone and was never announced.
            assert read_error(l1_manager, key) == L1Error.KEY_NOT_EXIST
            assert key not in listener.write_finished_keys()

            # The store controller stores the control key, so it has processed
            # every key announced before it; the failed key is not in the sink.
            write_key_to_l1(l1_manager, control, layout)
            assert wait_for_condition(lambda: sink.debug_has_key(control)), (
                "the store controller did not store the control key"
            )
            assert not sink.debug_has_key(key)
            assert sink.debug_get_stored_object_count() == 1

    def test_stop_with_load_in_flight_never_publishes_destination(
        self, l1_manager: L1Manager
    ) -> None:
        """Stopping the controller mid-load leaves no readable or announced key.

        The adapter never completes its load, so the controller is stopped with
        the destination reserved (write-locked) and the load in flight.
        """
        layout = make_layout()
        key = make_object_key(3)

        with ExitStack() as stack:
            pending = PendingLoadAdapter(make_config())
            stack.callback(pending.close)
            store_keys_in_l2(pending, [key], layout)

            listener = RecordingListener()
            l1_manager.register_listener(listener)
            ctrl = PrefetchController(
                l1_manager=l1_manager,
                l2_adapters=[pending],
                adapter_descriptors=[make_descriptor(0)],
                policy=DefaultPrefetchPolicy(),
            )
            ctrl.start()
            stopped = threading.Event()

            def stop_controller() -> None:
                if not stopped.is_set():
                    stopped.set()
                    ctrl.stop()

            stack.callback(stop_controller)

            ctrl.submit_prefetch_request(PrefetchRequestSpec([key], {0: layout}))
            assert pending.load_submitted.wait(timeout=5.0), (
                "the load was never submitted to the adapter"
            )
            # The destination was reserved before the load was submitted.
            assert read_error(l1_manager, key) == L1Error.KEY_NOT_READABLE

            stop_controller()

            assert read_error(l1_manager, key) == L1Error.KEY_NOT_EXIST
            assert key not in listener.write_finished_keys()
