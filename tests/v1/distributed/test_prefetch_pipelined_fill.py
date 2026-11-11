# SPDX-License-Identifier: Apache-2.0
"""PrefetchController handling of pipelined loads that finish with a fill."""

# Standard
from collections.abc import Iterator
from threading import Lock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import pending_fill
from lmcache.v1.distributed.api import (
    FULL_ATTENTION_WINDOW_CHUNKS,
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchLockMode,
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
from lmcache.v1.distributed.pending_fill import FillTracker
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    BufferOnlyPrefetchPolicy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.memory_management import (
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
L1_BYTES = 64 * 4096
TIMEOUT_S = 5.0


def make_l1_config() -> L1ManagerConfig:
    return L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=L1_BYTES, use_lazy=False, shm_name="")
    )


def make_key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "pipelined-fill-test", 0)


def wait_until(predicate) -> bool:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class FillingAdapter(MockL2Adapter):
    """MockL2Adapter whose loads finish with a fill the test settles."""

    def __init__(self) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0))
        self._fills_lock = Lock()
        self.fills: list[FillTracker] = []
        self._fill_by_task: dict[L2TaskId, FillTracker] = {}

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        # Held across the submit so the controller cannot ask for the fill
        # before it is recorded.
        with self._fills_lock:
            task_id = super().submit_load_task(keys, objects)
            fill = FillTracker(objects)
            self.fills.append(fill)
            self._fill_by_task[task_id] = fill
            return task_id

    def take_load_fill(self, task_id: L2TaskId) -> FillTracker | None:
        with self._fills_lock:
            return self._fill_by_task.pop(task_id, None)


def store_in_l2(adapter: MockL2Adapter, keys: list[ObjectKey]) -> None:
    objs: list[MemoryObj] = []
    for _ in keys:
        tensor = torch.randn(LAYOUT.shapes[0], dtype=LAYOUT.dtypes[0])
        metadata = MemoryObjMetadata(
            shape=LAYOUT.shapes[0],
            dtype=LAYOUT.dtypes[0],
            address=0,
            phy_size=tensor.nelement() * tensor.element_size(),
            ref_count=0,
        )
        objs.append(TensorMemoryObj(tensor, metadata, parent_allocator=None))
    adapter.submit_store_task(keys, objs)
    assert wait_until(lambda: all(adapter.debug_has_key(k) for k in keys))


def is_read_locked(l1: DramL1Manager, keys: list[ObjectKey]) -> bool:
    results = l1.unsafe_read(keys)
    return all(results[k][0] == L1Error.SUCCESS for k in keys)


def is_absent(l1: DramL1Manager, keys: list[ObjectKey]) -> bool:
    results = l1.reserve_read(keys)
    return all(results[k][0] == L1Error.KEY_NOT_EXIST for k in keys)


@pytest.fixture
def l1(monkeypatch: pytest.MonkeyPatch) -> Iterator[DramL1Manager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    manager = DramL1Manager(make_l1_config())
    yield manager
    manager.close()


@pytest.fixture
def prefetched(
    l1: DramL1Manager,
) -> Iterator[tuple[list[ObjectKey], FillTracker]]:
    """Prefetch two keys through a FillingAdapter; yield them and the fill."""
    adapter = FillingAdapter()
    keys = [make_key(i) for i in range(2)]
    store_in_l2(adapter, keys)
    ctrl = PrefetchController(
        l1_managers=[l1],
        l1_manager_descriptors=[L1ManagerDescriptor(index=0, config=make_l1_config())],
        l2_adapters=[adapter],
        adapter_descriptors=[
            L2AdapterDescriptor(
                index=0,
                config=MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0),
            )
        ],
        # Loaded objects are temporary, so they vanish once the last read
        # lock (the reader's or the fill's) is dropped.
        policy=BufferOnlyPrefetchPolicy(),
    )
    ctrl.start()
    try:
        spec = PrefetchTaskSpec(
            key_groups=[
                GroupedObjectKeys(
                    keys=keys,
                    object_group_id=0,
                    layout_desc=LAYOUT,
                    sliding_window_size=FULL_ATTENTION_WINDOW_CHUNKS,
                    num_kv_readers=1,
                )
            ],
            fetching_policy="prefix",
            lock_mode=PrefetchLockMode.LOCK,
        )
        request_id = ctrl.submit_prefetch_request(spec)
        assert ctrl.wait_prefetch_result(request_id, TIMEOUT_S)
        result = ctrl.query_prefetch_result(request_id)
        assert result is not None
        assert result.hit_cells[0].get_indices_list() == [0, 1]
        (fill,) = adapter.fills
        yield keys, fill
    finally:
        ctrl.stop()
        adapter.close()


def test_fill_holds_a_read_lock_until_it_succeeds(
    l1: DramL1Manager, prefetched: tuple[list[ObjectKey], FillTracker]
) -> None:
    keys, fill = prefetched
    # The reader is done, but the fill still holds the objects.
    l1.finish_read(keys)
    assert is_read_locked(l1, keys)

    fill.settle(True)

    assert is_absent(l1, keys)


def test_failed_fill_is_deleted_once_unread(
    l1: DramL1Manager, prefetched: tuple[list[ObjectKey], FillTracker]
) -> None:
    keys, fill = prefetched
    objs = [obj for _, obj in l1.unsafe_read(keys).values()]

    fill.settle(False)

    # The reader still holds the objects: they stay registered as failed
    # across a few controller iterations.
    time.sleep(0.6)
    assert is_read_locked(l1, keys)
    assert all(pending_fill.find(obj) is fill for obj in objs)
    assert pending_fill.wait_filled(objs) == [False, False]

    l1.finish_read(keys)

    assert wait_until(lambda: all(pending_fill.find(obj) is None for obj in objs))
    assert is_absent(l1, keys)
