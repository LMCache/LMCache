# SPDX-License-Identifier: Apache-2.0
"""Failed runtime attachment must leave no adapter to attach or retry later."""

# Standard
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
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
from lmcache.v1.distributed.storage_controllers.store_controller import StoreController
from lmcache.v1.distributed.storage_controllers.store_policy import DefaultStorePolicy
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.distributed.storage_manager import StorageManager

pytestmark = pytest.mark.no_shared_allocator


def l1_config() -> L1ManagerConfig:
    """Create a small host pool independent of the shared GPU test allocator."""
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=8 * 1024 * 1024,
            use_lazy=False,
            align_bytes=4096,
            shm_name="",
        )
    )


def adapter_config() -> MockL2AdapterConfig:
    """Create a lightweight adapter with real completion eventfds."""
    return MockL2AdapterConfig(max_size_gb=0.001, mock_bandwidth_gb=10.0)


def make_controller(
    kind: str, manager: L1Manager
) -> StoreController | PrefetchController:
    """Create an unstarted controller with no initially attached adapters."""
    if kind == "store":
        return StoreController(manager, [], [], DefaultStorePolicy())
    return PrefetchController(
        [manager],
        [L1ManagerDescriptor(index=manager.l1_manager_id, config=manager.config)],
        [],
        [],
        DefaultPrefetchPolicy(),
    )


@pytest.fixture
def l1_manager() -> Iterator[L1Manager]:
    """Keep the L1 pool alive until the test has stopped its controller."""
    manager = L1Manager(l1_config())
    yield manager
    manager.close()


@pytest.mark.parametrize("kind", ["store", "prefetch"])
def test_timed_out_add_does_not_attach_when_loop_starts(
    kind: str, l1_manager: L1Manager
) -> None:
    """A failed add cannot become active after the controller resumes."""
    controller = make_controller(kind, l1_manager)
    failed_adapter = MockL2Adapter(adapter_config())
    retry_adapter = MockL2Adapter(adapter_config())
    started = False
    try:
        with pytest.raises(RuntimeError, match="did not attach adapter 0 in time"):
            controller.add_adapter(
                0, failed_adapter, L2AdapterDescriptor(0, adapter_config())
            )
        controller.start()
        started = True
        # The successful add also establishes that the earlier queued op ran.
        controller.add_adapter(
            1, retry_adapter, L2AdapterDescriptor(1, adapter_config())
        )
        assert controller.report_status()["num_active_adapters"] == 1
    finally:
        if not started:
            controller.start()
        controller.stop()
        failed_adapter.close()
        retry_adapter.close()


class BlockingAttachAdapter(MockL2Adapter):
    """Hold eventfd registration across the caller's acknowledgement deadline."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.entered = threading.Event()
        self.release = threading.Event()
        super().__init__(adapter_config())

    def get_store_event_fd(self) -> int:
        """Hold store registration until the test releases it."""
        if self.kind == "store":
            self.entered.set()
            if not self.release.wait(timeout=15.0):
                raise AssertionError("Test did not release store registration")
        return super().get_store_event_fd()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Hold prefetch registration until the test releases it."""
        if self.kind == "prefetch":
            self.entered.set()
            if not self.release.wait(timeout=15.0):
                raise AssertionError("Test did not release prefetch registration")
        return super().get_lookup_and_lock_event_fd()


@pytest.mark.parametrize("kind", ["store", "prefetch"])
def test_add_already_being_applied_reports_success(
    kind: str, l1_manager: L1Manager
) -> None:
    """The deadline cannot report failure while attachment is committing."""
    controller = make_controller(kind, l1_manager)
    adapter = BlockingAttachAdapter(kind)
    controller.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                controller.add_adapter,
                0,
                adapter,
                L2AdapterDescriptor(0, adapter_config()),
            )
            try:
                assert adapter.entered.wait(timeout=3.0)
                with pytest.raises(FuturesTimeoutError):
                    pending.result(timeout=5.7)
            finally:
                adapter.release.set()
            pending.result(timeout=3.0)
        assert controller.report_status()["num_active_adapters"] == 1
    finally:
        adapter.release.set()
        controller.stop()
        adapter.close()


@pytest.mark.parametrize("kind", ["store", "prefetch"])
def test_manager_rolls_back_failed_attachment_and_allows_retry(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A controller rejection leaves no registry entry or live adapter."""
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=l1_config(),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        )
    )
    closed_after_detach: list[bool] = []
    close = MockL2Adapter.close

    def record_close(adapter: MockL2Adapter) -> None:
        status = manager.report_status()
        closed_after_detach.append(
            not manager.l2_adapters()
            and status["store_controller"]["num_l2_adapters"] == 0
            and status["prefetch_controller"]["num_l2_adapters"] == 0
        )
        close(adapter)

    def reject_add(
        controller: StoreController | PrefetchController,
        adapter_id: int,
        adapter: L2AdapterInterface,
        descriptor: L2AdapterDescriptor,
    ) -> None:
        raise RuntimeError("attachment acknowledgement timed out")

    monkeypatch.setattr(MockL2Adapter, "close", record_close)
    try:
        with monkeypatch.context() as fault:
            target = StoreController if kind == "store" else PrefetchController
            fault.setattr(target, "add_adapter", reject_add)
            with pytest.raises(RuntimeError, match="acknowledgement timed out"):
                manager.add_l2_adapter(adapter_config())
        assert manager.l2_adapters() == []
        assert closed_after_detach == [True]

        adapter_id = manager.add_l2_adapter(adapter_config())
        assert [descriptor.index for descriptor, _ in manager.l2_adapters()] == [
            adapter_id
        ]
        assert manager.report_status()["store_controller"]["num_active_adapters"] == 1
        assert (
            manager.report_status()["prefetch_controller"]["num_active_adapters"] == 1
        )
    finally:
        for descriptor, _ in manager.l2_adapters():
            manager.delete_l2_adapter(descriptor.index)
        manager.close()
