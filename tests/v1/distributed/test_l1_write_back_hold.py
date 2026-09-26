# SPDX-License-Identifier: Apache-2.0
"""Tests for L1 write-back holds (issue #5350).

A key written to L1 must not be evictable before the StoreController has
reserved it for the L2 store; otherwise eviction silently drops data that was
never written back. CPU-only: uses a real L1Manager and MockL2Adapter.
"""

# Standard
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener
from lmcache.v1.distributed.l1_manager import L1Manager
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

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


class _RecordingListener(L1ManagerListener):
    def __init__(self) -> None:
        self.written: list[ObjectKey] = []
        self.read_finished: list[ObjectKey] = []

    def on_l1_keys_reserved_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_read_finished(self, keys: list[ObjectKey]) -> None:
        self.read_finished.extend(keys)

    def on_l1_keys_reserved_write(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_write_finished(self, keys: list[ObjectKey]) -> None:
        self.written.extend(keys)

    def on_l1_keys_finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_deleted_by_manager(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l1_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def _write(
    l1_manager: L1Manager, keys: list[ObjectKey], temporary: bool = False
) -> None:
    results = l1_manager.reserve_write(
        keys=keys, is_temporary=[temporary] * len(keys), layout_desc=_LAYOUT
    )
    assert all(err == L1Error.SUCCESS for err, _ in results.values())
    l1_manager.finish_write(keys)


def _evict(l1_manager: L1Manager, keys: list[ObjectKey]) -> None:
    """Evict like L1EvictionController: filter by eligibility, then delete."""
    l1_manager.delete([k for k in keys if l1_manager.is_key_evictable(k)])


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def l1_manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=32 * 1024 * 1024,
            use_lazy=False,
            init_size_in_bytes=32 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


class TestWriteBackHold:
    def test_written_key_is_held_until_released(self, l1_manager):
        listener = _RecordingListener()
        l1_manager.register_write_back_listener(listener)
        key = _key(1)

        _write(l1_manager, [key])

        assert listener.written == [key]
        assert not l1_manager.is_key_evictable(key)
        _evict(l1_manager, [key])
        assert l1_manager.get_object_state(key) is not None

        l1_manager.release_write_back_holds([key])

        assert l1_manager.is_key_evictable(key)
        assert listener.read_finished == []
        assert l1_manager.report_status()["read_locked_count"] == 0

    def test_explicit_delete_ignores_hold(self, l1_manager):
        l1_manager.register_write_back_listener(_RecordingListener())
        key = _key(5)
        _write(l1_manager, [key])

        assert l1_manager.delete([key])[key] == L1Error.SUCCESS
        l1_manager.release_write_back_holds([key])

    def test_plain_listener_takes_no_hold(self, l1_manager):
        l1_manager.register_listener(_RecordingListener())
        key = _key(2)

        _write(l1_manager, [key])

        assert l1_manager.is_key_evictable(key)

    def test_temporary_key_takes_no_hold(self, l1_manager):
        listener = _RecordingListener()
        l1_manager.register_write_back_listener(listener)
        key = _key(3)

        _write(l1_manager, [key], temporary=True)

        assert listener.written == []
        assert l1_manager.is_key_evictable(key)

    def test_release_skips_unheld_and_missing_keys(self, l1_manager):
        l1_manager.register_write_back_listener(_RecordingListener())
        key = _key(4)
        _write(l1_manager, [key])

        l1_manager.release_write_back_holds([key, _key(99)])
        l1_manager.release_write_back_holds([key])

        assert l1_manager.is_key_evictable(key)
        assert l1_manager.report_status()["read_locked_count"] == 0


class TestStoreControllerEvictionRace:
    def test_key_evicted_before_store_loop_runs_still_reaches_l2(self, l1_manager):
        adapter = MockL2Adapter(
            MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=10.0)
        )
        descriptor = L2AdapterDescriptor(
            index=0, config=MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=10.0)
        )
        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[adapter],
            adapter_descriptors=[descriptor],
            policy=DefaultStorePolicy(),
        )
        keys = [_key(i) for i in range(10, 14)]
        try:
            # The store loop is not running yet, so the keys sit in its queue.
            _write(l1_manager, keys)

            # What L1EvictionController does under memory pressure.
            _evict(l1_manager, keys)

            ctrl.start()
            assert _wait_for(
                lambda: adapter.debug_get_stored_object_count() == len(keys)
            ), "keys evicted before the store loop ran never reached L2"
            assert _wait_for(
                lambda: l1_manager.report_status()["read_locked_count"] == 0
            ), "write-back holds or store read locks leaked"
        finally:
            ctrl.stop()
            adapter.close()

    def test_stop_releases_holds_of_unprocessed_keys(self, l1_manager):
        entered = threading.Event()
        unblock = threading.Event()

        class _BlockingPolicy(DefaultStorePolicy):
            def select_store_targets(self, keys, adapters):
                entered.set()
                unblock.wait(timeout=5.0)
                return super().select_store_targets(keys, adapters)

        ctrl = StoreController(
            l1_manager=l1_manager,
            l2_adapters=[],
            adapter_descriptors=[],
            policy=_BlockingPolicy(),
        )
        ctrl.start()
        first, queued = _key(20), _key(21)
        _write(l1_manager, [first])
        assert entered.wait(timeout=5.0)
        # The loop is busy with ``first``; ``queued`` waits in its queue.
        _write(l1_manager, [queued])

        stopper = threading.Thread(target=ctrl.stop)
        stopper.start()
        time.sleep(0.1)
        unblock.set()
        stopper.join(timeout=10.0)

        assert l1_manager.is_key_evictable(first)
        assert l1_manager.is_key_evictable(queued)
