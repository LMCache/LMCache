# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for FSL2Adapter capacity accounting and deletion.
"""

# Standard
from collections.abc import Callable, Generator
from pathlib import Path
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _legacy_object_key_to_filename,
    _object_key_to_filename,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd


def create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    object_group_id: int = 0,
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        object_group_id=object_group_id,
    )


def create_memory_obj(size: int = 16, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def wait_until(predicate: Callable[[], bool], timeout: float = 4.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture
def adapter(tmp_path: Path) -> Generator[FSL2Adapter, None, None]:
    config = FSL2AdapterConfig(
        base_path=str(tmp_path),
        max_capacity_gb=0.001,
    )
    a = FSL2Adapter(config)
    yield a
    a.close()


class RecordingListener(L2AdapterListener):
    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.append(list(keys))


def store(
    adapter: FSL2Adapter,
    key: ObjectKey,
    obj: TensorMemoryObj,
) -> L2StoreResult:
    task_id = adapter.submit_store_task([key], [obj])
    assert wait_for_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id]


def lookup(adapter: FSL2Adapter, key: ObjectKey) -> Bitmap:
    task_id = adapter.submit_lookup_and_lock_task([key])
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return bitmap


class TestConfig:
    def test_from_dict_parses_max_capacity_and_eviction(self, tmp_path: Path) -> None:
        cfg = FSL2AdapterConfig.from_dict(
            {
                "type": "fs",
                "base_path": str(tmp_path),
                "max_capacity_gb": 2.5,
                "eviction": {
                    "eviction_policy": "LRU",
                    "trigger_watermark": 0.7,
                    "eviction_ratio": 0.3,
                },
            }
        )
        assert cfg.max_capacity_gb == 2.5
        assert cfg.eviction_config is not None
        assert cfg.eviction_config.eviction_policy == "LRU"

    @pytest.mark.parametrize("bad_value", [-1, "1", True])
    def test_from_dict_rejects_invalid_max_capacity(
        self,
        tmp_path: Path,
        bad_value: object,
    ) -> None:
        with pytest.raises(ValueError, match="max_capacity_gb"):
            FSL2AdapterConfig.from_dict(
                {
                    "type": "fs",
                    "base_path": str(tmp_path),
                    "max_capacity_gb": bad_value,
                }
            )


class TestUsageAndDelete:
    def test_usage_grows_and_duplicate_store_does_not_double_count(
        self,
        adapter: FSL2Adapter,
    ) -> None:
        key = create_object_key(1)
        obj = create_memory_obj()

        result = store(adapter, key, obj)
        assert result.is_successful()
        assert result.bytes_transferred() == obj.get_size()
        assert adapter.get_usage().total_bytes_used == obj.get_size()

        duplicate = store(adapter, key, obj)
        assert duplicate.is_successful()
        assert duplicate.bytes_transferred() == 0
        assert adapter.get_usage().total_bytes_used == obj.get_size()

    def test_delete_removes_file_and_shrinks_usage(
        self,
        adapter: FSL2Adapter,
        tmp_path: Path,
    ) -> None:
        key = create_object_key(1)
        obj = create_memory_obj()
        store(adapter, key, obj)

        data_path = tmp_path / _object_key_to_filename(key)
        assert data_path.exists()
        adapter.delete([key])

        assert not data_path.exists()
        assert adapter.get_usage().total_bytes_used == 0

    def test_lookup_lock_blocks_delete_until_unlock(
        self,
        adapter: FSL2Adapter,
        tmp_path: Path,
    ) -> None:
        key = create_object_key(1)
        obj = create_memory_obj()
        store(adapter, key, obj)

        bitmap = lookup(adapter, key)
        assert bitmap.test(0) is True

        data_path = tmp_path / _object_key_to_filename(key)
        adapter.delete([key])
        assert data_path.exists()
        assert adapter.get_usage().total_bytes_used == obj.get_size()

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not data_path.exists()
        assert adapter.get_usage().total_bytes_used == 0

    def test_load_success_fires_access_event(self, adapter: FSL2Adapter) -> None:
        key = create_object_key(1)
        obj = create_memory_obj(fill_value=3.0)
        store(adapter, key, obj)

        listener = RecordingListener()
        adapter.register_listener(listener)
        dst = create_memory_obj(fill_value=0.0)
        task_id = adapter.submit_load_task([key], [dst])
        assert wait_for_event_fd(adapter.get_load_event_fd())
        bitmap = adapter.query_load_result(task_id)

        assert bitmap is not None and bitmap.test(0) is True
        dst_tensor = dst.tensor
        assert dst_tensor is not None
        assert torch.allclose(dst_tensor, torch.full((16,), 3.0))
        assert any(key in batch for batch in listener.accessed)


class TestRecovery:
    def test_restart_recovers_usage_and_can_delete(self, tmp_path: Path) -> None:
        key = create_object_key(1)
        obj = create_memory_obj()
        first = FSL2Adapter(
            FSL2AdapterConfig(base_path=str(tmp_path), max_capacity_gb=0.001)
        )
        try:
            store(first, key, obj)
        finally:
            first.close()

        reopened = FSL2Adapter(
            FSL2AdapterConfig(base_path=str(tmp_path), max_capacity_gb=0.001)
        )
        try:
            assert reopened.get_usage().total_bytes_used == obj.get_size()
            reopened.delete([key])
            assert reopened.get_usage().total_bytes_used == 0
            assert not (tmp_path / _object_key_to_filename(key)).exists()
        finally:
            reopened.close()

    def test_legacy_data_file_recovers_hits_loads_and_deletes(
        self,
        tmp_path: Path,
    ) -> None:
        key = ObjectKey(
            chunk_hash=bytes.fromhex(
                "fdedede53cfd5c28a427fb854eff39864330ca65973abd8612825412b94e4617"
            ),
            model_name="/aigcmodels01/modelscope/QuantTrio/GLM-5___1-AWQ",
            kv_rank=0x01000100,
            object_group_id=0,
        )
        src = create_memory_obj(fill_value=7.0)
        legacy_filename = _legacy_object_key_to_filename(key)
        assert legacy_filename is not None
        legacy_path = tmp_path / legacy_filename
        legacy_path.write_bytes(bytes(src.byte_array))

        reopened = FSL2Adapter(
            FSL2AdapterConfig(base_path=str(tmp_path), max_capacity_gb=0.001)
        )
        try:
            assert reopened.get_usage().total_bytes_used == src.get_size()

            bitmap = lookup(reopened, key)
            assert bitmap.test(0) is True

            dst = create_memory_obj(fill_value=0.0)
            task_id = reopened.submit_load_task([key], [dst])
            assert wait_for_event_fd(reopened.get_load_event_fd())
            load_bitmap = reopened.query_load_result(task_id)
            assert load_bitmap is not None and load_bitmap.test(0) is True
            dst_tensor = dst.tensor
            assert dst_tensor is not None
            assert torch.allclose(dst_tensor, torch.full((16,), 7.0))

            reopened.submit_unlock([key])
            reopened.delete([key])
            assert not legacy_path.exists()
            assert reopened.get_usage().total_bytes_used == 0
        finally:
            reopened.close()

    def test_recovery_skips_invalid_and_empty_files(self, tmp_path: Path) -> None:
        (tmp_path / "invalid.data").write_bytes(b"not-a-cache-file")
        (tmp_path / "llama@0x0000002a@0@deadbeef.data").write_bytes(b"")

        reopened = FSL2Adapter(
            FSL2AdapterConfig(base_path=str(tmp_path), max_capacity_gb=0.001)
        )
        try:
            assert reopened.get_usage().total_bytes_used == 0
        finally:
            reopened.close()


class TestEvictionIntegration:
    def test_eviction_controller_deletes_existing_fs_key(
        self,
        tmp_path: Path,
    ) -> None:
        key = create_object_key(1)
        obj = create_memory_obj()
        adapter = FSL2Adapter(
            FSL2AdapterConfig(
                base_path=str(tmp_path),
                max_capacity_gb=obj.get_size() / (1024**3),
            )
        )
        controller = None
        try:
            store(adapter, key, obj)
            eviction_state = L2AdapterEvictionState(
                adapter=adapter,
                eviction_config=EvictionConfig(
                    eviction_policy="LRU",
                    trigger_watermark=0.5,
                    eviction_ratio=1.0,
                ),
            )
            controller = L2EvictionController([eviction_state])
            controller.start()

            assert wait_until(lambda: adapter.get_usage().total_bytes_used == 0)
            assert not (tmp_path / _object_key_to_filename(key)).exists()
        finally:
            if controller is not None:
                controller.stop()
            adapter.close()
