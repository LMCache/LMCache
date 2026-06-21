# SPDX-License-Identifier: Apache-2.0
"""Tests for variable-size distributed serde payload handling."""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
import os
import shutil
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.serde import (
    AsyncSerdeProcessor,
    Deserializer,
    SerdeConfig,
    Serializer,
    register_serde_factory,
)
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.platform import create_event_notifier


def _key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i),
        model_name="test-model",
        kv_rank=0,
    )


def _byte_obj(n: int) -> MemoryObj:
    raw = torch.arange(n, dtype=torch.uint8)
    return TensorMemoryObj(
        raw_data=raw,
        metadata=MemoryObjMetadata(
            shape=torch.Size([n]),
            dtype=torch.uint8,
            address=-1,
            phy_size=n,
            ref_count=1,
            shapes=[torch.Size([n])],
            dtypes=[torch.uint8],
        ),
        parent_allocator=None,
    )


def _wait_for_store_result(adapter: L2AdapterInterface, task_id: int) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            assert completed[task_id].is_successful()
            return
        time.sleep(0.05)
    raise AssertionError("store task did not complete")


def _wait_for_condition(
    predicate: Callable[[], bool],
    timeout: float = 10.0,
    poll_interval: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def _wait_for_prefetch(sm: Any, handle: Any) -> int | None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        result = sm.query_prefetch_status(handle)
        if result is not None:
            return result.count_leading_ones()
        time.sleep(0.05)
    return None


def _assert_temp_keys_match_originals(
    temp_keys: list[ObjectKey],
    original_keys: list[ObjectKey],
) -> None:
    assert len(temp_keys) == len(original_keys)
    for temp_key, original_key in zip(temp_keys, original_keys, strict=True):
        assert temp_key.model_name == original_key.model_name
        assert temp_key.kv_rank == original_key.kv_rank
        assert temp_key.object_group_id == original_key.object_group_id
        assert temp_key.cache_salt == original_key.cache_salt
        assert temp_key.chunk_hash.startswith(original_key.chunk_hash)
        assert len(temp_key.chunk_hash) == len(original_key.chunk_hash) + 16


def _import_storage_manager_deps() -> tuple[type, type, type, type, type, type]:
    try:
        # First Party
        from lmcache.v1.distributed.config import (
            EvictionConfig,
            L1ManagerConfig,
            L1MemoryManagerConfig,
            StorageManagerConfig,
        )
        from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
        from lmcache.v1.distributed.storage_manager import StorageManager
    except ImportError as exc:
        pytest.skip(f"StorageManager native dependencies are unavailable: {exc}")

    return (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
        L2AdaptersConfig,
        StorageManager,
    )


class _PrefixSerializer(Serializer):
    """Write a small prefix to exercise variable-size serialized payloads."""

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Copy only the first eight source bytes into ``dst``.

        Args:
            src: Source object containing a tensor.
            dst: Destination byte-buffer object.

        Returns:
            The number of bytes written.

        Raises:
            ValueError: If either object does not expose a tensor.
        """
        if src.tensor is None or dst.tensor is None:
            raise ValueError("test serializer requires tensors")
        payload = src.tensor.flatten().view(torch.uint8)[:8].clone()
        dst.tensor.flatten()[: payload.numel()].copy_(payload)
        return int(payload.numel())

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return an intentionally larger upper bound for the serialized buffer.

        Args:
            layout_desc: Source layout description. Unused by this test serde.

        Returns:
            A fixed 64-byte estimate.
        """
        return 64


class _PrefixDeserializer(Deserializer):
    """Restore the stored prefix and leave the remainder zeroed."""

    def deserialize(self, src: MemoryObj, dst: MemoryObj) -> None:
        """Copy all serialized bytes from ``src`` into the start of ``dst``.

        Args:
            src: Source serialized byte buffer.
            dst: Destination object containing the original layout.

        Raises:
            ValueError: If either object does not expose a tensor.
        """
        if src.tensor is None or dst.tensor is None:
            raise ValueError("test deserializer requires tensors")
        dst.tensor.flatten().zero_()
        dst.tensor.flatten().view(torch.uint8)[: src.get_size()].copy_(
            src.tensor.flatten().view(torch.uint8)[: src.get_size()]
        )


def _create_prefix_serde(kwargs: dict[str, object]) -> AsyncSerdeProcessor:
    """Create the variable-size test serde processor."""
    return AsyncSerdeProcessor(_PrefixSerializer(), _PrefixDeserializer())


class _EmptySerializer(Serializer):
    """Write an empty payload to exercise exact zero-size L2 loads."""

    def serialize(self, src: MemoryObj, dst: MemoryObj) -> int:
        """Return zero bytes written.

        Args:
            src: Source object. Unused by this test serde.
            dst: Destination byte-buffer object. Unused by this test serde.

        Returns:
            Zero bytes written.
        """
        return 0

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Return a non-zero estimate for an empty actual payload.

        Args:
            layout_desc: Source layout description. Unused by this test serde.

        Returns:
            A fixed 64-byte estimate.
        """
        return 64


def _create_empty_serde(kwargs: dict[str, object]) -> AsyncSerdeProcessor:
    """Create the empty-payload test serde processor."""
    return AsyncSerdeProcessor(_EmptySerializer(), _PrefixDeserializer())


try:
    register_serde_factory("test-prefix-variable-size", _create_prefix_serde)
except ValueError:
    pass

try:
    register_serde_factory("test-empty-variable-size", _create_empty_serde)
except ValueError:
    pass


def test_fs_adapter_get_object_sizes_reports_file_lengths(tmp_path: Path) -> None:
    cfg = FSL2AdapterConfig(base_path=str(tmp_path))
    adapter = FSL2Adapter(cfg)
    try:
        key = _key(1)
        task_id = adapter.submit_store_task([key], [_byte_obj(5)])
        _wait_for_store_result(adapter, task_id)
        assert adapter.get_object_sizes([key, _key(2)]) == {key: 5}
    finally:
        adapter.close()


def test_mock_adapter_get_object_sizes_reports_stored_logical_sizes() -> None:
    adapter = MockL2Adapter(MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1))
    try:
        key = _key(1)
        obj = _byte_obj(32)
        obj.set_used_size(7)
        task_id = adapter.submit_store_task([key], [obj])
        _wait_for_store_result(adapter, task_id)
        assert adapter.get_object_sizes([key, _key(2)]) == {key: 7}
    finally:
        adapter.close()


def _byte_size(layout_desc: MemoryLayoutDesc) -> int:
    return sum(
        int(shape.numel()) * dtype.itemsize
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
    )


def test_serde_wrapper_store_batches_equal_temp_layout_reservations() -> None:
    """Store uses one batched temp reservation for identical serialized layouts."""

    class _CountingL1Manager:
        def __init__(self) -> None:
            self.reserve_calls: list[
                tuple[list[ObjectKey], list[bool], MemoryLayoutDesc, str]
            ] = []
            self.finished_write_read_keys: list[ObjectKey] = []
            self.finished_read_keys: list[ObjectKey] = []

        def reserve_write(
            self,
            keys: list[ObjectKey],
            is_temporary: list[bool],
            layout_desc: MemoryLayoutDesc,
            mode: str = "all",
        ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
            self.reserve_calls.append(
                (list(keys), list(is_temporary), layout_desc, mode)
            )
            return {
                key: (L1Error.SUCCESS, _byte_obj(_byte_size(layout_desc)))
                for key in keys
            }

        def finish_write(self, keys: list[ObjectKey]) -> None:
            pass

        def finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
            self.finished_write_read_keys.extend(keys)

        def finish_read(self, keys: list[ObjectKey]) -> None:
            self.finished_read_keys.extend(keys)

        def delete(self, keys: list[ObjectKey]) -> None:
            pass

    fake_l1 = _CountingL1Manager()
    inner = MockL2Adapter(MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1))
    serde = AsyncSerdeProcessor(_PrefixSerializer(), _PrefixDeserializer())
    wrapper = SerdeL2AdapterWrapper(inner, serde, cast(L1Manager, fake_l1))
    try:
        keys = [_key(20), _key(21), _key(22)]
        task_id = wrapper.submit_store_task(keys, [_byte_obj(16) for _ in keys])
        _wait_for_store_result(wrapper, task_id)

        assert len(fake_l1.reserve_calls) == 1
        reserve_keys, is_temporary, reserve_layout, mode = fake_l1.reserve_calls[0]
        _assert_temp_keys_match_originals(reserve_keys, keys)
        assert is_temporary == [True] * len(keys)
        assert reserve_layout == MemoryLayoutDesc(
            shapes=[torch.Size([64])],
            dtypes=[torch.uint8],
        )
        assert mode == "new"
        assert fake_l1.finished_write_read_keys == reserve_keys
        assert fake_l1.finished_read_keys == reserve_keys
    finally:
        wrapper.close()


def test_serde_wrapper_store_releases_all_temp_reservations_on_failure() -> None:
    """Batched temp reservation failures release every successful temp key."""

    class _PartialFailureL1Manager:
        def __init__(self) -> None:
            self.reserved_keys: list[ObjectKey] = []
            self.finished_keys: list[ObjectKey] = []
            self.deleted_keys: list[ObjectKey] = []

        def reserve_write(
            self,
            keys: list[ObjectKey],
            is_temporary: list[bool],
            layout_desc: MemoryLayoutDesc,
            mode: str = "all",
        ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
            self.reserved_keys = list(keys)
            return {
                key: (
                    (L1Error.OUT_OF_MEMORY, None)
                    if index == 1
                    else (L1Error.SUCCESS, _byte_obj(1))
                )
                for index, key in enumerate(keys)
            }

        def finish_write(self, keys: list[ObjectKey]) -> None:
            self.finished_keys.extend(keys)

        def finish_write_and_reserve_read(self, keys: list[ObjectKey]) -> None:
            pass

        def finish_read(self, keys: list[ObjectKey]) -> None:
            pass

        def delete(self, keys: list[ObjectKey]) -> None:
            self.deleted_keys.extend(keys)

    fake_l1 = _PartialFailureL1Manager()
    inner = MockL2Adapter(MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1))
    serde = AsyncSerdeProcessor(_PrefixSerializer(), _PrefixDeserializer())
    wrapper = SerdeL2AdapterWrapper(inner, serde, cast(L1Manager, fake_l1))
    try:
        keys = [_key(30), _key(31), _key(32)]
        task_id = wrapper.submit_store_task(keys, [_byte_obj(16) for _ in keys])
        completed = wrapper.pop_completed_store_tasks()

        assert task_id in completed
        assert not completed[task_id].is_successful()
        _assert_temp_keys_match_originals(fake_l1.reserved_keys, keys)
        released_keys = [fake_l1.reserved_keys[0], fake_l1.reserved_keys[2]]
        assert fake_l1.finished_keys == released_keys
        assert fake_l1.deleted_keys == released_keys
    finally:
        wrapper.close()


def test_native_adapter_does_not_report_advisory_object_sizes() -> None:
    class _NativeClient:
        def __init__(self) -> None:
            self._notifier = create_event_notifier()
            self._next_future_id = 0
            self._completions: list[tuple[int, bool, str, list[bool] | None]] = []

        def event_fd(self) -> int:
            return self._notifier.fileno()

        def submit_batch_set(
            self,
            keys: list[str],
            memoryviews: list[memoryview],
        ) -> int:
            future_id = self._next_future_id
            self._next_future_id += 1
            self._completions.append((future_id, True, "", [True] * len(keys)))
            self._notifier.notify()
            return future_id

        def drain_completions(
            self,
        ) -> list[tuple[int, bool, str, list[bool] | None]]:
            completions = self._completions
            self._completions = []
            self._notifier.consume()
            return completions

        def close(self) -> None:
            self._notifier.close()

    adapter = NativeConnectorL2Adapter(_NativeClient())
    try:
        key = _key(3)
        task_id = adapter.submit_store_task([key], [_byte_obj(9)])
        _wait_for_store_result(adapter, task_id)
        assert adapter.get_object_sizes([key]) == {}
    finally:
        adapter.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fs_serde_load_uses_actual_stored_payload_size() -> None:
    (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
        L2AdaptersConfig,
        StorageManager,
    ) = _import_storage_manager_deps()

    disk_path = tempfile.mkdtemp(prefix="lmcache_variable_size_")
    sm: Any | None = None
    try:
        fs_cfg = FSL2AdapterConfig(base_path=disk_path)
        fs_cfg.serde_config = SerdeConfig(type="test-prefix-variable-size")
        sm = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=512 << 20,
                        use_lazy=True,
                        init_size_in_bytes=128 << 20,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
        )
        key = _key(10)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([16])],
            dtypes=[torch.uint8],
        )

        reserved = sm.reserve_write([key], layout, mode="new")
        assert key in reserved
        tensor = reserved[key].tensor
        assert tensor is not None
        tensor.copy_(torch.arange(16, dtype=torch.uint8))
        sm.finish_write([key])

        assert _wait_for_condition(
            lambda: any(
                entry.is_file() and not entry.name.endswith(".tmp")
                for entry in os.scandir(disk_path)
            )
        )
        stored_files = [entry for entry in os.scandir(disk_path) if entry.is_file()]
        assert len(stored_files) == 1
        assert stored_files[0].stat().st_size == 8

        sm.clear(force=True)
        handle = sm.submit_prefetch_task([key], layout)
        assert _wait_for_prefetch(sm, handle) == 1
        with sm.read_prefetched_results([key]) as mem_objs:
            assert mem_objs is not None
            assert mem_objs[0].tensor is not None
            got = mem_objs[0].tensor.flatten().view(torch.uint8)
            assert torch.equal(got[:8].cpu(), torch.arange(8, dtype=torch.uint8))
            assert torch.equal(got[8:].cpu(), torch.zeros(8, dtype=torch.uint8))
        sm.finish_read_prefetched([key])
    finally:
        if sm is not None:
            sm.close()
        shutil.rmtree(disk_path, ignore_errors=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fs_serde_load_uses_exact_zero_stored_payload_size() -> None:
    (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
        L2AdaptersConfig,
        StorageManager,
    ) = _import_storage_manager_deps()

    disk_path = tempfile.mkdtemp(prefix="lmcache_zero_size_")
    sm: Any | None = None
    try:
        fs_cfg = FSL2AdapterConfig(base_path=disk_path)
        fs_cfg.serde_config = SerdeConfig(type="test-empty-variable-size")
        sm = StorageManager(
            StorageManagerConfig(
                l1_manager_config=L1ManagerConfig(
                    memory_config=L1MemoryManagerConfig(
                        size_in_bytes=512 << 20,
                        use_lazy=True,
                        init_size_in_bytes=128 << 20,
                    ),
                ),
                eviction_config=EvictionConfig(eviction_policy="LRU"),
                l2_adapter_config=L2AdaptersConfig(adapters=[fs_cfg]),  # type: ignore[list-item]
            )
        )
        key = _key(11)
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([16])],
            dtypes=[torch.uint8],
        )

        reserved = sm.reserve_write([key], layout, mode="new")
        assert key in reserved
        tensor = reserved[key].tensor
        assert tensor is not None
        tensor.copy_(torch.arange(16, dtype=torch.uint8))
        sm.finish_write([key])

        assert _wait_for_condition(
            lambda: any(
                entry.is_file() and not entry.name.endswith(".tmp")
                for entry in os.scandir(disk_path)
            )
        )
        stored_files = [entry for entry in os.scandir(disk_path) if entry.is_file()]
        assert len(stored_files) == 1
        assert stored_files[0].stat().st_size == 0

        sm.clear(force=True)
        handle = sm.submit_prefetch_task([key], layout)
        assert _wait_for_prefetch(sm, handle) == 1
        with sm.read_prefetched_results([key]) as mem_objs:
            assert mem_objs is not None
            assert mem_objs[0].tensor is not None
            got = mem_objs[0].tensor.flatten().view(torch.uint8)
            assert torch.equal(got.cpu(), torch.zeros(16, dtype=torch.uint8))
        sm.finish_read_prefetched([key])
    finally:
        if sm is not None:
            sm.close()
        shutil.rmtree(disk_path, ignore_errors=True)
