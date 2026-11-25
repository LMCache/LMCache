# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from unittest.mock import patch
import os
import select
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener
from lmcache.v1.distributed.l2_adapters.raw_block import encode_object_key
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import (
    IoUringBlockIO,
    IoUringBlockIOConfig,
    PosixBlockIO,
    RawBlockIOConfig,
)
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from tests.v1.distributed.raw_block.test_utils import _RecordingRawDevice
import lmcache.v1.distributed.l2_adapters.raw_block.core as raw_block_core
import lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter as raw_block_l2_adapter

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


requires_raw_block_ext = pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self.deleted.append(list(keys))


class _FailingListener(L2AdapterListener):
    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        del keys
        raise RuntimeError("store listener failed")

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        raise RuntimeError("access listener failed")

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        del keys
        raise RuntimeError("delete listener failed")


def _create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _create_memory_obj(size: int = 1024, fill_value: float = 0.0) -> TensorMemoryObj:
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


def _create_complex_memory_obj(
    size: int = 1024,
    fill_value: complex = 0j,
) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.complex64)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.complex64,
        address=0,
        phy_size=raw_data.numel() * raw_data.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _make_config(
    device_path: str,
    *,
    capacity_bytes: int = 0,
    io_engine: str = "posix",
    use_uring_cmd: bool = False,
    max_data_transfer_size: int = 0,
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path=device_path,
        capacity_bytes=capacity_bytes,
        use_odirect=False,
        io_engine=io_engine,
        use_uring_cmd=use_uring_cmd,
        max_data_transfer_size=max_data_transfer_size,
        block_align=4096,
        header_bytes=4096,
        meta_total_bytes=1 * 1024 * 1024,
        meta_enable_periodic=False,
        num_store_workers=2,
        num_lookup_workers=1,
        num_load_workers=2,
    )


def _config_dict(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "device_path": "/tmp/raw-block-test-device",
        "use_odirect": False,
    }
    config.update(overrides)
    return config


def test_raw_block_l2_adapter_config_default_io_engine():
    config = RawBlockL2AdapterConfig.from_dict(_config_dict())

    assert config.io_engine == "posix"


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_l2_adapter_config_accepts_io_engine_values(io_engine):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine=io_engine))

    assert config.io_engine == io_engine


def test_raw_block_l2_adapter_config_rejects_invalid_io_engine():
    with pytest.raises(ValueError, match="io_engine"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine="uring"))


def test_raw_block_l2_adapter_buffered_iouring_accepts_unaligned_l1() -> None:
    """Buffered io_uring does not require block-aligned L1 memory."""
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine="io_uring"))
    descriptor = L1MemoryDesc(ptr=0x4000, size=8192, align_bytes=1024)
    with patch(
        "lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter.RawBlockCore",
        autospec=True,
    ) as core_cls:
        core_cls.return_value.report_status.return_value = {}
        core_cls.return_value.snapshot_indexed_allocations.return_value = []
        adapter = RawBlockL2Adapter(config, descriptor)
        adapter.close()


@pytest.mark.parametrize("legacy_key", ["use_iouring", "use_uring"])
def test_raw_block_l2_adapter_config_legacy_use_uring_maps_to_iouring(legacy_key):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(**{legacy_key: True}))

    assert config.io_engine == "io_uring"


def test_raw_block_l2_adapter_config_explicit_io_engine_wins_over_legacy_flag():
    config = RawBlockL2AdapterConfig.from_dict(
        _config_dict(io_engine="posix", use_iouring=True)
    )

    assert config.io_engine == "posix"


def test_raw_block_l2_adapter_config_validates_iouring_queue_depth():
    with pytest.raises(ValueError, match="iouring_queue_depth"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(iouring_queue_depth=0))


@pytest.mark.parametrize("block_align", [0, -1, 3, 4095])
def test_raw_block_l2_adapter_config_rejects_non_power_of_2_block_align(
    block_align: int,
) -> None:
    """from_dict rejects invalid block_align values."""
    with pytest.raises(ValueError, match="block_align"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(block_align=block_align))


def _run_store(adapter: RawBlockL2Adapter, keys, objects) -> bool:
    task_id = adapter.submit_store_task(keys, objects)
    assert _wait_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id].is_successful()


def _run_lookup(adapter: RawBlockL2Adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
    return task_id, adapter.query_lookup_and_lock_result(task_id)


def _run_load(adapter: RawBlockL2Adapter, keys, objects):
    task_id = adapter.submit_load_task(keys, objects)
    assert _wait_event_fd(adapter.get_load_event_fd())
    return task_id, adapter.query_load_result(task_id)


def test_raw_block_l2_adapter_config_parses_uring_flags():
    cfg = RawBlockL2AdapterConfig.from_dict(
        {
            "type": "raw_block",
            "device_path": "/tmp/raw-block-dev",
            "slot_bytes": 64 * 1024,
            "use_odirect": False,
            "io_engine": "io_uring",
        }
    )

    assert cfg.io_engine == "io_uring"
    assert cfg.use_uring_cmd is False

    with pytest.raises(ValueError, match="use_uring_cmd requires io_uring"):
        RawBlockL2AdapterConfig.from_dict(
            {
                "type": "raw_block",
                "device_path": "/tmp/raw-block-dev",
                "slot_bytes": 64 * 1024,
                "use_uring_cmd": True,
            }
        )


def test_raw_block_l2_adapter_uring_cmd_rejects_regular_file():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        with pytest.raises(ValueError, match="NVMe namespace character device"):
            RawBlockL2Adapter(
                _make_config(
                    dev_path,
                    io_engine="io_uring",
                    use_uring_cmd=True,
                )
            )


@requires_raw_block_ext
def test_raw_block_l2_adapter_store_lookup_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key1 = _create_object_key(1)
            key_miss = _create_object_key(2)
            key3 = _create_object_key(3)
            obj1 = _create_memory_obj(fill_value=1.0)
            obj3 = _create_memory_obj(fill_value=3.0)

            assert _run_store(adapter, [key1, key3], [obj1, obj3]) is True

            lookup_task_id, lookup_bitmap = _run_lookup(
                adapter,
                [key1, key_miss, key3],
            )
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_lookup_and_lock_result(lookup_task_id) is None

            load_buffers = [
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
            ]
            load_task_id, load_bitmap = _run_load(
                adapter,
                [key1, key_miss, key3],
                load_buffers,
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_load_result(load_task_id) is None
            assert torch.equal(load_buffers[0].tensor, obj1.tensor)
            assert torch.equal(load_buffers[2].tensor, obj3.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            adapter.submit_unlock([key1, key_miss, key3])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_delete_respects_lock_until_unlock():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(11)
            obj = _create_memory_obj(fill_value=11.0)
            assert _run_store(adapter, [key], [obj]) is True

            _, bitmap = _run_lookup(adapter, [key])
            assert bitmap is not None
            assert bitmap.get_indices_list() == [0]

            adapter.delete([key])
            _, still_present = _run_lookup(adapter, [key])
            assert still_present is not None
            assert still_present.get_indices_list() == [0]
            adapter.submit_unlock([key, key])

            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
            adapter.submit_unlock([key])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_charges_a_key_repeated_in_a_batch_once():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)
        adapter = RawBlockL2Adapter(_make_config(dev_path))
        listener = _RecordingListener()
        adapter.register_listener(listener)
        try:
            key = _create_object_key(31)
            obj = _create_memory_obj(fill_value=31.0)
            assert _run_store(adapter, [key, key], [obj, obj]) is True
            used = adapter.get_usage().total_bytes_used
            assert used == adapter.report_status()["core"]["allocated_bytes"]
            adapter.delete([key])
            assert adapter.get_usage().total_bytes_used == 0
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_capacity_drops_when_deleted_space_is_retired():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)
        adapter = RawBlockL2Adapter(_make_config(dev_path, io_engine="posix"))
        real_write = PosixBlockIO.write_buffers

        def failing_header_write(self, offsets, *args, **kwargs):
            if len(offsets) == 1:
                raise OSError("injected")
            return real_write(self, offsets, *args, **kwargs)

        try:
            key = _create_object_key(41)
            assert _run_store(adapter, [key], [_create_memory_obj(fill_value=41.0)])
            before = adapter.get_usage()
            with patch.object(PosixBlockIO, "write_buffers", failing_header_write):
                adapter.delete([key])
            after = adapter.get_usage()
            assert after.total_bytes_used == 0
            assert after.total_capacity_bytes == (
                before.total_capacity_bytes - before.total_bytes_used
            )
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_usage_stays_truthful_while_stores_find_no_range():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)
        # Room for four default objects (8 KiB each) after the marker block.
        capacity_bytes = (1 * 1024 * 1024) + 4096 + 4 * 8 * 1024
        adapter = RawBlockL2Adapter(
            _make_config(dev_path, capacity_bytes=capacity_bytes)
        )
        try:
            keys = [_create_object_key(50 + i) for i in range(4)]
            for i, key in enumerate(keys):
                assert _run_store(adapter, [key], [_create_memory_obj(fill_value=i)])
            adapter.delete([keys[0], keys[2]])
            assert adapter.get_usage().usage_fraction == 0.5
            big = _create_memory_obj(size=3 * 1024, fill_value=9.0)  # 16 KiB object
            assert _run_store(adapter, [_create_object_key(60)], [big]) is False
            # Usage stays truthful although the store found no range.
            assert adapter.get_usage().usage_fraction == 0.5
            adapter.delete([keys[1]])
            assert adapter.get_usage().usage_fraction == 0.25
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_uses_global_eviction_accounting():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        # Room for exactly one default object (a 4 KiB header and 4 KiB
        # payload) after the 4 KiB format-marker block.
        object_bytes = 8 * 1024
        capacity_bytes = (1 * 1024 * 1024) + 4096 + object_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                capacity_bytes=capacity_bytes,
            )
        )
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key1 = _create_object_key(21)
            key2 = _create_object_key(22)
            obj1 = _create_memory_obj(fill_value=21.0)
            obj2 = _create_memory_obj(fill_value=22.0)

            assert _run_store(adapter, [key1], [obj1]) is True
            assert _run_store(adapter, [key2], [obj2]) is False

            assert listener.stored == [[key1]]
            assert listener.deleted == []

            usage = adapter.get_usage()
            assert usage.total_bytes_used == object_bytes
            assert usage.total_capacity_bytes == object_bytes
            assert 0.0 < usage.usage_fraction <= 1.0
            assert adapter.supports_global_eviction is True

            status = adapter.report_status()
            assert status["is_healthy"] is True
            assert status["type"] == "RawBlockL2Adapter"
            assert status["core"]["usable_capacity_bytes"] == object_bytes

            _, bitmap1 = _run_lookup(adapter, [key1])
            assert bitmap1 is not None
            assert bitmap1.get_indices_list() == [0]
            _, bitmap2 = _run_lookup(adapter, [key2])
            assert bitmap2 is not None
            assert bitmap2.get_indices_list() == []
            adapter.submit_unlock([key1, key2])

            adapter.delete([key1])
            assert listener.deleted[-1] == [key1]
            assert adapter.get_usage().total_bytes_used == 0

            assert _run_store(adapter, [key2], [obj2]) is True
            assert listener.stored[-1] == [key2]
            _, bitmap_after_delete = _run_lookup(adapter, [key1, key2])
            assert bitmap_after_delete is not None
            assert bitmap_after_delete.get_indices_list() == [1]
            adapter.submit_unlock([key1, key2])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_does_not_notify_duplicate_store():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key = _create_object_key(25)
            obj = _create_memory_obj(fill_value=25.0)

            assert _run_store(adapter, [key], [obj]) is True
            assert _run_store(adapter, [key], [obj]) is True

            assert listener.stored == [[key]]
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_listener_errors_do_not_block_eventfds():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        adapter.register_listener(_FailingListener())

        try:
            key = _create_object_key(29)
            obj = _create_memory_obj(fill_value=29.0)

            store_task_id = adapter.submit_store_task([key], [obj])
            assert _wait_event_fd(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_task_id].is_successful()

            load_buffer = _create_memory_obj(fill_value=0.0)
            load_task_id = adapter.submit_load_task([key], [load_buffer])
            assert _wait_event_fd(adapter.get_load_event_fd())
            load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_from_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(31)
        obj = _create_memory_obj(fill_value=31.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            _, lookup_bitmap = _run_lookup(adapter2, [key])
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0]

            load_buffer = _create_memory_obj(fill_value=0.0)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert torch.equal(load_buffer.tensor, obj.tensor)
            adapter2.submit_unlock([key])
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_seeds_usage_by_cache_salt():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        object_bytes = 8 * 1024  # 4 KiB header + 4 KiB payload
        config = _make_config(dev_path)
        key = _create_object_key(33, cache_salt="u1")
        obj = _create_memory_obj(fill_value=33.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == object_bytes
            assert dict(usage.bytes_by_cache_salt) == {"u1": object_bytes}

            adapter2.delete([key])
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == 0
            assert dict(usage.bytes_by_cache_salt) == {}
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovered_keys_seed_l2_eviction_state():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(34)
        obj = _create_memory_obj(fill_value=34.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            state = L2AdapterEvictionState(
                0,
                adapter2,
                EvictionConfig(eviction_policy="LRU", eviction_ratio=1.0),
            )
            assert state.eviction_policy.get_eviction_candidates(1) == [key]

            actions = state.eviction_policy.get_eviction_actions(1.0)
            assert len(actions) == 1
            assert actions[0].keys == [key]
            adapter2.delete(actions[0].keys)

            assert adapter2.get_usage().total_bytes_used == 0
            assert state.eviction_policy.get_eviction_candidates(1) == []
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovers_unknown_checkpoint_dtype():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(35)
        obj = _create_complex_memory_obj(fill_value=1 + 2j)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            load_buffer = _create_complex_memory_obj(fill_value=0j)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert load_buffer.metadata.dtype is torch.complex64
            assert torch.equal(load_buffer.tensor, obj.tensor)
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_error_bitmaps_keep_submitted_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            keys = [_create_object_key(41), _create_object_key(42)]
            objects = [_create_memory_obj(), _create_memory_obj()]

            with patch.object(
                adapter, "_run_lookup_task", side_effect=RuntimeError("lookup failed")
            ):
                lookup_task_id = adapter.submit_lookup_and_lock_task(
                    keys, {0: _EMPTY_LAYOUT}
                )
                assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
                lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
            assert lookup_bitmap is not None
            assert str(lookup_bitmap) == "00"

            with patch.object(
                adapter, "_run_load_task", side_effect=RuntimeError("load failed")
            ):
                load_task_id = adapter.submit_load_task(keys, objects)
                assert _wait_event_fd(adapter.get_load_event_fd())
                load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert str(load_bitmap) == "00"
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_uecc_preserves_healthy_slots(
    tmp_path: Path,
) -> None:
    """A payload UECC retires its slot while healthy siblings continue serving."""
    fake = _RecordingRawDevice(size=8 * 1024 * 1024)
    config = _make_config(
        str(tmp_path / "device"),
        capacity_bytes=fake.size,
        io_engine="io_uring",
        use_uring_cmd=True,
        max_data_transfer_size=4096,
    )

    def make_engine(cfg: RawBlockIOConfig) -> IoUringBlockIO:
        assert isinstance(cfg, IoUringBlockIOConfig)
        engine = IoUringBlockIO.from_config(cfg)
        engine._raw = fake
        return engine

    with (
        patch.object(IoUringBlockIO, "validate_uring_cmd_device"),
        patch.object(raw_block_core, "make_raw_block_io", make_engine),
    ):
        adapter = RawBlockL2Adapter(config)
    try:
        keys = [_create_object_key(51), _create_object_key(52)]
        encoded = [encode_object_key(key).encoded for key in keys]
        sizes = [4096, 1024]  # float32 elements: four and one 4 KiB chunks
        objects: list[MemoryObj] = [
            _create_memory_obj(size=size, fill_value=1.0) for size in sizes
        ]
        assert _run_store(adapter, keys, objects)
        assert adapter.pop_completed_store_tasks() == {}
        core = adapter._core
        entry_offset = core.entry_offset(encoded[0])
        assert entry_offset is not None
        failing_offset = entry_offset + core.header_bytes + 8192
        destinations = [_create_memory_obj(size=size) for size in sizes]

        listener = _RecordingListener()
        adapter.register_listener(listener)
        for uecc in (False, True):
            lookup_id, hits = _run_lookup(adapter, keys)
            assert str(hits) == "11"
            assert adapter.query_lookup_and_lock_result(lookup_id) is None
            assert [core.lock_refcount(key) for key in encoded] == [1, 1]
            if uecc:
                fake.uecc_on_next_wait_batch = True
            else:
                fake.fail_completion_entries.add(2)
            load_id, loaded = _run_load(adapter, keys, destinations)
            assert fake.last_read_lens == [4096] * 5
            assert fake.last_read_offsets[2] == failing_offset
            assert str(loaded) == "01"
            assert adapter.query_load_result(load_id) is None
            adapter.submit_unlock(keys)
            assert [core.lock_refcount(key) for key in encoded] == [0, 0]
            assert adapter.report_status()["is_healthy"]
            if not uecc:
                assert listener.deleted == []
                fake.fail_completion_entries.clear()
        # Each key is charged its header plus payload: 20 KiB and 8 KiB.
        assert listener.deleted == [[keys[0]]]
        assert adapter.get_usage().total_bytes_used == 8 * 1024
        lookup_id, hits = _run_lookup(adapter, keys)
        assert str(hits) == "01"
        assert adapter.query_lookup_and_lock_result(lookup_id) is None
        load_id, loaded = _run_load(adapter, keys[1:], destinations[1:])
        assert str(loaded) == "1"
        assert adapter.query_load_result(load_id) is None
        adapter.submit_unlock(keys[1:])
        assert _run_store(adapter, keys[:1], objects[:1])
        assert core.entry_offset(encoded[0]) not in (None, entry_offset)
        assert adapter.get_usage().total_bytes_used == 28 * 1024
        _, restored = _run_load(adapter, keys, destinations)
        assert str(restored) == "11"
    finally:
        adapter.close()


def test_raw_block_l2_adapter_config_accepts_and_ignores_slot_bytes():
    """Configs written for fixed slots keep parsing; slot_bytes no longer
    limits object size."""
    with patch.object(raw_block_l2_adapter.logger, "warning") as warn:
        config = RawBlockL2AdapterConfig.from_dict(_config_dict(slot_bytes=4096))
    assert "slot_bytes is deprecated" in warn.call_args.args[0]
    assert not hasattr(config, "slot_bytes")


@requires_raw_block_ext
def test_raw_block_l2_adapter_stores_objects_larger_than_a_legacy_slot():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)
        config = RawBlockL2AdapterConfig.from_dict(
            _config_dict(
                device_path=dev_path,
                slot_bytes=64 * 1024,
                meta_total_bytes=1024 * 1024,
                meta_enable_periodic=False,
            )
        )
        adapter = RawBlockL2Adapter(config)
        try:
            key = _create_object_key(70)
            obj = _create_memory_obj(size=64 * 1024, fill_value=7.0)  # 256 KiB
            assert _run_store(adapter, [key], [obj]) is True
            assert adapter.get_usage().total_bytes_used == 4096 + 256 * 1024
            dest = _create_memory_obj(size=64 * 1024)
            _, loaded = _run_load(adapter, [key], [dest])
            assert str(loaded) == "1"
            assert torch.equal(dest.tensor, obj.tensor)
        finally:
            adapter.close()
