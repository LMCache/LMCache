# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend

# A chunk whose allocation carries alignment padding: `byte_array` exposes the
# logical bytes, while `get_physical_size` reports the whole allocation. The two
# differ, so `usage` can only stay right if it counts one of them on both the
# way in and the way out.
LOGICAL_SIZE = 1000
PHYSICAL_SIZE = 1024


def _create_key(key_id: int) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


def _create_backend(tmp_path) -> LocalDiskBackend:
    backend = object.__new__(LocalDiskBackend)
    backend.dict = {}
    backend.usage = 0
    backend.current_cache_size = 0
    backend.max_cache_size = 1 << 30
    backend.disk_lock = threading.Lock()
    backend.cache_policy = MagicMock()
    backend.stats_monitor = MagicMock()
    backend.batched_msg_sender = None
    backend.disk_worker = MagicMock()
    backend.write_file = MagicMock()  # type: ignore[method-assign]
    backend._key_to_path = lambda key, **kwargs: str(  # type: ignore[method-assign]
        tmp_path / "chunk.bin"
    )
    return backend


def _create_memory_obj(logical: int, physical: int) -> MemoryObj:
    memory_obj = MagicMock(spec=MemoryObj)
    memory_obj.tensor = torch.empty(1, dtype=torch.uint8)
    memory_obj.byte_array = memoryview(bytearray(logical))
    memory_obj.get_physical_size.return_value = physical
    memory_obj.metadata = SimpleNamespace(
        shape=torch.Size([logical]),
        dtype=torch.uint8,
        fmt=MemoryFormat.KV_2LTD,
        cached_positions=None,
    )
    return memory_obj


def test_usage_counts_the_size_remove_gives_back(tmp_path) -> None:
    backend = _create_backend(tmp_path)
    key = _create_key(1)

    backend.async_save_bytes_to_disk(
        key, _create_memory_obj(LOGICAL_SIZE, PHYSICAL_SIZE)
    )

    # The physical size, which is what the metadata records, what `remove`
    # gives back and what `submit_put_task` admitted. Pinned as a number too,
    # so that moving both sides to the logical size does not quietly satisfy
    # this while leaving the admission accounting counting something else.
    assert backend.usage == PHYSICAL_SIZE
    assert backend.dict[key].size == PHYSICAL_SIZE
    backend.stats_monitor.update_local_storage_usage.assert_called_with(PHYSICAL_SIZE)


def test_usage_returns_to_zero_after_put_then_remove(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("os.remove", lambda path: None)
    backend = _create_backend(tmp_path)
    key = _create_key(1)

    backend.async_save_bytes_to_disk(
        key, _create_memory_obj(LOGICAL_SIZE, PHYSICAL_SIZE)
    )
    assert backend.remove(key) is True

    assert backend.usage == 0
    backend.stats_monitor.update_local_storage_usage.assert_called_with(0)


def test_usage_returns_to_zero_over_repeated_put_remove_cycles(
    tmp_path, monkeypatch
) -> None:
    """The drift accumulated, so one cycle understated it."""
    monkeypatch.setattr("os.remove", lambda path: None)
    backend = _create_backend(tmp_path)

    for key_id in range(5):
        key = _create_key(key_id)
        backend.async_save_bytes_to_disk(
            key, _create_memory_obj(LOGICAL_SIZE, PHYSICAL_SIZE)
        )
        assert backend.remove(key) is True

    assert backend.usage == 0
