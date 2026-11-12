# SPDX-License-Identifier: Apache-2.0
"""FS L2 adapter capacity: ``max_capacity_gb`` declares the capacity, so usage
accounting and capacity eviction work as for fs_native; without it the adapter
stays unbounded, as before."""

# Standard
from pathlib import Path
from typing import cast
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_management import MemoryObj


class _Buf:
    def __init__(self, data: bytes) -> None:
        self._data = bytearray(data)

    @property
    def byte_array(self) -> memoryview:
        return memoryview(self._data)


def _key(h: bytes) -> ObjectKey:
    return ObjectKey(chunk_hash=h, model_name="llama", kv_rank=0, cache_salt="")


def _adapter(tmp_path: Path, **extra) -> FSL2Adapter:
    return FSL2Adapter(
        FSL2AdapterConfig.from_dict({"base_path": str(tmp_path), **extra})
    )


def _store_and_wait(adp: FSL2Adapter, keys, payloads) -> None:
    task_id = adp.submit_store_task(
        keys, cast("list[MemoryObj]", [_Buf(p) for p in payloads])
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        done = adp.pop_completed_store_tasks()
        if task_id in done:
            assert int(done[task_id]) >= 0
            return
        time.sleep(0.01)
    pytest.fail("store task did not complete within 5s")


def test_default_is_unbounded(tmp_path: Path) -> None:
    adp = _adapter(tmp_path)
    try:
        assert (
            FSL2AdapterConfig.from_dict({"base_path": str(tmp_path)}).max_capacity_gb
            == 0
        )
        assert adp.supports_global_eviction is False
        assert (
            StorageManager._should_enable_l2_eviction(
                None, adp, EvictionConfig(eviction_policy="LRU")
            )
            is False
        )
    finally:
        adp.close()


def test_capacity_enables_global_eviction(tmp_path: Path) -> None:
    adp = _adapter(tmp_path, max_capacity_gb=3840)
    try:
        assert adp.supports_global_eviction is True
        assert adp.get_usage().total_capacity_bytes == 3840 * 1024**3
        assert (
            StorageManager._should_enable_l2_eviction(
                None, adp, EvictionConfig(eviction_policy="LRU")
            )
            is True
        )
    finally:
        adp.close()


def test_usage_tracks_stored_and_deleted_bytes(tmp_path: Path) -> None:
    adp = _adapter(tmp_path, max_capacity_gb=1e-6)  # ~1073 bytes
    try:
        k1, k2 = _key(b"\x01" * 4), _key(b"\x02" * 4)
        _store_and_wait(adp, [k1, k2], [b"x" * 300, b"y" * 200])
        u = adp.get_usage()
        assert u.total_bytes_used == 500
        assert u.total_capacity_bytes == int(1e-6 * 1024**3)
        adp.delete([k1])
        assert adp.get_usage().total_bytes_used == 200
    finally:
        adp.close()


@pytest.mark.parametrize("bad", [-1, "10", True])
def test_invalid_capacity_rejected(tmp_path: Path, bad) -> None:
    with pytest.raises(ValueError):
        FSL2AdapterConfig.from_dict(
            {"base_path": str(tmp_path), "max_capacity_gb": bad}
        )
