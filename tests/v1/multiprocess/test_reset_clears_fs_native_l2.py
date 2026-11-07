# SPDX-License-Identifier: Apache-2.0
"""The reset RPC (ManagementModule.clear) must drop KV stored in fs_native L2.

Regression: the RPC cleared L1 only, so after an RL weight update prefetch
promoted old-weight KV from L2 back into L1.
"""

# Standard
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.fs_native_l2_adapter import (
    FSNativeL2AdapterConfig,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from tests.v1.distributed.utils import should_use_lazy_alloc, single_row_spec

if not torch_dev.is_available():
    pytest.skip(
        f"Requires available {torch_device_type} runtime", allow_module_level=True
    )
pytest.importorskip("lmcache.lmcache_fs")

# First Party
from lmcache.v1.distributed.storage_manager import (  # noqa: E402
    StorageManager,
    l1_owners_of,
)

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([100, 2, 512])], dtypes=[torch.bfloat16])


def _wait(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _prefetch_hits(sm: StorageManager, keys: list[ObjectKey]) -> int:
    handle = sm.submit_prefetch_task(single_row_spec(keys, _LAYOUT))
    hits: list[int] = []
    assert _wait(lambda: _poll(sm, handle, hits))
    return hits[0]


def _poll(sm: StorageManager, handle: Any, out: list[int]) -> bool:
    result = sm.query_prefetch_status(handle)
    if result is None:
        return False
    out.append(result.hit_cells[0].count_leading_ones())
    return True


def test_reset_rpc_drops_kv_stored_in_fs_native_l2(tmp_path: Path) -> None:
    sm = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=should_use_lazy_alloc(),
                    init_size_in_bytes=64 * 1024 * 1024,
                    align_bytes=0x1000,
                ),
                write_ttl_seconds=600,
                read_ttl_seconds=300,
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[FSNativeL2AdapterConfig(base_path=str(tmp_path))]
            ),
        )
    )
    try:
        keys = [
            ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)
            for i in range(4)
        ]
        reserved = sm.reserve_write(keys, _LAYOUT)
        sm.finish_write(list(reserved), l1_owners_of(reserved))
        l1 = sm.l1_managers()[0]
        assert _wait(
            lambda: (
                len(list(tmp_path.glob("*.data"))) == len(keys)
                and all(l1.is_key_evictable(k) for k in keys)
            )
        )
        l1.delete(keys)  # the old-weight KV now lives only in L2
        assert _prefetch_hits(sm, keys) == len(keys)
        sm.finish_read_prefetched(keys)  # the engine has consumed the prefetch
        l1.delete(keys)

        ManagementModule(MagicMock(storage_manager=sm)).clear()

        assert _prefetch_hits(sm, keys) == 0
        assert list(tmp_path.glob("*.data")) == []
    finally:
        sm.close()
