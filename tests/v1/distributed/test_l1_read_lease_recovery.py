# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for expired L1 read-lease recovery."""

# Standard
from unittest.mock import MagicMock, patch

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager


def test_tp4_expired_read_lease_is_recovered_exactly_once() -> None:
    """Two TP readers restore one four-reader reservation, not two."""
    memory_obj = MagicMock(name="memory_obj")
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    layout = MemoryLayoutDesc(
        shapes=[torch.Size([1])],
        dtypes=[torch.bfloat16],
    )
    key = ObjectKey(
        chunk_hash=b"lease-recovery",
        model_name="test-model",
        kv_rank=0,
    )

    with patch(
        "lmcache.v1.distributed.l1_manager.L1MemoryManager"
    ) as memory_manager_cls:
        memory_manager_cls.return_value.allocate.return_value = (
            L1Error.SUCCESS,
            [memory_obj],
        )
        manager = L1Manager(config)

        manager.reserve_write([key], [False], layout)
        manager.finish_write([key])
        manager.reserve_read([key], extra_count=3)

        # Releasing the reservation produces the same persistent, readable,
        # unlocked state that remains after TTLLock's read TTL expires.
        manager.finish_read([key], extra_count=3)
        assert manager.unsafe_read([key])[key][0] == L1Error.KEY_NOT_READABLE

        first = manager.unsafe_read([key], recover_expired=True, extra_count=3)
        second = manager.unsafe_read([key], recover_expired=True, extra_count=3)
        assert first[key] == (L1Error.SUCCESS, memory_obj)
        assert second[key] == (L1Error.SUCCESS, memory_obj)

        # Four TP completions exhaust the recovered lease. If the second read
        # had restored it again, another four claims would still be held.
        for _ in range(4):
            assert manager.finish_read([key])[key] == L1Error.SUCCESS
        assert manager.unsafe_read([key])[key][0] == L1Error.KEY_NOT_READABLE

        manager.close()
