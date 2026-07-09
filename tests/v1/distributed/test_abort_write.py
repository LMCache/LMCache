# SPDX-License-Identifier: Apache-2.0
"""
Tests for abort_write: releasing write reservations without publishing them.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.server_transfer import PickleTransferStrategy
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

try:
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager
except ImportError:
    pytest.skip(
        "Skipping because StorageManager cannot be imported", allow_module_level=True
    )

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)


@pytest.fixture
def basic_layout():
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 2, 16, 16])],
        dtypes=[torch.bfloat16],
    )


@pytest.fixture
def storage_manager():
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=128 * 1024 * 1024,
                use_lazy=torch.cuda.is_available(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )
    sm = StorageManager(config)
    yield sm
    sm.close()


def make_object_key(chunk_hash: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_hash),
        model_name="test_model",
        kv_rank=0,
    )


def _locked_and_total(sm: StorageManager) -> tuple[int, int]:
    status = sm.report_status()["l1_manager"]
    return status["write_locked_count"], status["total_object_count"]


def test_abort_write_releases_locks_and_frees_objects(storage_manager, basic_layout):
    keys = [make_object_key(i) for i in range(3)]
    reserved = storage_manager.reserve_write(keys, basic_layout, "new")
    assert len(reserved) == 3
    assert _locked_and_total(storage_manager) == (3, 3)

    storage_manager.abort_write(keys)

    assert _locked_and_total(storage_manager) == (0, 0)
    # Aborted keys must be reservable again as "new".
    reserved_again = storage_manager.reserve_write(keys, basic_layout, "new")
    assert len(reserved_again) == 3
    storage_manager.abort_write(keys)


def test_abort_write_does_not_publish_to_store_listeners(
    storage_manager, basic_layout
):
    events: list[list[ObjectKey]] = []

    class _Listener:
        def on_l1_keys_reserved_read(self, keys):
            pass

        def on_l1_keys_read_finished(self, keys):
            pass

        def on_l1_keys_reserved_write(self, keys):
            pass

        def on_l1_keys_write_finished(self, keys):
            events.append(list(keys))

        def on_l1_keys_finish_write_and_reserve_read(self, keys):
            pass

        def on_l1_keys_deleted_by_manager(self, keys):
            pass

        def on_l1_keys_accessed(self, keys):
            pass

        def on_l2_keys_stored(self, keys, sizes):
            pass

        def on_l2_keys_accessed(self, keys):
            pass

        def on_l2_keys_deleted(self, keys):
            pass

    storage_manager._l1_manager.register_listener(_Listener())
    keys = [make_object_key(10)]
    storage_manager.reserve_write(keys, basic_layout, "new")
    storage_manager.abort_write(keys)

    assert events == [] or all(not e for e in events)


def test_l1_abort_write_rejects_unlocked_keys(storage_manager, basic_layout):
    keys = [make_object_key(20)]
    storage_manager.reserve_write(keys, basic_layout, "new")
    storage_manager.finish_write(keys)

    result = storage_manager._l1_manager.abort_write(keys)
    assert result[keys[0]] == L1Error.KEY_IN_WRONG_STATE

    missing = [make_object_key(21)]
    result = storage_manager._l1_manager.abort_write(missing)
    assert result[missing[0]] == L1Error.KEY_NOT_EXIST


def test_pickle_commit_store_aborts_skipped_reservations(
    storage_manager, basic_layout
):
    """A payload with fewer chunks than object keys must not leak the extra
    write reservations (regression: leaked write locks filled L1 and every
    subsequent reserve failed)."""
    # Standard
    import pickle

    strategy = PickleTransferStrategy(storage_manager)
    obj_keys = [make_object_key(30 + i) for i in range(3)]
    metadata = EngineDrivenContextMetadata(
        layout_desc=basic_layout, block_size=16, use_mla=False
    )
    key = IPCCacheServerKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        token_ids=tuple(range(48)),
        start=0,
        end=48,
        request_id="req-abort-test",
    )

    # Only one chunk for three keys: two reservations must be aborted.
    payload = pickle.dumps([torch.ones(2, 2, 16, 16, dtype=torch.bfloat16)])
    ok = strategy.commit_store(
        key=key,
        instance_id=1,
        cpu_data=payload,
        context=metadata,
        resolve_obj_keys=lambda _k: obj_keys,
    )

    assert ok is False
    locked, total = _locked_and_total(storage_manager)
    assert locked == 0
    # Only the written key may remain resident.
    assert total <= 1
