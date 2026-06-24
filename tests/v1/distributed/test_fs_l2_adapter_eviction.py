# SPDX-License-Identifier: Apache-2.0
"""
Tests for capacity-aware eviction in the ``fs`` L2 adapter.

The ``fs`` adapter participates in the shared L2 eviction framework like
``native_connector``: it declares a capacity, reports usage via the base
``_notify_keys_*`` helpers, and implements ``delete`` to unlink files.
These tests verify the consistency guarantees that wiring relies on:

* byte accounting tracks store/delete exactly once per key (no double
  count under dedup),
* ``delete`` removes files, frees accounting, and notifies listeners,
* load touches keys so an LRU policy keeps them warm,
* startup recovery accounts for and seeds pre-existing files,
* a real LRU policy drives end-to-end eviction of the right victims, and
* deleting a file concurrently with a read never corrupts data (POSIX).
"""

# Standard
from pathlib import Path
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.eviction import L2EvictionPolicy
from lmcache.v1.distributed.eviction_policy import CreateEvictionPolicy
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter,
    FSL2AdapterConfig,
    _object_key_to_filename,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

# Bytes per element of the float32 payloads used below.
_F32 = 4

# The fs adapter ignores the layout descriptor in lookup, so an empty one
# is sufficient to drive ``submit_lookup_and_lock_task`` in tests.
_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])


def _gb(num_bytes: int) -> float:
    """Convert a byte budget to the GB unit the config expects."""
    return num_bytes / (1024**3)


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(num_elems: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    """A float32 ``TensorMemoryObj`` of ``num_elems * 4`` bytes."""
    raw_data = torch.empty(num_elems, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([num_elems]),
        dtype=torch.float32,
        address=0,
        phy_size=num_elems * _F32,
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


def store_blocking(
    adapter: FSL2Adapter,
    keys: list[ObjectKey],
    objs: list[MemoryObj],
    timeout: float = 5.0,
) -> L2StoreResult:
    """Submit a store task and block until it completes."""
    task_id = adapter.submit_store_task(keys, objs)
    assert wait_for_event_fd(adapter.get_store_event_fd(), timeout)
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id]


def load_blocking(
    adapter: FSL2Adapter,
    keys: list[ObjectKey],
    objs: list[MemoryObj],
    timeout: float = 5.0,
) -> Bitmap:
    """Submit a load task and block until it completes; return the bitmap."""
    task_id = adapter.submit_load_task(keys, objs)
    assert wait_for_event_fd(adapter.get_load_event_fd(), timeout)
    bitmap = adapter.query_load_result(task_id)
    assert bitmap is not None
    return bitmap


def lookup_lock_blocking(
    adapter: FSL2Adapter,
    keys: list[ObjectKey],
    timeout: float = 5.0,
) -> Bitmap:
    """Submit a lookup-and-lock task and block until it completes."""
    task_id = adapter.submit_lookup_and_lock_task(keys, _EMPTY_LAYOUT)
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd(), timeout)
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return bitmap


class RecordingListener:
    """L2 adapter listener that records every callback invocation."""

    def __init__(self) -> None:
        self.stored: list[tuple[list[ObjectKey], list[int]]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.append((list(keys), list(sizes)))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.append(list(keys))


@pytest.fixture
def make_adapter(tmp_path):
    """Factory that builds adapters and closes them at teardown."""
    created: list[FSL2Adapter] = []

    def _make(**kwargs) -> FSL2Adapter:
        kwargs.setdefault("base_path", str(tmp_path / "fsl2"))
        adapter = FSL2Adapter(FSL2AdapterConfig(**kwargs))
        created.append(adapter)
        return adapter

    yield _make

    for adapter in created:
        adapter.close()


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_from_dict_defaults(self):
        cfg = FSL2AdapterConfig.from_dict({"base_path": "/tmp/x"})
        assert cfg.max_capacity_gb == 0
        assert cfg.recover_on_start is True

    def test_from_dict_capacity_and_recover(self):
        cfg = FSL2AdapterConfig.from_dict(
            {"base_path": "/tmp/x", "max_capacity_gb": 2, "recover_on_start": False}
        )
        assert cfg.max_capacity_gb == 2.0
        assert cfg.recover_on_start is False

    def test_from_dict_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="max_capacity_gb"):
            FSL2AdapterConfig.from_dict({"base_path": "/tmp/x", "max_capacity_gb": -1})

    def test_from_dict_invalid_capacity_type_raises(self):
        with pytest.raises(ValueError, match="max_capacity_gb"):
            FSL2AdapterConfig.from_dict(
                {"base_path": "/tmp/x", "max_capacity_gb": "big"}
            )

    def test_from_dict_invalid_recover_type_raises(self):
        with pytest.raises(ValueError, match="recover_on_start"):
            FSL2AdapterConfig.from_dict(
                {"base_path": "/tmp/x", "recover_on_start": "yes"}
            )

    def test_help_mentions_capacity(self):
        assert "max_capacity_gb" in FSL2AdapterConfig.help()


# ---------------------------------------------------------------------------
# Capacity reporting
# ---------------------------------------------------------------------------


class TestCapacityReporting:
    def test_no_capacity_disables_eviction(self, make_adapter):
        adapter = make_adapter()
        assert adapter.supports_global_eviction is False
        assert adapter.get_usage().usage_fraction == -1.0

    def test_capacity_enables_eviction(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(4096))
        assert adapter.supports_global_eviction is True
        assert adapter.get_usage().total_capacity_bytes > 0


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


class TestUsageAccounting:
    def test_store_increases_usage(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(256) for _ in range(3)]
        result = store_blocking(adapter, keys, objs)
        assert result.is_successful()

        expected = sum(len(o.byte_array) for o in objs)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == expected
        assert usage.usage_fraction == pytest.approx(
            expected / usage.total_capacity_bytes
        )

    def test_dedup_does_not_double_count(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        key = create_object_key(0)
        size = len(create_memory_obj(256).byte_array)

        store_blocking(adapter, [key], [create_memory_obj(256)])
        assert adapter.get_usage().total_bytes_used == size

        # Re-storing the same key skips the write (already on disk) and
        # must not inflate accounting.
        store_blocking(adapter, [key], [create_memory_obj(256)])
        assert adapter.get_usage().total_bytes_used == size

    def test_delete_decreases_usage_and_removes_files(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(256) for _ in range(3)]
        store_blocking(adapter, keys, objs)
        size = len(objs[0].byte_array)

        adapter.delete([keys[0], keys[2]])

        usage = adapter.get_usage()
        assert usage.total_bytes_used == size  # only keys[1] remains
        assert not (base / _object_key_to_filename(keys[0])).exists()
        assert (base / _object_key_to_filename(keys[1])).exists()
        assert not (base / _object_key_to_filename(keys[2])).exists()

    def test_delete_unknown_key_is_noop(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        # Never stored -- delete must not raise or drive accounting negative.
        adapter.delete([create_object_key(99)])
        assert adapter.get_usage().total_bytes_used == 0

    def test_no_capacity_store_stays_stateless(self, make_adapter):
        # With eviction disabled (max_capacity_gb=0, the default) the store
        # path must not accumulate per-key accounting -- the fs adapter keeps
        # its historic stateless, zero-overhead behaviour.
        adapter = make_adapter()  # capacity 0
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(256) for _ in range(3)]
        store_blocking(adapter, keys, objs)

        usage = adapter.get_usage()
        assert usage.total_bytes_used == 0
        assert usage.usage_fraction == -1.0


# ---------------------------------------------------------------------------
# Listener notifications
# ---------------------------------------------------------------------------


class TestListenerNotifications:
    def test_store_notifies_listener_once(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        listener = RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(0)
        store_blocking(adapter, [key], [create_memory_obj(256)])
        store_blocking(adapter, [key], [create_memory_obj(256)])  # dedup

        flat = [k for batch, _ in listener.stored for k in batch]
        assert flat.count(key) == 1

    def test_load_notifies_accessed_for_hits_only(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        listener = RecordingListener()
        adapter.register_listener(listener)

        stored_key = create_object_key(0)
        missing_key = create_object_key(1)
        store_blocking(adapter, [stored_key], [create_memory_obj(256)])

        dst = [create_memory_obj(256), create_memory_obj(256)]
        load_blocking(adapter, [stored_key, missing_key], dst)

        accessed = [k for batch in listener.accessed for k in batch]
        assert stored_key in accessed
        assert missing_key not in accessed

    def test_delete_notifies_deleted(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        listener = RecordingListener()
        adapter.register_listener(listener)

        key = create_object_key(0)
        store_blocking(adapter, [key], [create_memory_obj(256)])
        adapter.delete([key])

        deleted = [k for batch in listener.deleted for k in batch]
        assert deleted == [key]


# ---------------------------------------------------------------------------
# Startup recovery
# ---------------------------------------------------------------------------


def _seed_file(base_dir: Path, key: ObjectKey, num_bytes: int) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / _object_key_to_filename(key)).write_bytes(b"\xab" * num_bytes)


class TestRecovery:
    def test_recovery_accounts_existing_files(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        keys = [create_object_key(i) for i in range(2)]
        for key in keys:
            _seed_file(base, key, 1024)

        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        assert adapter.get_usage().total_bytes_used == 2048

    def test_recovery_seeds_eviction_policy(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        key = create_object_key(0)
        _seed_file(base, key, 1024)

        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="LRU"))
        adapter.register_listener(L2EvictionPolicy(policy))

        # The recovered file is known to the policy and is evictable.
        actions = policy.get_eviction_actions(1.0)
        evictable = [k for action in actions for k in action.keys]
        assert key in evictable

    def test_recovery_does_not_seed_non_eviction_listener(self, tmp_path, make_adapter):
        # A non-eviction listener (e.g. the coordinator event reporter) must
        # NOT receive recovered files as fresh store events, which would
        # double-report the warm cache to the fleet on every restart.
        base = tmp_path / "fsl2"
        _seed_file(base, create_object_key(0), 1024)

        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        listener = RecordingListener()
        adapter.register_listener(listener)

        assert listener.stored == []

    def test_recovery_skips_foreign_files(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        key = create_object_key(0)
        _seed_file(base, key, 1024)
        (base / "not-a-kv-file.txt").write_bytes(b"junk")
        (base / "malformed.data").write_bytes(b"junk")

        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        # Only the one valid file counts.
        assert adapter.get_usage().total_bytes_used == 1024

    def test_recover_on_start_false_skips_scan(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        _seed_file(base, create_object_key(0), 1024)

        adapter = make_adapter(
            base_path=str(base),
            max_capacity_gb=_gb(1 << 20),
            recover_on_start=False,
        )
        assert adapter.get_usage().total_bytes_used == 0

    def test_no_capacity_skips_recovery(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        _seed_file(base, create_object_key(0), 1024)

        adapter = make_adapter(base_path=str(base))  # cap 0
        # Eviction disabled -> no usage signal, no recovery cost.
        assert adapter.get_usage().usage_fraction == -1.0
        assert adapter.get_usage().total_bytes_used == 0


# ---------------------------------------------------------------------------
# End-to-end eviction with a real LRU policy
# ---------------------------------------------------------------------------


class TestLRUEvictionIntegration:
    def test_lru_evicts_least_recently_used_victims(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        policy = CreateEvictionPolicy(EvictionConfig(eviction_policy="LRU"))
        adapter.register_listener(L2EvictionPolicy(policy))

        keys = [create_object_key(i) for i in range(5)]
        for key in keys:
            store_blocking(adapter, [key], [create_memory_obj(256)])

        # Touch keys[0] via a load so it becomes most-recently-used and
        # must survive eviction even though it was stored first.
        load_blocking(adapter, [keys[0]], [create_memory_obj(256)])

        before = adapter.get_usage().total_bytes_used
        actions = policy.get_eviction_actions(0.4)
        victims = [k for action in actions for k in action.keys]
        assert victims, "LRU should select victims to evict"
        assert keys[0] not in victims  # protected by the recent load

        adapter.delete(victims)

        # Accounting dropped, victim files gone, survivors intact.
        per_obj = len(create_memory_obj(256).byte_array)
        usage = adapter.get_usage()
        assert usage.total_bytes_used == before - per_obj * len(victims)
        for victim in victims:
            assert not (base / _object_key_to_filename(victim)).exists()
        assert (base / _object_key_to_filename(keys[0])).exists()

        # keys[0] still loads with the correct payload.
        dst = create_memory_obj(256, fill_value=0.0)
        bitmap = load_blocking(adapter, [keys[0]], [dst])
        assert bitmap.test(0)
        assert torch.equal(dst.raw_data, torch.ones(256, dtype=torch.float32))


# ---------------------------------------------------------------------------
# POSIX delete-during-read safety
# ---------------------------------------------------------------------------


class TestEvictionLocking:
    def test_locked_key_survives_eviction_then_unlock_frees_it(
        self, tmp_path, make_adapter
    ):
        base = tmp_path / "fsl2"
        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        key = create_object_key(0)
        store_blocking(adapter, [key], [create_memory_obj(256)])
        size = len(create_memory_obj(256).byte_array)

        # lookup_and_lock pins the key; a concurrent eviction must skip it.
        assert lookup_lock_blocking(adapter, [key]).test(0)
        adapter.delete([key])
        assert adapter.get_usage().total_bytes_used == size
        assert (base / _object_key_to_filename(key)).exists()

        # After unlock the key is evictable again.
        adapter.submit_unlock([key])
        adapter.delete([key])
        assert adapter.get_usage().total_bytes_used == 0
        assert not (base / _object_key_to_filename(key)).exists()

    def test_store_in_progress_lock_is_balanced(self, tmp_path, make_adapter):
        # After a completed store the key holds no residual lock, so it is
        # immediately evictable (the store lock is released in finally).
        base = tmp_path / "fsl2"
        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        key = create_object_key(0)
        store_blocking(adapter, [key], [create_memory_obj(256)])

        adapter.delete([key])
        assert adapter.get_usage().total_bytes_used == 0


class TestHardening:
    def test_store_accounts_preexisting_untracked_file(self, tmp_path, make_adapter):
        # recover_on_start=False leaves a prior-run file untracked; a later
        # store of that key must account it so usage/eviction include it.
        base = tmp_path / "fsl2"
        key = create_object_key(0)
        _seed_file(base, key, 1024)
        adapter = make_adapter(
            base_path=str(base),
            max_capacity_gb=_gb(1 << 20),
            recover_on_start=False,
        )
        assert adapter.get_usage().total_bytes_used == 0  # not tracked yet

        store_blocking(adapter, [key], [create_memory_obj(256)])  # exists -> skip
        assert adapter.get_usage().total_bytes_used == 1024  # accounted at disk size

    def test_recovery_skips_symlink(self, tmp_path, make_adapter):
        base = tmp_path / "fsl2"
        _seed_file(base, create_object_key(0), 1024)  # one real 1024B file
        # A symlink named like a valid key pointing at a larger external file.
        target = tmp_path / "external_blob"
        target.write_bytes(b"\x00" * 4096)
        os.symlink(target, base / _object_key_to_filename(create_object_key(1)))

        adapter = make_adapter(base_path=str(base), max_capacity_gb=_gb(1 << 20))
        # Only the real file is counted; the symlink target is not.
        assert adapter.get_usage().total_bytes_used == 1024


class TestDeleteSafety:
    def test_load_after_delete_is_miss(self, make_adapter):
        adapter = make_adapter(max_capacity_gb=_gb(1 << 20))
        key = create_object_key(0)
        store_blocking(adapter, [key], [create_memory_obj(256)])
        adapter.delete([key])

        bitmap = load_blocking(adapter, [key], [create_memory_obj(256)])
        assert not bitmap.test(0)  # miss, no exception, no corruption

    def test_open_descriptor_survives_unlink(self, tmp_path):
        """The POSIX property delete() relies on: an already-open fd keeps
        reading a file's data after it is unlinked."""
        path = tmp_path / "probe.bin"
        payload = b"\xcd" * 4096
        path.write_bytes(payload)

        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.unlink(path)  # concurrent eviction
            assert not path.exists()
            assert os.read(fd, len(payload)) == payload  # still readable
        finally:
            os.close(fd)
