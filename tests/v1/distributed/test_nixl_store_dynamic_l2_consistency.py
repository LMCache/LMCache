# SPDX-License-Identifier: Apache-2.0
"""
Consistency tests for DynamicNixlStoreL2Adapter.

The eviction controller calls ``delete()`` from its own thread while
lookup / store / load run on the adapter's asyncio loop thread. ``delete()``
drops a key from the in-memory index under the lock and unlinks the file
outside it. These tests pause ``delete()`` exactly inside that window and
interleave the other operations, then check that the index and the files on
disk agree.

U1  eviction vs lookup  -- a lookup must not re-add a key that is being deleted
U1b eviction vs store   -- a store of a key that is being deleted must not leave
                           an index entry whose file is then unlinked
U2  load ENOENT         -- a missing file drops the whole chunk from the index
U3  shard dir removed   -- the store recreates the shard directory once
U4  whole-chunk delete  -- deleting one object of a chunk deletes its siblings;
                           a pinned sibling keeps the whole chunk
"""

# Standard
from pathlib import Path
import os
import select
import shutil
import tempfile
import threading

# Third Party
import pytest
import torch

nixl = pytest.importorskip("nixl")

# First Party
from lmcache import torch_device_type  # noqa: E402
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.internal_api import (  # noqa: E402
    L1MemoryDesc,
    L2AdapterListener,
)
from lmcache.v1.distributed.l2_adapters.config import PersistConfig  # noqa: E402
from lmcache.v1.distributed.l2_adapters.nixl_store_agents.file_dynamic_nixl_store_agent import (  # noqa: E402, E501
    FileDynamicNixlStorageAgent,
)
from lmcache.v1.distributed.l2_adapters.nixl_store_dynamic_l2_adapter import (  # noqa: E402, E501
    DynamicNixlStoreL2Adapter,
    DynamicNixlStoreL2AdapterConfig,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

if torch_device_type == "xpu":
    pytest.skip(
        "NIXL dynamic store backends unavailable on XPU", allow_module_level=True
    )

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])
PAGE_SIZE = 4096
NUM_BUFFER_PAGES = 32
MAX_CAPACITY_GB = 0.001  # ~1 MB


# =============================================================================
# Helpers
# =============================================================================


class _RecordingListener(L2AdapterListener):
    def __init__(self) -> None:
        self.stored: list[ObjectKey] = []
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        self.stored.extend(keys)

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


def make_key(
    chunk_id: int, group: int = 0, kv_rank: int = 0, salt: str = ""
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=kv_rank,
        object_group_id=group,
        cache_salt=salt,
    )


def make_obj(buffer: torch.Tensor, page_index: int, fill: float) -> TensorMemoryObj:
    start = page_index * PAGE_SIZE
    raw = buffer[start : start + PAGE_SIZE].view(torch.float32)
    raw.fill_(fill)
    meta = MemoryObjMetadata(
        shape=torch.Size([PAGE_SIZE // 4]),
        dtype=torch.float32,
        address=page_index * PAGE_SIZE,
        phy_size=PAGE_SIZE,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def wait_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    if poll.poll(timeout * 1000):
        try:
            consume_fd(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def store(adpt: DynamicNixlStoreL2Adapter, keys, objs) -> bool:
    task_id = adpt.submit_store_task(keys, objs)
    assert wait_fd(adpt.get_store_event_fd())
    return adpt.pop_completed_store_tasks()[task_id].is_successful()


def lookup(adpt: DynamicNixlStoreL2Adapter, keys) -> list[bool]:
    task_id = adpt.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    assert wait_fd(adpt.get_lookup_and_lock_event_fd())
    bitmap = adpt.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def load(adpt: DynamicNixlStoreL2Adapter, keys, objs) -> list[bool]:
    task_id = adpt.submit_load_task(keys, objs)
    assert wait_fd(adpt.get_load_event_fd())
    bitmap = adpt.query_load_result(task_id)
    assert bitmap is not None
    return [bitmap.test(i) for i in range(len(keys))]


def unlock_and_flush(adpt: DynamicNixlStoreL2Adapter, keys) -> None:
    """Unlock and wait until the loop thread has run the unlock callback."""
    adpt.submit_unlock(keys)
    done = threading.Event()
    adpt._loop.call_soon_threadsafe(done.set)
    assert done.wait(5)


def file_of(adpt: DynamicNixlStoreL2Adapter, key: ObjectKey) -> str:
    agent = adpt.nixl_agent
    assert isinstance(agent, FileDynamicNixlStorageAgent)
    return agent._get_file_path_for_key(key)


def in_index(adpt: DynamicNixlStoreL2Adapter, key: ObjectKey) -> bool:
    with adpt._lock:
        return key in adpt._memory_objects


class PausedDelete:
    """Run ``adpt.delete(keys)`` in a thread and stop it before the first unlink.

    When ``entered`` is set, the keys are already gone from the index but their
    files still exist: the exact window between "remove from index" and
    "unlink".
    """

    def __init__(self, adpt: DynamicNixlStoreL2Adapter, keys: list[ObjectKey]):
        self.entered = threading.Event()
        self.release = threading.Event()
        agent = adpt.nixl_agent
        original = agent.dynamic_delete

        def paused(key: ObjectKey) -> None:
            self.entered.set()
            assert self.release.wait(10), "test never released the delete"
            original(key)

        agent.dynamic_delete = paused  # type: ignore[method-assign]
        self._restore = lambda: setattr(agent, "dynamic_delete", original)
        self.thread = threading.Thread(target=adpt.delete, args=(keys,), daemon=True)

    def __enter__(self) -> "PausedDelete":
        self.thread.start()
        assert self.entered.wait(5), "delete() never reached the unlink"
        return self

    def __exit__(self, *exc) -> None:
        self.release.set()
        self.thread.join(10)
        self._restore()


def build_adapter(tmp_dir: str, buffer: torch.Tensor, shard_dirs: bool = False):
    l1 = L1MemoryDesc(ptr=buffer.data_ptr(), size=buffer.numel(), align_bytes=PAGE_SIZE)
    cfg = DynamicNixlStoreL2AdapterConfig(
        backend="POSIX",
        backend_params={
            "file_path": tmp_dir,
            "use_direct_io": "false",
            "max_capacity_gb": str(MAX_CAPACITY_GB),
            "shard_dirs": str(shard_dirs).lower(),
        },
    )
    cfg.persist_config = PersistConfig(persist_enabled=True)
    return DynamicNixlStoreL2Adapter(cfg, l1)


@pytest.fixture(params=[False, True], ids=["flat", "sharded"])
def env(request):
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_consistency_")
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8)
    adpt = build_adapter(tmp_dir, buffer, shard_dirs=request.param)
    listener = _RecordingListener()
    adpt.register_listener(listener)
    yield adpt, buffer, Path(tmp_dir), listener
    adpt.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sharded_env():
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_consistency_shard_")
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8)
    adpt = build_adapter(tmp_dir, buffer, shard_dirs=True)
    yield adpt, buffer, Path(tmp_dir)
    adpt.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def assert_index_matches_disk(adpt: DynamicNixlStoreL2Adapter, keys) -> None:
    for key in keys:
        assert in_index(adpt, key) == os.path.exists(file_of(adpt, key)), (
            f"index/disk mismatch for {key}: index={in_index(adpt, key)} "
            f"file={os.path.exists(file_of(adpt, key))}"
        )


# =============================================================================
# U1: eviction vs lookup
# =============================================================================


def test_u1_lookup_does_not_resurrect_key_being_deleted(env):
    adpt, buf, _, _ = env
    key = make_key(1)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])

    with PausedDelete(adpt, [key]):
        # Index entry is gone, file still on disk: the secondary lookup would
        # stat it and re-add the key.
        assert not in_index(adpt, key)
        assert os.path.exists(file_of(adpt, key))
        hit = lookup(adpt, [key])
        assert hit == [False], "lookup re-added a key that is being deleted"

    assert not in_index(adpt, key)
    assert not os.path.exists(file_of(adpt, key))
    # The follow-up load must miss cleanly rather than hit ENOENT.
    assert load(adpt, [key], [make_obj(buf, 1, 0.0)]) == [False]
    assert adpt._total_bytes == 0


# =============================================================================
# U1b: eviction vs store of the same key
# =============================================================================


def test_u1b_store_during_delete_never_leaves_dangling_index(env):
    adpt, buf, _, _ = env
    key = make_key(2)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])

    with PausedDelete(adpt, [key]):
        store(adpt, [key], [make_obj(buf, 1, 2.0)])

    # Whatever the store decided, the index and the disk must agree.
    assert_index_matches_disk(adpt, [key])

    # Once the delete is finished the key can be stored and loaded again.
    assert store(adpt, [key], [make_obj(buf, 2, 3.0)])
    assert in_index(adpt, key)
    dst = make_obj(buf, 3, 0.0)
    assert load(adpt, [key], [dst]) == [True]
    assert torch.all(dst.tensor == 3.0)
    assert_index_matches_disk(adpt, [key])


# =============================================================================
# U2: load ENOENT drops the whole chunk
# =============================================================================


def test_u2_load_enoent_drops_whole_chunk(env):
    adpt, buf, _, listener = env
    k0, k1 = make_key(3, group=0), make_key(3, group=1)
    other = make_key(4)
    assert store(adpt, [k0, k1, other], [make_obj(buf, i, 1.0) for i in range(3)])
    bytes_before = adpt._total_bytes

    os.unlink(file_of(adpt, k0))  # file vanished behind the adapter's back
    keys = [k0, k1]
    assert lookup(adpt, keys) == [True, True]  # stale index still answers hit
    got = load(adpt, keys, [make_obj(buf, 10 + i, 0.0) for i in range(2)])
    unlock_and_flush(adpt, keys)
    assert got[0] is False

    # Both objects of the chunk leave the index, the unrelated chunk stays.
    assert not in_index(adpt, k0)
    assert not in_index(adpt, k1)
    assert in_index(adpt, other)
    assert adpt._total_bytes == bytes_before - 2 * PAGE_SIZE
    assert {k0, k1} <= set(listener.deleted)
    # Next lookup misses the chunk instead of failing on load again.
    assert lookup(adpt, keys) == [False, False]
    assert_index_matches_disk(adpt, [k0, k1, other])


# =============================================================================
# U3: shard directory removed externally
# =============================================================================


def test_u3_store_recreates_removed_shard_dir(sharded_env):
    adpt, buf, _ = sharded_env
    k0, k1 = make_key(5, group=0), make_key(5, group=1)  # same shard dir
    assert store(adpt, [k0], [make_obj(buf, 0, 1.0)])
    shard = os.path.dirname(file_of(adpt, k0))
    shutil.rmtree(shard)  # e.g. an operator wiped the L2 directory

    assert store(adpt, [k1], [make_obj(buf, 1, 2.0)])
    assert os.path.exists(file_of(adpt, k1))


def test_u3_store_does_not_retry_other_errors(sharded_env, monkeypatch):
    adpt, buf, _ = sharded_env
    calls = {"n": 0}
    real_open = os.open

    def deny(path, flags, *args, **kwargs):
        if ".tmp." in str(path):
            calls["n"] += 1
            raise PermissionError(13, "denied", path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny)
    assert not store(adpt, [make_key(6)], [make_obj(buf, 0, 1.0)])
    assert calls["n"] == 1


# =============================================================================
# U4: whole-chunk eviction
# =============================================================================


def _chunk_keys(chunk_id: int) -> list[ObjectKey]:
    return [make_key(chunk_id, group=g, kv_rank=r) for g in (0, 1) for r in (0, 1)]


def test_u4_delete_one_object_deletes_whole_chunk(env):
    adpt, buf, _, listener = env
    chunk = _chunk_keys(7)
    other = _chunk_keys(8)
    keys = chunk + other
    assert store(adpt, keys, [make_obj(buf, i, 1.0) for i in range(len(keys))])

    adpt.delete([chunk[0]])

    for key in chunk:
        assert not in_index(adpt, key)
    for key in other:
        assert in_index(adpt, key)
    assert set(chunk) <= set(listener.deleted)
    assert adpt._total_bytes == len(other) * PAGE_SIZE
    assert_index_matches_disk(adpt, keys)


def test_u4_pinned_sibling_keeps_whole_chunk(env):
    adpt, buf, _, _ = env
    chunk = _chunk_keys(9)
    assert store(adpt, chunk, [make_obj(buf, i, 1.0) for i in range(len(chunk))])

    assert lookup(adpt, [chunk[3]]) == [True]  # pins one object
    adpt.delete([chunk[0]])
    for key in chunk:
        assert in_index(adpt, key), "pinned sibling must keep the whole chunk"

    unlock_and_flush(adpt, [chunk[3]])
    adpt.delete([chunk[0]])
    for key in chunk:
        assert not in_index(adpt, key)
    assert_index_matches_disk(adpt, chunk)
    assert adpt._total_bytes == 0


def test_u4_salt_and_model_separate_chunks(env):
    adpt, buf, _, _ = env
    plain = make_key(10)
    salted = make_key(10, salt="tenantA")
    assert store(adpt, [plain, salted], [make_obj(buf, i, 1.0) for i in range(2)])
    adpt.delete([plain])
    assert not in_index(adpt, plain)
    assert in_index(adpt, salted)
