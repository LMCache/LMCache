# SPDX-License-Identifier: Apache-2.0
"""
Consistency tests for DynamicNixlStoreL2Adapter.

The eviction controller calls ``delete()`` from its own thread while
lookup / store / load run on the adapter's asyncio loop thread. These tests
pause ``delete()`` exactly between "key removed from the index" and "file
unlinked" and interleave the other operations, then check, through the public
interface only, that what the adapter reports (lookup results, usage, listener
events) agrees with the files on disk.

U1  eviction vs lookup   -- a lookup must not re-add a key that is being deleted
U1b eviction vs store    -- a store of a key that is being deleted must not leave
                            an index entry whose file is then unlinked
U2  load, file missing   -- ENOENT/ESTALE drops only that key from the index;
                            other errors leave it alone
U3  shard dir removed    -- the store recreates the shard directory once
U4  unlink failure       -- delete() keeps usage accounting consistent when an
                            unlink fails
"""

# Standard
from pathlib import Path
import errno
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
        self.deleted: list[ObjectKey] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]) -> None:
        pass

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        pass

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        self.deleted.extend(keys)


def make_key(chunk_id: int, group: int = 0, tags: dict[str, str] | None = None):
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
        object_group_id=group,
        tags=tags,
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


def files_of(root: Path, key: ObjectKey) -> list[Path]:
    """Data files of ``key``, flat or sharded (the hash hex is in the name).

    Tags are part of the file name after an ``@``; an untagged key's name ends
    right after the hash.
    """
    tagged = bool(key.identity.tags)
    return [
        p
        for p in root.rglob(f"*_{key.chunk_hash.hex()}*.bin")
        if f"_{key.object_group_id}_" in p.name and ("@" in p.name) == tagged
    ]


def file_of(root: Path, key: ObjectKey) -> Path:
    (path,) = files_of(root, key)
    return path


def used_bytes(adpt: DynamicNixlStoreL2Adapter) -> int:
    return adpt.get_usage().total_bytes_used


def assert_usage_matches_disk(adpt: DynamicNixlStoreL2Adapter, root: Path) -> None:
    on_disk = sum(p.stat().st_size for p in root.rglob("*.bin"))
    assert used_bytes(adpt) == on_disk


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


def build_adapter(
    tmp_dir: str,
    buffer: torch.Tensor,
    shard_dirs: bool = False,
    shared: bool = False,
):
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
    cfg.shared = shared
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


# =============================================================================
# U1: eviction vs lookup
# =============================================================================


def test_u1_lookup_does_not_resurrect_key_being_deleted(env):
    adpt, buf, root, _ = env
    key = make_key(1)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])

    with PausedDelete(adpt, [key]):
        # Index entry is gone, file still on disk: the secondary lookup would
        # stat it and re-add the key.
        assert files_of(root, key)
        assert lookup(adpt, [key]) == [False], "lookup re-added a key being deleted"

    assert not files_of(root, key)
    # The follow-up load must miss cleanly rather than hit ENOENT.
    assert load(adpt, [key], [make_obj(buf, 1, 0.0)]) == [False]
    assert used_bytes(adpt) == 0


# =============================================================================
# U1b: eviction vs store of the same key
# =============================================================================


def test_u1b_store_during_delete_never_leaves_dangling_index(env):
    adpt, buf, root, _ = env
    key = make_key(2)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])

    with PausedDelete(adpt, [key]):
        store(adpt, [key], [make_obj(buf, 1, 2.0)])

    # Whatever the store decided, usage and the disk must agree.
    assert_usage_matches_disk(adpt, root)

    # Once the delete is finished the key can be stored and loaded again.
    assert store(adpt, [key], [make_obj(buf, 2, 3.0)])
    dst = make_obj(buf, 3, 0.0)
    assert load(adpt, [key], [dst]) == [True]
    assert torch.all(dst.tensor == 3.0)
    assert_usage_matches_disk(adpt, root)


# =============================================================================
# U2: load finds the file missing
# =============================================================================


def test_u2_missing_file_drops_only_that_key(env):
    adpt, buf, root, listener = env
    gone, kept = make_key(3, group=0), make_key(3, group=1)
    assert store(adpt, [gone, kept], [make_obj(buf, i, 1.0) for i in range(2)])

    file_of(root, gone).unlink()  # vanished behind the adapter's back
    keys = [gone, kept]
    assert lookup(adpt, keys) == [True, True]  # stale index still answers hit
    got = load(adpt, keys, [make_obj(buf, 10 + i, 0.0) for i in range(2)])
    assert got == [False, True]
    adpt.submit_unlock(keys)

    assert listener.deleted == [gone]
    assert used_bytes(adpt) == PAGE_SIZE
    # The stale key now misses instead of failing on every later load.
    assert lookup(adpt, keys) == [False, True]
    assert_usage_matches_disk(adpt, root)


def test_u2_missing_file_does_not_touch_tag_variants(env):
    adpt, buf, root, listener = env
    base = make_key(4)
    tagged = make_key(4, tags={"lora": "a"})  # separate file, same chunk hash
    assert store(adpt, [base, tagged], [make_obj(buf, i, 1.0) for i in range(2)])

    file_of(root, base).unlink()
    assert lookup(adpt, [base]) == [True]
    assert load(adpt, [base], [make_obj(buf, 10, 0.0)]) == [False]
    adpt.submit_unlock([base])

    assert files_of(root, tagged), "valid file of a tag variant was unlinked"
    dst = make_obj(buf, 11, 0.0)
    assert lookup(adpt, [tagged]) == [True]
    assert load(adpt, [tagged], [dst]) == [True]
    assert torch.all(dst.tensor == 1.0)


def test_u2_estale_is_treated_as_missing(env):
    adpt, buf, root, listener = env
    key = make_key(5)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])
    agent = adpt.nixl_agent
    original = agent.dynamic_load

    async def stale(mem_indices, k):
        raise OSError(errno.ESTALE, "Stale file handle")

    agent.dynamic_load = stale  # type: ignore[method-assign]
    try:
        assert lookup(adpt, [key]) == [True]
        assert load(adpt, [key], [make_obj(buf, 1, 0.0)]) == [False]
        adpt.submit_unlock([key])
    finally:
        agent.dynamic_load = original  # type: ignore[method-assign]

    assert listener.deleted == [key]
    assert used_bytes(adpt) == 0


def test_u2_other_load_errors_keep_the_entry(env):
    adpt, buf, _, listener = env
    key = make_key(6)
    assert store(adpt, [key], [make_obj(buf, 0, 1.0)])
    agent = adpt.nixl_agent
    original = agent.dynamic_load

    async def denied(mem_indices, k):
        raise PermissionError(errno.EACCES, "denied")

    agent.dynamic_load = denied  # type: ignore[method-assign]
    try:
        assert lookup(adpt, [key]) == [True]
        assert load(adpt, [key], [make_obj(buf, 1, 0.0)]) == [False]
        adpt.submit_unlock([key])
    finally:
        agent.dynamic_load = original  # type: ignore[method-assign]

    assert listener.deleted == []
    assert used_bytes(adpt) == PAGE_SIZE


# =============================================================================
# U3: shard directory removed externally
# =============================================================================


def test_u3_store_recreates_removed_shard_dir(sharded_env):
    adpt, buf, root = sharded_env
    k0, k1 = make_key(7, group=0), make_key(7, group=1)  # same shard dir
    assert store(adpt, [k0], [make_obj(buf, 0, 1.0)])
    shutil.rmtree(file_of(root, k0).parent)  # e.g. an operator wiped the dir

    assert store(adpt, [k1], [make_obj(buf, 1, 2.0)])
    assert files_of(root, k1)


def test_u3_store_does_not_retry_other_errors(sharded_env, monkeypatch):
    adpt, buf, _ = sharded_env
    calls = {"n": 0}
    real_open = os.open

    def deny(path, flags, *args, **kwargs):
        if ".tmp." in str(path):
            calls["n"] += 1
            raise PermissionError(errno.EACCES, "denied", path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny)
    assert not store(adpt, [make_key(8)], [make_obj(buf, 0, 1.0)])
    assert calls["n"] == 1


# =============================================================================
# U4: unlink failure during delete
# =============================================================================


def test_u4_failed_unlink_does_not_skew_usage(env):
    adpt, buf, root, listener = env
    ok, bad = make_key(9, group=0), make_key(9, group=1)
    assert store(adpt, [ok, bad], [make_obj(buf, i, 1.0) for i in range(2)])
    agent = adpt.nixl_agent
    original = agent.dynamic_delete

    def flaky(key: ObjectKey) -> None:
        if key == bad:
            raise PermissionError(errno.EACCES, "denied")
        original(key)

    agent.dynamic_delete = flaky  # type: ignore[method-assign]
    try:
        adpt.delete([bad, ok])  # must not raise, must still process `ok`
    finally:
        agent.dynamic_delete = original  # type: ignore[method-assign]

    assert set(listener.deleted) == {ok, bad}
    assert not files_of(root, ok)
    # The leftover file of `bad` is found by the secondary lookup again, which
    # re-registers its bytes, so usage and disk agree.
    assert lookup(adpt, [bad]) == [True]
    assert_usage_matches_disk(adpt, root)


# =============================================================================
# U5: shared directory, lookup hit verifies the file
# =============================================================================


@pytest.fixture(params=[True, False], ids=["shared", "not_shared"])
def shared_env(request):
    tmp_dir = tempfile.mkdtemp(prefix="nixl_dyn_l2_consistency_shared_")
    buffer = torch.empty(PAGE_SIZE * NUM_BUFFER_PAGES, dtype=torch.uint8)
    adpt = build_adapter(tmp_dir, buffer, shared=request.param)
    listener = _RecordingListener()
    adpt.register_listener(listener)
    yield adpt, buffer, Path(tmp_dir), listener, request.param
    adpt.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_u5_shared_lookup_misses_file_deleted_by_another_server(shared_env):
    adpt, buf, root, listener, shared = shared_env
    gone, kept = make_key(20, group=0), make_key(20, group=1)
    assert store(adpt, [gone, kept], [make_obj(buf, i, 1.0) for i in range(2)])

    file_of(root, gone).unlink()  # another server evicted it
    hits = lookup(adpt, [gone, kept])

    if shared:
        assert hits == [False, True]
        assert listener.deleted == [gone]
        assert used_bytes(adpt) == PAGE_SIZE
        assert_usage_matches_disk(adpt, root)
    else:
        # Single-server setups do not pay the extra stat: the stale entry is
        # only noticed by the load (see U2).
        assert hits == [True, True]
        assert listener.deleted == []


def test_u5_shared_lookup_keeps_present_files(shared_env):
    adpt, buf, root, listener, _ = shared_env
    key = make_key(21)
    assert store(adpt, [key], [make_obj(buf, 0, 4.0)])

    assert lookup(adpt, [key]) == [True]
    dst = make_obj(buf, 1, 0.0)
    assert load(adpt, [key], [dst]) == [True]
    assert torch.all(dst.tensor == 4.0)
    assert listener.deleted == []
