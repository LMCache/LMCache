# SPDX-License-Identifier: Apache-2.0
"""Cross-TP (heterogeneous TP) lookup through LookupModule's public handlers.

Objects stored at one TP size (the source, ``A``) are looked up at another
(``B``). The engine context is real (L1-only CPU StorageManager, layout
registry, session manager, token hasher), so read locks and bytes are the
storage manager's own. Lock state is probed with ``delete_l1_keys``: a locked
key is refused (counted as skipped) and left in place, an unlocked one is
deleted.
"""

# Standard
from collections.abc import Callable, Iterator
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

MODEL = "hetero-model"
CHUNK = 4
NUM_CHUNKS = 2
HEAD_ELEMS = 2
# Layers per kernel group; two kernel groups exercise the per-group copy.
KERNEL_GROUP_LAYERS = (1, 2)
TIMEOUT = 5.0


def _head_range(num_heads: int, tp: int, rank: int) -> tuple[int, int]:
    """vLLM GQA sharding, written independently of the planner under test."""
    n = max(1, num_heads // tp)
    return (rank // max(1, tp // num_heads)) * n, n


def _layout(num_heads: int, tp: int) -> MemoryLayoutDesc:
    """One rank's object layout: ``[kv, L, T, local_heads * head_elems]``."""
    width = max(1, num_heads // tp) * HEAD_ELEMS
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, n, CHUNK, width]) for n in KERNEL_GROUP_LAYERS],
        dtypes=[torch.float32] * len(KERNEL_GROUP_LAYERS),
    )


def _global_kv(num_heads: int, chunk: int, offset: float = 0.0) -> list[torch.Tensor]:
    """All heads of one chunk, distinct per chunk and kernel group."""
    return [
        torch.arange(2 * n * CHUNK * num_heads * HEAD_ELEMS, dtype=torch.float32)
        .reshape(2, n, CHUNK, num_heads * HEAD_ELEMS)
        .add(offset + 1000.0 * chunk + 100000.0 * group)
        for group, n in enumerate(KERNEL_GROUP_LAYERS)
    ]


def _rank_slice(
    full: list[torch.Tensor], num_heads: int, tp: int, rank: int
) -> list[torch.Tensor]:
    """The heads ``rank`` holds at ``tp``, per kernel group."""
    first, n = _head_range(num_heads, tp, rank)
    return [t[..., first * HEAD_ELEMS : (first + n) * HEAD_ELEMS] for t in full]


def _lookup_key(world_size: int, request_id: str = "req") -> IPCCacheServerKey:
    """A lookup key (all workers) over ``NUM_CHUNKS`` full chunks."""
    return IPCCacheServerKey.from_token_ids(
        model_name=MODEL,
        world_size=world_size,
        worker_id=None,
        token_ids=tuple(range(CHUNK * NUM_CHUNKS)),
        start=0,
        end=CHUNK * NUM_CHUNKS,
        request_id=request_id,
        num_kv_readers=world_size,
    )


def _keys(ctx: MPCacheServerContext, world_size: int) -> list[ObjectKey]:
    """Object keys of the test prompt at a world size, chunk-major."""
    key = _lookup_key(world_size)
    hashes = ctx.token_hasher.compute_packed_chunk_hashes(key.token_bytes, end=key.end)
    assert len(hashes) == NUM_CHUNKS
    return ipc_key_to_object_keys(key, hashes, [0])[0]


def _store(
    ctx: MPCacheServerContext, num_heads: int, tp: int, offset: float = 0.0
) -> list[ObjectKey]:
    """Store the test prompt's objects as written by a TP-``tp`` engine."""
    sm = ctx.storage_manager
    keys = _keys(ctx, tp)
    objs = sm.reserve_write(keys, _layout(num_heads, tp))
    assert len(objs) == len(keys)
    for chunk in range(NUM_CHUNKS):
        full = _global_kv(num_heads, chunk, offset)
        for rank in range(tp):
            obj = objs[keys[chunk * tp + rank]]
            for group, src in enumerate(_rank_slice(full, num_heads, tp, rank)):
                tensor = obj.get_tensor(group)
                assert tensor is not None
                tensor.copy_(src)
    sm.finish_write(keys)
    return keys


def _assert_holds(
    ctx: MPCacheServerContext,
    keys: list[ObjectKey],
    num_heads: int,
    tp: int,
    offset: float = 0.0,
) -> None:
    """Assert the (read-locked) objects hold their TP-``tp`` head slices."""
    got_keys, objs = ctx.storage_manager.unsafe_read(keys)
    assert got_keys == keys
    for chunk in range(NUM_CHUNKS):
        full = _global_kv(num_heads, chunk, offset)
        for rank in range(tp):
            obj = objs[chunk * tp + rank]
            for group, want in enumerate(_rank_slice(full, num_heads, tp, rank)):
                torch.testing.assert_close(obj.get_tensor(group), want, rtol=0, atol=0)


def _num_locked(sm: StorageManager, keys: list[ObjectKey]) -> int:
    """Count locked keys; unlocked keys are deleted as a side effect."""
    _deleted, skipped = sm.delete_l1_keys(keys)
    return skipped


def _num_resident(sm: StorageManager, keys: list[ObjectKey]) -> int:
    """Count resident keys, deleting the unlocked ones."""
    deleted, skipped = sm.delete_l1_keys(keys)
    return deleted + skipped


def _poll(module: LookupModule, request_id: str) -> int:
    """Poll ``query_prefetch_status`` until the lookup resolves."""
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        result = module.query_prefetch_status(request_id)
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError(f"lookup {request_id} never resolved")


def _spy_submits(
    monkeypatch: pytest.MonkeyPatch, sm: StorageManager
) -> list[tuple[PrefetchTaskSpec, PrefetchHandle]]:
    """Record every prefetch the storage manager is asked to run."""
    calls: list[tuple[PrefetchTaskSpec, PrefetchHandle]] = []
    submit = sm.submit_prefetch_task

    def spy(spec: PrefetchTaskSpec, external_request_id: str = "") -> PrefetchHandle:
        handle = submit(spec, external_request_id=external_request_id)
        calls.append((spec, handle))
        return handle

    monkeypatch.setattr(sm, "submit_prefetch_task", spy)
    return calls


@pytest.fixture
def mp_ctx(monkeypatch: pytest.MonkeyPatch) -> Iterator[MPCacheServerContext]:
    """A real engine context over a small L1-only CPU storage manager."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    ctx = MPCacheServerContext(
        StorageManagerConfig(
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=1 << 20, use_lazy=False, shm_name=""
                )
            ),
            EvictionConfig("noop"),
        ),
        chunk_size=CHUNK,
    )
    yield ctx
    ctx.close()


ModuleFactory = Callable[[int, int, list[int]], LookupModule]


@pytest.fixture
def make_module(mp_ctx: MPCacheServerContext) -> Iterator[ModuleFactory]:
    """Register the model at the lookup's TP and build a LookupModule."""
    modules: list[LookupModule] = []

    def create(num_heads: int, world_size: int, sources: list[int]) -> LookupModule:
        mp_ctx.layout_desc_registry.register(
            MODEL,
            world_size,
            _layout(num_heads, world_size),
            total_num_kv_heads=num_heads,
        )
        module = LookupModule(mp_ctx, hetero_tp_source_world_sizes=sources)
        modules.append(module)
        return module

    yield create
    for module in modules:
        module.close()


def _gate_reserve_write(
    monkeypatch: pytest.MonkeyPatch, sm: StorageManager
) -> tuple[threading.Event, threading.Event]:
    """Hold the reshard inside ``reserve_write`` until ``release`` is set.

    Returns:
        ``(entered, release)``: set when the reshard reaches the gate, and
        set by the test to let it continue.
    """
    entered, release = threading.Event(), threading.Event()
    reserve = sm.reserve_write

    def gated(keys: list[ObjectKey], layout_desc: MemoryLayoutDesc) -> dict:
        entered.set()
        assert release.wait(TIMEOUT)
        return reserve(keys, layout_desc)

    monkeypatch.setattr(sm, "reserve_write", gated)
    return entered, release


# (total KV heads, source TP, lookup TP); the last three replicate heads.
RESHARD_CASES = [(2, 2, 1), (2, 1, 2), (4, 2, 4), (2, 4, 1), (2, 1, 4), (2, 4, 2)]


@pytest.mark.parametrize("num_heads,src_tp,dst_tp", RESHARD_CASES)
def test_reshard_hit_returns_all_chunks_with_exact_heads(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    num_heads: int,
    src_tp: int,
    dst_tp: int,
) -> None:
    """A miss at B served from A objects yields every chunk, B-sharded."""
    module = make_module(num_heads, dst_tp, [src_tp])
    _store(mp_ctx, num_heads, src_tp)
    dst_keys = _keys(mp_ctx, dst_tp)

    module.lookup(_lookup_key(dst_tp), dst_tp)

    assert _poll(module, "req") == NUM_CHUNKS
    _assert_holds(mp_ctx, dst_keys, num_heads, dst_tp)


@pytest.mark.parametrize("num_heads,src_tp,dst_tp", RESHARD_CASES)
def test_reshard_leaves_no_read_locks(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    num_heads: int,
    src_tp: int,
    dst_tp: int,
) -> None:
    """The reshard drops A locks; free_lookup_locks drops the B locks."""
    module = make_module(num_heads, dst_tp, [src_tp])
    src_keys = _store(mp_ctx, num_heads, src_tp)
    dst_keys = _keys(mp_ctx, dst_tp)
    key = _lookup_key(dst_tp)

    module.lookup(key, dst_tp)
    assert _poll(module, "req") == NUM_CHUNKS

    # The final prefetch still holds every B object for the engine.
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == len(dst_keys)
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0

    module.free_lookup_locks(key, dst_tp)
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == 0


@pytest.mark.parametrize("sources", [[], [4], [3]])
def test_no_usable_source_keeps_normal_miss(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
    sources: list[int],
) -> None:
    """No source, only the lookup's own TP, or an unshardable TP: one prefetch."""
    module = make_module(4, 4, sources)
    src_keys = _store(mp_ctx, 4, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(4), 4)

    assert _poll(module, "req") == 0
    assert len(calls) == 1
    assert _num_resident(mp_ctx.storage_manager, _keys(mp_ctx, 4)) == 0
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0


def test_primary_hit_skips_source_stage(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B objects already present: served as-is, A objects never read."""
    module = make_module(2, 1, [2])
    src_keys = _store(mp_ctx, 2, 2)
    dst_keys = _store(mp_ctx, 2, 1, offset=0.5)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)

    assert _poll(module, "req") == NUM_CHUNKS
    assert len(calls) == 1
    _assert_holds(mp_ctx, dst_keys, 2, 1, offset=0.5)
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    module.free_lookup_locks(key, 1)
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == 0


def test_free_locks_after_source_stage_releases_and_misses(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeing a lookup whose source stage hit drops A locks and reports a miss."""
    module = make_module(2, 1, [2])
    src_keys = _store(mp_ctx, 2, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)
    deadline = time.monotonic() + TIMEOUT
    while len(calls) < 2:
        assert time.monotonic() < deadline, "source stage never submitted"
        assert module.query_prefetch_status("req") is None
        time.sleep(0.005)
    source_handle = calls[1][1]
    assert mp_ctx.storage_manager.wait_prefetch_status(source_handle, TIMEOUT)
    # The finished source stage holds its A objects until someone settles it.
    assert _num_locked(mp_ctx.storage_manager, src_keys) == len(src_keys)

    module.free_lookup_locks(key, 1)

    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    assert len(calls) == 2
    session = mp_ctx.session_manager.get("req")
    assert session is not None
    assert session.prefetch_hit_chunks == 0
    assert module.query_prefetch_status("req") == 0
    assert _num_resident(mp_ctx.storage_manager, _keys(mp_ctx, 1)) == 0


@pytest.mark.parametrize("use_wait", [False, True])
def test_reshard_failure_resolves_as_miss(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
    use_wait: bool,
) -> None:
    """A failing reshard answers the lookup with a miss and drops A locks."""
    module = make_module(2, 1, [2])
    src_keys = _store(mp_ctx, 2, 2)

    def fail(keys: list[ObjectKey], layout_desc: MemoryLayoutDesc) -> dict:
        raise RuntimeError("injected reserve failure")

    monkeypatch.setattr(mp_ctx.storage_manager, "reserve_write", fail)

    module.lookup(_lookup_key(1), 1)

    if use_wait:
        assert module.wait_prefetch_status("req", TIMEOUT) == 0
    else:
        assert _poll(module, "req") == 0
    assert module.query_prefetch_status("req") == 0
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    assert _num_resident(mp_ctx.storage_manager, _keys(mp_ctx, 1)) == 0


def test_wait_follows_stage_chain_in_one_call(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One wait covers primary, source and final stages and returns the hit."""
    module = make_module(2, 1, [2])
    _store(mp_ctx, 2, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(1), 1)

    assert module.wait_prefetch_status("req", TIMEOUT) == NUM_CHUNKS
    assert len(calls) == 3
    # Source stage: A keys, one reader; final stage: B keys, L1 only.
    source_spec, final_spec = calls[1][0], calls[2][0]
    assert source_spec.num_kv_readers == 1
    assert {k for row in source_spec.key_groups for k in row.keys} == set(
        _keys(mp_ctx, 2)
    )
    assert final_spec.allowed_l2_memory_tags == frozenset()
    assert module.query_prefetch_status("req") == 0
    _assert_holds(mp_ctx, _keys(mp_ctx, 1), 2, 1)


def test_second_source_tried_after_first_misses(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
) -> None:
    """Candidates are tried in order; a miss at one moves to the next."""
    module = make_module(4, 4, [1, 2])
    _store(mp_ctx, 4, 2)

    module.lookup(_lookup_key(4), 4)

    assert module.wait_prefetch_status("req", TIMEOUT) == NUM_CHUNKS
    _assert_holds(mp_ctx, _keys(mp_ctx, 4), 4, 4)


def _start_reshard(module: LookupModule, entered: threading.Event) -> None:
    """Poll until the lookup's source-stage hit is handed to the reshard."""
    deadline = time.monotonic() + TIMEOUT
    while not entered.is_set():
        assert time.monotonic() < deadline, "reshard never started"
        assert module.query_prefetch_status("req") is None
        time.sleep(0.005)


def test_query_does_not_block_on_reshard(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While the reshard runs, polling returns None at once; then the hit."""
    module = make_module(2, 1, [2])
    _store(mp_ctx, 2, 2)
    entered, release = _gate_reserve_write(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(1), 1)
    _start_reshard(module, entered)

    started = time.monotonic()
    assert module.query_prefetch_status("req") is None
    assert time.monotonic() - started < 1.0
    release.set()
    assert _poll(module, "req") == NUM_CHUNKS
    _assert_holds(mp_ctx, _keys(mp_ctx, 1), 2, 1)


def test_wait_times_out_during_reshard_then_resolves(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wait shorter than the reshard returns None; a later wait gets the hit."""
    module = make_module(2, 1, [2])
    _store(mp_ctx, 2, 2)
    entered, release = _gate_reserve_write(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(1), 1)
    _start_reshard(module, entered)

    assert module.wait_prefetch_status("req", 0.05) is None
    release.set()
    assert module.wait_prefetch_status("req", TIMEOUT) == NUM_CHUNKS


def test_free_during_reshard_waits_and_releases_everything(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeing mid-reshard waits for it, then drops both A and B locks."""
    module = make_module(2, 1, [2])
    src_keys = _store(mp_ctx, 2, 2)
    dst_keys = _keys(mp_ctx, 1)
    entered, release = _gate_reserve_write(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)
    _start_reshard(module, entered)

    threading.Timer(0.05, release.set).start()
    module.free_lookup_locks(key, 1)

    assert release.is_set()
    session = mp_ctx.session_manager.get("req")
    assert session is not None
    assert session.prefetch_hit_chunks == NUM_CHUNKS
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == 0
    assert module.query_prefetch_status("req") == 0


def test_final_stage_failure_resolves_as_miss(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the final prefetch cannot be submitted, the lookup reports a miss."""
    module = make_module(2, 1, [2])
    src_keys = _store(mp_ctx, 2, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    submit = mp_ctx.storage_manager.submit_prefetch_task

    def fail_third(
        spec: PrefetchTaskSpec, external_request_id: str = ""
    ) -> PrefetchHandle:
        if len(calls) == 2:
            raise RuntimeError("injected final-stage failure")
        return submit(spec, external_request_id=external_request_id)

    monkeypatch.setattr(mp_ctx.storage_manager, "submit_prefetch_task", fail_third)

    module.lookup(_lookup_key(1), 1)

    assert module.wait_prefetch_status("req", TIMEOUT) == 0
    assert module.query_prefetch_status("req") == 0
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
