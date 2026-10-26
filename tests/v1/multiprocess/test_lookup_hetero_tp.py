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
import logging
import sys
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    L2AdaptersConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import L1WriteCompletion, StorageManager
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.tp_reshard import HeteroTP, PrefetchStage
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
    """Contiguous KV head sharding, written independently of the planner."""
    n = num_heads // tp
    return rank * n, n


def _layout(num_heads: int, tp: int) -> MemoryLayoutDesc:
    """One rank's object layout: ``[kv, L, T, local_heads * head_elems]``.

    A TP size above the head count gives each rank one (replicated) head.
    """
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
    ctx: MPCacheServerContext,
    num_heads: int,
    tp: int,
    offset: float = 0.0,
    num_chunks: int = NUM_CHUNKS,
) -> list[ObjectKey]:
    """Store the test prompt's first ``num_chunks`` chunks as a TP-``tp`` engine."""
    sm = ctx.storage_manager
    keys = _keys(ctx, tp)[: num_chunks * tp]
    objs = sm.reserve_write(keys, _layout(num_heads, tp))
    assert len(objs) == len(keys)
    for chunk in range(num_chunks):
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
def mp_ctx(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[MPCacheServerContext]:
    """A real engine context over a small CPU storage manager.

    L1-only by default; indirect parametrization with ``True`` adds a mock L2.
    """
    adapters: list[L2AdapterConfigBase] = (
        [MockL2AdapterConfig(max_size_gb=0.01, mock_bandwidth_gb=10.0)]
        if getattr(request, "param", False)
        else []
    )
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
            L2AdaptersConfig(adapters),
        ),
        chunk_size=CHUNK,
    )
    yield ctx
    ctx.close()


ModuleFactory = Callable[[int, int, int | None], LookupModule]


@pytest.fixture
def make_module(mp_ctx: MPCacheServerContext) -> Iterator[ModuleFactory]:
    """Register the model at the lookup's TP and build a LookupModule."""
    modules: list[LookupModule] = []

    def create(num_heads: int, world_size: int, source: int | None) -> LookupModule:
        mp_ctx.layout_desc_registry.register(
            MODEL,
            world_size,
            _layout(num_heads, world_size),
            total_num_kv_heads=num_heads,
        )
        module = LookupModule(mp_ctx, hetero_tp_source_world_size=source)
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


# (total KV heads, source TP, lookup TP)
RESHARD_CASES = [(2, 2, 1), (2, 1, 2), (4, 2, 4), (4, 4, 1), (4, 1, 4), (4, 4, 2)]


@pytest.mark.parametrize("num_heads,src_tp,dst_tp", RESHARD_CASES)
def test_reshard_hit_returns_all_chunks_with_exact_heads(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    num_heads: int,
    src_tp: int,
    dst_tp: int,
) -> None:
    """A miss at B served from A objects yields every chunk, B-sharded."""
    module = make_module(num_heads, dst_tp, src_tp)
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
    module = make_module(num_heads, dst_tp, src_tp)
    src_keys = _store(mp_ctx, num_heads, src_tp)
    dst_keys = _keys(mp_ctx, dst_tp)
    key = _lookup_key(dst_tp)

    module.lookup(key, dst_tp)
    assert _poll(module, "req") == NUM_CHUNKS

    # The reshard left every B object read-locked for the engine.
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == len(dst_keys)
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0

    module.free_lookup_locks(key, dst_tp)
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == 0


@pytest.mark.parametrize("num_heads,src_tp,dst_tp", [(2, 1, 2), (4, 1, 4)])
def test_reshard_locks_once_per_kv_reader(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    num_heads: int,
    src_tp: int,
    dst_tp: int,
) -> None:
    """Each B object stays locked until every worker's retrieve released it."""
    module = make_module(num_heads, dst_tp, src_tp)
    _store(mp_ctx, num_heads, src_tp)
    dst_keys = _keys(mp_ctx, dst_tp)

    module.lookup(_lookup_key(dst_tp), dst_tp)
    assert _poll(module, "req") == NUM_CHUNKS

    sm = mp_ctx.storage_manager
    for _ in range(dst_tp - 1):
        sm.finish_read_prefetched(dst_keys, read_locks=1)
        assert _num_locked(sm, dst_keys) == len(dst_keys)
    sm.finish_read_prefetched(dst_keys, read_locks=1)
    assert _num_locked(sm, dst_keys) == 0


def _same_tp_stage(windows: list[int]) -> PrefetchStage:
    """The same-TP stage of a TP-4 lookup over chunk hashes ``a`` and ``b``."""
    return PrefetchStage(
        key=_lookup_key(4),
        chunk_hashes=[b"a", b"b"],
        group_layout_descs={g: _layout(4, 4) for g in range(len(windows))},
        attn_desc=AttnWindowDesc(windows, world_size=4),
        num_kv_readers=4,
    )


@pytest.fixture
def hetero_tp(mp_ctx: MPCacheServerContext) -> Iterator[HeteroTP]:
    """Cross-TP lookups from source TP 2 for a model registered at TP 4."""
    mp_ctx.layout_desc_registry.register(MODEL, 4, _layout(4, 4), total_num_kv_heads=4)
    hetero_tp = HeteroTP(mp_ctx.layout_desc_registry, mp_ctx.storage_manager, 2)
    yield hetero_tp
    hetero_tp.close()


def test_cross_tp_stage_covers_the_rest_at_the_source_tp(hetero_tp: HeteroTP) -> None:
    """The cross-TP stage starts after the same-TP hit, keyed and shaped for A."""
    plan = hetero_tp.plan(_same_tp_stage([-1]))
    assert plan is not None

    stage = plan.cross_tp_stage(1)

    assert stage.key.world_size == 2
    assert stage.chunk_hashes == [b"b"]
    assert stage.group_layout_descs == {0: _layout(4, 2)}
    assert stage.attn_desc.world_size == 2
    assert stage.num_kv_readers == 1


@pytest.mark.parametrize("windows,qualifies", [([-1, -1], True), ([-1, 2], False)])
def test_plan_requires_full_attention(
    hetero_tp: HeteroTP, windows: list[int], qualifies: bool
) -> None:
    """A model with a sliding-window group gets no cross-TP lookup."""
    assert (hetero_tp.plan(_same_tp_stage(windows)) is not None) == qualifies


@pytest.mark.parametrize("source", [None, 4, 3, 8])
def test_no_usable_source_keeps_normal_miss(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
    source: int | None,
) -> None:
    """No source, the lookup's own TP, or a TP not dividing the heads: a miss."""
    module = make_module(4, 4, source)
    src_keys = _store(mp_ctx, 4, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(4), 4)

    assert _poll(module, "req") == 0
    assert len(calls) == 1
    assert _num_resident(mp_ctx.storage_manager, _keys(mp_ctx, 4)) == 0
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0


@pytest.mark.parametrize("mp_ctx", [True], indirect=True)
def test_reshard_drops_source_copies_loaded_from_l2(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A objects loaded from L2 are temporary under the default prefetch
    policy, so they leave L1 with the reshard; only B objects stay."""
    # MockL2Adapter never completes loads of multi-tensor objects.
    monkeypatch.setattr(sys.modules[__name__], "KERNEL_GROUP_LAYERS", (2,))
    sm = mp_ctx.storage_manager
    module = make_module(4, 2, 4)
    src_keys = _store(mp_ctx, 4, 4)
    adapter = sm.l2_adapters()[0][1]
    assert isinstance(adapter, MockL2Adapter)
    deadline = time.monotonic() + TIMEOUT
    while not all(adapter.debug_has_key(k) for k in src_keys):
        assert time.monotonic() < deadline, "source objects never reached L2"
        time.sleep(0.005)
    assert sm.delete_l1_keys(src_keys) == (len(src_keys), 0)

    module.lookup(_lookup_key(2), 2)

    assert _poll(module, "req") == NUM_CHUNKS
    _assert_holds(mp_ctx, _keys(mp_ctx, 2), 4, 2)
    assert _num_resident(sm, src_keys) == 0


def test_reshard_keeps_source_objects_already_in_l1(
    mp_ctx: MPCacheServerContext, make_module: ModuleFactory
) -> None:
    """A objects in L1 before the lookup (a TP=A engine on this server) stay."""
    module = make_module(4, 2, 4)
    src_keys = _store(mp_ctx, 4, 4)

    module.lookup(_lookup_key(2), 2)

    assert _poll(module, "req") == NUM_CHUNKS
    assert _num_resident(mp_ctx.storage_manager, src_keys) == len(src_keys)


def test_lookup_tp_above_head_count_keeps_normal_miss(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lookup TP larger than the KV head count is a plain miss, not an error."""
    module = make_module(2, 4, 1)
    src_keys = _store(mp_ctx, 2, 1)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)

    with caplog.at_level(logging.ERROR):
        module.lookup(_lookup_key(4), 4)
        assert _poll(module, "req") == 0

    assert len(calls) == 1
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_full_same_tp_hit_skips_cross_tp_stage(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B objects already present: served as-is, A objects never read."""
    module = make_module(2, 1, 2)
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


def test_partial_same_tp_hit_reshards_only_the_rest(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-TP hit of chunk 0 is kept; only chunk 1 is looked up at A and
    resharded; the answer locks both chunks once, with no lock left over."""
    module = make_module(2, 1, 2)
    src_keys = _store(mp_ctx, 2, 2)
    _store(mp_ctx, 2, 1, num_chunks=1)
    dst_keys = _keys(mp_ctx, 1)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)

    assert _poll(module, "req") == NUM_CHUNKS
    source_spec = calls[1][0]
    assert {k for row in source_spec.key_groups for k in row.keys} == set(src_keys[2:])
    _assert_holds(mp_ctx, dst_keys, 2, 1)
    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    module.free_lookup_locks(key, 1)
    assert _num_locked(mp_ctx.storage_manager, dst_keys) == 0


@pytest.mark.parametrize("source_hits", [False, True])
def test_partial_same_tp_hit_survives_a_missing_or_freed_source(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
    source_hits: bool,
) -> None:
    """With the source missing, or freed before its reshard starts, the
    lookup answers with its same-TP hit and holds exactly its locks."""
    module = make_module(2, 1, 2)
    src_keys = _store(mp_ctx, 2, 2) if source_hits else []
    same_tp_keys = _store(mp_ctx, 2, 1, num_chunks=1)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)
    if source_hits:
        deadline = time.monotonic() + TIMEOUT
        while len(calls) < 2:
            assert time.monotonic() < deadline, "cross-TP stage never submitted"
            assert module.query_prefetch_status("req") is None
            time.sleep(0.005)
        assert mp_ctx.storage_manager.wait_prefetch_status(calls[1][1], TIMEOUT)
        module.free_lookup_locks(key, 1)
        session = mp_ctx.session_manager.get("req")
        assert session is not None
        assert session.prefetch_hit_chunks == 1
    else:
        assert _poll(module, "req") == 1
        assert _num_locked(mp_ctx.storage_manager, same_tp_keys) == len(same_tp_keys)
        module.free_lookup_locks(key, 1)

    assert _num_locked(mp_ctx.storage_manager, src_keys) == 0
    assert _num_locked(mp_ctx.storage_manager, same_tp_keys) == 0


@pytest.mark.parametrize("mp_ctx", [True], indirect=True)
def test_same_tp_prefix_loaded_from_l2_stays_locked_with_the_hit(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A same-TP prefix loaded from L2 is temporary in L1; the same-TP stage's
    locks are part of the answer, so it is still there to be served."""
    # MockL2Adapter never completes loads of multi-tensor objects.
    monkeypatch.setattr(sys.modules[__name__], "KERNEL_GROUP_LAYERS", (2,))
    sm = mp_ctx.storage_manager
    module = make_module(4, 2, 4)
    same_tp_keys = _store(mp_ctx, 4, 2, num_chunks=1)
    adapter = sm.l2_adapters()[0][1]
    assert isinstance(adapter, MockL2Adapter)
    deadline = time.monotonic() + TIMEOUT
    while not all(adapter.debug_has_key(k) for k in same_tp_keys):
        assert time.monotonic() < deadline, "same-TP prefix never reached L2"
        time.sleep(0.005)
    assert sm.delete_l1_keys(same_tp_keys) == (len(same_tp_keys), 0)
    _store(mp_ctx, 4, 4)
    key = _lookup_key(2)

    module.lookup(key, 2)

    assert _poll(module, "req") == NUM_CHUNKS
    _assert_holds(mp_ctx, _keys(mp_ctx, 2), 4, 2)
    module.free_lookup_locks(key, 2)
    assert _num_resident(sm, same_tp_keys) == 0


def test_free_locks_after_cross_tp_stage_releases_and_misses(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeing a lookup whose cross-TP stage hit drops A locks and answers with
    the same-TP hit (none here)."""
    module = make_module(2, 1, 2)
    src_keys = _store(mp_ctx, 2, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)
    key = _lookup_key(1)

    module.lookup(key, 1)
    deadline = time.monotonic() + TIMEOUT
    while len(calls) < 2:
        assert time.monotonic() < deadline, "cross-TP stage never submitted"
        assert module.query_prefetch_status("req") is None
        time.sleep(0.005)
    source_handle = calls[1][1]
    assert mp_ctx.storage_manager.wait_prefetch_status(source_handle, TIMEOUT)
    # The finished cross-TP stage holds its A objects until someone settles it.
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
    """A failing reshard answers with the same-TP hit (none here), drops A locks."""
    module = make_module(2, 1, 2)
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
    """One wait covers the same-TP stage, the cross-TP stage and the reshard."""
    module = make_module(2, 1, 2)
    _store(mp_ctx, 2, 2)
    calls = _spy_submits(monkeypatch, mp_ctx.storage_manager)

    module.lookup(_lookup_key(1), 1)

    assert module.wait_prefetch_status("req", TIMEOUT) == NUM_CHUNKS
    # Cross-TP stage: A keys, one reader. The reshard locks its own objects,
    # so no further prefetch follows.
    assert len(calls) == 2
    source_spec = calls[1][0]
    assert source_spec.num_kv_readers == 1
    assert {k for row in source_spec.key_groups for k in row.keys} == set(
        _keys(mp_ctx, 2)
    )
    assert module.query_prefetch_status("req") == 0
    _assert_holds(mp_ctx, _keys(mp_ctx, 1), 2, 1)


def _start_reshard(module: LookupModule, entered: threading.Event) -> None:
    """Poll until the lookup's cross-TP stage hit is handed to the reshard."""
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
    module = make_module(2, 1, 2)
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
    module = make_module(2, 1, 2)
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
    module = make_module(2, 1, 2)
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


def test_free_during_cross_tp_prefetch_waits_and_releases_everything(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeing while the cross-TP prefetch runs waits for it, then drops the
    same-TP hit's locks and the A locks."""
    module = make_module(2, 1, 2)
    same_tp_keys = _store(mp_ctx, 2, 1, num_chunks=1)
    src_keys = _store(mp_ctx, 2, 2)
    sm = mp_ctx.storage_manager
    calls = _spy_submits(monkeypatch, sm)
    release = threading.Event()
    query, wait = sm.query_prefetch_status, sm.wait_prefetch_status

    def running(handle: PrefetchHandle) -> bool:
        return len(calls) > 1 and handle is calls[1][1] and not release.is_set()

    def held_query(handle: PrefetchHandle) -> PrefetchResult | None:
        return None if running(handle) else query(handle)

    def held_wait(handle: PrefetchHandle, timeout: float) -> bool:
        if running(handle) and not release.wait(timeout):
            return False
        return wait(handle, timeout)

    monkeypatch.setattr(sm, "query_prefetch_status", held_query)
    monkeypatch.setattr(sm, "wait_prefetch_status", held_wait)
    key = _lookup_key(1)

    module.lookup(key, 1)
    deadline = time.monotonic() + TIMEOUT
    while len(calls) < 2:
        assert time.monotonic() < deadline, "cross-TP stage never submitted"
        assert module.query_prefetch_status("req") is None
        time.sleep(0.005)

    threading.Timer(0.05, release.set).start()
    module.free_lookup_locks(key, 1)

    assert release.is_set()
    session = mp_ctx.session_manager.get("req")
    assert session is not None
    assert session.prefetch_hit_chunks == 1
    assert _num_locked(sm, same_tp_keys) == 0
    assert _num_locked(sm, src_keys) == 0
    assert module.query_prefetch_status("req") == 0


def _drop_reservation(
    monkeypatch: pytest.MonkeyPatch, sm: StorageManager, dropped: ObjectKey
) -> None:
    """Make ``reserve_write`` fail for ``dropped`` as if L1 were full."""
    reserve = sm.reserve_write

    def partial_reserve(
        keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> dict[ObjectKey, MemoryObj]:
        return reserve([k for k in keys if k != dropped], layout_desc)

    monkeypatch.setattr(sm, "reserve_write", partial_reserve)


def test_reshard_hit_stops_at_the_first_unwritten_object(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination object that cannot be written ends the hit there; the
    objects written after it are not left locked."""
    module = make_module(2, 2, 1)
    src_keys = _store(mp_ctx, 2, 1)
    dst_keys = _keys(mp_ctx, 2)
    sm = mp_ctx.storage_manager
    # Chunk 1 rank 0: chunk 1 is incomplete, its rank-1 object is written.
    _drop_reservation(monkeypatch, sm, dst_keys[2])
    key = _lookup_key(2)

    module.lookup(key, 2)

    assert module.wait_prefetch_status("req", TIMEOUT) == 1
    assert _num_locked(sm, dst_keys[2:]) == 0
    module.free_lookup_locks(key, 2)
    assert _num_locked(sm, dst_keys) == 0
    assert _num_locked(sm, src_keys) == 0


def test_reshard_failure_after_locking_releases_its_locks(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reshard that fails after locking some objects answers with the
    same-TP hit (none here) and leaves no object locked."""
    module = make_module(2, 2, 1)
    src_keys = _store(mp_ctx, 2, 1)
    dst_keys = _keys(mp_ctx, 2)
    sm = mp_ctx.storage_manager
    _drop_reservation(monkeypatch, sm, dst_keys[2])

    def fail(keys: list[ObjectKey], read_locks: int = 1) -> dict[ObjectKey, int]:
        raise RuntimeError("injected reserve_read failure")

    # Fails after the written objects were locked, on the one not written.
    monkeypatch.setattr(sm, "reserve_read", fail)

    module.lookup(_lookup_key(2), 2)

    assert module.wait_prefetch_status("req", TIMEOUT) == 0
    assert _num_locked(sm, dst_keys) == 0
    assert _num_locked(sm, src_keys) == 0


def test_concurrent_lookups_of_one_prefix_both_hit(
    mp_ctx: MPCacheServerContext,
    make_module: ModuleFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second lookup of a prefix being resharded waits for it and hits too."""
    module = make_module(2, 1, 2)
    _store(mp_ctx, 2, 2)
    sm = mp_ctx.storage_manager
    # Hold the first reshard after it reserved its destination objects, the
    # point where a second reshard of the same keys finds them taken.
    entered, release = threading.Event(), threading.Event()
    prepare = sm.prepare_write_completion

    def gated(reserved: dict[ObjectKey, MemoryObj]) -> L1WriteCompletion:
        if not entered.is_set():
            entered.set()
            assert release.wait(TIMEOUT)
        return prepare(reserved)

    monkeypatch.setattr(sm, "prepare_write_completion", gated)

    module.lookup(_lookup_key(1, "first"), 1)
    deadline = time.monotonic() + TIMEOUT
    while not entered.is_set():
        assert time.monotonic() < deadline, "first reshard never started"
        assert module.query_prefetch_status("first") is None
        time.sleep(0.005)
    module.lookup(_lookup_key(1, "second"), 1)
    for _ in range(20):
        assert module.query_prefetch_status("second") is None
        time.sleep(0.005)

    release.set()
    assert _poll(module, "first") == NUM_CHUNKS
    assert _poll(module, "second") == NUM_CHUNKS
    _assert_holds(mp_ctx, _keys(mp_ctx, 1), 2, 1)
