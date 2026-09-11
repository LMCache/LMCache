# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the blend retrieve planner (native flat-plan fast path):
invariant-spec caching and re-stamping, work-table encoding, double-buffered
wave slotting, and the fallback gates (non-lazy objects, compressed groups).

Moved from test_blend_load_store_opts.py in the blend package split."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# First Party
from lmcache import device_ops  # noqa: F401
from lmcache.v1.multiprocess.modules.blend import retrieve as retrieve_mod
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState
import lmcache.lmcache_native as lmcache_native

# ---------------------------------------------------------------------------
# Retrieve: native plan builder (execute_cb_retrieve_plan fast path)
# ---------------------------------------------------------------------------


def _native_retrieve_plan_available() -> bool:
    """Return whether the C++ native retrieve-plan interfaces are available."""

    return retrieve_mod._HAS_NATIVE_RETRIEVE_PLAN and hasattr(device_ops, "CBGroupSpec")


native_retrieve_plan_required = pytest.mark.skipif(
    not _native_retrieve_plan_available(),
    reason="requires the native blend retrieve-plan C++ support",
)


def _build_plan_engine_and_context(
    num_groups: int = 2,
    max_batch: int = 2,
    spc: int = 4,
    num_layers: int = 2,
    head_size: int = 8,
    n_heads: int = 2,
):
    """Engine with the real ``_build_cb_retrieve_plan_flat`` bound, a fake GPU
    context with real CPU tensors, and a real ``_CBRopeState``. Kernel
    groups are plain (non-fused) K/V, so hidden_dim = n_heads * head_size."""
    # Standard
    import weakref

    # Third Party
    import torch

    eng = MagicMock(spec=BlendModule)
    for name in (
        "_build_cb_retrieve_plan_flat",
        "_resolve_cb_plan_invariants",
        "_cb_slot_buffers",
        "_cb_staged_groups",
    ):
        setattr(eng, name, getattr(BlendModule, name).__get__(eng))
    eng._cb_plan_invariants = weakref.WeakKeyDictionary()
    eng._cb_slot_staging = weakref.WeakKeyDictionary()
    eng._cb_plan_done_events = weakref.WeakKeyDictionary()

    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.kv_layer_groups import ObjectGroupInfo

    hidden_dim = n_heads * head_size
    gpu_context = MagicMock()
    gpu_context.device = torch.device("cpu")
    # Legacy fused layout: one object group holding every kernel group.
    gpu_context.kv_layer_groups_manager.get_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1]
    )
    gpu_context.kv_layer_groups_manager.object_groups = [
        ObjectGroupInfo(kernel_group_indices=list(range(num_groups)))
    ]
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(
            tokens_per_block=4,
            slots_per_block=4,
            engine_group_idx=0,
            engine_kv_format=lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            shape_desc=SimpleNamespace(nb=100),
        )
        for _ in range(num_groups)
    ]
    kv_buffers = {
        (slot, group): torch.zeros(2, num_layers, spc, hidden_dim)
        for slot in range(max_batch)
        for group in range(num_groups)
    }
    gpu_context.get_temp_kernel_group_buffer.side_effect = lambda s, g: kv_buffers[
        (s, g)
    ]
    ptr_tensors = [torch.zeros(num_layers, dtype=torch.long) for _ in range(num_groups)]
    gpu_context.get_kernel_group_kv_pointers.side_effect = lambda g: ptr_tensors[g]
    gpu_context.get_engine_kv_format.side_effect = lambda g: (
        lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    )
    # One object group; each chunk memory object fills one flat slot.
    obj_bytes = sum(kv_buffers[(0, g)].numel() * 4 for g in range(num_groups))
    obj_buffers = [torch.zeros(obj_bytes, dtype=torch.uint8) for _ in range(max_batch)]
    gpu_context.get_temp_object_group_buffer.side_effect = lambda s, og: obj_buffers[s]

    rope_state = _CBRopeState(
        head_size=head_size,
        is_neox_style=True,
        cos_sin_caches=[torch.zeros(64, head_size)],
        group_to_cache=[],
    )
    return eng, gpu_context, rope_state, obj_bytes


def _lazy_memory_obj(obj_bytes: int, address: int):
    """MemoryObj stand-in that passes the lazy-allocator gate and
    build_staging_copies' size/pointer checks."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.memory_allocators.lazy_memory_allocator import (
        LazyMemoryAllocator,
    )

    obj = MagicMock()
    obj.parent.return_value = MagicMock(spec=LazyMemoryAllocator)
    obj.raw_tensor = torch.zeros(obj_bytes, dtype=torch.uint8)
    obj.get_size.return_value = obj_bytes
    obj.data_ptr = obj.raw_tensor.data_ptr()
    obj.meta.address = address
    return obj


@native_retrieve_plan_required
def test_native_plan_specs_stamped_and_cached():
    """3 chunks, max_batch=2: per-group slot-mapping rows staged into the
    persistent device buffer and stamped into the cached invariant specs; a
    second build for the same context reuses the same spec objects (and the
    same staging buffer) and re-stamps them."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Chunks 0/1 shifted (old != cur), chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    group_specs, (_staging, _ropes, _scatters, step_offsets), keepalive = plan

    assert len(group_specs) == 2
    # keepalive: the persistent (num_groups, cap) device staging buffer.
    assert len(keepalive) == 1
    dev = keepalive[0]
    assert dev[0, :12].tolist() == [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    assert dev[1, :12].tolist() == [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]
    # Each cached spec is stamped with its row of the staging buffer.
    assert group_specs[0].slot_mapping_base == dev[0].data_ptr()
    assert group_specs[0].slot_mapping_capacity == 12
    assert group_specs[1].slot_mapping_base == dev[1].data_ptr()
    # Wave split: max_batch=2 -> double-buffered waves of 1 chunk each -> 3 steps.
    assert step_offsets.shape[0] == 3

    # Second build for the same context reuses the cached invariant specs
    # (same objects) and the same staging buffer, re-stamped per request.
    def pair2(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    plan2 = eng._build_cb_retrieve_plan_flat(
        gpu_context,
        rope_state,
        cpu_block_tables,
        [[pair2(4, 8, 200)]],
        max_batch=2,
    )
    assert plan2 is not None
    group_specs2, _, keepalive2 = plan2
    assert group_specs2[0] is group_specs[0]  # cached, not rebuilt
    assert keepalive2[0] is dev  # staging buffer reused, not reallocated
    assert group_specs2[0].slot_mapping_base == keepalive2[0][0].data_ptr()
    assert group_specs2[0].slot_mapping_capacity == 4
    # pos 4..8 -> block 11 -> slots 44..47 for group 0.
    assert keepalive2[0][0, :4].tolist() == [44, 45, 46, 47]


@native_retrieve_plan_required
def test_flat_plan_tables_encode_every_work_item():
    """The flat tables encode one staging row per chunk (dest = its wave
    slot's buffer), rope rows only for shifted chunks x groups, scatter rows
    for all chunks x groups with cumulative token offsets, and monotone
    per-step CSR offsets."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Chunks 0/1 shifted, chunk 2 prefix (old == cur).
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 8)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert plan is not None
    _specs, (staging, ropes, scatters, step_offsets), _keep = plan

    # 3 chunks -> 3 staging rows; wave=1 alternates slots 0,1,0. The
    # destinations live in the retrieve-owned private pool (NOT the shared
    # temp buffers), so assert the alternation contract on the pointers:
    # rows 0 and 2 share slot 0's buffer, row 1 uses a distinct one.
    assert staging.shape == (3, 4)
    dests = staging[:, 0].tolist()
    assert dests[0] == dests[2]
    assert dests[1] != dests[0]
    shared_ptr = gpu_context.get_temp_object_group_buffer(0, 0).data_ptr()
    assert shared_ptr not in dests, "staging must not target the shared pool"
    # Rope rows: 2 shifted chunks x 2 groups.
    assert ropes.shape == (4, 4)
    assert sorted(set(ropes[:, 2].tolist())) == [100, 104]  # old_st values
    # Scatter rows: 3 chunks x 2 groups, token offsets 0,4,8 repeated per group.
    assert scatters.shape == (6, 4)
    assert scatters[:, 2].tolist() == [0, 0, 4, 4, 8, 8]
    assert scatters[:, 3].tolist() == [4] * 6
    # Step CSR: 3 steps of 1 chunk; scatter ends = chunks x groups.
    assert step_offsets.shape == (3, 3)
    assert step_offsets[:, 0].tolist() == [1, 2, 3]
    assert step_offsets[:, 2].tolist() == [2, 4, 6]
    assert bool(np.all(np.diff(step_offsets[:, 1]) >= 0))


@native_retrieve_plan_required
def test_flat_plan_emits_no_rope_rows_for_a_nope_model():
    """A NoPE model (zero cos/sin caches) needs no re-RoPE for shifted
    matches, so the plan must emit no rope rows: the table unpack runs under
    the GIL and the executor would walk entries that cannot change a value.
    """
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()

    def pair(cur_st, cur_ed, old_st):
        return (
            SimpleNamespace(cur_st=cur_st, cur_ed=cur_ed, old_st=old_st),
            (_lazy_memory_obj(obj_bytes, address=cur_st * 1000),),
        )

    # Every chunk shifted (old != cur) — the worst case for rope rows.
    runs = [[pair(0, 4, 100), pair(4, 8, 104), pair(8, 12, 108)]]
    cpu_block_tables = [
        (np.array([10, 11, 12], dtype=np.int64), 4),
        (np.array([20, 21, 22], dtype=np.int64), 4),
    ]

    with_rope = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert with_rope is not None
    _, (_, ropes_r, scatters_r, offsets_r), _ = with_rope
    assert ropes_r.shape[0] == 6  # 3 shifted chunks x 2 groups

    rope_state.cos_sin_caches = []  # NoPE
    nope = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
    )
    assert nope is not None
    _, (staging_n, ropes_n, scatters_n, offsets_n), _ = nope

    assert ropes_n.shape[0] == 0, "NoPE must emit no rope rows"
    assert (offsets_n[:, 1] == 0).all(), "rope CSR offsets must stay at zero"
    # The actual data movement is untouched: same staging and scatter tables.
    assert np.array_equal(scatters_n, scatters_r)
    assert offsets_n.shape == offsets_r.shape
    assert np.array_equal(offsets_n[:, 0], offsets_r[:, 0])
    assert np.array_equal(offsets_n[:, 2], offsets_r[:, 2])
    assert staging_n.shape[0] == 3


@native_retrieve_plan_required
def test_flat_tables_alternate_disjoint_slot_halves():
    """Same double-buffer contract, asserted on the flat-table encoding."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context(
        max_batch=4
    )
    runs = [
        [
            (
                SimpleNamespace(cur_st=i * 4, cur_ed=i * 4 + 4, old_st=i * 4 + 100),
                (_lazy_memory_obj(obj_bytes, address=i * 4),),
            )
            for i in range(6)
        ]
    ]
    cpu_block_tables = [
        (np.arange(12, dtype=np.int64), 4),
        (np.arange(12, dtype=np.int64) + 100, 4),
    ]
    plan = eng._build_cb_retrieve_plan_flat(
        gpu_context, rope_state, cpu_block_tables, runs, max_batch=4
    )
    assert plan is not None
    _specs, (_staging, _ropes, scatters, step_offsets), _keep = plan

    prev_slots: set[int] | None = None
    c0 = 0
    for c1 in step_offsets[:, 2].tolist():
        slots = set(np.asarray(scatters[c0:c1, 1]).tolist())
        assert slots <= {0, 1} or slots <= {2, 3}, "step must stay in one half"
        if prev_slots is not None:
            assert not (slots & prev_slots)
        prev_slots = slots
        c0 = c1


@native_retrieve_plan_required
def test_native_plan_falls_back_for_non_lazy_objects():
    """A non-lazy-allocator memory object disables the native plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    obj = _lazy_memory_obj(obj_bytes, address=0)
    obj.parent.return_value = object()  # not a LazyMemoryAllocator
    runs = [[(SimpleNamespace(cur_st=0, cur_ed=4, old_st=100), (obj,))]]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )


@native_retrieve_plan_required
def test_native_plan_falls_back_for_compressed_group():
    """A compressed group (tokens != slots per block) disables the plan."""
    # Third Party
    import numpy as np

    eng, gpu_context, rope_state, obj_bytes = _build_plan_engine_and_context()
    gpu_context.kv_layer_groups_manager.kernel_groups[1].slots_per_block = 2
    runs = [
        [
            (
                SimpleNamespace(cur_st=0, cur_ed=4, old_st=100),
                (_lazy_memory_obj(obj_bytes, address=0),),
            )
        ]
    ]
    cpu_block_tables = [
        (np.array([10], dtype=np.int64), 4),
        (np.array([20], dtype=np.int64), 4),
    ]
    assert (
        eng._build_cb_retrieve_plan_flat(
            gpu_context, rope_state, cpu_block_tables, runs, max_batch=2
        )
        is None
    )


def test_reason_table():
    """The RetrieveReason -> (scatter_ran, publish) table is a contract:
    scatter_ran=False is the only client-degrade outcome, and
    CB_RETRIEVE_NOOP publishes only when reuse was actually lost."""
    # First Party
    from lmcache.v1.multiprocess.modules.blend.retrieve import RetrieveReason

    expected = {
        "ok": (True, False),
        "already_applied": (True, False),
        "matches_beyond_alloc": (True, False),
        "matches_straddle_alloc": (False, True),
        "no_object_keys": (True, True),
    }
    actual = {r.value: (r.scatter_ran, r.publish) for r in RetrieveReason}
    assert actual == expected


# ---------------------------------------------------------------------------
# L2: obj_keys cache lifecycle
# ---------------------------------------------------------------------------


def _fake_obj_key(chunk_hash: bytes, worker_id: int) -> SimpleNamespace:
    return SimpleNamespace(chunk_hash=chunk_hash, worker_id=worker_id)


def test_obj_keys_cache_round_trip_tp1():
    """At world_size=1, retrieve can rebuild from the session stash exactly."""
    # First Party
    from lmcache.v1.multiprocess.session import Session

    session = Session(request_id="req-1", hasher=MagicMock())

    # Simulate what the lookup's classify stores.
    chunk_hashes = [b"h1", b"h2", b"h3"]
    obj_keys_per_chunk = {h: [_fake_obj_key(h, 0)] for h in chunk_hashes}
    session.extras[BlendModule.UNRETRIEVED_KEYS_EXTRA] = {
        "read_locks": 1,
        "per_hash": obj_keys_per_chunk,
    }

    # Simulate retrieve consuming the stash (take-once).
    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    stash = session.extras.pop(BlendModule.UNRETRIEVED_KEYS_EXTRA, None)
    cached = stash["per_hash"] if stash else None

    assert cached is not None
    assert all(r.hash in cached for r in matches_sorted)
    rebuilt = [k for r in matches_sorted for k in cached[r.hash]]
    assert len(rebuilt) == 3
    assert [k.chunk_hash for k in rebuilt] == chunk_hashes
    # Stash is now empty: a second take (the session destroy listener after
    # a successful retrieve) releases nothing twice.
    assert session.extras.pop(BlendModule.UNRETRIEVED_KEYS_EXTRA, None) is None


def test_obj_keys_cache_round_trip_tp_expanded():
    """world_size>1: cached entry per hash is a list of length world_size,
    rebuilt list is flat chunk-major."""
    # First Party
    from lmcache.v1.multiprocess.session import Session

    session = Session(request_id="req-tp", hasher=MagicMock())

    ws = 4
    chunk_hashes = [b"h1", b"h2"]
    per_hash = {h: [_fake_obj_key(h, w) for w in range(ws)] for h in chunk_hashes}
    session.extras[BlendModule.UNRETRIEVED_KEYS_EXTRA] = {
        "read_locks": 1,
        "per_hash": per_hash,
    }

    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    stash = session.extras.pop(BlendModule.UNRETRIEVED_KEYS_EXTRA, None)
    cached = stash["per_hash"] if stash else None
    assert cached is not None
    rebuilt = [k for r in matches_sorted for k in cached[r.hash]]
    # Length = 2 chunks × 4 workers.
    assert len(rebuilt) == 8
    # Chunk-major: first 4 entries are h1's workers 0..3, then h2's.
    assert [k.chunk_hash for k in rebuilt[:4]] == [b"h1"] * 4
    assert [k.worker_id for k in rebuilt[:4]] == [0, 1, 2, 3]
    assert [k.chunk_hash for k in rebuilt[4:]] == [b"h2"] * 4


def test_obj_keys_cache_miss_falls_back():
    """If the cache doesn't contain every match's hash, retrieve must
    fall back to recompute (handled in the engine; this test just pins
    the detection logic)."""
    cached = {b"h1": ["k1"]}
    matches = [SimpleNamespace(hash=b"h1"), SimpleNamespace(hash=b"h_missing")]
    all_present = all(r.hash in cached for r in matches)
    assert all_present is False


# ---------------------------------------------------------------------------
# L2: sparse-prefetch read-lock release after retrieve
# ---------------------------------------------------------------------------


def _release_match(i: int):
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    chunk = 256
    return CBMatchResult(
        old_st=i * chunk,
        old_ed=(i + 1) * chunk,
        cur_st=(i + 3) * chunk,
        cur_ed=(i + 4) * chunk,
        hash=bytes([i]) * 32,
    )


def _release_applied(matches, applied, n_read, stream):
    """Call the helper unbound on a bare mock ``self``; return (n, keys, cb)."""

    keys = [f"k{i}g{g}" for i in range(len(matches)) for g in range(n_read)]
    with patch.object(retrieve_mod, "submit_callback_to_stream") as cb:
        n = BlendModule._release_applied_read_locks(
            MagicMock(), matches, applied, keys, n_read, stream
        )
    return n, keys, cb


def test_release_applied_read_locks_all_keys_on_retrieve_stream():
    """Every applied match releases its n_read keys, chunk-major, via a
    ``finish_read_prefetched`` callback ordered on the retrieve stream."""
    stream = MagicMock(name="retrieve_cupy_stream")
    matches = [_release_match(i) for i in range(3)]
    n, keys, cb = _release_applied(matches, matches, 2, stream)

    assert n == 6
    cb.assert_called_once()
    got_stream, kind, payload = cb.call_args.args
    assert got_stream is stream
    assert kind == "finish_read_prefetched"
    assert payload == keys


def test_release_applied_read_locks_keeps_dropped_matches_locked():
    """Beyond-slot matches are retried on vLLM's full-alloc call: not released."""
    matches = [_release_match(i) for i in range(4)]
    applied = [matches[0], matches[2]]
    n, keys, cb = _release_applied(matches, applied, 1, MagicMock())

    assert n == 2
    assert cb.call_args.args[2] == [keys[0], keys[2]]


def test_release_applied_read_locks_nothing_applied():
    n, _keys, cb = _release_applied([_release_match(0)], [], 1, MagicMock())

    assert n == 0
    cb.assert_not_called()


def test_release_applied_read_locks_selects_by_identity():
    """Equal-valued but distinct match objects: only the scattered one is
    released (mirrors how ``pairs`` is built in the retrieve)."""
    a, b = _release_match(0), _release_match(0)
    n, keys, cb = _release_applied([a, b], [b], 1, MagicMock())

    assert n == 1
    assert cb.call_args.args[2] == [keys[1]]


# ---------------------------------------------------------------------------
# L2: sparse-prefetch read-lock release when the request never retrieves
# ---------------------------------------------------------------------------
#
# The unified lookup read-locks every found chunk's object keys and stashes
# them on the session (``Session.extras``) for the retrieve. A client that
# drops all its matches -- e.g. every match falls inside vLLM's local prefix
# cache coverage -- never sends ``CB_RETRIEVE_PRE_COMPUTED``, and before this
# fix nothing released those locks: the retrieve's orphan sweep never ran, and
# ``free_lookup_locks`` covers only the prefix leg's lock model. The chunks
# stayed pinned in L1 for the server's lifetime, with counts stacking on every
# repeat lookup.
#
# The fix: ``BlendModule`` registers a ``SessionManager`` destroy listener that
# releases whatever the request never consumed -- covering END_SESSION removal
# at request end and the TTL reaper for clients that died without one.
#
# Unlike the mocked sections above, these drive the real ``BlendModule``,
# ``BlendTokenRangeMatcher``, ``SessionManager`` and key expansion; only the
# storage manager is a lock-counting fake, so the assertions are on actual
# lock accounting.

_UNRETRIEVED_CHUNK = 256
_UNRETRIEVED_N_CHUNKS = 4
# A TTL no test can reach, so only the explicit ttl=0.0 case reaps.
_UNRETRIEVED_TTL_NEVER = 3600.0


class _LockCountingStorageManager:
    """Counts read locks per key: the sparse prefetch locks every submitted
    key; ``finish_read_prefetched`` is the only release. Over-release raises.
    """

    def __init__(self) -> None:
        self.locks: dict = {}

    def submit_prefetch_task(self, spec, external_request_id=None):
        # First Party
        from lmcache.v1.distributed.api import TrimPolicy

        if spec.policy == TrimPolicy.SPARSE:
            n = int(getattr(spec, "num_kv_readers", 1) or 1)
            for key in spec.keys:
                self.locks[key] = self.locks.get(key, 0) + n
        handle = MagicMock()
        handle.keys = list(spec.keys)
        handle.l2_orig_indices = []
        return handle

    def query_prefetch_status(self, handle):
        bitmap = MagicMock()
        bitmap.get_indices_list.return_value = list(range(len(handle.keys)))
        return bitmap

    def finish_read_prefetched(self, keys, read_locks: int = 1) -> None:
        for key in keys:
            held = self.locks.get(key, 0)
            if held < read_locks:
                raise AssertionError(f"over-release on {key}")
            if held == read_locks:
                del self.locks[key]
            else:
                self.locks[key] = held - read_locks

    def outstanding(self) -> int:
        return sum(self.locks.values())


def _unretrieved_ctx(
    storage_manager: _LockCountingStorageManager,
    ttl: float = _UNRETRIEVED_TTL_NEVER,
) -> MagicMock:
    """Mock server context with a REAL SessionManager (no cleanup thread)."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc
    from lmcache.v1.multiprocess.session import SessionManager

    ctx = MagicMock()
    ctx.chunk_size = _UNRETRIEVED_CHUNK
    ctx.storage_manager = storage_manager
    ctx.event_bus.has_subscribers.return_value = False
    # Prefix leg: no full chunk hashes -> handle None -> 0 coverage.
    ctx.token_hasher.compute_chunk_hashes.return_value = []
    # One registered attention-only object group.
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1], world_size=1, group_kinds=("attention",)
    )
    ctx.session_manager = SessionManager(
        hasher=MagicMock(), ttl=ttl, cleanup_interval=None
    )
    return ctx


def _unretrieved_blend(ctx: MagicMock):
    """The real BlendModule under the mocked context."""

    return BlendModule(ctx, lmcache_driven_transfer=MagicMock())


def _run_unretrieved_lookup(blend, request_id: str, num_kv_readers: int = 1):
    """Register ``_UNRETRIEVED_N_CHUNKS`` fingerprints and run a lookup that
    finds them all shifted (prefix coverage 0), i.e. the sparse leg locks
    every chunk with ``num_kv_readers`` locks each. Returns the lookup key."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

    n_tokens = _UNRETRIEVED_N_CHUNKS * _UNRETRIEVED_CHUNK
    stored_tokens = list(range(1000, 1000 + n_tokens))
    token_hashes = [
        f"{request_id}-hash{i}".encode() for i in range(_UNRETRIEVED_N_CHUNKS)
    ]
    indexed = blend._token_range_matcher.on_new_token_hashes(
        stored_tokens, token_hashes, start_chunk_idx=0, position_offset=0
    )
    assert indexed == _UNRETRIEVED_N_CHUNKS

    query = list(range(50_000, 50_128)) + stored_tokens
    key = IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=num_kv_readers,
        worker_id=None,
        token_ids=tuple(query),
        start=0,
        end=len(query),
        request_id=request_id,
    )
    result = blend.cb_unified_lookup(key, tp_size=1)
    assert result is not None
    assert result.prefix_coverage_tokens == 0
    assert len(result.non_prefix_segments) == _UNRETRIEVED_N_CHUNKS
    return key


def test_session_end_releases_unretrieved_sparse_locks():
    """The leak scenario: lookup locks N chunks, the client never retrieves
    (all matches shadowed by its local prefix cache), the request ends.
    END_SESSION's session removal must release every sparse read lock."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)

    _run_unretrieved_lookup(blend, "req-shadowed")
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS

    # No retrieve. Request ends: END_SESSION removes the session.
    ctx.session_manager.remove("req-shadowed")
    assert storage.outstanding() == 0


def test_session_end_releases_whole_reservation_mla():
    """MLA-style lookup (num_kv_readers=8) reserves 8 locks per key; the
    destroy listener must release the whole reservation, not 1 per key."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)

    _run_unretrieved_lookup(blend, "req-mla", num_kv_readers=8)
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS * 8

    ctx.session_manager.remove("req-mla")
    assert storage.outstanding() == 0


def test_repeat_lookup_releases_superseded_stash():
    """A second lookup for the same request (e.g. re-issued after a
    preemption) replaces the stash; the superseded reservation must be
    released at overwrite, and the live one at session end."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)

    key = _run_unretrieved_lookup(blend, "req-repeat", num_kv_readers=2)
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS * 2

    # Repeat lookup: a fresh reservation is taken and the previous stash's
    # reservation is released at overwrite — never both held.
    result = blend.cb_unified_lookup(key, tp_size=1)
    assert result is not None
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS * 2

    ctx.session_manager.remove("req-repeat")
    assert storage.outstanding() == 0


def test_retrieve_take_prevents_double_release_at_session_end():
    """A consumed stash releases nothing at session end: the retrieve's
    take empties it, so the destroy listener is a no-op (the counting fake
    raises on any over-release)."""

    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)

    _run_unretrieved_lookup(blend, "req-retrieved")

    # Emulate the retrieve's consumption + release of the taken keys.
    session = ctx.session_manager.get("req-retrieved")
    assert session is not None
    stash = session.extras.pop(BlendModule.UNRETRIEVED_KEYS_EXTRA, None)
    assert stash is not None and len(stash["per_hash"]) == _UNRETRIEVED_N_CHUNKS
    storage.finish_read_prefetched(
        [key for keys in stash["per_hash"].values() for key in keys],
        read_locks=stash["read_locks"],
    )
    assert storage.outstanding() == 0

    # Session end must not release again.
    ctx.session_manager.remove("req-retrieved")
    assert storage.outstanding() == 0


def test_ttl_cleanup_releases_unretrieved_sparse_locks():
    """A session reaped by TTL cleanup (client died without END_SESSION)
    releases its unretrieved locks through the same destroy listener."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage, ttl=0.0)
    blend = _unretrieved_blend(ctx)

    _run_unretrieved_lookup(blend, "req-abandoned")
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS

    assert ctx.session_manager.cleanup_expired() == 1
    assert storage.outstanding() == 0


def test_close_unregisters_the_destroy_listener():
    """After BlendModule.close(), destroying sessions calls nothing on the
    (now torn down) module."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)

    _run_unretrieved_lookup(blend, "req-late")
    blend.close()
    # The stash is still on the session; with the listener gone, removal
    # releases nothing (server shutdown path -- locks die with the process).
    ctx.session_manager.remove("req-late")
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS


def test_destroy_listener_failure_does_not_break_removal():
    """A raising listener is logged and swallowed; removal still completes."""
    # First Party
    from lmcache.v1.multiprocess.session import Session, SessionManager

    def _boom(session: Session) -> None:
        raise RuntimeError("listener failure")

    manager = SessionManager(hasher=MagicMock(), cleanup_interval=None)
    manager.destroy_listeners.append(_boom)
    manager.get_or_create("req-x")
    assert manager.remove("req-x") is not None
    assert manager.get("req-x") is None
