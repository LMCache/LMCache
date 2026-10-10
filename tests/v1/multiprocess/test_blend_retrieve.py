# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the blend retrieve planner (native flat-plan fast path):
invariant-spec caching and re-stamping, work-table encoding, double-buffered
wave slotting, and the fallback gates (non-lazy objects, compressed groups).

Moved from test_blend_load_store_opts.py in the blend package split."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import random
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import device_ops  # noqa: F401
from lmcache.v1.multiprocess.modules.blend import retrieve as retrieve_mod
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.read_locks import ReadLockReservation
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
        "read_locks_not_held": (False, True),
    }
    actual = {r.value: (r.scatter_ran, r.publish) for r in RetrieveReason}
    assert actual == expected


# ---------------------------------------------------------------------------
# L2: this rank's object keys from the lookup's reservation
# ---------------------------------------------------------------------------


def _fake_obj_key(chunk_hash: bytes, worker_id) -> tuple:
    """A hashable stand-in for an ObjectKey (the ledger keys a dict by it)."""
    return (chunk_hash, worker_id)


def _reservation(per_hash, ends=None, read_locks: int = 1) -> ReadLockReservation:
    return ReadLockReservation(
        read_locks=read_locks,
        per_hash=per_hash,
        ends=ends if ends is not None else dict.fromkeys(per_hash, 0),
    )


def _ipc_key(worker_id, world_size, end: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        worker_id=worker_id, world_size=world_size, end=end, request_id="req"
    )


def test_rank_keys_tp1_returns_every_reserved_key():
    """At world_size=1 a match's keys are its reserved keys, chunk-major."""
    hashes = [b"h1", b"h2", b"h3"]
    res = _reservation({h: [_fake_obj_key(h, 0)] for h in hashes})
    matches = [SimpleNamespace(hash=h) for h in hashes]
    out = res.rank_keys(matches, _ipc_key(None, 1), 1)
    assert [ks[0][0] for ks in out] == hashes


def test_rank_keys_tp_expanded_selects_this_ranks_keys():
    """world_size>1: the reservation holds every rank's key per read group,
    group-major and rank-minor; each rank gets its own, or TP mispairs."""
    ws, n_read = 4, 2
    per_hash = {
        b"h1": [_fake_obj_key(b"h1", (g, r)) for g in range(n_read) for r in range(ws)]
    }
    res = _reservation(per_hash)
    out = res.rank_keys([SimpleNamespace(hash=b"h1")], _ipc_key(2, ws), n_read)
    assert [k[1] for k in out[0]] == [(0, 2), (1, 2)]


def test_rank_keys_none_for_a_hash_the_reservation_lacks():
    """A match the reservation has no keys for yields None; the retrieve
    derives its keys, and its claim then fails unless they are held."""
    res = _reservation({b"h1": ["k1"]})
    matches = [SimpleNamespace(hash=b"h1"), SimpleNamespace(hash=b"h_missing")]
    assert res.rank_keys(matches, _ipc_key(None, 1), 1) == [["k1"], None]


def test_assemble_obj_keys_derives_only_the_missing_matches(monkeypatch):
    derived_for = []

    def fake_derive(key, hashes, gids):
        derived_for.append(list(hashes))
        return [f"d-{h.decode()}-{g}" for h in hashes for g in gids]

    monkeypatch.setattr(retrieve_mod, "_cb_chunk_major_object_keys", fake_derive)
    matches = [SimpleNamespace(hash=b"a"), SimpleNamespace(hash=b"b")]
    out = retrieve_mod._assemble_obj_keys(
        _ipc_key(None, 1), matches, [["a0", "a1"], None], (0, 1), 2
    )
    assert out == ["a0", "a1", "d-b-0", "d-b-1"]
    assert derived_for == [[b"b"]]


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
        if spec.fetching_policy == "full":
            n = int(getattr(spec, "num_kv_readers", 1) or 1)
            for row in spec.key_groups:
                for key in row.keys:
                    self.locks[key] = self.locks.get(key, 0) + n
        handle = MagicMock()
        handle.key_groups = list(spec.key_groups)
        handle.total_requested_keys = 0
        return handle

    def query_prefetch_status(self, handle):
        # Every submitted key is found and was loaded from L2: one all-set
        # bitmap per row.
        # First Party
        from lmcache.lmcache_native import Bitmap
        from lmcache.v1.distributed.api import PrefetchResult

        def rows():
            return [Bitmap(len(row.keys), len(row.keys)) for row in handle.key_groups]

        return PrefetchResult(
            hit_cells=rows(),
            l1_hit_cells=[Bitmap(len(row.keys)) for row in handle.key_groups],
            l2_hit_cells=rows(),
        )

    def finish_read_prefetched(self, keys, read_locks: int = 1, l1_owners=None) -> None:
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


def test_a_retrieved_reservation_releases_nothing_twice_at_session_end():
    """A retrieve that claims every key and releases its claims leaves
    nothing for session end (the counting fake raises on over-release)."""

    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-retrieved")
    session = ctx.session_manager.get("req-retrieved")

    claim = blend._claim_read_locks(
        session, _reserved_matches(session), key, (0,), slot_bound=key.end
    )
    assert claim is not None
    storage.finish_read_prefetched(claim[0])  # the retrieve's own release
    assert storage.outstanding() == 0

    # Called directly: the session manager logs and swallows a listener that
    # raises, so a release with nothing left to release must be checked here.
    blend._release_unretrieved_locks(session)
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


# ---------------------------------------------------------------------------
# Sparse classify: found / partial / stale chunk accounting
# ---------------------------------------------------------------------------


def _classify_engine():
    """Engine with the real ``_sparse_classify`` bound over mocked state."""
    # Standard
    import threading

    eng = MagicMock(spec=BlendModule)
    eng._sparse_classify = BlendModule._sparse_classify.__get__(eng)
    eng.UNRETRIEVED_KEYS_EXTRA = BlendModule.UNRETRIEVED_KEYS_EXTRA
    eng._STALE_STRIKE_THRESHOLD = 2
    eng._pending_fp_lock = threading.Lock()
    eng._cb_retain_lock = threading.Lock()
    eng._pending_fp_hashes = set()
    eng._stale_strike = {}
    eng._ctx = MagicMock()
    eng._ctx.session_manager.get_or_create.return_value = SimpleNamespace(extras={})
    eng._event_bus = MagicMock()
    eng._token_range_matcher = MagicMock()
    return eng


def _classify_key(request_id: str = "req-classify", num_kv_readers: int = 1):
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=num_kv_readers,
        worker_id=None,
        token_ids=(1, 2, 3),
        start=0,
        end=3,
        request_id=request_id,
    )


def _classify_match(col: int, h: bytes):
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    n = _UNRETRIEVED_CHUNK
    return CBMatchResult(
        old_st=0, old_ed=n, cur_st=col * n, cur_ed=(col + 1) * n, hash=h
    )


@pytest.mark.parametrize("l1_owners", [None, {"k-partial-r0": 1}])
def test_sparse_classify_partial_column_releases_and_takes_no_strike(l1_owners):
    """A chunk with some but not all rows loaded is not blendable and not
    stale: its landed keys' locks are released immediately (not held for the
    read TTL) and it takes no eviction strike -- the content is still stored,
    it merely did not fit L1."""
    # First Party
    from lmcache.lmcache_native import Bitmap

    eng = _classify_engine()
    key = _classify_key(num_kv_readers=8)
    matches = [_classify_match(0, b"whole"), _classify_match(1, b"partial")]
    per_hash_obj_keys = {
        b"whole": ["k-whole-r0", "k-whole-r1"],
        b"partial": ["k-partial-r0", "k-partial-r1"],
    }
    hash_to_col = {b"whole": 0, b"partial": 1}
    # Row 0 loaded both columns; row 1 only column 0.
    row0 = Bitmap(2, 2)
    row1 = Bitmap(2)
    row1.set(0)

    found = eng._sparse_classify(
        key,
        matches,
        [row0, row1],
        per_hash_obj_keys,
        hash_to_col,
        l1_owners=l1_owners,
    )

    assert [r.hash for r in found] == [b"whole"]
    # The partial chunk's landed key released the whole reservation, now.
    eng._ctx.storage_manager.finish_read_prefetched.assert_called_once_with(
        ["k-partial-r0"], read_locks=8, l1_owners=l1_owners
    )
    # No strike and no matcher eviction for the partial chunk.
    assert eng._stale_strike == {}
    eng._token_range_matcher.remove_chunks.assert_not_called()


def test_sparse_classify_fully_missing_chunk_still_strikes():
    """A chunk with NO rows loaded keeps the stale-strike path: struck once
    below the threshold, evicted from the matcher at the threshold."""
    # First Party
    from lmcache.lmcache_native import Bitmap

    eng = _classify_engine()
    key = _classify_key()
    matches = [_classify_match(0, b"gone")]
    per_hash_obj_keys = {b"gone": ["k-gone-r0", "k-gone-r1"]}
    hash_to_col = {b"gone": 0}
    empty_rows = [Bitmap(1), Bitmap(1)]

    assert (
        eng._sparse_classify(key, matches, empty_rows, per_hash_obj_keys, hash_to_col)
        == []
    )
    assert eng._stale_strike == {b"gone": 1}
    eng._token_range_matcher.remove_chunks.assert_not_called()
    eng._ctx.storage_manager.finish_read_prefetched.assert_not_called()

    assert (
        eng._sparse_classify(key, matches, empty_rows, per_hash_obj_keys, hash_to_col)
        == []
    )
    eng._token_range_matcher.remove_chunks.assert_called_once_with([b"gone"])
    assert eng._stale_strike == {}


def test_sparse_classify_unstaged_but_found_chunk_takes_no_strike():
    """With the prefetch's found view available, a chunk that exists in
    storage (pinned by the L2 lookup) but landed NO rows is skipped -- no
    strike, nothing to release -- while a chunk absent from the found view
    keeps the stale path."""
    # First Party
    from lmcache.lmcache_native import Bitmap

    eng = _classify_engine()
    key = _classify_key()
    matches = [_classify_match(0, b"unstaged"), _classify_match(1, b"gone")]
    per_hash_obj_keys = {
        b"unstaged": ["k-u-r0", "k-u-r1"],
        b"gone": ["k-g-r0", "k-g-r1"],
    }
    hash_to_col = {b"unstaged": 0, b"gone": 1}
    landed_rows = [Bitmap(2), Bitmap(2)]  # nothing landed at all
    avail0 = Bitmap(2)
    avail0.set(0)  # column 0 exists in both rows; column 1 nowhere
    avail1 = Bitmap(2)
    avail1.set(0)

    found = eng._sparse_classify(
        key, matches, landed_rows, per_hash_obj_keys, hash_to_col, [avail0, avail1]
    )

    assert found == []
    # "unstaged" exists -> skipped without a strike or a release.
    eng._ctx.storage_manager.finish_read_prefetched.assert_not_called()
    # "gone" is absent from every tier -> the stale path, unchanged.
    assert eng._stale_strike == {b"gone": 1}


# ---------------------------------------------------------------------------
# Read-lock reservation: a request retrieved over several calls (one per
# engine prefill chunk) claims one lock per key it reads, releases the
# matches no later call can send, and never reads or releases a lock it no
# longer holds.
# ---------------------------------------------------------------------------


def _reserved_matches(session) -> list:
    """The lookup's matches as the client would send them, by position."""
    res = session.extras[BlendModule.UNRETRIEVED_KEYS_EXTRA]
    return [
        SimpleNamespace(hash=h, cur_ed=res.ends[h])
        for h in sorted(res.per_hash, key=lambda h: res.ends[h])
    ]


def test_reservation_holds_every_key_read_locks_times():
    res = _reservation({b"a": ["a0", "a1"], b"b": ["b0"]}, read_locks=3)
    assert res.held == {"a0": 3, "a1": 3, "b0": 3}


def test_claim_is_all_or_nothing():
    res = _reservation({b"a": ["a0"], b"b": ["b0"]})
    assert res.claim(["b0"])
    assert not res.claim(["a0", "b0"]), "b0 has no lock left"
    assert res.held == {"a0": 1, "b0": 0}, "a failed claim takes nothing"


def test_claim_counts_a_repeated_key_per_occurrence():
    res = _reservation({b"a": ["a0"]}, read_locks=1)
    assert not res.claim(["a0", "a0"]), "one lock cannot cover two reads"
    assert res.held == {"a0": 1}


def test_unclaim_hands_a_lock_back():
    res = _reservation({b"a": ["a0"]})
    assert res.claim(["a0"])
    res.unclaim(["a0"])
    assert res.claim(["a0"])


def test_each_mla_reader_claims_one_lock():
    """MLA shares a key across num_kv_readers readers: each claims one."""
    res = _reservation({b"a": ["a0"]}, read_locks=2)
    assert res.claim(["a0"]) and res.claim(["a0"])
    assert not res.claim(["a0"])


def test_sweep_releases_unsent_matches_the_call_has_passed():
    res = _reservation(
        {b"a": ["a0"], b"b": ["b0"], b"c": ["c0"]},
        ends={b"a": 256, b"b": 512, b"c": 768},
    )
    # The call was sent "b": "a" ends before it, so no later call sends it;
    # "c" may still come in a later window.
    assert res.sweep({b"b"}, upto=512, final=False) == {1: ["a0"]}
    assert set(res.per_hash) == {b"b", b"c"}


def test_final_sweep_releases_every_unsent_match():
    res = _reservation({b"a": ["a0"], b"b": ["b0"]}, ends={b"a": 256, b"b": 512})
    assert res.sweep({b"a"}, upto=256, final=True) == {1: ["b0"]}


def test_sweep_never_releases_a_claimed_lock():
    res = _reservation({b"a": ["a0"], b"b": ["b0"]}, ends={b"a": 256, b"b": 512})
    assert res.claim(["a0"])
    # A later call sent "b": "a" is swept, but its only lock is claimed by
    # the earlier call, which releases it itself.
    assert res.sweep({b"b"}, upto=512, final=True) == {}


def test_release_all_groups_what_is_still_held():
    res = _reservation({b"a": ["a0", "a1"], b"b": ["b0"]}, read_locks=2)
    assert res.claim(["a0"])
    assert res.release_all() == {1: ["a0"], 2: ["a1", "b0"]}
    assert res.held == {} and res.per_hash == {}


def test_lookup_installs_a_reservation_with_match_ends():
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    _run_unretrieved_lookup(blend, "req-ends", num_kv_readers=2)
    res = ctx.session_manager.get("req-ends").extras[BlendModule.UNRETRIEVED_KEYS_EXTRA]
    # The query is 128 filler tokens, then the stored chunks.
    assert sorted(res.ends.values()) == [
        128 + (i + 1) * _UNRETRIEVED_CHUNK for i in range(_UNRETRIEVED_N_CHUNKS)
    ]
    assert set(res.held.values()) == {2}


def test_windows_claim_their_matches_and_release_the_rest():
    """Chunked prefill: each call claims its window's matches; a match no
    call sends (the engine recomputes it) is released once a later call has
    passed it, and on the call that sees the whole prompt allocated."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-win")
    session = ctx.session_manager.get("req-win")
    m0, m1, m2, m3 = _reserved_matches(session)

    # Window 1 sends m0. m1 straddles the window's end, so the client never
    # sends it; it stays held until a later call passes it.
    claim = blend._claim_read_locks(
        session, [m0], key, (0,), slot_bound=m0.cur_ed + 128
    )
    assert claim is not None
    storage.finish_read_prefetched(claim[0])
    assert storage.outstanding() == 3

    # Window 2 (the last) sends m2: m1 and m3 can no longer be sent.
    claim = blend._claim_read_locks(session, [m2], key, (0,), slot_bound=key.end)
    assert claim is not None
    assert storage.outstanding() == 1  # only m2, claimed by this call
    storage.finish_read_prefetched(claim[0])
    assert storage.outstanding() == 0

    ctx.session_manager.remove("req-win")  # nothing left to release twice
    assert storage.outstanding() == 0


def test_single_shot_retrieve_releases_every_unsent_match():
    """One call with the whole prompt allocated releases every match it was
    not given, as a single-shot retrieve always has."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-single", num_kv_readers=2)
    session = ctx.session_manager.get("req-single")
    m1 = _reserved_matches(session)[1]

    claim = blend._claim_read_locks(session, [m1], key, (0,), slot_bound=key.end)
    assert claim is not None
    # Every other match released in full; m1 keeps one lock per reader.
    assert storage.outstanding() == 2


def test_unknown_slot_bound_releases_nothing_early():
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-nobound")
    session = ctx.session_manager.get("req-nobound")
    m0 = _reserved_matches(session)[0]

    claim = blend._claim_read_locks(session, [m0], key, (0,), slot_bound=None)
    assert claim is not None
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS
    storage.finish_read_prefetched(claim[0])
    ctx.session_manager.remove("req-nobound")
    assert storage.outstanding() == 0


def test_a_released_key_is_never_read_again():
    """Re-sending a match whose lock this request already released (e.g. a
    re-scatter) claims nothing: reading it would rely on another request's
    lock, and releasing it would take that lock."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-again")
    session = ctx.session_manager.get("req-again")
    m0 = _reserved_matches(session)[0]

    claim = blend._claim_read_locks(session, [m0], key, (0,), slot_bound=None)
    storage.finish_read_prefetched(claim[0])
    # Another request now holds the same object.
    storage.locks[claim[0][0]] = 1

    assert blend._claim_read_locks(session, [m0], key, (0,), None) is None
    ctx.session_manager.remove("req-again")
    assert storage.locks == {claim[0][0]: 1}, "the other request's lock is intact"


def test_a_failed_read_spends_the_claim():
    """A failed read releases the keys it did read (one lock each); a retry
    of the same matches must not read them again."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-fail")
    session = ctx.session_manager.get("req-fail")
    m0, m1 = _reserved_matches(session)[:2]

    keys = blend._claim_read_locks(session, [m0, m1], key, (0,), None)[0]
    # The storage read context: m0 read and released, m1 not readable.
    storage.finish_read_prefetched([keys[0]])
    del storage.locks[keys[1]]

    assert blend._claim_read_locks(session, [m0, m1], key, (0,), None) is None
    ctx.session_manager.remove("req-fail")
    assert storage.outstanding() == 0


def test_a_retrieve_after_session_end_reads_nothing():
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-late-retrieve")
    session = ctx.session_manager.get("req-late-retrieve")
    m0 = _reserved_matches(session)[0]

    ctx.session_manager.remove("req-late-retrieve")
    assert storage.outstanding() == 0
    assert blend._claim_read_locks(session, [m0], key, (0,), key.end) is None


def test_repeat_lookup_releases_only_unclaimed_locks():
    """A repeat lookup (e.g. after preemption) while a retrieve still owns
    its claims releases only the superseded reservation's unclaimed locks;
    the retrieve releases its own."""
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-repeat-claim")
    session = ctx.session_manager.get("req-repeat-claim")
    m0 = _reserved_matches(session)[0]

    in_flight = blend._claim_read_locks(session, [m0], key, (0,), None)[0]
    assert blend.cb_unified_lookup(key, tp_size=1) is not None
    # The new reservation holds all 4; the in-flight retrieve holds m0's.
    assert storage.outstanding() == _UNRETRIEVED_N_CHUNKS + 1

    storage.finish_read_prefetched(in_flight)
    ctx.session_manager.remove("req-repeat-claim")
    assert storage.outstanding() == 0


def test_mla_readers_each_read_once_and_session_end_releases_the_rest():
    storage = _LockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    key = _run_unretrieved_lookup(blend, "req-mla-read", num_kv_readers=2)
    session = ctx.session_manager.get("req-mla-read")
    m0 = _reserved_matches(session)[0]

    for _reader in range(2):
        claim = blend._claim_read_locks(session, [m0], key, (0,), None)
        assert claim is not None
        storage.finish_read_prefetched(claim[0])
    assert blend._claim_read_locks(session, [m0], key, (0,), None) is None

    ctx.session_manager.remove("req-mla-read")
    assert storage.outstanding() == 0


def test_handshake_reports_version_2_and_accepts_version_1_clients():
    # First Party
    from lmcache.v1.multiprocess.modules.blend.module import _handshake_response

    assert _handshake_response(1) == (2, True)
    assert _handshake_response(2) == (2, True)
    assert _handshake_response(3) == (2, False)


class _ThreadSafeLockCountingStorageManager(_LockCountingStorageManager):
    """The lock-counting fake, safe to release from several threads."""

    def __init__(self) -> None:
        super().__init__()
        self._mutex = threading.Lock()

    def finish_read_prefetched(self, keys, read_locks: int = 1, l1_owners=None) -> None:
        with self._mutex:
            super().finish_read_prefetched(keys, read_locks, l1_owners)


class _SlowLookupDict(dict):
    """A dict whose ``get`` yields the GIL after reading. Used for the
    reservation's ledger, a claim reads a key's count, yields, then counts it
    down: a racing thread the module lock does not exclude (session end, or
    the other rank) then sees the same lock as held and releases or claims it
    a second time."""

    def get(self, key, default=None):
        value = super().get(key, default)
        time.sleep(0.0002)
        return value


def _race_tp_ranks_against_session_end(
    round_no: int, n_chunks: int, end_delay_s: float
) -> None:
    """One race: two TP ranks retrieve ``n_chunks`` windows (claiming their
    keys, releasing what they read, sweeping passed matches) while session
    end releases the reservation ``end_delay_s`` after they start. Asserts
    every read lock was released exactly once."""
    n_hashes, ws, n_read, chunk = 8, 2, 2, 2
    storage = _ThreadSafeLockCountingStorageManager()
    ctx = _unretrieved_ctx(storage)
    blend = _unretrieved_blend(ctx)
    rid = f"req-tp-{round_no}"
    session = ctx.session_manager.get_or_create(rid)
    hashes = [f"{rid}-h{i}".encode() for i in range(n_hashes)]
    per_hash = _SlowLookupDict()
    for h in hashes:
        # Group-major, rank-minor: [g0r0, g0r1, g1r0, g1r1].
        per_hash[h] = [(h, g, r) for g in range(n_read) for r in range(ws)]
        for k in per_hash[h]:
            storage.locks[k] = 1
    # Hash i ends at token i + 1; window c is allocated up to (c + 1) * chunk.
    reservation = ReadLockReservation(
        read_locks=1,
        per_hash=per_hash,
        ends={h: i + 1 for i, h in enumerate(hashes)},
    )
    reservation.held = _SlowLookupDict(reservation.held)
    session.extras[BlendModule.UNRETRIEVED_KEYS_EXTRA] = reservation
    errors: list[BaseException] = []
    start = threading.Barrier(ws + 1)

    def rank(worker_id: int) -> None:
        try:
            start.wait()
            for c in range(n_chunks):
                # The engine recomputes the window's last match: never sent.
                window = hashes[c * chunk : (c + 1) * chunk]
                matches = [
                    SimpleNamespace(hash=h, cur_ed=hashes.index(h) + 1)
                    for h in window[:-1]
                ]
                claim = blend._claim_read_locks(
                    session,
                    matches,
                    _ipc_key(worker_id, ws, end=n_hashes),
                    tuple(range(n_read)),
                    (c + 1) * chunk,
                )
                if claim is None:
                    return
                storage.finish_read_prefetched(claim[0])  # read and released
        except BaseException as exc:  # reported by the assert below
            errors.append(exc)

    def end_session() -> None:
        try:
            start.wait()
            time.sleep(end_delay_s)
            ctx.session_manager.remove(rid)
        except BaseException as exc:  # reported by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=rank, args=(r,)) for r in range(ws)]
    threads.append(threading.Thread(target=end_session))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a thread hung"
    assert not errors, errors
    assert storage.outstanding() == 0, f"round {round_no}: leaked locks"


def test_tp_ranks_and_session_end_release_each_lock_once(monkeypatch):
    """Two TP ranks claim and release their keys window by window while
    session end races them. Whatever the interleaving, every read lock is
    released exactly once: by the rank that claimed it, by a sweep, or by
    session end. The fake raises on any over-release, and a leak leaves a
    lock outstanding."""
    # After session end a rank derives keys for its matches; they are never
    # held, so its claim fails without reading.
    monkeypatch.setattr(
        retrieve_mod,
        "_cb_chunk_major_object_keys",
        lambda key, hashes, gids: [("derived", h, g) for h in hashes for g in gids],
    )
    rng = random.Random(0)
    for round_no in range(200):
        # Session end lands anywhere from before the first claim to after the
        # last, and the ranks may stop short of the last window.
        _race_tp_ranks_against_session_end(
            round_no, n_chunks=rng.randint(1, 4), end_delay_s=rng.uniform(0, 0.004)
        )
