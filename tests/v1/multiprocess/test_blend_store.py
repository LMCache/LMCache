# SPDX-License-Identifier: Apache-2.0
"""Store-side blend units: the async fingerprint path, plus lookup
(coordinator leg, prefix overlap), rope registration, and read-set
classification. Load/retrieve-side tests live in test_blend_retrieve.py.

These tests exercise the wiring/state changes without touching CUDA or
the storage controller; the matcher inside the async fingerprint worker
is mocked.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache import device_ops  # noqa: F401
from lmcache.v1.mp_coordinator.blend_client import PENDING  # noqa: F401
from lmcache.v1.multiprocess.modules.blend import retrieve as retrieve_mod  # noqa: F401
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.read_set import (
    _cb_chunk_major_object_keys,
    _classify_cb_read_groups,
    _narrow_attn_desc,
)
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState

# ---------------------------------------------------------------------------
# S1: async fingerprint registration
# ---------------------------------------------------------------------------


def _make_engine_with_mocked_matcher():
    """Construct a real BlendModule with the matcher mocked so we can
    observe `on_new_token_hashes` calls without setting up storage."""

    eng_mock = MagicMock(spec=BlendModule)
    eng_mock._fingerprint_stop = threading.Event()
    eng_mock._token_range_matcher = MagicMock()
    # The drainer reports the matcher's indexed-chunk count on the event.
    eng_mock._token_range_matcher.on_new_token_hashes.return_value = 1
    eng_mock._token_range_matcher.chunk_size = 256
    eng_mock._pending_fp_lock = threading.Lock()
    eng_mock._pending_fp_hashes = set()
    eng_mock._event_bus = MagicMock()
    # Bind the real drainer + its per-job and registration-event helpers.
    eng_mock._emit_fingerprints_registered = (
        BlendModule._emit_fingerprints_registered.__get__(eng_mock)
    )
    eng_mock._register_fp_job = BlendModule._register_fp_job.__get__(eng_mock)
    eng_mock._drain_fingerprint_queue = BlendModule._drain_fingerprint_queue.__get__(
        eng_mock
    )
    return eng_mock


def test_fingerprint_queue_drains_in_order():
    """Jobs enqueued by store() flow through the worker in submission order."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()

    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    try:
        jobs = [
            ([1, 2, 3], [b"h1"], 0, 0, "req-a"),
            ([4, 5, 6], [b"h2"], 1, 3, "req-b"),
            ([7, 8, 9], [b"h3"], 0, 6, "req-c"),
        ]
        for j in jobs:
            eng._fingerprint_queue.put(j)
        # Wait for the queue to drain (worker calls task_done implicitly
        # only via get(); we just poll until matcher has all calls).
        deadline = time.monotonic() + 2.0
        while (
            eng._token_range_matcher.on_new_token_hashes.call_count < len(jobs)
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        eng._fingerprint_stop.set()
        worker.join(timeout=1.0)

    # All three were registered.
    assert eng._token_range_matcher.on_new_token_hashes.call_count == 3
    # In submission order.
    calls = eng._token_range_matcher.on_new_token_hashes.call_args_list
    assert calls[0].args[0] == [1, 2, 3]
    assert calls[1].args[0] == [4, 5, 6]
    assert calls[2].args[0] == [7, 8, 9]
    # kwargs are preserved (start_chunk_idx, position_offset).
    assert calls[1].kwargs == {"start_chunk_idx": 1, "position_offset": 3}


def test_fingerprint_worker_survives_kernel_exception():
    """A failing matcher call doesn't kill the worker."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()
    # First call raises, subsequent succeed.
    eng._token_range_matcher.on_new_token_hashes.side_effect = [
        RuntimeError("boom"),
        1,
    ]

    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    try:
        eng._fingerprint_queue.put(([1], [b"h1"], 0, 0, "req-a"))
        eng._fingerprint_queue.put(([2], [b"h2"], 0, 1, "req-b"))
        deadline = time.monotonic() + 2.0
        while (
            eng._token_range_matcher.on_new_token_hashes.call_count < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        eng._fingerprint_stop.set()
        worker.join(timeout=1.0)

    assert eng._token_range_matcher.on_new_token_hashes.call_count == 2
    assert not worker.is_alive()


def test_fingerprint_worker_stops_on_signal():
    """``_fingerprint_stop`` event halts the drainer cleanly."""
    # Standard
    from queue import Queue

    eng = _make_engine_with_mocked_matcher()
    eng._fingerprint_queue = Queue()
    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    eng._fingerprint_stop.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


# ---------------------------------------------------------------------------
# Coordinator (global) leg: conversion to retrievable CBMatchResult + deadline
# ---------------------------------------------------------------------------


def _coord_engine(chunk_size: int = 4):
    """A BlendModule mock with the coordinator-leg methods bound."""

    eng = MagicMock(spec=BlendModule)
    eng._ctx = SimpleNamespace(chunk_size=chunk_size)
    # _event_bus is an instance attr (set in __init__), so spec= omits it;
    # _poll_coordinator_match publishes CB_COORDINATOR_MATCH_END through it.
    eng._event_bus = MagicMock()
    eng._build_global_segments = BlendModule._build_global_segments.__get__(eng)
    eng._poll_coordinator_match = BlendModule._poll_coordinator_match.__get__(eng)
    return eng


def test_build_global_segments_are_retrievable_cbmatchresults():
    """Coordinator chunk_hash hex round-trips to the hash the retrieve path
    resolves via ipc_key_to_object_keys; positions span one chunk."""
    # First Party
    from lmcache.v1.mp_coordinator.api import BlendMatch
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    eng = _coord_engine(chunk_size=4)
    raw = bytes.fromhex("00") * 0 + b"\xab\xcd\xef\x01"
    matches = [BlendMatch(chunk_hash=raw, old_st=8, cur_st=20)]

    segs = eng._build_global_segments(matches)

    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, CBMatchResult)
    assert seg.hash == raw  # hex -> exact bytes the retrieve path expands
    assert (seg.old_st, seg.old_ed, seg.cur_st, seg.cur_ed) == (8, 12, 20, 24)


def test_poll_coordinator_match_deferred_then_resolved():
    """PENDING within deadline defers (None); a list resolves to segments."""
    # First Party
    from lmcache.v1.mp_coordinator.api import BlendMatch

    eng = _coord_engine(chunk_size=4)
    coordinator = MagicMock()
    eng._coordinator = coordinator
    job = SimpleNamespace(coord_submitted=True, coord_deadline=time.monotonic() + 60)

    coordinator.poll_match.return_value = PENDING
    assert eng._poll_coordinator_match(job, "rid") is None  # defer
    coordinator.take_match.assert_not_called()

    coordinator.poll_match.return_value = [BlendMatch(b"\xaa", old_st=0, cur_st=4)]
    out = eng._poll_coordinator_match(job, "rid")
    assert [s.cur_st for s in out] == [4]
    coordinator.take_match.assert_called_once_with("rid")


def test_poll_coordinator_match_gives_up_past_deadline():
    """PENDING past the deadline degrades to local-only ([]) and drops state."""
    # First Party

    eng = _coord_engine(chunk_size=4)
    coordinator = MagicMock()
    eng._coordinator = coordinator
    coordinator.poll_match.return_value = PENDING
    job = SimpleNamespace(coord_submitted=True, coord_deadline=time.monotonic() - 1)

    assert eng._poll_coordinator_match(job, "rid") == []
    coordinator.take_match.assert_called_once_with("rid")


def test_non_overlapping_after_prefix():
    """Prefix filter + leftmost-greedy overlap dedup, filter applied first."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    f = BlendModule._non_overlapping_after_prefix

    def m(cur_st: int, cur_ed: int) -> CBMatchResult:
        return CBMatchResult(
            old_st=0, old_ed=cur_ed - cur_st, cur_st=cur_st, cur_ed=cur_ed, hash=b""
        )

    assert f([], 0) == []

    # Overlap dedup + ascending cur_st: 10-20 overlaps the kept 5-15, dropped.
    out = f([m(10, 20), m(5, 15), m(15, 25)], 0)
    assert [(r.cur_st, r.cur_ed) for r in out] == [(5, 15), (15, 25)]

    # Prefix filter drops matches starting before the coverage.
    out = f([m(0, 10), m(10, 20)], 5)
    assert [r.cur_st for r in out] == [10]

    # Filter precedes dedup: a prefix-covered match (5-13) must NOT suppress the
    # usable 10-18 in the greedy pass (dedup-first would drop both -> []).
    out = f([m(5, 13), m(10, 18)], 8)
    assert [r.cur_st for r in out] == [10]


# ---------------------------------------------------------------------------
# Dual-RoPE: per-group cache selection + registration validation
# ---------------------------------------------------------------------------


def test_cache_for_group_uniform_and_mapped():
    """Empty map -> every group uses cache 0; a map indexes per group;
    a group past the map's end raises instead of guessing."""

    local, global_ = MagicMock(), MagicMock()

    uniform = _CBRopeState(
        head_size=32, is_neox_style=True, cos_sin_caches=[local], group_to_cache=[]
    )
    assert uniform.cache_for_group(0) is local
    assert uniform.cache_for_group(5) is local

    mapped = _CBRopeState(
        head_size=32,
        is_neox_style=True,
        cos_sin_caches=[local, global_],
        group_to_cache=[0, 1],
    )
    assert mapped.cache_for_group(0) is local
    assert mapped.cache_for_group(1) is global_
    with pytest.raises(RuntimeError, match="no rope cache mapping"):
        mapped.cache_for_group(2)


def _rope_registration_engine(engine_group_indices: list[int]):
    """A BlendModule mock with ``cb_register_rope`` bound and a registered
    instance whose kernel groups span the given engine group indices."""

    eng = MagicMock(spec=BlendModule)
    eng._cb_rope_state = {}
    eng._transfer_module = MagicMock()
    entry = SimpleNamespace(
        cache_context=SimpleNamespace(
            kv_layer_groups_manager=SimpleNamespace(
                kernel_groups=[
                    SimpleNamespace(engine_group_idx=idx)
                    for idx in engine_group_indices
                ]
            )
        )
    )
    eng._transfer_module.get_and_touch_context_entry.return_value = entry
    eng.cb_register_rope = BlendModule.cb_register_rope.__get__(eng)
    return eng


def _unit_rope_cache_ipc():
    """An IPC-wrapper mock whose tensor has unit magnitude (cos=1, sin=0),
    so registration skips mscale normalization."""
    # Third Party

    cache = torch.zeros(4, 8)
    cache[:, :4] = 1.0
    ipc = MagicMock()
    ipc.to_tensor.return_value = cache
    return ipc


def test_register_rope_dual_cache_round_trip():
    """Two caches + a full engine-group map register and land in rope state."""
    eng = _rope_registration_engine(engine_group_indices=[0, 1])

    eng.cb_register_rope(
        instance_id=7,
        cos_sin_caches_ipc=[_unit_rope_cache_ipc(), _unit_rope_cache_ipc()],
        head_size=8,
        is_neox_style=True,
        group_to_cache=[0, 1],
    )

    state = eng._cb_rope_state[7]
    assert len(state.cos_sin_caches) == 2
    assert state.group_to_cache == [0, 1]
    assert state.cache_for_group(1) is state.cos_sin_caches[1]


def test_register_rope_rejects_invalid_group_to_cache():
    """Out-of-range / negative cache indices and a map that does not cover
    every engine group of the registered model are rejected."""
    eng = _rope_registration_engine(engine_group_indices=[0, 1])
    caches = [_unit_rope_cache_ipc(), _unit_rope_cache_ipc()]

    with pytest.raises(ValueError, match="outside"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[0, 2])
    with pytest.raises(ValueError, match="outside"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[-1, 0])
    # Model has engine groups {0, 1} but the map only covers group 0.
    with pytest.raises(ValueError, match="engine groups up to index 1"):
        eng.cb_register_rope(1, caches, 8, True, group_to_cache=[0])
    # A map referencing caches that were never sent is rejected even when the
    # cache list is empty (the NoPE form requires an empty map too).
    with pytest.raises(ValueError, match="outside"):
        eng.cb_register_rope(1, [], 8, True, group_to_cache=[0, 0])


def test_register_rope_accepts_nope_zero_caches():
    """NoPE models register zero cos/sin caches. The rope state is still
    stored — it carries the head layout the scatter needs — and every group
    reports its re-RoPE skipped."""
    eng = _rope_registration_engine(engine_group_indices=[0, 1])

    eng.cb_register_rope(1, [], 128, True, group_to_cache=[])

    state = eng._cb_rope_state[1]
    assert state.cos_sin_caches == []
    assert state.head_size == 128
    # No cache for any group => every re-RoPE consumer skips rotation.
    assert state.cache_for_group(0) is None
    assert state.cache_for_group(1) is None


def test_register_rope_requires_registered_instance():
    """CB_REGISTER_ROPE before REGISTER_KV_CACHE is rejected."""
    eng = _rope_registration_engine(engine_group_indices=[0])
    eng._transfer_module.get_and_touch_context_entry.return_value = None

    with pytest.raises(ValueError, match="no paged KV cache registered"):
        eng.cb_register_rope(1, [_unit_rope_cache_ipc()], 8, True, group_to_cache=[])


def test_union_of_local_and_fleet_matches_collapses_duplicates():
    """Local and fleet matching are additive, so the overlap dedup is what
    keeps a chunk both sources report from scattering twice."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    dedup = BlendModule._non_overlapping_after_prefix
    shared = CBMatchResult(old_st=0, old_ed=4, cur_st=8, cur_ed=12, hash=b"\x01")
    local_only = CBMatchResult(old_st=4, old_ed=8, cur_st=12, cur_ed=16, hash=b"\x02")
    fleet_only = CBMatchResult(old_st=8, old_ed=12, cur_st=16, cur_ed=20, hash=b"\x03")

    # Same chunk reported by both sources, plus one unique to each.
    union = [shared, local_only] + [shared, fleet_only]
    kept = dedup(union, 0)

    assert [r.hash for r in kept] == [b"\x01", b"\x02", b"\x03"]


def test_union_recall_is_at_least_either_source_alone():
    """The union can only add matches: whatever one source finds outside the
    prefix survives the merge."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    dedup = BlendModule._non_overlapping_after_prefix
    local = [CBMatchResult(old_st=0, old_ed=4, cur_st=4, cur_ed=8, hash=b"\x01")]
    fleet = [CBMatchResult(old_st=0, old_ed=4, cur_st=12, cur_ed=16, hash=b"\x02")]

    assert len(dedup(local, 0)) == 1
    assert len(dedup(fleet, 0)) == 1
    assert len(dedup(local + fleet, 0)) == 2


# ---------------------------------------------------------------------------
# Multi-object-group read set (separate-object-groups support)
# ---------------------------------------------------------------------------


def test_classify_read_groups_single_group_is_legacy():
    """A single-object-group layout maps to group 0 with no aux group,
    regardless of kind labels (legacy fused layout)."""

    read = _classify_cb_read_groups(1, ())
    assert read.blend_gids == (0,)
    assert read.prefix_gids == (0,)
    assert read.recurrent_gids == ()
    assert read.attn_gid == 0


def test_classify_read_groups_hybrid_layout():
    """Per-leg sets: blend = (attention, aux), never recurrent; prefix =
    (attention, recurrent), never aux — each leg keys/locks exactly the
    planes it consumes."""

    read = _classify_cb_read_groups(3, ("attention", "recurrent", "aux"))
    assert read.blend_gids == (0, 2)
    assert read.prefix_gids == (0, 1)
    assert read.recurrent_gids == (1,)
    assert read.attn_gid == 0


def test_classify_read_groups_multi_recurrent():
    """Every recurrent group joins the prefix set (a hybrid may bucket its
    state pages into more than one object group under separation)."""

    read = _classify_cb_read_groups(4, ("recurrent", "attention", "recurrent", "aux"))
    assert read.blend_gids == (1, 3)
    assert read.prefix_gids == (0, 1, 2)
    assert read.recurrent_gids == (0, 2)


def test_classify_read_groups_rejects_two_aux_groups():
    """Blend supports at most one connector-private aux group."""

    with pytest.raises(RuntimeError):
        _classify_cb_read_groups(3, ("attention", "aux", "aux"))


def test_narrow_attn_desc_selects_the_leg_gids():
    """The fold stride is groups x ranks, so each leg's descriptor must cover
    exactly its own gids."""
    # First Party
    from lmcache.v1.distributed.api import AttnWindowDesc

    full = AttnWindowDesc(
        num_chunks_in_sw=[1, -1, -1],
        world_size=2,
        group_kinds=("recurrent", "attention", "aux"),
    )
    prefix = _narrow_attn_desc(full, (0, 1))
    assert prefix.num_chunks_in_sw == [1, -1]
    assert prefix.group_kinds == ("recurrent", "attention")
    assert prefix.world_size == 2
    blend = _narrow_attn_desc(full, (1, 2))
    assert blend.num_chunks_in_sw == [-1, -1]
    assert blend.group_kinds == ("attention", "aux")


def test_classify_read_groups_rejects_unresolvable_layouts():
    """Multi-group layouts without kinds, with several attention buckets, or
    with several aux groups are refused loudly (silent mis-addressing
    would corrupt reads)."""

    with pytest.raises(RuntimeError):
        _classify_cb_read_groups(2, ())
    with pytest.raises(RuntimeError):
        _classify_cb_read_groups(2, ("attention", "attention"))
    with pytest.raises(RuntimeError):
        _classify_cb_read_groups(3, ("attention", "aux", "aux"))


def test_chunk_major_object_keys_ordering():
    """Keys come out chunk-major: for each hash, every read group's key(s)
    are contiguous, groups ascending — the invariant behind every per-chunk
    stride (coverage math, found-classification, retrieve pairing)."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

    key = IPCCacheServerKey.from_token_ids(
        model_name="m", world_size=2, worker_id=None, token_ids=[1, 2, 3]
    )
    hashes = [b"\x01" * 8, b"\x02" * 8]
    keys = _cb_chunk_major_object_keys(key, hashes, (0, 2))
    # 2 hashes x 2 groups x world_size 2.
    assert len(keys) == 8
    assert [k.object_group_id for k in keys] == [0, 0, 2, 2, 0, 0, 2, 2]
    assert [k.chunk_hash for k in keys][:4] == [hashes[0]] * 4
    assert [k.chunk_hash for k in keys][4:] == [hashes[1]] * 4

    # Worker-specific key: expansion of 1 per (hash, group).
    wkey = IPCCacheServerKey.from_token_ids(
        model_name="m", world_size=2, worker_id=1, token_ids=[1, 2, 3]
    )
    wkeys = _cb_chunk_major_object_keys(wkey, hashes, (0, 2))
    assert len(wkeys) == 4
    assert [k.object_group_id for k in wkeys] == [0, 2, 0, 2]


def test_classify_read_groups_recurrent_first_layout():
    """Object groups are numbered by registration order, so a hybrid whose
    recurrent layers register first makes object group 0 RECURRENT. Blend
    must key its layout off attn_gid, never off group 0 (that would hand it
    the state-page layout)."""

    read = _classify_cb_read_groups(3, ("recurrent", "attention", "aux"))
    assert read.attn_gid == 1
    # Read set ascending, recurrent excluded.
    assert read.blend_gids == (1, 2)
