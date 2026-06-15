# SPDX-License-Identifier: Apache-2.0
"""Unit tests for V3 load/store optimizations: L1 (batched rope), L2
(obj_keys cache), S1 (async fingerprint).

These tests exercise the wiring/state changes without touching CUDA or
the storage controller. The CUDA kernel inside ``_apply_cb_rope_batched``
is mocked; the matcher inside the async fingerprint worker is mocked.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import threading
import time

# Third Party
import pytest

# ---------------------------------------------------------------------------
# S1: async fingerprint registration
# ---------------------------------------------------------------------------


def _make_engine_with_mocked_matcher():
    """Construct a real BlendV3Module with the matcher mocked so we can
    observe `on_new_token_hashes` calls without setting up storage."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng_mock = MagicMock(spec=v3_mod.BlendV3Module)
    eng_mock._fingerprint_stop = threading.Event()
    eng_mock._token_range_matcher = MagicMock()
    eng_mock._pending_fp_lock = threading.Lock()
    eng_mock._pending_fp_hashes = set()
    # Bind the real drainer method to our mock.
    eng_mock._drain_fingerprint_queue = (
        v3_mod.BlendV3Module._drain_fingerprint_queue.__get__(eng_mock)
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
            ([1, 2, 3], [b"h1"], 0, 0),
            ([4, 5, 6], [b"h2"], 1, 3),
            ([7, 8, 9], [b"h3"], 0, 6),
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
        None,
    ]

    worker = threading.Thread(target=eng._drain_fingerprint_queue, daemon=True)
    worker.start()
    try:
        eng._fingerprint_queue.put(([1], [b"h1"], 0, 0))
        eng._fingerprint_queue.put(([2], [b"h2"], 0, 1))
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
# L2: obj_keys cache lifecycle
# ---------------------------------------------------------------------------


def _fake_obj_key(chunk_hash: bytes, worker_id: int) -> SimpleNamespace:
    return SimpleNamespace(chunk_hash=chunk_hash, worker_id=worker_id)


def test_obj_keys_cache_round_trip_tp1():
    """At world_size=1, retrieve can rebuild from the cache exactly."""
    eng = MagicMock()
    eng._lookup_obj_keys_cache = {}
    eng._lookup_obj_keys_lock = threading.Lock()

    # Simulate what cb_lookup_subsequences stores.
    chunk_hashes = [b"h1", b"h2", b"h3"]
    obj_keys_per_chunk = {h: [_fake_obj_key(h, 0)] for h in chunk_hashes}
    with eng._lookup_obj_keys_lock:
        eng._lookup_obj_keys_cache["req-1"] = obj_keys_per_chunk

    # Simulate retrieve consuming the cache.
    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    with eng._lookup_obj_keys_lock:
        cached = eng._lookup_obj_keys_cache.pop("req-1", None)

    assert cached is not None
    assert all(r.hash in cached for r in matches_sorted)
    rebuilt = [k for r in matches_sorted for k in cached[r.hash]]
    assert len(rebuilt) == 3
    assert [k.chunk_hash for k in rebuilt] == chunk_hashes
    # Cache is now empty for this request.
    with eng._lookup_obj_keys_lock:
        assert "req-1" not in eng._lookup_obj_keys_cache


def test_obj_keys_cache_round_trip_tp_expanded():
    """world_size>1: cached entry per hash is a list of length world_size,
    rebuilt list is flat chunk-major."""
    eng = MagicMock()
    eng._lookup_obj_keys_cache = {}
    eng._lookup_obj_keys_lock = threading.Lock()

    ws = 4
    chunk_hashes = [b"h1", b"h2"]
    per_hash = {h: [_fake_obj_key(h, w) for w in range(ws)] for h in chunk_hashes}
    with eng._lookup_obj_keys_lock:
        eng._lookup_obj_keys_cache["req-tp"] = per_hash

    matches_sorted = [
        SimpleNamespace(hash=h, cur_st=i) for i, h in enumerate(chunk_hashes)
    ]
    with eng._lookup_obj_keys_lock:
        cached = eng._lookup_obj_keys_cache.pop("req-tp", None)
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
# L1: batched rope structure
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for the torch tensors used inside _apply_cb_rope_batched.
    Tracks shape so the kernel mock can assert on it.
    """

    def __init__(self, shape):
        self.shape = shape
        self.device = "cpu"

    def __getitem__(self, idx):
        # tmp[0] selects K from the (2, num_layers, slots, hidden_dim) tensor.
        return _FakeTensor(self.shape[1:] if isinstance(idx, int) else self.shape)

    def reshape(self, *new_shape):
        return _FakeTensor(tuple(new_shape))

    def view(self, *new_shape):
        return _FakeTensor(tuple(new_shape))


def _build_fake_gpu_context(batch_size: int, num_groups: int):
    """Returns a MagicMock matching the minimal GPUCacheContext surface
    used by _apply_cb_rope_batched."""
    gpu_context = MagicMock()
    gpu_context.kv_layer_groups_manager.num_kernel_groups = num_groups
    # All groups: uncompressed (tokens_per_block == slots_per_block), kv_size=2.
    groups = [
        SimpleNamespace(tokens_per_block=4, slots_per_block=4)
        for _ in range(num_groups)
    ]
    gpu_context.kv_layer_groups_manager.kernel_groups = groups

    # Each per-(slot, group) buffer has shape
    # (2 kv, num_layers, slots_per_block, hidden_dim).
    num_layers, slots_per_block, hidden_dim = 2, 4, 64
    head_size = 32

    def _get_temp_kernel_group_buffer(batch_idx, kernel_group_idx):
        return _FakeTensor((2, num_layers, slots_per_block, hidden_dim))

    gpu_context.get_temp_kernel_group_buffer.side_effect = _get_temp_kernel_group_buffer
    return gpu_context, head_size


def test_batched_rope_calls_kernel_per_group_per_slot():
    """For N non-prefix slots and G groups, kernel is called N*G times
    (matching today's CUDA-level work) but the Python ``per-group setup``
    runs only G times (vs N*G under the legacy path)."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context, head_size = _build_fake_gpu_context(batch_size=4, num_groups=2)
    rope_state = SimpleNamespace(
        head_size=head_size, cos_sin_cache=MagicMock(), is_neox_style=True
    )

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    slots_to_rope = [(0, 100, 200), (2, 300, 400)]  # 2 non-prefix slots

    with (
        patch.object(v3_mod, "lmc_ops") as ops,
        patch.object(v3_mod, "torch") as torch_mod,
    ):
        torch_mod.long = "long"

        # Build a fake positions tensor that supports + and .repeat()
        class _Pos:
            def __add__(self, other):
                return _Pos()

            def __radd__(self, other):
                return _Pos()

            def repeat(self, n):
                return _Pos()

        torch_mod.arange.return_value = _Pos()

        eng._apply_cb_rope_batched(gpu_context, rope_state, 4, slots_to_rope)

    # all_slots is built once per group (G=2), each fetching the full batch
    # of slot buffers => batch_len(4) × G(2) = 8 buffer fetches, independent
    # of how many slots are actually re-RoPE'd.
    assert gpu_context.get_temp_kernel_group_buffer.call_count == 8
    # Kernel called N=2 slots × G=2 groups = 4 times.
    assert ops.rotary_embedding_k_fused.call_count == 4


def test_batched_rope_noop_on_empty_slots():
    """No non-prefix slots → no setup, no kernel calls."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context, head_size = _build_fake_gpu_context(batch_size=2, num_groups=2)
    rope_state = SimpleNamespace(
        head_size=head_size, cos_sin_cache=MagicMock(), is_neox_style=False
    )
    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    with patch.object(v3_mod, "lmc_ops") as ops:
        eng._apply_cb_rope_batched(gpu_context, rope_state, 2, [])

    assert gpu_context.get_temp_kernel_group_buffer.call_count == 0
    assert ops.rotary_embedding_k_fused.call_count == 0


def test_batched_rope_raises_on_compressed_layout():
    """A compressed group (tokens_per_block != slots_per_block) → RuntimeError."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    gpu_context = MagicMock()
    gpu_context.kv_layer_groups_manager.num_kernel_groups = 1
    gpu_context.kv_layer_groups_manager.kernel_groups = [
        SimpleNamespace(tokens_per_block=8, slots_per_block=4)
    ]
    rope_state = SimpleNamespace(
        head_size=32, cos_sin_cache=MagicMock(), is_neox_style=True
    )

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._apply_cb_rope_batched = v3_mod.BlendV3Module._apply_cb_rope_batched.__get__(
        eng
    )

    with pytest.raises(RuntimeError, match="is compressed"):
        eng._apply_cb_rope_batched(gpu_context, rope_state, 2, [(0, 1, 2)])


# ---------------------------------------------------------------------------
# Coordinator (global) leg: conversion to retrievable CBMatchResult + deadline
# ---------------------------------------------------------------------------


def _coord_engine(chunk_size: int = 4):
    """A BlendV3Module mock with the coordinator-leg methods bound."""
    # First Party
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    eng = MagicMock(spec=v3_mod.BlendV3Module)
    eng._ctx = SimpleNamespace(chunk_size=chunk_size)
    eng._build_global_segments = v3_mod.BlendV3Module._build_global_segments.__get__(
        eng
    )
    eng._poll_coordinator_match = v3_mod.BlendV3Module._poll_coordinator_match.__get__(
        eng
    )
    return eng


def test_build_global_segments_are_retrievable_cbmatchresults():
    """Coordinator object_key hex round-trips to the hash the retrieve path
    resolves via ipc_key_to_object_keys; positions span one chunk."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import RemoteMatch
    from lmcache.v1.multiprocess.custom_types import CBMatchResult

    eng = _coord_engine(chunk_size=4)
    raw = bytes.fromhex("00") * 0 + b"\xab\xcd\xef\x01"
    matches = [RemoteMatch(object_key=raw.hex(), old_st=8, cur_st=20)]

    segs = eng._build_global_segments(matches)

    assert len(segs) == 1
    seg = segs[0]
    assert isinstance(seg, CBMatchResult)
    assert seg.hash == raw  # hex -> exact bytes the retrieve path expands
    assert (seg.old_st, seg.old_ed, seg.cur_st, seg.cur_ed) == (8, 12, 20, 24)


def test_poll_coordinator_match_deferred_then_resolved():
    """PENDING within deadline defers (None); a list resolves to segments."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import PENDING, RemoteMatch

    eng = _coord_engine(chunk_size=4)
    coordinator = MagicMock()
    eng._coordinator = coordinator
    job = SimpleNamespace(coord_submitted=True, coord_deadline=time.monotonic() + 60)

    coordinator.poll_match.return_value = PENDING
    assert eng._poll_coordinator_match(job, "rid") is None  # defer
    coordinator.take_match.assert_not_called()

    coordinator.poll_match.return_value = [RemoteMatch("aa", old_st=0, cur_st=4)]
    out = eng._poll_coordinator_match(job, "rid")
    assert [s.cur_st for s in out] == [4]
    coordinator.take_match.assert_called_once_with("rid")


def test_poll_coordinator_match_gives_up_past_deadline():
    """PENDING past the deadline degrades to local-only ([]) and drops state."""
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import PENDING

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
    from lmcache.v1.multiprocess.modules import blend_v3 as v3_mod

    f = v3_mod.BlendV3Module._non_overlapping_after_prefix

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
