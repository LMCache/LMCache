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
