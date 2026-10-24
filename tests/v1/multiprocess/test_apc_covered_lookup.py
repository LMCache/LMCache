# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the APC-covered-lookup skip.

The engine marks the APC-covered prefix by setting the lookup key's ``start``;
the server touches ``[0, start)`` and locks/prefetches only ``[start, end)``.
Covers the correctness-critical pieces that do not require the native compute
backend:

- ``resolve_prefetched_obj_keys`` flooring the release range to the key's
  ``start``, so a release never touches a chunk the lookup did not lock
  (over-release would drop a concurrent prefix-sharing request's read lock).
- The connector freezing the boundary only on a real submit, going sticky on
  a shrink, and remembering the abandoned lookup's locked range for the FREE.
"""

# Standard
from typing import Any, cast
from unittest.mock import MagicMock
import importlib.util
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import resolve_prefetched_obj_keys
from lmcache.v1.multiprocess.token_codec import pack_token_ids

# Test helpers
from tests.v1.mp_scheduler_adapter_test_utils import (
    make_request_client,
    make_scheduler_adapter,
)

CHUNK_SIZE = 16

# The connector and its tracker import vLLM at module top, so the tests that
# drive them cannot run in environments built without it (the k3 unit suite
# installs LMCache alone). The server-side tests below have no such dependency
# and must keep running there, so guard per-test rather than per-module.
requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="drives the vLLM connector, which imports vLLM at module top",
)


class _FakeRequest:
    """Duck-typed vLLM Request carrying only what the tracker reads."""

    def __init__(self, prompt_token_ids: list[int]):
        # Third Party
        from vllm.v1.request import RequestStatus

        self.request_id = "req-covered"
        self.status = RequestStatus.WAITING
        self.cache_salt = ""
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(prompt_token_ids)
        self.all_token_ids = list(prompt_token_ids)
        self.num_computed_tokens = 0
        self.mm_features: list[Any] = []
        self.sampling_params = None


def _make_key(n_chunks: int, start_chunks: int = 0) -> IPCCacheServerKey:
    n_tokens = n_chunks * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        model_name="m",
        world_size=1,
        worker_id=None,
        token_ids=range(n_tokens),
        start=start_chunks * CHUNK_SIZE,
        end=n_tokens,
        request_id="r",
        num_kv_readers=1,
    )


def _ctx_with_hashes(n_chunks: int) -> MagicMock:
    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher.compute_packed_chunk_hashes.return_value = [
        bytes([i]) * 4 for i in range(n_chunks)
    ]
    return ctx


def test_resolve_sliding_window_floored_by_key_start():
    # Sliding window w=2, hit=4 => window [2,4); start=3 floors it to [3,4).
    ctx = _ctx_with_hashes(1)  # hashes for [3, 4)
    key = _make_key(4, start_chunks=3)
    obj_keys = resolve_prefetched_obj_keys(
        ctx, key, hit_chunks=4, locked_gids=(), group_windows=(2,)
    )
    assert len(obj_keys) == 1


def test_touch_cached_keys_hits_both_tiers():
    """``StorageManager.touch_cached_keys`` refreshes L1 *and* every L2 adapter."""
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

    sm = MagicMock(spec=StorageManager)
    l1 = MagicMock()
    l2_a, l2_b = MagicMock(), MagicMock()
    sm._adapters_lock = threading.Lock()
    sm._l1_managers_by_id = {0: l1}
    sm._l2_adapters = {0: l2_a, 1: l2_b}

    keys = [ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(7), model_name="m", kv_rank=0)]
    StorageManager.touch_cached_keys(sm, keys)

    l1.touch_keys.assert_called_once_with(keys)
    l2_a.submit_touch.assert_called_once_with(keys)
    l2_b.submit_touch.assert_called_once_with(keys)


def test_submit_sends_key_start_and_reports_whether_it_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``maybe_submit_lookup_request`` carries ``start`` on the key.

    The caller must key its frozen covered boundary off the return value: a
    poll that no-ops because a lookup is already in flight returns ``False``.
    """
    client = make_request_client(CHUNK_SIZE)
    adapter = make_scheduler_adapter(monkeypatch, {"u": client}, CHUNK_SIZE)
    token_ids = list(range(8 * CHUNK_SIZE))

    assert adapter.maybe_submit_lookup_request(
        "req", packed_token_ids=pack_token_ids(token_ids), start=3 * CHUNK_SIZE
    )
    client.lookup.assert_called_once()
    key = client.lookup.call_args.args[0]
    assert (key.start, key.end) == (3 * CHUNK_SIZE, 8 * CHUNK_SIZE)

    # Second poll: the lookup is still in flight, so nothing is sent.
    assert not adapter.maybe_submit_lookup_request(
        "req", packed_token_ids=pack_token_ids(token_ids), start=3 * CHUNK_SIZE
    )
    client.lookup.assert_called_once()


@requires_vllm
def test_covered_boundary_frozen_only_when_a_lookup_is_sent():
    """A no-op poll must not move ``apc_covered_tokens``.

    ``lookup_started_at`` is cleared once a result lands while the request
    stays tracked by the adapter, so a later poll re-enters the submit path.
    If that poll re-froze the boundary from the APC hit it happens to see,
    the frozen value would diverge from the one the in-flight lookup used --
    making the shrink check compare against a boundary the server never had.
    """
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    # A lookup is already in flight, so the adapter sends nothing.
    adapter.maybe_submit_lookup_request.return_value = False
    adapter.check_lookup_result.return_value = None

    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = False
    connector._hit_alignment_tokens = CHUNK_SIZE
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False

    request = _FakeRequest(list(range(8 * CHUNK_SIZE)))
    tracker = connector._get_or_create_request_tracker(request)
    # The in-flight lookup was built on a 4-chunk APC hit.
    tracker.apc_covered_tokens = 4 * CHUNK_SIZE
    tracker.lookup_started_at = None  # cleared when the previous result landed

    # Poll again while a 7-chunk APC hit is visible. Before the fix this
    # re-froze the boundary to 7 chunks even though no LOOKUP went out.
    connector.get_num_new_matched_tokens(request, 7 * CHUNK_SIZE)

    assert tracker.apc_covered_tokens == 4 * CHUNK_SIZE


def _shrink_connector(covered_chunks: int = 4):
    """A connector poised at the shrink check inside the lookup poll.

    Models the second poll of a request whose first poll already submitted a
    lookup: ``maybe_submit_lookup_request`` returns ``False`` (one is still in
    flight, so the frozen boundary is not rewritten) and the lookup result is
    in. Callers vary ``num_computed_tokens`` to move vLLM's APC hit relative
    to that boundary.
    """
    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector

    adapter = MagicMock()
    adapter.lmcache_tokens_per_chunk = CHUNK_SIZE
    adapter.maybe_submit_lookup_request.return_value = False
    adapter.check_lookup_result.return_value = 6 * CHUNK_SIZE
    connector = LMCacheMPConnector.__new__(LMCacheMPConnector)
    connector.request_trackers = {}
    connector.scheduler_adapter = adapter
    connector._reserve_last_token_for_lookup = False
    connector._hit_alignment_tokens = CHUNK_SIZE
    connector._connector_stats = MagicMock()
    connector.lazy_offload = False

    request = _FakeRequest(list(range(8 * CHUNK_SIZE)))
    tracker = connector._get_or_create_request_tracker(cast(Any, request))
    tracker.apc_covered_tokens = covered_chunks * CHUNK_SIZE
    return connector, adapter, request, tracker


@requires_vllm
def test_shrink_goes_sticky_and_remembers_the_stale_locked_range():
    """A shrink re-looks-up immediately and keeps the stale range for the FREE.

    The abandoned lookup locked ``[c0, ret)``. The connector records that
    range in ``pending_free_ranges`` -- released with the request's next
    ``FREE_LOOKUP_LOCKS`` -- and resubmits the full lookup in the same
    scheduler step instead of waiting for the next poll.
    """
    connector, adapter, request, tracker = _shrink_connector()
    tracker.num_stored_tokens = 6 * CHUNK_SIZE
    tracker.num_lmcache_hit_tokens = 6 * CHUNK_SIZE

    # APC now covers 2 chunks, below the frozen 4.
    result = connector.get_num_new_matched_tokens(request, 2 * CHUNK_SIZE)

    # Scheduler is told to re-poll rather than given a hit length.
    assert result == (None, True)
    # The stale [c0, ret) range rides the next FREE; nothing is sent now.
    adapter.free_lookup_locks.assert_not_called()
    assert tracker.pending_free_ranges == [(4 * CHUNK_SIZE, 6 * CHUNK_SIZE)]
    adapter.cleanup_lookup_result.assert_called_once_with(request.request_id)
    # The full re-lookup is submitted in the same step: once at the top of
    # the poll, once after the shrink was detected.
    assert adapter.maybe_submit_lookup_request.call_count == 2
    assert "start" not in adapter.maybe_submit_lookup_request.call_args.kwargs
    # Sticky, so a second shrink cannot recur for this request.
    assert tracker.need_full_lookup is True
    # Per-lookup state reset so the re-lookup starts clean.
    assert tracker.apc_covered_tokens == 0
    assert tracker.num_stored_tokens == 0
    assert tracker.num_lmcache_hit_tokens == 0


@requires_vllm
def test_no_shrink_when_apc_hit_still_covers_the_boundary():
    """An APC hit at or above the frozen boundary is not a shrink."""
    connector, adapter, request, tracker = _shrink_connector()

    # Exactly at the boundary.
    connector.get_num_new_matched_tokens(request, 4 * CHUNK_SIZE)

    adapter.free_lookup_locks.assert_not_called()
    assert tracker.need_full_lookup is False
    assert tracker.pending_free_ranges == []
    assert tracker.apc_covered_tokens == 4 * CHUNK_SIZE


@requires_vllm
def test_shrink_cannot_recur_once_fallen_back():
    """``need_full_lookup`` makes the fallback fire only once.

    Acting on an already-fallen-back request would record a bogus stale
    range against the replacement full lookup's locks.
    """
    connector, adapter, request, tracker = _shrink_connector(covered_chunks=0)
    tracker.need_full_lookup = True

    connector.get_num_new_matched_tokens(request, 0)

    adapter.free_lookup_locks.assert_not_called()
    adapter.cleanup_lookup_result.assert_not_called()
    assert tracker.pending_free_ranges == []


def _lookup_ctx(n_chunks: int) -> MagicMock:
    """Mock engine context sufficient to drive ``LookupModule.lookup``."""
    ctx = _ctx_with_hashes(n_chunks)
    ctx.layout_desc_registry.find.return_value = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value = MagicMock(
        num_chunks_in_sw=[-1], num_object_groups=1
    )
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {0: MagicMock()}
    return ctx


def test_all_covered_lookup_touches_and_submits_no_prefetch():
    """A key whose ``start`` reaches the aligned end locks and fetches nothing.

    The empty job still resolves: its hit is the covered prefix, offset from
    ``start_chunk``, so the engine's accounting (``num_stored_tokens``) stays
    absolute without a single chunk being re-fetched.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = _lookup_ctx(4)
    module = LookupModule(ctx)
    module.lookup(_make_key(4, start_chunks=4), 1)

    ctx.storage_manager.touch_cached_keys.assert_called_once()
    ctx.storage_manager.submit_prefetch_task.assert_not_called()
    with module._prefetch_job_lock:
        job = module._prefetch_jobs["r"]
    assert job.handle.prefetch_request_id == -1
    assert job.start_chunk == 4


def test_server_honors_the_lookup_key_start():
    """A lookup key with ``start > 0`` skips the lock and the prefetch below it.

    The LRU touch is the observable proof that the start was honored -- it
    happens only on the skip path. A key starting at 0 keeps the old
    behavior, so the prefix is still locked and prefetched.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    covered_ctx = _lookup_ctx(4)
    LookupModule(covered_ctx).lookup(_make_key(4, start_chunks=2), 1)
    covered_ctx.storage_manager.touch_cached_keys.assert_called_once()

    plain_ctx = _lookup_ctx(4)
    LookupModule(plain_ctx).lookup(_make_key(4), 1)
    plain_ctx.storage_manager.touch_cached_keys.assert_not_called()


def test_end_session_refreshes_both_tiers():
    """``end_session``'s unified touch must reach L2, not just L1.

    Skipping splits one request's recency in two: the covered prefix is
    touched when the lookup runs, while the retrieved tail is touched when it
    is actually read, much later. An L1-only refresh at the end would leave
    the prefix the older L2 entry and make the most-shared data the first
    thing L2 evicts.
    """
    # First Party
    from lmcache.v1.multiprocess.modules.lookup import LookupModule

    ctx = _lookup_ctx(4)
    session = MagicMock()
    session.lookup_ipc_key = _make_key(4)
    session.get_hashes.return_value = [1, 2, 3, 4]
    ctx.session_manager.remove.return_value = session

    LookupModule(ctx).end_session("r")

    ctx.storage_manager.touch_cached_keys.assert_called_once()
    touched = ctx.storage_manager.touch_cached_keys.call_args.args[0]
    assert len(touched) == 4


def test_serde_wrapper_forwards_touch_to_the_inner_adapter():
    """The serde wrapper must delegate ``submit_touch``, not inherit it.

    ``register_listener`` forwards to the inner adapter, so the wrapper's own
    listener list is always empty. Inheriting the base implementation would
    notify that empty list and silently skip the L2 refresh entirely whenever
    a serde config is set.
    """
    # First Party
    from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper

    wrapper = SerdeL2AdapterWrapper.__new__(SerdeL2AdapterWrapper)
    inner = MagicMock()
    wrapper._inner = inner
    keys = [ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(3), model_name="m", kv_rank=0)]

    SerdeL2AdapterWrapper.submit_touch(wrapper, keys)

    inner.submit_touch.assert_called_once_with(keys)
