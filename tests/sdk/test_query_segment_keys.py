# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the segment-relative key chain used by query tensors.
Query tensors might be only needed for the trailing part of the sequence,
so their cache keys must chain from the first token of that segment.
This is needed to accommodate retrieve(), so that it will not need to read-lock
the chunks before the segment, which the pass never wrote and the server cannot
serve. The KV cache covers every token, so its keys chain from token 0, but
query tensors exist only for the tokens a pass computed, so their keys chain
from that pass's first token.
These tests pin both halves of the contract: the store side keys each pass from
its own first computed token, and the retrieve side addresses that same chain.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.context import FULL_WINDOW, LMCacheSDKContext
from lmcache.sdk.qringbuffer import QRingBufferAdapter
from lmcache.sdk.request import LMCacheRequestStream, LMCacheRequestStreamError
from lmcache.sdk.wrapper.paged_pool import RecurrentState
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK = 4


def windowed_range(
    window_tokens: int, sw_size_tokens: int, chunk_size: int, chunk_windowed: bool
) -> tuple[int, int]:
    """Run the context's window math against bare state."""
    state = SimpleNamespace(
        _sw_size_tokens=sw_size_tokens,
        _chunk_size=chunk_size,
        _chunk_windowed=chunk_windowed,
    )
    return LMCacheSDKContext.windowed_range(state, window_tokens)  # type: ignore[arg-type]


def test_kv_chains_from_token_zero():
    """KV covers every token, so its chain roots at 0 regardless of segment."""
    assert LMCacheSDKCacheKind.KV.key_origin(segment_start=512) == 0


def test_query_chains_from_the_segment():
    """Query rows exist only past the cached prefix, so the chain roots there."""
    assert LMCacheSDKCacheKind.QUERY.key_origin(segment_start=512) == 512


def test_full_window_reads_everything():
    """The full window starts at the first token, whichever kind it serves."""
    assert windowed_range(20, FULL_WINDOW, CHUNK, chunk_windowed=False) == (0, 20)


def test_chunk_window_reads_its_trailing_chunks():
    """With object-group separation, a two-chunk window reads the last two."""
    assert windowed_range(20, 2 * CHUNK, CHUNK, chunk_windowed=True) == (12, 8)


def test_chunk_window_clamps_to_the_addressable_window():
    """A window longer than what is addressable stops at its first token:
    nothing before it exists under this kind's key chain."""
    assert windowed_range(20, 10 * CHUNK, CHUNK, chunk_windowed=True) == (0, 20)


def test_sub_chunk_window_keeps_the_tail_of_every_chunk():
    """Without object-group separation, every chunk keeps its last rows."""
    assert windowed_range(20, 2, CHUNK, chunk_windowed=False) == (0, 5 * 2)


def test_sub_chunk_window_reads_only_the_last_chunk_when_separated():
    """With separation, a sub-chunk window is the tail of the last chunk."""
    assert windowed_range(20, 2, CHUNK, chunk_windowed=True) == (16, 2)


class _FakeOp:
    """Minimal LoadStoreOp stand-in for a query store."""

    def __init__(self, token_ids: list[int], start: int, end: int) -> None:
        self.token_ids = token_ids
        self.start = start
        self.end = end


def _q_adapter() -> tuple[QRingBufferAdapter, MagicMock]:
    """A QRingBufferAdapter whose worker adapter records the keys it builds."""
    worker = MagicMock()
    worker.is_healthy = True
    worker.instance_id = 7
    worker.blocks_in_chunk = 1
    worker._create_key.side_effect = lambda token_ids, start, end, **kw: (
        IPCCacheServerKey(
            model_name="m",
            world_size=1,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=kw["request_id"],
            cache_salt=kw.get("cache_salt", ""),
        )
    )
    q_adapter = QRingBufferAdapter(worker, "m##query")
    q_adapter.q_ring = MagicMock()
    return q_adapter, worker


def _stored_key(
    tokens: list[int], op_start: int, op_end: int, segment_start: int
) -> IPCCacheServerKey:
    """Drive submit_q_store_request and return the key it stored under."""
    q_adapter, worker = _q_adapter()

    q_adapter.submit_q_store_request(
        "req",
        _FakeOp(tokens, op_start, op_end),  # type: ignore[arg-type]
        [0],
        MagicMock(),
        segment_start=segment_start,
    )

    worker.transfer_ctx.submit_q_store.assert_called_once()
    return worker.transfer_ctx.submit_q_store.call_args.args[1]


def test_store_key_is_rebased_on_the_segment():
    """The stored key drops the tokens before the segment and shifts its range,
    so the chunk hashes chain from the segment's first token."""
    tokens = list(range(40))

    key = _stored_key(tokens, op_start=20, op_end=28, segment_start=20)

    assert key.token_ids == tuple(tokens[20:])
    assert (key.start, key.end) == (0, 8)


def test_store_key_for_a_first_pass_is_unchanged():
    """A pass that computes from token 0 keys exactly as before."""
    tokens = list(range(40))

    key = _stored_key(tokens, op_start=0, op_end=8, segment_start=0)

    assert key.token_ids == tuple(tokens)
    assert (key.start, key.end) == (0, 8)


def test_store_key_keeps_the_query_model_name():
    """Re-basing does not disturb the query namespace."""
    key = _stored_key(list(range(40)), op_start=20, op_end=28, segment_start=20)

    assert key.model_name == "m##query"


def test_store_key_uses_its_own_server_session():
    """The server extends one rolling hash chain per request id, so the query
    store cannot share the request's session with the KV store: it would be
    handed the KV chain's hashes instead of the segment's."""
    key = _stored_key(list(range(40)), op_start=20, op_end=28, segment_start=20)

    assert key.request_id == "req##query"
    assert LMCacheSDKCacheKind.KV.server_session_id("req") == "req"


def test_store_past_the_segment_start_is_skipped():
    """A segment starting after the op would shift the range negative; the
    store is dropped and its ring blocks are freed instead."""
    q_adapter, worker = _q_adapter()

    q_adapter.submit_q_store_request(
        "req",
        _FakeOp(list(range(40)), 8, 16),
        [3],
        MagicMock(),
        segment_start=12,
    )

    worker.transfer_ctx.submit_q_store.assert_not_called()
    q_adapter.q_ring.free.assert_called_once_with([3])


class _RecordingContext:
    """A cache context that records the retrieve window it is handed."""

    def __init__(
        self,
        kind: LMCacheSDKCacheKind,
        tensor: torch.Tensor | None,
        window: tuple[int, bool] = (FULL_WINDOW, False),
    ) -> None:
        self.kind = kind
        self._window = window
        self.chunk_size = CHUNK
        self.is_hybrid = False
        self._tensor = tensor
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.stored: list[tuple[int, ...]] = []
        self.cached_tokens = 0

    def windowed_range(self, window_tokens: int) -> tuple[int, int]:
        return windowed_range(window_tokens, self._window[0], CHUNK, self._window[1])

    def lookup(self, tokens, cache_salt: str = "") -> int:
        return min(self.cached_tokens, (len(tokens) // CHUNK) * CHUNK)

    def retrieve(
        self, tokens, cache_salt: str = "", start_token_id: int = 0
    ) -> torch.Tensor | None:
        self.calls.append((tuple(tokens), start_token_id))
        return self._tensor

    def store(self, kv, tokens, cache_salt: str = "", recurrent_state=None) -> bool:
        self.stored.append(tuple(tokens))
        self.cached_tokens = (len(tokens) // CHUNK) * CHUNK
        return True


def _stream(
    tokens: list[int],
    segment_start: int,
    q_tensor: torch.Tensor | None,
    q_window: tuple[int, bool] = (FULL_WINDOW, False),
) -> tuple[LMCacheRequestStream, _RecordingContext]:
    """A stream that has run a pass computing tokens[segment_start:].

    Built the way the SDK does it: a first pass over the cached prefix, then a
    pass that decodes the rest.
    """
    kv_ctx = _RecordingContext(
        LMCacheSDKCacheKind.KV, torch.zeros(2, 2, len(tokens), 8)
    )
    q_ctx = _RecordingContext(LMCacheSDKCacheKind.QUERY, q_tensor, window=q_window)
    tail = tokens[segment_start:]
    stream = LMCacheRequestStream(
        contexts=[kv_ctx, q_ctx],  # type: ignore[list-item]
        post_completion=MagicMock(),
        prompt_token_ids=tokens[:segment_start],
    )
    stream.generate({"max_tokens": 0})
    # The pass that decodes the tail loads the prefix from the cache.
    kv_ctx.cached_tokens = segment_start
    stream.post_completion = lambda *a, **kw: [  # type: ignore[assignment]
        SimpleNamespace(token_id=t, text="") for t in tail
    ]
    stream.generate({"max_tokens": len(tail)})
    # The engine stored KV for every complete chunk the pass computed.
    kv_ctx.cached_tokens = (len(tokens) // CHUNK) * CHUNK
    return stream, q_ctx


def _modify(stream: LMCacheRequestStream, timeout: float = 0.0) -> list[int]:
    """Run modify_kv with an editor that keeps the sequence as-is."""
    captured: list[int] = []

    def keep(tensors, tokens):
        captured.extend(tokens)
        return tensors[LMCacheSDKCacheKind.KV], tokens

    stream.modify_kv(keep, timeout=timeout, poll_interval=0.0)
    return captured


def test_modify_addresses_the_segment_window():
    """modify_kv reads the query window through the segment's own chain: the
    window starts at the segment and the offset is relative to it, while KV
    keeps its chain at token 0."""
    tokens = list(range(40))
    stream, q_ctx = _stream(tokens, segment_start=20, q_tensor=torch.zeros(1, 2, 20, 8))
    kv_ctx = stream._contexts[LMCacheSDKCacheKind.KV]

    _modify(stream)

    assert q_ctx.calls == [(tuple(tokens[20:40]), 0)]
    assert kv_ctx.calls == [(tuple(tokens), 0)]


def test_modify_offsets_a_trailing_window_within_the_segment():
    """A window inside the segment is addressed relative to the segment."""
    tokens = list(range(40))
    stream, q_ctx = _stream(
        tokens,
        segment_start=20,
        q_tensor=torch.zeros(1, 2, 8, 8),
        q_window=(2 * CHUNK, True),
    )

    _modify(stream)

    window, relative_start = q_ctx.calls[0]
    assert window == tuple(tokens[20:40])
    assert relative_start == 12


def test_modify_expects_the_rows_a_sub_chunk_window_keeps():
    """A sub-chunk window returns fewer rows than tokens: 5 chunks x 2 rows."""
    tokens = list(range(40))
    stream, q_ctx = _stream(
        tokens,
        segment_start=20,
        q_tensor=torch.zeros(1, 2, 10, 8),
        q_window=(2, False),
    )

    _modify(stream)

    assert q_ctx.calls == [(tuple(tokens[20:40]), 0)]


def test_modify_tells_the_editor_where_the_decoded_segment_starts():
    """A windowed query covers only the segment's end, so the editor is told
    the segment start instead of inferring it from the query length."""
    tokens = list(range(40))
    stream, _ = _stream(
        tokens,
        segment_start=20,
        q_tensor=torch.zeros(1, 2, 2, 8),
        q_window=(2, True),
    )
    seen: list[int] = []

    def keep(tensors, tokens):
        seen.append(tensors.segment_start_token_id)
        assert set(tensors) == {LMCacheSDKCacheKind.KV, LMCacheSDKCacheKind.QUERY}
        return tensors[LMCacheSDKCacheKind.KV], tokens

    stream.modify_kv(keep, timeout=0.0, poll_interval=0.0)

    assert seen == [20]


def test_modify_fails_fast_on_an_empty_window():
    """A window with nothing to read raises instead of polling until timeout."""
    tokens = list(range(40))
    stream, _ = _stream(tokens, segment_start=40, q_tensor=None)

    with pytest.raises(LMCacheRequestStreamError, match="empty"):
        _modify(stream, timeout=30.0)


def test_modify_reports_the_chain_root_when_the_window_is_missing():
    """The error names the chain root, since a mismatched root is the way this
    fails."""
    tokens = list(range(40))
    stream, _ = _stream(tokens, segment_start=20, q_tensor=None)

    with pytest.raises(LMCacheRequestStreamError, match="chained from token 20"):
        _modify(stream)


def test_each_pass_takes_its_chain_root_from_the_cache():
    """The engine computes from wherever the cache runs out, so each pass asks
    the cache rather than assuming what an earlier store left behind."""
    kv_ctx = _RecordingContext(LMCacheSDKCacheKind.KV, None)
    stream = LMCacheRequestStream(
        contexts=[kv_ctx],  # type: ignore[list-item]
        post_completion=lambda *a, **kw: [],
        prompt_token_ids=list(range(40)),
    )

    stream.generate({"max_tokens": 0})
    assert stream._segment_start_token_id == 0

    kv_ctx.cached_tokens = 20
    stream.generate({"max_tokens": 0})
    assert stream._segment_start_token_id == 20


def test_a_short_cache_hit_moves_the_chain_root_back():
    """A hit shorter than what update() stored (eviction, a restarted server)
    moves the root with it, instead of leaving the retrieve addressing a chain
    the pass never wrote."""
    kv_ctx = _RecordingContext(LMCacheSDKCacheKind.KV, None)
    stream = LMCacheRequestStream(
        contexts=[kv_ctx],  # type: ignore[list-item]
        post_completion=lambda *a, **kw: [],
        prompt_token_ids=list(range(40)),
    )

    stream.update(LMCacheSDKCacheKind.KV, torch.zeros(2, 2, 22, 8), list(range(22)))
    assert stream._segment_start_token_id == 20  # what the store kept

    kv_ctx.cached_tokens = 8  # ... but only this much survives
    stream.generate({"max_tokens": 0})

    assert stream._segment_start_token_id == 8


def test_update_moves_the_chain_to_the_stored_prefix():
    """After an edit, the next pass reloads what store() kept (whole chunks
    only) and starts its chain there."""
    kv_ctx = _RecordingContext(LMCacheSDKCacheKind.KV, None)
    stream = LMCacheRequestStream(
        contexts=[kv_ctx],  # type: ignore[list-item]
        post_completion=lambda *a, **kw: [],
        prompt_token_ids=list(range(40)),
    )

    stream.update(LMCacheSDKCacheKind.KV, torch.zeros(2, 2, 22, 8), list(range(22)))
    stream.generate({"max_tokens": 0})

    assert stream._segment_start_token_id == 20


def test_stored_and_retrieved_chunks_hash_identically():
    """The regression: a chunk stored by a pass must be addressable by the
    modify that follows it. Both sides are hashed the way the server does.
    """
    tokens = list(range(40))
    segment_start = 20
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")

    # Store side: one pass writing [20, 40) as it decodes.
    store_key = _stored_key(tokens, op_start=20, op_end=40, segment_start=20)
    stored = hasher.compute_chunk_hashes(
        list(store_key.token_ids), start=store_key.start, end=store_key.end
    )

    # Retrieve side: the modify that follows reads the same window.
    origin = LMCacheSDKCacheKind.QUERY.key_origin(segment_start)
    window_tokens = len(tokens) - origin
    start_offset, _ = windowed_range(window_tokens, FULL_WINDOW, CHUNK, False)
    requested = hasher.compute_chunk_hashes(
        tokens[origin:], start=start_offset, end=window_tokens
    )

    assert stored == requested
    assert len(stored) == (40 - 20) // CHUNK


def test_chain_rooted_at_zero_would_not_match():
    """Guards the reason for re-basing: keeping the KV chain for query tensors
    puts the tail under hashes the compacted sequence never reproduces."""
    tokens = list(range(40))
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")

    from_zero = hasher.compute_chunk_hashes(tokens, start=20, end=40)
    from_segment = hasher.compute_chunk_hashes(tokens[20:], start=0, end=20)

    assert from_zero != from_segment


class _StubSDKContext:
    """Bare LMCacheSDKContext state for exercising retrieve() alone."""

    def __init__(
        self, hit_tokens: int, window: tuple[int, bool] = (FULL_WINDOW, False)
    ) -> None:
        self.chunk_size = CHUNK
        self.kind = LMCacheSDKCacheKind.QUERY
        self._window = window
        self.instance_id = 1
        self._hit_tokens = hit_tokens
        self.ended: list[str] = []
        self.retrieved_keys: list[IPCCacheServerKey] = []
        self.transfer_ctx = SimpleNamespace(retrieve=self._retrieve)

    # retrieve() runs the context's own range logic against this stub state.
    _retrieve_range = LMCacheSDKContext._retrieve_range

    def windowed_range(self, window_tokens: int) -> tuple[int, int]:
        return windowed_range(window_tokens, self._window[0], CHUNK, self._window[1])

    def maybe_submit_lookup_request(self, request_id, token_ids, cache_salt, **kw):
        self.lookup_tokens = list(token_ids)

    def _await_lookup_result(self, request_id: str) -> int:
        return self._hit_tokens

    def end_session(self, request_id: str) -> None:
        self.ended.append(request_id)

    def _create_key(self, token_ids, start, end, request_id, cache_salt="", **kw):
        return IPCCacheServerKey(
            model_name="m##query",
            world_size=1,
            worker_id=0,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )

    def _retrieve(self, key, instance_id):
        self.retrieved_keys.append(key)
        return torch.zeros(1, 2, key.end - key.start, 8), None


def _context_retrieve(
    hit_tokens: int,
    start_token_id: int,
    window: tuple[int, bool] = (FULL_WINDOW, False),
) -> tuple[_StubSDKContext, torch.Tensor | None]:
    """Call the real retrieve() against stub state."""
    ctx = _StubSDKContext(hit_tokens, window)
    result = LMCacheSDKContext.retrieve(
        ctx,  # type: ignore[arg-type]
        list(range(40)),
        "",
        start_token_id,
    )
    return ctx, result


def test_retrieve_stops_at_the_lookup_hit():
    """Chunks past the lookup's hit are not read-locked, so the read stops
    there instead of asking for a range the server cannot serve."""
    ctx, result = _context_retrieve(hit_tokens=24, start_token_id=8)

    assert result is not None
    key = ctx.retrieved_keys[0]
    assert (key.start, key.end) == (8, 24)


def test_retrieve_clamps_to_a_chunk_window():
    """Only a chunk window's trailing chunks are readable, so a retrieve from
    token 0 starts at the window instead of asking for unwritten chunks."""
    window = (2 * CHUNK, True)
    ctx, result = _context_retrieve(hit_tokens=24, start_token_id=0, window=window)

    assert result is not None
    key = ctx.retrieved_keys[0]
    assert (key.start, key.end) == (16, 24)


def test_retrieve_returns_none_when_the_hit_misses_the_start():
    """A hit that stops before the requested start yields nothing readable."""
    ctx, result = _context_retrieve(hit_tokens=8, start_token_id=8)

    assert result is None
    assert ctx.ended  # the session is released on the miss path


def test_retrieve_never_exceeds_the_chunk_aligned_range():
    """A hit longer than the token range is clamped to the range."""
    ctx, result = _context_retrieve(hit_tokens=400, start_token_id=0)

    assert ctx.retrieved_keys[0].end == 40


class _HybridContext:
    """A hybrid KV context recording what modify_kv stores."""

    kind = LMCacheSDKCacheKind.KV
    chunk_size = CHUNK
    is_hybrid = True

    def __init__(self, kv: torch.Tensor, state: RecurrentState) -> None:
        self._result = (kv, state)
        self.stored: list[tuple[tuple[int, ...], RecurrentState | None]] = []

    def lookup(self, tokens, cache_salt: str = "") -> int:
        return (len(tokens) // CHUNK) * CHUNK

    def retrieve_with_state(self, tokens, cache_salt: str = ""):
        return self._result

    def store(self, kv, tokens, cache_salt: str = "", recurrent_state=None) -> bool:
        self.stored.append((tuple(tokens), recurrent_state))
        return True


def test_modify_carries_the_retrieved_state_to_the_edited_prefix():
    """A hybrid edit only sees attention K/V; the recurrent state at the end
    of the retrieved prefix is stored as the edited prefix's."""
    kv = torch.randn(2, 2, 2 * CHUNK, 8)
    state = RecurrentState((torch.randn(1, 2, CHUNK, 1, 8),))
    ctx = _HybridContext(kv, state)
    stream = LMCacheRequestStream(
        contexts=[ctx],  # type: ignore[list-item]
        post_completion=MagicMock(),
        prompt_token_ids=list(range(2 * CHUNK + 3)),
    )
    seen = {}

    def drop_first_chunk(tensors, tokens):
        seen["kv"] = tensors[LMCacheSDKCacheKind.KV]
        return seen["kv"][:, :, CHUNK:], tokens[CHUNK:]

    stream.modify_kv(drop_first_chunk, timeout=0.0, poll_interval=0.0)

    assert seen["kv"] is kv
    assert ctx.stored == [(tuple(range(CHUNK, 2 * CHUNK)), state)]
