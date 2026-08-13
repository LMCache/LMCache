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
from lmcache.sdk.cache_kind import (
    LMCacheSDKCacheKind,
    LMCacheSDKCacheSpan,
    LMCacheSDKCacheSpanKind,
)
from lmcache.sdk.context import LMCacheSDKContext
from lmcache.sdk.qringbuffer import QRingBufferAdapter
from lmcache.sdk.request import LMCacheRequestStream, LMCacheRequestStreamError
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK = 4


def test_kv_chains_from_token_zero():
    """KV covers every token, so its chain roots at 0 regardless of segment."""
    assert LMCacheSDKCacheKind.KV.key_origin(segment_start=512) == 0


def test_query_chains_from_the_segment():
    """Query rows exist only past the cached prefix, so the chain roots there."""
    assert LMCacheSDKCacheKind.QUERY.key_origin(segment_start=512) == 512


def test_all_span_reads_the_whole_window():
    """ALL starts at the window's first token, whichever kind it belongs to."""
    span = LMCacheSDKCacheSpan()

    assert span.kind is LMCacheSDKCacheSpanKind.ALL
    assert span.start_offset(window_tokens=20, chunk_size=CHUNK) == 0
    assert span.expected_tokens(window_tokens=20, chunk_size=CHUNK) == 20


def test_trailing_span_starts_a_fixed_number_of_chunks_from_the_end():
    """TRAILING takes its last trailing_chunks chunks."""
    span = LMCacheSDKCacheSpan(kind=LMCacheSDKCacheSpanKind.TRAILING, trailing_chunks=2)

    assert span.start_offset(window_tokens=20, chunk_size=CHUNK) == 12
    assert span.expected_tokens(window_tokens=20, chunk_size=CHUNK) == 8


def test_trailing_span_clamps_to_the_window():
    """A range longer than the window stops at its first token: nothing before
    the window exists under this kind's key chain."""
    span = LMCacheSDKCacheSpan(
        kind=LMCacheSDKCacheSpanKind.TRAILING, trailing_chunks=10
    )

    assert span.start_offset(window_tokens=20, chunk_size=CHUNK) == 0
    assert span.expected_tokens(window_tokens=20, chunk_size=CHUNK) == 20


def test_every_span_kind_is_handled():
    """No span kind falls through to the unhandled-kind raise."""
    for kind in LMCacheSDKCacheSpanKind:
        span = LMCacheSDKCacheSpan(kind=kind)
        assert span.start_offset(window_tokens=20, chunk_size=CHUNK) >= 0


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
    q_adapter = QRingBufferAdapter(worker, "m##query", MagicMock())
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
        span: LMCacheSDKCacheSpan | None = None,
    ) -> None:
        self.kind = kind
        self.span = span if span is not None else LMCacheSDKCacheSpan()
        self.chunk_size = CHUNK
        self._tensor = tensor
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.stored: list[tuple[int, ...]] = []
        self.cached_tokens = 0

    def lookup(self, tokens, cache_salt: str = "") -> int:
        return min(self.cached_tokens, (len(tokens) // CHUNK) * CHUNK)

    def retrieve(
        self, tokens, cache_salt: str = "", start_token_id: int = 0
    ) -> torch.Tensor | None:
        self.calls.append((tuple(tokens), start_token_id))
        return self._tensor

    def store(self, kv, tokens, cache_salt: str = "") -> bool:
        self.stored.append(tuple(tokens))
        self.cached_tokens = (len(tokens) // CHUNK) * CHUNK
        return True


def _stream(
    tokens: list[int],
    segment_start: int,
    q_tensor: torch.Tensor | None,
    q_span: LMCacheSDKCacheSpan | None = None,
) -> tuple[LMCacheRequestStream, _RecordingContext]:
    """A stream that has run a pass computing tokens[segment_start:].

    Built the way the SDK does it: a first pass over the cached prefix, then a
    pass that decodes the rest.
    """
    kv_ctx = _RecordingContext(
        LMCacheSDKCacheKind.KV, torch.zeros(2, 2, len(tokens), 8)
    )
    q_ctx = _RecordingContext(LMCacheSDKCacheKind.QUERY, q_tensor, span=q_span)
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
    """modify_kv reads the query span through the segment's own chain: the
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
        q_span=LMCacheSDKCacheSpan(
            kind=LMCacheSDKCacheSpanKind.TRAILING, trailing_chunks=2
        ),
    )

    _modify(stream)

    window, relative_start = q_ctx.calls[0]
    assert window == tuple(tokens[20:40])
    assert relative_start == 12


def test_modify_fails_fast_on_an_empty_span():
    """A span with nothing to read raises instead of polling until timeout."""
    tokens = list(range(40))
    stream, _ = _stream(tokens, segment_start=40, q_tensor=None)

    with pytest.raises(LMCacheRequestStreamError, match="empty"):
        _modify(stream, timeout=30.0)


def test_modify_reports_the_chain_root_when_the_span_is_missing():
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

    # Retrieve side: the modify that follows reads the same span.
    span = LMCacheSDKCacheSpan()
    origin = LMCacheSDKCacheKind.QUERY.key_origin(segment_start)
    window_tokens = len(tokens) - origin
    start_offset = span.start_offset(window_tokens, CHUNK)
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

    def __init__(self, hit_tokens: int) -> None:
        self.chunk_size = CHUNK
        self.kind = LMCacheSDKCacheKind.QUERY
        self.instance_id = 1
        self._hit_tokens = hit_tokens
        self.ended: list[str] = []
        self.retrieved_keys: list[IPCCacheServerKey] = []
        self.transfer_ctx = SimpleNamespace(retrieve=self._retrieve)

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
        return torch.zeros(1, 2, key.end - key.start, 8)


def _context_retrieve(
    hit_tokens: int, start_token_id: int
) -> tuple[_StubSDKContext, torch.Tensor | None]:
    """Call the real retrieve() against stub state."""
    ctx = _StubSDKContext(hit_tokens)
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


def test_retrieve_returns_none_when_the_hit_misses_the_span():
    """A hit that stops before the requested start yields nothing readable."""
    ctx, result = _context_retrieve(hit_tokens=8, start_token_id=8)

    assert result is None
    assert ctx.ended  # the session is released on the miss path


def test_retrieve_never_exceeds_the_chunk_aligned_range():
    """A hit longer than the token range is clamped to the range."""
    ctx, result = _context_retrieve(hit_tokens=400, start_token_id=0)

    assert ctx.retrieved_keys[0].end == 40
