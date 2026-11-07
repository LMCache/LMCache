# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the segment-relative key chain used by query tensors.
The KV cache covers every token, so its keys chain from token 0, but query
tensors exist only for the tokens a pass computed (the KV before them came from
the cache), so their keys chain from that pass's first token. retrieve() then
never read-locks chunks before the segment, which the pass never wrote and the
server cannot serve.
These tests pin both halves of the contract: the store side keys each pass from
its own first computed token, and the retrieve side addresses that same chain.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import contextlib

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.context import LMCacheSDKContext
from lmcache.sdk.qringbuffer import QRingBufferAdapter
from lmcache.sdk.request import LMCacheRequestStream, LMCacheRequestStreamError
from lmcache.v1.common.token_codec import pack_token_ids, unpack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

CHUNK = 4


def test_kv_chains_from_token_zero_and_query_from_the_segment():
    """KV covers every token; query rows exist only past the cached prefix."""
    assert LMCacheSDKCacheKind.KV.key_origin(segment_start=512) == 0
    assert LMCacheSDKCacheKind.QUERY.key_origin(segment_start=512) == 512


class _FakeOp:
    """Minimal LoadStoreOp stand-in for a query store."""

    def __init__(self, token_ids: list[int], start: int, end: int) -> None:
        self.token_bytes = pack_token_ids(token_ids)
        self.start = start
        self.end = end


def _q_adapter() -> tuple[QRingBufferAdapter, MagicMock]:
    """A QRingBufferAdapter whose worker adapter records the keys it builds."""
    worker = MagicMock()
    worker.is_healthy = True
    worker.instance_id = 7
    worker.blocks_in_chunk = 1
    worker._create_key.side_effect = lambda token_bytes, start, end, **kw: (
        IPCCacheServerKey(
            model_name="m",
            world_size=1,
            worker_id=None,
            token_bytes=token_bytes,
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


@pytest.mark.parametrize("segment_start", [0, 20])
def test_store_key_is_rebased_on_the_segment(segment_start):
    """The stored key drops the tokens before the segment and shifts its range,
    under the query namespace and its own server session (the server extends
    one hash chain per request id, so it cannot share the KV store's)."""
    tokens = list(range(40))

    key = _stored_key(tokens, segment_start, segment_start + 8, segment_start)

    assert unpack_token_ids(key.token_bytes) == tokens[segment_start:]
    assert (key.start, key.end) == (0, 8)
    assert (key.model_name, key.request_id) == ("m##query", "req##query")


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

    def __init__(self, kind: LMCacheSDKCacheKind, tensor: torch.Tensor | None) -> None:
        self.kind = kind
        self.chunk_size = CHUNK
        self._tensor = tensor
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.stored: list[tuple[int, ...]] = []
        self.cached_tokens = 0

    def lookup(self, tokens, cache_salt: str = "") -> int:
        return min(self.cached_tokens, (len(tokens) // CHUNK) * CHUNK)

    def retrieve(
        self, tokens, cache_salt: str = "", start_token_id: int = 0
    ) -> tuple[torch.Tensor, None] | None:
        self.calls.append((tuple(tokens), start_token_id))
        return (self._tensor, None) if self._tensor is not None else None

    def store(self, kv, tokens, cache_salt: str = "") -> bool:
        self.stored.append(tuple(tokens))
        self.cached_tokens = (len(tokens) // CHUNK) * CHUNK
        return True


def _stream(
    tokens: list[int],
    segment_start: int,
    q_tensor: torch.Tensor | None,
) -> tuple[LMCacheRequestStream, _RecordingContext]:
    """A stream that has run a pass computing tokens[segment_start:].

    Built the way the SDK does it: a first pass over the cached prefix, then a
    pass that decodes the rest.
    """
    kv_ctx = _RecordingContext(
        LMCacheSDKCacheKind.KV, torch.zeros(2, 2, len(tokens), 8)
    )
    q_ctx = _RecordingContext(LMCacheSDKCacheKind.QUERY, q_tensor)
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


def test_modify_reads_query_from_the_segment_and_kv_from_zero():
    """modify_kv reads the query window from the segment, KV from token 0, and
    tells the editor where the segment starts."""
    tokens = list(range(40))
    stream, q_ctx = _stream(tokens, segment_start=20, q_tensor=torch.zeros(1, 2, 20, 8))
    kv_ctx = stream._contexts[LMCacheSDKCacheKind.KV]
    seen: list[int] = []

    def keep(tensors, tokens):
        seen.append(tensors.segment_start_token_id)
        return tensors[LMCacheSDKCacheKind.KV], tokens

    stream.modify_kv(keep, timeout=0.0, poll_interval=0.0)

    assert q_ctx.calls == [(tuple(tokens[20:40]), 0)]
    assert kv_ctx.calls == [(tuple(tokens), 0)]
    assert seen == [20]


def test_modify_fails_fast_on_an_empty_window():
    """A window with nothing to read raises instead of polling until timeout."""
    tokens = list(range(40))
    stream, _ = _stream(tokens, segment_start=40, q_tensor=None)

    with pytest.raises(LMCacheRequestStreamError, match="empty"):
        _modify(stream, timeout=30.0)


def test_modify_waits_for_every_query_row():
    """Query stores may still be in flight: a retrieve returning fewer rows
    than the segment holds is retried until the timeout, then reported."""
    tokens = list(range(40))
    stream, q_ctx = _stream(tokens, segment_start=20, q_tensor=torch.zeros(1, 2, 8, 8))

    with pytest.raises(LMCacheRequestStreamError, match="returned 8 rows"):
        _modify(stream, timeout=0.05)
    assert len(q_ctx.calls) > 1  # retried before giving up
    assert q_ctx.calls[0] == (tuple(tokens[20:40]), 0)


def test_modify_waits_for_the_generated_kv_to_be_stored():
    """The KV of the latest pass must be stored before it is retrieved, or
    the edit would drop its tail."""
    tokens = list(range(40))
    stream, _ = _stream(tokens, segment_start=20, q_tensor=torch.zeros(1, 2, 20, 8))
    stream._contexts[LMCacheSDKCacheKind.KV].cached_tokens = 32  # type: ignore[attr-defined]

    with pytest.raises(LMCacheRequestStreamError, match="still incomplete"):
        _modify(stream)


def test_public_retrieve_of_query_reads_from_the_segment():
    """retrieve(QUERY) from token 0 addresses the query chain from its root,
    rather than a negative offset into it."""
    tokens = list(range(40))
    stream, q_ctx = _stream(tokens, segment_start=20, q_tensor=torch.zeros(1, 2, 20, 8))

    stream.retrieve(LMCacheSDKCacheKind.QUERY, timeout=0.0)

    assert q_ctx.calls == [(tuple(tokens[20:40]), 0)]


def test_each_pass_takes_its_chain_root_from_the_cache():
    """Each pass roots its chain where the cache runs out: after update() that
    is the stored whole chunks, and a shorter hit (eviction) moves it back."""
    kv_ctx = _RecordingContext(LMCacheSDKCacheKind.KV, None)
    stream = LMCacheRequestStream(
        contexts=[kv_ctx],  # type: ignore[list-item]
        post_completion=lambda *a, **kw: [],
        prompt_token_ids=list(range(40)),
    )

    stream.generate({"max_tokens": 0})
    assert stream._segment_start_token_id == 0

    stream.update(LMCacheSDKCacheKind.KV, torch.zeros(2, 2, 22, 8), list(range(22)))
    stream.generate({"max_tokens": 0})
    assert stream._segment_start_token_id == 20

    kv_ctx.cached_tokens = 8
    stream.generate({"max_tokens": 0})
    assert stream._segment_start_token_id == 8


def test_stored_and_retrieved_chunks_hash_identically():
    """A chunk stored by a pass is addressable by the modify that follows it,
    which a chain rooted at token 0 would not be."""
    tokens = list(range(40))
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")

    key = _stored_key(tokens, op_start=20, op_end=40, segment_start=20)
    stored = hasher.compute_chunk_hashes(
        unpack_token_ids(key.token_bytes), start=key.start, end=key.end
    )
    requested = hasher.compute_chunk_hashes(tokens[20:], start=0, end=20)

    assert stored == requested and len(stored) == 20 // CHUNK
    assert hasher.compute_chunk_hashes(tokens, start=20, end=40) != requested


class _StubSDKContext:
    """Bare LMCacheSDKContext state for exercising retrieve() alone."""

    def __init__(self, hit_tokens: int) -> None:
        self.chunk_size = CHUNK
        self.kind = LMCacheSDKCacheKind.QUERY
        self.instance_id = 1
        self._hit_tokens = hit_tokens
        self.ended: list[str] = []
        self.retrieved_keys: list[IPCCacheServerKey] = []
        self.transfer_ctx = SimpleNamespace(
            retrieve=self._retrieve, transfer_slot=contextlib.nullcontext()
        )
        self._read_ttl = 300
        self._world_size = 1
        self._mq_timeout = 1.0
        self._req_client = SimpleNamespace(
            free_lookup_locks=lambda *_a: SimpleNamespace(
                result=lambda timeout=None: None
            )
        )

    def wait_for_lookup(
        self, request_id, tokens, cache_salt, request_configs, total_tokens
    ) -> int:
        return min(self._hit_tokens, total_tokens)

    def end_session(self, request_id: str) -> None:
        self.ended.append(request_id)

    def _create_key(self, token_ids, start, end, request_id, cache_salt="", **kw):
        return IPCCacheServerKey.from_token_ids(
            model_name="m##query",
            world_size=1,
            worker_id=0,
            token_ids=list(token_ids),
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
) -> tuple[_StubSDKContext, tuple[torch.Tensor, None] | None]:
    """Call the real retrieve() against stub state."""
    ctx = _StubSDKContext(hit_tokens)
    result = LMCacheSDKContext.retrieve(
        ctx,  # type: ignore[arg-type]
        list(range(40)),
        "",
        start_token_id=start_token_id,
    )
    return ctx, result


@pytest.mark.parametrize(
    ("hit_tokens", "start_token_id", "expected"), [(24, 8, (8, 24)), (400, 0, (0, 40))]
)
def test_retrieve_reads_up_to_the_lookup_hit(hit_tokens, start_token_id, expected):
    """The read stops at the lookup's hit (chunks past it are not read-locked),
    clamped to the chunk-aligned token range."""
    ctx, result = _context_retrieve(hit_tokens, start_token_id)

    assert result is not None
    tensor, state = result
    assert tensor.shape[2] == expected[1] - expected[0] and state is None
    key = ctx.retrieved_keys[0]
    assert (key.start, key.end) == expected


def test_retrieve_returns_none_when_the_hit_misses_the_start():
    """A hit that stops before the requested start yields nothing readable."""
    ctx, result = _context_retrieve(hit_tokens=8, start_token_id=8)

    assert result is None
    # Both the re-lookup's session and the warm-up's are ended on the miss path.
    assert len(ctx.ended) == 2
    assert sum(id.startswith("retrieve-warmup-") for id in ctx.ended) == 1


def test_retrieve_rejects_an_unaligned_start():
    with pytest.raises(Exception, match="multiple of chunk_size"):
        _context_retrieve(hit_tokens=24, start_token_id=2)
