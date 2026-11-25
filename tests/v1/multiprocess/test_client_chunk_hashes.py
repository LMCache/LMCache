# SPDX-License-Identifier: Apache-2.0
"""Keys carrying client-computed chunk hashes instead of token ids."""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import random

# Third Party
import pytest

# First Party
from lmcache.integration.atom import multi_process_adapter as atom_adapter
from lmcache.integration.atom.multi_process_adapter import (
    AtomMPParallelConfig,
    AtomMPSchedulerAdapter,
)
from lmcache.v1.client.chunk_hashes import ChunkHashCache, lmcache_chunk_hashes
from lmcache.v1.common.custom_types import CHUNK_HASH_BYTES, IPCCacheServerKey
from lmcache.v1.common.token_codec import pack_token_ids
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import DEFAULT_SHARD_REPLICATION_DESC, AttnWindowDesc
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.engine_context import (
    MPCacheServerContext,
    check_chunk_hash_key,
)
from lmcache.v1.multiprocess.modules.lookup import (
    LookupModule,
    resolve_prefetched_obj_keys,
)
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.transport.futures import MessagingFuture

CHUNK = 16


def _tokens(n: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 1 << 20) for _ in range(n)]


def _hash_key(tokens: list[int], start: int, end: int) -> IPCCacheServerKey:
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    return IPCCacheServerKey.from_chunk_hashes(
        model_name="m",
        world_size=1,
        worker_id=0,
        chunk_hashes=hasher.compute_chunk_hashes(tokens, end=end),
        start=start,
        end=end,
        request_id="req",
    )


def _scheduler(
    monkeypatch: pytest.MonkeyPatch, **kwargs: object
) -> tuple[AtomMPSchedulerAdapter, MagicMock]:
    client = MagicMock(name="req_client")
    done: MessagingFuture[None] = MessagingFuture()
    done.set_result(None)
    client.lookup.return_value = done
    factory = MagicMock(name="request_client_factory")
    factory.create.return_value = client
    monkeypatch.setattr(atom_adapter, "RequestClientFactory", factory)
    monkeypatch.setattr(atom_adapter, "_get_chunk_size", lambda *args: CHUNK)
    scheduler = AtomMPSchedulerAdapter(
        server_url="tcp://127.0.0.1:5555",
        context=MagicMock(name="zmq_context"),
        model_name="m",
        block_size=CHUNK,
        parallel_config=AtomMPParallelConfig(1, 0, 1),
        **kwargs,  # type: ignore[arg-type]
    )
    return scheduler, client


def test_atom_client_hashes_match_server_token_hasher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lookups from a growing prompt carry the server's hashes, not tokens."""
    scheduler, client = _scheduler(monkeypatch, client_chunk_hashes=True)
    server = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(10 * CHUNK + 5)

    for length in (3 * CHUNK + 1, 7 * CHUNK, 10 * CHUNK + 5):
        # A fresh request id per lookup: lookups are submitted once per id.
        scheduler.maybe_submit_lookup_request(f"req-{length}", tokens[:length])
        key = client.lookup.call_args.args[0]
        aligned = length - length % CHUNK
        assert key.token_bytes == b""
        assert key.end == aligned
        assert key.split_chunk_hashes() == server.compute_chunk_hashes(
            tokens[:length], end=aligned
        )


def test_atom_client_hashes_follow_prompt_growth_and_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One request's hashes extend as it grows and restart if it changes."""
    scheduler, client = _scheduler(monkeypatch, client_chunk_hashes=True)
    server = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(8 * CHUNK)
    changed = tokens[: 3 * CHUNK] + _tokens(5 * CHUNK, seed=1)

    for prompt in (tokens[: 4 * CHUNK], tokens, changed):
        scheduler.free_lookup_locks(prompt, 0, len(prompt), "req")
        key = client.free_lookup_locks.call_args.args[0]
        assert key.split_chunk_hashes() == server.compute_chunk_hashes(prompt)


def test_chunk_hash_cache_matches_token_hasher() -> None:
    """The shared cache returns the server's hashes as a prompt grows."""
    cache = ChunkHashCache(CHUNK, max_requests=1)
    server = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(9 * CHUNK + 3)

    for length in (2 * CHUNK, 5 * CHUNK + 1, 9 * CHUNK + 3):
        assert cache.hashes("a", tokens[:length], length) == (
            server.compute_chunk_hashes(tokens[:length])
        )
    # end bounds the hashes; evicting "a" (max_requests=1) only drops a cache.
    assert cache.hashes("a", tokens, 3 * CHUNK + 2) == server.compute_chunk_hashes(
        tokens, end=3 * CHUNK
    )
    assert cache.hashes("b", tokens[:CHUNK], CHUNK) == server.compute_chunk_hashes(
        tokens[:CHUNK]
    )
    assert cache.hashes("a", tokens, len(tokens)) == server.compute_chunk_hashes(tokens)
    assert cache.hashes("a", tokens[: CHUNK - 1], CHUNK - 1) == []


def test_atom_default_keys_carry_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without client_chunk_hashes, keys keep sending the token prefix."""
    scheduler, client = _scheduler(monkeypatch)
    tokens = _tokens(4 * CHUNK)

    scheduler.maybe_submit_lookup_request("req", tokens)

    key = client.lookup.call_args.args[0]
    assert key.chunk_hashes == b""
    assert key.token_bytes == pack_token_ids(tokens)


@pytest.mark.parametrize(("start", "end"), [(0, 6 * CHUNK), (2 * CHUNK, 6 * CHUNK)])
def test_session_resolves_hash_keys_like_token_keys(start: int, end: int) -> None:
    """A session fed chunk hashes returns the hashes its tokens would give."""
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(6 * CHUNK + 3)
    by_tokens = Session(request_id="a", hasher=hasher)
    by_tokens.update_from_key(
        IPCCacheServerKey.from_token_ids("m", 1, 0, tokens, start, end, "a")
    )
    by_hashes = Session(request_id="b", hasher=hasher)
    by_hashes.update_from_key(_hash_key(tokens, start, end))

    assert by_hashes.get_hashes(start, end) == by_tokens.get_hashes(start, end)
    assert by_hashes.get_hashes(0) == by_tokens.get_hashes(0)
    with pytest.raises(ValueError, match="exceeds"):
        by_hashes.get_hashes(0, end + CHUNK)


def test_session_mixes_hash_and_token_keys() -> None:
    """Token keys extend a chain started from client hashes, and back."""
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(9 * CHUNK)
    session = Session(request_id="r", hasher=hasher)

    session.update_from_key(_hash_key(tokens, 0, 3 * CHUNK))
    session.update_from_key(
        IPCCacheServerKey.from_token_ids("m", 1, 0, tokens[: 6 * CHUNK], 0, 6 * CHUNK)
    )
    session.update_from_key(_hash_key(tokens, 6 * CHUNK, 9 * CHUNK))

    assert session.get_hashes(0, 9 * CHUNK) == hasher.compute_chunk_hashes(tokens)


def test_server_with_other_hash_algorithm_rejects_client_hashes() -> None:
    """Client hashes are blake3; another server algorithm refuses them."""
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="builtin")
    session = Session(request_id="r", hasher=hasher)

    with pytest.raises(ValueError, match="blake3"):
        session.update_from_key(_hash_key(_tokens(2 * CHUNK), 0, 2 * CHUNK))


def test_hash_key_validation() -> None:
    """A key carries tokens or whole hashes, never an unfull chunk."""
    key = _hash_key(_tokens(2 * CHUNK), 0, 2 * CHUNK)

    assert key.num_tokens == 2 * CHUNK
    assert key.no_worker_id_version().chunk_hashes == key.chunk_hashes
    assert not _hash_key(_tokens(3 * CHUNK), 0, 2 * CHUNK + 5).has_unfull_chunk(CHUNK)
    with pytest.raises(ValueError, match="either"):
        IPCCacheServerKey(
            "m", 1, 0, pack_token_ids([1]), 0, 1, "r", "", None, b"x" * 32
        )
    with pytest.raises(ValueError, match="whole number"):
        IPCCacheServerKey("m", 1, 0, b"", 0, 1, "r", "", None, b"x" * 31)


@pytest.mark.parametrize("hit_chunks", [-1, 4])
def test_lock_release_resolves_hash_keys_like_token_keys(hit_chunks: int) -> None:
    """free_lookup_locks resolves the same object keys from either key form."""
    ctx = MagicMock()
    ctx.layout_desc_registry.get_chunk_size.return_value = CHUNK
    ctx.layout_desc_registry.get_token_hasher.return_value = TokenHasher(
        chunk_size=CHUNK, hash_algorithm="blake3"
    )
    ctx.layout_desc_registry.find_attn_desc.return_value = AttnWindowDesc(
        num_chunks_in_sw=[-1, 2], world_size=1
    )
    ctx.layout_desc_registry.find_shard_replication_desc.return_value = (
        DEFAULT_SHARD_REPLICATION_DESC
    )
    tokens = _tokens(6 * CHUNK)
    token_key = IPCCacheServerKey.from_token_ids(
        "m", 1, None, tokens, CHUNK, 6 * CHUNK, "req"
    )
    hash_key = _hash_key(tokens, CHUNK, 6 * CHUNK).no_worker_id_version()

    by_hashes = resolve_prefetched_obj_keys(ctx, hash_key, hit_chunks, ())

    assert by_hashes
    assert by_hashes == resolve_prefetched_obj_keys(ctx, token_key, hit_chunks, ())


@pytest.mark.parametrize("bindings_needed", [False, True])
def test_hash_keys_fail_while_cache_events_need_tokens(bindings_needed: bool) -> None:
    """KV-event/coordinator reporting needs token ids, so hash keys fail."""
    ctx = MagicMock()
    ctx.event_bus.has_subscribers.side_effect = lambda event_type: (
        bindings_needed and event_type == EventType.MP_TOKENS
    )
    ctx.layout_desc_registry.get_chunk_size.return_value = CHUNK
    token_key = IPCCacheServerKey.from_token_ids("m", 1, 0, _tokens(CHUNK), 0, CHUNK)
    hash_key = _hash_key(_tokens(CHUNK), 0, CHUNK)

    check_chunk_hash_key(ctx, token_key)
    if bindings_needed:
        with pytest.raises(ValueError, match="token ids"):
            check_chunk_hash_key(ctx, hash_key)
        # A lookup fails before it publishes or hashes anything, and so does
        # resolving a store's or retrieve's object keys.
        with pytest.raises(ValueError, match="token ids"):
            LookupModule(ctx).lookup(hash_key, tp_size=1)
        ctx.event_bus.publish.assert_not_called()
        with pytest.raises(ValueError, match="token ids"):
            MPCacheServerContext.resolve_obj_keys(ctx, hash_key, [0])
        ctx.session_manager.get_or_create.assert_not_called()
    else:
        check_chunk_hash_key(ctx, hash_key)


@pytest.mark.parametrize("registered", [False, True])
def test_hash_keys_with_wrong_hash_count_fail(registered: bool) -> None:
    """A key must carry one hash per complete chunk of [0, end)."""
    ctx = MagicMock()
    ctx.event_bus.has_subscribers.return_value = False
    ctx.layout_desc_registry.get_chunk_size.return_value = CHUNK
    if not registered:
        ctx.layout_desc_registry.find.return_value = None
    tokens = _tokens(4 * CHUNK)
    # Hashed at twice the server's chunk size: half the hashes for [0, end).
    other = TokenHasher(chunk_size=2 * CHUNK, hash_algorithm="blake3")
    short = IPCCacheServerKey.from_chunk_hashes(
        "m", 1, 0, other.compute_chunk_hashes(tokens), 0, 4 * CHUNK, "req"
    )
    # One hash too many for the key's end.
    long = replace(_hash_key(tokens, 0, 4 * CHUNK), end=3 * CHUNK)

    check_chunk_hash_key(ctx, _hash_key(tokens, 0, 4 * CHUNK))
    check_chunk_hash_key(ctx, _hash_key(tokens, 0, 4 * CHUNK - 1))
    for key in (short, long):
        if not registered:
            # Left to the caller, which handles an unregistered key.
            check_chunk_hash_key(ctx, key)
            continue
        with pytest.raises(ValueError, match="chunk hash"):
            key.check_chunk_hash_count(CHUNK)
        with pytest.raises(ValueError, match="chunk hash"):
            check_chunk_hash_key(ctx, key)
        with pytest.raises(ValueError, match="chunk hash"):
            LookupModule(ctx).lookup(key, tp_size=1)
        with pytest.raises(ValueError, match="chunk hash"):
            LookupModule(ctx).free_lookup_locks(key, tp_size=1)
        with pytest.raises(ValueError, match="chunk hash"):
            MPCacheServerContext.resolve_obj_keys(ctx, key, [0])
        ctx.event_bus.publish.assert_not_called()
        ctx.session_manager.get_or_create.assert_not_called()


# -- callers that send the hashes themselves --------------------------------


def test_lmcache_chunk_hashes_match_server_token_hasher() -> None:
    """The exported chain is the server's, and continues from a prior link."""
    server = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    tokens = _tokens(7 * CHUNK + 3)
    expected = b"".join(server.compute_chunk_hashes(tokens))

    assert lmcache_chunk_hashes(tokens, CHUNK) == expected
    head = lmcache_chunk_hashes(tokens[: 3 * CHUNK], CHUNK)
    tail = lmcache_chunk_hashes(tokens[3 * CHUNK :], CHUNK, head[-CHUNK_HASH_BYTES:])
    assert head + tail == expected
    assert lmcache_chunk_hashes(tokens[: CHUNK - 1], CHUNK) == b""


def test_caller_hash_lookup_and_lock_keys_match_token_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller hashes give the very keys the adapter builds from the tokens."""
    by_tokens, token_client = _scheduler(monkeypatch, client_chunk_hashes=True)
    by_hashes, hash_client = _scheduler(monkeypatch, client_chunk_hashes=True)
    tokens = _tokens(9 * CHUNK + 7)
    hashes = lmcache_chunk_hashes(tokens, CHUNK)

    by_tokens.maybe_submit_lookup_request("req", tokens)
    by_hashes.maybe_submit_lookup_request("req", [], chunk_hashes=hashes)
    token_key = token_client.lookup.call_args.args[0]
    hash_key = hash_client.lookup.call_args.args[0]
    assert hash_key == token_key
    assert hash_key.chunk_hashes == token_key.chunk_hashes == hashes
    assert hash_key.end == 9 * CHUNK

    for start, end in ((0, 4 * CHUNK), (2 * CHUNK, 9 * CHUNK)):
        by_tokens.free_lookup_locks(tokens, start, end, "req")
        by_hashes.free_lookup_locks(
            [],
            start,
            end,
            "req",
            chunk_hashes=hashes[: end // CHUNK * CHUNK_HASH_BYTES],
        )
        token_key = token_client.free_lookup_locks.call_args.args[0]
        hash_key = hash_client.free_lookup_locks.call_args.args[0]
        assert hash_key == token_key
        assert hash_key.chunk_hashes == token_key.chunk_hashes


def test_caller_hashes_must_cover_whole_chunks_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashes must be whole and one per complete chunk of [0, end)."""
    scheduler, client = _scheduler(monkeypatch, client_chunk_hashes=True)
    hashes = lmcache_chunk_hashes(_tokens(4 * CHUNK), CHUNK)

    with pytest.raises(ValueError, match="whole number"):
        scheduler.free_lookup_locks([], 0, 4 * CHUNK, "req", chunk_hashes=hashes[:-1])
    with pytest.raises(ValueError, match="whole number"):
        scheduler.maybe_submit_lookup_request("req", [], chunk_hashes=b"x")
    with pytest.raises(ValueError, match="expected 3"):
        scheduler.free_lookup_locks([], 0, 3 * CHUNK + 5, "req", chunk_hashes=hashes)
    # A partial last chunk adds no hash.
    scheduler.free_lookup_locks(
        [], CHUNK, 3 * CHUNK + 5, "req", chunk_hashes=hashes[: 3 * CHUNK_HASH_BYTES]
    )
    assert client.free_lookup_locks.call_args.args[0].end == 3 * CHUNK + 5


def test_caller_hashes_need_client_chunk_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter sending tokens refuses caller hashes instead of mixing."""
    scheduler, client = _scheduler(monkeypatch)
    hashes = lmcache_chunk_hashes(_tokens(2 * CHUNK), CHUNK)

    assert AtomMPSchedulerAdapter.ACCEPTS_CALLER_CHUNK_HASHES
    with pytest.raises(ValueError, match="client_chunk_hashes"):
        scheduler.maybe_submit_lookup_request("req", [], chunk_hashes=hashes)
    with pytest.raises(ValueError, match="client_chunk_hashes"):
        scheduler.free_lookup_locks([], 0, 2 * CHUNK, "req", chunk_hashes=hashes)
    # Also when there is no complete chunk to look up or release.
    with pytest.raises(ValueError, match="client_chunk_hashes"):
        scheduler.maybe_submit_lookup_request("empty", [], chunk_hashes=b"")
    with pytest.raises(ValueError, match="client_chunk_hashes"):
        scheduler.free_lookup_locks([], 0, 0, "req", chunk_hashes=b"")
    client.lookup.assert_not_called()
    client.free_lookup_locks.assert_not_called()
