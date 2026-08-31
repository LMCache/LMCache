# SPDX-License-Identifier: Apache-2.0
"""Sparse-prefetch read locks must be released when a request never retrieves.

The blend unified lookup read-locks every found chunk's object keys and
stashes them (``Session.extras``) for the retrieve. A client that drops all
its matches — e.g. every match falls inside vLLM's local prefix-cache
coverage — never sends ``CB_RETRIEVE_PRE_COMPUTED``, and before this fix
nothing released those locks: the retrieve's orphan sweep never ran, and
``free_lookup_locks`` covers only the prefix leg's lock model. The chunks
stayed pinned in L1 for the server's lifetime, with counts stacking on every
repeat lookup.

The fix: ``BlendModule`` registers a ``SessionManager`` destroy listener that
releases whatever the request never consumed — covering END_SESSION removal
at request end and the TTL reaper for clients that died without one.

These tests drive the real ``BlendModule``, ``BlendTokenRangeMatcher``,
``SessionManager`` and key expansion; only the storage manager is a
lock-counting fake, so the assertions are on actual lock accounting.
"""

# Standard
from unittest.mock import MagicMock

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, TrimPolicy
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.blend import BlendModule
from lmcache.v1.multiprocess.session import Session, SessionManager

CHUNK = 256
N_CHUNKS = 4


class _LockCountingStorageManager:
    """Counts read locks per key: the sparse prefetch locks every submitted
    key; ``finish_read_prefetched`` is the only release. Over-release raises.
    """

    def __init__(self) -> None:
        self.locks: dict = {}

    def submit_prefetch_task(self, spec, external_request_id=None):
        if spec.policy == TrimPolicy.SPARSE:
            for key in spec.keys:
                self.locks[key] = self.locks.get(key, 0) + 1
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


def _make_ctx(
    storage_manager: _LockCountingStorageManager,
    ttl: float = SessionManager.DEFAULT_SESSION_TTL,
) -> MagicMock:
    """Mock server context with a REAL SessionManager (no cleanup thread)."""
    ctx = MagicMock()
    ctx.chunk_size = CHUNK
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


def _lookup_key(query: list[int], request_id: str) -> IPCCacheServerKey:
    return IPCCacheServerKey(
        model_name="m",
        world_size=1,
        num_kv_readers=1,
        worker_id=None,
        token_ids=tuple(query),
        start=0,
        end=len(query),
        request_id=request_id,
    )


def _run_lookup(blend: BlendModule, request_id: str) -> None:
    """Register N_CHUNKS fingerprints and run a lookup that finds them all
    shifted (prefix coverage 0), i.e. the sparse leg locks every chunk."""
    stored_tokens = list(range(1000, 1000 + N_CHUNKS * CHUNK))
    token_hashes = [f"{request_id}-hash{i}".encode() for i in range(N_CHUNKS)]
    indexed = blend._token_range_matcher.on_new_token_hashes(
        stored_tokens, token_hashes, start_chunk_idx=0, position_offset=0
    )
    assert indexed == N_CHUNKS
    query = list(range(50_000, 50_128)) + stored_tokens
    result = blend.cb_unified_lookup(_lookup_key(query, request_id), tp_size=1)
    assert result is not None
    assert result.prefix_coverage_tokens == 0
    assert len(result.non_prefix_segments) == N_CHUNKS


def test_session_end_releases_unretrieved_sparse_locks():
    """The leak scenario: lookup locks N chunks, the client never retrieves
    (all matches shadowed by its local prefix cache), the request ends.
    END_SESSION's session removal must release every sparse read lock."""
    storage = _LockCountingStorageManager()
    ctx = _make_ctx(storage)
    blend = BlendModule(ctx, lmcache_driven_transfer=MagicMock())

    _run_lookup(blend, "req-shadowed")
    assert storage.outstanding() == N_CHUNKS

    # No retrieve. Request ends: END_SESSION removes the session.
    ctx.session_manager.remove("req-shadowed")
    assert storage.outstanding() == 0


def test_retrieve_take_prevents_double_release_at_session_end():
    """A consumed stash releases nothing at session end: the retrieve's
    take empties it, so the destroy listener is a no-op (the counting fake
    raises on any over-release)."""
    # First Party
    from lmcache.v1.multiprocess.modules.blend import _take_unretrieved_keys

    storage = _LockCountingStorageManager()
    ctx = _make_ctx(storage)
    blend = BlendModule(ctx, lmcache_driven_transfer=MagicMock())

    _run_lookup(blend, "req-retrieved")

    # Emulate the retrieve's consumption + release of the taken keys.
    cached = _take_unretrieved_keys(ctx.session_manager.get("req-retrieved"))
    assert cached is not None and len(cached) == N_CHUNKS
    storage.finish_read_prefetched([key for keys in cached.values() for key in keys])
    assert storage.outstanding() == 0

    # Session end must not release again.
    ctx.session_manager.remove("req-retrieved")
    assert storage.outstanding() == 0


def test_ttl_cleanup_releases_unretrieved_sparse_locks():
    """A session reaped by TTL cleanup (client died without END_SESSION)
    releases its unretrieved locks through the same destroy listener."""
    storage = _LockCountingStorageManager()
    ctx = _make_ctx(storage, ttl=0.0)
    blend = BlendModule(ctx, lmcache_driven_transfer=MagicMock())

    _run_lookup(blend, "req-abandoned")
    assert storage.outstanding() == N_CHUNKS

    assert ctx.session_manager.cleanup_expired() == 1
    assert storage.outstanding() == 0


def test_close_unregisters_the_destroy_listener():
    """After BlendModule.close(), destroying sessions calls nothing on the
    (now torn down) module."""
    storage = _LockCountingStorageManager()
    ctx = _make_ctx(storage)
    blend = BlendModule(ctx, lmcache_driven_transfer=MagicMock())

    _run_lookup(blend, "req-late")
    blend.close()
    # The stash is still on the session; with the listener gone, removal
    # releases nothing (server shutdown path — locks die with the process).
    ctx.session_manager.remove("req-late")
    assert storage.outstanding() == N_CHUNKS


def test_destroy_listener_failure_does_not_break_removal():
    """A raising listener is logged and swallowed; removal still completes."""

    def _boom(session: Session) -> None:
        raise RuntimeError("listener failure")

    manager = SessionManager(hasher=MagicMock(), cleanup_interval=None)
    manager.add_destroy_listener(_boom)
    manager.get_or_create("req-x")
    assert manager.remove("req-x") is not None
    assert manager.get("req-x") is None


def test_remove_destroy_listener_is_idempotent():
    """Removing an unregistered listener is a no-op."""
    manager = SessionManager(hasher=MagicMock(), cleanup_interval=None)

    def _listener(session: Session) -> None:
        pass

    manager.remove_destroy_listener(_listener)  # not registered: no-op
    manager.add_destroy_listener(_listener)
    manager.remove_destroy_listener(_listener)
    manager.remove_destroy_listener(_listener)  # second removal: no-op
