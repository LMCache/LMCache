# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any, Optional, overload
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    PeriodicThreadRegistry,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    token_ids: list[int] = field(default_factory=list)
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)
    lookup_ipc_key: Optional[IPCCacheServerKey] = None
    prefetch_hit_chunks: int = -1
    prefetch_locked_gids: tuple = ()
    prefetch_group_windows: tuple[int, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
    _lookup_generation: int = field(default=0, repr=False)
    _failed_retrieve_releases: set[tuple[int, int, int, int, int]] = field(
        default_factory=set, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        with self._lock:
            self.token_ids = full_token_ids

    @overload
    def get_hashes(self, start: int, end: int) -> list: ...

    @overload
    def get_hashes(self, start: int) -> list: ...

    def get_hashes(self, start: int, end: int | None = None) -> list:
        """Compute and return chunk hashes for the [start, end) token range.

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks.

        Two calling conventions are supported (declared via ``@overload``)::

            get_hashes(start, end)  # explicit end
            get_hashes(start)       # end = last full-chunk boundary

        Args:
            start: Start token index (must be aligned to chunk_size).
            end: End token index (must be aligned to chunk_size).
                When omitted (``None``), automatically set to the last
                full-chunk boundary of the current token sequence.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).

        Raises:
            ValueError: If an explicit ``end`` exceeds the current tokens
                (hashing past them yields valid-looking garbage).
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size

        with self._lock:
            if end is not None and end > len(self.token_ids):
                raise ValueError(
                    f"get_hashes end ({end}) exceeds the session's "
                    f"{len(self.token_ids)} token(s); the session may have "
                    "been recreated after request cleanup"
                )
            if end is None:
                # No explicit end: use the last full-chunk boundary.
                # Lock must be held here because `self.token_ids` may be
                # concurrently replaced by `set_tokens` from another thread.
                end = len(self.token_ids) - (len(self.token_ids) % chunk_size)
            assert end % chunk_size == 0, (
                f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
            )
            end_chunk = end // chunk_size
            self._compute_hash(end_chunk)
            return self.chunk_hashes[start_chunk:end_chunk]

    def _compute_hash(self, end_chunk: int) -> None:
        """Compute rolling hashes up to end_chunk.

        Uses cached state to skip already-computed chunks.

        Args:
            end_chunk: Compute hashes up to (but not including) this chunk.
        """
        chunk_size = self.hasher.chunk_size

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * chunk_size
            ce = cs + chunk_size
            chunk = self.token_ids[cs:ce]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else self.hasher.none_hash
            )
            h = self.hasher.hash_tokens(chunk, prefix)
            self.last_prefix_hash = h
            self.chunk_hashes.append(h)
            self.num_chunks_processed += 1

    def begin_lookup(
        self,
        key: IPCCacheServerKey,
        group_windows: tuple[int, ...],
    ) -> None:
        """Record a new lookup and reset its per-lookup release state."""
        with self._lock:
            self.lookup_ipc_key = key
            self.prefetch_hit_chunks = -1
            self.prefetch_locked_gids = ()
            self.prefetch_group_windows = group_windows
            self._lookup_generation += 1
            self._failed_retrieve_releases.clear()

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
    ) -> None:
        """Record the lock set acquired by the current lookup."""
        with self._lock:
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids

    def prepare_failed_retrieve_release(
        self,
        key: IPCCacheServerKey,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Return a stable snapshot for a failed worker's lock release."""
        if key.worker_id is None:
            return None

        with self._lock:
            lookup_key = self.lookup_ipc_key
            hit_chunks = self.prefetch_hit_chunks
            if lookup_key is None or hit_chunks < 0:
                return None

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_ids == lookup_key.token_ids
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return None

            return (
                hit_chunks,
                self.prefetch_locked_gids,
                self.prefetch_group_windows,
                self._lookup_generation,
            )

    def claim_failed_retrieve_release(
        self,
        instance_id: int,
        key: IPCCacheServerKey,
        lookup_generation: int,
    ) -> bool:
        """Atomically claim one failed worker's prepared lock release.

        A scheduler lookup acquires read locks for every KV worker (or one
        reader count per TP worker for MLA), while RETRIEVE responses are
        per worker instance.  When a worker has lost its GPU registration,
        only that instance's share may be released.  Claiming after key
        resolution makes duplicate failed RETRIEVEs idempotent without
        consuming the claim when resolution itself fails.

        ``False`` is also returned when the session cannot prove ownership of
        the requested range.  In that case it is safer to leave the lock to
        its TTL than to decrement a concurrent request's anonymous L1 count.
        """
        if key.worker_id is None:
            return False

        with self._lock:
            lookup_key = self.lookup_ipc_key
            if lookup_key is None or lookup_generation != self._lookup_generation:
                return False

            same_lookup = (
                key.model_name == lookup_key.model_name
                and key.world_size == lookup_key.world_size
                and key.token_ids == lookup_key.token_ids
                and key.cache_salt == lookup_key.cache_salt
                and key.start >= lookup_key.start
                and key.end <= lookup_key.end
            )
            if not same_lookup:
                return False

            owner = (
                lookup_generation,
                instance_id,
                key.worker_id,
                key.start,
                key.end,
            )
            if owner in self._failed_retrieve_releases:
                return False
            self._failed_retrieve_releases.add(owner)
            return True


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes
    DEFAULT_CLEANUP_INTERVAL = 60.0

    def __init__(
        self,
        hasher: TokenHasher,
        ttl: float = DEFAULT_SESSION_TTL,
        cleanup_interval: float | None = DEFAULT_CLEANUP_INTERVAL,
    ) -> None:
        self._hasher = hasher
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_thread: PeriodicThread | None = None
        if cleanup_interval is not None and cleanup_interval > 0:
            self._cleanup_thread = self._create_cleanup_thread(cleanup_interval)
            self._cleanup_thread.start()

    def get_or_create(self, request_id: str) -> Session:
        """Get existing session or create a new one.

        Args:
            request_id: Unique request identifier.

        Returns:
            The Session for this request_id.
        """
        with self._lock:
            if request_id not in self._sessions:
                self._sessions[request_id] = Session(
                    request_id=request_id, hasher=self._hasher
                )
                logger.debug("Created session for request_id=%s", request_id)
            return self._sessions[request_id]

    def get(self, request_id: str) -> Optional[Session]:
        """Return an existing session without creating ownership state."""
        with self._lock:
            return self._sessions.get(request_id)

    def remove(self, request_id: str) -> Optional[Session]:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            if request_id in self._sessions:
                session = self._sessions[request_id]
                del self._sessions[request_id]
                logger.debug("Removed session for request_id=%s", request_id)
                return session
            return None

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired = []
        with self._lock:
            for rid, session in self._sessions.items():
                if now - session.created_at > self._ttl:
                    expired.append(rid)
            for rid in expired:
                del self._sessions[rid]

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def active_count(self) -> int:
        """Return the number of active sessions.

        Returns:
            Number of currently tracked sessions.
        """
        with self._lock:
            return len(self._sessions)

    def close(self) -> None:
        """Stop the background cleanup thread and unregister it."""
        if self._cleanup_thread is None:
            return

        PeriodicThreadRegistry.get_instance().unregister(self._cleanup_thread.name)
        self._cleanup_thread.stop()
        self._cleanup_thread = None

    def _create_cleanup_thread(self, cleanup_interval: float) -> PeriodicThread:
        def execute_cleanup() -> ThreadRunSummary:
            removed = self.cleanup_expired()
            return ThreadRunSummary(
                success=True,
                message=f"Removed {removed} expired sessions",
                extra_info={"removed_sessions": str(removed)},
            )

        return create_periodic_thread(
            name=f"SessionManager-cleanup-thread-{id(self):x}",
            interval=cleanup_interval,
            execute_fn=execute_cleanup,
            level=ThreadLevel.MEDIUM,
        )
