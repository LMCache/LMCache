# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, overload
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1ReadReservation
from lmcache.v1.distributed.l1_manager import validate_read_locks
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.token_codec import TOKEN_STRIDE, num_packed_tokens
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

    Tokens are held in the packed form of
    :mod:`lmcache.v1.multiprocess.token_codec`, the layout that arrives on
    the wire and that blake3 hashes; only callers needing ints unpack.

    Thread-safe: all public methods are protected by an internal lock
    to allow concurrent access from multiple TP worker threads.
    """

    request_id: str
    hasher: TokenHasher
    tokens: bytes = b""
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)
    lookup_ipc_key: Optional[IPCCacheServerKey] = None
    prefetch_hit_chunks: int = -1
    prefetch_locked_gids: tuple = ()
    prefetch_read_generations: dict[ObjectKey, int] = field(default_factory=dict)
    prefetch_group_windows: tuple[int, ...] = ()
    _prefetch_owners: dict[ObjectKey, int] = field(default_factory=dict, repr=False)
    extras: dict[str, Any] = field(default_factory=dict)
    _lookup_generation: int = field(default=0, repr=False)
    _unclaimed_reads: dict[ObjectKey, int] = field(default_factory=dict, repr=False)
    _claimed_worker_reads: set[tuple[int, int, ObjectKey]] = field(
        default_factory=set, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_tokens(self, packed_token_ids: bytes) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            packed_token_ids: Complete packed token sequence, starting at
                position 0.
        """
        with self._lock:
            self.tokens = packed_token_ids

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
            held = num_packed_tokens(self.tokens)
            if end is not None and end > held:
                raise ValueError(
                    f"get_hashes end ({end}) exceeds the session's "
                    f"{held} token(s); the session may have "
                    "been recreated after request cleanup"
                )
            if end is None:
                # No explicit end: use the last full-chunk boundary.
                # Lock must be held here because `self.tokens` may be
                # concurrently replaced by `set_tokens` from another thread.
                end = held - (held % chunk_size)
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
        stride = self.hasher.chunk_size * TOKEN_STRIDE
        # Slicing a memoryview does not copy; slicing the bytes would.
        view = memoryview(self.tokens)

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * stride
            chunk = view[cs : cs + stride]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else self.hasher.none_hash
            )
            h = self.hasher.hash_packed_chunk(chunk, prefix)
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
            self.prefetch_read_generations = {}
            self.prefetch_group_windows = group_windows
            self._lookup_generation += 1
            self._unclaimed_reads.clear()
            self._claimed_worker_reads.clear()
            self._prefetch_owners.clear()

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
        l1_owners: dict[ObjectKey, int] | None = None,
        *,
        read_generations: dict[ObjectKey, int] | None = None,
    ) -> None:
        """Record the lookup's retained L1 owners and acquired lock generations."""
        with self._lock:
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids
            self._prefetch_owners = dict(l1_owners) if l1_owners is not None else {}
            self.prefetch_read_generations = read_generations or {}
            count = (
                validate_read_locks(self.lookup_ipc_key.require_num_kv_readers())
                if self.lookup_ipc_key is not None
                else 0
            )
            self._unclaimed_reads = {
                key: count for key in self.prefetch_read_generations
            }

    def get_read_generations(self, keys: list[ObjectKey]) -> dict[ObjectKey, int]:
        """Snapshot the lookup's original lock generations for cleanup."""
        with self._lock:
            return {
                k: self.prefetch_read_generations[k]
                for k in keys
                if k in self.prefetch_read_generations
            }

    def get_prefetch_owners(self) -> dict[ObjectKey, int]:
        """Snapshot the L1 owners whose read locks this session's lookup retained."""
        with self._lock:
            return self._prefetch_owners.copy()

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
                and key.token_bytes == lookup_key.token_bytes
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

    def claim_worker_read(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        keys: list[ObjectKey],
        *,
        lookup_generation: int | None = None,
    ) -> tuple[L1ReadReservation, ...] | None:
        """Transfer one share per key to a worker, excluding subsequent FREE.

        MLA TP instances share a KV worker ID, so the existing protocol's
        instance ID distinguishes their shares. A repeated/overlapping call
        from one instance cannot claim twice; disjoint ranges remain valid.

        Args:
            key: Worker request matching this lookup's model and token identity.
            instance_id: Worker incarnation carried by the RETRIEVE protocol.
            keys: Exact retained objects this worker will consume or abandon.
            lookup_generation: Optional epoch captured before key resolution.

        Returns:
            Immutable original reservations, or None if any share is unavailable.
            The owner must settle every returned share, including failed pins.
        """
        with self._lock:
            if (
                not self._matches_lookup(key)
                or key.worker_id is None
                or not 0 <= key.worker_id < key.world_size
                or (
                    lookup_generation is not None
                    and lookup_generation != self._lookup_generation
                )
            ):
                return None
            unique_keys = list(dict.fromkeys(keys))
            if any(
                self._unclaimed_reads.get(obj_key, 0) == 0
                or (instance_id, key.worker_id, obj_key) in self._claimed_worker_reads
                for obj_key in unique_keys
            ):
                return None
            claimed = tuple(
                L1ReadReservation(
                    obj_key,
                    self.prefetch_read_generations[obj_key],
                    self._prefetch_owners.get(obj_key),
                    1,
                )
                for obj_key in unique_keys
            )
            for obj_key in unique_keys:
                self._unclaimed_reads[obj_key] -= 1
                self._claimed_worker_reads.add((instance_id, key.worker_id, obj_key))
            return claimed

    def claim_free_reads(
        self, key: IPCCacheServerKey, keys: list[ObjectKey]
    ) -> tuple[L1ReadReservation, ...]:
        """Take only lookup shares not already transferred to a worker.

        Args:
            key: Cancellation request with this lookup's model/token identity.
            keys: Resolved objects in the canceled ranges, possibly overlapping.

        Returns:
            Original per-key shares to release exactly once. A repeated FREE
            returns no shares. Already claimed transfers retain their ownership.
        """
        with self._lock:
            if not self._matches_lookup(key):
                return ()
            claims = []
            for obj_key in dict.fromkeys(keys):
                count = self._unclaimed_reads.pop(obj_key, 0)
                if count:
                    claims.append(
                        L1ReadReservation(
                            obj_key,
                            self.prefetch_read_generations[obj_key],
                            self._prefetch_owners.get(obj_key),
                            count,
                        )
                    )
            return tuple(claims)

    def _matches_lookup(self, key: IPCCacheServerKey) -> bool:
        lookup = self.lookup_ipc_key
        return (
            lookup is not None
            and self.prefetch_hit_chunks >= 0
            and key.request_id == self.request_id
            and key.model_name == lookup.model_name
            and key.world_size == lookup.world_size
            and key.token_bytes == lookup.token_bytes
            and key.cache_salt == lookup.cache_salt
        )


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
        #: Callbacks invoked for every destroyed session, so owners of
        #: per-request state (e.g. entries in ``Session.extras``) can release
        #: it however the request ended. See :meth:`_notify_destroyed`.
        self.destroy_listeners: list[Callable[[Session], None]] = []
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
            if request_id not in self._sessions:
                return None
            session = self._sessions[request_id]
            del self._sessions[request_id]
            logger.debug("Removed session for request_id=%s", request_id)
        self._notify_destroyed(session)
        return session

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired: list[Session] = []
        with self._lock:
            for session in self._sessions.values():
                if now - session.created_at > self._ttl:
                    expired.append(session)
            for session in expired:
                del self._sessions[session.request_id]

        for session in expired:
            self._notify_destroyed(session)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    def _notify_destroyed(self, session: Session) -> None:
        """Run every destroy listener for one removed session.

        Called without the manager lock held: a listener may take other locks
        (e.g. the storage manager's). A listener that raises is logged and
        skipped so destruction always completes.

        Args:
            session: The session that was just removed.
        """
        for listener in list(self.destroy_listeners):
            try:
                listener(session)
            except Exception:
                logger.exception(
                    "Session destroy listener failed for request_id=%s",
                    session.request_id,
                )

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
