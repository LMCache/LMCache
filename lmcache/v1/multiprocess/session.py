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
from lmcache.v1.common.custom_types import IPCCacheServerKey
from lmcache.v1.common.token_codec import TOKEN_STRIDE, num_packed_tokens
from lmcache.v1.common.token_hasher import TokenHasher
from lmcache.v1.distributed.api import ObjectKey
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
    :mod:`lmcache.v1.common.token_codec`, the layout that arrives on
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
    _prefetch_owners: dict[ObjectKey, int] = field(default_factory=dict, repr=False)
    _streaming_ranges: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _streaming_owners: list[dict[ObjectKey, int]] = field(
        default_factory=list, repr=False
    )
    _current_streaming_range: tuple[int, int] | None = field(default=None, repr=False)
    _current_streaming_finished: bool = field(default=False, repr=False)
    extras: dict[str, Any] = field(default_factory=dict)
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

    def get_unfull_chunk_hash(self, boundary: int) -> Any:
        """Return the hash of the unfull chunk with boundary ``boundary``.

        A boundary is a token count: the unfull chunk covers the tokens from
        the last chunk boundary at or below it up to it,
        ``[floor(boundary / chunk_size) * chunk_size, boundary)``. It is
        hashed like a full chunk, chaining from the rolling hash of the full
        chunks before it. It has fewer than chunk_size tokens, so its hash
        differs from every full chunk's.

        Args:
            boundary: Token count the unfull chunk reaches; must not be a
                multiple of chunk_size.

        Returns:
            The unfull chunk's hash value.

        Raises:
            ValueError: If ``boundary`` is chunk-aligned or exceeds the current
                tokens.
        """
        chunk_size = self.hasher.chunk_size
        if boundary % chunk_size == 0:
            raise ValueError(
                f"unfull chunk boundary ({boundary}) must not be a multiple of "
                f"chunk_size ({chunk_size})"
            )
        with self._lock:
            num_tokens = num_packed_tokens(self.tokens)
            if boundary > num_tokens:
                raise ValueError(
                    f"unfull chunk boundary ({boundary}) exceeds the session's "
                    f"{num_tokens} token(s)"
                )
            num_full_chunks = boundary // chunk_size
            self._compute_hash(num_full_chunks)
            prefix = (
                self.chunk_hashes[num_full_chunks - 1]
                if num_full_chunks > 0
                else self.hasher.none_hash
            )
            return self.hasher.hash_packed_chunk(
                memoryview(self.tokens)[
                    num_full_chunks * chunk_size * TOKEN_STRIDE : boundary
                    * TOKEN_STRIDE
                ],
                prefix,
            )

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

    def begin_lookup(self, key: IPCCacheServerKey) -> None:
        """Record a new lookup and reset the previous lookup's result."""
        with self._lock:
            if (
                self._current_streaming_range is not None
                and self._current_streaming_finished
            ):
                self._streaming_ranges.pop()
                self._streaming_owners.pop()
            self.lookup_ipc_key = key
            self.prefetch_hit_chunks = -1
            self.prefetch_locked_gids = ()
            self._prefetch_owners.clear()
            self._current_streaming_range = None
            self._current_streaming_finished = False

    def record_prefetch_result(
        self,
        hit_chunks: int,
        locked_gids: tuple[int, ...],
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        """Record the lock set acquired by the current lookup."""
        with self._lock:
            self.prefetch_hit_chunks = hit_chunks
            self.prefetch_locked_gids = locked_gids
            self._prefetch_owners = dict(l1_owners) if l1_owners is not None else {}

    def get_prefetch_owners(self) -> dict[ObjectKey, int]:
        """Snapshot the L1 owners whose read locks this session's lookup retained."""
        with self._lock:
            return self._prefetch_owners.copy()

    def add_streaming_range(
        self, start: int, end: int, l1_owners: dict[ObjectKey, int] | None = None
    ) -> None:
        """Record a mixed range and its L1 read-lock owners."""
        with self._lock:
            self._streaming_ranges.append((start, end))
            self._streaming_owners.append(dict(l1_owners or {}))
            self._current_streaming_range = (start, end)
            self._current_streaming_finished = False

    def finish_current_streaming_range(self, start: int, end: int) -> None:
        """Mark the current streaming lookup finished when its full range is done."""
        with self._lock:
            current = self._current_streaming_range
            if current is not None and start <= current[0] and end >= current[1]:
                self._current_streaming_finished = True

    def finish_current_streaming_retrieve(self) -> None:
        """Mark the current lookup scheduled for retrieval and lock release."""
        with self._lock:
            if self._current_streaming_range is not None:
                self._current_streaming_finished = True

    def abandoned_streaming_range_owners(
        self, start: int, end: int
    ) -> dict[ObjectKey, int] | None:
        """Return owners of the oldest matching abandoned streaming lookup."""
        with self._lock:
            index = self._abandoned_streaming_range_index(start, end)
            return None if index is None else dict(self._streaming_owners[index])

    def finish_abandoned_streaming_range(self, start: int, end: int) -> None:
        """Forget the oldest abandoned lookup after its entire range is freed."""
        with self._lock:
            index = self._abandoned_streaming_range_index(start, end)
            if index is not None and self._streaming_ranges[index] == (start, end):
                self._streaming_ranges.pop(index)
                self._streaming_owners.pop(index)

    def _abandoned_streaming_range_index(self, start: int, end: int) -> int | None:
        """Find a prior range while the session lock is held."""
        limit = len(self._streaming_ranges) - int(
            self._current_streaming_range is not None
        )
        for exact in (True, False):
            for index in range(limit):
                lo, hi = self._streaming_ranges[index]
                if (lo == start and hi == end) or (
                    not exact and lo <= start and end <= hi
                ):
                    return index
        return None

    def streaming_range_owners(
        self, start: int, end: int
    ) -> dict[ObjectKey, int] | None:
        """Return L1 owners for a streaming range, or None for a regular lookup."""
        with self._lock:
            ranges = list(
                zip(self._streaming_ranges, self._streaming_owners, strict=True)
            )
            for (lo, hi), owners in reversed(ranges):
                if lo == start and hi == end:
                    return dict(owners)
            for (lo, hi), owners in reversed(ranges):
                if lo <= start and end <= hi:
                    return dict(owners)
            return None

    def extend_current_streaming_range(
        self, end: int, l1_owners: dict[ObjectKey, int]
    ) -> None:
        """Include a read-locked partial L1 chunk in the current mixed range."""
        with self._lock:
            if self._current_streaming_range is None:
                raise ValueError("no current streaming range")
            start, previous_end = self._current_streaming_range
            if end <= previous_end:
                raise ValueError("partial chunk must extend the streaming range")
            self._streaming_ranges[-1] = (start, end)
            self._streaming_owners[-1].update(l1_owners)
            self._current_streaming_range = (start, end)

    def is_streaming_range(self, start: int, end: int) -> bool:
        """Return whether a range belongs to an FS streaming lookup."""
        with self._lock:
            return any(lo <= start and end <= hi for lo, hi in self._streaming_ranges)

    def is_current_streaming_range(self, start: int, end: int) -> bool:
        """Return whether the latest lookup serves this range from FS windows."""
        with self._lock:
            current = self._current_streaming_range
            return current is not None and current[0] <= start and end <= current[1]


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
