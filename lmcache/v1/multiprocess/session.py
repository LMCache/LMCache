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
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.token_hasher import TokenHasher

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
    total_tokens: int = 0
    _retrieved_start: int = 0
    _retrieved_end: int = 0
    lookup_time: float = 0.0
    retrieve_time: float = 0.0
    store_time: float = 0.0
    lookup_chunks: int = 0
    store_chunks: int = 0
    lookup_ipc_key: Optional[IPCCacheEngineKey] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def retrieved_tokens(self) -> int:
        """Number of retrieved tokens, derived from the range."""
        return self._retrieved_end - self._retrieved_start

    @property
    def retrieve_chunks(self) -> int:
        """Number of retrieved chunks, derived from the range."""
        chunk_size = self.hasher.chunk_size
        return (self._retrieved_end - self._retrieved_start) // chunk_size

    def update_retrieved_range(self, start: int, end: int) -> None:
        """Update the retrieved token range (idempotent union).

        Multiple TP workers may call this with the same range;
        the result is always the union of all reported ranges.

        Args:
            start: Start token index of the retrieved range.
            end: End token index of the retrieved range.
        """
        with self._lock:
            if self._retrieved_start == self._retrieved_end:
                # First call: initialize the range
                self._retrieved_start = start
                self._retrieved_end = end
            else:
                self._retrieved_start = min(self._retrieved_start, start)
                self._retrieved_end = max(self._retrieved_end, end)

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
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size

        with self._lock:
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


class SessionManager:
    """Thread-safe manager for per-request sessions."""

    DEFAULT_SESSION_TTL = 600  # 10 minutes

    def __init__(self, hasher: TokenHasher, ttl: float = DEFAULT_SESSION_TTL):
        self._hasher = hasher
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

        # Cumulative stats accumulated when sessions end
        self._stats_lock = threading.Lock()
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._total_retrieved_tokens: int = 0

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

    def remove(self, request_id: str, reason: str = "normal") -> Optional[Session]:
        """Remove a session and accumulate its stats.

        Args:
            request_id: Unique request identifier.
            reason: Why the session is being removed, e.g. ``"normal"``
                for an explicit end_session from vLLM or ``"expired"``
                for TTL-based cleanup. Included in the END_SESSION log.

        Returns:
            The removed session, or None if no session was found.
        """
        with self._lock:
            session = self._sessions.pop(request_id, None)
        if session is not None:
            with self._stats_lock:
                self._total_requests += 1
                self._total_tokens += session.total_tokens
                self._total_retrieved_tokens += session.retrieved_tokens
            hit_rate = (
                session.retrieved_tokens / session.total_tokens
                if session.total_tokens > 0
                else 0.0
            )
            logger.info(
                "END_SESSION[%s] reason=%s: total_tokens=%d, "
                "retrieved_tokens=%d, hit_rate=%.2f%%, "
                "lookup=%d chunks/%.3fs, "
                "retrieve=%d chunks/%.3fs, "
                "store=%d chunks/%.3fs",
                request_id,
                reason,
                session.total_tokens,
                session.retrieved_tokens,
                hit_rate * 100,
                session.lookup_chunks,
                session.lookup_time,
                session.retrieve_chunks,
                session.retrieve_time,
                session.store_chunks,
                session.store_time,
            )
        return session

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded their TTL.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        with self._lock:
            expired = [
                rid
                for rid, s in self._sessions.items()
                if now - s.created_at > self._ttl
            ]

        for rid in expired:
            self.remove(rid, reason="expired")

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

    def report_hit_stats(self) -> dict[str, int | float]:
        """Return cumulative hit statistics.

        Returns:
            Dict with total_requests, total_tokens,
            total_retrieved_tokens, and hit_rate.
        """
        with self._stats_lock:
            total_req = self._total_requests
            total_tok = self._total_tokens
            retrieved_tok = self._total_retrieved_tokens
        hit_rate = round(retrieved_tok / total_tok, 4) if total_tok > 0 else 0.0
        return {
            "total_requests": total_req,
            "total_tokens": total_tok,
            "total_retrieved_tokens": retrieved_tok,
            "hit_rate": hit_rate,
        }
