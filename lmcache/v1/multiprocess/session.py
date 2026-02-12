# SPDX-License-Identifier: Apache-2.0
"""
Session and SessionManager for tracking per-request state
in the multiprocess cache server.
"""

# Standard
from dataclasses import dataclass, field
from typing import Any
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.token_hasher import TokenHasher

logger = init_logger(__name__)


@dataclass
class Session:
    """Tracks accumulated token IDs and computed chunk hashes for a request."""

    request_id: str
    hasher: TokenHasher
    token_ids: list[int] = field(default_factory=list)
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)

    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        self.token_ids = full_token_ids

    def get_hashes(self, start: int, end: int) -> list:
        """Compute and return chunk hashes for the [start, end) token range.

        Internally computes rolling hashes up to end_chunk, skipping
        already-computed chunks.

        Args:
            start: Start token index.
            end: End token index.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).
        """
        chunk_size = self.hasher.chunk_size
        assert start % chunk_size == 0, (
            f"start ({start}) must be a multiple of chunk_size ({chunk_size})"
        )
        assert end % chunk_size == 0, (
            f"end ({end}) must be a multiple of chunk_size ({chunk_size})"
        )
        start_chunk = start // chunk_size
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

    def remove(self, request_id: str) -> None:
        """Remove a session by request_id.

        Args:
            request_id: Unique request identifier.
        """
        with self._lock:
            if request_id in self._sessions:
                del self._sessions[request_id]
                logger.debug("Removed session for request_id=%s", request_id)

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
