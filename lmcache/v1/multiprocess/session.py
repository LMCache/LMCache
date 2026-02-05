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
    token_ids: list[int] = field(default_factory=list)
    chunk_hashes: list = field(default_factory=list)
    last_prefix_hash: Any = None
    num_chunks_processed: int = 0
    created_at: float = field(default_factory=time.time)

    def set_tokens_and_hash_range(
        self,
        full_token_ids: list[int],
        start: int,
        end: int,
        hasher: TokenHasher,
    ) -> list:
        """Idempotently set tokens, compute hashes up to end, return hashes
        for [start, end).

        - Sets token_ids (idempotent, not append)
        - Computes rolling hashes up to end_chunk using cached prefix state
        - Returns chunk hashes for the [start_chunk, end_chunk) range
        - Safe to call multiple times: already-computed chunks are skipped

        Args:
            full_token_ids: Complete token sequence (replaces, not extends).
            start: Start token index.
            end: End token index.
            hasher: TokenHasher instance for hash computation.

        Returns:
            List of hash values for chunks in [start_chunk, end_chunk).
        """
        self.token_ids = full_token_ids

        chunk_size = hasher.chunk_size
        start_chunk = start // chunk_size
        end_chunk = end // chunk_size

        while self.num_chunks_processed < end_chunk:
            cs = self.num_chunks_processed * chunk_size
            ce = cs + chunk_size
            chunk = self.token_ids[cs:ce]

            prefix = (
                self.last_prefix_hash
                if self.last_prefix_hash is not None
                else hasher.none_hash
            )
            h = hasher.hash_tokens(chunk, prefix)
            self.last_prefix_hash = h
            self.chunk_hashes.append(h)
            self.num_chunks_processed += 1

        return self.chunk_hashes[start_chunk:end_chunk]

    def compute_all_hashes(self, token_ids: list[int], hasher: TokenHasher) -> list:
        """Set full token_ids and compute all chunk hashes from scratch.

        Args:
            token_ids: Complete token sequence.
            hasher: TokenHasher instance for hash computation.

        Returns:
            List of all computed hash values.
        """
        self.token_ids = list(token_ids)
        self.chunk_hashes = []
        self.last_prefix_hash = None
        self.num_chunks_processed = 0

        total_tokens = len(self.token_ids)
        return self.set_tokens_and_hash_range(self.token_ids, 0, total_tokens, hasher)


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
                self._sessions[request_id] = Session(request_id=request_id)
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
