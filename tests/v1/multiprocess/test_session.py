# SPDX-License-Identifier: Apache-2.0
"""Tests for Session and SessionManager."""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.session import Session, SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@pytest.fixture
def hasher() -> TokenHasher:
    """TokenHasher with small chunk_size for testing."""
    return TokenHasher(chunk_size=4, hash_algorithm="blake3")


@pytest.fixture
def session(hasher: TokenHasher) -> Session:
    """A fresh Session instance."""
    return Session(request_id="req-1", hasher=hasher)


@pytest.fixture
def session_manager(hasher: TokenHasher) -> SessionManager:
    """A SessionManager with short TTL for testing expiry."""
    return SessionManager(hasher, ttl=0.1)


class TestSession:
    def test_set_tokens_replaces(self, session: Session) -> None:
        session.set_tokens([1, 2, 3])
        assert session.token_ids == [1, 2, 3]
        session.set_tokens([4, 5, 6])
        assert session.token_ids == [4, 5, 6]

    def test_get_hashes_basic(self, session: Session) -> None:
        """8 tokens, chunk_size=4 produces 2 hashes."""
        session.set_tokens(list(range(8)))
        hashes = session.get_hashes(0, 8)
        assert len(hashes) == 2

    def test_get_hashes_incremental(self, session: Session) -> None:
        """Calling get_hashes incrementally should produce same results."""
        tokens = list(range(12))
        session.set_tokens(tokens)

        # First call: compute chunks 0-1
        h_first = session.get_hashes(0, 8)
        assert len(h_first) == 2
        assert session.num_chunks_processed == 2

        # Second call: compute chunk 2 (chunks 0-1 already cached)
        h_second = session.get_hashes(8, 12)
        assert len(h_second) == 1
        assert session.num_chunks_processed == 3

        # Full range should match
        h_all = session.get_hashes(0, 12)
        assert h_all == h_first + h_second

    def test_get_hashes_idempotent(self, session: Session) -> None:
        """Calling get_hashes twice with same range returns same result."""
        session.set_tokens(list(range(8)))
        h1 = session.get_hashes(0, 8)
        h2 = session.get_hashes(0, 8)
        assert h1 == h2
        assert session.num_chunks_processed == 2  # not recomputed

    def test_get_hashes_matches_hasher(
        self, hasher: TokenHasher, session: Session
    ) -> None:
        """Session hashes should match standalone TokenHasher hashes."""
        tokens = list(range(8))
        session.set_tokens(tokens)
        session_hashes = session.get_hashes(0, 8)
        hasher_hashes = hasher.compute_chunk_hashes(tokens)
        assert session_hashes == hasher_hashes

    def test_rolling_state_across_calls(self, session: Session) -> None:
        """Simulates the lookup then store flow: lookup hashes all, store
        hashes a subrange. The rolling state from lookup should be reused."""
        tokens = list(range(12))
        session.set_tokens(tokens)

        # Lookup: hash everything
        all_hashes = session.get_hashes(0, 12)
        assert len(all_hashes) == 3

        # Store: request hashes for chunk 1-2 (tokens 4:12)
        # Should reuse already-computed hashes
        store_hashes = session.get_hashes(4, 12)
        assert store_hashes == all_hashes[1:]
        assert session.num_chunks_processed == 3  # no extra computation


class TestSessionManager:
    def test_get_or_create_new(self, session_manager: SessionManager) -> None:
        s = session_manager.get_or_create("req-1")
        assert s.request_id == "req-1"

    def test_get_or_create_existing(self, session_manager: SessionManager) -> None:
        s1 = session_manager.get_or_create("req-1")
        s2 = session_manager.get_or_create("req-1")
        assert s1 is s2

    def test_get_or_create_different_ids(self, session_manager: SessionManager) -> None:
        s1 = session_manager.get_or_create("req-1")
        s2 = session_manager.get_or_create("req-2")
        assert s1 is not s2

    def test_remove(self, session_manager: SessionManager) -> None:
        session_manager.get_or_create("req-1")
        session_manager.remove("req-1")
        # Should create a fresh session
        s = session_manager.get_or_create("req-1")
        assert s.num_chunks_processed == 0

    def test_remove_nonexistent(self, session_manager: SessionManager) -> None:
        """Removing a non-existent session should not raise."""
        session_manager.remove("does-not-exist")

    def test_cleanup_expired(self, session_manager: SessionManager) -> None:
        """Sessions older than TTL should be cleaned up."""
        session_manager.get_or_create("req-1")
        session_manager.get_or_create("req-2")
        # Wait for TTL to expire (ttl=0.1s)
        time.sleep(0.15)
        removed = session_manager.cleanup_expired()
        assert removed == 2
        # New session should be fresh
        s = session_manager.get_or_create("req-1")
        assert s.num_chunks_processed == 0

    def test_cleanup_keeps_fresh(self, session_manager: SessionManager) -> None:
        """Fresh sessions should not be cleaned up."""
        session_manager.get_or_create("req-1")
        removed = session_manager.cleanup_expired()
        assert removed == 0


class TestSessionThreadSafety:
    """Verify Session is safe under concurrent access
    from multiple TP worker threads."""

    def test_concurrent_get_hashes(self, hasher: TokenHasher) -> None:
        """Multiple threads calling get_hashes on the same Session
        must produce correct hashes (no duplicates, no corruption).
        """
        session = Session(request_id="req-mt", hasher=hasher)
        tokens = list(range(20))  # 5 chunks of 4
        session.set_tokens(tokens)

        # Reference hashes computed single-threaded
        expected = hasher.compute_chunk_hashes(tokens)
        errors: list[str] = []
        barrier = threading.Barrier(8)

        def worker(tid: int) -> None:
            try:
                barrier.wait(timeout=5)
                hashes = session.get_hashes(0, 20)
                if hashes != expected:
                    errors.append(
                        "Thread %d: got %r, expected %r" % (tid, hashes, expected)
                    )
            except Exception as exc:
                errors.append("Thread %d: %s" % (tid, exc))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, "\n".join(errors)
        assert session.num_chunks_processed == 5

    def test_concurrent_set_and_get(self, hasher: TokenHasher) -> None:
        """set_tokens and get_hashes called concurrently must not
        corrupt internal state."""
        session = Session(request_id="req-mt2", hasher=hasher)
        tokens = list(range(8))  # 2 chunks
        session.set_tokens(tokens)
        expected = hasher.compute_chunk_hashes(tokens)
        errors: list[str] = []
        barrier = threading.Barrier(4)

        def reader(tid: int) -> None:
            try:
                barrier.wait(timeout=5)
                hashes = session.get_hashes(0, 8)
                if hashes != expected:
                    errors.append("Reader %d: mismatch" % tid)
            except Exception as exc:
                errors.append("Reader %d: %s" % (tid, exc))

        def writer() -> None:
            try:
                barrier.wait(timeout=5)
                # Re-set same tokens (idempotent)
                session.set_tokens(tokens)
            except Exception as exc:
                errors.append("Writer: %s" % exc)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, "\n".join(errors)
