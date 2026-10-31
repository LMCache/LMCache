# SPDX-License-Identifier: Apache-2.0
"""Idle-TTL behavior for SessionManager."""

# Standard
import time

# First Party
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher


def _manager(ttl: float = 0.2) -> SessionManager:
    """Create a manager with no background cleanup thread.

    Args:
        ttl: Idle timeout in seconds.

    Returns:
        A session manager.
    """
    return SessionManager(
        TokenHasher(chunk_size=4, hash_algorithm="blake3"),
        ttl=ttl,
        cleanup_interval=None,
    )


def test_get_refreshes_idle_ttl() -> None:
    """A session accessed through get() must not be reaped as idle."""
    manager = _manager()
    session = manager.get_or_create("req-1")
    session.last_used_at = time.time() - 10.0

    assert manager.get("req-1") is session
    assert manager.cleanup_expired() == 0


def test_get_or_create_refreshes_idle_ttl() -> None:
    """A session accessed through get_or_create() must not be reaped."""
    manager = _manager()
    session = manager.get_or_create("req-1")
    session.last_used_at = time.time() - 10.0

    assert manager.get_or_create("req-1") is session
    assert manager.cleanup_expired() == 0


def test_idle_session_is_reaped() -> None:
    """A session with no recent access must be reaped."""
    manager = _manager()
    session = manager.get_or_create("req-1")
    session.last_used_at = time.time() - 10.0

    assert manager.cleanup_expired() == 1
