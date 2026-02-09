# SPDX-License-Identifier: Apache-2.0
# Stub for the native_storage_ops C++ extension (implemented in csrc/storage_manager/).

"""Native storage operations for LMCache."""

class TTLLock:
    """
    A thread-safe lock with TTL (Time-To-Live) support.

    The lock maintains a counter that can be incremented (lock) and decremented
    (unlock). If the TTL expires, the lock is considered unlocked regardless
    of the counter value.
    """

    def __init__(self, ttl_second: int = 300) -> None:
        """
        Construct a TTLLock with the specified TTL duration in seconds.

        Args:
            ttl_second: TTL duration in seconds. Default is 300.
        """
        ...

    def lock(self) -> None:
        """
        Increment the lock counter by 1 and update the TTL.
        If the previous TTL has expired, reset counter to 1.
        """
        ...

    def unlock(self) -> None:
        """Decrement the lock counter by 1 (minimum 0)."""
        ...

    def is_locked(self) -> bool:
        """
        Check if the lock is held (counter > 0 and TTL not expired).

        Returns:
            True if the lock is held, False otherwise.
        """
        ...

    def reset(self) -> None:
        """Reset the lock to initial state (counter = 0, TTL expired)."""
        ...
