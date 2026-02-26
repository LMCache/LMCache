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

class Bitmap:
    """
    A bitmap for tracking the state of L2 storage operation results.

    Each bit represents the success or failure of a key.
    """

    def __init__(self, size: int) -> None:
        """
        Construct a Bitmap with the specified number of bits.

        Args:
            size: The number of bits in the bitmap.
        """
        ...

    def set(self, index: int) -> None:
        """Set the bit at the specified index to 1."""
        ...

    def clear(self, index: int) -> None:
        """Clear the bit at the specified index to 0."""
        ...

    def test(self, index: int) -> bool:
        """
        Test the bit at the specified index.

        Returns:
            True if the bit is set to 1, False otherwise.
        """
        ...

    def popcount(self) -> int:
        """Return the number of bits set to 1."""
        ...

    def count_leading_zeros(self) -> int:
        """Return the number of leading zeros."""
        ...

    def count_leading_ones(self) -> int:
        """Return the number of leading ones."""
        ...

    def __and__(self, other: Bitmap) -> Bitmap:
        """
        Bitwise AND with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        ...

    def __repr__(self) -> str:
        """String representation: '1' for set bits, '0' for clear bits."""
        ...

class ParallelPatternMatcher:
    """
    Pattern matcher for integer vectors.

    This class performs pattern matching on a vector of integers.
    It finds all positions where a given pattern occurs in the input data.
    """

    def __init__(self, pattern: list[int]) -> None:
        """
        Construct a ParallelPatternMatcher with the specified pattern.

        Args:
            pattern: The pattern to search for. Must not be empty.

        Raises:
            ValueError: If pattern is empty.
        """
        ...

    def match(self, data: list[int]) -> list[int]:
        """
        Match the pattern in the given data.

        Args:
            data: The data to search in.

        Returns:
            A sorted list of positions where the pattern starts.
            Returns an empty list if no matches are found.
        """
        ...
