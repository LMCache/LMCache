# SPDX-License-Identifier: Apache-2.0
# Stub for the native_storage_ops C++ extension (implemented in csrc/storage_manager/).

"""Native storage operations for LMCache."""

# Standard
from typing import Any, Set, overload

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

    @overload
    def __init__(self, size: int) -> None:
        """
        Construct a Bitmap with the specified number of bits.

        Args:
            size: The number of bits in the bitmap.
        """
        ...
    @overload
    def __init__(self, size: int, prefix_bits: int) -> None:
        """
        Construct a Bitmap with the specified number of bits and prefix.

        Args:
            size: The number of bits in the bitmap.
            prefix_bits: The first N bits are set to 1.
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

    def __invert__(self) -> Bitmap:
        """Bitwise NOT (flip all bits)."""
        ...

    def __or__(self, other: Bitmap) -> Bitmap:
        """
        Bitwise OR with another bitmap.
        If sizes differ, the result is truncated to the smaller size.
        """
        ...

    def get_indices_list(self) -> list[int]:
        """Return a list of indices where the bit is set to 1, in ascending order."""
        ...

    def get_indices_set(self) -> Set[int]:
        """Return a set of indices where the bit is set to 1."""
        ...

    def gather(self, items: list[Any]) -> list[Any]:
        """
        Return elements from items at indices where the bit is set to 1.

        Args:
            items: A list of objects. Length should match the bitmap size.

        Returns:
            A list of objects from items at positions where the bitmap bit is 1.
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

class RangePatternMatcher:
    """
    Range pattern matcher for integer vectors.

    This class performs range pattern matching on a vector of integers.
    It finds ranges that start with a start pattern and end with an end pattern.
    When multiple end patterns exist after a start pattern, it matches the first
    one (minimal range).
    """

    def __init__(self, start_pattern: list[int], end_pattern: list[int]) -> None:
        """
        Construct a RangePatternMatcher with start and end patterns.

        Args:
            start_pattern: The pattern marking the start of a range.
            end_pattern: The pattern marking the end of a range.

        Raises:
            ValueError: If either pattern is empty or has more than 5 elements.
        """
        ...

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        """
        Match ranges in the given data.

        Finds all ranges that start with the start pattern and end with the end
        pattern. When multiple end patterns exist after a start pattern, matches
        the first one (minimal range).

        Args:
            data: The data to search in.

        Returns:
            A list of (start_pos, end_pos) tuples where:
            - start_pos is the beginning index of the start pattern
            - end_pos is the exclusive index after the end pattern
            Returns an empty list if no ranges are found.

        Example:
            >>> start = [1, 2]
            >>> end = [3, 4]
            >>> matcher = RangePatternMatcher(start, end)
            >>> data = [1, 2, 0, 3, 4, 0, 3, 4, 1, 2, 0, 0, 3, 4]
            >>> matcher.match(data)
            [(0, 5), (8, 14)]
        """
        ...
