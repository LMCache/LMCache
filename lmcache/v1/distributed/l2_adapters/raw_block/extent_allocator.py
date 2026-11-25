# SPDX-License-Identifier: Apache-2.0
"""Best-fit free-extent allocator for the raw-block data region."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable
import bisect

Extent = tuple[int, int]
"""A device byte range as ``(offset, length)``."""


def _last(bucket: dict[int, None]) -> int:
    """Return the most recently added key of a non-empty bucket in O(1).

    ``next(iter(d))`` skips the deleted entries CPython leaves at the front
    of a dict, so taking-and-deleting the first key repeatedly is O(n) each
    time; ``popitem`` takes from the back and trims deleted entries there.
    """
    start, _ = bucket.popitem()
    bucket[start] = None
    return start


class ExtentAllocator:
    """Allocate aligned byte ranges from one contiguous region.

    Free space is a set of maximal free extents, indexed by start, by end
    (for O(1) coalescing on free) and by length (for best fit); Ceph
    BlueStore's AVL allocator keeps the same two orders in trees. An
    allocation takes the head of the smallest free extent that fits;
    ``allocate_extents`` can instead serve a request no single free extent
    fits from several, largest first. Freed ranges are coalesced with their
    free neighbours immediately.

    Among equal candidates the most recently freed extent is taken, as a
    LIFO free list would.

    A free extent may carry a placement tag (the FDP placement identifier
    that last wrote it). Tags only break ties between free extents of exactly
    the requested length; they never block coalescing.

    Complexity: allocate and free are O(1) dictionary operations plus one
    insertion into or removal from a sorted list of distinct free lengths,
    which is O(k) memmove for k distinct lengths.

    The allocator is not thread-safe; the caller serializes access.
    """

    def __init__(
        self,
        base: int,
        size: int,
        unit: int,
        used: Iterable[Extent] = (),
    ) -> None:
        """Create an allocator over ``[base, base + size)``.

        Args:
            base: First byte of the region; a multiple of ``unit``.
            size: Region length in bytes; a multiple of ``unit``.
            unit: Allocation unit in bytes; every offset and length handed
                out or accepted is a multiple of it.
            used: Extents already in use, which must lie inside the region and
                not overlap each other. The rest of the region starts free.

        Raises:
            ValueError: If the geometry is not unit-aligned or ``used`` is
                out of range, misaligned, or overlapping.
        """
        if unit <= 0 or base % unit or size % unit or size <= 0:
            raise ValueError("base and size must be positive multiples of unit")
        self.base = base
        self.size = size
        self.unit = unit
        self._by_start: dict[int, int] = {}
        self._by_end: dict[int, int] = {}
        self._by_len: dict[int, dict[int, None]] = {}
        self._lens: list[int] = []
        self._tag: dict[int, int] = {}
        self._by_len_tag: dict[tuple[int, int], dict[int, None]] = {}
        self._free_bytes = 0

        cursor = base
        for offset, length in sorted(used):
            self._check_extent(offset, length)
            if offset < cursor:
                raise ValueError(f"used extent at {offset} overlaps another")
            if offset > cursor:
                self._insert(cursor, offset - cursor, None)
            cursor = offset + length
        if cursor < base + size:
            self._insert(cursor, base + size - cursor, None)

    @property
    def free_bytes(self) -> int:
        """Total free bytes."""
        return self._free_bytes

    @property
    def free_extent_count(self) -> int:
        """Number of maximal free extents."""
        return len(self._by_start)

    @property
    def largest_free_extent(self) -> int:
        """Length of the largest free extent, or 0 when nothing is free."""
        return self._lens[-1] if self._lens else 0

    def has_tagged_extent(self, nbytes: int, tag: int) -> bool:
        """Return whether a free extent of exactly ``nbytes`` carries ``tag``."""
        return bool(self._by_len_tag.get((nbytes, tag)))

    def allocate(self, nbytes: int, *, tag: int | None = None) -> int | None:
        """Allocate ``nbytes`` contiguous bytes, best fit.

        Args:
            nbytes: Bytes to allocate; a positive multiple of ``unit``.
            tag: Placement tag preferred among free extents of exactly
                ``nbytes``.

        Returns:
            The offset of the allocated range, or None when no free extent is
            long enough. Nothing is allocated on None.

        Raises:
            ValueError: If ``nbytes`` is not a positive multiple of ``unit``.
        """
        if nbytes <= 0 or nbytes % self.unit:
            raise ValueError("nbytes must be a positive multiple of unit")
        if tag is not None:
            tagged = self._by_len_tag.get((nbytes, tag))
            if tagged:
                return self._take(_last(tagged), nbytes)
        i = bisect.bisect_left(self._lens, nbytes)
        if i == len(self._lens):
            return None
        return self._take(_last(self._by_len[self._lens[i]]), nbytes)

    def allocate_extents(
        self, nbytes: int, max_extents: int, *, min_first: int = 0
    ) -> list[Extent] | None:
        """Allocate ``nbytes`` as several extents when no single one fits.

        Whole free extents are taken largest first until the remainder has a
        best fit, so the first extent is the longest.

        Args:
            nbytes: Bytes to allocate; a positive multiple of ``unit``.
            max_extents: Most extents the result may have; at least 2.
            min_first: Least length of the first extent (e.g. an object
                header that must not be split).

        Returns:
            Extents whose lengths sum to ``nbytes``, longest first, or None
            when the ``max_extents`` longest free extents cannot hold it or
            the longest is shorter than ``min_first``. Nothing is allocated on
            None.

        Raises:
            ValueError: If ``nbytes`` is not a positive multiple of ``unit``
                or ``max_extents`` is below 2.
        """
        if nbytes <= 0 or nbytes % self.unit:
            raise ValueError("nbytes must be a positive multiple of unit")
        if max_extents < 2:
            raise ValueError("max_extents must be >= 2")
        if (
            not self._lens
            or self._lens[-1] < min_first
            or not self._fits_in(nbytes, max_extents)
        ):
            return None
        extents: list[Extent] = []
        remaining = nbytes
        while remaining:
            i = bisect.bisect_left(self._lens, remaining)
            length = self._lens[i] if i < len(self._lens) else self._lens[-1]
            take = min(length, remaining)
            extents.append((self._take(_last(self._by_len[length]), take), take))
            remaining -= take
        return extents

    def free(self, offset: int, length: int, *, tag: int | None = None) -> None:
        """Return a range to free space, coalescing with free neighbours.

        Args:
            offset: Start of a range previously handed out and not freed since.
            length: Its length.
            tag: Placement tag recorded on the freed space.

        Raises:
            ValueError: If the range is misaligned, outside the region, or
                starts or ends where free space does (a double free of that
                range). A range that overlaps free space without starting or
                ending where a free extent does (inside it, or across one of
                its ends) is not detected.
        """
        self._check_extent(offset, length)
        start, end = offset, offset + length
        if start in self._by_start or end in self._by_end:
            raise ValueError(f"range ({offset}, {length}) is already free")
        prev_start = self._by_end.get(start)
        if prev_start is not None:
            prev_tag = self._tag.get(prev_start)
            self._remove(prev_start)
            tag_merged = tag if prev_tag == tag else None
            start = prev_start
        else:
            tag_merged = tag
        next_len = self._by_start.get(end)
        if next_len is not None:
            if self._tag.get(end) != tag_merged:
                tag_merged = None
            self._remove(end)
            end += next_len
        self._insert(start, end - start, tag_merged)

    def remove_free_range(self, offset: int, length: int) -> int:
        """Withdraw any free bytes inside a range from future allocation.

        Used to retire a damaged device range. Allocated bytes in the range
        are left alone; the caller must not free them back later.

        Args:
            offset: First byte of the range; unit-aligned.
            length: Range length; a positive multiple of ``unit``.

        Returns:
            Number of free bytes withdrawn.
        """
        self._check_extent(offset, length)
        end = offset + length
        withdrawn = 0
        # Retirement is rare, so a linear scan over free extents is fine.
        for start, free_len in list(self._by_start.items()):
            free_end = start + free_len
            lo, hi = max(start, offset), min(free_end, end)
            if lo >= hi:
                continue
            tag = self._tag.get(start)
            self._remove(start)
            if start < lo:
                self._insert(start, lo - start, tag)
            if hi < free_end:
                self._insert(hi, free_end - hi, tag)
            withdrawn += hi - lo
        return withdrawn

    def free_extents(self) -> list[Extent]:
        """Return all free extents sorted by offset (for tests and status)."""
        return sorted(self._by_start.items())

    def _fits_in(self, nbytes: int, max_extents: int) -> bool:
        """Return whether the ``max_extents`` longest free extents cover nbytes."""
        total = 0
        count = 0
        for length in reversed(self._lens):
            n = min(len(self._by_len[length]), max_extents - count)
            total += n * length
            count += n
            if total >= nbytes:
                return True
            if count == max_extents:
                return False
        return False

    def _take(self, start: int, nbytes: int) -> int:
        """Allocate the head of the free extent at ``start``."""
        length = self._by_start[start]
        tag = self._tag.get(start)
        self._remove(start)
        if length > nbytes:
            self._insert(start + nbytes, length - nbytes, tag)
        return start

    def _insert(self, start: int, length: int, tag: int | None) -> None:
        self._by_start[start] = length
        self._by_end[start + length] = start
        bucket = self._by_len.get(length)
        if bucket is None:
            bucket = self._by_len[length] = {}
            bisect.insort(self._lens, length)
        bucket[start] = None
        if tag is not None:
            self._tag[start] = tag
            self._by_len_tag.setdefault((length, tag), {})[start] = None
        self._free_bytes += length

    def _remove(self, start: int) -> None:
        length = self._by_start.pop(start)
        del self._by_end[start + length]
        bucket = self._by_len[length]
        del bucket[start]
        if not bucket:
            del self._by_len[length]
            del self._lens[bisect.bisect_left(self._lens, length)]
        tag = self._tag.pop(start, None)
        if tag is not None:
            tagged = self._by_len_tag[(length, tag)]
            del tagged[start]
            if not tagged:
                del self._by_len_tag[(length, tag)]
        self._free_bytes -= length

    def _check_extent(self, offset: int, length: int) -> None:
        if (
            length <= 0
            or offset % self.unit
            or length % self.unit
            or offset < self.base
            or offset + length > self.base + self.size
        ):
            raise ValueError(f"invalid extent ({offset}, {length})")
