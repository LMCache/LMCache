# SPDX-License-Identifier: Apache-2.0

# Standard
import random

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block.extent_allocator import (
    ExtentAllocator,
)

UNIT = 4096


def _free_runs(occupied: list[bool]) -> list[tuple[int, int]]:
    """Return maximal free runs of a unit occupancy map as (start_unit, units)."""
    runs = []
    start = None
    for i, used in enumerate(occupied + [True]):
        if not used and start is None:
            start = i
        elif used and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def _check_against_model(
    alloc: ExtentAllocator, occupied: list[bool], base: int
) -> None:
    """Free space must equal the model exactly, as maximal (coalesced) extents."""
    expected = [(base + s * UNIT, n * UNIT) for s, n in _free_runs(occupied)]
    assert alloc.free_extents() == expected
    assert alloc.free_bytes == sum(n for _, n in expected)
    assert alloc.free_extent_count == len(expected)
    assert alloc.largest_free_extent == max((n for _, n in expected), default=0)


def _mark(occupied: list[bool], base: int, extents, value: bool) -> None:
    for offset, length in extents:
        for u in range((offset - base) // UNIT, (offset + length - base) // UNIT):
            assert occupied[u] != value, "double allocation or double free"
            occupied[u] = value


@pytest.mark.parametrize("seed", range(20))
def test_random_alloc_free_matches_reference_model(seed: int):
    rng = random.Random(seed)
    base = 7 * UNIT
    units = 600
    alloc = ExtentAllocator(base, units * UNIT, UNIT)
    occupied = [False] * units
    live: list[tuple[int, int]] = []
    sizes = [rng.randint(1, 40) for _ in range(4)]

    for _ in range(3000):
        if live and (rng.random() < 0.45 or alloc.free_bytes < UNIT):
            offset, length = live.pop(rng.randrange(len(live)))
            alloc.free(offset, length)
            _mark(occupied, base, [(offset, length)], False)
        else:
            want = rng.choice(sizes) * UNIT
            fitting = [
                (n * UNIT, s) for s, n in _free_runs(occupied) if n * UNIT >= want
            ]
            got = alloc.allocate(want)
            if got is None:
                assert not fitting
            else:
                # Best fit: the head of a smallest fitting run.
                smallest = min(n for n, _ in fitting)
                assert any(n == smallest and base + s * UNIT == got for n, s in fitting)
                _mark(occupied, base, [(got, want)], True)
                live.append((got, want))
        _check_against_model(alloc, occupied, base)


def test_used_extents_define_initial_free_space():
    alloc = ExtentAllocator(
        UNIT, 10 * UNIT, UNIT, used=[(3 * UNIT, 2 * UNIT), (8 * UNIT, UNIT)]
    )
    assert alloc.free_extents() == [
        (UNIT, 2 * UNIT),
        (5 * UNIT, 3 * UNIT),
        (9 * UNIT, 2 * UNIT),
    ]


@pytest.mark.parametrize(
    "used",
    [
        [(0, UNIT)],  # before base
        [(10 * UNIT, 2 * UNIT)],  # past end
        [(UNIT + 1, UNIT)],  # misaligned
        [(2 * UNIT, 2 * UNIT), (3 * UNIT, UNIT)],  # overlap
    ],
)
def test_invalid_used_extents_are_rejected(used):
    with pytest.raises(ValueError):
        ExtentAllocator(UNIT, 10 * UNIT, UNIT, used=used)


@pytest.mark.parametrize(
    "base, size, unit",
    [(0, 10 * UNIT, 0), (1, 10 * UNIT, UNIT), (0, UNIT + 1, UNIT), (0, 0, UNIT)],
)
def test_invalid_geometry_is_rejected(base, size, unit):
    with pytest.raises(ValueError):
        ExtentAllocator(base, size, unit)


def _separated(alloc: ExtentAllocator, *lengths: int) -> list[int]:
    """Allocate ranges of ``lengths`` units with a held unit after each, so
    freeing them leaves separate free extents."""
    offsets = []
    for n in lengths:
        offset = alloc.allocate(n * UNIT)
        assert offset is not None and alloc.allocate(UNIT) is not None
        offsets.append(offset)
    return offsets


def test_tag_prefers_exact_length_extent_with_same_tag():
    alloc = ExtentAllocator(0, 20 * UNIT, UNIT)
    y, x = _separated(alloc, 2, 2)
    alloc.free(y, 2 * UNIT, tag=9)
    alloc.free(x, 2 * UNIT, tag=5)  # freed last: an untagged pick takes it
    assert alloc.has_tagged_extent(2 * UNIT, 9)
    assert not alloc.has_tagged_extent(2 * UNIT, 7)
    assert alloc.allocate(2 * UNIT, tag=9) == y
    assert alloc.allocate(2 * UNIT, tag=9) == x  # no tag match: best fit


def test_split_remainder_keeps_its_tag():
    alloc = ExtentAllocator(0, 20 * UNIT, UNIT)
    (a,) = _separated(alloc, 4)
    alloc.free(a, 4 * UNIT, tag=3)
    assert alloc.allocate(UNIT, tag=3) == a
    assert alloc.has_tagged_extent(3 * UNIT, 3)


def test_coalescing_keeps_common_tag_and_drops_mixed_tags():
    alloc = ExtentAllocator(0, 6 * UNIT, UNIT)
    a, b, c = (alloc.allocate(2 * UNIT) for _ in range(3))
    assert (a, b, c) == (0, 2 * UNIT, 4 * UNIT)
    alloc.free(a, 2 * UNIT, tag=1)
    alloc.free(b, 2 * UNIT, tag=1)  # merges with the previous neighbour
    assert alloc.has_tagged_extent(4 * UNIT, 1)
    alloc.free(c, 2 * UNIT, tag=2)
    # Mixed tags merge into one untagged extent.
    assert alloc.free_extents() == [(0, 6 * UNIT)]
    assert not alloc.has_tagged_extent(6 * UNIT, 1)
    assert not alloc.has_tagged_extent(6 * UNIT, 2)


def test_coalescing_with_the_next_neighbour_drops_a_different_tag():
    alloc = ExtentAllocator(0, 4 * UNIT, UNIT)
    a, b = alloc.allocate(2 * UNIT), alloc.allocate(2 * UNIT)
    alloc.free(b, 2 * UNIT, tag=2)
    alloc.free(a, 2 * UNIT, tag=1)
    assert alloc.free_extents() == [(0, 4 * UNIT)]
    assert not alloc.has_tagged_extent(4 * UNIT, 1)
    alloc2 = ExtentAllocator(0, 4 * UNIT, UNIT)
    a, b = alloc2.allocate(2 * UNIT), alloc2.allocate(2 * UNIT)
    alloc2.free(b, 2 * UNIT, tag=1)
    alloc2.free(a, 2 * UNIT, tag=1)
    assert alloc2.has_tagged_extent(4 * UNIT, 1)


def test_remove_free_range_withdraws_only_free_bytes():
    alloc = ExtentAllocator(0, 10 * UNIT, UNIT)
    held = alloc.allocate(3 * UNIT)
    assert held == 0
    assert alloc.remove_free_range(2 * UNIT, 4 * UNIT) == 3 * UNIT
    assert alloc.free_extents() == [(6 * UNIT, 4 * UNIT)]
    # Freeing part of the held range does not resurrect the retired range.
    alloc.free(0, 2 * UNIT)
    assert alloc.free_extents() == [(0, 2 * UNIT), (6 * UNIT, 4 * UNIT)]


def test_remove_free_range_keeps_the_tag_of_what_remains():
    alloc = ExtentAllocator(0, 10 * UNIT, UNIT)
    (a,) = _separated(alloc, 6)
    alloc.free(a, 6 * UNIT, tag=4)
    alloc.remove_free_range(a + 2 * UNIT, UNIT)
    assert alloc.has_tagged_extent(2 * UNIT, 4)
    assert alloc.has_tagged_extent(3 * UNIT, 4)


@pytest.mark.parametrize("nbytes", [0, -UNIT, UNIT + 1])
def test_invalid_request_size_raises(nbytes):
    with pytest.raises(ValueError):
        ExtentAllocator(0, 10 * UNIT, UNIT).allocate(nbytes)


def test_request_larger_than_any_free_extent_returns_none_without_change():
    alloc = ExtentAllocator(0, 4 * UNIT, UNIT)
    assert alloc.allocate(5 * UNIT) is None
    assert alloc.free_extents() == [(0, 4 * UNIT)]
    held = [alloc.allocate(UNIT) for _ in range(4)]
    alloc.free(held[0], UNIT)
    alloc.free(held[2], UNIT)
    # Two free units, but not contiguous.
    assert alloc.allocate(2 * UNIT) is None
    assert alloc.free_bytes == 2 * UNIT


@pytest.mark.parametrize(
    "offset, length",
    [
        (0, UNIT),  # starts where free space starts
        (UNIT, UNIT),  # ends where free space ends
        (UNIT + 1, UNIT),  # misaligned
        (9 * UNIT, 2 * UNIT),  # past the end
    ],
)
def test_bad_free_is_rejected_without_change(offset, length):
    alloc = ExtentAllocator(0, 10 * UNIT, UNIT)
    held = [alloc.allocate(UNIT) for _ in range(10)]
    alloc.free(held[0], UNIT)
    alloc.free(held[1], UNIT)
    before = alloc.free_extents()
    with pytest.raises(ValueError):
        alloc.free(offset, length)
    assert alloc.free_extents() == before
    assert alloc.free_bytes == 2 * UNIT


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("pattern", ["alternating", "phase", "small-holes"])
def test_adversarial_size_patterns_near_full_match_reference_model(
    seed: int, pattern: str
):
    """Small/large sizes that alternate, come in phases, or free only the small
    objects between large ones keep the device near full and fragmented; free
    space must still match the model, and an allocation fails only when no
    free run is long enough."""
    rng = random.Random(seed)
    units = 800
    alloc = ExtentAllocator(0, units * UNIT, UNIT)
    occupied = [False] * units
    live: list[tuple[int, int]] = []
    small, large = 1, 37
    stores = 0
    for step in range(4000):
        if pattern == "alternating":
            want = small if stores % 2 == 0 else large
        elif pattern == "phase":
            want = small if (step // 500) % 2 == 0 else large
        else:  # fill with small/large pairs, then free only the small ones
            want = large if step % 1000 >= 500 else rng.choice((small, large))
        got = alloc.allocate(want * UNIT)
        if got is None:
            assert max((n for _, n in _free_runs(occupied)), default=0) < want
            # Near full: evict, as the eviction controller would.
            victims = live
            if pattern == "small-holes":
                victims = [e for e in live if e[1] == small * UNIT] or live
            for extent in rng.sample(victims, max(1, len(victims) // 10)):
                live.remove(extent)
                alloc.free(*extent)
                _mark(occupied, 0, [extent], False)
        else:
            _mark(occupied, 0, [(got, want * UNIT)], True)
            live.append((got, want * UNIT))
            stores += 1
        if step % 50 == 0:
            _check_against_model(alloc, occupied, 0)
    _check_against_model(alloc, occupied, 0)


def _fewest_extents(runs: list[int], nbytes: int) -> int | None:
    """Fewest free runs whose lengths cover ``nbytes`` (longest first)."""
    total = 0
    for count, length in enumerate(sorted(runs, reverse=True), start=1):
        total += length
        if total >= nbytes:
            return count
    return None


@pytest.mark.parametrize("seed", range(20))
def test_random_split_allocations_match_reference_model(seed: int):
    rng = random.Random(seed)
    base = 3 * UNIT
    units = 400
    max_extents = 4
    min_first = 2 * UNIT
    alloc = ExtentAllocator(base, units * UNIT, UNIT)
    occupied = [False] * units
    live: list[list[tuple[int, int]]] = []

    for _ in range(3000):
        if live and (rng.random() < 0.4 or alloc.free_bytes < UNIT):
            extents = live.pop(rng.randrange(len(live)))
            for offset, length in extents:
                alloc.free(offset, length)
            _mark(occupied, base, extents, False)
        else:
            want = rng.randint(1, 60) * UNIT
            runs = [n * UNIT for _, n in _free_runs(occupied)]
            before = alloc.free_extents()
            got = alloc.allocate_extents(want, max_extents, min_first=min_first)
            fewest = _fewest_extents(runs, want)
            if (
                fewest is None
                or fewest > max_extents
                or max(runs, default=0) < min_first
            ):
                assert got is None
                assert alloc.free_extents() == before
            else:
                assert got is not None
                assert sum(n for _, n in got) == want
                # Longest first, so the header always fits in the first one.
                assert got[0][1] >= min(min_first, want)
                assert len(got) == min(fewest, len(got)) <= max_extents
                _mark(occupied, base, got, True)
                live.append(got)
        _check_against_model(alloc, occupied, base)


def _holes(*lengths: int) -> tuple[ExtentAllocator, list[int]]:
    """An allocator whose only free space is holes of ``lengths`` units."""
    alloc = ExtentAllocator(0, sum(n + 1 for n in lengths) * UNIT, UNIT)
    holes = _separated(alloc, *lengths)
    for offset, n in zip(holes, lengths, strict=True):
        alloc.free(offset, n * UNIT)
    return alloc, holes


def test_split_allocation_takes_whole_extents_then_a_best_fit():
    alloc, holes = _holes(8, 5, 3, 2)
    got = alloc.allocate_extents(14 * UNIT, 8)
    # The longest (8) whole; no hole fits the 6 left, so the longest left (5)
    # whole; then 1 from the smallest hole that fits it (2).
    assert got == [
        (holes[0], 8 * UNIT),
        (holes[1], 5 * UNIT),
        (holes[3], UNIT),
    ]


@pytest.mark.parametrize(
    "nbytes, max_extents, min_first",
    [(19 * UNIT, 8, 0), (17 * UNIT, 3, 0), (10 * UNIT, 8, 9 * UNIT)],
    ids=["more-than-free", "too-many-extents", "header-longer-than-any"],
)
def test_split_allocation_that_cannot_fit_changes_nothing(
    nbytes, max_extents, min_first
):
    alloc, _ = _holes(8, 5, 3, 2)
    before = alloc.free_extents()
    assert alloc.allocate_extents(nbytes, max_extents, min_first=min_first) is None
    assert alloc.free_extents() == before


@pytest.mark.parametrize("max_extents", [0, 1])
def test_split_allocation_needs_at_least_two_extents(max_extents):
    alloc = ExtentAllocator(0, 8 * UNIT, UNIT)
    with pytest.raises(ValueError):
        alloc.allocate_extents(UNIT, max_extents)
