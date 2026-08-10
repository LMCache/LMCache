# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ``compute_mr_slice_regions``, the pure slicing math behind
sliced NIXL memory registration.

Unlike ``test_nixl_impl.py`` these tests need no nixl runtime (the function is
deliberately pure), so they always run in CI and pin down the invariants that
the hardware-dependent integration tests cannot: the regions exactly partition
the buffer, respect the slice cap, and land on alignment boundaries.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (
    compute_mr_slice_regions,
)


def _check_partition_invariants(
    regions: list[tuple[int, int, int, str]],
    ptr: int,
    size: int,
    mr_slice_bytes: int,
    align_bytes: int,
) -> None:
    """Assert regions exactly partition [ptr, ptr + size) per the contract."""
    assert len(regions) >= 1
    # Consecutive coverage: no gaps, no overlaps, exact start and end.
    assert regions[0][0] == ptr
    for (addr, length, dev_id, meta), nxt in zip(regions, regions[1:], strict=False):
        assert addr + length == nxt[0]
    last_addr, last_len, _, _ = regions[-1]
    assert last_addr + last_len == ptr + size
    for addr, length, dev_id, meta in regions:
        assert length > 0
        assert length <= mr_slice_bytes or mr_slice_bytes == 0
        assert dev_id == 0
        assert meta == ""
        # Every boundary between two regions is ptr + a multiple of align.
        if addr != ptr:
            assert (addr - ptr) % align_bytes == 0


def test_zero_mr_slice_bytes_registers_single_region():
    regions = compute_mr_slice_regions(0x1000, 4096, 0, 256)
    assert regions == [(0x1000, 4096, 0, "")]


def test_even_split_into_exact_slices():
    regions = compute_mr_slice_regions(0x1000, 4096, 1024, 256)
    assert regions == [
        (0x1000, 1024, 0, ""),
        (0x1000 + 1024, 1024, 0, ""),
        (0x1000 + 2048, 1024, 0, ""),
        (0x1000 + 3072, 1024, 0, ""),
    ]


def test_last_slice_is_shorter_when_size_is_not_a_multiple():
    regions = compute_mr_slice_regions(0, 2560, 1024, 256)
    assert regions == [(0, 1024, 0, ""), (1024, 1024, 0, ""), (2048, 512, 0, "")]


def test_slice_cap_is_rounded_down_to_alignment():
    # 1100 rounds down to 1024 (4 x 256), so the layout matches slice=1024.
    regions = compute_mr_slice_regions(0, 4096, 1100, 256)
    assert [length for _, length, _, _ in regions] == [1024, 1024, 1024, 1024]


def test_slice_larger_than_buffer_yields_single_region():
    regions = compute_mr_slice_regions(0x2000, 1024, 4096, 256)
    assert regions == [(0x2000, 1024, 0, "")]


def test_negative_mr_slice_bytes_raises_value_error():
    with pytest.raises(ValueError):
        compute_mr_slice_regions(0, 4096, -1, 256)


def test_mr_slice_bytes_below_alignment_raises_value_error():
    with pytest.raises(ValueError):
        compute_mr_slice_regions(0, 4096, 255, 256)


@pytest.mark.parametrize("ptr", [0, 0x1000, 0xDEAD0000])
@pytest.mark.parametrize("size_pages", [1, 3, 7, 64, 1023])
@pytest.mark.parametrize("align_bytes", [256, 4096, 65536])
@pytest.mark.parametrize("slice_pages", [1, 2, 5, 16, 10_000])
def test_partition_invariants_hold_across_parameter_sweep(
    ptr: int, size_pages: int, align_bytes: int, slice_pages: int
):
    """Sweep buffer/slice geometries; every combination must exactly partition
    the buffer (production-scale ratios are covered by page counts, so the
    sweep stays fast)."""
    size = size_pages * align_bytes
    # Perturb the cap so rounding-down is exercised, not just exact multiples.
    mr_slice_bytes = slice_pages * align_bytes + align_bytes - 1
    regions = compute_mr_slice_regions(ptr, size, mr_slice_bytes, align_bytes)
    _check_partition_invariants(regions, ptr, size, mr_slice_bytes, align_bytes)


def test_production_geometry_24_gib_l1_with_3_5_gib_slices():
    """The shipped default: a 24 GiB L1 sliced at 3.5 GiB -> 7 regions."""
    align = 65536
    size = 24 * 1024**3
    regions = compute_mr_slice_regions(0, size, 3_758_096_384, align)
    assert len(regions) == 7
    _check_partition_invariants(regions, 0, size, 3_758_096_384, align)
