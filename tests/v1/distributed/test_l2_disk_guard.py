# SPDX-License-Identifier: Apache-2.0
"""DiskGuard: an adapter's capacity follows the room left on its filesystem."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.disk_guard import DiskGuard

GB = 1 << 30
TB = 1 << 40


def _fs(total_tb: float, used_tb: float):
    """statvfs for a filesystem of ``total_tb`` with ``used_tb`` in use."""
    frsize = 4096
    blocks = int(total_tb * TB) // frsize
    free = blocks - int(used_tb * TB) // frsize
    return lambda path: SimpleNamespace(
        f_frsize=frsize, f_blocks=blocks, f_bfree=free, f_bavail=free
    )


def _guard(total_tb, used_tb, **limits):
    return DiskGuard("/l2", statvfs=_fs(total_tb, used_tb), ttl=0.0, **limits)


def test_needs_at_least_one_limit():
    with pytest.raises(ValueError):
        DiskGuard("/l2")
    with pytest.raises(ValueError):
        DiskGuard("/l2", high_watermark=1.5)
    with pytest.raises(ValueError):
        DiskGuard("/l2", min_free_bytes=-1)


def test_headroom_is_the_room_under_the_watermark():
    guard = _guard(10, 6, high_watermark=0.8)
    assert guard.headroom_bytes() == pytest.approx(2 * TB, rel=1e-6)


def test_headroom_is_the_room_above_the_free_floor():
    guard = _guard(10, 6, min_free_bytes=3 * TB)
    assert guard.headroom_bytes() == pytest.approx(1 * TB, rel=1e-6)


def test_the_tighter_limit_wins():
    guard = _guard(10, 6, high_watermark=0.8, min_free_bytes=3 * TB)
    assert guard.headroom_bytes() == pytest.approx(1 * TB, rel=1e-6)


def test_capacity_is_untouched_while_the_disk_has_room():
    guard = _guard(10, 2, high_watermark=0.8)
    assert guard.effective_capacity(100 * GB, 1500 * GB) == 1500 * GB


def test_capacity_shrinks_to_what_the_disk_allows():
    # 7.5 of 10 TB used, 0.5 TB of it by the adapter; 0.5 TB left under 80%.
    guard = _guard(10, 7.5, high_watermark=0.8)
    capacity = guard.effective_capacity(500 * GB, 1500 * GB)
    assert capacity == pytest.approx(500 * GB + TB // 2, rel=1e-6)


def test_disk_past_the_limit_reads_as_over_capacity():
    # Filled by something else: the adapter holds 300 GB and must give back.
    guard = _guard(10, 8.5, high_watermark=0.8)
    capacity = guard.effective_capacity(300 * GB, 1500 * GB)
    assert 0 < capacity < 300 * GB


def test_capacity_never_reads_as_unlimited():
    guard = _guard(10, 9.9, high_watermark=0.8)
    assert guard.effective_capacity(0, 1500 * GB) == 1
    assert guard.effective_capacity(10 * GB, 1500 * GB) == 1


def test_unreadable_filesystem_does_not_limit():
    def broken(path):
        raise OSError("gone")

    guard = DiskGuard("/l2", high_watermark=0.8, statvfs=broken, ttl=0.0)
    assert guard.headroom_bytes() is None
    assert guard.effective_capacity(10 * GB, 1500 * GB) == 1500 * GB


def test_reading_is_reused_within_the_ttl():
    calls = []

    def counting(path):
        calls.append(path)
        return _fs(10, 6)(path)

    guard = DiskGuard("/l2", high_watermark=0.8, statvfs=counting, ttl=60.0)
    guard.headroom_bytes()
    guard.headroom_bytes()
    assert len(calls) == 1
