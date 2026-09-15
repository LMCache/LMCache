# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.multiprocess.posix_shm``.

Validates the POSIX-SHM primitives and the ``mmap``-based pool helper
shared by SHM-based transports.
"""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess import posix_shm
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_map_readwrite,
    shm_munmap,
    shm_open_pool_as_mmap,
    shm_unlink,
)


def _unique_name(tag: str) -> str:
    # macOS shm_open caps names at 31 bytes incl. leading '/'.
    return "/lmc_pshm_%s_%d" % (tag, os.getpid())


def test_create_map_munmap_unlink_roundtrip():
    name = _unique_name("rt")
    addr = shm_create_readwrite(name, 4096)
    try:
        assert addr not in (0, None)
        # Map again from a fresh address: same segment, different vaddr.
        addr2 = shm_map_readwrite(name, 4096)
        try:
            assert addr2 not in (0, None)
        finally:
            shm_munmap(addr2, 4096)
    finally:
        shm_munmap(addr, 4096)
        shm_unlink(name)


def test_create_excl_collision():
    name = _unique_name("excl")
    addr = shm_create_readwrite(name, 4096)
    try:
        with pytest.raises(OSError):
            shm_create_readwrite(name, 4096)
    finally:
        shm_munmap(addr, 4096)
        shm_unlink(name)


def test_open_pool_as_mmap_zero_copy_view():
    name = _unique_name("pool")
    nbytes = 4096
    addr = shm_create_readwrite(name, nbytes)
    try:
        mm = shm_open_pool_as_mmap(name, nbytes)
        try:
            mm[0:4] = b"\x01\x02\x03\x04"
            mm2 = shm_open_pool_as_mmap(name, nbytes)
            try:
                assert bytes(mm2[0:4]) == b"\x01\x02\x03\x04"
            finally:
                mm2.close()
        finally:
            mm.close()
    finally:
        shm_munmap(addr, nbytes)
        shm_unlink(name)


def test_munmap_no_op_on_zero_addr():
    # Should not crash; best-effort no-op.
    shm_munmap(0, 4096)


def test_munmap_can_retry_after_exported_view_is_released():
    """A blocked close must keep the mapping reachable for a later retry."""
    name = _unique_name("retry")
    addr = shm_create_readwrite(name, 4096)
    mm = posix_shm._ADDR_TO_MMAP[addr]
    exported_view = memoryview(mm)

    try:
        shm_munmap(addr, 4096)
        assert addr in posix_shm._ADDR_TO_MMAP
        assert not mm.closed

        exported_view.release()
        shm_munmap(addr, 4096)
        assert addr not in posix_shm._ADDR_TO_MMAP
        assert mm.closed
    finally:
        exported_view.release()
        if not mm.closed:
            mm.close()
        posix_shm._ADDR_TO_MMAP.pop(addr, None)
        shm_unlink(name)
