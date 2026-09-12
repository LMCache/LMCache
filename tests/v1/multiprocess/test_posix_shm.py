# SPDX-License-Identifier: Apache-2.0
"""Tests for ``lmcache.v1.multiprocess.posix_shm``.

Validates the POSIX-SHM primitives and the ``mmap``-based pool helper
shared by SHM-based transports.
"""

# Standard
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest

# First Party
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


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_does_not_unlink_parent_segment():
    script = textwrap.dedent(
        """
        import os
        import sys

        from lmcache.v1.multiprocess.posix_shm import (
            shm_create_readwrite,
            shm_map_readwrite,
            shm_munmap,
            shm_unlink,
        )

        name = f"/lmc_pshm_fork_{os.getpid()}"
        child_name = f"/lmc_pshm_child_{os.getpid()}"
        nbytes = 4096
        addr = shm_create_readwrite(name, nbytes)
        child_pid = os.fork()
        if child_pid == 0:
            shm_create_readwrite(child_name, nbytes)
            sys.exit(0)

        try:
            _, status = os.waitpid(child_pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0

            mapped_addr = shm_map_readwrite(name, nbytes)
            shm_munmap(mapped_addr, nbytes)

            try:
                shm_map_readwrite(child_name, nbytes)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("child-owned segment was not unlinked")
        finally:
            shm_munmap(addr, nbytes)
            shm_unlink(name)
            shm_unlink(child_name)
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
