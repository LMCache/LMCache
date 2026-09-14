# SPDX-License-Identifier: Apache-2.0
"""Tests for ``NixlFilePool`` slot-file creation.

The pool pre-opens one file per slot and hands the fds to NIXL as ``FILE_SEG``
descriptors.  These tests cover the three things that go wrong at construction
time: a filesystem that refuses ``O_DIRECT``, a failure part-way through the
loop, and the permissions of the files that get created.

Requires the ``nixl`` package only because ``nixl_storage_backend`` imports it
at module scope; no NIXL agent, transport, or hardware is involved.
"""

# Standard
from unittest import mock
import errno
import os
import stat

# Third Party
import pytest

pytest.importorskip("nixl")

# First Party
from lmcache.v1.storage_backend.nixl_storage_backend import (  # noqa: E402
    DEFAULT_FILE_CREATE_MODE,
    NixlFilePool,
)
from lmcache.v1.storage_backend.path_sharder import PathSharder  # noqa: E402

_POOL_SIZE = 6


def _make_sharder(path: str) -> PathSharder:
    return PathSharder(
        raw_csv=path,
        strategy="by_gpu",
        dst_device="cuda:0",
        create_dirs=True,
    )


def _open_fd_count() -> int:
    """Number of fds held by this process (Linux)."""
    return len(os.listdir("/proc/self/fd"))


@pytest.mark.parametrize(
    "refuse_errno",
    [errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM, errno.EINVAL],
)
def test_file_pool_falls_back_when_odirect_refused(tmp_path, refuse_errno):
    """A filesystem refusing O_DIRECT yields a buffered pool, not an exception.

    Only the *platform* used to be checked (``hasattr(os, "O_DIRECT")``), so a
    refusal at ``open()`` propagated the ``OSError`` and aborted pool
    construction -- despite the log promising a fallback to buffered I/O.

    The refusal is injected here rather than provoked from a real filesystem:
    xfs, ext4, tmpfs, overlayfs and FUSE all accept O_DIRECT on the kernels
    this was developed against, so no locally mountable filesystem exercises
    the path. The errnos are parametrized because the flag may be refused with
    EINVAL (the documented value), EOPNOTSUPP/ENOTSUP, or EPERM under policy.

    Also asserts the refusal is latched: every slot in a pool lives on one
    filesystem, so O_DIRECT must be attempted once, not once per slot.
    """
    sharder = _make_sharder(str(tmp_path))
    real_open = os.open
    odirect = getattr(os, "O_DIRECT", 0)
    direct_attempts = []

    def refusing_open(path, flags, *args, **kwargs):
        if odirect and (flags & odirect):
            direct_attempts.append(path)
            raise OSError(refuse_errno, os.strerror(refuse_errno))
        return real_open(path, flags, *args, **kwargs)

    with mock.patch(
        "lmcache.v1.storage_backend.nixl_storage_backend.os.open",
        side_effect=refusing_open,
    ):
        pool = NixlFilePool(_POOL_SIZE, sharder, use_direct_io=True)

    try:
        assert len(pool.fds) == _POOL_SIZE
        for fd in pool.fds:
            assert os.fstat(fd).st_size == 0  # usable fd
        assert len(direct_attempts) == 1, (
            f"O_DIRECT should be attempted once and then latched off, "
            f"got {len(direct_attempts)} attempts"
        )
    finally:
        pool.close()


def test_file_pool_closes_fds_when_construction_fails(tmp_path):
    """A failure part-way through construction must not leak fds.

    ``__init__`` raising means the object is never constructed, so ``close()``
    never runs and Python cannot reclaim raw int fds -- the ones opened before
    the failure would leak for the lifetime of the process.
    """
    sharder = _make_sharder(str(tmp_path))
    real_open = os.open
    fail_after = 3
    calls = {"n": 0}

    def failing_open(path, flags, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > fail_after:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real_open(path, flags, *args, **kwargs)

    before = _open_fd_count()
    with mock.patch(
        "lmcache.v1.storage_backend.nixl_storage_backend.os.open",
        side_effect=failing_open,
    ):
        with pytest.raises(OSError) as excinfo:
            NixlFilePool(_POOL_SIZE, sharder, use_direct_io=False)
    assert excinfo.value.errno == errno.ENOSPC

    after = _open_fd_count()
    assert after == before, (
        f"leaked {after - before} fd(s) after a failed pool construction "
        f"(opened {fail_after} before the failure)"
    )


def test_file_pool_creates_files_with_expected_mode(tmp_path):
    """Slot files are created 0o644, not 0o777 & ~umask.

    The module defines ``DEFAULT_FILE_CREATE_MODE`` for this, and the dynamic
    backend passes it; the pool used to omit the mode argument entirely, which
    left KV cache files world-readable (0o755 under a typical umask).
    """
    sharder = _make_sharder(str(tmp_path))
    pool = NixlFilePool(_POOL_SIZE, sharder, use_direct_io=False)
    try:
        created = sorted(p for p in os.listdir(sharder.selected) if p.endswith(".bin"))
        assert len(created) == _POOL_SIZE
        for name in created:
            mode = stat.S_IMODE(os.stat(os.path.join(sharder.selected, name)).st_mode)
            assert mode == DEFAULT_FILE_CREATE_MODE, (
                f"{name} has mode {oct(mode)}, expected {oct(DEFAULT_FILE_CREATE_MODE)}"
            )
    finally:
        pool.close()


def test_file_pool_close_is_idempotent(tmp_path):
    """``close()`` twice must not raise EBADF or double-close a reused fd.

    Double-closing is worse than noisy: the fd number can have been handed to
    something else by then, so the second close would shut down an unrelated
    file.
    """
    sharder = _make_sharder(str(tmp_path))
    pool = NixlFilePool(_POOL_SIZE, sharder, use_direct_io=False)
    pool.close()
    assert pool.fds == []
    pool.close()  # must be a no-op
    assert pool.fds == []
