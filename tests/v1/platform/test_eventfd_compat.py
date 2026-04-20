# SPDX-License-Identifier: Apache-2.0
"""Tests for lmcache.v1.platform.eventfd_compat."""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.platform.eventfd_compat import (
    HAS_EVENTFD,
    _compat_eventfd,
    _compat_eventfd_read,
    _compat_eventfd_write,
    _pipe_registry,
    eventfd_close,
)


class TestCompatEventfd:
    """Tests for _compat_eventfd pipe-based creation."""

    def test_returns_valid_fd(self) -> None:
        efd = _compat_eventfd(0)
        try:
            assert isinstance(efd, int)
            assert efd in _pipe_registry
        finally:
            eventfd_close(efd)

    def test_registered_in_pipe_registry(self) -> None:
        efd = _compat_eventfd(0)
        try:
            r, w = _pipe_registry[efd]
            assert r == efd
            assert w != efd
        finally:
            eventfd_close(efd)

    def test_initval_seeds_counter(self) -> None:
        efd = _compat_eventfd(initval=42)
        try:
            val = _compat_eventfd_read(efd)
            assert val == 42
        finally:
            eventfd_close(efd)

    def test_initval_zero_no_data(self) -> None:
        efd = _compat_eventfd(initval=0)
        try:
            with pytest.raises(BlockingIOError):
                _compat_eventfd_read(efd)
        finally:
            eventfd_close(efd)

    def test_flags_accepted(self) -> None:
        """flags param is accepted (no-op for compat)."""
        efd = _compat_eventfd(initval=0, flags=0)
        try:
            assert efd in _pipe_registry
        finally:
            eventfd_close(efd)


class TestCompatEventfdWrite:
    """Tests for _compat_eventfd_write."""

    def test_write_and_read_roundtrip(self) -> None:
        efd = _compat_eventfd(0)
        try:
            _compat_eventfd_write(efd, 7)
            val = _compat_eventfd_read(efd)
            assert val == 7
        finally:
            eventfd_close(efd)

    def test_multiple_writes_accumulate(self) -> None:
        efd = _compat_eventfd(0)
        try:
            _compat_eventfd_write(efd, 3)
            _compat_eventfd_write(efd, 5)
            val = _compat_eventfd_read(efd)
            assert val == 8
        finally:
            eventfd_close(efd)

    def test_write_unknown_fd_raises(self) -> None:
        with pytest.raises(OSError, match="unknown fd"):
            _compat_eventfd_write(99999, 1)

    def test_value_is_required(self) -> None:
        """value has no default, matching native os.eventfd_write."""
        efd = _compat_eventfd(0)
        try:
            with pytest.raises(TypeError):
                _compat_eventfd_write(efd)  # type: ignore[call-arg]
        finally:
            eventfd_close(efd)


class TestCompatEventfdRead:
    """Tests for _compat_eventfd_read."""

    def test_raises_when_no_data(self) -> None:
        efd = _compat_eventfd(0)
        try:
            with pytest.raises(
                (BlockingIOError, OSError),
            ):
                _compat_eventfd_read(efd)
        finally:
            eventfd_close(efd)

    def test_drains_all_pending(self) -> None:
        efd = _compat_eventfd(0)
        try:
            for i in range(1, 6):
                _compat_eventfd_write(efd, i)
            val = _compat_eventfd_read(efd)
            assert val == 15  # 1+2+3+4+5
        finally:
            eventfd_close(efd)

    def test_read_after_drain_raises(self) -> None:
        efd = _compat_eventfd(0)
        try:
            _compat_eventfd_write(efd, 1)
            _compat_eventfd_read(efd)
            with pytest.raises(
                (BlockingIOError, OSError),
            ):
                _compat_eventfd_read(efd)
        finally:
            eventfd_close(efd)


class TestEventfdClose:
    """Tests for eventfd_close."""

    def test_closes_both_ends(self) -> None:
        efd = _compat_eventfd(0)
        r, w = _pipe_registry[efd]
        eventfd_close(efd)
        assert efd not in _pipe_registry
        # Both fds should be closed now.
        with pytest.raises(OSError):
            os.fstat(r)
        with pytest.raises(OSError):
            os.fstat(w)

    def test_double_close_does_not_crash(self) -> None:
        efd = _compat_eventfd(0)
        eventfd_close(efd)
        # Second close should just delegate to os.close
        # which will raise, but eventfd_close itself
        # should not crash the process.
        with pytest.raises(OSError):
            eventfd_close(efd)

    def test_close_plain_fd(self) -> None:
        """eventfd_close on a non-compat fd delegates to os.close."""
        r, w = os.pipe()
        os.close(w)
        eventfd_close(r)
        with pytest.raises(OSError):
            os.fstat(r)


@pytest.mark.skipif(
    HAS_EVENTFD,
    reason="os.close patch only installed on non-Linux platforms",
)
class TestOsClosePatched:
    """Tests for the os.close safety-net patch."""

    def test_os_close_cleans_up_pipe_pair(self) -> None:
        efd = _compat_eventfd(0)
        r, w = _pipe_registry[efd]
        os.close(efd)
        assert efd not in _pipe_registry
        with pytest.raises(OSError):
            os.fstat(r)
        with pytest.raises(OSError):
            os.fstat(w)

    def test_os_close_plain_fd_still_works(self) -> None:
        r, w = os.pipe()
        os.close(r)
        os.close(w)
        with pytest.raises(OSError):
            os.fstat(r)


class TestInstallEventfdCompat:
    """Tests for install_eventfd_compat idempotency."""

    def test_os_eventfd_is_patched(self) -> None:
        """After install, os.eventfd should be available."""
        assert hasattr(os, "eventfd")
        assert hasattr(os, "eventfd_read")
        assert hasattr(os, "eventfd_write")

    def test_os_efd_constants_exist(self) -> None:
        assert hasattr(os, "EFD_NONBLOCK")
        assert hasattr(os, "EFD_CLOEXEC")
