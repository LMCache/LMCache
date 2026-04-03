# SPDX-License-Identifier: Apache-2.0
# Standard
from unittest.mock import patch
import os
import select

# Third Party
import pytest

# First Party
from lmcache.v1.compat_eventfd import (
    _pipe_registry,
    compat_eventfd,
    compat_eventfd_close,
    compat_eventfd_read,
    compat_eventfd_write,
)


class TestCompatEventfdAPI:
    """Test the public API works on the current platform."""

    def test_create_and_close(self):
        """Create an efd and close it without error."""
        efd = compat_eventfd()
        assert isinstance(efd, int)
        assert efd >= 0
        compat_eventfd_close(efd)

    def test_write_then_read(self):
        """Write a signal, then read it back."""
        efd = compat_eventfd()
        try:
            compat_eventfd_write(efd, 1)
            val = compat_eventfd_read(efd)
            assert val >= 1
        finally:
            compat_eventfd_close(efd)

    def test_pollable_after_write(self):
        """The fd should be poll-readable after a write."""
        efd = compat_eventfd()
        try:
            compat_eventfd_write(efd, 1)
            poller = select.poll()
            poller.register(efd, select.POLLIN)
            events = poller.poll(100)  # 100 ms
            assert len(events) > 0
            assert events[0][1] & select.POLLIN
        finally:
            compat_eventfd_close(efd)

    def test_multiple_writes_single_read(self):
        """Multiple writes should be consumed by a single read."""
        efd = compat_eventfd()
        try:
            compat_eventfd_write(efd, 1)
            compat_eventfd_write(efd, 1)
            compat_eventfd_write(efd, 1)
            val = compat_eventfd_read(efd)
            assert val >= 1
        finally:
            compat_eventfd_close(efd)

    def test_close_idempotent_for_unknown_fd(self):
        """Closing an unknown fd should be a no-op (not raise)."""
        # Use a fd that was never created by compat_eventfd
        compat_eventfd_close(999999)


class TestPipeFallback:
    """Force the pipe-based fallback path regardless of platform."""

    def _create_pipe_efd(self):
        """Create an efd via the pipe fallback path."""
        with patch("lmcache.v1.compat_eventfd._HAS_EVENTFD", False):
            return compat_eventfd()

    def _write_pipe(self, efd, value=1):
        with patch("lmcache.v1.compat_eventfd._HAS_EVENTFD", False):
            compat_eventfd_write(efd, value)

    def _read_pipe(self, efd):
        with patch("lmcache.v1.compat_eventfd._HAS_EVENTFD", False):
            return compat_eventfd_read(efd)

    def _close_pipe(self, efd):
        with patch("lmcache.v1.compat_eventfd._HAS_EVENTFD", False):
            compat_eventfd_close(efd)

    def test_pipe_registered(self):
        """Pipe fallback should register the fd pair."""
        efd = self._create_pipe_efd()
        try:
            assert efd in _pipe_registry
            r, w = _pipe_registry[efd]
            assert r == efd
            assert w != efd
        finally:
            self._close_pipe(efd)

    def test_pipe_close_removes_registry(self):
        """Closing should remove the fd from the registry."""
        efd = self._create_pipe_efd()
        assert efd in _pipe_registry
        self._close_pipe(efd)
        assert efd not in _pipe_registry

    def test_pipe_write_read_roundtrip(self):
        """Write and read through the pipe fallback."""
        efd = self._create_pipe_efd()
        try:
            self._write_pipe(efd)
            val = self._read_pipe(efd)
            assert val == 1
        finally:
            self._close_pipe(efd)

    def test_pipe_write_unknown_fd_raises(self):
        """Writing to an unknown fd should raise OSError."""
        with patch("lmcache.v1.compat_eventfd._HAS_EVENTFD", False):
            with pytest.raises(OSError, match="unknown fd"):
                compat_eventfd_write(999999)

    def test_pipe_read_without_write_returns_1(self):
        """Reading without a prior write should not block
        and return 1."""
        efd = self._create_pipe_efd()
        try:
            val = self._read_pipe(efd)
            assert val == 1
        finally:
            self._close_pipe(efd)

    def test_pipe_pollable(self):
        """Pipe fd should be pollable with select.poll."""
        efd = self._create_pipe_efd()
        try:
            self._write_pipe(efd)
            poller = select.poll()
            poller.register(efd, select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
        finally:
            self._close_pipe(efd)

    def test_pipe_close_both_ends(self):
        """Closing should close both read and write ends."""
        efd = self._create_pipe_efd()
        r, w = _pipe_registry[efd]
        self._close_pipe(efd)
        # Both fds should now be invalid
        with pytest.raises(OSError):
            os.fstat(r)
        with pytest.raises(OSError):
            os.fstat(w)

    def test_pipe_multiple_create_close_cycles(self):
        """Create and close multiple efds without leaking."""
        for _ in range(10):
            efd = self._create_pipe_efd()
            self._write_pipe(efd)
            self._read_pipe(efd)
            self._close_pipe(efd)
        # Registry should be clean
        # (only check our test fds are gone; others may exist)
