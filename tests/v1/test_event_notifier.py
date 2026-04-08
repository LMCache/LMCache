# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import select

# Third Party
import pytest

# First Party
from lmcache.v1.event_notifier import (
    EventNotifier,
    PipeNotifier,
    consume_fd,
    create_event_notifier,
)


class TestEventNotifierAPI:
    """Test the public API on the current platform."""

    def test_create_returns_event_notifier(self):
        """Factory returns an EventNotifier subclass."""
        n = create_event_notifier()
        try:
            assert isinstance(n, EventNotifier)
        finally:
            n.close()

    def test_fileno_is_nonnegative(self):
        """fileno() returns a valid fd."""
        n = create_event_notifier()
        try:
            assert n.fileno() >= 0
        finally:
            n.close()

    def test_notify_then_consume(self):
        """notify() makes the fd readable; consume() drains it."""
        n = create_event_notifier()
        try:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
            n.consume()
        finally:
            n.close()

    def test_multiple_notify_single_consume(self):
        """Multiple notify() calls are coalesced by one consume()."""
        n = create_event_notifier()
        try:
            n.notify()
            n.notify()
            n.notify()
            n.consume()
            # After consume, fd should not be readable
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(50)
            assert len(events) == 0
        finally:
            n.close()

    def test_consume_without_notify_is_noop(self):
        """consume() without prior notify() does not block or raise."""
        n = create_event_notifier()
        try:
            n.consume()  # should not block or raise
        finally:
            n.close()

    def test_close_is_idempotent(self):
        """Calling close() twice does not raise."""
        n = create_event_notifier()
        n.close()
        n.close()  # should not raise

    def test_context_manager(self):
        """EventNotifier works as a context manager."""
        with create_event_notifier() as n:
            n.notify()
            n.consume()
            assert n.fileno() >= 0


class TestPipeNotifier:
    """Force the pipe-based fallback path."""

    def test_both_fds_closed(self):
        """close() releases both read and write fds."""
        n = PipeNotifier()
        r = n._read_fd
        w = n._write_fd
        n.close()
        with pytest.raises(OSError):
            os.fstat(r)
        with pytest.raises(OSError):
            os.fstat(w)

    def test_notify_when_pipe_full_is_noop(self):
        """notify() does not raise when pipe buffer is full."""
        n = PipeNotifier()
        try:
            # Fill the pipe buffer
            for _ in range(100000):
                try:
                    os.write(n._write_fd, b"\x01" * 4096)
                except BlockingIOError:
                    break
            # This should not raise
            n.notify()
        finally:
            n.close()

    def test_pollable(self):
        """PipeNotifier fd is pollable with select.poll."""
        n = PipeNotifier()
        try:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
        finally:
            n.close()

    def test_multiple_create_close_cycles(self):
        """Create and close multiple notifiers without leaking."""
        for _ in range(20):
            n = PipeNotifier()
            n.notify()
            n.consume()
            n.close()


class TestConsumeFd:
    """Test the consume_fd utility function."""

    def test_consume_fd_after_notify(self):
        """consume_fd() drains a notifier's fd."""
        n = create_event_notifier()
        try:
            n.notify()
            consume_fd(n.fileno())
            # After consume, fd should not be readable
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(50)
            assert len(events) == 0
        finally:
            n.close()

    def test_consume_fd_without_signal(self):
        """consume_fd() on unsignaled fd does not block."""
        n = create_event_notifier()
        try:
            consume_fd(n.fileno())  # should not block
        finally:
            n.close()
