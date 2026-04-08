# SPDX-License-Identifier: Apache-2.0
"""Cross-platform event notification abstraction.

Provides a unified ``EventNotifier`` interface for signaling between
threads using poll-able file descriptors.  On Linux (Python 3.10+),
uses ``os.eventfd``; on other platforms (macOS, etc.), falls back to
``os.pipe`` with non-blocking I/O.

Usage::

    from lmcache.v1.platform import create_event_notifier

    notifier = create_event_notifier()
    notifier.notify()          # signal
    notifier.consume()         # drain
    fd = notifier.fileno()     # for select.poll()
    notifier.close()           # release resources

Or as a context manager::

    with create_event_notifier() as notifier:
        notifier.notify()
        notifier.consume()
"""

# Standard
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Optional, Type
import os

# First Party
from lmcache.v1.platform.capabilities import HAS_EVENTFD


class EventNotifier(ABC):
    """Abstract base class for cross-platform event notification.

    An ``EventNotifier`` models a **binary signal**: calling
    ``notify()`` makes the notifier readable via ``poll()``/
    ``select()``, and ``consume()`` resets it.  Multiple
    ``notify()`` calls before a ``consume()`` are coalesced.
    """

    @abstractmethod
    def fileno(self) -> int:
        """Return a poll-able file descriptor.

        The fd becomes readable after ``notify()`` is called.
        """

    @abstractmethod
    def notify(self) -> None:
        """Signal the notifier (idempotent if already signaled)."""

    @abstractmethod
    def consume(self) -> None:
        """Consume the pending signal (non-blocking)."""

    @abstractmethod
    def close(self) -> None:
        """Release underlying OS resources.  Idempotent."""

    # Context manager support

    def __enter__(self) -> "EventNotifier":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


class EventfdNotifier(EventNotifier):
    """Linux eventfd-based notifier (Python 3.10+)."""

    def __init__(self) -> None:
        self._efd: int = os.eventfd(  # type: ignore[attr-defined]
            0,
            os.EFD_NONBLOCK | os.EFD_CLOEXEC,  # type: ignore[attr-defined]
        )

    def fileno(self) -> int:
        return self._efd

    def notify(self) -> None:
        os.eventfd_write(self._efd, 1)  # type: ignore[attr-defined]

    def consume(self) -> None:
        try:
            os.eventfd_read(self._efd)  # type: ignore[attr-defined]
        except (BlockingIOError, OSError):
            pass

    def close(self) -> None:
        if self._efd >= 0:
            try:
                os.close(self._efd)
            except OSError:
                pass
            self._efd = -1


class PipeNotifier(EventNotifier):
    """Pipe-based fallback notifier for non-Linux platforms."""

    def __init__(self) -> None:
        # Standard
        import fcntl

        r, w = os.pipe()
        for fd in (r, w):
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        self._read_fd: int = r
        self._write_fd: int = w

    def fileno(self) -> int:
        return self._read_fd

    def notify(self) -> None:
        try:
            os.write(self._write_fd, b"\x01")
        except BlockingIOError:
            pass  # pipe buffer full — signal already pending

    def consume(self) -> None:
        try:
            while os.read(self._read_fd, 4096):
                pass
        except (BlockingIOError, OSError):
            pass

    def close(self) -> None:
        for attr in ("_read_fd", "_write_fd"):
            fd = getattr(self, attr, -1)
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, -1)


def create_event_notifier() -> EventNotifier:
    """Create a platform-appropriate EventNotifier.

    On Linux (Python 3.10+), returns an ``EventfdNotifier``.
    On other platforms, returns a ``PipeNotifier``.
    """
    if HAS_EVENTFD:
        return EventfdNotifier()
    return PipeNotifier()


def consume_fd(fd: int) -> None:
    """Consume a pending signal from a raw file descriptor.

    This is a convenience function for code that only has a
    raw fd (e.g., obtained from ``adapter.get_store_event_fd()``)
    and needs to drain it after ``poll()`` reports it readable.

    On Linux, uses ``os.eventfd_read()``; on other platforms,
    drains all bytes via ``os.read()``.
    """
    try:
        if HAS_EVENTFD:
            os.eventfd_read(fd)  # type: ignore[attr-defined]
        else:
            while os.read(fd, 4096):
                pass
    except (BlockingIOError, OSError):
        pass
