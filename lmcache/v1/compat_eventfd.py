# SPDX-License-Identifier: Apache-2.0
"""Cross-platform eventfd compatibility layer.

On Linux (Python 3.10+), delegates to ``os.eventfd`` / ``os.eventfd_read``
/ ``os.eventfd_write``.  On other platforms (macOS, etc.), emulates the
same semantics with ``os.pipe`` + non-blocking I/O.

Usage::

    from lmcache.v1.compat_eventfd import (
        compat_eventfd,
        compat_eventfd_read,
        compat_eventfd_write,
        compat_eventfd_close,
    )

    efd = compat_eventfd()
    compat_eventfd_write(efd, 1)
    compat_eventfd_read(efd)
    compat_eventfd_close(efd)
"""

# Standard
import fcntl
import os

_HAS_EVENTFD = hasattr(os, "eventfd")

# When using pipe-based fallback, map the "public" read-end fd
# to the (read_fd, write_fd) pair so we can write to the correct end.
_pipe_registry: dict[int, tuple[int, int]] = {}


def compat_eventfd() -> int:
    """Create an eventfd-like file descriptor.

    Returns:
        A file descriptor that can be polled with ``select.poll``.
    """
    if _HAS_EVENTFD:
        return os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

    r, w = os.pipe()
    # Set both ends non-blocking
    for fd in (r, w):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
    _pipe_registry[r] = (r, w)
    return r


def compat_eventfd_write(efd: int, value: int = 1) -> None:
    """Signal the eventfd (or pipe)."""
    if _HAS_EVENTFD:
        os.eventfd_write(efd, value)
        return

    pair = _pipe_registry.get(efd)
    if pair is None:
        raise OSError("compat_eventfd_write: unknown fd %d" % efd)
    _, w = pair
    try:
        os.write(w, b"\x01" * value)
    except BlockingIOError:
        pass  # pipe buffer full, already signaled


def compat_eventfd_read(efd: int) -> int:
    """Consume the eventfd (or pipe) signal.

    Returns:
        The eventfd counter value (always 1 for pipe fallback).
    """
    if _HAS_EVENTFD:
        return os.eventfd_read(efd)

    # Drain all bytes from the pipe
    try:
        while os.read(efd, 4096):
            pass
    except (BlockingIOError, OSError):
        pass
    return 1


def compat_eventfd_close(efd: int) -> None:
    """Close the eventfd (or both pipe ends)."""
    if _HAS_EVENTFD:
        try:
            os.close(efd)
        except OSError:
            pass
        return

    pair = _pipe_registry.pop(efd, None)
    if pair is not None:
        r, w = pair
        try:
            os.close(r)
        except OSError:
            pass
        try:
            os.close(w)
        except OSError:
            pass
