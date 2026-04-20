# SPDX-License-Identifier: Apache-2.0
"""Patch ``os`` with eventfd shims on non-Linux platforms.

Linux exposes ``os.eventfd`` natively; macOS and other systems
do not.  The helpers here emulate eventfd semantics using
``os.pipe`` so that call-sites can keep using ``os.eventfd``
transparently.
"""

# Standard
import os
import struct

HAS_EVENTFD: bool = hasattr(os, "eventfd")

# When using pipe-based fallback, map the "public" read-end fd
# to the (read_fd, write_fd) pair so we can write to the
# correct end.
_pipe_registry: dict[int, tuple[int, int]] = {}

_eventfd_compat_installed: bool = False


def _compat_eventfd(initval: int, flags: int = 0) -> int:
    """Create an eventfd-like file descriptor via ``os.pipe``."""
    # Standard
    import fcntl

    r, w = os.pipe()
    for fd in (r, w):
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)

    if initval > 0:
        os.write(w, struct.pack("<Q", initval))

    _pipe_registry[r] = (r, w)
    return r


def _compat_eventfd_write(efd: int, value: int) -> None:
    """Signal the pipe-based eventfd.

    Real eventfd_write always writes an 8-byte uint64 counter.
    We replicate that here so the read side can decode properly.
    """
    pair = _pipe_registry.get(efd)
    if pair is None:
        raise OSError("compat_eventfd_write: unknown fd %d" % efd)
    _, w = pair
    try:
        os.write(w, struct.pack("<Q", value))
    except BlockingIOError:
        pass  # pipe buffer full, already signaled


def _compat_eventfd_read(efd: int) -> int:
    """Consume the pipe-based eventfd signal.

    Drains all pending 8-byte counter values and returns
    their sum, matching real eventfd semantics.
    """
    total = 0
    try:
        while True:
            data = os.read(efd, 4096)
            if not data:
                break
            # Each write is an 8-byte uint64 frame.
            for off in range(0, len(data), 8):
                chunk = data[off : off + 8]
                if len(chunk) == 8:
                    total += struct.unpack("<Q", chunk)[0]
    except (BlockingIOError, OSError):
        if total == 0:
            raise
    return total


def eventfd_close(efd: int) -> None:
    """Close a compat eventfd and its hidden write-end.

    Prefer this over ``os.close`` for pipe-based eventfds so
    that both ends of the underlying pipe are released.  On
    native Linux eventfds this simply delegates to ``os.close``.
    """
    pair = _pipe_registry.pop(efd, None)
    if pair is not None:
        r, w = pair
        for fd in (r, w):
            try:
                os.close(fd)
            except OSError:
                pass
    else:
        os.close(efd)


def install_eventfd_compat() -> None:
    """Patch ``os`` with eventfd shims when missing.

    Must be called exactly once, at platform package init time.
    """
    global _eventfd_compat_installed  # noqa: PLW0603
    if _eventfd_compat_installed or HAS_EVENTFD:
        return
    _eventfd_compat_installed = True

    os.eventfd = _compat_eventfd  # type: ignore[attr-defined,misc]
    os.eventfd_read = _compat_eventfd_read  # type: ignore[attr-defined,assignment]
    os.eventfd_write = _compat_eventfd_write  # type: ignore[attr-defined,assignment]
    os.EFD_NONBLOCK = 0  # type: ignore[attr-defined]
    os.EFD_CLOEXEC = 0  # type: ignore[attr-defined]

    # Wrap os.close as a lightweight safety net so that
    # callers who forget to use eventfd_close still clean
    # up the hidden write-end.
    _orig_close = os.close

    def _patched_close(fd: int) -> None:
        pair = _pipe_registry.pop(fd, None)
        if pair is not None:
            r, w = pair
            for f in (r, w):
                try:
                    _orig_close(f)
                except OSError:
                    pass
        else:
            _orig_close(fd)

    os.close = _patched_close  # type: ignore[assignment]
