# SPDX-License-Identifier: Apache-2.0
"""Shared-memory primitives shared by SHM-based transports.

Thin POSIX-SHM facade exposing the legacy
``shm_create_readwrite`` / ``shm_map_readwrite`` / ``shm_munmap`` /
``shm_unlink`` / ``shm_open_pool_as_mmap`` quartet, so the CPU
KV-cache wrapper, the MP non-GPU SHM transport, and the existing
tests keep working unchanged.

We deliberately route through CPython's bundled ``_posixshmem`` C
extension (used internally by :mod:`multiprocessing.shared_memory`)
rather than the higher-level :class:`SharedMemory` wrapper. The
wrapper keeps an internal ``memoryview`` over its own ``mmap``;
when callers also export a buffer (via
``ctypes.c_uint8.from_buffer(shm.buf)`` / ``torch.frombuffer(...)``),
:meth:`SharedMemory.close` invoked from ``__del__`` at interpreter
shutdown raises ``BufferError: cannot close exported pointers
exist``. Owning the ``mmap`` ourselves and pairing every alloc with
an explicit ``shm_munmap`` keeps shutdown silent on macOS and Linux
alike.

The previous hand-rolled libc/librt implementation tripped over
macOS' shm_open MAC label propagation when certain native
extensions (torch + a few others) were already loaded in the
parent process, producing spurious ``errno=13 / EACCES`` failures
on the child side. Routing through ``_posixshmem.shm_open`` -- the
same underlying entry point CPython's stdlib uses -- fixes that
and is identical on Linux.
"""

# Future
from __future__ import annotations

# Standard
import ctypes
import mmap as _mmap
import os
import threading

# Third Party
import _posixshmem  # type: ignore[import-not-found]


def _strip_leading_slash(name: str) -> str:
    """Normalise a name to the bare form (no leading ``/``).

    Callers historically embed the POSIX leading slash; we keep the
    on-wire name slash-prefixed but feed the bare form to
    ``_posixshmem.shm_open``-derived helpers that prepend it again.
    """
    return name[1:] if name.startswith("/") else name


def _slashed(name: str) -> str:
    """Inverse of :func:`_strip_leading_slash` for shm_open calls."""
    return name if name.startswith("/") else "/" + name


# Per-process registry mapping the public ``int`` address back to the
# ``mmap`` object that owns the mapping, so a later ``shm_munmap`` can
# call ``mmap.close()`` exactly once and avoid leaking pages. Owners
# (creators) also remember the name so ``shm_unlink`` can find it
# without a re-open round-trip.
_REGISTRY_LOCK = threading.Lock()
_ADDR_TO_MMAP: dict[int, _mmap.mmap] = {}
_OWNED_NAMES: set[str] = set()


def _open_and_mmap(name: str, nbytes: int, *, create: bool) -> tuple[_mmap.mmap, int]:
    """Open (or create) a POSIX SHM segment and ``mmap`` it.

    Returns a ``(mmap_obj, base_addr)`` pair. The fd is always closed
    before returning so we don't leak descriptors; the kernel keeps
    the mapping alive as long as ``mmap_obj`` stays alive.
    """
    flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
    fd = _posixshmem.shm_open(_slashed(name), flags, mode=0o600)
    mm: _mmap.mmap | None = None
    try:
        if create:
            os.ftruncate(fd, nbytes)
        mm = _mmap.mmap(fd, nbytes, access=_mmap.ACCESS_WRITE)
    except BaseException:
        if create:
            try:
                _posixshmem.shm_unlink(_slashed(name))
            except OSError:
                pass
        raise
    finally:
        os.close(fd)
    addr = _addr_of_mmap(mm)
    return mm, addr


def _addr_of_mmap(mm: _mmap.mmap) -> int:
    """Return the base address of an ``mmap`` without leaking a buffer view.

    A ctypes buffer view is created just long enough to read the
    address, then dropped before this function returns; once it is
    out of scope the mmap has no exported pointers, so a later
    ``mm.close()`` can complete cleanly.
    """
    view = (ctypes.c_uint8 * len(mm)).from_buffer(mm)
    addr = ctypes.addressof(view)
    del view
    return addr


def shm_create_readwrite(name: str, nbytes: int) -> int:
    """Create a new shared-memory segment and return its mmap address.

    Mirrors the previous ``shm_open(O_CREAT|O_EXCL) + ftruncate +
    mmap`` sequence: collisions raise ``OSError`` (``FileExistsError``
    is a subclass), and a failure mid-way fully tears down what was
    allocated.
    """
    sm_name = _strip_leading_slash(name)
    mm, addr = _open_and_mmap(sm_name, nbytes, create=True)
    with _REGISTRY_LOCK:
        _ADDR_TO_MMAP[addr] = mm
        _OWNED_NAMES.add(sm_name)
    return addr


def shm_map_readwrite(name: str, nbytes: int) -> int:
    """Open an existing shared-memory segment and return its address.

    ``nbytes`` must match the segment's actual size; ``mmap`` will
    raise on a mismatch.
    """
    sm_name = _strip_leading_slash(name)
    mm, addr = _open_and_mmap(sm_name, nbytes, create=False)
    with _REGISTRY_LOCK:
        _ADDR_TO_MMAP[addr] = mm
    return addr


def shm_munmap(addr: int, nbytes: int) -> None:
    """Best-effort release of a previously mapped segment by address.

    The underlying mmap is closed exactly once; subsequent calls with
    the same address are no-ops.
    """
    if not addr:
        return
    with _REGISTRY_LOCK:
        mm = _ADDR_TO_MMAP.pop(addr, None)
    if mm is None:
        return
    try:
        mm.close()
    except (BufferError, ValueError):
        # ``BufferError`` means callers still hold an exported view
        # (e.g. a torch tensor backed by this mmap); they will release
        # the mapping themselves on GC. ``ValueError`` means already
        # closed -- treat both as best-effort no-ops.
        pass


def shm_unlink(name: str) -> None:
    """Best-effort segment removal.

    Idempotent: a missing segment is treated as a successful
    no-op so callers can blindly call this on shutdown.
    """
    sm_name = _strip_leading_slash(name)
    with _REGISTRY_LOCK:
        _OWNED_NAMES.discard(sm_name)
    try:
        _posixshmem.shm_unlink(_slashed(sm_name))
    except FileNotFoundError:
        pass
    except OSError:
        # Mirrors the historical "best effort" contract -- e.g.
        # double-unlink on shutdown should never raise.
        pass


def shm_open_pool_as_mmap(name: str, nbytes: int) -> _mmap.mmap:
    """Open an existing segment as an independent ``mmap.mmap`` object.

    Convenience helper for non-GPU SHM transports that consume the
    segment via ``torch.frombuffer(mmap_obj, ...)`` rather than a raw
    address. The returned mmap is independent of any registry entry,
    so the caller takes ownership and is responsible for closing it.
    """
    sm_name = _strip_leading_slash(name)
    fd = _posixshmem.shm_open(_slashed(sm_name), os.O_RDWR, mode=0o600)
    try:
        return _mmap.mmap(fd, nbytes, access=_mmap.ACCESS_WRITE)
    finally:
        os.close(fd)
