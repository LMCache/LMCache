# SPDX-License-Identifier: Apache-2.0
"""``FS_IOC_FIEMAP`` query and slab extent validation (``<linux/fiemap.h>``)."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import bisect
import ctypes
import fcntl

FS_IOC_FIEMAP = 0xC020660B

# One call covers a 128–256 GiB fallocated slab (~2 GiB extents on this ext4).
_FIEMAP_EXTENT_BATCH = 256

FIEMAP_FLAG_SYNC = 0x1
FIEMAP_EXTENT_LAST = 0x00000001
FIEMAP_EXTENT_UNKNOWN = 0x00000002
FIEMAP_EXTENT_DELALLOC = 0x00000004
FIEMAP_EXTENT_ENCODED = 0x00000008
FIEMAP_EXTENT_DATA_ENCRYPTED = 0x00000080
FIEMAP_EXTENT_NOT_ALIGNED = 0x00000100
FIEMAP_EXTENT_DATA_INLINE = 0x00000200
FIEMAP_EXTENT_DATA_TAIL = 0x00000400
FIEMAP_EXTENT_UNWRITTEN = 0x00000800

# Reject extents the kernel does not describe as plain on-disk blocks. NDS
# issues libnds I/O at FIEMAP physical offsets on the backing block device.
# ``FIEMAP_EXTENT_UNWRITTEN`` is valid reserved space because both PREAD and
# PWRITE address the block device directly and LMCache only reads written slots.
_UNSUPPORTED_FIEMAP_FLAGS = (
    FIEMAP_EXTENT_UNKNOWN
    | FIEMAP_EXTENT_DELALLOC
    | FIEMAP_EXTENT_ENCODED
    | FIEMAP_EXTENT_DATA_ENCRYPTED
    | FIEMAP_EXTENT_NOT_ALIGNED
    | FIEMAP_EXTENT_DATA_INLINE
    | FIEMAP_EXTENT_DATA_TAIL
)
# Logical length and physical offset must be multiples of this (libnds 512-byte
# I/O alignment).
_EXTENT_SECTOR = 512


class _FiemapExtent(ctypes.Structure):
    _fields_ = [
        ("fe_logical", ctypes.c_uint64),
        ("fe_physical", ctypes.c_uint64),
        ("fe_length", ctypes.c_uint64),
        ("fe_reserved64", ctypes.c_uint64 * 2),
        ("fe_flags", ctypes.c_uint32),
        ("fe_reserved", ctypes.c_uint32 * 3),
    ]


class _Fiemap(ctypes.Structure):
    _fields_ = [
        ("fm_start", ctypes.c_uint64),
        ("fm_length", ctypes.c_uint64),
        ("fm_flags", ctypes.c_uint32),
        ("fm_mapped_extents", ctypes.c_uint32),
        ("fm_extent_count", ctypes.c_uint32),
        ("fm_reserved", ctypes.c_uint32),
        ("fm_extents", _FiemapExtent * _FIEMAP_EXTENT_BATCH),
    ]


@dataclass(frozen=True)
class FileExtent:
    """One file extent: logical byte range and the disk physical range."""

    logical: int
    physical: int
    length: int
    flags: int


def fiemap_extents(fd: int, size: int) -> list[FileExtent]:
    """Query ``FS_IOC_FIEMAP`` for ``[0, size)`` on an open file fd.

    Returns extents trimmed to ``size``, in ascending logical order, until
    ``FIEMAP_EXTENT_LAST`` or the range is exhausted. A record that contributes
    no bytes inside ``[0, size)`` (zero length, or entirely outside the range)
    is skipped. This function does not raise for that.
    :meth:`FileExtentMap.load` validates that the records which remain cover
    ``[0, size)`` without gaps.
    """
    fm_start = 0
    out: list[FileExtent] = []
    while fm_start < size:
        req = _Fiemap(
            fm_start=fm_start,
            fm_length=size - fm_start,
            fm_flags=FIEMAP_FLAG_SYNC,
            fm_extent_count=_FIEMAP_EXTENT_BATCH,
        )
        fcntl.ioctl(fd, FS_IOC_FIEMAP, req, True)
        if req.fm_mapped_extents == 0:
            break
        mapped = min(int(req.fm_mapped_extents), _FIEMAP_EXTENT_BATCH)
        last_logical = fm_start
        for i in range(mapped):
            raw = req.fm_extents[i]
            logical = int(raw.fe_logical)
            physical = int(raw.fe_physical)
            length = int(raw.fe_length)
            flags = int(raw.fe_flags)
            # No bytes inside [0, size). Skip; the caller checks coverage.
            if logical + length <= 0 or length == 0:
                continue
            if logical >= size:
                continue
            trim_end = min(logical + length, size)
            trim_start = max(logical, 0)
            if trim_end <= trim_start:
                continue
            out.append(
                FileExtent(
                    logical=trim_start,
                    physical=physical + (trim_start - logical),
                    length=trim_end - trim_start,
                    flags=flags,
                )
            )
            last_logical = max(last_logical, logical + length)
            if flags & FIEMAP_EXTENT_LAST:
                if last_logical >= size:
                    return out
                break
        if last_logical <= fm_start:
            break
        fm_start = last_logical
    return out


def validate_slab_extents(extents: Sequence[FileExtent], size: int) -> None:
    """Reject slabs NDS cannot use: gaps, short coverage, alignment, bad flags."""
    if not extents:
        raise ValueError("NDS slab has no FIEMAP extents")
    pos = 0
    for ext in extents:
        if ext.logical != pos:
            raise ValueError(
                f"NDS slab extents have a gap at byte {pos} (got logical {ext.logical})"
            )
        if ext.length <= 0 or ext.length % _EXTENT_SECTOR:
            raise ValueError(
                "NDS extent length must be a positive multiple of "
                f"{_EXTENT_SECTOR}: {ext}"
            )
        if ext.physical <= 0 or ext.physical % _EXTENT_SECTOR:
            raise ValueError(
                "NDS extent physical offset must be a positive multiple of "
                f"{_EXTENT_SECTOR}: {ext}"
            )
        bad = ext.flags & _UNSUPPORTED_FIEMAP_FLAGS
        if bad:
            raise ValueError(
                f"NDS extent has unsupported FIEMAP flags 0x{bad:x}: {ext}"
            )
        pos += ext.length
    if pos < size:
        raise ValueError(f"NDS slab extents cover {pos} bytes, need {size}")


@dataclass(frozen=True)
class FileExtentMap:
    """Validated FIEMAP coverage of ``[0, size)`` for an NDS slab."""

    extents: tuple[FileExtent, ...]
    size: int

    def __post_init__(self) -> None:
        validate_slab_extents(self.extents, self.size)

    @classmethod
    def load(cls, fd: int, size: int) -> FileExtentMap:
        """Query ``FS_IOC_FIEMAP`` and reject a map NDS cannot use."""
        return cls(tuple(fiemap_extents(fd, size)), size)

    def extent_at(self, logical_offset: int) -> FileExtent:
        if 0 <= logical_offset < self.size:
            idx = (
                bisect.bisect_right(
                    self.extents, logical_offset, key=lambda ext: ext.logical
                )
                - 1
            )
            if idx >= 0:
                ext = self.extents[idx]
                if ext.logical <= logical_offset < ext.logical + ext.length:
                    return ext
        raise RuntimeError(
            f"NDS slab logical offset {logical_offset} is not covered by FIEMAP"
        )
