# SPDX-License-Identifier: Apache-2.0
"""Blocked container format for group-addressable accelerated compression.

A serialized object is a header followed by independently compressed blocks::

    [header][block 0][block 1] ... [block N-1]

Blocks never straddle a *group* boundary, so any single group can be
decompressed without touching the others. That is what lets a consumer
overlap decompression of group ``g + 1`` with its use of group ``g``.

Groups are a plain byte-range partition of the uncompressed buffer, so the
format is layout-agnostic. Choosing ``num_groups`` so that the boundaries
land on layer boundaries (layer-major layouts such as ``[L, 2, T, D]``) is
the caller's concern.

Header (little-endian)::

    magic       4s              b"LKCB"
    version     H               format version
    num_groups  H               independently addressable groups
    total_size  Q               uncompressed size in bytes
    group_size  I               uncompressed bytes per group (last may be short)
    block_size  I               uncompressed bytes per codec block
    num_blocks  I               number of codec blocks
    sizes       I * num_blocks  compressed size of each block

The block partition is fully determined by ``(total_size, group_size,
block_size)``, so only the compressed sizes need to be stored.
"""

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import struct

_MAGIC = b"LKCB"
_VERSION = 1

_HEADER = struct.Struct("<4sHHQIII")


def as_bytes(buf: memoryview | bytes | bytearray) -> memoryview:
    """Return a C-contiguous unsigned-byte view of ``buf``."""
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    if view.format == "B" and view.contiguous:
        return view
    return view.cast("B")


@dataclass(frozen=True)
class BlockPlan:
    """Partition of an uncompressed buffer into groups and codec blocks.

    Attributes:
        total_size: Uncompressed size in bytes.
        group_size: Uncompressed bytes per group; the last group may be shorter.
        block_size: Uncompressed bytes per codec block, capped by the backend's
            maximum single-call input size.
        block_offsets: Uncompressed start offset of each block.
        block_lengths: Uncompressed length of each block.
        group_first_block: Index of each group's first block, plus a trailing
            sentinel equal to the total block count.
    """

    total_size: int
    group_size: int
    block_size: int
    block_offsets: tuple[int, ...]
    block_lengths: tuple[int, ...]
    group_first_block: tuple[int, ...]

    @property
    def num_groups(self) -> int:
        return len(self.group_first_block) - 1

    @property
    def num_blocks(self) -> int:
        return len(self.block_lengths)

    @property
    def header_size(self) -> int:
        return _HEADER.size + 4 * self.num_blocks

    def group_range(self, group_index: int) -> tuple[int, int]:
        """Return ``(offset, length)`` of a group in the uncompressed buffer."""
        self._check_group(group_index)
        offset = group_index * self.group_size
        return offset, min(self.group_size, self.total_size - offset)

    def group_blocks(self, group_index: int) -> range:
        """Return the block indices belonging to a group."""
        self._check_group(group_index)
        return range(
            self.group_first_block[group_index],
            self.group_first_block[group_index + 1],
        )

    def _check_group(self, group_index: int) -> None:
        if not 0 <= group_index < self.num_groups:
            raise IndexError(
                f"group_index {group_index} out of range [0, {self.num_groups})"
            )


def _build(total_size: int, group_size: int, block_size: int) -> BlockPlan:
    """Derive the block partition from the three sizes that the header stores."""
    if group_size <= 0 or block_size <= 0:
        raise ValueError(
            f"group_size and block_size must be positive, "
            f"got {group_size} and {block_size}"
        )

    block_offsets: list[int] = []
    block_lengths: list[int] = []
    group_first_block: list[int] = []

    group_start = 0
    while group_start < total_size:
        group_first_block.append(len(block_lengths))
        group_end = min(group_start + group_size, total_size)
        offset = group_start
        while offset < group_end:
            length = min(block_size, group_end - offset)
            block_offsets.append(offset)
            block_lengths.append(length)
            offset += length
        group_start = group_end
    group_first_block.append(len(block_lengths))

    return BlockPlan(
        total_size=total_size,
        group_size=group_size,
        block_size=block_size,
        block_offsets=tuple(block_offsets),
        block_lengths=tuple(block_lengths),
        group_first_block=tuple(group_first_block),
    )


def plan_blocks(
    total_size: int, num_groups: int, max_block_size: int | None = None
) -> BlockPlan:
    """Partition ``total_size`` bytes into groups and codec blocks.

    Args:
        total_size: Uncompressed size in bytes.
        num_groups: Requested number of independently addressable groups. It is
            clamped to ``total_size``, and ceiling division may yield one group
            fewer than requested; use ``BlockPlan.num_groups`` for the result.
        max_block_size: Backend limit on a single compress call, or ``None``
            for no limit.

    Returns:
        The resulting partition.
    """
    if total_size <= 0:
        raise ValueError(f"total_size must be positive, got {total_size}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")

    group_size = -(-total_size // min(num_groups, total_size))
    block_size = min(group_size, max_block_size) if max_block_size else group_size
    return _build(total_size, group_size, block_size)


def pack_header(
    dst: memoryview, plan: BlockPlan, compressed_sizes: Sequence[int]
) -> int:
    """Write the header into the front of ``dst``.

    Returns:
        The header size in bytes.
    """
    if len(compressed_sizes) != plan.num_blocks:
        raise ValueError(
            f"expected {plan.num_blocks} compressed sizes, got {len(compressed_sizes)}"
        )
    if len(dst) < plan.header_size:
        raise ValueError(
            f"destination too small for header: {len(dst)} < {plan.header_size}"
        )

    _HEADER.pack_into(
        dst,
        0,
        _MAGIC,
        _VERSION,
        plan.num_groups,
        plan.total_size,
        plan.group_size,
        plan.block_size,
        plan.num_blocks,
    )
    struct.pack_into(f"<{plan.num_blocks}I", dst, _HEADER.size, *compressed_sizes)
    return plan.header_size


def unpack_header(src: memoryview) -> tuple[BlockPlan, tuple[int, ...]]:
    """Parse the header at the front of ``src``.

    Returns:
        The block partition and the per-block compressed sizes.

    Raises:
        ValueError: If the buffer is truncated, or the header is malformed or
            internally inconsistent.
    """
    if len(src) < _HEADER.size:
        raise ValueError(f"buffer too small for header: {len(src)} < {_HEADER.size}")

    (
        magic,
        version,
        num_groups,
        total_size,
        group_size,
        block_size,
        num_blocks,
    ) = _HEADER.unpack_from(src, 0)

    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}, expected {_MAGIC!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported format version {version}, expected {_VERSION}")

    index_end = _HEADER.size + 4 * num_blocks
    if len(src) < index_end:
        raise ValueError(f"buffer too small for block index: {len(src)} < {index_end}")

    plan = _build(total_size, group_size, block_size)
    if plan.num_blocks != num_blocks or plan.num_groups != num_groups:
        raise ValueError(
            "header is inconsistent with its own block partition: "
            f"declared {num_groups} groups / {num_blocks} blocks, "
            f"derived {plan.num_groups} / {plan.num_blocks}"
        )

    sizes = struct.unpack_from(f"<{num_blocks}I", src, _HEADER.size)
    return plan, sizes


def compressed_offsets(
    plan: BlockPlan, compressed_sizes: Sequence[int]
) -> tuple[int, ...]:
    """Return each block's start offset within the serialized buffer."""
    offsets: list[int] = []
    cursor = plan.header_size
    for size in compressed_sizes:
        offsets.append(cursor)
        cursor += size
    return tuple(offsets)
