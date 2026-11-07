# SPDX-License-Identifier: Apache-2.0
"""Small, engine-neutral reference model for hybrid KV-cache geometry.

The implementation deliberately enumerates logical token positions instead of
sharing the production slicing helpers.  It is intended as a differential-test
oracle for serving-engine adapters that expose multiple paged-block address
spaces with different block sizes, compression ratios, or sliding windows.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupGeometry:
    """Logical and physical geometry for one engine block-address space."""

    group_id: int
    tokens_per_block: int
    physical_slots_per_block: int
    window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.tokens_per_block <= 0:
            raise ValueError("tokens_per_block must be positive")
        if self.physical_slots_per_block <= 0:
            raise ValueError("physical_slots_per_block must be positive")
        if self.tokens_per_block % self.physical_slots_per_block:
            raise ValueError(
                "tokens_per_block must be a multiple of physical_slots_per_block"
            )
        if self.window_tokens is not None:
            if self.window_tokens <= 0:
                raise ValueError("window_tokens must be positive when present")
            if self.window_tokens % self.tokens_per_block:
                raise ValueError(
                    "window_tokens must align to the group's tokens_per_block"
                )

    @property
    def compression_ratio(self) -> int:
        """Return logical tokens represented by one physical slot."""
        return self.tokens_per_block // self.physical_slots_per_block


def reference_block_ids_for_token_range(
    allocated_block_ids: Mapping[int, Sequence[int]],
    geometries: Sequence[GroupGeometry],
    token_start: int,
    token_end: int,
) -> dict[int, list[int]]:
    """Map an aligned logical-token range to block IDs for every group.

    Unlike the production implementation, this oracle visits each logical
    token in the requested range and records the block whenever its address
    changes.  This keeps the implementation independent enough to catch unit,
    offset, and mixed-block-size regressions in optimized slicing code.
    """
    ordered = _validate_geometries(geometries)
    _validate_tables(allocated_block_ids, ordered)
    if token_start < 0 or token_end < 0:
        raise ValueError("token range endpoints must be non-negative")
    if token_end < token_start:
        raise ValueError("token_end must not precede token_start")

    result: dict[int, list[int]] = {}
    for geometry in ordered:
        if token_start % geometry.tokens_per_block or (
            token_end % geometry.tokens_per_block
        ):
            raise ValueError(
                f"token range [{token_start}, {token_end}) does not align to "
                f"group {geometry.group_id} tokens_per_block "
                f"{geometry.tokens_per_block}"
            )
        table = allocated_block_ids[geometry.group_id]
        selected: list[int] = []
        previous_position: int | None = None
        for token_position in range(token_start, token_end):
            block_position = token_position // geometry.tokens_per_block
            if block_position == previous_position:
                continue
            if block_position >= len(table):
                raise ValueError(
                    f"block table for group {geometry.group_id} is too short: "
                    f"position {block_position} is required but length is "
                    f"{len(table)}"
                )
            selected.append(table[block_position])
            previous_position = block_position
        result[geometry.group_id] = selected
    return result


def reference_windowed_block_ids(
    block_ids: Mapping[int, Sequence[int]],
    geometries: Sequence[GroupGeometry],
    logical_chunk_tokens: int,
) -> dict[int, list[int]]:
    """Keep the required tail blocks of each logical chunk for every group."""
    ordered = _validate_geometries(geometries)
    _validate_tables(block_ids, ordered)
    if logical_chunk_tokens <= 0:
        raise ValueError("logical_chunk_tokens must be positive")

    result: dict[int, list[int]] = {}
    for geometry in ordered:
        if logical_chunk_tokens % geometry.tokens_per_block:
            raise ValueError(
                f"logical chunk size {logical_chunk_tokens} does not align to "
                f"group {geometry.group_id} tokens_per_block "
                f"{geometry.tokens_per_block}"
            )
        blocks_per_chunk = logical_chunk_tokens // geometry.tokens_per_block
        table = block_ids[geometry.group_id]
        if len(table) % blocks_per_chunk:
            raise ValueError(
                f"block table for group {geometry.group_id} has length "
                f"{len(table)}, which is not a multiple of {blocks_per_chunk}"
            )
        keep_blocks = blocks_per_chunk
        if (
            geometry.window_tokens is not None
            and geometry.window_tokens < logical_chunk_tokens
        ):
            keep_blocks = geometry.window_tokens // geometry.tokens_per_block

        selected: list[int] = []
        for chunk_start in range(0, len(table), blocks_per_chunk):
            chunk = table[chunk_start : chunk_start + blocks_per_chunk]
            selected.extend(chunk[-keep_blocks:])
        result[geometry.group_id] = selected
    return result


def reference_num_physical_slots(
    logical_tokens: int,
    geometry: GroupGeometry,
) -> int:
    """Return the exact physical slots needed for a logical token count."""
    if logical_tokens < 0:
        raise ValueError("logical_tokens must be non-negative")
    if logical_tokens % geometry.compression_ratio:
        raise ValueError(
            f"logical_tokens {logical_tokens} does not align to compression "
            f"ratio {geometry.compression_ratio} for group {geometry.group_id}"
        )
    slots = 0
    for logical_position in range(
        0,
        logical_tokens,
        geometry.compression_ratio,
    ):
        del logical_position
        slots += 1
    return slots


def _validate_geometries(
    geometries: Sequence[GroupGeometry],
) -> tuple[GroupGeometry, ...]:
    ordered = tuple(sorted(geometries, key=lambda item: item.group_id))
    ids = [item.group_id for item in geometries]
    if len(ids) != len(set(ids)):
        raise ValueError("group IDs must be unique")
    if [item.group_id for item in ordered] != list(range(len(ordered))):
        raise ValueError("group IDs must be dense and start at zero")
    return ordered


def _validate_tables(
    block_ids: Mapping[int, Sequence[int]],
    geometries: Sequence[GroupGeometry],
) -> None:
    expected = {item.group_id for item in geometries}
    missing = expected - block_ids.keys()
    if missing:
        raise ValueError(f"missing block table for group(s) {sorted(missing)}")
    extra = block_ids.keys() - expected
    if extra:
        raise ValueError(f"unexpected block table for group(s) {sorted(extra)}")
    for group_id, table in block_ids.items():
        if any(
            isinstance(block_id, bool) or not isinstance(block_id, int) or block_id < 0
            for block_id in table
        ):
            raise ValueError(
                f"block table for group {group_id} must contain non-negative integers"
            )
