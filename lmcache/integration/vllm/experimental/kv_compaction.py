# SPDX-License-Identifier: Apache-2.0
"""Apply physical-slot copies for in-place paged-KV compaction."""

from __future__ import annotations

import torch

_MAX_COMPACTION_PAYLOAD_BYTES = 32 * 1024 * 1024


def apply_kv_compaction_moves(
    kv_cache: torch.Tensor,
    physical_slot_copies: torch.Tensor,
) -> None:
    """Apply physical-slot copies to paged KV in place.

    The caller must ensure all touched blocks are private and order copies so
    an earlier destination is never needed as a source by a later chunk.

    Args:
        kv_cache: Paged KV shaped [num_blocks, physical_slots_per_block, ...].
            Trailing dimensions form one KV entry and move together.
        physical_slot_copies: [num_copies, 2] source/destination physical slot
            pairs on the same device and in the same address space as kv_cache.
    """
    if physical_slot_copies.numel() == 0:
        return

    physical_slots_per_block = kv_cache.shape[1]
    kv_entry_bytes = kv_cache[0, 0].numel() * kv_cache.element_size()
    copies_per_chunk = max(1, _MAX_COMPACTION_PAYLOAD_BYTES // kv_entry_bytes)

    for start in range(0, physical_slot_copies.shape[0], copies_per_chunk):
        chunk = physical_slot_copies[start : start + copies_per_chunk]
        source_physical_slots, destination_physical_slots = chunk.unbind(dim=1)

        # Advanced indexing snapshots every source in this chunk before any
        # destination is overwritten.
        source_payload = kv_cache[
            source_physical_slots // physical_slots_per_block,
            source_physical_slots % physical_slots_per_block,
        ]
        kv_cache[
            destination_physical_slots // physical_slots_per_block,
            destination_physical_slots % physical_slots_per_block,
        ] = source_payload
        # Avoid keeping two chunks alive across the next gather.
        del source_payload
