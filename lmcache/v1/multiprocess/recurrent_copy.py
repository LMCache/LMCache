# SPDX-License-Identifier: Apache-2.0
"""Scatter recurrent checkpoint objects into private device destinations."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Set
from typing import TYPE_CHECKING

# First Party
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.multiprocess.object_group_transfer import downsample_and_stage_block_ids

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.recurrent_lease import RecurrentReadLease
    from lmcache.v1.platform.base.cache_context import BaseCacheContext


def enqueue_recurrent_copies(
    context: BaseCacheContext,
    lease: RecurrentReadLease,
    rank: int,
    destinations: Mapping[int, Mapping[int, int]],
    primary_blocks: Set[int],
) -> None:
    """Enqueue this rank's extra checkpoint copies on the current transfer stream.

    Args:
        context: The worker's registered cache buffers and transfer context.
        lease: Read-locked sources for the full checkpoint set.
        rank: Worker rank that has claimed this lease.
        destinations: Token positions mapped to engine groups and private block IDs.
        primary_blocks: Block IDs used by the primary load or shared prefix.

    The caller must keep the lease and destinations alive until a stream fence.

    Raises:
        ValueError: The geometry or destination layout is inconsistent.
    """
    if destinations.keys() != lease.keys.keys():
        raise ValueError("History destinations differ from the reserved checkpoints")

    manager = context.kv_layer_groups_manager
    limits: dict[int, int] = {}
    for group in lease.groups:
        for index in manager.object_groups[group].kernel_group_indices:
            kernel = manager.kernel_groups[index]
            if kernel.tokens_per_block != context.lmcache_tokens_per_chunk:
                raise ValueError("History requires one state block per chunk")
            engine_group = kernel.engine_group_idx
            limit = kernel.shape_desc.nb
            limits[engine_group] = min(limits.get(engine_group, limit), limit)

    occupied = set(primary_blocks)
    for row in destinations.values():
        if row.keys() != limits.keys():
            raise ValueError("History destinations differ from the recurrent groups")
        for group, block_id in row.items():
            if not 0 < block_id < limits[group] or block_id in occupied:
                raise ValueError("History destination is out of range or aliased")
            occupied.add(block_id)

    for position, row in destinations.items():
        host_ids = [[row.get(k.engine_group_idx, 0)] for k in manager.kernel_groups]
        staged = downsample_and_stage_block_ids(context, host_ids)
        sources = lease.sources_for(rank, position)
        for group, source in zip(lease.groups, sources, strict=True):
            context.transfer_object_group(
                staged,
                [source],
                object_group_id=group,
                batch_size=1,
                skip_first_n_tokens=0,
                direction=TransferDirection.H2D,
                transfer_key=f"recurrent-{rank}-{position}",
                block_ids_host=host_ids,
            )
