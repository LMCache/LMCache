# SPDX-License-Identifier: Apache-2.0
"""Control RPCs for optional restoration of earlier recurrent checkpoints."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import time

# Third Party
import msgspec

from lmcache.v1.multiprocess.recurrent_lease import reserve_recurrent_history
from lmcache.v1.multiprocess.recurrent_reservations import (
    HistoryReservation,
    RecurrentReservations,
)

# First Party
from lmcache.v1.transport.recurrent_history import (
    HistoryRequest,
    HistoryResponse,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
    from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import ContextEntry


def handle_recurrent_history(
    payload: bytes,
    context: MPCacheServerContext,
    reservations: RecurrentReservations,
    entries: dict[int, ContextEntry],
) -> bytes:
    """Process a history control request on the server's existing RPC path.

    Args:
        payload: Encoded HistoryRequest received by the server module router.
        context: Shared server state and L1 storage.
        reservations: Server-owned history reservations.
        entries: Snapshot of the registered device contexts.

    Returns:
        Encoded HistoryResponse for the plan or lifecycle operation.

    Raises:
        ValueError: The operation or registered copy geometry is unsupported.
    """
    if len(payload) > 4 * 1024 * 1024:
        raise ValueError("Recurrent history control payload exceeds 4 MiB")
    reservations.reap(time.monotonic())
    request = msgspec.msgpack.decode(payload, type=HistoryRequest)
    if request.action in ("status", "cancel"):
        result = reservations.query(request.lease, cancel=request.action == "cancel")
        return msgspec.msgpack.encode(result)
    if request.action == "acknowledge":
        result = reservations.query(request.lease)
        reservations.acknowledge(request.lease)
        return msgspec.msgpack.encode(result)
    if request.action != "prepare" or request.key is None:
        raise ValueError("Expected a history prepare with a scheduler prefix key")
    key = request.key
    matching = [
        e.cache_context.kv_layer_groups_manager
        for e in entries.values()
        if e.model_name == key.model_name and e.world_size == key.world_size
    ]
    if not matching:
        return msgspec.msgpack.encode(HistoryResponse())
    reads = reserve_recurrent_history(context, key, set(request.already_local))
    if reads is None:
        return msgspec.msgpack.encode(HistoryResponse())
    group_maps = [
        sorted(
            {
                manager.kernel_groups[k].engine_group_idx
                for g in reads.groups
                for k in manager.object_groups[g].kernel_group_indices
            }
        )
        for manager in matching
    ]
    if any(groups != group_maps[0] for groups in group_maps):
        reads.progress.cancel()
        reads.release_completed()
        raise ValueError("Registered recurrent engine groups differ across workers")
    entry = HistoryReservation(key, reads, group_maps[0])
    token = reservations.add(entry)
    return msgspec.msgpack.encode(entry.result(token))
