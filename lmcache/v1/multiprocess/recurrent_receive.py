# SPDX-License-Identifier: Apache-2.0
"""Run an optional history copy after the primary retrieve has been enqueued."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# First Party
from lmcache import torch_dev
from lmcache.v1.common.token_codec import TOKEN_STRIDE
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.recurrent_copy import enqueue_recurrent_copies
from lmcache.v1.multiprocess.recurrent_reservations import RecurrentReservations
from lmcache.v1.platform.stream import synchronize_stream
from lmcache.v1.transport.recurrent_history import HistoryCopy

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.platform.base.cache_context import BaseCacheContext


def receive_recurrent_history(
    context: BaseCacheContext,
    reservations: RecurrentReservations,
    key: IPCCacheServerKey,
    copy: HistoryCopy,
    primary_blocks: set[int],
    primary_succeeded: bool,
) -> None:
    """Copy and fence a claimed worker's private history destinations.

    Args:
        context: Registered context used for the primary receive.
        reservations: CPU reservation table shared by all worker handlers.
        key: Worker prefix key used for the primary load.
        copy: History reservation token and allocated device destinations.
        primary_blocks: Blocks occupied by the primary load or shared prefix.
        primary_succeeded: Whether the primary load succeeded.

    Raises:
        ValueError: The copy key differs from its reserved source prefix.
        RuntimeError: A transfer or stream fence fails.
    """
    with reservations.lock:
        entry = reservations.entries.get(copy.lease)
        if entry is None:
            return
        source = entry.key
        if (
            key.worker_id is None
            or key.model_name != source.model_name
            or key.world_size != source.world_size
            or key.request_id != source.request_id
            or key.cache_salt != source.cache_salt
            or key.token_bytes[: source.end * TOKEN_STRIDE]
            != source.token_bytes[: source.end * TOKEN_STRIDE]
            or key.end != source.end
        ):
            raise ValueError("History copy does not match its reserved prefix")
        rank = key.worker_id
        if not entry.reads.progress.start(rank):
            return
    succeeded = False
    try:
        if primary_succeeded:
            with torch_dev.device(context.device), torch_dev.stream(context.stream):
                enqueue_recurrent_copies(
                    context, entry.reads, rank, copy.destinations, primary_blocks
                )
            succeeded = True
    finally:
        # If the fence fails, keep the running claim and all memory references.
        synchronize_stream(context.device, context.stream)
        with reservations.lock:
            entry.reads.progress.finish(rank, succeeded)
            entry.result(copy.lease)
