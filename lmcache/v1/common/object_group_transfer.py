# SPDX-License-Identifier: Apache-2.0
"""Callback contract for a cache context's default object-group transfer."""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    # Standard
    from collections.abc import Sequence

    # Third Party
    import torch

    # First Party
    from lmcache.v1.common.memory_management import MemoryObj
    from lmcache.v1.platform.base.cache_context import BaseCacheContext
    import lmcache.lmcache_native as lmcache_native


class ObjectGroupTransfer(Protocol):
    """Enqueue one object group's copies on the supplied context's stream.

    The server supplies the default callback; device contexts may instead
    override ``transfer_object_group`` to match their staging layout.
    """

    def __call__(
        self,
        cache_context: BaseCacheContext,
        block_ids_gpu: list[torch.Tensor],
        memory_objs: Sequence[MemoryObj | None],
        object_group_id: int,
        batch_size: int,
        skip_first_n_tokens: int,
        direction: lmcache_native.TransferDirection,
        *,
        transfer_key: str,
        block_ids_host: Sequence[Sequence[int]] = (),
    ) -> None:
        """Transfer one object group without committing storage.

        Args:
            cache_context: KV layout, pointers, staging buffers and stream.
            block_ids_gpu: Staged block IDs per kernel group, in chunk order.
            memory_objs: One object per chunk. Retrieve rejects None entries;
                store skips their batches.
            object_group_id: Object group index.
            batch_size: Chunks per batch.
            skip_first_n_tokens: Leading tokens to leave untouched.
            direction: H2D for retrieve or D2H for store.
            transfer_key: Operation ID for phase timing.
            block_ids_host: Host block IDs; empty forces the kernel copy path.

        Returns:
            None after copies have been enqueued on the context's stream.

        Raises:
            ValueError: If a retrieve encounters a missing object.
            NotImplementedError: If an object is owned by a peer.

        Copy failures propagate to the caller. The caller owns completion,
        storage commit and the lifetime of the context and memory objects.
        """
        ...
