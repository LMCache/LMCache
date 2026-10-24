# SPDX-License-Identifier: Apache-2.0
"""Node-local warm-prefetch operations (submit + status).

:class:`PrefetchService` resolves a token sequence to per-rank keys and submits
a **warm** load (retained, unlocked) from L2 into L1, returning a ``request_id``
the caller polls. It validates its own inputs and raises transport-agnostic
domain errors (see :mod:`cache_control.errors`); the HTTP layer maps those to
status codes. It owns the node's :class:`WarmPrefetchJobs` table.
"""

# Standard
from typing import Any

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.multiprocess.cache_control.errors import (
    InvalidRequest,
    NotFound,
    Unavailable,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    resolve_grouped_object_keys,
)
from lmcache.v1.multiprocess.warm_prefetch import (
    COMPLETED,
    UNKNOWN,
    WarmPrefetchJobs,
)

# Warm prefetch loads from L2 into L1; other directions are rejected.
_SOURCE_TIER = Tier.L2
_TARGET_TIER = Tier.L1


class PrefetchService:
    """Submit and poll warm prefetches on one node.

    Args:
        engine: The node's cache engine (resolves tokens and runs the load).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._jobs = WarmPrefetchJobs()

    def submit(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        source_tier: Tier,
        target_tier: Tier,
    ) -> dict[str, object]:
        """Submit a warm prefetch of a token sequence's chunks from L2 into L1.

        Returns:
            ``{"request_id", "chunks", "status": "submitted"}``, or
            ``{"chunks": 0, "status": "noop"}`` for a sub-chunk sequence.

        Raises:
            InvalidRequest: unsupported direction, token cap exceeded, or an
                invalid key field.
            Unavailable: no layout registered for the model (not on this node).
        """
        if source_tier != _SOURCE_TIER or target_tier != _TARGET_TIER:
            raise InvalidRequest(
                f"unsupported prefetch direction {source_tier.value!r}->"
                f"{target_tier.value!r}; only {_SOURCE_TIER.value!r}->"
                f"{_TARGET_TIER.value!r}"
            )
        ctx = self._engine.context
        registry = ctx.layout_desc_registry
        # Every object group the model registered, each with its own layout
        # and attention window -- the same rows the lookup path submits. One
        # layout for group 0 would warm a fraction of each chunk under
        # --separate-object-groups and report success.
        unregistered = Unavailable(
            f"no layout registered for model_name={model_name!r} "
            f"world_size={world_size}; the model has not allocated "
            f"KV cache on this node yet"
        )
        group_layout_descs = registry.find_group_layout_descs(model_name, world_size)
        if not group_layout_descs:
            raise unregistered
        try:
            attn_desc = registry.find_attn_desc(model_name, world_size)
        except ValueError:
            # Unregistered between the two reads: the same answer, not a
            # malformed request.
            raise unregistered from None
        worker_group_layout_descs = registry.find_worker_group_layout_descs(
            model_name, world_size
        )
        try:
            key_groups, chunks = resolve_grouped_object_keys(
                ctx.token_hasher,
                model_name,
                world_size,
                token_ids,
                cache_salt,
                group_layout_descs,
                attn_desc,
                worker_group_layout_descs=worker_group_layout_descs,
            )
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from None
        if not chunks:
            return {"chunks": 0, "status": "noop"}
        request_id = self._jobs.submit(self._engine.storage_manager, key_groups)
        return {"request_id": request_id, "chunks": chunks, "status": "submitted"}

    def status(self, request_id: str) -> dict[str, object]:
        """Report a job's status, finalizing it on the first completed poll.

        Returns:
            ``{"request_id", "status": "pending"}`` or ``{"request_id",
            "status": "completed", "found_keys", "total_keys"}``.

        Raises:
            NotFound: unknown id (already completed-and-consumed, or never
                submitted).
        """
        status = self._jobs.poll(self._engine.storage_manager, request_id)
        if status.state == UNKNOWN:
            raise NotFound(
                f"unknown prefetch request_id={request_id!r} "
                f"(already completed or never submitted)"
            )
        if status.state == COMPLETED:
            return {
                "request_id": request_id,
                "status": COMPLETED,
                "found_keys": status.found_keys,
                "total_keys": status.total_keys,
            }
        return {"request_id": request_id, "status": status.state}
