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
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.multiprocess.cache_control.errors import InvalidRequest, NotFound
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    MAX_TOKEN_IDS,
    resolve_l1_keys,
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
            Unavailable: no layout registered for the model (via the resolver).
        """
        if source_tier != _SOURCE_TIER or target_tier != _TARGET_TIER:
            raise InvalidRequest(
                f"unsupported prefetch direction {source_tier.value!r}->"
                f"{target_tier.value!r}; only {_SOURCE_TIER.value!r}->"
                f"{_TARGET_TIER.value!r}"
            )
        if len(token_ids) > MAX_TOKEN_IDS:
            raise InvalidRequest(
                f"too many token_ids in a single request "
                f"(limit={MAX_TOKEN_IDS}, got={len(token_ids)})"
            )
        obj_keys, chunks, layout_desc = resolve_l1_keys(
            self._engine, model_name, world_size, token_ids, cache_salt
        )
        if not chunks:
            return {"chunks": 0, "status": "noop"}
        request_id = self._jobs.submit(
            self._engine.storage_manager, obj_keys, layout_desc
        )
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
