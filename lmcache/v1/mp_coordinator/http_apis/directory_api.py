# SPDX-License-Identifier: Apache-2.0
"""Key-directory endpoints on the coordinator (fleet-level).

The ``/directory`` surface, thin over the :class:`KeyDirectory` carried on
the typed :class:`CoordinatorContext`: cache-event ingestion from MP
servers, placement lookup, key/token-id listing, and stats. See
``docs/design/v1/mp_coordinator/key_directory.md``.
"""

# Standard
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.key_directory import ApplyResult, DirectoryStats
from lmcache.v1.mp_coordinator.schemas import (
    DirectoryEventsRequest,
    DirectoryEventsResponse,
    DirectoryKeyInfo,
    DirectoryKeyPlacements,
    DirectoryListResponse,
    DirectoryLookupRequest,
    DirectoryLookupResponse,
    KeyTokenIds,
    TokenIdsRequest,
    TokenIdsResponse,
    TokenPlacementLookupRequest,
    TokenPlacementLookupResponse,
)
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

router = APIRouter()


@router.post("/directory/events")
async def report_cache_events(
    body: DirectoryEventsRequest, request: Request
) -> DirectoryEventsResponse:
    """Apply a batch of cache-event batches to the key directory.

    Batches are applied in list order; per instance they must be sent in
    emission order. Duplicate and stale-incarnation batches are dropped
    and counted, not errors. Applied batches also fan out to the
    usage/eviction consumers via the context's event router.

    Args:
        body: The event batches to apply.

    Returns:
        Counts of applied and dropped batches.
    """
    ctx = get_context(request)
    response = DirectoryEventsResponse()
    for batch in body.batches:
        result = ctx.key_directory.apply_batch(batch)
        if result == ApplyResult.APPLIED:
            ctx.event_broadcaster.broadcast(batch)
            response.applied += 1
        elif result == ApplyResult.DUPLICATE:
            response.duplicates += 1
        else:
            response.stale += 1
    return response


@router.post("/directory/lookup")
async def lookup_placements(
    body: DirectoryLookupRequest, request: Request
) -> DirectoryLookupResponse:
    """Resolve keys to their known placements across the fleet.

    Args:
        body: The keys to resolve.

    Returns:
        One result per requested key, in request order; placements are
        empty for unknown keys.
    """
    directory = get_context(request).key_directory
    keys = [encoded.to_object_key() for encoded in body.keys]
    return DirectoryLookupResponse(
        results=[
            DirectoryKeyPlacements(key=encoded, placements=placements)
            for encoded, placements in zip(
                body.keys, directory.lookup(keys), strict=True
            )
        ]
    )


@router.post("/directory/lookup_tokens")
async def lookup_placements_by_tokens(
    body: TokenPlacementLookupRequest, request: Request
) -> TokenPlacementLookupResponse:
    """Resolve a token sequence to keys and return their placements.

    Hashes ``token_ids`` with the fleet's token hasher, expands each
    complete chunk into its per-rank object keys, and looks each key up
    in the directory.

    Args:
        body: The token sequence and the key-resolution parameters.

    Returns:
        Chunk count plus one result per resolved key; empty when the
        sequence is shorter than one chunk.

    Raises:
        HTTPException: 400 when the token sequence exceeds the
            per-request cap or a key field is invalid.
    """
    ctx = get_context(request)
    try:
        obj_keys, chunks = resolve_object_keys(
            token_hasher=ctx.token_hasher,
            model_name=body.model_name,
            world_size=body.world_size,
            token_ids=body.token_ids,
            cache_salt=body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TokenPlacementLookupResponse(
        chunks=chunks,
        results=[
            DirectoryKeyPlacements(
                key=key.to_encoded_object_key(), placements=placements
            )
            for key, placements in zip(
                obj_keys, ctx.key_directory.lookup(obj_keys), strict=True
            )
        ],
    )


@router.get("/directory/keys")
async def list_directory_keys(
    request: Request,
    tier: Tier = Tier.ALL,
    instance_id: str = "",
    backend: str = "",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> DirectoryListResponse:
    """List directory keys and their placements, one page at a time.

    A snapshot for inspection: pages of a changing directory may skip
    or repeat keys.

    Args:
        request: The FastAPI request carrying the coordinator context.
        tier: Keep placements on this tier (``all`` keeps every tier).
        instance_id: Keep placements reported by this instance (empty
            keeps every instance).
        backend: Keep placements on this backend (empty keeps every
            backend).
        offset: Matching keys to skip.
        limit: Maximum keys to return.

    Returns:
        The number of keys matching the filters plus the requested page,
        each key with its matching placements and the number of token
        ids known for its chunk.
    """
    directory = get_context(request).key_directory

    def _scan() -> DirectoryListResponse:
        total, page = directory.list_keys(tier, instance_id, backend, offset, limit)
        token_ids = directory.get_token_ids([key.chunk_hash for key in page])
        return DirectoryListResponse(
            total=total,
            keys=[
                DirectoryKeyInfo(
                    key=key.to_encoded_object_key(),
                    placements=placements,
                    num_tokens=len(tokens),
                )
                for (key, placements), tokens in zip(
                    page.items(), token_ids, strict=True
                )
            ],
        )

    # ``total`` walks every matching record — keep the scan off the event loop.
    return await asyncio.to_thread(_scan)


@router.post("/directory/token_ids")
async def lookup_token_ids(body: TokenIdsRequest, request: Request) -> TokenIdsResponse:
    """Return the known token ids for each requested key.

    Args:
        body: The keys whose chunks' token ids to return.

    Returns:
        One result per requested key, in request order — ``token_ids``
        is empty for chunks the directory has no tokens for.
    """
    directory = get_context(request).key_directory
    chunk_hashes = [encoded.to_object_key().chunk_hash for encoded in body.keys]
    return TokenIdsResponse(
        results=[
            KeyTokenIds(key=encoded, token_ids=list(token_ids))
            for encoded, token_ids in zip(
                body.keys, directory.get_token_ids(chunk_hashes), strict=True
            )
        ]
    )


@router.get("/directory/stats")
async def directory_stats(request: Request) -> DirectoryStats:
    """Return a point-in-time summary of directory contents.

    Returns:
        Key/placement counts plus per-instance stream state (incarnation,
        last applied seq, gap flag), keyed by ``instance_id``.
    """
    return get_context(request).key_directory.stats()
