# SPDX-License-Identifier: Apache-2.0
"""Key-directory endpoints on the coordinator (fleet-level).

The ``/directory`` surface, thin over the :class:`KeyDirectory` carried
on the typed :class:`CoordinatorContext`: placement lookup, fragment
(blend) lookup, key/token-id listing, and stats — all read-only. The
directory is written by the cache-event stream, which arrives on
``POST /events`` (see :mod:`events_api`). See
``docs/design/v1/mp_coordinator/views/key_directory.md``.
"""

# Standard
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    BlendLookupRequest,
    BlendLookupResponse,
    BlendMatchModel,
    DirectoryKeyInfo,
    DirectoryKeyPlacements,
    DirectoryListResponse,
    DirectoryLookupRequest,
    DirectoryLookupResponse,
    decode_tokens,
)
from lmcache.v1.mp_coordinator.views.key_directory import DirectoryStats, KeyDirectory
from lmcache.v1.multiprocess.cache_control.key_resolver import resolve_object_keys

router = APIRouter()


@router.post("/directory/lookup")
async def lookup_placements(
    body: DirectoryLookupRequest, request: Request
) -> DirectoryLookupResponse:
    """Resolve keys — or a request's token sequence — to placements.

    The tokens form hashes ``token_ids`` with the fleet's token hasher
    and expands each complete chunk into its per-rank object keys.
    Chunk hashes are prefix-chained, so ``token_ids`` must be the
    request's full sequence from position 0; trailing incomplete chunks
    are ignored.

    Args:
        body: Either the keys to resolve or the token sequence and its
            key-resolution parameters.

    Returns:
        Chunk count plus one result per resolved key, in request order,
        each with its known placements and the chunk's token ids
        (both empty when the directory knows nothing about the key).

    Raises:
        HTTPException: 400 when the token sequence exceeds the
            per-request cap or a key field is invalid.
    """
    ctx = get_context(request)
    if body.keys:
        encoded_keys = list(body.keys)
        obj_keys = [encoded.to_object_key() for encoded in encoded_keys]
        chunks = len(obj_keys)
    else:
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
        encoded_keys = [key.to_encoded_object_key() for key in obj_keys]
    directory = ctx.views.get(KeyDirectory)
    placements = directory.lookup(obj_keys)
    token_ids = directory.get_token_ids([key.chunk_hash for key in obj_keys])
    return DirectoryLookupResponse(
        chunks=chunks,
        results=[
            DirectoryKeyPlacements(
                key=encoded, placements=key_placements, token_ids=list(tokens)
            )
            for encoded, key_placements, tokens in zip(
                encoded_keys, placements, token_ids, strict=True
            )
        ],
    )


@router.post("/directory/blend-lookup")
async def blend_lookup(
    body: BlendLookupRequest, request: Request
) -> BlendLookupResponse:
    """Find cached chunk content anywhere inside a query sequence.

    The fragment counterpart to ``/directory/lookup``: the query need not
    be a prefix, and each match reports both where the content sits in
    the query and where it sat when stored, so the caller can re-RoPE it.

    Args:
        body: The query tokens.
        request: The FastAPI request carrying the coordinator context.

    Returns:
        Matched chunks, ascending by query position.
    """
    directory = get_context(request).views.get(KeyDirectory)
    tokens = decode_tokens(body.tokens_b64)

    def _match() -> BlendLookupResponse:
        """Run the fragment match and shape it for the wire."""
        return BlendLookupResponse(
            matches=[
                BlendMatchModel(
                    chunk_hash=match.chunk_hash.hex(),
                    old_st=match.old_st,
                    cur_st=match.cur_st,
                )
                for match in directory.blend_match(tokens)
            ]
        )

    # The rolling hash walks the whole query — keep it off the event loop.
    return await asyncio.to_thread(_match)


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
    directory = get_context(request).views.get(KeyDirectory)

    def _scan() -> DirectoryListResponse:
        """Page the directory and shape the rows for the wire."""
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


@router.get("/directory/stats")
async def directory_stats(request: Request) -> DirectoryStats:
    """Return a point-in-time summary of directory contents.

    Returns:
        Key/placement counts, per-instance L1 key counts, and the
        blend-index counts. Per-emitter stream state (incarnation,
        applied seq, gap flag) lives on the ingest gate and is not
        exposed yet.
    """
    return get_context(request).views.get(KeyDirectory).stats()
