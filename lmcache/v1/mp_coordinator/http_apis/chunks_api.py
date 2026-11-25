# SPDX-License-Identifier: Apache-2.0
"""Chunk → requests endpoint on the coordinator.

``GET /chunks/{chunk_hash_hex}/requests`` lists the tracked requests whose
sequence contains a chunk, from the :class:`RequestIndex`. See
``docs/design/v1/mp_coordinator/views/request_index.md``.
"""

# Standard
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import ChunkRequestInfo, ChunkRequestsResponse
from lmcache.v1.mp_coordinator.views.request_index import RequestIndex

router = APIRouter()


@router.get("/chunks/{chunk_hash_hex}/requests")
async def list_chunk_requests(
    chunk_hash_hex: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=10000),
) -> ChunkRequestsResponse:
    """List the requests whose sequence contains a chunk, one page at a time.

    Args:
        chunk_hash_hex: Hex ``ObjectKey.chunk_hash`` of the chunk.
        request: The FastAPI request carrying the coordinator context.
        offset: Requests to skip.
        limit: Maximum requests to return.

    Returns:
        The number of such requests and the requested page, sorted by
        ``(instance_id, request_id)``.

    Raises:
        HTTPException: 400 when ``chunk_hash_hex`` is not hex; 404 when the
            index does not hold the chunk.
    """
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    index = get_context(request).views.get(RequestIndex)
    # The walk visits every chunk built on this one -- keep it off the loop.
    refs = await asyncio.to_thread(index.get_chunk_requests, chunk_hash)
    if refs is None:
        raise HTTPException(
            status_code=404, detail=f"chunk {chunk_hash_hex} is not in the index"
        )
    return ChunkRequestsResponse(
        chunk_hash_hex=chunk_hash.hex(),
        total=len(refs),
        requests=[
            ChunkRequestInfo(
                instance_id=ref.instance_id,
                request_id=ref.request_id,
                model_name=ref.namespace.model_name,
                cache_salt=ref.namespace.cache_salt,
            )
            for ref in refs[offset : offset + limit]
        ],
    )
