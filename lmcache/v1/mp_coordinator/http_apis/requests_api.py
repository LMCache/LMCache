# SPDX-License-Identifier: Apache-2.0
"""Request recovery endpoint on the coordinator.

``GET /requests/{instance_id}/{request_id}`` returns a request's tokens
and, per chunk, where its KV lives now: tokens from the
:class:`RequestIndex`, placements from the :class:`KeyDirectory`. See
``docs/design/v1/mp_coordinator/views/request_index.md``.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    RequestChunkInfo,
    RequestKeyPlacements,
    RequestTraceResponse,
)
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.views.request_index import RequestIndex

router = APIRouter()


@router.get("/requests/{instance_id}/{request_id}")
async def get_request(
    instance_id: str, request_id: str, request: Request
) -> RequestTraceResponse:
    """Return a request's recovered tokens and its chunks' placements.

    Args:
        instance_id: The MP server that reported the request.
        request_id: The request id on that server.
        request: The FastAPI request carrying the coordinator context.

    Returns:
        The request's tokens and chunks in sequence order. A key with no
        placements has been evicted; its tokens are still returned.

    Raises:
        HTTPException: 404 when the index holds no chunk of the request: it
            was never seen, aged out, or its tail chunk was never stored.
    """
    views = get_context(request).views
    trace = views.get(RequestIndex).get_request(instance_id, request_id)
    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=f"no retained chunk of request {request_id!r} on {instance_id!r}",
        )
    directory = views.get(KeyDirectory)
    chunks = []
    for chunk in trace.chunks:
        placements = directory.lookup(list(chunk.keys))
        chunks.append(
            RequestChunkInfo(
                chunk_hash_hex=chunk.chunk_hash.hex(),
                token_offset=chunk.token_offset,
                num_tokens=len(chunk.token_ids),
                keys=[
                    RequestKeyPlacements(
                        key=key.to_encoded_object_key(), placements=key_placements
                    )
                    for key, key_placements in zip(chunk.keys, placements, strict=True)
                ],
            )
        )
    return RequestTraceResponse(
        instance_id=instance_id,
        request_id=request_id,
        start_offset=trace.start_offset,
        complete=trace.complete,
        token_ids=trace.token_ids,
        chunks=chunks,
    )
