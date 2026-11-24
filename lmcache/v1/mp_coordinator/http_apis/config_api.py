# SPDX-License-Identifier: Apache-2.0
"""Fleet-wide coordinator configuration binding endpoints."""

# Third Party
from fastapi import APIRouter, HTTPException, Request

# First Party
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    ChunkSizeBindRequest,
    ChunkSizeBindResponse,
)

router = APIRouter()


@router.put("/config/chunk-size")
async def bind_chunk_size(
    body: ChunkSizeBindRequest, request: Request
) -> ChunkSizeBindResponse:
    """Bind chunk hashing and blend lookup to the MP fleet's final size."""
    try:
        chunk_size = get_context(request).bind_chunk_size(body.chunk_size)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ChunkSizeBindResponse(chunk_size=chunk_size)
