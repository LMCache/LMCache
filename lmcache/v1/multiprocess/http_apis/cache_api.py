# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

router = APIRouter()


@router.post("/api/clear-cache")
async def clear_cache(request: Request) -> Any:
    """
    Force-clear all KV cache data stored in L1 (CPU) memory.

    This clears all objects including those with active
    read/write locks. In-flight store or prefetch operations
    may be corrupted.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "reason": "engine not initialized",
            },
        )

    engine.clear()
    logger.info("Cache cleared via HTTP API")
    return {"status": "ok"}
