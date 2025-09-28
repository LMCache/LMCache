# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, List, Optional

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.delete("/cache/clear")
async def clear(
    request: Request,
    locations: Annotated[Optional[List[str]], Query()] = None,
    request_configs: Optional[dict] = None,
):
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        if not hasattr(lmcache_adapter, "lmcache_engine"):
            return PlainTextResponse(
                content="/cache/clear api only work for lmcache_engine", status_code=500
            )

        lmcache_engine = lmcache_adapter.lmcache_engine
        if not lmcache_engine:
            return PlainTextResponse(
                content="/cache/clear api only work for lmcache_engine", status_code=500
            )
        num_removed = lmcache_engine.clear(
            locations=locations, request_configs=request_configs
        )
        return PlainTextResponse(
            content=f"num_removed: {num_removed}",
        )
    except Exception as e:
        return PlainTextResponse(
            content=f"Error: Failed to clear cache - {str(e)}",
            status_code=500,
        )
