# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, List, Optional
import json

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
    """Clear cached data from the LMCache engine.

    This endpoint provides a way to clear cached KV (Key-Value) data from the
    LMCache engine. It can clear all cached data or selectively clear data
    from specific storage locations.

    Args:
        request (Request): The FastAPI request object containing application state.
        locations (Optional[List[str]], optional): List of storage backend locations
            to clear cache from. If None, clears from all available locations.
            Common values include ["LocalCPUBackend", "LocalDiskBackend"].
            Defaults to None.
        request_configs (Optional[dict], optional): Additional configuration
            parameters for the clear operation. Currently unused but reserved
            for future extensions. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response

    Example:
        Clear all cached data:
        ```bash
        curl -X DELETE "http://localhost:8000/cache/clear"
        # Response: {"status": "success", "num_removed": 10,
        #           "locations": null, "request_configs": null}
        ```

        Clear cache from specific locations:
        ```bash
        curl -X DELETE "http://localhost:8000/cache/clear?locations=LocalCPUBackend&locations=LocalDiskBackend"
        # Response: {"status": "success", "num_removed": 5,
        #           "locations": ["LocalCPUBackend", "LocalDiskBackend"],
        #           "request_configs": null}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/cache/clear API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )
        num_removed = lmcache_engine.clear(
            locations=locations, request_configs=request_configs
        )
        success_info = {
            "status": "success",
            "num_removed": num_removed,
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_info = {"error": "Failed to clear cache", "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
