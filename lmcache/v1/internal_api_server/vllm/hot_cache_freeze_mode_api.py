# SPDX-License-Identifier: Apache-2.0
# Standard
import json

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

router = APIRouter()


@router.put("/hot_cache_freeze_mode/enable")
async def enable_hot_cache_freeze_mode(request: Request):
    """
    Enable hot cache freeze mode for the LMCache engine.

    When hot cache freeze mode is enabled:
    - All store operations will be skipped (no new data stored)
    - Only local_cpu backend will be used for retrieval
    - No admit/evict messages will be generated
    This protects the local_cpu hot cache from changes.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response indicating the operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/hot_cache_freeze_mode/enable"
        # Response: {"status": "success", "hot_cache_freeze_mode": true}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/hot_cache_freeze_mode/enable API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        lmcache_engine.set_hot_cache_freeze_mode(True)
        success_info = {
            "status": "success",
            "hot_cache_freeze_mode": True,
            "message": "Hot cache freeze mode enabled successfully",
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to enable hot cache freeze mode"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.put("/hot_cache_freeze_mode/disable")
async def disable_hot_cache_freeze_mode(request: Request):
    """
    Disable hot cache freeze mode for the LMCache engine.

    When hot cache freeze mode is disabled, store operations will proceed normally.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A JSON response indicating the operation status.

    Example:
        ```bash
        curl -X PUT "http://localhost:8000/hot_cache_freeze_mode/disable"
        # Response: {"status": "success", "hot_cache_freeze_mode": false}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/hot_cache_freeze_mode/disable API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        lmcache_engine.set_hot_cache_freeze_mode(False)
        success_info = {
            "status": "success",
            "hot_cache_freeze_mode": False,
            "message": "Hot cache freeze mode disabled successfully",
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to disable hot cache freeze mode"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.get("/hot_cache_freeze_mode/status")
async def get_hot_cache_freeze_mode_status(request: Request):
    """
    Get the current hot cache freeze mode status of the LMCache engine.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: JSON response with current hot cache freeze mode status.

    Example:
        ```bash
        curl -X GET "http://localhost:8000/hot_cache_freeze_mode/status"
        # Response: {"status": "success", "hot_cache_freeze_mode": true}
        ```
    """
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        if not lmcache_engine:
            error_info = {
                "error": "/hot_cache_freeze_mode/status API is unavailable",
                "message": "LMCache engine not configured.",
            }
            return PlainTextResponse(
                content=json.dumps(error_info, indent=2),
                media_type="application/json",
                status_code=503,  # Service Unavailable
            )

        hot_cache_freeze_mode = lmcache_engine.get_hot_cache_freeze_mode()
        mode_str = "enabled" if hot_cache_freeze_mode else "disabled"
        success_info = {
            "status": "success",
            "hot_cache_freeze_mode": hot_cache_freeze_mode,
            "message": "Hot cache freeze mode is " + mode_str,
        }
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        error_msg = "Failed to get hot cache freeze mode status"
        error_info = {"error": error_msg, "message": str(e)}
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
