# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple, Union
import json
import time

# Third Party
from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)

router = APIRouter()


def _create_json_response(data: dict, status_code: int = 200) -> PlainTextResponse:
    """Create a JSON response with consistent formatting."""
    return PlainTextResponse(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _get_lookup_client(
    request: Request,
) -> Tuple[
    Optional[Union[LookupClientInterface, ChunkStatisticsLookupClient]],
    Optional[PlainTextResponse],
]:
    """Get lookup client from request or return error response."""
    lmcache_adapter = request.app.state.lmcache_adapter
    lookup_client = getattr(lmcache_adapter, "lookup_client", None)

    if not lookup_client:
        error_response = _create_json_response(
            {
                "error": "Chunk statistics API unavailable",
                "message": "Lookup client not configured.",
            },
            status_code=503,
        )
        return None, error_response

    return lookup_client, None


def _get_statistics_client(
    request: Request,
) -> Tuple[Optional[ChunkStatisticsLookupClient], Optional[PlainTextResponse]]:
    """Get ChunkStatisticsLookupClient or return error response."""
    lookup_client, error_response = _get_lookup_client(request)
    if error_response:
        return None, error_response

    if not isinstance(lookup_client, ChunkStatisticsLookupClient):
        error_response = _create_json_response(
            {
                "error": "Chunk statistics not available",
                "message": "Current lookup client does not support statistics.",
            },
            status_code=400,
        )
        return None, error_response

    return lookup_client, None


def _handle_exception(operation: str, error: Exception) -> PlainTextResponse:
    """Create error response for exceptions."""
    return _create_json_response(
        {
            "error": f"Failed to {operation}",
            "message": str(error),
        },
        status_code=500,
    )


@router.post("/chunk_statistics/start")
async def start_chunk_statistics(request: Request):
    """Start chunk statistics collection.

        This endpoint enables chunk statistics tracking in the LMCache system.
        It allows monitoring chunk reuse patterns and collecting metrics.

        Args:
            request (Request): The FastAPI request object containing application state.

        Returns:
            PlainTextResponse: A plain text response with operation status.

        Example:
    ```bash
            curl -X POST "http://localhost:8000/chunk_statistics/start"
            # Response: {"status": "success", "message": "Chunk statistics "
            #          "collection started"}
            ```
    """
    try:
        lookup_client, error_response = _get_lookup_client(request)
        if error_response:
            return error_response

        assert lookup_client is not None

        if isinstance(lookup_client, ChunkStatisticsLookupClient):
            lookup_client.start_statistics()
            message = "Chunk statistics collection started"
        else:
            return _create_json_response(
                {
                    "error": "Chunk statistics not available",
                    "message": "Current lookup client does not support statistics.",
                },
                status_code=400,
            )

        return _create_json_response({"status": "success", "message": message})
    except Exception as e:
        return _handle_exception("start chunk statistics", e)


@router.post("/chunk_statistics/stop")
async def stop_chunk_statistics(request: Request):
    """Stop chunk statistics collection.

        This endpoint disables chunk statistics tracking in the LMCache system.

        Args:
            request (Request): The FastAPI request object containing application state.

        Returns:
            PlainTextResponse: A plain text response with operation status.

        Example:
    ```bash
            curl -X POST "http://localhost:8000/chunk_statistics/stop"
            # Response: {"status": "success", "message": "Chunk statistics "
            #          "collection stopped"}
            ```
    """
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response

        assert stats_client is not None
        stats_client.stop_statistics()
        return _create_json_response(
            {
                "status": "success",
                "message": "Chunk statistics collection stopped",
            }
        )
    except Exception as e:
        return _handle_exception("stop chunk statistics", e)


@router.post("/chunk_statistics/reset")
async def reset_chunk_statistics(request: Request):
    """Reset chunk statistics.

    This endpoint resets all collected chunk statistics to initial state.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A plain text response with operation status.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/chunk_statistics/reset"
        # Response: {"status": "success", "message": "Chunk statistics reset"}
        ```
    """
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response

        assert stats_client is not None
        stats_client.reset_statistics()
        return _create_json_response(
            {"status": "success", "message": "Chunk statistics reset"}
        )
    except Exception as e:
        return _handle_exception("reset chunk statistics", e)


@router.get("/chunk_statistics/status")
async def get_chunk_statistics_status(request: Request):
    """Get current chunk statistics status.

    This endpoint returns the current status of chunk statistics collection
    including all collected metrics.

    Args:
        request (Request): The FastAPI request object containing application state.

    Returns:
        PlainTextResponse: A plain text response with statistics information.

    Example:
        ```bash
        curl "http://localhost:8000/chunk_statistics/status"
        # Response: {
        #   "enabled": true,
        #   "total_requests": 100,
        #   "total_chunks": 500,
        #   "unique_chunks": 400,
        #   "duplicate_chunks": 100,
        #   "reuse_rate": 0.2,
        #   "bloom_filter": {...}
        # }
        ```
    """
    try:
        stats_client, error_response = _get_statistics_client(request)
        if error_response:
            return error_response

        assert stats_client is not None
        config = request.app.state.lmcache_adapter.config
        stats = stats_client.get_statistics()

        runtime_info = {
            "timestamp": time.time(),
            "auto_exit_enabled": (
                config.chunk_statistics_auto_exit_timeout_hours > 0.0
                or config.chunk_statistics_auto_exit_target_unique_chunks is not None
            ),
            "auto_exit_timeout_hours": config.chunk_statistics_auto_exit_timeout_hours,
            "auto_exit_target_unique_chunks": (
                config.chunk_statistics_auto_exit_target_unique_chunks
            ),
        }

        stats.update(runtime_info)
        return _create_json_response(stats)
    except Exception as e:
        return _handle_exception("get chunk statistics status", e)
