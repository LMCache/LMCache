# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, Callable, List, Optional, Tuple
import json
import traceback

# Third Party
from fastapi import APIRouter, Query
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine

logger = init_logger(__name__)

router = APIRouter()


def _parse_tokens_from_params(
    tokens_mock: Optional[str],
) -> Tuple[Optional[List[int]], Optional[dict]]:
    """Parse tokens from input parameters.

    Args:
        tokens_mock: Two comma-separated numbers specifying start and end of token range
            - Example: "0,100" generates tokens [0, 1, 2, ..., 99]
            - Example: "50,75" generates tokens [50, 51, 52, ..., 74]

    Returns:
        Tuple of (tokens list, error dict).
        If error dict is not None, tokens will be None.
    """
    # TODO(baoloongmao): Add support for tokens_input parameter to read tokens from file
    if tokens_mock:
        try:
            parts = tokens_mock.split(",")
            if len(parts) != 2:
                raise ValueError("tokens_mock must contain exactly 2 numbers")
            start, end = int(parts[0].strip()), int(parts[1].strip())
            if start >= end:
                raise ValueError("start must be less than end")
            tokens = list(range(start, end))
            return tokens, None
        except ValueError as e:
            return None, {
                "error": "Invalid tokens_mock format",
                "message": f"tokens_mock must be 'start,end': {str(e)}",
            }
    else:
        return None, {
            "error": "Missing parameters",
            "message": "Must specify either tokens_input or tokens_mock",
        }


def _create_error_response(error_info: dict, status_code: int) -> PlainTextResponse:
    """Create a standardized error response.

    Args:
        error_info: Dictionary containing error information
        status_code: HTTP status code

    Returns:
        PlainTextResponse with error information
    """
    return PlainTextResponse(
        content=json.dumps(error_info, indent=2),
        media_type="application/json",
        status_code=status_code,
    )


def _check_lmcache_engine(
    request: Request,
) -> Tuple[Optional["LMCacheEngine"], Optional[PlainTextResponse]]:
    """Check if LMCache engine is available.

    Args:
        request: FastAPI request object

    Returns:
        Tuple of (lmcache_engine, error_response).
        If error_response is not None, engine will be None.
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
    if not lmcache_engine:
        error_info = {
            "error": "LMCache API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return None, _create_error_response(error_info, 503)
    return lmcache_engine, None


def _get_kvcaches_and_device(engine):
    """Get kvcaches and device from engine's gpu_connector.

    Args:
        engine: LMCache engine instance

    Returns:
        Tuple of (kvcaches, device).
        kvcaches may be None if not available.
        device defaults to "cpu" if kvcaches not available.
    """
    kvcaches = None
    device = "cpu"  # Default device

    if engine.gpu_connector:
        kvcaches = engine.gpu_connector.kvcaches
        if kvcaches is not None and len(kvcaches) > 0:
            device = kvcaches[0].device
            logger.debug(f"Using kvcaches device: {device}")
        else:
            logger.warning(
                "gpu_connector.kvcaches is None or empty. "
                "Make sure post_init was called with kvcaches."
            )

    return kvcaches, device


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
        lmcache_engine, error_response = _check_lmcache_engine(request)
        if error_response:
            return error_response

        assert lmcache_engine is not None
        num_removed = lmcache_engine.clear(  # type: ignore[attr-defined]
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
        return _create_error_response(error_info, 500)


def _process_tokens_request(
    request: Request,
    tokens_mock: Optional[str],
) -> Tuple[Optional[object], Optional[List[int]], Optional[PlainTextResponse]]:
    """Process tokens request and validate parameters.

    Args:
        request: FastAPI request object
        tokens_mock: Mock token range specification

    Returns:
        Tuple of (lmcache_engine, tokens, error_response).
        If error_response is not None, the other values will be None.
    """
    lmcache_engine, error_response = _check_lmcache_engine(request)
    if error_response:
        return None, None, error_response

    tokens, error_info = _parse_tokens_from_params(tokens_mock)
    if error_info:
        status_code = 400 if error_info["error"] != "File not found" else 404
        return None, None, _create_error_response(error_info, status_code)

    return lmcache_engine, tokens, None


def _execute_cache_operation(
    operation_name: str,
    operation_func: Callable,
    lmcache_engine: object,
    tokens: List[int],
) -> PlainTextResponse:
    """Execute a cache operation and return standardized response.

    Args:
        operation_name: Name of the operation for error messages
        operation_func: Function to execute the operation
        lmcache_engine: LMCache engine instance
        tokens: List of token IDs

    Returns:
        PlainTextResponse with operation result
    """
    try:
        result = operation_func(lmcache_engine, tokens)
        success_info = {
            "status": "success",
            "num_tokens": len(tokens),
        }
        if result is not None:
            success_info.update(result)
        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )
    except Exception as e:
        # Log the full traceback for debugging
        tb_str = traceback.format_exc()
        logger.error(f"Failed to {operation_name}: {str(e)}\\n{tb_str}")

        # Include more detailed error info in response
        error_message = str(e) if str(e) else f"Exception type: {type(e).__name__}"
        error_info = {
            "error": f"Failed to {operation_name}",
            "message": error_message,
            "exception_type": type(e).__name__,
        }
        return _create_error_response(error_info, 500)


@router.post("/cache/store")
async def store(
    request: Request,
    tokens_mock: Optional[str] = None,
):
    """Store KV cache data into the LMCache engine.

    This endpoint provides a way to store KV cache data by generating mock tokens.

    Args:
        request (Request): The FastAPI request object containing application state.
        tokens_mock (Optional[str], optional): Two comma-separated numbers specifying
            the start and end of a token range. Example: "0,100" generates tokens
            from 0 to 99. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response with operation status

    Example:
        Store with mock tokens:
        ```bash
        curl -X POST "http://localhost:8000/cache/store?tokens_mock=0,100"
        # Response: {"status": "success", "num_tokens": 100}
        ```
    """
    lmcache_engine, tokens, error_response = _process_tokens_request(
        request, tokens_mock
    )
    if error_response:
        return error_response

    assert tokens is not None
    assert lmcache_engine is not None

    def _store_operation(engine, token_list):
        # Get kvcaches and device using the shared function
        kvcaches, device = _get_kvcaches_and_device(engine)

        # Create slot mapping for the tokens
        slot_mapping = torch.arange(len(token_list), dtype=torch.long, device=device)

        logger.debug(
            f"Storing {len(token_list)} tokens with slot_mapping on device {device}"
        )

        engine.store(
            tokens=token_list,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )
        return None

    return _execute_cache_operation(
        "store cache", _store_operation, lmcache_engine, tokens
    )


@router.post("/cache/retrieve")
async def retrieve(
    request: Request,
    tokens_mock: Optional[str] = None,
):
    """Retrieve KV cache data from the LMCache engine.

    This endpoint provides a way to retrieve KV cache data by generating mock tokens.

    Args:
        request (Request): The FastAPI request object containing application state.
        tokens_mock (Optional[str], optional): Two comma-separated numbers specifying
            the start and end of a token range. Example: "0,100" generates tokens
            from 0 to 99. Defaults to None.

    Returns:
        PlainTextResponse: A plain text response with operation status

    Example:
        Retrieve with mock tokens:
        ```bash
        curl -X POST "http://localhost:8000/cache/retrieve?tokens_mock=0,100"
        # Response: {"status": "success", "num_tokens": 100, "num_retrieved": 80}
        ```
    """
    lmcache_engine, tokens, error_response = _process_tokens_request(
        request, tokens_mock
    )
    if error_response:
        return error_response

    assert tokens is not None
    assert lmcache_engine is not None

    def _retrieve_operation(engine, token_list):
        # Get kvcaches and device using the shared function
        kvcaches, device = _get_kvcaches_and_device(engine)

        # Create slot_mapping for retrieve operation
        slot_mapping = torch.arange(len(token_list), dtype=torch.long, device=device)

        logger.debug(
            f"Retrieving {len(token_list)} tokens with slot_mapping on device {device}"
        )

        ret_mask = engine.retrieve(
            tokens=token_list,
            slot_mapping=slot_mapping,
            kvcaches=kvcaches,
        )
        num_retrieved = int(ret_mask.sum().item())
        return {"num_retrieved": num_retrieved}

    return _execute_cache_operation(
        "retrieve cache", _retrieve_operation, lmcache_engine, tokens
    )
