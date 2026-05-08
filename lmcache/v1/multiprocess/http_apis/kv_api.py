# SPDX-License-Identifier: Apache-2.0
"""HTTP routes for bytes-level KV cache access.

These routes expose ``MPCacheEngine.store_bytes / retrieve_bytes /
lookup_bytes`` to external developers over HTTP. They are control-plane
endpoints intended for cache priming, debugging, and future editing
workflows — not the inference hot path.

The wire format is the canonical KV_2LTD layout (``[2, num_layers,
num_tokens, hidden_dim]``) with all TP shards aggregated along the hidden
dim and all chunks concatenated along the token dim. See
``docs/design/v1/multiprocess/http_apis/kv_edit_api.md``.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.server import MPCacheEngine

logger = init_logger(__name__)

router = APIRouter()


_KV_CONTENT_TYPE = "application/x-lmcache-kv; v=1"

_HEADER_MODEL_NAME = "X-LMCache-Model-Name"
_HEADER_TOKENS = "X-LMCache-Tokens"
_HEADER_CACHE_SALT = "X-LMCache-Cache-Salt"
_HEADER_HIT_TOKENS = "X-LMCache-Hit-Tokens"
_HEADER_HIT_CHUNKS = "X-LMCache-Hit-Chunks"
_HEADER_TOTAL_TOKENS = "X-LMCache-Total-Tokens"
_HEADER_TOTAL_CHUNKS = "X-LMCache-Total-Chunks"


class _LookupOrRetrieveBody(BaseModel):
    """JSON body for /api/kv/retrieve and /api/kv/lookup.

    ``tokens`` is sent in the body rather than the URL because token
    sequences for real workloads are too long for query parameters.
    """

    model_name: str = Field(..., description="Registered model name.")
    tokens: list[int] = Field(..., description="Token sequence to address.")
    cache_salt: str = Field(
        default="",
        description="Optional per-namespace isolation salt; passed through "
        "to ObjectKey.cache_salt.",
    )


def _engine_or_503(request: Request) -> MPCacheEngine:
    """Resolve the MP engine off ``app.state`` or raise a 503 JSON error."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="engine not initialized",
        )
    return engine


def _parse_int_list(header_value: str) -> list[int]:
    """Parse a comma-separated list of ints from an HTTP header.

    Raises:
        ValueError: If any entry fails to parse as ``int``.
    """
    if not header_value:
        return []
    return [int(x) for x in header_value.split(",") if x]


@router.post("/api/kv/store")
async def store_kv(request: Request) -> dict[str, int | str]:
    """Store opaque KV cache bytes for a token sequence.

    The request body is the raw KV cache payload in the canonical KV_2LTD
    layout. Routing metadata (``model_name``, ``tokens``, optional
    ``cache_salt``) is carried in headers so that the body can be a single
    binary blob — multipart parsing for hundreds of MB of payload is
    unpleasant on both client and server.

    Headers:
        X-LMCache-Model-Name: required, must match a registered model.
        X-LMCache-Tokens: required, comma-separated token IDs.
        X-LMCache-Cache-Salt: optional namespace salt.
        Content-Type: should be ``application/x-lmcache-kv; v=1``.
    """
    engine = _engine_or_503(request)

    model_name = request.headers.get(_HEADER_MODEL_NAME)
    if not model_name:
        raise HTTPException(
            status_code=400, detail=f"missing {_HEADER_MODEL_NAME} header"
        )

    raw_tokens = request.headers.get(_HEADER_TOKENS, "")
    try:
        tokens = _parse_int_list(raw_tokens)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{_HEADER_TOKENS} must be a comma-separated list of ints",
        ) from exc
    if not tokens:
        raise HTTPException(
            status_code=400,
            detail=f"{_HEADER_TOKENS} header must contain at least one token",
        )

    cache_salt = request.headers.get(_HEADER_CACHE_SALT, "")
    payload = await request.body()

    try:
        result = engine.store_bytes(
            model_name=model_name,
            tokens=tokens,
            payload=payload,
            cache_salt=cache_salt,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"model_name {model_name!r} is not registered with this engine",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "total_tokens": result.total_tokens,
        "total_chunks": result.total_chunks,
        "stored_tokens": result.stored_tokens,
        "stored_chunks": result.stored_chunks,
    }


@router.post("/api/kv/retrieve")
async def retrieve_kv(body: _LookupOrRetrieveBody, request: Request) -> Response:
    """Retrieve KV cache bytes for the longest cached prefix.

    Returns the binary KV payload as the response body, with hit metadata
    in response headers. A 404 is returned with an empty body when nothing
    in the requested token sequence is cached.
    """
    engine = _engine_or_503(request)
    try:
        result = engine.retrieve_bytes(
            model_name=body.model_name,
            tokens=body.tokens,
            cache_salt=body.cache_salt,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"model_name {body.model_name!r} is not registered with this engine",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {
        _HEADER_HIT_TOKENS: str(result.hit_tokens),
        _HEADER_HIT_CHUNKS: str(result.hit_chunks),
        _HEADER_TOTAL_TOKENS: str(result.total_tokens),
        _HEADER_TOTAL_CHUNKS: str(result.total_chunks),
    }
    if result.hit_chunks == 0:
        return Response(
            status_code=404,
            content=b"",
            headers=headers,
            media_type=_KV_CONTENT_TYPE,
        )
    return Response(
        content=result.payload,
        headers=headers,
        media_type=_KV_CONTENT_TYPE,
    )


@router.post("/api/kv/lookup")
async def lookup_kv(body: _LookupOrRetrieveBody, request: Request) -> JSONResponse:
    """Probe how much of ``tokens`` is currently cached, without moving bytes.

    Useful for clients that want to decide whether to download a large
    KV payload, or to check the cached-prefix length after a store.
    """
    engine = _engine_or_503(request)
    try:
        result = engine.lookup_bytes(
            model_name=body.model_name,
            tokens=body.tokens,
            cache_salt=body.cache_salt,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"model_name {body.model_name!r} is not registered with this engine",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        {
            "total_tokens": result.total_tokens,
            "total_chunks": result.total_chunks,
            "hit_tokens": result.hit_tokens,
            "hit_chunks": result.hit_chunks,
        }
    )
