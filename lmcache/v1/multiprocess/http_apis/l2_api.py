# SPDX-License-Identifier: Apache-2.0
"""
HTTP endpoints for managing L2 KV cache keys.

Both endpoints target the **primary** L2 adapter — the first one
configured. The HTTP surface deliberately exposes no adapter selector:
a deployment that wants these endpoints to operate on a specific
adapter must list that adapter first in its L2 configuration.

- ``DELETE /l2`` — delete the KV cache for a caller-supplied list of
  keys. Idempotent: keys absent from the adapter are skipped silently;
  keys currently locked by in-flight store/load tasks (S3) are skipped
  so deletion never corrupts an active transfer.

- ``GET /l2/keys`` — paginate keys resident in the primary adapter,
  optionally filtered by ``model_name``. Returns 501 when the primary
  adapter does not implement listing (in v1 only ``S3L2Adapter`` does).

L1 is intentionally NOT touched by ``DELETE /l2`` — keys removed from
L2 may still return from L1 until natural L1 eviction expires them.
Callers that need an L1+L2 purge should layer their own L1 invalidation
or wait for the existing L1 eviction controller.
"""

# Standard
from dataclasses import dataclass
from typing import Any
import asyncio

# Third Party
from fastapi import APIRouter, HTTPException, Query, Request

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey

router = APIRouter()


_MAX_PAGE_SIZE = 5000
_DEFAULT_PAGE_SIZE = 500
# Hard cap on how many keys a single ``DELETE /l2`` request may
# target. Keeps the request body bounded and prevents a single call
# from monopolizing the adapter's I/O loop for an unbounded interval.
_MAX_DELETE_BATCH = 10_000


def _get_storage_manager(request: Request) -> Any:
    """Resolve the shared ``StorageManager`` or raise ``HTTPException``.

    Returns the live ``StorageManager`` instance. Raises
    ``HTTPException(503)`` when the engine isn't initialized yet,
    which FastAPI turns into a ``{"detail": "engine not initialized"}``
    JSON response.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return engine.storage_manager


# ---------------------------------------------------------------------------
# Wire schemas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeleteRequest:
    """Wire body for :py:func:`delete_l2`.

    FastAPI / Pydantic still validate per-item :class:`EncodedObjectKey`
    field types before the handler runs (missing fields, wrong type for
    ``kv_rank``, ...) — those surface as automatic 422s. The
    ``_MAX_DELETE_BATCH`` cap and the :class:`ObjectKey`-invariant
    checks (hex parse, ``@``-in-model-name, salt charset) run inside
    the handler and raise ``HTTPException(400)``.
    """

    keys: list[EncodedObjectKey]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _resolve_primary_l2(request: Request):
    """Look up the primary L2 adapter or raise the right ``HTTPException``.

    Wraps ``StorageManager.primary_l2()``: maps the "no L2 adapters
    configured" ``ValueError`` to HTTP 503 so handlers don't repeat
    the try/except boilerplate.

    Returns:
        ``(descriptor, adapter)``.
    """
    sm = _get_storage_manager(request)
    try:
        return sm.primary_l2()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.delete("/l2", response_model=None)
async def delete_l2(body: DeleteRequest, request: Request) -> dict[str, object]:
    """Delete a caller-supplied list of keys from the primary L2 adapter.

    Body schema: :class:`DeleteRequest` —
    ``{"keys": [EncodedObjectKey, ...]}``.

    Responses:
        200: ``{"requested": N, "adapter": "<type_name>", "ok": <bool>}``
            (with optional ``"error"`` field on ``ok=False``).
        400: batch exceeds ``_MAX_DELETE_BATCH`` OR a key's payload
            survived field-level typing but violates an ``ObjectKey``
            invariant (bad hex, ``@`` in ``model_name``, forbidden
            ``cache_salt`` char, ...).
        422: field-level validation failure (missing ``keys``, wrong
            types).
        503: engine not initialized OR no L2 adapters configured.
    """
    if len(body.keys) > _MAX_DELETE_BATCH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too many keys in a single request "
                f"(limit={_MAX_DELETE_BATCH}, got={len(body.keys)})"
            ),
        )

    parsed = []
    for i, cache_key in enumerate(body.keys):
        try:
            parsed.append(cache_key.to_object_key())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"keys[{i}]: {exc}") from None

    desc, adapter = _resolve_primary_l2(request)

    response: dict[str, object] = {
        "requested": len(parsed),
        "adapter": desc.type_name,
    }
    try:
        # ``adapter.delete`` is synchronous and blocks on adapter I/O
        # (the S3 adapter bounces the call through
        # ``run_coroutine_threadsafe(...).result(timeout=30.0)``).
        # Off-load to a worker thread so the FastAPI event loop stays
        # free for other requests.
        await asyncio.to_thread(adapter.delete, parsed)
    except Exception as exc:
        # Best-effort contract: adapter exceptions are reported in the
        # 200 body so operators see the failure cleanly, not as a 500
        # with a stack trace.
        response["ok"] = False
        response["error"] = str(exc)
        return response
    response["ok"] = True
    return response


@router.get("/l2/keys", response_model=None)
async def list_l2_keys(
    request: Request,
    model_name: str | None = Query(default=None),
    page_size: int = Query(default=_DEFAULT_PAGE_SIZE, ge=1, le=_MAX_PAGE_SIZE),
    page_token: str | None = Query(default=None),
) -> dict[str, object]:
    """List keys resident in the primary L2 adapter, paginated.

    Query parameters:
        model_name: restrict to one model name. Omit to return all.
        page_size: max entries per page. Clamped to ``[1, 5000]``;
            default ``500``.
        page_token: opaque cursor returned by the previous page. Omit
            on the first call. Pass back verbatim to get the next page.

    Responses:
        200: ``{"adapter": "<type>",
                "entries": [{"key": <EncodedObjectKey>, "size_bytes": N},
                            ...],
                "next_page_token": "<opaque>" | null}``.
        400: malformed page_token (adapter-level).
        501: primary adapter does not implement listing.
        503: engine not initialized OR no L2 adapters configured.
    """
    desc, adapter = _resolve_primary_l2(request)

    try:
        # Same rationale as ``delete_l2``: ``list_l2_keys`` is a
        # synchronous adapter call that issues blocking S3
        # ``ListObjectsV2`` requests, so off-load it to a worker
        # thread to keep the event loop responsive.
        page = await asyncio.to_thread(
            adapter.list_l2_keys,
            model_name=model_name,
            page_size=page_size,
            cursor=page_token,
        )
    except ValueError as exc:
        # Adapter-level validation failure (e.g. malformed page_token).
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"primary L2 adapter does not support listing: {exc}",
        ) from None

    return {
        "adapter": desc.type_name,
        "entries": page.entries,
        "next_page_token": page.next_page_token,
    }
