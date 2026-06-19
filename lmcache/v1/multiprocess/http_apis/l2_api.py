# SPDX-License-Identifier: Apache-2.0
"""
HTTP endpoints for managing L2 KV cache keys.

Three endpoints:

- ``GET /l2/adapters`` — enumerate every configured L2 adapter with
  its ``type_name`` (and whether it's the primary). Operators read this
  to know which value to pass as ``?adapter=`` on the other endpoints.

- ``DELETE /l2`` — delete the KV cache for a caller-supplied list of
  keys. Idempotent: keys absent from the adapter are skipped silently;
  keys currently locked by in-flight store/load tasks (S3) are skipped
  so deletion never corrupts an active transfer.

- ``GET /l2/keys`` — paginate keys resident in the selected adapter,
  optionally filtered by ``model_name``. Returns 501 when the selected
  adapter does not implement listing (in v1 only ``S3L2Adapter`` does).

Both ``DELETE /l2`` and ``GET /l2/keys`` accept an optional
``?adapter=<type_name>`` query parameter to target a specific adapter.
Omitting the selector defaults to the **primary** (first-configured)
adapter, preserving the v1 behavior. When multiple adapters share a
``type_name``, the first match wins.

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
    """Wire body for :py:func:`delete_from_l2`.

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


def _resolve_l2_adapter(request: Request, selector: str | None):
    """Resolve the ``(descriptor, adapter)`` pair the handler targets.

    ``selector=None`` picks the primary adapter; a string picks the
    first adapter whose ``type_name`` matches.

    Raises:
        HTTPException: 503 if no L2 adapters are configured; 404 if
            ``selector`` doesn't match any.
    """
    adapters = _get_storage_manager(request).l2_adapters()
    if not adapters:
        raise HTTPException(status_code=503, detail="no L2 adapters configured")
    if selector is None:
        return adapters[0]
    for desc, adapter in adapters:
        if desc.type_name == selector:
            return desc, adapter
    raise HTTPException(
        status_code=404,
        detail=f"no L2 adapter with type_name={selector!r}",
    )


@router.get("/l2/adapters", response_model=None)
async def list_adapters(request: Request) -> dict[str, object]:
    """Enumerate configured L2 adapters.

    Responses:
        200: ``{"adapters": [{"index", "type_name", "primary"}, ...]}``.
            Empty list when no L2 backends are configured.
        503: engine not initialized.
    """
    adapters = _get_storage_manager(request).l2_adapters()
    return {
        "adapters": [
            {"index": i, "type_name": desc.type_name, "primary": i == 0}
            for i, (desc, _) in enumerate(adapters)
        ]
    }


@router.delete("/l2", response_model=None)
async def delete_from_l2(
    body: DeleteRequest,
    request: Request,
    adapter: str | None = Query(default=None, alias="adapter"),
) -> dict[str, object]:
    """Delete a caller-supplied list of keys from one L2 adapter.

    Body: ``{"keys": [EncodedObjectKey, ...]}``.

    Query parameters:
        adapter: ``type_name`` of the target adapter (see
            ``GET /l2/adapters``). Omit to target the primary.

    Responses:
        200: ``{"requested", "adapter", "ok"[, "error"]}``.
        400: batch exceeds ``_MAX_DELETE_BATCH`` or a key violates
            ``ObjectKey`` invariants.
        404: ``?adapter=<name>`` does not match any adapter.
        422: request body fails field-level validation.
        503: engine not initialized, or no L2 adapters configured.
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

    desc, target_adapter = _resolve_l2_adapter(request, selector=adapter)

    response: dict[str, object] = {
        "requested": len(parsed),
        "adapter": desc.type_name,
    }
    try:
        # ``adapter.delete`` is sync-blocking; off-load to a worker
        # thread to keep the event loop free for other requests.
        await asyncio.to_thread(target_adapter.delete, parsed)
    except Exception as exc:
        # Adapter exceptions are surfaced in the 200 body so operators
        # see a structured failure, not a 500.
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
    adapter: str | None = Query(default=None, alias="adapter"),
) -> dict[str, object]:
    """List keys resident in one L2 adapter, paginated.

    Query parameters:
        model_name: restrict to one model name. Omit to return all.
        page_size: max entries per page. Clamped to ``[1, 5000]``.
        page_token: opaque cursor from the previous page; omit on the
            first call.
        adapter: ``type_name`` of the target adapter (see
            ``GET /l2/adapters``). Omit to target the primary.

    Responses:
        200: ``{"adapter", "entries", "next_page_token"}``.
        400: malformed ``page_token``.
        404: ``?adapter=<name>`` does not match any adapter.
        501: selected adapter does not implement listing.
        503: engine not initialized, or no L2 adapters configured.
    """
    desc, target_adapter = _resolve_l2_adapter(request, selector=adapter)

    try:
        # Sync-blocking; off-load to a worker thread.
        page = await asyncio.to_thread(
            target_adapter.list_l2_keys,
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
            detail=f"L2 adapter {desc.type_name!r} does not support listing: {exc}",
        ) from None

    return {
        "adapter": desc.type_name,
        "entries": page.entries,
        "next_page_token": page.next_page_token,
    }
