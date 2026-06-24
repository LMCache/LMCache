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

- ``POST /l2/prefetch`` — submit a warm prefetch of a token sequence's
  chunks from L2 into L1. The caller sends ``token_ids`` (not keys); the
  server hashes them, expands them across the node's ranks, submits a retain
  prefetch that loads them **unlocked** (no read lock), and returns a
  ``request_id`` immediately. The loaded chunks are retained (permanent) and
  immediately a normal, evictable L1 entry. Coalesces across all configured
  L2 adapters (no selector).

- ``GET /l2/prefetch/{request_id}`` — poll a submitted warm prefetch
  (``pending`` / ``completed``). Reactive: the caller drives completion;
  there is no server-side polling loop and nothing to release.

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
from lmcache.v1.distributed.api import EncodedObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.warm_prefetch import (
    COMPLETED,
    UNKNOWN,
    WarmPrefetchJobs,
)

router = APIRouter()


_MAX_PAGE_SIZE = 5000
_DEFAULT_PAGE_SIZE = 500
# Hard cap on how many keys a single ``DELETE /l2`` request may
# target. Keeps the request body bounded and prevents a single call
# from monopolizing the adapter's I/O loop for an unbounded interval.
_MAX_DELETE_BATCH = 10_000
# Hard cap on how many token ids a single ``POST /l2/prefetch`` request may
# carry. Keeps the request body bounded and the synchronous hashing /
# key-construction work in the handler proportionate.
_MAX_PREFETCH_TOKENS = 1_000_000


def _get_engine(request: Request) -> Any:
    """Resolve the shared engine (``MPCacheServer``) or raise ``HTTPException``.

    Raises ``HTTPException(503)`` when the engine isn't initialized yet.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    return engine


def _get_storage_manager(request: Request) -> Any:
    """Resolve the shared ``StorageManager`` or raise ``HTTPException``.

    Returns the live ``StorageManager`` instance. Raises
    ``HTTPException(503)`` when the engine isn't initialized yet,
    which FastAPI turns into a ``{"detail": "engine not initialized"}``
    JSON response.
    """
    return _get_engine(request).storage_manager


def _warm_jobs(request: Request) -> WarmPrefetchJobs:
    """Return the per-app warm-prefetch job table, created lazily.

    Holds the in-flight ``submit`` handles so a later status poll can report
    completion. The warm holds no lock, so there is nothing to release.
    """
    jobs = getattr(request.app.state, "warm_prefetch_jobs", None)
    if jobs is None:
        jobs = WarmPrefetchJobs()
        request.app.state.warm_prefetch_jobs = jobs
    return jobs


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


@dataclass(frozen=True)
class PrefetchRequest:
    """Wire body for :py:func:`prefetch_to_l1`.

    Callers describe content by ``token_ids`` -- the same unit the lookup
    path speaks -- never by internal cache keys, which they cannot construct
    (a key is a content hash plus a per-rank layout bitmap). The server hashes
    the tokens and expands them into the per-rank :class:`ObjectKey`s itself.

    ``model_name`` / ``world_size`` select the registered
    :class:`MemoryLayoutDesc` and the rank fan-out; ``cache_salt`` isolates
    the keys per tenant. FastAPI validates field types (422); the token cap
    and ``cache_salt`` invariants raise ``HTTPException(400)`` in the handler.
    """

    model_name: str
    world_size: int
    token_ids: list[int]
    cache_salt: str = ""


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


@router.post("/l2/prefetch", response_model=None, status_code=202)
async def prefetch_to_l1(
    body: PrefetchRequest,
    request: Request,
) -> dict[str, object]:
    """Submit a warm prefetch of a token sequence's chunks from L2 into L1.

    Hashes ``body.token_ids`` into chunk keys (the same path the lookup handler
    uses), expands them across the node's ranks, and submits a retain prefetch
    that loads them **unlocked** (no read lock). Returns immediately with a
    ``request_id``; the load runs in the storage manager's own thread. Poll
    ``GET /l2/prefetch/{request_id}`` to observe completion. The loaded chunks
    are resident, retained, and immediately re-lookupable/evictable -- there is
    no lock to release and no server-side polling loop. Coalesces across all
    configured L2 adapters, so there is no ``?adapter=`` selector (unlike
    ``DELETE /l2``).

    Body: ``{"model_name", "world_size", "token_ids": [int, ...], "cache_salt"}``.

    Responses:
        202: ``{"request_id", "chunks", "status": "submitted"}``, or
            ``{"chunks": 0, "status": "noop"}`` when the sequence is shorter
            than one chunk (nothing submitted; no ``request_id`` to poll).
        400: token count exceeds ``_MAX_PREFETCH_TOKENS`` or ``cache_salt``
            violates its invariants.
        409: no layout registered for ``(model_name, world_size)`` -- the
            model has not allocated KV cache on this node yet.
        422: request body fails field-level validation.
        503: engine not initialized.
    """
    if len(body.token_ids) > _MAX_PREFETCH_TOKENS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"too many token_ids in a single request "
                f"(limit={_MAX_PREFETCH_TOKENS}, got={len(body.token_ids)})"
            ),
        )

    ctx = _get_engine(request).context
    layout_desc = ctx.layout_desc_registry.find(body.model_name, body.world_size)
    if layout_desc is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"no layout registered for model_name={body.model_name!r} "
                f"world_size={body.world_size}; the model has not allocated "
                f"KV cache on this node yet"
            ),
        )

    # worker_id=None expands each chunk to one key per rank, warming the whole
    # node's L1 (mirrors how the lookup path fans a chunk across workers).
    try:
        ipc_key = IPCCacheServerKey(
            model_name=body.model_name,
            world_size=body.world_size,
            worker_id=None,
            token_ids=tuple(body.token_ids),
            start=0,
            end=len(body.token_ids),
            request_id="",
            cache_salt=body.cache_salt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    chunk_hashes = ctx.token_hasher.compute_chunk_hashes(list(body.token_ids))
    if not chunk_hashes:
        # Fewer than one full chunk of tokens -- nothing to warm.
        return {"chunks": 0, "status": "noop"}

    obj_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes, [0])[0]
    request_id = _warm_jobs(request).submit(ctx.storage_manager, obj_keys, layout_desc)
    return {
        "request_id": request_id,
        "chunks": len(chunk_hashes),
        "status": "submitted",
    }


@router.get("/l2/prefetch/{request_id}", response_model=None)
async def prefetch_status(request_id: str, request: Request) -> dict[str, object]:
    """Report a warm-prefetch job's status and finalize it on completion.

    The first poll that observes completion drops the job (and pops the
    controller's result bookkeeping), so a later poll for the same id returns
    404 (exactly-once). The warm holds no lock, so there is nothing to release.

    Responses:
        200: ``{"status": "pending"}`` while the load runs, or
            ``{"status": "completed", "found_keys", "total_keys"}`` once done.
        404: unknown ``request_id`` (already completed-and-consumed, or never
            submitted).
        503: engine not initialized.
    """
    status = _warm_jobs(request).poll(_get_storage_manager(request), request_id)
    if status.state == UNKNOWN:
        raise HTTPException(
            status_code=404,
            detail=(
                f"unknown prefetch request_id={request_id!r} "
                f"(already completed or never submitted)"
            ),
        )
    if status.state == COMPLETED:
        return {
            "status": COMPLETED,
            "found_keys": status.found_keys,
            "total_keys": status.total_keys,
        }
    return {"status": status.state}


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
