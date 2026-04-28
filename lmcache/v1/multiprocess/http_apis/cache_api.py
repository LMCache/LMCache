# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Optional
import asyncio
import hashlib

# Third Party
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    compress_slot_mapping,
    parse_mixed_slot_mapping,
)
from lmcache.v1.utils.kv_slot_ops import (
    extract_kv_at_slots,
    slice_by_slot_dim,
)

logger = init_logger(__name__)

router = APIRouter()


@router.post("/clear-cache")
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


@router.get("/api/kvcache/check")
async def kvcache_check(
    request: Request,
    slot_mapping: Optional[str] = None,
    chunk_size: Optional[int] = None,
    instance_id: int = 0,
    layerwise: bool = False,
) -> JSONResponse:
    """Compute MD5 checksums for KV cache slots.

    Groups the referenced slot indices into ``chunk_size``-sized
    chunks and returns an MD5 digest per chunk (optionally per
    layer). Intended for diagnostics and round-trip integrity
    verification, e.g. from the ``lmcache bench kvcache``
    command.

    Args:
        request: FastAPI request (used to fetch the attached
            engine / GPU contexts from ``app.state``).
        slot_mapping: Slot indices in mixed format (comma- and
            dash-separated, e.g. ``"0-15,32,48-63"``). Required.
        chunk_size: Positive integer — slots are grouped into
            contiguous chunks of this size before hashing.
            Required.
        instance_id: GPU context ID to look up on the engine
            (default 0).
        layerwise: If True, return per-layer checksums under
            each chunk; otherwise a single aggregated checksum
            per chunk.

    Returns:
        :class:`fastapi.responses.JSONResponse` whose body on
        success contains::

            {
                "status": "success",
                "chunk_size": <int>,
                "num_chunks": <int>,
                "chunk_checksums": <see below>,
                "layerwise": <bool>,
                "slot_mapping_ranges": "<compressed ranges>"
            }

        The type of ``chunk_checksums`` depends on the
        ``layerwise`` query parameter — mirroring the vLLM
        internal API contract:

        * ``layerwise=false`` (default): ``list[str]`` with one
          aggregated MD5 digest per chunk. Callers may safely
          ``"".join`` the list.
        * ``layerwise=true``: ``dict[str, list[str]]`` keyed by
          ``"layer_<idx>"``; each value is the per-chunk digest
          list for that layer. Callers that expect a flat list
          **must** branch on ``layerwise`` before consuming.

        Error responses carry ``{"error": "..."}`` with one of
        the HTTP status codes listed below.

    HTTP status codes:
        200: Success; body holds the checksum result.
        400: ``slot_mapping`` missing / malformed, or
            ``chunk_size`` is missing or non-positive.
        404: ``instance_id`` is not registered on the engine,
            or the engine has no KV tensors allocated.
        501: The attached engine type does not expose GPU
            contexts (checksum endpoint unsupported).
        503: Engine is not yet initialised on ``app.state``.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"error": "engine not initialized"},
        )

    gpu_ctxs = getattr(engine, "gpu_contexts", None)
    if gpu_ctxs is None:
        return JSONResponse(
            status_code=501,
            content={
                "error": "checksum not supported for this engine type",
            },
        )

    ctx = gpu_ctxs.get(instance_id)
    if ctx is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "instance_id %d not registered" % instance_id,
            },
        )

    if not slot_mapping:
        return JSONResponse(
            status_code=400,
            content={"error": "slot_mapping is required"},
        )

    slot_indices, error_info = parse_mixed_slot_mapping(
        slot_mapping,
    )
    if error_info:
        logger.warning(
            "Invalid slot_mapping from client: %s",
            error_info,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid slot_mapping format",
            },
        )
    if slot_indices is None:
        return JSONResponse(
            status_code=400,
            content={"error": "failed to parse slot_mapping"},
        )

    if chunk_size is None or chunk_size <= 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "chunk_size must be positive",
            },
        )

    kv_tensors = ctx.kv_tensors
    if not kv_tensors:
        return JSONResponse(
            status_code=404,
            content={"error": "kv_caches empty"},
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_mp_checksums(
            kv_tensors,
            slot_indices,
            chunk_size,
            layerwise,
        ),
    )
    result["slot_mapping_ranges"] = compress_slot_mapping(
        slot_indices,
    )
    return JSONResponse(content=result)


def _compute_mp_checksums(
    kv_tensors: list[torch.Tensor],
    slot_indices: list[int],
    chunk_size: int,
    layerwise: bool,
) -> dict[str, Any]:
    """Compute MD5 checksums over KV cache slots.

    This function is fully layout-agnostic: the knowledge of how
    to gather slots out of a per-layer KV tensor is delegated to
    :func:`extract_kv_at_slots`, and chunk-level slicing is
    delegated to :func:`slice_by_slot_dim`. Future KV layouts
    only need to be taught to ``lmcache.v1.utils.kv_slot_ops``
    and this code path works unchanged.
    """
    num_slots = len(slot_indices)
    num_chunks = (num_slots + chunk_size - 1) // chunk_size
    # Build the slot index tensor on CPU, then memoize a copy
    # per distinct KV device. KV tensors are registered via
    # CUDA IPC and live on GPU; indexing with a CPU slot tensor
    # would trigger an implicit H2D copy per layer (and is
    # rejected by some PyTorch/CUDA builds). Mirrors the
    # device-aligned pattern used in the vLLM-side
    # ``compute_kvcache_checksums``.
    slot_t_cpu = torch.tensor(slot_indices, dtype=torch.long)
    slot_t_by_device: dict[torch.device, torch.Tensor] = {
        slot_t_cpu.device: slot_t_cpu,
    }

    layer_data: list[torch.Tensor] = []
    for kv in kv_tensors:
        slot_t = slot_t_by_device.get(kv.device)
        if slot_t is None:
            slot_t = slot_t_cpu.to(kv.device)
            slot_t_by_device[kv.device] = slot_t
        sliced = extract_kv_at_slots(kv, slot_t)
        # Move to CPU once per layer to save GPU memory and avoid
        # repeated transfers in the chunking loop below.
        sliced = sliced.cpu()
        # numpy() below does not support bfloat16 - upcast if needed.
        if sliced.dtype == torch.bfloat16:
            sliced = sliced.to(torch.float32)
        layer_data.append(sliced)

    if layerwise:
        checksums: dict[str, list[str]] = {}
        for li, ld in enumerate(layer_data):
            cks: list[str] = []
            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, num_slots)
                chunk = slice_by_slot_dim(ld, s, e).contiguous()
                cks.append(hashlib.md5(chunk.numpy().tobytes()).hexdigest())
            checksums["layer_%d" % li] = cks
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": checksums,
            "layerwise": True,
        }

    cks_list: list[str] = []
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, num_slots)
        md5 = hashlib.md5()
        for ld in layer_data:
            chunk = slice_by_slot_dim(ld, s, e).contiguous()
            md5.update(chunk.numpy().tobytes())
        cks_list.append(md5.hexdigest())
    return {
        "status": "success",
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunk_checksums": cks_list,
        "layerwise": False,
    }
