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

    Args:
        slot_mapping: Slot indices (mixed format).
        chunk_size: Group slots into chunks.
        instance_id: GPU context ID (default 0).
        layerwise: Per-layer checksums if True.
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

    Each kv_tensor shape: [2, NB, BS, NH, HS].
    Slots are mapped via reshape to [2, NB*BS, NH, HS].
    """
    num_slots = len(slot_indices)
    num_chunks = (num_slots + chunk_size - 1) // chunk_size
    slot_t = torch.tensor(
        slot_indices,
        dtype=torch.long,
    )

    # kv: [2, NB, BS, NH, HS] -> [2, NB*BS, NH, HS]
    layer_data: list[torch.Tensor] = []
    for kv in kv_tensors:
        flat = kv.reshape(2, -1, *kv.shape[3:])
        # Move to CPU once per layer to save GPU memory
        # and avoid repeated transfers in the chunking loop
        sliced = flat[:, slot_t, ...].cpu()
        # Handle BFloat16 which is not supported by numpy
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
                chunk = ld[:, s:e, ...].contiguous()
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
            chunk = ld[:, s:e, ...].contiguous()
            md5.update(chunk.numpy().tobytes())
        cks_list.append(md5.hexdigest())
    return {
        "status": "success",
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunk_checksums": cks_list,
        "layerwise": False,
    }
