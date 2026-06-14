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
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

router = APIRouter()


# Per-format axis of the ``num_blocks`` dimension inside a per-layer KV tensor.
# The MP /kvcache/check endpoint gathers KV data by block IDs along this
# axis, which preserves the block_size dimension verbatim so chunking is a
# clean slice on a known axis.
#
# Formats that fuse num_blocks and block_size into a single page-buffer
# dimension (TWO_X_NL_X_NBBS_NH_HS, NL_X_NBBS_ONE_HS) and the cross-layer
# NB_NL_TWO_BS_NH_HS layout are intentionally not listed: the block-level
# semantics don't map cleanly onto a single layer tensor, and the diagnostic
# API declines them with HTTP 501 until a real need appears.
_BLOCK_AXIS_BY_FORMAT: dict[Any, int] = {
    lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS: 1,  # [2, NB, BS, NH, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS: 0,  # [NB, 2, BS, NH, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_BS_HS: 0,  # MLA: [NB, BS, HS]
    lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS: 1,  # [2, NB, NH, BS, HS]
    lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS: 0,  # [NB, 2, NH, BS, HS]
}


@router.post("/clear-cache")
async def clear_cache(request: Request) -> Any:
    """Force-clear all KV cache data stored in L1 (CPU) memory.

    This clears all objects including those with active read/write locks.
    In-flight store or prefetch operations may be corrupted.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "reason": "engine not initialized"},
        )

    engine.clear()
    logger.info("Cache cleared via HTTP API")
    return {"status": "ok"}


@router.get("/kvcache/check")
async def kvcache_check(
    request: Request,
    block_ids: Optional[str] = None,
    chunk_size: Optional[int] = None,
    instance_id: int = 0,
    layerwise: bool = False,
) -> JSONResponse:
    """Compute MD5 checksums for KV cache blocks, grouped ``chunk_size``
    blocks at a time.

    MP mode addresses KV storage by block IDs natively (same as
    ``STORE``/``RETRIEVE``), so this endpoint is fully block-centric:
    ``block_ids`` enumerates the target blocks and ``chunk_size`` counts
    blocks per hashed chunk. Intended for diagnostics / round-trip
    integrity checks from ``lmcache bench server``.

    Args:
        request: FastAPI request.
        block_ids: GPU block IDs in mixed format (e.g. ``"0,[2,5],8"``).
        chunk_size: Positive integer — number of blocks per hashed chunk.
        instance_id: GPU context ID on the engine (default 0).
        layerwise: If True, return per-layer checksums keyed by
            ``"layer_<idx>"``; otherwise a single aggregated digest per
            chunk (all layers combined).

    Returns:
        JSON body on success::

            {
                "status": "success",
                "chunk_size": <int>,          # blocks per chunk
                "num_chunks": <int>,
                "chunk_checksums": <list[str] | dict[str, list[str]]>,
                "layerwise": <bool>,
                "block_id_ranges": "<compressed block id ranges>",
            }

        ``chunk_checksums`` is ``list[str]`` when ``layerwise=False`` and
        ``dict[str, list[str]]`` when ``layerwise=True``.

    HTTP status codes:
        200: success.
        400: ``block_ids`` missing/malformed, or ``chunk_size``
            missing/non-positive.
        404: ``instance_id`` not registered, or KV tensors empty.
        501: engine has no ``gpu_contexts``, or the GPU KV format is
            not supported by this endpoint.
        503: engine not yet initialised on ``app.state``.
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
            content={"error": "checksum not supported for this engine type"},
        )

    ctx = gpu_ctxs.get(instance_id)
    if ctx is None:
        return JSONResponse(
            status_code=404,
            content={"error": "instance_id %d not registered" % instance_id},
        )

    if not block_ids:
        return JSONResponse(
            status_code=400,
            content={"error": "block_ids is required"},
        )

    parsed_blocks, block_err = parse_mixed_slot_mapping(block_ids)
    if block_err or parsed_blocks is None:
        if block_err:
            logger.warning("Invalid block_ids from client: %s", block_err)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid block_ids format"},
        )

    if chunk_size is None or chunk_size <= 0:
        return JSONResponse(
            status_code=400,
            content={"error": "chunk_size must be positive"},
        )

    kv_tensors = ctx.kv_tensors
    if not kv_tensors:
        return JSONResponse(
            status_code=404,
            content={"error": "kv_caches empty"},
        )

    engine_kv_format = ctx.engine_kv_format_
    block_axis = _BLOCK_AXIS_BY_FORMAT.get(engine_kv_format)
    if block_axis is None:
        return JSONResponse(
            status_code=501,
            content={
                "error": "checksum not supported for GPU KV format %s"
                % engine_kv_format.name
            },
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: _compute_block_checksums(
            kv_tensors, parsed_blocks, block_axis, chunk_size, layerwise
        ),
    )
    result["block_id_ranges"] = compress_slot_mapping(parsed_blocks)
    return JSONResponse(content=result)


def _compute_block_checksums(
    kv_tensors: list[torch.Tensor],
    block_ids: list[int],
    block_axis: int,
    chunk_size: int,
    layerwise: bool,
) -> dict[str, Any]:
    """Compute MD5 checksums over KV cache blocks, grouped ``chunk_size``
    blocks per chunk.

    The gather is performed in block space on ``block_axis`` of each
    per-layer KV tensor using :func:`torch.index_select`. ``block_axis``
    is looked up once per request from :data:`_BLOCK_AXIS_BY_FORMAT`, so
    adding a new KV format to the endpoint is a single dict entry.

    Each layer's gathered tensor is moved to CPU once, then chunk-level
    checksums slice it along ``block_axis`` in steps of ``chunk_size``
    blocks. ``bfloat16`` is upcast to ``float32`` only if present, since
    :meth:`torch.Tensor.numpy` does not support it.
    """
    num_blocks = len(block_ids)
    num_chunks = (num_blocks + chunk_size - 1) // chunk_size

    # Build the block index tensor on CPU once, memoise per KV device to
    # avoid an implicit H2D copy (and mixed-device indexing) per layer.
    block_idx_cpu = torch.tensor(block_ids, dtype=torch.long)
    block_idx_by_device: dict[torch.device, torch.Tensor] = {
        block_idx_cpu.device: block_idx_cpu,
    }

    # Pre-gather each layer's blocks to CPU (one H2D→D2H transfer per
    # layer), then all chunking is CPU-only below.
    layer_blocks: list[torch.Tensor] = []
    for kv in kv_tensors:
        idx = block_idx_by_device.get(kv.device)
        if idx is None:
            idx = block_idx_cpu.to(kv.device)
            block_idx_by_device[kv.device] = idx
        gathered = kv.index_select(block_axis, idx).cpu()
        if gathered.dtype == torch.bfloat16:
            gathered = gathered.to(torch.float32)
        layer_blocks.append(gathered)

    def _chunk_slice(t: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return t.narrow(block_axis, start, end - start).contiguous()

    if layerwise:
        per_layer: dict[str, list[str]] = {}
        for li, blocks in enumerate(layer_blocks):
            digests: list[str] = []
            for ci in range(num_chunks):
                s = ci * chunk_size
                e = min(s + chunk_size, num_blocks)
                chunk = _chunk_slice(blocks, s, e)
                digests.append(hashlib.md5(chunk.numpy().tobytes()).hexdigest())
            per_layer["layer_%d" % li] = digests
        return {
            "status": "success",
            "chunk_size": chunk_size,
            "num_chunks": num_chunks,
            "chunk_checksums": per_layer,
            "layerwise": True,
        }

    aggregated: list[str] = []
    for ci in range(num_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, num_blocks)
        md5 = hashlib.md5()
        for blocks in layer_blocks:
            chunk = _chunk_slice(blocks, s, e)
            md5.update(chunk.numpy().tobytes())
        aggregated.append(md5.hexdigest())
    return {
        "status": "success",
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunk_checksums": aggregated,
        "layerwise": False,
    }
