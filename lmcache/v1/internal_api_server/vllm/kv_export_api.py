# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Third Party
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import torch

# First Party
from lmcache.logging import init_logger

router = APIRouter()
logger = init_logger(__name__)

# Thread pool for concurrent KV export/import operations
# These operations are CPU-bound (tensor ops) so we use threads to avoid blocking the event loop
KV_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="kv_worker")


class KVExportRequest(BaseModel):
    """Request model for KV cache export."""
    tokens: List[int]
    locations: Optional[List[str]] = None


def _do_export(lmcache_engine, tokens: List[int], locations: Optional[List[str]]):
    """Sync worker function for KV export - runs in thread pool."""
    import time
    import struct
    
    timings = {}
    total_start = time.time()
    
    storage_manager = lmcache_engine.storage_manager
    token_database = lmcache_engine.token_database
    
    if storage_manager is None or token_database is None:
        return {"error": "Engine not fully initialized", "status_code": 503}
    
    # Process tokens to get cache keys
    t0 = time.time()
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    chunk_info_list = list(token_database.process_tokens(tokens=tokens_tensor))
    timings['token_processing'] = (time.time() - t0) * 1000
    
    if not chunk_info_list:
        return {"body": b'', "num_chunks": 0, "num_tokens": 0}
    
    # Check which chunks exist
    t0 = time.time()
    keys = [chunk_info[2] for chunk_info in chunk_info_list]
    hit_chunks, block_mapping = storage_manager.batched_contains(
        keys, search_range=locations, pin=False,
    )
    timings['batched_contains'] = (time.time() - t0) * 1000
    
    if hit_chunks == 0:
        return {"body": b'', "num_chunks": 0, "num_tokens": 0}
    
    # Get memory objects
    t0 = time.time()
    location = next(iter(block_mapping.keys())) if block_mapping else None
    memory_objs = storage_manager.batched_get(keys[:hit_chunks], location=location)
    timings['batched_get'] = (time.time() - t0) * 1000
    
    if not memory_objs:
        return {"body": b'', "num_chunks": 0, "num_tokens": 0}
    
    # Build binary response: [header][chunk1][chunk2]...
    # Header: num_chunks (4 bytes)
    # Each chunk: [metadata_len (4B)][metadata_json][tensor_len (4B)][tensor_bytes]
    
    chunks_binary = []
    num_tokens_cached = 0
    time_contiguous = 0
    time_tobytes = 0
    total_tensor_bytes = 0
    
    for memory_obj, chunk_info in zip(memory_objs, chunk_info_list[:hit_chunks]):
        if memory_obj is None:
            continue
            
        start_idx, end_idx, key = chunk_info
        num_tokens_cached = end_idx
        tensor = memory_obj.tensor
        
        if tensor is not None:
            # Ensure contiguous
            t0 = time.time()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            time_contiguous += (time.time() - t0) * 1000
            
            # Get raw bytes - handle bfloat16 (numpy doesn't support it)
            t0 = time.time()
            if tensor.dtype == torch.bfloat16:
                # View as uint16 (same byte layout), then get bytes
                tensor_bytes = tensor.view(torch.uint16).numpy().tobytes()
            else:
                tensor_bytes = tensor.numpy().tobytes()
            time_tobytes += (time.time() - t0) * 1000
            total_tensor_bytes += len(tensor_bytes)
            
            # Metadata as compact JSON
            metadata = json.dumps({
                "key": str(key),
                "start": start_idx,
                "end": end_idx,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "fmt": memory_obj.get_memory_format().name,
            }).encode('utf-8')
            
            # Pack: [metadata_len][metadata][tensor_len][tensor_bytes]
            chunk_binary = (
                struct.pack('<I', len(metadata)) +
                metadata +
                struct.pack('<I', len(tensor_bytes)) +
                tensor_bytes
            )
            chunks_binary.append(chunk_binary)
        
        memory_obj.ref_count_down()
    
    timings['contiguous'] = time_contiguous
    timings['tobytes'] = time_tobytes
    
    # Combine all chunks
    t0 = time.time()
    response_body = struct.pack('<I', len(chunks_binary)) + b''.join(chunks_binary)
    timings['combine'] = (time.time() - t0) * 1000
    
    timings['total'] = (time.time() - total_start) * 1000
    
    # Log timing
    data_mb = total_tensor_bytes / 1024 / 1024
    logger.info(f"[/kv/export] {len(chunks_binary)} chunks, {data_mb:.1f}MB")
    logger.info(f"   ⏱️ token_processing: {timings['token_processing']:.1f}ms")
    logger.info(f"   ⏱️ batched_contains: {timings['batched_contains']:.1f}ms")
    logger.info(f"   ⏱️ batched_get: {timings['batched_get']:.1f}ms")
    logger.info(f"   ⏱️ contiguous: {timings['contiguous']:.1f}ms")
    logger.info(f"   ⏱️ tobytes: {timings['tobytes']:.1f}ms ({data_mb * 1000 / max(timings['tobytes'], 0.1):.1f} MB/s)")
    logger.info(f"   ⏱️ combine: {timings['combine']:.1f}ms")
    logger.info(f"   ⏱️ TOTAL: {timings['total']:.1f}ms ({data_mb * 1000 / max(timings['total'], 0.1):.1f} MB/s)")
    
    return {
        "body": response_body,
        "num_chunks": len(chunks_binary),
        "num_tokens": num_tokens_cached,
    }


@router.post("/kv/export")
async def export_kv(
    request: Request,
    export_request: KVExportRequest,
):
    """Export KV cache data as raw binary.
    
    Runs in thread pool for concurrent execution.
    
    Binary format:
    - [num_chunks: 4B uint32]
    - For each chunk:
      - [metadata_len: 4B uint32][metadata_json: bytes]
      - [tensor_len: 4B uint32][tensor_bytes: bytes]
    """
    from starlette.responses import Response
    
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        
        if not lmcache_engine:
            return PlainTextResponse(
                content=json.dumps({"error": "LMCache engine not configured"}),
                media_type="application/json",
                status_code=503,
            )
        
        tokens = export_request.tokens
        locations = export_request.locations
        
        # Run heavy computation in thread pool for concurrent execution
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            KV_EXECUTOR,
            _do_export,
            lmcache_engine,
            tokens,
            locations,
        )
        
        # Handle error result
        if "error" in result:
            return PlainTextResponse(
                content=json.dumps({"error": result["error"]}),
                media_type="application/json",
                status_code=result.get("status_code", 500),
            )
        
        return Response(
            content=result["body"],
            media_type="application/octet-stream",
            headers={
                "X-Num-Chunks": str(result["num_chunks"]),
                "X-Num-Tokens": str(result["num_tokens"]),
            },
        )
        
    except Exception as e:
        logger.exception(f"[/kv/export] Error: {e}")
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


def _do_import(lmcache_engine, tokens: List[int], body: bytes, read_body_time: float):
    """Sync worker function for KV import - runs in thread pool."""
    import time
    import struct
    import numpy as np
    from lmcache.v1.memory_management import MemoryFormat
    
    timings = {'read_body': read_body_time}
    total_start = time.time()
    
    storage_manager = lmcache_engine.storage_manager
    token_database = lmcache_engine.token_database
    
    if storage_manager is None or token_database is None:
        return {"error": "Engine not fully initialized", "status_code": 503}
    
    if len(body) < 4:
        return {"num_imported": 0, "empty": True}
    
    # Process tokens to get cache keys
    t0 = time.time()
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    chunk_info_list = list(token_database.process_tokens(tokens=tokens_tensor))
    timings['token_processing'] = (time.time() - t0) * 1000
    
    if not chunk_info_list:
        return {"num_imported": 0, "error_hint": "no_chunks"}
    
    # Parse binary format
    t0 = time.time()
    offset = 0
    num_chunks = struct.unpack('<I', body[offset:offset+4])[0]
    offset += 4
    
    chunks_parsed = []
    for _ in range(num_chunks):
        # Metadata
        meta_len = struct.unpack('<I', body[offset:offset+4])[0]
        offset += 4
        metadata = json.loads(body[offset:offset+meta_len].decode('utf-8'))
        offset += meta_len
        
        # Tensor bytes
        tensor_len = struct.unpack('<I', body[offset:offset+4])[0]
        offset += 4
        tensor_bytes = body[offset:offset+tensor_len]
        offset += tensor_len
        
        chunks_parsed.append((metadata, tensor_bytes))
    timings['parse_binary'] = (time.time() - t0) * 1000
    
    # Import chunks
    dtype_map = {
        'torch.float16': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.float32': torch.float32,
        'torch.int8': torch.int8,
        'torch.int16': torch.int16,
        'torch.int32': torch.int32,
        'torch.int64': torch.int64,
    }
    
    num_imported = 0
    time_reconstruct = 0
    time_allocate = 0
    time_copy = 0
    time_submit = 0
    total_bytes = 0
    
    for chunk_idx, (metadata, tensor_bytes) in enumerate(chunks_parsed):
        try:
            total_bytes += len(tensor_bytes)
            
            # Reconstruct tensor
            t0 = time.time()
            shape = metadata.get('shape')
            dtype_str = metadata.get('dtype', 'torch.bfloat16')
            dtype = dtype_map.get(dtype_str, torch.bfloat16)
            
            np_dtype = {
                torch.float16: np.float16,
                torch.bfloat16: np.uint16,
                torch.float32: np.float32,
                torch.int8: np.int8,
                torch.int16: np.int16,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }.get(dtype, np.float16)
            
            np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(np_array.copy())
            if dtype == torch.bfloat16:
                tensor = tensor.view(torch.bfloat16)
            time_reconstruct += (time.time() - t0) * 1000
            
            if chunk_idx < len(chunk_info_list):
                start_idx, end_idx, key = chunk_info_list[chunk_idx]
                
                fmt_name = metadata.get('fmt', 'KV_2LTD')
                try:
                    fmt = MemoryFormat[fmt_name]
                except KeyError:
                    fmt = MemoryFormat.KV_2LTD
                
                local_cpu = storage_manager.storage_backends.get("LocalCPUBackend")
                if local_cpu:
                    t0 = time.time()
                    memory_obj = local_cpu.allocate(
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        fmt=fmt,
                    )
                    time_allocate += (time.time() - t0) * 1000
                    
                    if memory_obj and memory_obj.tensor is not None:
                        t0 = time.time()
                        memory_obj.tensor.copy_(tensor)
                        time_copy += (time.time() - t0) * 1000
                        
                        t0 = time.time()
                        local_cpu.submit_put_task(key, memory_obj)
                        time_submit += (time.time() - t0) * 1000
                        
                        num_imported += 1
                        
        except Exception as e:
            logger.warning(f"[/kv/import] chunk[{chunk_idx}] error: {e}")
    
    timings['reconstruct'] = time_reconstruct
    timings['allocate'] = time_allocate
    timings['copy'] = time_copy
    timings['submit'] = time_submit
    timings['total'] = (time.time() - total_start) * 1000 + read_body_time
    
    data_mb = total_bytes / 1024 / 1024
    logger.info(f"[/kv/import] {num_imported}/{num_chunks} chunks, {data_mb:.1f}MB")
    logger.info(f"   ⏱️ read_body: {timings['read_body']:.1f}ms")
    logger.info(f"   ⏱️ token_processing: {timings['token_processing']:.1f}ms")
    logger.info(f"   ⏱️ parse_binary: {timings['parse_binary']:.1f}ms")
    logger.info(f"   ⏱️ reconstruct: {timings['reconstruct']:.1f}ms ({data_mb * 1000 / max(timings['reconstruct'], 0.1):.1f} MB/s)")
    logger.info(f"   ⏱️ allocate: {timings['allocate']:.1f}ms")
    logger.info(f"   ⏱️ copy: {timings['copy']:.1f}ms")
    logger.info(f"   ⏱️ submit: {timings['submit']:.1f}ms")
    logger.info(f"   ⏱️ TOTAL: {timings['total']:.1f}ms ({data_mb * 1000 / max(timings['total'], 0.1):.1f} MB/s)")
    
    return {"num_imported": num_imported}


@router.post("/kv/import")
async def import_kv(request: Request):
    """Import KV cache from raw binary data.
    
    Runs in thread pool for concurrent execution.
    
    Binary format (matches export):
    - [num_chunks: 4B uint32]
    - For each chunk:
      - [metadata_len: 4B uint32][metadata_json: bytes]
      - [tensor_len: 4B uint32][tensor_bytes: bytes]
    
    Headers:
    - X-Tokens: JSON array of token IDs
    """
    import time
    from starlette.responses import Response
    
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        
        if not lmcache_engine:
            return PlainTextResponse(
                content=json.dumps({"error": "LMCache engine not configured"}),
                media_type="application/json",
                status_code=503,
            )
        
        # Get tokens from header
        tokens_header = request.headers.get("X-Tokens")
        if not tokens_header:
            return PlainTextResponse(
                content=json.dumps({"error": "Missing X-Tokens header"}),
                media_type="application/json",
                status_code=400,
            )
        tokens = json.loads(tokens_header)
        
        # Read binary body (async, before moving to thread)
        t0 = time.time()
        body = await request.body()
        read_body_time = (time.time() - t0) * 1000
        
        # Run heavy computation in thread pool for concurrent execution
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            KV_EXECUTOR,
            _do_import,
            lmcache_engine,
            tokens,
            body,
            read_body_time,
        )
        
        # Handle error result
        if "error" in result:
            return PlainTextResponse(
                content=json.dumps({"error": result["error"]}),
                media_type="application/json",
                status_code=result.get("status_code", 500),
            )
        
        headers = {"X-Num-Imported": str(result["num_imported"])}
        if result.get("error_hint"):
            headers["X-Error"] = result["error_hint"]
        
        return Response(
            content=b'OK',
            media_type="application/octet-stream",
            headers=headers,
        )
        
    except Exception as e:
        logger.exception(f"[/kv/import] Error: {e}")
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500,
        )


@router.get("/kv/info")
async def kv_info(request: Request):
    """Get KV cache storage info."""
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        
        if not lmcache_engine:
            return PlainTextResponse(
                content=json.dumps({"error": "LMCache engine not configured"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        storage_manager = lmcache_engine.storage_manager
        if storage_manager is None:
            return PlainTextResponse(
                content=json.dumps({"error": "Storage manager not available"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        backends_info = {}
        for name, backend in storage_manager.storage_backends.items():
            try:
                backends_info[name] = {
                    "type": type(backend).__name__,
                    "size": getattr(backend, 'size', lambda: 'N/A')(),
                }
            except Exception:
                backends_info[name] = {"type": type(backend).__name__, "error": "failed to get info"}
        
        return PlainTextResponse(
            content=json.dumps({
                "status": "success",
                "chunk_size": lmcache_engine.config.chunk_size,
                "backends": backends_info,
            }, indent=2),
            media_type="application/json",
        )
        
    except Exception as e:
        logger.exception(f"[/kv/info] Error: {e}")
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}, indent=2),
            media_type="application/json",
            status_code=500,
        )
