# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional
import base64
import json
import io

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


class KVExportRequest(BaseModel):
    """Request model for KV cache export."""
    tokens: List[int]
    locations: Optional[List[str]] = None


class KVImportRequest(BaseModel):
    """Request model for KV cache import."""
    tokens: List[int]
    chunks: List[dict]


@router.post("/kv/export")
async def export_kv(
    request: Request,
    export_request: KVExportRequest,
):
    """Export KV cache data for given tokens."""
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        
        if not lmcache_engine:
            return PlainTextResponse(
                content=json.dumps({"error": "LMCache engine not configured"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        tokens = export_request.tokens
        locations = export_request.locations
        storage_manager = lmcache_engine.storage_manager
        token_database = lmcache_engine.token_database
        
        if storage_manager is None or token_database is None:
            return PlainTextResponse(
                content=json.dumps({"error": "Engine not fully initialized"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        # Process tokens to get cache keys
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        chunk_info_list = list(token_database.process_tokens(tokens=tokens_tensor))
        
        if not chunk_info_list:
            return PlainTextResponse(
                content=json.dumps({
                    "status": "success",
                    "num_tokens_cached": 0,
                    "num_chunks": 0,
                    "chunks": [],
                }, indent=2),
                media_type="application/json",
            )
        
        # Check which chunks exist
        keys = [chunk_info[2] for chunk_info in chunk_info_list]
        hit_chunks, block_mapping = storage_manager.batched_contains(
            keys, search_range=locations, pin=False,
        )
        
        if hit_chunks == 0:
            return PlainTextResponse(
                content=json.dumps({
                    "status": "success",
                    "num_tokens_cached": 0,
                    "num_chunks": 0,
                    "chunks": [],
                }, indent=2),
                media_type="application/json",
            )
        
        # Get memory objects
        location = next(iter(block_mapping.keys())) if block_mapping else None
        memory_objs = storage_manager.batched_get(keys[:hit_chunks], location=location)
        
        if not memory_objs:
            return PlainTextResponse(
                content=json.dumps({
                    "status": "success",
                    "num_tokens_cached": 0,
                    "num_chunks": 0,
                    "chunks": [],
                }, indent=2),
                media_type="application/json",
            )
        
        # Serialize memory objects
        chunks_data = []
        num_tokens_cached = 0
        
        for memory_obj, chunk_info in zip(memory_objs, chunk_info_list[:hit_chunks]):
            if memory_obj is None:
                continue
                
            start_idx, end_idx, key = chunk_info
            num_tokens_cached = end_idx
            tensor = memory_obj.tensor
            
            if tensor is not None:
                # Clone to avoid serializing the entire memory pool
                tensor_cpu = tensor.cpu().clone().contiguous()
                buffer = io.BytesIO()
                torch.save(tensor_cpu, buffer, pickle_protocol=4)
                tensor_bytes = buffer.getvalue()
                
                chunks_data.append({
                    "key": str(key),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "format": memory_obj.get_memory_format().name,
                    "data": base64.b64encode(tensor_bytes).decode('utf-8'),
                })
            
            memory_obj.ref_count_down()
        
        return PlainTextResponse(
            content=json.dumps({
                "status": "success",
                "num_tokens_cached": num_tokens_cached,
                "num_chunks": len(chunks_data),
                "chunks": chunks_data,
            }, indent=2),
            media_type="application/json",
        )
        
    except Exception as e:
        logger.exception(f"[/kv/export] Error: {e}")
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}, indent=2),
            media_type="application/json",
            status_code=500,
        )


@router.post("/kv/import")
async def import_kv(
    request: Request,
    import_request: KVImportRequest,
):
    """Import KV cache data for given tokens."""
    try:
        lmcache_adapter = request.app.state.lmcache_adapter
        lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)
        
        if not lmcache_engine:
            return PlainTextResponse(
                content=json.dumps({"error": "LMCache engine not configured"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        tokens = import_request.tokens
        chunks = import_request.chunks
        storage_manager = lmcache_engine.storage_manager
        token_database = lmcache_engine.token_database
        
        if storage_manager is None or token_database is None:
            return PlainTextResponse(
                content=json.dumps({"error": "Engine not fully initialized"}, indent=2),
                media_type="application/json",
                status_code=503,
            )
        
        # Process tokens to get cache keys
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        chunk_info_list = list(token_database.process_tokens(tokens=tokens_tensor))
        
        if not chunk_info_list:
            return PlainTextResponse(
                content=json.dumps({
                    "status": "error",
                    "message": "Failed to process tokens into chunks",
                }, indent=2),
                media_type="application/json",
                status_code=400,
            )
        
        # Import each chunk
        num_imported = 0
        errors = []
        
        for chunk_idx, chunk_data in enumerate(chunks):
            try:
                # Decode tensor
                tensor_bytes = base64.b64decode(chunk_data['data'])
                buffer = io.BytesIO(tensor_bytes)
                tensor = torch.load(buffer, weights_only=False)
                
                # Get corresponding key
                if chunk_idx < len(chunk_info_list):
                    start_idx, end_idx, key = chunk_info_list[chunk_idx]
                    
                    from lmcache.v1.memory_management import MemoryFormat
                    fmt_name = chunk_data.get('format', 'KV_2LTD')
                    try:
                        fmt = MemoryFormat[fmt_name]
                    except KeyError:
                        fmt = MemoryFormat.KV_2LTD
                    
                    # Store to LocalCPUBackend
                    local_cpu = storage_manager.storage_backends.get("LocalCPUBackend")
                    if local_cpu:
                        memory_obj = local_cpu.allocate(
                            shape=tensor.shape,
                            dtype=tensor.dtype,
                            fmt=fmt,
                        )
                        if memory_obj and memory_obj.tensor is not None:
                            memory_obj.tensor.copy_(tensor)
                            local_cpu.submit_put_task(key, memory_obj)
                            num_imported += 1
                        else:
                            errors.append(f"chunk[{chunk_idx}]: allocation failed")
                    else:
                        errors.append(f"chunk[{chunk_idx}]: LocalCPUBackend not found")
                else:
                    errors.append(f"chunk[{chunk_idx}]: no corresponding key")
                    
            except Exception as e:
                errors.append(f"chunk[{chunk_idx}]: {str(e)}")
        
        return PlainTextResponse(
            content=json.dumps({
                "status": "success",
                "num_chunks_imported": num_imported,
                "num_chunks_requested": len(chunks),
                "errors": errors[:10] if errors else [],
            }, indent=2),
            media_type="application/json",
        )
        
    except Exception as e:
        logger.exception(f"[/kv/import] Error: {e}")
        return PlainTextResponse(
            content=json.dumps({"error": str(e)}, indent=2),
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
