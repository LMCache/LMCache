# SPDX-License-Identifier: Apache-2.0
# Standard
import base64
import binascii
import io
import threading
from typing import Optional, Literal

# Third Party
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Body, File, Form, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from safetensors.torch import save, load
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.server import MPCacheEngine
from lmcache.v1.memory_management import MemoryFormat

logger = init_logger(__name__)

app = FastAPI(title="LMCache HTTP API", version="1.0.0")


# ----------------------------
# Tensor serialization helpers
# ----------------------------

def tensor_to_npy_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.is_cuda:
        tensor = tensor.cpu()

    tensor = tensor.detach().contiguous()

    # numpy doesn't support bfloat16 -> cast
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float16)  # or torch.float32 if you want exact-ish

    arr = tensor.numpy()
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()



def npy_bytes_to_tensor(data: bytes) -> torch.Tensor:
    """
    Deserialize .npy bytes into a CPU torch.Tensor.
    """
    buf = io.BytesIO(data)
    arr = np.load(buf, allow_pickle=False)
    return torch.from_numpy(arr)


# ----------------------------
# Hash encoding helpers
# ----------------------------

HashEncoding = Literal["hex", "b64url"]


# ----------------------------
# Request models
# ----------------------------

class DownloadRequest(BaseModel):
    chunk_hash: str
    model_name: Optional[str] = None
    world_size: Optional[int] = None
    worker_id: Optional[int] = None
    hash_encoding: HashEncoding = "hex"


def hash_bytes_to_string(b: bytes, encoding: HashEncoding = "hex") -> str:
    """
    Canonical bytes -> string for client transport.
    """
    if encoding == "hex":
        return b.hex()
    if encoding == "b64url":
        # urlsafe, no padding is common; keep padding for strictness unless you decide otherwise
        return base64.urlsafe_b64encode(b).decode("ascii")
    raise ValueError(f"Unsupported encoding: {encoding}")


def hash_string_to_bytes(hash_str: str, encoding: HashEncoding = "hex") -> bytes:
    """
    Canonical string -> bytes. No guessing. Deterministic.
    """
    s = hash_str.strip()
    try:
        if encoding == "hex":
            if s.startswith(("0x", "0X")):
                s = s[2:]
            # Strict hex parse: will raise ValueError on odd length / bad chars
            return bytes.fromhex(s)

        if encoding == "b64url":
            # Strict-ish base64; accept missing padding by fixing it
            s = s.replace("-", "+").replace("_", "/")
            pad = (-len(s)) % 4
            s = s + ("=" * pad)
            return base64.b64decode(s, validate=True)

        raise ValueError(f"Unsupported encoding: {encoding}")

    except (ValueError, binascii.Error) as e:
        raise ValueError(f"Invalid hash string for encoding={encoding}: {hash_str}") from e


def get_engine() -> Optional[MPCacheEngine]:
    return getattr(get_engine, "_engine", None)


def set_engine(engine: MPCacheEngine) -> None:
    get_engine._engine = engine


@app.get("/")
async def root():
    return {"status": "ok", "service": "LMCache HTTP API"}


@app.get("/all_hashes")
async def get_all_hashes(
    encoding: HashEncoding = Query("hex", description="Hash encoding to return: 'hex' or 'b64url'"),
):
    """
    Return all chunk hashes in a canonical string encoding.
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        all_keys = engine.storage_manager.get_all_keys()
        return [hash_bytes_to_string(k.chunk_hash, encoding=encoding) for k in all_keys]
    except Exception as e:
        logger.error("Error retrieving all hashes: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/kv_cache/{hash_str}")
async def get_kv_cache(
    hash_str: str,
    model_name: Optional[str] = Query(None, description="Model name for the key"),
    world_size: Optional[int] = Query(None, description="World size for the key"),
    worker_id: Optional[int] = Query(None, description="Worker ID for the key"),
    response_format: Literal["npy", "json"] = Query("npy", description="Response format: 'npy' or 'json'"),
    hash_encoding: HashEncoding = Query("hex", description="Hash encoding of hash_str: 'hex' or 'b64url'"),
):
    """
    Get KV cache tensor by hash.
    - response_format=npy: returns raw .npy bytes
    - response_format=json: returns base64 of .npy bytes
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        chunk_hash_bytes = hash_string_to_bytes(hash_str, encoding=hash_encoding)

        if model_name is not None and world_size is not None and worker_id is not None:
            keys = [IPCCacheEngineKey(
                model_name=model_name,
                world_size=world_size,
                worker_id=worker_id,
                chunk_hash=chunk_hash_bytes,
            )]
        else:
            keys = engine.search_keys_by_hash(chunk_hash_bytes, model_name, world_size, worker_id)
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = engine.get_memory_objects(keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        npy_bytes = tensor_to_npy_bytes(tensor)

        if response_format == "json":
            return JSONResponse(content={
                "hash": hash_str,
                "hash_encoding": hash_encoding,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "data_b64": base64.b64encode(npy_bytes).decode("ascii"),
                "data_format": "npy_base64",
            })

        # response_format == "npy"
        return Response(
            content=npy_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="kv_cache_{hash_str}.npy"',
                "X-Tensor-Shape": str(list(tensor.shape)),
                "X-Tensor-Dtype": str(tensor.dtype),
                "X-Hash-Encoding": hash_encoding,
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error retrieving KV cache for hash %s: %s", hash_str, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/kv_cache/{hash_str}/metadata")
async def get_kv_cache_metadata(
    hash_str: str,
    model_name: Optional[str] = Query(None),
    world_size: Optional[int] = Query(None),
    worker_id: Optional[int] = Query(None),
    hash_encoding: HashEncoding = Query("hex", description="Hash encoding of hash_str: 'hex' or 'b64url'"),
):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        chunk_hash_bytes = hash_string_to_bytes(hash_str, encoding=hash_encoding)

        if model_name is not None and world_size is not None and worker_id is not None:
            keys = [IPCCacheEngineKey(
                model_name=model_name,
                world_size=world_size,
                worker_id=worker_id,
                chunk_hash=chunk_hash_bytes,
            )]
        else:
            keys = engine.search_keys_by_hash(chunk_hash_bytes, model_name, world_size, worker_id)
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = engine.get_memory_objects(keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        return JSONResponse(content={
            "hash": hash_str,
            "hash_encoding": hash_encoding,
            "key": {
                "model_name": keys[0].model_name,
                "world_size": keys[0].world_size,
                "worker_id": keys[0].worker_id,
            },
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "element_size": tensor.element_size(),
            "size_bytes": tensor.numel() * tensor.element_size(),
        })

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error retrieving KV cache metadata for hash %s: %s", hash_str, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/kv-cache/download")
async def download_kv_cache(request: DownloadRequest = Body(...)):
    """
    Download KV cache as safetensors file.
    Accepts a POST request with JSON body containing chunk_hash and optional metadata.
    Returns the tensor as a safetensors file for download.
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        chunk_hash_bytes = hash_string_to_bytes(request.chunk_hash, encoding=request.hash_encoding)

        if request.model_name is not None and request.world_size is not None and request.worker_id is not None:
            keys = [IPCCacheEngineKey(
                model_name=request.model_name,
                world_size=request.world_size,
                worker_id=request.worker_id,
                chunk_hash=chunk_hash_bytes,
            )]
        else:
            keys = engine.search_keys_by_hash(chunk_hash_bytes, request.model_name, request.world_size, request.worker_id)
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = engine.get_memory_objects(keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        # Convert tensor to safetensors format
        safetensors_bytes = save({"tensor_bytes": tensor.cpu().contiguous()})

        return Response(
            content=safetensors_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="kv_cache_{request.chunk_hash}.safetensors"',
                "X-Tensor-Shape": str(list(tensor.shape)),
                "X-Tensor-Dtype": str(tensor.dtype),
                "X-Hash-Encoding": request.hash_encoding,
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error downloading KV cache for hash %s: %s", request.chunk_hash, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/kv-cache/set")
async def set_kv_cache(
    chunk_hash: str = Form(...),
    safetensors: UploadFile = File(...),
    # Optional filters; if omitted we infer metadata by existing key(s) with this hash
    model_name: Optional[str] = Form(None),
    world_size: Optional[int] = Form(None),
    worker_id: Optional[int] = Form(None),
    hash_encoding: HashEncoding = Form("hex"),
):
    """
    Upload a safetensors file and write it into the cache entry identified by `chunk_hash`.

    Behavior (for your "-2 := -1" test):
    - If an entry for this chunk_hash exists, OVERWRITE its data in-place.
    - If it does not exist, CREATE a new entry using MemoryFormat.KV_2LTD,
      but only if (model_name, world_size, worker_id) are provided.
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    storage = engine.storage_manager

    try:
        chunk_hash_bytes = hash_string_to_bytes(chunk_hash, encoding=hash_encoding)

        # Read & parse safetensors
        file_contents = await safetensors.read()
        if not file_contents:
            raise HTTPException(status_code=400, detail="Empty safetensors file")

        tensors_dict = load(file_contents)
        if "tensor_bytes" not in tensors_dict:
            raise HTTPException(status_code=400, detail="safetensors must contain 'tensor_bytes' key")

        uploaded_tensor = tensors_dict["tensor_bytes"]
        if uploaded_tensor.is_cuda:
            uploaded_tensor = uploaded_tensor.cpu()
        uploaded_tensor = uploaded_tensor.contiguous()

        # Resolve key metadata:
        # Prefer inference from existing key(s) for this hash (guarantees correct metadata).
        inferred_key: Optional[IPCCacheEngineKey] = None
        keys = engine.search_keys_by_hash(chunk_hash_bytes, model_name, world_size, worker_id)
        if keys:
            inferred_key = keys[0]
        else:
            raise HTTPException(status_code=404, detail="KV cache not found")

        if inferred_key is not None:
            key = inferred_key
        else:
            # No existing entry: require full metadata to create a new key
            if model_name is None or world_size is None or worker_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="No existing entry for this hash; provide model_name/world_size/worker_id to create one.",
                )
            key = IPCCacheEngineKey(
                model_name=model_name,
                world_size=world_size,
                worker_id=worker_id,
                chunk_hash=chunk_hash_bytes,
            )

        # 1) Overwrite path (most common for your -2 := -1 test)
        with storage._buffer_lock:  # internal but OK for debug server
            existing_obj = storage._commited_memory_objects.get(key, None)

        if existing_obj is not None:
            dst = existing_obj.tensor
            if dst is None:
                raise HTTPException(status_code=500, detail="Existing MemoryObj has no tensor")
            if tuple(dst.shape) != tuple(uploaded_tensor.shape):
                raise HTTPException(
                    status_code=400,
                    detail=f"Shape mismatch: existing={list(dst.shape)} upload={list(uploaded_tensor.shape)}",
                )
            if dst.dtype != uploaded_tensor.dtype:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dtype mismatch: existing={str(dst.dtype)} upload={str(uploaded_tensor.dtype)}",
                )

            dst.copy_(uploaded_tensor, non_blocking=False)

            return JSONResponse(content={
                "status": "success",
                "mode": "overwrite",
                "chunk_hash": chunk_hash,
                "hash_encoding": hash_encoding,
                "key": {
                    "model_name": key.model_name,
                    "world_size": key.world_size,
                    "worker_id": key.worker_id,
                },
                "shape": list(uploaded_tensor.shape),
                "dtype": str(uploaded_tensor.dtype),
            })

        # 2) Create-new path (only if metadata provided above)
        fmt = MemoryFormat.KV_2LTD  # confirmed correct
        reserve_handle, reserved_dict = storage.reserve(
            [key], uploaded_tensor.shape, uploaded_tensor.dtype, fmt=fmt
        )
        if key not in reserved_dict:
            raise HTTPException(status_code=500, detail="Failed to reserve memory for KV cache")

        obj = reserved_dict[key]
        if obj.tensor is None:
            raise HTTPException(status_code=500, detail="Reserved MemoryObj has no tensor")

        obj.tensor.copy_(uploaded_tensor, non_blocking=False)
        storage.commit(reserve_handle)

        return JSONResponse(content={
            "status": "success",
            "mode": "reserve_commit",
            "chunk_hash": chunk_hash,
            "hash_encoding": hash_encoding,
            "key": {
                "model_name": key.model_name,
                "world_size": key.world_size,
                "worker_id": key.worker_id,
            },
            "shape": list(uploaded_tensor.shape),
            "dtype": str(uploaded_tensor.dtype),
        })

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Error storing KV cache for hash %s: %s", chunk_hash, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def run_http_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    engine: Optional[MPCacheEngine] = None,
):
    if engine is not None:
        set_engine(engine)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)
    logger.info("Starting LMCache HTTP server on http://%s:%d", host, port)
    server.run()


def start_http_server_thread(
    host: str = "0.0.0.0",
    port: int = 8000,
    engine: Optional[MPCacheEngine] = None,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_http_server,
        args=(host, port, engine),
        daemon=True,
        name="LMCacheHTTPServer",
    )
    thread.start()
    return thread
