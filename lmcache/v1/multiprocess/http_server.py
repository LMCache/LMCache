# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Annotated, Literal, Optional
import argparse
import base64
import binascii
import io
import threading

# Third Party
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from safetensors.torch import load, save
import numpy as np
import torch
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.server import MPCacheEngine, run_cache_server

logger = init_logger(__name__)

app = FastAPI(title="LMCache HTTP API", version="1.0.0")

# ----------------------------
# Global server lifecycle state
# ----------------------------
_server_instance: Optional[uvicorn.Server] = None
_server_lock = threading.Lock()
_zmq_server = None  # ZMQ MessageQueueServer reference for cleanup


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
        # urlsafe, no padding is common; keep padding for strictness
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
        raise ValueError(
            f"Invalid hash string for encoding={encoding}: {hash_str}"
        ) from e


_engine_instance: Optional[MPCacheEngine] = None


def get_engine() -> Optional[MPCacheEngine]:
    return _engine_instance


def set_engine(engine: MPCacheEngine) -> None:
    global _engine_instance
    _engine_instance = engine


# ----------------------------
# Storage helper functions
# ----------------------------


def search_keys_by_hash(
    engine: MPCacheEngine,
    chunk_hash: bytes,
    model_name: str | None = None,
    world_size: int | None = None,
    worker_id: int | None = None,
) -> list[IPCCacheEngineKey]:
    """
    Search for keys matching the given hash and optional filters.

    Args:
        engine: The MPCacheEngine instance
        chunk_hash: The chunk hash to search for
        model_name: Optional model name filter
        world_size: Optional world size filter
        worker_id: Optional worker ID filter
    Returns:
        List of matching IPCCacheEngineKey objects
    """
    matching_keys = []
    with engine.lock:
        all_keys = engine.storage_manager.get_all_keys()

        for key in all_keys:
            if key.chunk_hash != chunk_hash:
                continue
            if model_name is not None and key.model_name != model_name:
                continue
            if world_size is not None and key.world_size != world_size:
                continue
            if worker_id is not None and key.worker_id != worker_id:
                continue
            matching_keys.append(key)

    return matching_keys


def get_memory_objects(
    engine: MPCacheEngine,
    keys: list[IPCCacheEngineKey],
) -> list[MemoryObj]:
    """
    Get memory objects for the given keys without locking/unlocking.
    This is used for HTTP retrieval where we don't need the full retrieve
    context manager flow.

    Args:
        engine: The MPCacheEngine instance
        keys: List of keys to retrieve

    Returns:
        List of MemoryObj objects (filtered to only include found objects)
    """
    memory_objs = []
    storage_manager = engine.storage_manager
    # Use storage manager's buffer lock for thread safety
    with storage_manager._buffer_lock:
        for key in keys:
            if storage_manager._has_key(key):
                obj = storage_manager._commited_memory_objects[key]
                # Touch the cache policy to update LRU
                storage_manager._cache_policy.update_on_hit(
                    key, storage_manager._commited_memory_objects
                )
                memory_objs.append(obj)

    return memory_objs


@app.get("/")
async def root():
    return {"status": "ok", "service": "LMCache HTTP API"}


@app.get("/all_hashes")
async def get_all_hashes(
    encoding: Annotated[
        HashEncoding,
        Query(description="Hash encoding to return: 'hex' or 'b64url'"),
    ] = "hex",
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
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/kv_cache/{hash_str}")
async def get_kv_cache(
    hash_str: str,
    model_name: Annotated[
        Optional[str], Query(description="Model name for the key")
    ] = None,
    world_size: Annotated[
        Optional[int], Query(description="World size for the key")
    ] = None,
    worker_id: Annotated[
        Optional[int], Query(description="Worker ID for the key")
    ] = None,
    response_format: Annotated[
        Literal["npy", "json"], Query(description="Response format: 'npy' or 'json'")
    ] = "npy",
    hash_encoding: Annotated[
        HashEncoding, Query(description="Hash encoding of hash_str: 'hex' or 'b64url'")
    ] = "hex",
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
            keys = [
                IPCCacheEngineKey(
                    model_name=model_name,
                    world_size=world_size,
                    worker_id=worker_id,
                    chunk_hash=chunk_hash_bytes,
                )
            ]
        else:
            keys = search_keys_by_hash(
                engine, chunk_hash_bytes, model_name, world_size, worker_id
            )
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = get_memory_objects(engine, keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        npy_bytes = tensor_to_npy_bytes(tensor)

        if response_format == "json":
            return JSONResponse(
                content={
                    "hash": hash_str,
                    "hash_encoding": hash_encoding,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "data_b64": base64.b64encode(npy_bytes).decode("ascii"),
                    "data_format": "npy_base64",
                }
            )

        # response_format == "npy"
        return Response(
            content=npy_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="kv_cache_{hash_str}.npy"'
                ),
                "X-Tensor-Shape": str(list(tensor.shape)),
                "X-Tensor-Dtype": str(tensor.dtype),
                "X-Hash-Encoding": hash_encoding,
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Error retrieving KV cache for hash %s: %s", hash_str, e, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/kv_cache/{hash_str}/metadata")
async def get_kv_cache_metadata(
    hash_str: str,
    model_name: Annotated[Optional[str], Query()] = None,
    world_size: Annotated[Optional[int], Query()] = None,
    worker_id: Annotated[Optional[int], Query()] = None,
    hash_encoding: Annotated[
        HashEncoding, Query(description="Hash encoding of hash_str: 'hex' or 'b64url'")
    ] = "hex",
):
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        chunk_hash_bytes = hash_string_to_bytes(hash_str, encoding=hash_encoding)

        if model_name is not None and world_size is not None and worker_id is not None:
            keys = [
                IPCCacheEngineKey(
                    model_name=model_name,
                    world_size=world_size,
                    worker_id=worker_id,
                    chunk_hash=chunk_hash_bytes,
                )
            ]
        else:
            keys = search_keys_by_hash(
                engine, chunk_hash_bytes, model_name, world_size, worker_id
            )
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = get_memory_objects(engine, keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        return JSONResponse(
            content={
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
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Error retrieving KV cache metadata for hash %s: %s",
            hash_str,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/kv-cache/download")
async def download_kv_cache(request: Annotated[DownloadRequest, Body()]):
    """
    Download KV cache as safetensors file.
    Accepts a POST request with JSON body containing chunk_hash and optional metadata.
    Returns the tensor as a safetensors file for download.
    """
    engine = get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Cache engine not initialized")

    try:
        chunk_hash_bytes = hash_string_to_bytes(
            request.chunk_hash, encoding=request.hash_encoding
        )

        if (
            request.model_name is not None
            and request.world_size is not None
            and request.worker_id is not None
        ):
            keys = [
                IPCCacheEngineKey(
                    model_name=request.model_name,
                    world_size=request.world_size,
                    worker_id=request.worker_id,
                    chunk_hash=chunk_hash_bytes,
                )
            ]
        else:
            keys = search_keys_by_hash(
                engine,
                chunk_hash_bytes,
                request.model_name,
                request.world_size,
                request.worker_id,
            )
            if not keys:
                raise HTTPException(status_code=404, detail="KV cache not found")

        memory_objs = get_memory_objects(engine, keys)
        if not memory_objs or memory_objs[0].tensor is None:
            raise HTTPException(status_code=404, detail="KV cache not found")

        tensor = memory_objs[0].tensor
        # Convert tensor to safetensors format
        safetensors_bytes = save({"tensor_bytes": tensor.cpu().contiguous()})

        return Response(
            content=safetensors_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="kv_cache_{request.chunk_hash}.safetensors"'
                ),
                "X-Tensor-Shape": str(list(tensor.shape)),
                "X-Tensor-Dtype": str(tensor.dtype),
                "X-Hash-Encoding": request.hash_encoding,
            },
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Error downloading KV cache for hash %s: %s",
            request.chunk_hash,
            e,
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/api/kv-cache/set")
async def set_kv_cache(
    chunk_hash: Annotated[str, Form()],
    safetensors: Annotated[UploadFile, File()],
    # Optional filters; if omitted we infer metadata by existing key(s) with this hash
    model_name: Annotated[Optional[str], Form()] = None,
    world_size: Annotated[Optional[int], Form()] = None,
    worker_id: Annotated[Optional[int], Form()] = None,
    hash_encoding: Annotated[HashEncoding, Form()] = "hex",
):
    """
    Upload a safetensors file and write it into the cache entry
    identified by `chunk_hash`.

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
            raise HTTPException(
                status_code=400, detail="safetensors must contain 'tensor_bytes' key"
            )

        uploaded_tensor = tensors_dict["tensor_bytes"]
        if uploaded_tensor.is_cuda:
            uploaded_tensor = uploaded_tensor.cpu()
        uploaded_tensor = uploaded_tensor.contiguous()

        # Resolve key metadata:
        # Prefer inference from existing key(s) for this hash
        # (guarantees correct metadata).
        inferred_key: Optional[IPCCacheEngineKey] = None
        keys = search_keys_by_hash(
            engine, chunk_hash_bytes, model_name, world_size, worker_id
        )
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
                    detail=(
                        "No existing entry for this hash; "
                        "provide model_name/world_size/worker_id to create one."
                    ),
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
                raise HTTPException(
                    status_code=500, detail="Existing MemoryObj has no tensor"
                )
            if tuple(dst.shape) != tuple(uploaded_tensor.shape):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Shape mismatch: existing={list(dst.shape)} "
                        f"upload={list(uploaded_tensor.shape)}"
                    ),
                )
            if dst.dtype != uploaded_tensor.dtype:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Dtype mismatch: existing={dst.dtype} "
                        f"upload={uploaded_tensor.dtype}"
                    ),
                )

            dst.copy_(uploaded_tensor, non_blocking=False)

            return JSONResponse(
                content={
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
                }
            )

        # 2) Create-new path (only if metadata provided above)
        fmt = MemoryFormat.KV_2LTD  # confirmed correct
        reserve_handle, reserved_dict = storage.reserve(
            [key], uploaded_tensor.shape, uploaded_tensor.dtype, fmt=fmt
        )
        if key not in reserved_dict:
            raise HTTPException(
                status_code=500, detail="Failed to reserve memory for KV cache"
            )

        obj = reserved_dict[key]
        if obj.tensor is None:
            raise HTTPException(
                status_code=500, detail="Reserved MemoryObj has no tensor"
            )

        obj.tensor.copy_(uploaded_tensor, non_blocking=False)
        storage.commit(reserve_handle)

        return JSONResponse(
            content={
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
            }
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            "Error storing KV cache for hash %s: %s", chunk_hash, e, exc_info=True
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e


def run_http_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    zmq_host: str = "localhost",
    zmq_port: int = 5555,
    chunk_size: int = 256,
    cpu_buffer_size: float = 5.0,
    max_workers: int = 1,
):
    """
    Run the LMCache HTTP server with integrated MP (ZMQ) server.

    This function starts the ZMQ cache server in the background, then runs
    the HTTP server (blocking). On shutdown, both servers are cleaned up.

    Args:
        host: HTTP server host
        port: HTTP server port
        zmq_host: ZMQ server host
        zmq_port: ZMQ server port
        chunk_size: Chunk size for KV cache operations
        cpu_buffer_size: CPU buffer size in GB
        max_workers: Maximum number of worker threads for ZMQ server
    """
    global _server_instance, _zmq_server

    # Start ZMQ MP server and get engine
    zmq_server, engine = run_cache_server(
        host=zmq_host,
        port=zmq_port,
        chunk_size=chunk_size,
        cpu_buffer_size=cpu_buffer_size,
        max_workers=max_workers,
        return_engine=True,
    )
    _zmq_server = zmq_server
    set_engine(engine)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(config)

    with _server_lock:
        _server_instance = server

    logger.info("Starting LMCache HTTP server on http://%s:%d", host, port)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    finally:
        # Clean up after server stops
        with _server_lock:
            _server_instance = None
        if _zmq_server is not None:
            logger.info("Shutting down ZMQ server...")
            _zmq_server.close()
            _zmq_server = None
        logger.info("LMCache HTTP server stopped")


def stop_http_server(timeout: float = 5.0) -> bool:
    """
    Stop the HTTP server gracefully.

    Args:
        timeout: Maximum time to wait for server to stop (seconds)

    Returns:
        True if server was stopped, False if not running or timeout
    """
    global _server_instance, _zmq_server

    with _server_lock:
        if _server_instance is None:
            logger.info("HTTP server is not running")
            return False

        logger.info("Stopping HTTP server...")
        _server_instance.should_exit = True

    # Also stop ZMQ server
    if _zmq_server is not None:
        logger.info("Stopping ZMQ server...")
        _zmq_server.close()

    logger.info("HTTP server stopped successfully")
    return True


def is_http_server_running() -> bool:
    """
    Check if the HTTP server is currently running.

    Returns:
        True if server is running, False otherwise
    """
    with _server_lock:
        return _server_instance is not None and not _server_instance.should_exit


def parse_args():
    parser = argparse.ArgumentParser(
        description="LMCache HTTP Server with integrated MP Cache Server"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the HTTP server"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the HTTP server"
    )
    parser.add_argument(
        "--zmq-host", type=str, default="localhost", help="Host to bind the ZMQ server"
    )
    parser.add_argument(
        "--zmq-port", type=int, default=5555, help="Port to bind the ZMQ server"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256, help="Chunk size for KV cache operations"
    )
    parser.add_argument(
        "--cpu-buffer-size", type=float, default=5.0, help="CPU buffer size in GB"
    )
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Maximum number of worker threads"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_http_server(
        host=args.host,
        port=args.port,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
        chunk_size=args.chunk_size,
        cpu_buffer_size=args.cpu_buffer_size,
        max_workers=args.max_workers,
    )
