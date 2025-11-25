# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, List, Optional
import asyncio
import json

# Third Party
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.remote_backend import RemoteBackend


class LoadFSChunksRequest(BaseModel):
    """Request model for loading FS chunks."""

    config_path: str
    max_chunks: Optional[int] = None
    max_failed_keys: int = 10


router = APIRouter()
logger = init_logger(__name__)


@router.post("/cache/load-fs-chunks")
async def load_fs_chunks(
    request: Request,
    request_body: LoadFSChunksRequest,
):
    """
    Load chunk files from FSConnector into LocalCPUBackend hot cache.

    This endpoint loads all chunk files from the specified FSConnector directory
    into the LocalCPUBackend's hot cache by:
    1. Loading configuration from the specified config file
    2. Initializing RemoteBackend with FSConnector
    3. Listing all chunk files in the FSConnector directory
    4. Constructing CacheEngineKey from filenames
    5. Loading MemoryObj from files and putting into hot cache

    Args:
        request: The FastAPI request object
        config_path: Path to the configuration file for RemoteBackend
        max_chunks: Maximum number of chunks to load (optional, loads all if None)

    Returns:
        PlainTextResponse: JSON response with loading statistics

    Example:
        ```bash
        curl -X POST "http://localhost:8000/cache/load-fs-chunks" \
             -H "Content-Type: application/json" \
             -d '{"config_path": "/path/to/config.json", "max_chunks": 100}'
        ```
    """
    lmcache_adapter = request.app.state.lmcache_adapter
    lmcache_engine = getattr(lmcache_adapter, "lmcache_engine", None)

    if not lmcache_engine:
        error_info = {
            "error": "/cache/load-fs-chunks API is unavailable",
            "message": "LMCache engine not configured.",
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=503,
        )

    remote_backend = None
    try:
        config = await _load_config_from_file(request_body.config_path)
        local_cpu_backend = lmcache_engine.storage_manager.allocator_backend

        remote_backend = await _initialize_remote_backend(
            config,
            lmcache_engine.metadata,
            local_cpu_backend,
            lmcache_engine.storage_manager.loop,
        )

        result = await _load_chunks_from_fs_connector(
            remote_backend,
            local_cpu_backend,
            request_body.max_chunks,
            request_body.max_failed_keys,
        )

        success_info = {
            "status": "success",
            "loaded_chunks": result["loaded_chunks"],
            "total_files": result["total_files"],
            "failed_keys": result["failed_keys"],
            "config_path": request_body.config_path,
        }

        return PlainTextResponse(
            content=json.dumps(success_info, indent=2),
            media_type="application/json",
        )

    except Exception as e:
        error_info = {
            "error": "Failed to load chunks from FSConnector",
            "message": str(e),
            "config_path": request_body.config_path,
        }
        return PlainTextResponse(
            content=json.dumps(error_info, indent=2),
            media_type="application/json",
            status_code=500,
        )
    finally:
        if remote_backend is not None:
            remote_backend.close()


async def _load_config_from_file(config_path: str) -> LMCacheEngineConfig:
    """Load configuration from yaml file."""
    try:
        return LMCacheEngineConfig.from_file(config_path)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Invalid configuration file: {str(e)}"
        ) from e


async def _initialize_remote_backend(
    config: LMCacheEngineConfig, metadata, local_cpu_backend, loop
) -> RemoteBackend:
    """Initialize RemoteBackend with FSConnector."""
    try:
        remote_backend = RemoteBackend(
            config=config,
            metadata=metadata,
            loop=loop,
            local_cpu_backend=local_cpu_backend,
            dst_device="cpu",
        )
        remote_backend.init_connection()
        return remote_backend
    except Exception as e:
        logger.error(f"Failed to initialize RemoteBackend: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize RemoteBackend: {str(e)}"
        ) from e


async def _load_chunks_from_fs_connector(
    remote_backend: RemoteBackend,
    local_cpu_backend,
    max_chunks: Optional[int] = None,
    max_failed_keys: int = 10,
) -> Dict:
    """Load chunks from FSConnector into LocalCPUBackend."""
    connector = remote_backend.connection
    if not connector:
        raise HTTPException(status_code=500, detail="FSConnector not initialized")

    try:
        chunk_files = await connector.list()
        total_files = len(chunk_files)

        if max_chunks:
            chunk_files = chunk_files[:max_chunks]

        logger.info(f"Found {len(chunk_files)} chunk files to load")

        loaded_chunks = 0
        failed_keys: List[str] = []

        for chunk_filename in chunk_files:
            try:
                key_str = chunk_filename.replace("-SEP-", "/")
                key = CacheEngineKey.from_string(key_str)

                memory_obj = await asyncio.get_event_loop().run_in_executor(
                    None, remote_backend.get_blocking, key
                )

                if memory_obj:
                    local_cpu_backend.submit_put_task(key, memory_obj)
                    memory_obj.ref_count_down()
                    loaded_chunks += 1

                    if loaded_chunks % 100 == 0:
                        logger.info(f"Loaded {loaded_chunks} chunks...")
                else:
                    failed_keys.append(key_str)
                    logger.warning(f"Failed to load chunk: {key_str}")

            except Exception as e:
                failed_keys.append(chunk_filename)
                logger.warning(f"Error loading chunk {chunk_filename}: {e}")

        logger.info(
            f"Successfully loaded {loaded_chunks} chunks from {total_files} files"
        )

        return {
            "loaded_chunks": loaded_chunks,
            "total_files": total_files,
            "failed_keys": failed_keys[:max_failed_keys],
        }

    except Exception as e:
        logger.error(f"Error in chunk loading process: {e}")
        raise HTTPException(
            status_code=500, detail=f"Chunk loading failed: {str(e)}"
        ) from e
