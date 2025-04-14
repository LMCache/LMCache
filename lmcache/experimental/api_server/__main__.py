import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lmcache.experimental.cache_controller import LMCacheClusterExecutor
from lmcache.logging import init_logger

logger = init_logger(__name__)


def create_app(lmcache_instance_ids: List[str]) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_cluster_executor = LMCacheClusterExecutor(lmcache_instance_ids)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start background task here
        lmcache_cluster_monitor_task = asyncio.create_task(
            lmcache_cluster_executor.monitor_metadata_loop())
        yield
        # Optionally cancel the task on shutdown
        lmcache_cluster_monitor_task.cancel()
        try:
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)

    class LookupRequest(BaseModel):
        instance_id: str
        tokens: List[int]
        worker_ids: Optional[List[int]] = []

    @app.post("/lookup")
    async def lookup(req: LookupRequest):
        try:
            kwargs = {"tokens": req.tokens, "worker_ids": req.worker_ids}
            return await lmcache_cluster_executor.execute(
                req.instance_id, "lookup", **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearCacheRequest(BaseModel):
        instance_id: str
        tokens: Optional[List[int]] = []
        worker_ids: Optional[List[int]] = []

    @app.post("/clear")
    async def clear(req: ClearCacheRequest):
        try:
            kwargs = {"tokens": req.tokens, "worker_ids": req.worker_ids}
            return await lmcache_cluster_executor.execute(
                req.instance_id, "clear", **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lmcache-instance-ids",
                        type=str,
                        nargs="+",
                        default=["lmcache_default_instance"])
    args = parser.parse_args()

    try:
        app = create_app(args.lmcache_instance_ids)

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"LMCache instance ids: {args.lmcache_instance_ids}")

        uvicorn.run(app, host=args.host, port=args.port)  #, reload=True)
    except TimeoutError as e:
        logger.error(e)
