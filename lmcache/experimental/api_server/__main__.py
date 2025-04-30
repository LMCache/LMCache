import argparse
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lmcache.experimental.cache_controller.controller_manager import \
    LMCacheControllerManager
from lmcache.experimental.cache_controller.message import (ClearMsg,
                                                           ClearRetMsg,
                                                           LookupMsg,
                                                           LookupRetMsg)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def create_app(controller_url: str) -> FastAPI:
    """
    Create a FastAPI application with endpoints for LMCache operations.
    """
    lmcache_controller_manager = LMCacheControllerManager(controller_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start background task here
        lmcache_cluster_monitor_task = asyncio.create_task(
            lmcache_controller_manager.start_all())
        yield
        # Optionally cancel the task on shutdown
        lmcache_cluster_monitor_task.cancel()
        try:
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)

    class LookupRequest(BaseModel):
        tokens: List[int]

    @app.post("/lookup")
    async def lookup(req: LookupRequest):
        try:
            msg = LookupMsg(tokens=req.tokens, )
            ret_msg = await lmcache_controller_manager.\
                handle_orchestration_message(msg)
            assert isinstance(ret_msg, LookupRetMsg)
            return {"res": ret_msg.best_instance_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearCacheRequest(BaseModel):
        instance_id: str
        tokens: Optional[List[int]] = []
        worker_ids: Optional[List[int]] = []

    @app.post("/clear")
    async def clear(req: ClearCacheRequest):
        try:
            msg = ClearMsg(
                instance_id=req.instance_id,
                worker_ids=req.worker_ids,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.\
                handle_orchestration_message(msg)
            assert isinstance(ret_msg, ClearRetMsg)
            return {"res": ret_msg.success}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--monitor-port", type=int, default=9001)

    args = parser.parse_args()

    try:
        app = create_app(f"{args.host}:{args.monitor_port}")

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"Monitoring lmcache workers at port {args.monitor_port}")

        uvicorn.run(app, host=args.host, port=args.port)  #, reload=True)
    except TimeoutError as e:
        logger.error(e)
