# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
import argparse
import asyncio

# Third Party
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controller_manager import LMCacheControllerManager
from lmcache.v1.cache_controller.message import (  # noqa: E501
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    HealthMsg,
    HealthRetMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
    QueryInstMsg,
    QueryInstRetMsg,
)

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
            lmcache_controller_manager.start_all()
        )
        yield
        # Optionally cancel the task on shutdown
        lmcache_cluster_monitor_task.cancel()
        try:
            await lmcache_cluster_monitor_task
        except asyncio.CancelledError:
            pass

    app = FastAPI(lifespan=lifespan)

    class QueryInstRequest(BaseModel):
        ip: str

    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            msg = QueryInstMsg(
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, QueryInstRetMsg)
            return {"res": ret_msg.instance_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]

    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            msg = LookupMsg(
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(layout_info=ret_msg.layout_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        locations: Optional[List[str]] = []
        tokens: Optional[List[int]] = []

    class ClearResponse(BaseModel):
        success: bool

    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            msg = ClearMsg(
                instance_id=req.instance_id,
                tokens=req.tokens,
                locations=req.locations,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(success=ret_msg.success)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PinRequest(BaseModel):
        instance_id: str
        locations: Optional[List[str]] = []
        tokens: Optional[List[int]] = []

    class PinResponse(BaseModel):
        success: bool

    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            msg = PinMsg(
                instance_id=req.instance_id,
                locations=req.locations,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(success=ret_msg.success)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        locations: Optional[List[str]] = []
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        event_id: str

    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            msg = CompressMsg(
                instance_id=req.instance_id,
                method=req.method,
                locations=req.locations,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(success=ret_msg.event_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class MoveRequest(BaseModel):
        # (instance_id, location)
        old_position: Tuple[str, str]
        new_position: Tuple[str, str]
        tokens: Optional[List[int]] = []

    class MoveResponse(BaseModel):
        event_id: str

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            msg = MoveMsg(
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(success=ret_msg.event_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        alive: bool

    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            msg = HealthMsg(
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(alive=ret_msg.alive)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        event_id: str

    class CheckFinishResponse(BaseModel):
        finished: bool

    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                event_id=req.event_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, CheckFinishRetMsg)
            return CheckFinishResponse(finished=ret_msg.finished)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--monitor-port", type=int, default=9001)

    args = parser.parse_args()

    try:
        app = create_app(f"{args.host}:{args.monitor_port}")

        logger.info(f"Starting LMCache controller at {args.host}:{args.port}")
        logger.info(f"Monitoring lmcache workers at port {args.monitor_port}")

        uvicorn.run(app, host=args.host, port=args.port)
    except TimeoutError as e:
        logger.error(e)


if __name__ == "__main__":
    main()
