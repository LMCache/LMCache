# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
import argparse
import asyncio
import uuid

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
        eventId: str
        ip: str

    class QueryInstResponse(BaseModel):
        eventId: str
        res: str  # the instance id

    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            eventId = ("QueryInst" + str(uuid.uuid4()),)
            msg = QueryInstMsg(
                eventId=eventId,
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, QueryInstRetMsg)
            return QueryInstResponse(
                eventId=ret_msg.eventId,
                res=ret_msg.instanceId,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        eventId: str
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]

    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            eventId = "Lookup" + str(uuid.uuid4())
            msg = LookupMsg(
                eventId=eventId,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(
                eventId=ret_msg.eventId, layout_info=ret_msg.layout_info
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        locations: Optional[List[str]] = []
        tokens: Optional[List[int]] = []

    class ClearResponse(BaseModel):
        eventId: str
        success: bool

    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            eventId = "Clear" + str(uuid.uuid4())
            msg = ClearMsg(
                eventId=eventId,
                instance_id=req.instance_id,
                tokens=req.tokens,
                locations=req.locations,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(eventId=ret_msg.eventId, success=ret_msg.success)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PinRequest(BaseModel):
        instance_id: str
        locations: Optional[List[str]] = []
        tokens: Optional[List[int]] = []

    class PinResponse(BaseModel):
        eventId: str
        success: bool

    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            eventId = "Pin" + str(uuid.uuid4())
            msg = PinMsg(
                eventId=eventId,
                instance_id=req.instance_id,
                locations=req.locations,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(eventId=ret_msg.eventId, success=ret_msg.success)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        eventId: str
        num_tokens: int

    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            eventId = "Compress" + str(uuid.uuid4())
            msg = CompressMsg(
                eventId=eventId,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(
                eventId=ret_msg.eventId, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class MoveRequest(BaseModel):
        # (instance_id, location)
        old_position: Tuple[str, str]
        new_position: Tuple[str, str]
        tokens: Optional[List[int]] = []
        copy: Optional[bool] = False

    class MoveResponse(BaseModel):
        eventId: str
        num_tokens: int

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            eventId = "Move" + str(uuid.uuid4())
            msg = MoveMsg(
                eventId=eventId,
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
                copy=req.copy,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(
                eventId=ret_msg.eventId,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        eventId: str
        alive: bool

    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            eventId = "Health" + str(uuid.uuid4())
            msg = HealthMsg(
                eventId=eventId,
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(eventId=ret_msg.eventId, alive=ret_msg.alive)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        eventId: str

    class CheckFinishResponse(BaseModel):
        status: str

    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                eventId=req.eventId,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, CheckFinishRetMsg)
            return CheckFinishResponse(status=ret_msg.status)
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
