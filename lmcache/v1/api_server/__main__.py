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
    DecompressMsg,
    DecompressRetMsg,
    ErrorMsg,
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
        event_id: str
        ip: str

    class QueryInstResponse(BaseModel):
        event_id: str
        res: str  # the instance id

    @app.post("/query_instance")
    async def query_instance(req: QueryInstRequest):
        try:
            event_id = "QueryInst" + str(uuid.uuid4())
            msg = QueryInstMsg(
                event_id=event_id,
                ip=req.ip,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, QueryInstRetMsg)
            return QueryInstResponse(
                event_id=ret_msg.event_id,
                res=ret_msg.instance_id,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class LookupRequest(BaseModel):
        tokens: List[int]

    class LookupResponse(BaseModel):
        event_id: str
        # a list of (instance_id, location, token_count)
        layout_info: Dict[str, Tuple[str, int]]

    @app.post("/lookup", response_model=LookupResponse)
    async def lookup(req: LookupRequest):
        try:
            event_id = "Lookup" + str(uuid.uuid4())
            msg = LookupMsg(
                event_id=event_id,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, LookupRetMsg)
            return LookupResponse(
                event_id=ret_msg.event_id, layout_info=ret_msg.layout_info
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class ClearRequest(BaseModel):
        instance_id: str
        location: str

    class ClearResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/clear", response_model=ClearResponse)
    async def clear(req: ClearRequest):
        try:
            event_id = "Clear" + str(uuid.uuid4())
            msg = ClearMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, ClearRetMsg)
            return ClearResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class PinRequest(BaseModel):
        instance_id: str
        location: str
        tokens: list[int]

    class PinResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/pin", response_model=PinResponse)
    async def pin(req: PinRequest):
        try:
            event_id = "Pin" + str(uuid.uuid4())
            msg = PinMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, PinRetMsg)
            return PinResponse(event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class CompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    class DecompressRequest(BaseModel):
        instance_id: str
        method: str
        location: str
        tokens: Optional[List[int]] = []

    class DecompressResponse(BaseModel):
        event_id: str
        num_tokens: int

    @app.post("/compress", response_model=CompressResponse)
    async def compress(req: CompressRequest):
        try:
            event_id = "Compress" + str(uuid.uuid4())
            msg = CompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, CompressRetMsg)
            return CompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/decompress", response_model=DecompressResponse)
    async def decompress(req: DecompressRequest):
        try:
            event_id = "Decompress" + str(uuid.uuid4())
            msg = DecompressMsg(
                event_id=event_id,
                instance_id=req.instance_id,
                method=req.method,
                location=req.location,
                tokens=req.tokens,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert isinstance(ret_msg, DecompressRetMsg)
            return DecompressResponse(
                event_id=ret_msg.event_id, num_tokens=ret_msg.num_tokens
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
        event_id: str
        num_tokens: int

    @app.post("/move", response_model=MoveResponse)
    async def move(req: MoveRequest):
        try:
            event_id = "Move" + str(uuid.uuid4())
            msg = MoveMsg(
                event_id=event_id,
                old_position=req.old_position,
                new_position=req.new_position,
                tokens=req.tokens,
                copy=req.copy,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, MoveRetMsg)
            return MoveResponse(
                event_id=ret_msg.event_id,
                num_tokens=ret_msg.num_tokens,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class HealthRequest(BaseModel):
        instance_id: str

    class HealthResponse(BaseModel):
        event_id: str
        # worker_id -> error_code
        error_codes: dict[int, int]

    @app.post("/health", response_model=HealthResponse)
    async def health(req: HealthRequest):
        try:
            event_id = "health" + str(uuid.uuid4())
            msg = HealthMsg(
                event_id=event_id,
                instance_id=req.instance_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
            assert isinstance(ret_msg, HealthRetMsg)
            return HealthResponse(
                event_id=ret_msg.event_id, error_codes=ret_msg.error_codes
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    class CheckFinishRequest(BaseModel):
        event_id: str

    class CheckFinishResponse(BaseModel):
        status: str

    @app.post("/check_finish", response_model=CheckFinishResponse)
    async def check_finish(req: CheckFinishRequest):
        try:
            msg = CheckFinishMsg(
                event_id=req.event_id,
            )
            ret_msg = await lmcache_controller_manager.handle_orchestration_message(msg)
            assert not isinstance(ret_msg, ErrorMsg), ret_msg.error
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
