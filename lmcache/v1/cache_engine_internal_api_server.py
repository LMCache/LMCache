# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading

# Third Party
from fastapi import FastAPI
from prometheus_client import REGISTRY, generate_latest
from starlette.requests import Request
from starlette.responses import PlainTextResponse
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

app = FastAPI()


@app.get("/metrics")
async def get_metrics(request: Request):
    metrics_data = generate_latest(REGISTRY)
    return PlainTextResponse(content=metrics_data, media_type="text/plain")


class CacheEngineInternalAPIServer:
    def __init__(self, config: LMCacheEngineConfig):
        self.port = config.cache_engine_internal_api_server_port_start
        logger.info(f"Init cache engine internal API server on port {self.port}")
        config = uvicorn.Config(
            app, host="0.0.0.0", port=self.port, loop="uvloop", http="httptools"
        )
        server = uvicorn.Server(config)
        self.server = server

    async def run(self):
        logger.info(f"Running LMCache API server on port {self.port}")
        await self.server.serve()

    def start(self):
        logger.info(f"Starting lmcache internal API server on port {self.port}")
        threading.Thread(target=asyncio.run, args=(self.run(),), daemon=True).start()

    def stop(self):
        logger.info("Stopping LMCache internal API server")
        if self.server:
            self.server.should_exit = True
