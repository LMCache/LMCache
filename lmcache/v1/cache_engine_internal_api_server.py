# SPDX-License-Identifier: Apache-2.0
# Standard
import threading
import time

# Third Party
from fastapi import FastAPI
from prometheus_client import REGISTRY, generate_latest, make_asgi_app
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send
import prometheus_client
import regex as re
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

app = FastAPI()


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def mount_metrics(app: FastAPI, ttl: int = 0):
    """Mount prometheus metrics to a FastAPI app."""

    class MetricsCacheMiddleware:
        def __init__(self, app: ASGIApp):
            self.app = app
            self.cached_data = None
            self.last_update = 0

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope["path"] == "/metrics":
                current_time = time.time()
                if current_time - self.last_update >= ttl:
                    gen_start = time.perf_counter()
                    # filter registry to only include lmcache metrics
                    filtered_registry = REGISTRY.restricted_registry(
                        [
                            name
                            for name in REGISTRY._names_to_collectors
                            if name.startswith("lmcache")
                        ]
                    )
                    self.cached_data = generate_latest(filtered_registry)
                    gen_time = (time.perf_counter() - gen_start) * 1000
                    logger.debug(f"Metrics generation time: {gen_time:.2f}ms")

                    self.last_update = current_time
                else:
                    logger.info("Using cached metrics")
                response = PlainTextResponse(
                    content=self.cached_data,
                    media_type=prometheus_client.CONTENT_TYPE_LATEST,
                )
                await response(scope, receive, send)
            else:
                await self.app(scope, receive, send)

    app.add_middleware(MetricsCacheMiddleware)
    metrics_route = Mount("/metrics", make_asgi_app())

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)


@app.get("/test")
async def get_metrics(request: Request):
    logger.info(f"Test request {request}")
    return PlainTextResponse(content="Success", media_type="text/plain")


class CacheEngineInternalAPIServer:
    def __init__(self, config: LMCacheEngineConfig):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.port = config.cache_engine_internal_api_server_port_start
        ttl = config.cache_engine_internal_api_server_metrics_ttl
        mount_metrics(app, ttl)
        logger.info(
            f"Starting cache engine internal API server on port {self.port}, ttl {ttl}"
        )

    def run(self):
        logger.info(f"Starting LMCache API server on port {self.port}")
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self.port,
            log_config=None,
            access_log=False,
            timeout_keep_alive=60,
            workers=10,
            loop="uvloop",
            http="httptools",
            limit_concurrency=100,
        )
        server = uvicorn.Server(config)
        server.run()

    def start(self):
        self.thread.start()

    def stop(self):
        logger.info("LMCache API server stopped")
