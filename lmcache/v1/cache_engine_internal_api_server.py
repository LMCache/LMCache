# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio
import logging
import sys
import threading
import traceback

# Third Party
from fastapi import FastAPI, Query
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


@app.get("/threads")
async def get_threads(
    request: Request,
    name: Optional[str] = Query(
        None, description="Filter by thread name (fuzzy match)"
    ),
    thread_id: Optional[int] = Query(None, description="Filter by thread ID"),
):
    """Return information about active threads with optional filtering"""
    threads = threading.enumerate()

    filtered_threads = []
    for t in threads:
        # Apply filters
        if name and name.lower() not in t.name.lower():
            continue
        if thread_id and t.ident != thread_id:
            continue
        filtered_threads.append(t)

    thread_info = []

    for t in filtered_threads:
        # Basic thread info with creation time
        info = f"Thread: {t}\n"

        # Get stack trace if available
        try:
            stack_frames = sys._current_frames().get(t.ident)
            if stack_frames:
                stack_trace = traceback.format_stack(stack_frames)
                info += "Stack trace:\n" + "".join(stack_trace)
            else:
                info += "No stack trace available\n"
        except AttributeError:
            info += "Stack trace unavailable\n"

        thread_info.append(info)

    # Add summary section
    summary = "\n\n=== Thread Summary ===\n"
    summary += f"Total threads: {len(filtered_threads)}\n"

    return PlainTextResponse(
        content="\n\n".join(thread_info) + summary, media_type="text/plain"
    )


@app.get("/loglevel")
async def get_or_set_log_level(logger_name: str = None, level: str = None):
    """
    Get or set the log level for a logger.
    - No parameters: List all loggers and their levels.
    - With logger_name: Get the level of the specified logger.
    - With logger_name and level: Set the level of the specified logger.
    """
    # Use the global logger to record logs
    logger.debug(f"Get or set log level for logger {logger_name} to level {level}")
    if not logger_name and not level:
        # List all loggers and their levels
        loggers = logging.Logger.manager.loggerDict
        result = "=== Loggers and Levels ===\n"
        for name, logger_obj in loggers.items():
            if isinstance(logger_obj, logging.Logger):
                result += f"{name}: {logging.getLevelName(logger_obj.level)}\n"
        return PlainTextResponse(content=result, media_type="text/plain")
    elif logger_name and not level:
        # Get the level of the specified logger
        target_logger = logging.getLogger(logger_name)
        return PlainTextResponse(
            content=f"{logger_name}: {logging.getLevelName(target_logger.level)}",
            media_type="text/plain",
        )
    elif logger_name and level:
        # Set the level of the specified logger
        target_logger = logging.getLogger(logger_name)
        try:
            level_value = getattr(logging, level.upper())
            target_logger.setLevel(level_value)
            # Set the level of all handlers
            for handler in target_logger.handlers:
                handler.setLevel(level_value)
            return PlainTextResponse(
                content=f"Set {logger_name} level to {level.upper()} "
                "(including all handlers)",
                media_type="text/plain",
            )
        except AttributeError:
            return PlainTextResponse(
                content=f"Invalid log level: {level}",
                media_type="text/plain",
                status_code=400,
            )


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
