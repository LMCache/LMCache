# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING
import asyncio
import os
import threading

# Third Party
from fastapi import FastAPI
import uvicorn

# First Party
from lmcache.logging import init_logger

# Local
from .api_registry import APIRegistry

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

logger = init_logger(__name__)

app = FastAPI()

# Automatically register all APIs
registry = APIRegistry(app)
registry.register_all_apis()


class InternalAPIServer:
    def __init__(self, lmcache_adapter: "LMCacheConnectorV1Impl"):
        config = lmcache_adapter.config
        lmcache_engine = lmcache_adapter.lmcache_engine
        # 0 for scheduler, 1 for worker 0, 2 for worker 1, ...
        port_offset = 0 if not lmcache_engine else 1 + lmcache_engine.metadata.worker_id
        self.port = config.internal_api_server_port_start + port_offset
        self.socket_path_prefix = config.internal_api_server_socket_path_prefix
        self.socket_path = (
            f"{self.socket_path_prefix}_{self.port}"
            if self.socket_path_prefix
            else None
        )
        include_index_list = config.internal_api_server_include_index_list

        self.enable = True
        if not config.internal_api_server_enabled or (
            include_index_list and port_offset not in include_index_list
        ):
            logger.info(
                f"Internal API server disabled. internal_api_server_enabled="
                f"{config.internal_api_server_enabled}, port_offset={port_offset}, "
                f"port={self.port}, socket_path={self.socket_path}, "
                f"include_index_list={include_index_list}"
            )
            self.enable = False
            return

        uvicorn_config = {
            "app": app,
            "host": config.internal_api_server_host,
            "loop": "uvloop",
            "http": "httptools",
        }

        if self.socket_path:
            self.server_log_info = f"socket {self.socket_path}"
            logger.info(f"Init internal API server on {self.server_log_info}")
            uvicorn_config["uds"] = self.socket_path
            # Ensure socket directory exists
            os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
            # Remove existing socket file if exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
        else:
            self.server_log_info = f"port {self.port}"
            logger.info(f"Init internal API server on {self.server_log_info}")
            uvicorn_config["port"] = self.port

        self.server = uvicorn.Server(uvicorn.Config(**uvicorn_config))
        app.state.lmcache_adapter = lmcache_adapter

    async def run(self):
        logger.info(f"Running LMCache internal API server on {self.server_log_info}")
        if self.server:
            await self.server.serve()

    def start(self):
        if not self.enable:
            return
        logger.info(f"Starting LMCache internal API server on {self.server_log_info}")
        threading.Thread(target=asyncio.run, args=(self.run(),), daemon=True).start()

    def stop(self):
        if not self.enable:
            return
        logger.info("Stopping LMCache internal API server")
        if self.server:
            self.server.should_exit = True
            if self.socket_path and os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
