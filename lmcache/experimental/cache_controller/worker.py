import os
import fcntl
import json
import asyncio
import threading

from lmcache.experimental.cache_engine import LMCacheEngine
from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from rpc_utils import (get_server_socket, get_zmq_context, 
                       get_unix_socket_path, clean_old_sockets)
from lmcache.logging import init_logger

logger = init_logger(__name__)

class LMCacheWorker:
    def __init__(
        self, 
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        lmcache_engine: LMCacheEngine,
    ):
        self.lmcache_instance_id = config.lmcache_instance_id
        self.lmcache_engine = lmcache_engine
        self.worker_id = metadata.worker_id
        self.metadata_path = \
            f"/tmp/lmcache_instance_metadata_{self.lmcache_instance_id}.json"
        self._write_metadata()
        
        self.context = get_zmq_context()
        self.socket_path = get_unix_socket_path(
            self.lmcache_instance_id, self.worker_id)
        self.socket = get_server_socket(self.context, self.socket_path)
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start(), self.loop)
    
    async def handle_request(self):
        while True:
            try:
                request = await self.socket.recv_json()
                operation = request.get("operation")
                data = request.get("data", {})
                logger.info(f"Received operation: {operation}, data: {data}")

                if hasattr(self.lmcache_engine, operation):
                    method = getattr(self.lmcache_engine, operation)
                    result = await method(**data)
                else:
                    result = {"error": f"Unsupported operation '{operation}'"}

                await self.socket.send_json({"res": result})
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await self.socket.send_json({"error": str(e)})

    def _write_metadata(self):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        with open(self.metadata_path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                metadata = json.load(f)
                metadata[self.lmcache_instance_id] = \
                    metadata.get(self.lmcache_instance_id, [])
                metadata[self.lmcache_instance_id].append(self.worker_id)
                f.seek(0)
                json.dump(metadata, f)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    
    async def start(self):
        try:
            logger.info(f"Starting lcache worker {self.worker_id}"
                        f"for instance {self.lmcache_instance_id}")
            await self.handle_request()
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            clean_old_sockets(self.lmcache_instance_id, self.worker_id)