import asyncio
import fcntl
import inspect
import json
import os
import threading

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.cache_controller.rpc_utils import (
    clean_old_sockets, get_server_socket, get_unix_socket_path,
    get_zmq_context)
from lmcache.experimental.cache_engine import LMCacheEngine
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.logging import init_logger

logger = init_logger(__name__)


class LMCacheWorker:
    """
    LMCache Worker class to handle the execution of cache operations.
    This class is responsible for receiving requests from the executor and
    executing the corresponding operations on the LMCache engine.
    Each worker is associated with a specific LMCache instance and a worker id.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        lmcache_engine: LMCacheEngine,
    ):
        self.lmcache_instance_id = config.lmcache_instance_id
        assert self.lmcache_instance_id is not None
        self.lmcache_engine = lmcache_engine
        self.worker_id = metadata.worker_id
        self.metadata_path = \
            f"/tmp/lmcache_instance_metadata_{self.lmcache_instance_id}.json"
        self._update_metadata()

        self.context = get_zmq_context()
        self.socket_path = get_unix_socket_path(self.lmcache_instance_id,
                                                self.worker_id)
        self.socket = get_server_socket(self.context, self.socket_path)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever,
                                       daemon=True)
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
                    if inspect.iscoroutinefunction(method):
                        result = await method(**data)
                    else:
                        result = method(**data)
                else:
                    result = {"error": f"Unsupported operation '{operation}'"}
                logger.info(f"Operation result: {result}")
                await self.socket.send_json({"res": result})
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await self.socket.send_json({"error": str(e)})

    def _update_metadata(self, mode="append"):
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        try:
            with open(self.metadata_path, "x") as f:
                json.dump({}, f)
        except FileExistsError:
            pass
        with open(self.metadata_path, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                metadata = json.load(f)
                if mode == "append":
                    metadata[self.lmcache_instance_id] = \
                        metadata.get(self.lmcache_instance_id, [])
                    if self.worker_id not in metadata[
                            self.lmcache_instance_id]:
                        metadata[self.lmcache_instance_id].append(
                            self.worker_id)
                elif mode == "remove":
                    if self.lmcache_instance_id in metadata:
                        metadata[self.lmcache_instance_id].remove(
                            self.worker_id)
                        if not metadata[self.lmcache_instance_id]:
                            del metadata[self.lmcache_instance_id]
                f.seek(0)
                json.dump(metadata, f)
                f.truncate()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    async def start(self):
        try:
            logger.info(f"Starting lmcache worker {self.worker_id}"
                        f"for instance {self.lmcache_instance_id}")
            await self.handle_request()
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            assert self.lmcache_instance_id is not None
            clean_old_sockets(self.lmcache_instance_id, [self.worker_id])

    def close(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()

        self._update_metadata(mode="remove")
        assert self.lmcache_instance_id is not None
        clean_old_sockets(self.lmcache_instance_id, [self.worker_id])
