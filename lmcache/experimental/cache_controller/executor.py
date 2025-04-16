import asyncio
import fcntl
import json
import os
from collections import Counter
from typing import Dict, List, Optional

from lmcache.experimental.cache_controller.rpc_utils import (
    get_client_socket, get_unix_socket_path, get_zmq_context)
from lmcache.logging import init_logger

logger = init_logger(__name__)


# NOTE(Jiayi): `LMCacheClusterExecutor` might need to be in different processes
# in the future for the sake of performance.
# NOTE(Jiayi): Also, consider scaling up the number of cluster executors
# in the future.
class LMCacheClusterExecutor:
    """
    LMCache Cluster Executor class to handle the execution of cache operations.
    """

    def __init__(self, lmcache_instance_ids: List[str]):
        """
        Initialize the LMCache Executor with a cache instance.

        :param lmcache_instance_id: lmcache_instance_id
        """
        assert len(
            lmcache_instance_ids
        ) == 1, "Multiple lmcache instances are not supported for now"

        lmcache_instance_id = lmcache_instance_ids[0]

        # TODO(Jiayi): need to make this an argument
        self.instance_metadata_path = \
            f"/tmp/lmcache_instance_metadata_{lmcache_instance_id}.json"
        self.context = get_zmq_context()
        self.instances: Dict[str, LMCacheInstanceExecutor] = {}
        self.instance_lock = asyncio.Lock()

        self.last_mtime: Optional[float] = None
        #asyncio.create_task(self._monitor_metadata_loop())

    async def monitor_metadata_loop(self, poll_interval: float = 10.0):
        while True:
            try:
                if os.path.exists(self.instance_metadata_path):
                    mtime = os.path.getmtime(self.instance_metadata_path)
                    if self.last_mtime != mtime:
                        self.last_mtime = mtime
                        await self._update_instances()
            except Exception as e:
                logger.error(f"Metadata monitoring error: {e}")
            await asyncio.sleep(poll_interval)

    async def _update_instances(self):
        try:
            # TODO(Jiayi): file read can be async
            with open(self.instance_metadata_path, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                try:
                    instance_metadata = json.load(f)
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)

            for instance_id, worker_ids in instance_metadata.items():
                if instance_id not in self.instances:
                    async with self.instance_lock:
                        self.instances[instance_id] = LMCacheInstanceExecutor(
                            instance_id, worker_ids)
                    logger.info(f"Added instance {instance_id}, "
                                f"worker_ids: {worker_ids}")
                    continue
                if Counter(self.instances[instance_id].worker_ids) != Counter(
                        worker_ids):
                    # TODO(Jiayi): We can make this incremental
                    # instead of re-initializing
                    async with self.instance_lock:
                        self.instances[instance_id] = LMCacheInstanceExecutor(
                            instance_id, worker_ids)
                    logger.info(f"Updated instance {instance_id}, "
                                f"worker_ids: {worker_ids}")
        except Exception as e:
            print(f"Error in updating instances: {e}")

    async def execute(self, instance_id, operation, **kwargs):
        """
        Execute a cache operation with error handling.

        :param instance_id: The ID of the cache instance.
        :param operation: The operation to execute 
        (e.g., 'lookup', 'clear_all').
        :param kwargs: Keyword arguments for the operation.
        :return: The result of the operation or an error message.
        """
        try:
            instance = self.instances[instance_id]
            return await instance.execute(operation, **kwargs)
        except KeyError:
            return {"error": f"Instance ID '{instance_id}' not found."}
        except Exception as e:
            return {"error": str(e)}


class LMCacheInstanceExecutor:
    """
    LMCache Instance Executor class to handle the execution of cache operations.
    """

    def __init__(self, instance_id: str, worker_ids: List[int]):
        """
        Initialize the LMCache Executor with a cache instance.

        """
        self.context = get_zmq_context()
        self.sockets = {}
        self.worker_ids = worker_ids
        for worker_id in worker_ids:
            socket_path = get_unix_socket_path(instance_id, worker_id)
            self.sockets[worker_id] = get_client_socket(
                self.context, socket_path)

    async def lookup(self, **kwargs):
        """
        Perform a lookup operation on the cache.

        :param worker_ids: The IDs of the worker.
        :param tokens: The IDs of the tokens to look up.
        :return: The result of the lookup operation.
        """

        assert "worker_ids" in kwargs
        worker_ids = kwargs["worker_ids"]
        # Only queries the first worker if worker_ids is not specified
        if len(worker_ids) == 0:
            worker_ids = [0]

        if "tokens" not in kwargs:
            raise ValueError("tokens is required in lookup")
        print(f"worker_ids: {worker_ids}")
        res = await self.execute_worker("lookup",
                                        worker_ids,
                                        tokens=kwargs["tokens"])

        return res

    async def clear(self, **kwargs):
        """
        Perform a clear operation on the cache.

        :param worker_ids: The IDs of the worker.
        :param tokens: The IDs of the tokens to look up.
        :return: The number of cleared requests.
        """

        assert "worker_ids" in kwargs
        worker_ids = kwargs["worker_ids"]
        # Queries all workers if worker_ids is not specified
        if len(worker_ids) == 0:
            worker_ids = self.worker_ids

        if "tokens" not in kwargs:
            raise ValueError("tokens is required in lookup")

        res = await self.execute_worker("clear",
                                        worker_ids,
                                        tokens=kwargs["tokens"])

        return res

    async def execute_worker(self, operation: str, worker_ids: List[int],
                             **kwargs):
        """
        Execute an operation on the specified list of workers.
        :param operation: The operation to execute 
        (e.g., 'lookup', 'clear_all').
        :param worker_ids: The list of worker IDs to execute the operation on.
        :param kwargs: Keyword arguments for the operation.
        :return: A dictionary mapping worker IDs to their respective results.
        """
        tasks = []
        for worker_id in worker_ids:

            async def send_and_receive(s):
                await s.send_json({"operation": operation, "data": kwargs})
                return await s.recv_json()

            socket = self.sockets[worker_id]
            tasks.append(send_and_receive(socket))

        results = await asyncio.gather(*tasks)
        return {
            worker_id: result
            for worker_id, result in zip(worker_ids, results)
        }

    async def execute(self, operation, **kwargs):
        """
        Execute a cache operation with error handling.

        :param operation: The operation to execute 
        (e.g., 'lookup', 'clear_all').
        :param kwargs: Keyword arguments for the operation.
        :return: The result of the operation or an error message.
        """
        try:
            method = getattr(self, operation)
            return await method(**kwargs)
        except AttributeError:
            return {"error": f"Operation '{operation}' is not supported."}
        except Exception as e:
            return {"error": str(e)}
