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

from typing import Optional

import zmq
import zmq.asyncio

from lmcache.experimental.cache_controller.message import (DeRegisterMsg,
                                                           RegisterMsg)
from lmcache.experimental.cache_controller.rpc_utils import (close_zmq_socket,
                                                             get_zmq_context,
                                                             get_zmq_socket)
from lmcache.logging import init_logger

logger = init_logger(__name__)


class RegistrationController:

    def __init__(self):

        # Mapping from `instance_id` -> `worker_ids`
        self.worker_mapping: dict[str, list[int]] = {}
        # Mapping from `(instance_id, worker_id)` -> `url`
        self.socket_mapping: dict[tuple[str, int], zmq.asyncio.Socket] = {}

    def post_init(self, kv_controller, cluster_executor):
        """
        Post initialization of the Registration Controller.
        """
        self.kv_controller = kv_controller
        self.cluster_executor = cluster_executor

    def get_socket(self, instance_id: str,
                   worker_id: int) -> Optional[zmq.asyncio.Socket]:
        """
        Get the socket for a given instance and worker ID.
        """
        socket = self.socket_mapping.get((instance_id, worker_id))
        if socket is None:
            logger.warning(f"Instance-worker {(instance_id, worker_id)}"
                           " not registered")
        return socket

    def get_workers(self, instance_id: str) -> list[int]:
        """
        Get worker ids given an instance id.
        """
        return self.worker_mapping.get(instance_id, [])

    async def register(self, msg: RegisterMsg) -> None:
        """
        Register a new instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        url = msg.url

        context = get_zmq_context()
        socket = get_zmq_socket(context, url, protocol="tcp",
                                role=zmq.REQ)  # type: ignore[attr-defined]

        self.socket_mapping[(instance_id, worker_id)] = socket
        if instance_id not in self.worker_mapping:
            self.worker_mapping[instance_id] = []
        self.worker_mapping[instance_id].append(worker_id)
        logger.info(f"Registered instance-worker {(instance_id, worker_id)}"
                    f" with URL {url}")

    async def deregister(self, msg: DeRegisterMsg) -> None:
        """
        Deregister an instance-worker connection mapping.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        if instance_id in self.worker_mapping:
            self.worker_mapping[instance_id].remove(worker_id)
            if not self.worker_mapping[instance_id]:
                del self.worker_mapping[instance_id]
        else:
            logger.warning(f"Instance {instance_id} not registered")

        if (instance_id, worker_id) in self.socket_mapping:
            socket = self.socket_mapping.pop((instance_id, worker_id))
            close_zmq_socket(socket)
            self.kv_controller.deregister(instance_id, worker_id)
            logger.info(
                f"Deregistered instance-worker {(instance_id, worker_id)}")
        else:
            logger.warning(f"Instance-worker {(instance_id, worker_id)}"
                           "not registered")
