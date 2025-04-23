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

from dataclasses import dataclass

from lmcache.experimental.cache_controller.message import (  # noqa: E501
    ClearMsg, ClearRetMsg, KVAdmitMsg, KVEvictMsg, LookupMsg, LookupRetMsg)
from lmcache.experimental.token_database import ChunkedTokenDatabase


@dataclass
class KVChunkMetadata:
    """
    A class representing a KV chunk metadata.
    """
    instance_id: str
    worker_id: int
    location: str


# TODO(Jiayi): Need more efficient data structures (e.g., trie)
# to handle these operations (e.g., evict, deregister)
# more efficiently.


class KVController:

    def __init__(self):
        self.kv_pool: dict[str, list[KVChunkMetadata]] = {}

        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()
        self.token_database.chunk_size = 256

    def post_init(self, cluster_executor):
        """
        Post initialization of the KV controller.
        """
        self.cluster_executor = cluster_executor

    async def admit(self, msg: KVAdmitMsg) -> None:
        """
        Admit a new kv chunk.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        key = msg.key
        location = msg.location
        if instance_id not in self.kv_pool:
            self.kv_pool[key] = []
        self.kv_pool[key].append(
            KVChunkMetadata(instance_id, worker_id, location))

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        key = msg.key
        location = msg.location

        if key not in self.kv_pool:
            return

        remaining = [
            m for m in self.kv_pool[key]
            if not (m.instance_id == instance_id and m.worker_id == worker_id
                    and m.location == location)
        ]

        if remaining:
            self.kv_pool[key] = remaining
        else:
            del self.kv_pool[key]

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear all kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        """
        Deregister all kv chunks of an instance-worker.
        """
        for key in self.kv_pool:
            self.kv_pool[key] = [
                m for m in self.kv_pool[key] if not (
                    m.instance_id == instance_id and m.worker_id == worker_id)
            ]
            if not self.kv_pool[key]:
                del self.kv_pool[key]

    # TODO(Jiayi): The current implementation does not handle
    # the case where the prefix chunks are evicted while the
    # suffix chunk is still in the system. LMCache should guarantee
    # this does not happen.
    # TODO(Jiayi): The current implementation does not consider
    # the location of the kv chunks. It simply returns the
    # `instance_id` with longest prefix.
    # TODO(Jiayi): Need to get rid of the hash somehow
    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        target_instance = None
        tokens = msg.tokens
        for start, end, key in self.token_database.process_tokens(
                tokens, make_key=False):
            assert isinstance(key, str)
            if key not in self.kv_pool:
                break
            target_instance = self.kv_pool[key][0].instance_id
        return LookupRetMsg(target_instance)
