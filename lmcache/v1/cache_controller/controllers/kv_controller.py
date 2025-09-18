# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# First Party
from lmcache.v1.cache_controller.message import (
    CheckFinishMsg,
    CheckFinishRetMsg,
    ClearMsg,
    ClearRetMsg,
    CompressMsg,
    CompressRetMsg,
    DecompressMsg,
    DecompressRetMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    LookupRetMsg,
    MoveMsg,
    MoveRetMsg,
    PinMsg,
    PinRetMsg,
)
from lmcache.v1.token_database import ChunkedTokenDatabase


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
    def __init__(self) -> None:
        # NOTE (Jiayi): Even if we offload kv_pool to
        # redis. We might need a local cache for handling
        # messages like `check_finish`. Or everything should be
        # written to redis.
        self.kv_pool: dict[str, list[KVChunkMetadata]] = {}

        # TODO(Jiayi): remove this hardcode
        self.token_database = ChunkedTokenDatabase()

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
        key = str(msg.key)
        location = msg.location
        if key not in self.kv_pool:
            self.kv_pool[key] = []
        self.kv_pool[key].append(KVChunkMetadata(instance_id, worker_id, location))

    async def evict(self, msg: KVEvictMsg) -> None:
        """
        Evict a kv chunk.
        """
        instance_id = msg.instance_id
        worker_id = msg.worker_id
        key = str(msg.key)
        location = msg.location

        if key not in self.kv_pool:
            return

        remaining = [
            m
            for m in self.kv_pool[key]
            if not (
                m.instance_id == instance_id
                and m.worker_id == worker_id
                and m.location == location
            )
        ]

        if remaining:
            self.kv_pool[key] = remaining
        else:
            del self.kv_pool[key]

    async def clear(self, msg: ClearMsg) -> ClearRetMsg:
        """
        Clear kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("clear", msg)

    async def pin(self, msg: PinMsg) -> PinRetMsg:
        """
        Pin kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("pin", msg)

    async def compress(self, msg: CompressMsg) -> CompressRetMsg:
        """
        Compress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("compress", msg)

    async def decompress(self, msg: DecompressMsg) -> DecompressRetMsg:
        """
        Decompress kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("decompress", msg)

    async def move(self, msg: MoveMsg) -> MoveRetMsg:
        """
        Move kv chunks of instance-worker(s).
        """
        return await self.cluster_executor.execute("move", msg)

    async def check_finish(self, msg: CheckFinishMsg) -> CheckFinishRetMsg:
        """
        Check if an event is finished.
        """
        return await self.cluster_executor.execute("check_finish", msg)

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        """
        Deregister all kv chunks of an instance-worker.
        """
        for key in self.kv_pool:
            self.kv_pool[key] = [
                m
                for m in self.kv_pool[key]
                if not (m.instance_id == instance_id and m.worker_id == worker_id)
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
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            key = str(key)
            if key not in self.kv_pool:
                break
            matched_instance = self.kv_pool[key][0].instance_id
            matched_location = self.kv_pool[key][0].location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)
