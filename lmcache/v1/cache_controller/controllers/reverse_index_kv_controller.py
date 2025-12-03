# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict, deque
from dataclasses import dataclass

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_controller.controllers import KVController
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
    KVAdmitMsg,
    KVEvictMsg,
    LookupMsg,
    LookupRetMsg,
)

logger = init_logger(__name__)


@dataclass
class KVChunkMetadata:
    """
    A class representing a KV chunk metadata.
    """

    instance_id: str
    worker_id: int
    location: str

    def __hash__(self) -> int:
        """
        Hash method.
        """
        return hash((self.instance_id, self.worker_id, self.location))

    def __eq__(self, other) -> bool:
        """
        Equality comparison method.
        """
        if not isinstance(other, KVChunkMetadata):
            return False
        return (
            self.instance_id == other.instance_id
            and self.worker_id == other.worker_id
            and self.location == other.location
        )


"""
The reverse index kv controller is a kv controller that maintains a reverse index.
When the the number of instance is unknown or large, the time complexity of `lookup` 
in normal kv controller is O(n^2), while the reverse index kv controller is O(n). 
"""


class ReverseIndexKVController(KVController):
    def __init__(self):
        super().__init__()
        # Mapping from `chunk_hash` -> deque[KVChunkMetadata]
        self.reverse_index: dict[int, deque[KVChunkMetadata]] = defaultdict(deque)
        logger.info("created reverse index kv controller")

    async def admit(self, msg: KVAdmitMsg) -> None:
        await super().admit(msg)
        chunk_meta = KVChunkMetadata(msg.instance_id, msg.worker_id, msg.location)
        self.reverse_index[msg.key].append(chunk_meta)

    async def evict(self, msg: KVEvictMsg) -> None:
        await super().evict(msg)

        if msg.key not in self.reverse_index:
            return

        chunk_meta = KVChunkMetadata(msg.instance_id, msg.worker_id, msg.location)
        try:
            self.reverse_index[msg.key].remove(chunk_meta)
        except ValueError:
            pass
        if not self.reverse_index[msg.key]:
            del self.reverse_index[msg.key]

    async def deregister(self, instance_id: str, worker_id: int) -> None:
        for location, keys in self.kv_pool[(instance_id, worker_id)].items():
            for key in keys:
                if key not in self.reverse_index:
                    continue
                chunk_meta = KVChunkMetadata(instance_id, worker_id, location)
                try:
                    self.reverse_index[key].remove(chunk_meta)
                except ValueError:
                    pass
                if not self.reverse_index[key]:
                    del self.reverse_index[key]
        await super().deregister(instance_id, worker_id)

    async def lookup(self, msg: LookupMsg) -> LookupRetMsg:
        tokens = msg.tokens
        layout_info = {}
        for start, end, key in self.token_database.process_tokens(
            tokens, make_key=False
        ):
            if key not in self.reverse_index:
                break
            matched_instance = self.reverse_index[key][0].instance_id
            matched_location = self.reverse_index[key][0].location
            layout_info[matched_instance] = (matched_location, end)
        return LookupRetMsg(layout_info=layout_info, event_id=msg.event_id)

    async def batched_p2p_lookup(
        self, msg: BatchedP2PLookupMsg
    ) -> BatchedP2PLookupRetMsg:
        worker_id = msg.worker_id
        query_instance_id = msg.instance_id
        num_hit_chunks = 0
        instance_id = ""
        location = ""
        peer_init_url = ""
        for key in msg.hashes:
            # TODO(Jiayi): remove this string conversion
            if key not in self.reverse_index:
                break

            # TODO(Jiayi): Currently, we use the first matched
            # kv chunk metadata to do matching. The matching
            # logic can be improved.
            # TODO(Jiayi): The KV Cache could be from different
            # instances. We need to handle this case as well.
            matched_kv_chunk_meta = None
            for kv_chunk_meta in self.reverse_index[key]:
                if kv_chunk_meta.instance_id != query_instance_id:
                    # Found a matching instance_id that's not the
                    # same as the query_instance_id.
                    matched_kv_chunk_meta = kv_chunk_meta
                    break

            if matched_kv_chunk_meta is None:
                break
            if instance_id != "" and (
                instance_id != matched_kv_chunk_meta.instance_id
                or location != matched_kv_chunk_meta.location
            ):
                # We have already found a different instance_id
                # before. Stop here.
                break
            elif instance_id == "":
                instance_id = matched_kv_chunk_meta.instance_id
                location = matched_kv_chunk_meta.location
                peer_init_url = self.reg_controller.get_distributed_url(
                    instance_id, worker_id
                )
                assert peer_init_url is not None
            num_hit_chunks += 1

        return BatchedP2PLookupRetMsg(
            layout_info=[
                (instance_id, location, num_hit_chunks, peer_init_url),
            ]
        )
