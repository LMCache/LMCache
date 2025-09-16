# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Sequence
import abc

# Third Party
import torch
import msgspec

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.storage_backend.storage_backend_listener import StorageBackendListener
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket


class P2PMsgBase(msgspec.Struct, tag=True):
    """Base class for all P2P-related messages"""

    pass

class LookupAndRetrieveMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Lookup id
    lookup_id: str

    #CacheEngineKey in str form
    keys: list[str]

    # Indexes (remote) of allocated memory objects (to be written)
    local_indexes: list[int]

class LookupAndRetrieveMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # CacheEngineKey in string form
    keys: list[str]

    # Indexes (remote) of allocated memory objects (to be written)
    local_indexes: list[int]

class LookupAndRetrieveRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of hit chunks
    num_hit_chunks: int

# NOTE(Jiayi): Several notes about P2PBackend:
# 1. Put is not supported for now.
# 2. Only async contains and async get are supported.
# 3. Lookup is currently three-tier:
#    (1) local (local lookup cache) -> remote peer (goto (3) if hit)
#    (2) controller -> remote peer (goto (3) if hit)
#    (3) remote peer -> kv abd real retrieved lengths

# TODO(Jiayi): handle asymmetric TP.
class P2PBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
    ):
        super().__init__(dst_device=dst_device)
        self.config = config
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.lmcache_worker = lmcache_worker
        
        # A CacheEngineKey (in int form) -> a list of 
        # (peer_init_url, peer_url ,location)
        self.local_lookup_cache = {}
        # A set of peer_init_urls
        self.initialized_peers: set[str] = set()

        # A lookup_id -> (peer_init_url, peer_url, location)
        self.lookup_id_to_peer_mapping: dict[str, tuple[str, str]] = {}

        # FIXME: 
        self.tp_rank
        self.transfer_channel

        self.allocator_backend = 
        self.full_size_shape =
        self.fmt: MemoryFormat =
        self.dtype =

        self.context = 
        self.async_peer_socket =

        self.running = True

        # TODO(Jiayi): support local storage as well.
        self.local_cpu_backend = 

    # TODO(Jiayi): make this async function as well.
    def _ensure_peer_connection(
        self,
        peer_id: str,
    ) -> None:
        pass


    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:

        # Convert to hashes (int form)
        hashes = [key.chunk_hash for key in keys]

        # Tier 1 lookup: local lookup cache
        # TODO(Jiayi): Please implement the local lookup cache.
        
        # Tier 2 lookup in controller
        msg = BatchedP2PLookupMsg(
            worker_id=self.tp_rank,
            hashes=hashes,
        )
        ret_msg = await self.lmcache_worker.async_put_and_wait_msg(msg)
        assert isinstance(ret_msg, BatchedP2PLookupRetMsg)
        
        # NOTE(Jiayi): For now we only support one peer hit.
        layout_info = ret_msg.layout_info[0]
        _, location, num_hit_chunks, peer_init_url, peer_url = layout_info

        if num_hit_chunks > 0:
            self.lookup_id_to_peer_mapping[lookup_id] = (
                peer_init_url, peer_url, location
            )


        # TODO(Jiayi): We could potentially update the local cache here.
        # Or we can update after tier 3 lookup.
        
        # NOTE(Jiayi): Tier 3 lookup is in function 
        # `batched_get_non_blocking`. 

        return num_hit_chunks
    
    async def _handle_batched_get_non_blocking(self):
        """
        Handle `LookupAndRetrieveMsg` issued by peers in `batched_get_non_blocking`.
        """
        while self.running:
            msg_bytes = await self.async_peer_socket.recv()
            msg = msgspec.decode(msg_bytes, type=P2PMsg)
            assert isinstance(msg, LookupAndRetrieveMsg)

            lookup_id = msg.lookup_id
            keys = [CacheEngineKey.from_str(key) for key in msg.keys]

            # TODO(Jiayi): Optimally, there's no need to use async call
            # for some backends (e.g., local cpu) as there's overhead for 
            # async function call.
            num_hit_chunks = await self.local_cpu_backend.batched_async_contains(

            )
            
            self.local_cpu_backend.batched_get_non_blocking()




    async def _ensure_peer_connection(
        self,
        peer_init_url: str,
    ) -> None:
        if peer_init_url in self.initialized_peers:
            return
        
        await self.transfer_channel.async_lazy_init_peer_connection(
            peer_id=peer_init_url,
            peer_init_url=peer_init_url)
        
        self.initialized_peers.add(peer_init_url)


    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        offsets: list[int],
    ) -> list[MemoryObj]:

        peer_init_url, peer_url, location = self.lookup_id_to_peer_mapping[lookup_id]
        await self._ensure_peer_connection(peer_init_url)
        
        mem_objs = []
        for key in keys:
            shape = self.full_size_shape[self.fmt.token_dim()]

            # FIXME: check signiture
            mem_obj = self.allocator_backend.allocate(
                torch.Size(shape), self.dtype, self.fmt)
            mem_objs.append(mem_obj)
        
        local_indexes = self.transfer_channel.get_local_mem_indices(mem_objs)

        msg = LookupAndRetrieveMsg(
            keys=[str(key) for key in keys],
            local_indexes=local_indexes,
        )

        ret_msg = await self.async_peer_socket.send(
            msgspec.encode(msg, type=P2PMsg)
        )

        num_hit_chunks = ret_msg.num_hit_chunks

        hit_mem_objs = mem_objs[:num_hit_chunks]
        for missed_mem_obj in mem_objs[num_hit_chunks:]:
            missed_mem_obj.ref_count_down()

        return hit_mem_objs
        
        
        
    
    # NOTE: synchronous contain is not supported for now.
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        raise NotImplementedError

    # NOTE: put-related functions are not supported for now.
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    # NOTE: put-related functions are not supported for now.
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        raise NotImplementedError

    # NOTE: Synchronous get is not supported for now.
    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        raise NotImplementedError

    # NOTE: pin is useless for P2P backend now.
    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    # NOTE: unpin is useless for P2P backend now.
    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        return False

    # NOTE: remove is useless for P2P backend now.
    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return False

    def close(
        self,
    ) -> None:
        """
        Close the P2P backend.
        """
        pass