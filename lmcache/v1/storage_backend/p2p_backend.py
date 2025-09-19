# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union
import asyncio

# Third Party
import msgspec
import torch
import zmq

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedMixedMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    P2PInitSideRetMsg,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.worker import LMCacheWorker

logger = init_logger(__name__)


class P2PMsgBase(msgspec.Struct, tag=True):
    """Base class for all P2P-related messages"""

    pass


class BatchedLookupAndGetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    receiver_id: str

    # CacheEngineKey in string form
    keys: list[str]

    # Indexes (remote) of allocated memory objects (to be written)
    mem_indexes: list[int]


class BatchedLookupAndGetRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of hit chunks
    num_hit_chunks: int


P2PMsg = Union[
    BatchedLookupAndGetMsg,
    BatchedLookupAndGetRetMsg,
]

# NOTE(Jiayi): Several notes about P2PBackend:
# 1. Put is not supported for now.
# 2. Only async contains and async get are supported.
# 3. Lookup is currently three-tier:
#    (1) local (local lookup cache) -> remote peer (goto (3) if hit)
#    (2) controller -> remote peer (goto (3) if hit)
#    (3) remote peer -> kv and real retrieved lengths


# TODO(Jiayi): handle asymmetric TP.
class P2PBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        lmcache_worker: "LMCacheWorker",
    ):
        self.config = config
        self.loop = loop
        self.lmcache_worker = lmcache_worker

        assert config.peer_init_host is not None, "peer_init_host must be specified"
        assert config.peer_init_ports is not None, "peer_init_port must be specified"

        # tp rank is worker id for now
        self.tp_rank = metadata.worker_id

        self.peer_init_host = config.p2p_init_host
        self.peer_init_port = config.p2p_init_ports[self.tp_rank]
        self.peer_init_url = f"{self.peer_init_host}:{self.peer_init_port}"

        self.peer_lookup_host = config.peer_lookup_host
        self.peer_lookup_port = config.peer_lookup_ports[self.tp_rank]
        self.peer_lookup_url = f"{self.peer_lookup_host}:{self.peer_lookup_port}"

        # A CacheEngineKey (in int form) -> a list of
        # (peer_init_url, peer_lookup_url, location)
        self.local_lookup_cache: dict[int, tuple[str, str, str]] = {}
        # A set of peer_init_urls
        self.peer_id_to_lookup_url_mapping: dict[str, str] = {}

        # A lookup_id -> (peer_init_url, peer_lookup_url, location)
        self.lookup_id_to_peer_mapping: dict[str, tuple[str, str, str]] = {}

        self.dtype = metadata.kv_dtype
        self.full_size_shape = list(metadata.kv_shape)
        # TODO(Jiayi): remove this hardcode
        self.fmt: MemoryFormat = MemoryFormat.KV_2LTD

        # TODO(Jiayi): support gpu and local storage p2p as well.
        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedMixedMemoryAllocator)

        # FIXME: We need to change buffer_ptr...
        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            role="both",
            buffer_ptr=self.memory_allocator.cpu_allocator.buffer_ptr,
            buffer_size=self.memory_allocator.cpu_allocator.buffer_size,
            align_bytes=self.memory_allocator.cpu_allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=self.peer_init_url,
            peer_lookup_url=self.peer_lookup_url,
            backends=config.nixl_backends,
            event_loop=loop,
        )

        self.context = get_zmq_context()
        self.async_peer_socket = get_zmq_socket(
            self.context,
            self.peer_lookup_url,
            "tcp",
            zmq.REP,
            "bind",
        )

        self.running = True

        # FIXME: UCX_TLS=rc to enbal infiniband

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
        _, location, num_hit_chunks, peer_init_url = layout_info

        if num_hit_chunks > 0:
            await self._ensure_peer_connection(peer_init_url)
            self.lookup_id_to_peer_mapping[lookup_id] = (
                peer_init_url,
                self.peer_id_to_lookup_url_mapping[peer_init_url],
                location,
            )

        # TODO(Jiayi): We could potentially update the local cache here.
        # Or we can update after tier 3 lookup.

        # NOTE(Jiayi): Tier 3 lookup is in function
        # `batched_get_non_blocking`.

        return num_hit_chunks

    async def _handle_batched_get_non_blocking(self):
        """
        Handle `BatchedLookupAndGetMsg` issued by peers in `batched_get_non_blocking`.
        """
        while self.running:
            msg_bytes = await self.async_peer_socket.recv()
            msg = msgspec.decode(msg_bytes, type=P2PMsg)
            assert isinstance(msg, BatchedLookupAndGetMsg)

            lookup_id = msg.lookup_id
            keys = [CacheEngineKey.from_str(key) for key in msg.keys]

            # TODO(Jiayi): Optimally, there's no need to use async call
            # for some backends (e.g., local cpu) as there's overhead for
            # async function call.
            num_hit_chunks = await self.local_cpu_backend.batched_async_contains(
                lookup_id=lookup_id,
                keys=keys,
                pin=True,
            )

            mem_objs = await self.local_cpu_backend.batched_get_non_blocking(
                lookup_id=lookup_id,
                keys=keys[:num_hit_chunks],
            )

            channel_transfer_spec = {
                "receiver_id": msg.receiver_id,
                "remote_indexes": msg.mem_indexes[:num_hit_chunks],
            }

            await self.transfer_channel.async_batched_write(
                data=mem_objs,
                transfer_spec=channel_transfer_spec,
            )

            ret_msg = BatchedLookupAndGetRetMsg(
                num_hit_chunks=num_hit_chunks,
            )

            await self.async_peer_socket.send(msgspec.msgpack.encode(ret_msg))

            for mem_obj in mem_objs:
                mem_obj.ref_count_down()
                mem_obj.unpin()

    async def _ensure_peer_connection(
        self,
        peer_init_url: str,
    ) -> None:
        if peer_init_url in self.peer_id_to_lookup_url_mapping:
            return

        init_ret_msg = await self.transfer_channel.async_lazy_init_peer_connection(
            peer_id=peer_init_url, peer_init_url=peer_init_url
        )
        assert isinstance(init_ret_msg, P2PInitSideRetMsg)
        peer_lookup_url = init_ret_msg.peer_lookup_url
        self.peer_id_to_lookup_url_mapping[peer_init_url] = peer_lookup_url

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        peer_init_url, peer_lookup_url, location = self.lookup_id_to_peer_mapping.pop(
            lookup_id
        )

        assert isinstance(transfer_spec, dict)
        cum_chunk_lengths = transfer_spec.get("cum_chunk_lengths", None)
        assert cum_chunk_lengths is not None, "cum_chunk_lengths must be provided"

        mem_objs = []
        for idx, key in enumerate(keys):
            shape = self.full_size_shape.copy()
            shape[self.fmt.token_dim()] = (
                cum_chunk_lengths[idx + 1] - cum_chunk_lengths[idx]
            )
            mem_obj = self.local_cpu_backend.allocate(
                torch.Size(shape), self.dtype, self.fmt
            )
            mem_objs.append(mem_obj)

        local_indexes = self.transfer_channel.get_local_mem_indices(mem_objs)

        # NOTE(Jiayi): Tier 3 lookup is batched with retrieval.
        msg = BatchedLookupAndGetMsg(
            receiver_id=peer_init_url,
            keys=[str(key) for key in keys],
            mem_indexes=local_indexes,
        )

        ret_msg = await self.async_peer_socket.send(msgspec.msgpack.encode(msg))

        num_hit_chunks = ret_msg.num_hit_chunks

        hit_mem_objs = mem_objs[:num_hit_chunks]
        for missed_mem_obj in mem_objs[num_hit_chunks:]:
            missed_mem_obj.ref_count_down()

        return hit_mem_objs

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(
        self,
    ) -> None:
        """
        Close the P2P backend.
        """
        pass

    ############################################################
    # Not-supported functions
    ############################################################

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
        transfer_spec: Any = None,
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
