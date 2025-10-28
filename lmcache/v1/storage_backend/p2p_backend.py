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
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_controller.message import (
    BatchedP2PLookupMsg,
    BatchedP2PLookupRetMsg,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import (
    P2PInitSideMsg,
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

    lookup_id: str

    receiver_id: str

    # CacheEngineKey in string form
    keys: list[str]

    # Indexes (remote) of allocated memory objects (to be written)
    mem_indexes: list[int]


class BatchedLookupAndGetRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of hit chunks
    num_hit_chunks: int


class BatchedLookupAndPutMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    sender_id: str

    # CacheEngineKey in string form
    keys: list[str]

    # Number of tokens for each chunk
    offsets: list[int]

    # Indexes (remote) of allocated memory objects (to be read)
    mem_indexes: list[int]


class BatchedLookupAndPutRetMsg(P2PMsgBase):
    """Lookup and retrieve message"""

    # Number of read chunks
    num_read_chunks: int


P2PMsg = Union[
    BatchedLookupAndGetMsg,
    BatchedLookupAndGetRetMsg,
    BatchedLookupAndPutMsg,
    BatchedLookupAndPutRetMsg,
]


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
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        assert config.p2p_host is not None, "p2p_host must be specified"
        assert config.p2p_init_ports is not None, "p2p_init_ports must be specified"
        assert config.p2p_lookup_ports is not None, "p2p_lookup_ports must be specified"

        # tp rank is worker id for now
        self.tp_rank = metadata.worker_id

        self.peer_host = config.p2p_host
        self.peer_init_port = config.p2p_init_ports[self.tp_rank]
        self.peer_init_url = f"{self.peer_host}:{self.peer_init_port}"

        self.peer_lookup_port = config.p2p_lookup_ports[self.tp_rank]
        self.peer_lookup_url = f"{self.peer_host}:{self.peer_lookup_port}"

        self.lmcache_instance_id = config.lmcache_instance_id

        # A CacheEngineKey (in int form) -> a list of
        # (peer_init_url, peer_lookup_url, location)
        self.local_lookup_cache: dict[int, tuple[str, str, str]] = {}
        # A set of peer_init_urls
        self.peer_id_to_lookup_url_mapping: dict[str, str] = {}

        # A lookup_id -> (peer_init_url, peer_lookup_url, location)
        self.lookup_id_to_peer_mapping: dict[str, tuple[str, str, str]] = {}

        # TODO(Jiayi): support gpu and local storage p2p as well.
        self.local_cpu_backend = local_cpu_backend
        self.memory_allocator = local_cpu_backend.get_memory_allocator()
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.dtype = metadata.kv_dtype
        self.full_size_shape = list(self.memory_allocator.cpu_allocator.shape)
        # TODO(Jiayi): remove this hardcode
        self.fmt: MemoryFormat = MemoryFormat.KV_2LTD
        self.chunk_size = config.chunk_size

        self.transfer_channel = CreateTransferChannel(
            channel_type=config.transfer_channel,
            async_mode=True,
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

        self.running = True
        self.lookup_url_to_socket_mapping: dict[str, zmq.Socket] = {}
        self.lookup_url_to_lock_mapping: dict[str, asyncio.Lock] = {}
        asyncio.run_coroutine_threadsafe(self._handle_peer_requests(), loop)

    def __str__(self) -> str:
        return "P2PBackend"

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
            instance_id=self.lmcache_instance_id,
            worker_id=self.tp_rank,
            hashes=hashes,
        )
        ret_msg = await self.lmcache_worker.async_put_and_wait_msg(msg)
        assert isinstance(ret_msg, BatchedP2PLookupRetMsg)

        # NOTE(Jiayi): For now we only support one peer hit.
        layout_info = ret_msg.layout_info[0]
        _, location, num_hit_chunks, peer_init_url = layout_info

        logger.info(f"Got layout info from controller: {layout_info}")

        if num_hit_chunks > 0:
            await self._ensure_peer_connection(peer_init_url)
            self.lookup_id_to_peer_mapping[lookup_id] = (
                peer_init_url,
                self.peer_id_to_lookup_url_mapping[peer_init_url],
                location,
            )

        # TODO(Jiayi): We could potentially update the local cache here.
        # Or we can update after tier 3 lookup.

        # NOTE(Jiayi): Tier 3 lookup is batched together with get
        # in function `batched_get_non_blocking`.

        return num_hit_chunks

    async def _handle_peer_requests(self):
        """
        Handle `BatchedLookupAndGetMsg` issued by peers in `batched_get_non_blocking`.
        """

        logger.info(
            f"Starting P2P backend batched get handler at {self.peer_lookup_url}"
        )
        self.async_context = get_zmq_context()
        self.async_peer_socket = get_zmq_socket(
            self.async_context,
            self.peer_lookup_url,
            "tcp",
            zmq.REP,
            "bind",
        )

        while self.running:
            msg_bytes = await self.async_peer_socket.recv()
            msg = msgspec.msgpack.decode(msg_bytes, type=P2PMsg)

            num_tokens = len(msg.mem_indexes) * self.chunk_size
            monitor_req_id = self.stats_monitor.on_p2p_transfer_request(num_tokens)

            if isinstance(msg, BatchedLookupAndGetMsg):
                logger.info("Received P2P batched get msg")

                lookup_id = msg.lookup_id
                receiver_id = msg.receiver_id
                remote_mem_indexes = msg.mem_indexes
                keys = [CacheEngineKey.from_string(key) for key in msg.keys]

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
                    "receiver_id": receiver_id,
                    "remote_indexes": remote_mem_indexes[:num_hit_chunks],
                }
                await self.transfer_channel.async_batched_write(
                    objects=mem_objs,
                    transfer_spec=channel_transfer_spec,
                )

                ret_msg = BatchedLookupAndGetRetMsg(
                    num_hit_chunks=num_hit_chunks,
                )

                for mem_obj in mem_objs:
                    mem_obj.ref_count_down()
                    mem_obj.unpin()
            elif isinstance(msg, BatchedLookupAndPutMsg):
                logger.info("Received P2P batched put msg")

                sender_id = msg.sender_id
                r_mem_indexes = msg.mem_indexes
                keys = [CacheEngineKey.from_string(key) for key in msg.keys]
                offsets = msg.offsets

                # TODO(Jiayi): Need to support more backend
                r_mem_indexes_to_read = []
                keys_to_read = []
                local_mem_objs = []
                for idx, key in enumerate(keys):
                    if self.local_cpu_backend.contains(key, pin=False):
                        continue
                    r_mem_indexes_to_read.append(r_mem_indexes[idx])
                    shape = self.full_size_shape.copy()
                    shape[self.fmt.token_dim()] = offsets[idx]
                    local_mem_obj = self.local_cpu_backend.allocate(
                        torch.Size(shape), self.dtype, self.fmt
                    )
                    local_mem_objs.append(local_mem_obj)
                    keys_to_read.append(key)

                channel_transfer_spec = {
                    "sender_id": sender_id,
                    "remote_indexes": r_mem_indexes_to_read,
                }
                await self.transfer_channel.async_batched_read(
                    buffers=local_mem_objs,
                    transfer_spec=channel_transfer_spec,
                )

                self.local_cpu_backend.batched_submit_put_task(
                    keys=keys_to_read,
                    memory_objs=local_mem_objs,
                )

                ret_msg = BatchedLookupAndPutRetMsg(
                    num_read_chunks=len(local_mem_objs),
                )

            logger.info(f"P2P transfer finished for request {monitor_req_id}")
            self.stats_monitor.on_p2p_transfer_finished(monitor_req_id)

            await self.async_peer_socket.send(msgspec.msgpack.encode(ret_msg))

    async def _ensure_peer_connection(
        self,
        peer_init_url: str,
    ) -> None:
        if peer_init_url in self.peer_id_to_lookup_url_mapping:
            return
        init_side_msg = P2PInitSideMsg()
        init_ret_msg = await self.transfer_channel.async_lazy_init_peer_connection(
            local_id=self.peer_init_url,
            peer_id=peer_init_url,
            peer_init_url=peer_init_url,
            init_side_msg=init_side_msg,
        )
        assert isinstance(init_ret_msg, P2PInitSideRetMsg)
        peer_lookup_url = init_ret_msg.peer_lookup_url
        self.peer_id_to_lookup_url_mapping[peer_init_url] = peer_lookup_url

        lookup_socket = get_zmq_socket(
            self.async_context,
            peer_lookup_url,
            "tcp",
            zmq.REQ,
            "connect",
        )
        self.lookup_url_to_socket_mapping[peer_lookup_url] = lookup_socket
        self.lookup_url_to_lock_mapping[peer_lookup_url] = asyncio.Lock()
        logger.info(
            f"Established connection to peer_init_url {peer_init_url}."
            f" The peer_lookup_url: {peer_lookup_url}"
        )

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        peer_init_url, peer_lookup_url, location = self.lookup_id_to_peer_mapping.pop(
            lookup_id
        )
        lookup_socket = self.lookup_url_to_socket_mapping[peer_lookup_url]
        lookup_lock = self.lookup_url_to_lock_mapping[peer_lookup_url]

        assert isinstance(transfer_spec, dict)
        cum_chunk_lengths = transfer_spec.get("cum_chunk_lengths", None)
        assert cum_chunk_lengths is not None, "cum_chunk_lengths must be provided"

        mem_objs = []
        str_keys = []
        for idx, key in enumerate(keys):
            shape = self.full_size_shape.copy()
            shape[self.fmt.token_dim()] = (
                cum_chunk_lengths[idx + 1] - cum_chunk_lengths[idx]
            )
            mem_obj = self.local_cpu_backend.allocate(
                torch.Size(shape), self.dtype, self.fmt
            )
            mem_objs.append(mem_obj)
            str_keys.append(key.to_string())

        local_indexes = self.transfer_channel.get_local_mem_indices(mem_objs)

        # NOTE(Jiayi): Tier 3 lookup is batched with retrieval.
        msg = BatchedLookupAndGetMsg(
            lookup_id=lookup_id,
            receiver_id=self.peer_init_url,
            keys=str_keys,
            mem_indexes=local_indexes,
        )

        async with lookup_lock:
            await lookup_socket.send(msgspec.msgpack.encode(msg))
            ret_msg_bytes = await lookup_socket.recv()

        ret_msg = msgspec.msgpack.decode(ret_msg_bytes, type=P2PMsg)

        num_hit_chunks = ret_msg.num_hit_chunks

        hit_mem_objs = mem_objs[:num_hit_chunks]
        for missed_mem_obj in mem_objs[num_hit_chunks:]:
            missed_mem_obj.ref_count_down()
        return hit_mem_objs

    # NOTE: put-related functions are not supported for now.
    async def async_batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        # Code path for `move` operation in controller.
        assert isinstance(transfer_spec, dict)
        assert "peer_init_url" in transfer_spec
        assert "offsets" in transfer_spec

        peer_init_url = transfer_spec["peer_init_url"]
        offsets = transfer_spec["offsets"]

        await self._ensure_peer_connection(transfer_spec["peer_init_url"])

        str_keys = [key.to_string() for key in keys]
        local_indexes = self.transfer_channel.get_local_mem_indices(objs)

        msg = BatchedLookupAndPutMsg(
            sender_id=self.peer_init_url,
            keys=str_keys,
            offsets=offsets,
            mem_indexes=local_indexes,
        )

        peer_lookup_url = self.peer_id_to_lookup_url_mapping[peer_init_url]
        lookup_socket = self.lookup_url_to_socket_mapping[peer_lookup_url]
        lookup_lock = self.lookup_url_to_lock_mapping[peer_lookup_url]

        async with lookup_lock:
            await lookup_socket.send(msgspec.msgpack.encode(msg))
            ret_msg_bytes = await lookup_socket.recv()
        ret_msg = msgspec.msgpack.decode(ret_msg_bytes, type=P2PMsg)

        return ret_msg.num_read_chunks

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
        return False

    # NOTE: put-related functions are not supported for now.
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        raise NotImplementedError

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        pass

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
