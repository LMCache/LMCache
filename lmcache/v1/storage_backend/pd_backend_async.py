# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union
import asyncio
import os
import threading
import time
import uuid

# Third Party
import msgspec
import torch
import zmq
import zmq.asyncio

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    TORCH_DTYPE_TO_STR_DTYPE,
    CacheEngineKey,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    PagedCpuGpuMemoryAllocator,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc_utils import get_zmq_context, get_zmq_socket
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.transfer_channel import CreateTransferChannel
from lmcache.v1.transfer_channel.transfer_utils import get_correct_device

logger = init_logger(__name__)


class PDMsgBase(msgspec.Struct, tag=True):
    """Base class for all PD-related messages"""

    pass


class AllocRequest(PDMsgBase):
    """Allocation request message"""

    keys: list[str]  # len(keys) indicates num_chunks
    fmt: int
    shape: list[int]  # The shape of the memory objects
    dtype: str
    last_chunk_toks: int
    # req_id is used by the receiver for per-request chunk accounting and
    # fail-fast detection when C_req > max_inflight_chunks.  An empty string
    # means the sender does not provide an identifier (backwards-compatible);
    # in that case per-request chunk accounting and fail-fast detection are
    # skipped for this allocation request.
    req_id: str = ""
    # is_last_batch signals the final batch for this req_id so the receiver
    # can release admission and clean up per-request tracking.
    is_last_batch: bool = False


class AllocResponse(PDMsgBase):
    """Allocation response message"""

    # Indexes (remote) of allocated memory objects (to be written).
    # One entry per key in the request; -1 means allocation failed for that slot.
    remote_indexes: list[int]

    # Indexes (local) of already sent memory objects.
    # Always empty for PDBackendAsync (no dedup), but included for
    # wire-compatibility with sync PDBackend senders that expect this field.
    already_sent_indexes: list[int] = []


class ProxyNotif(PDMsgBase):
    req_id: str  # The request id to notify the proxy


PDMsg = Union[AllocRequest, AllocResponse, ProxyNotif]


@dataclass
class _TransferItem:
    """Item placed onto a per-request transfer queue."""

    keys: Sequence[CacheEngineKey]
    memory_objs: List[MemoryObj]
    receiver_id: str
    on_complete_callback: Optional[Callable[[CacheEngineKey], None]]
    transfer_spec: Any


@dataclass
class PDConfig:
    role: str

    peer_host: str
    peer_init_port: int
    peer_alloc_port: int

    proxy_host: str
    proxy_port: int

    buffer_size: int
    buffer_device: str

    allocation_timeout_sec: float
    shutdown_timeout_sec: float
    condition_poll_interval_sec: float

    @staticmethod
    def from_cache_engine_config(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        tp_rank: int,
    ) -> "PDConfig":
        """Convert the LMCacheEngineConfig to PDConfig"""

        role = config.pd_role

        # TODO(Jiayi): Could be both if we want to do dynamic role switch.
        assert role in ["sender", "receiver"], (
            f"Invalid role: {config.pd_role}, must be either sender or receiver"
        )

        assert config.pd_buffer_size is not None
        assert config.pd_buffer_device is not None

        if role == "receiver":
            assert config.pd_peer_host is not None
            assert config.pd_peer_init_port is not None
            assert config.pd_peer_alloc_port is not None
        elif role == "sender":
            assert config.pd_proxy_host is not None
            assert config.pd_proxy_port is not None

        corrected_device = get_correct_device(
            config.pd_buffer_device, metadata.worker_id
        )

        if config.pd_peer_alloc_port is not None:
            pd_peer_alloc_port = config.pd_peer_alloc_port[tp_rank]
        else:
            pd_peer_alloc_port = None

        if config.pd_peer_init_port is not None:
            pd_peer_init_port = config.pd_peer_init_port[tp_rank]
        else:
            pd_peer_init_port = None

        return PDConfig(
            role=role,
            peer_host=config.pd_peer_host,
            peer_init_port=pd_peer_init_port,
            peer_alloc_port=pd_peer_alloc_port,
            proxy_host=config.pd_proxy_host,
            proxy_port=config.pd_proxy_port,
            buffer_size=config.pd_buffer_size,
            buffer_device=corrected_device,
            allocation_timeout_sec=config.pd_allocation_timeout_sec,
            shutdown_timeout_sec=config.pd_shutdown_timeout_sec,
            condition_poll_interval_sec=config.pd_condition_poll_interval_sec,
        )


class PDBackendAsync(AllocatorBackendInterface):
    """
    Implementation of the StorageBackendInterface for PD Disaggregation.

    At the sender side, it will never save anything but directly write the data
    to the receiver side.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ):
        self.running = True

        self.tp_rank = metadata.worker_id

        self.pd_config = PDConfig.from_cache_engine_config(
            config, metadata, self.tp_rank
        )

        # Cache timing config values as instance attributes for convenient access.
        self._allocation_timeout = self.pd_config.allocation_timeout_sec
        self._condition_poll_interval = self.pd_config.condition_poll_interval_sec

        self.corrected_device = get_correct_device(
            config.pd_buffer_device,
            metadata.worker_id,
        )

        # NOTE(Jiayi): sender/prefiller will not use this pool;
        # only receiver/decoder will.
        self.data: dict[CacheEngineKey, MemoryObj] = {}
        self.data_lock = threading.Lock()

        self.memory_allocator = self.initialize_allocator(config, metadata)
        assert isinstance(self.memory_allocator, PagedCpuGpuMemoryAllocator)

        self.zmq_context = get_zmq_context(use_asyncio=False)
        self.running_threads: list[threading.Thread] = []
        self.side_channels: list[zmq.Socket] = []

        # Initialize transfer channel
        peer_init_url = None
        self.local_id = ""
        # TODO(Jiayi): both sender and receiver have to have
        # peer_init_url if they want to do instance flip.
        if self.pd_config.peer_init_port is not None:
            peer_init_url = (
                f"{self.pd_config.peer_host}:{self.pd_config.peer_init_port}"
            )
            self.local_id = self.pd_config.peer_host + str(
                self.pd_config.peer_init_port
            )

        # Fallback: ensure local_id is never empty so DEALER identity is unique.
        # Senders typically don't set pd_peer_init_port. In xP1D deployments
        # multiple Prefillers may share the same proxy, so proxy_host:proxy_port
        # alone is NOT unique. We include os.getpid() and a UUID fragment to
        # guarantee a globally unique identity.
        if not self.local_id:
            self.local_id = f"sender-pid{os.getpid()}-{uuid.uuid4().hex[:8]}"

        # Create the event loop before the transfer channel so it can be passed
        # into the channel constructor for async_mode initialization.
        if self.pd_config.role == "sender":
            self._sender_loop = asyncio.new_event_loop()
            self._sender_thread = threading.Thread(
                target=self._sender_loop.run_forever,
                daemon=True,
                name="pd-sender-async",
            )
            self._sender_thread.start()
            event_loop = self._sender_loop
        elif self.pd_config.role == "receiver":
            self._recv_loop = asyncio.new_event_loop()
            self._recv_thread = threading.Thread(
                target=self._recv_loop.run_forever,
                daemon=True,
                name="pd-receiver-async",
            )
            self._recv_thread.start()
            event_loop = self._recv_loop
        else:
            raise ValueError("Invalid PD role.")

        allocator = (
            self.memory_allocator.cpu_allocator
            if self.corrected_device == "cpu"
            else self.memory_allocator.gpu_allocator
        )
        self.transfer_channel = CreateTransferChannel(
            async_mode=True,
            channel_type=config.transfer_channel,
            role=self.pd_config.role,
            buffer_ptr=allocator.buffer_ptr,
            buffer_size=allocator.buffer_size,
            align_bytes=allocator.align_bytes,
            tp_rank=self.tp_rank,
            peer_init_url=peer_init_url,
            backends=config.nixl_backends,
            device=self.corrected_device,
            event_loop=event_loop,
        )

        if self.pd_config.role == "sender":
            self._init_sender()
            self.initialized_peers: set[str] = set()
            self._peer_connection_lock = threading.Lock()
            # Separate async ZMQ context for sender coroutines
            self._async_zmq_context = zmq.asyncio.Context()
            self._async_alloc_sockets: dict[str, zmq.asyncio.Socket] = {}
            self._async_alloc_locks: dict[str, asyncio.Lock] = {}
            # Sender staging buffer flow control: block cache_engine.store()
            # (which runs in a vLLM worker thread) when the staging buffer is
            # near-full so that in-flight RDMA transfers can drain before new
            # allocations are allowed.  threading.Condition is required because
            # allocate() is called from a worker thread, not the asyncio loop.
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            self._sender_staging_lock = threading.Lock()
            self._sender_staging_condition = threading.Condition(
                self._sender_staging_lock
            )
            self._sender_inflight_chunks = 0
            self._sender_max_inflight_chunks = total_chunks
            logger.info(
                "PDBackendAsync sender: staging flow control initialized with "
                "max_inflight=%d (total_chunks=%d)",
                self._sender_max_inflight_chunks,
                total_chunks,
            )
            # Two-level queue structure:
            #
            #   _receiver_req_queues:  {
            #       "recv-1": Queue([ req-A, req-C ]),   # requests for Decoder-1
            #       "recv-2": Queue([ req-B ]),           # requests for Decoder-2
            #   }
            #                          │
            #                          ▼
            #   _transfer_queues:  {
            #       "req-A": Queue([ chunk1, chunk2, chunk3 ]),
            #       "req-B": Queue([ chunk1, chunk2 ]),
            #       "req-C": Queue([ chunk1 ]),
            #   }
            #
            # A per-receiver worker picks req-A from _receiver_req_queues[recv-1],
            # drains all its chunks from _transfer_queues["req-A"], then moves on
            # to req-C.  A separate worker does the same for recv-2.  This lets
            # req-B→Decoder-2 proceed concurrently with req-A→Decoder-1 (1PxD),
            # while still guaranteeing that req-A and req-C to the same Decoder-1
            # are serialized (preventing receiver-buffer fragmentation/deadlock).
            #
            # Workers are created on demand in _enqueue_transfer when a new
            # receiver_id is first seen.
            #
            # Only accessed from coroutines on _sender_loop, no lock needed.
            self._transfer_queues: dict[str, asyncio.Queue] = {}
            self._receiver_req_queues: dict[str, asyncio.Queue] = {}
            self._receiver_worker_tasks: dict[str, asyncio.Task] = {}
        elif self.pd_config.role == "receiver":
            self._init_receiver()
            total_chunks = self._aligned_buffer_size // self._chunk_size_bytes
            self._max_inflight_chunks = total_chunks
            self._inflight_chunks = 0
            # The condition must be created on the receiver event loop
            future = asyncio.run_coroutine_threadsafe(
                self._create_inflight_condition(), self._recv_loop
            )
            future.result(timeout=5)
            logger.info(
                "PDBackendAsync receiver: inflight flow control initialized with "
                "max_inflight_chunks=%d (total_chunks=%d, buffer=%d bytes, "
                "chunk=%d bytes)",
                self._max_inflight_chunks,
                total_chunks,
                self._aligned_buffer_size,
                self._chunk_size_bytes,
            )
            # Per-request key tracking for fail-fast detection and rollback.
            # Maps req_id → list of key strings allocated across all batches.
            self._req_allocated_keys: dict[str, list[str]] = {}
            # Admission control: only one req_id's batches processed at a time.
            self._admission_owner: str = ""

        self.full_chunk_size_bytes = config.chunk_size

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> PagedCpuGpuMemoryAllocator:
        if self.corrected_device != "cpu":
            logger.info(f"Setting cuda device to {self.corrected_device} ")
            torch.cuda.set_device(self.corrected_device)

        paged_mem_allocator = PagedCpuGpuMemoryAllocator()

        init_func = (
            paged_mem_allocator.init_cpu_memory_allocator
            if self.corrected_device == "cpu"
            else paged_mem_allocator.init_gpu_memory_allocator
        )

        # Calculate the chunk size (align_bytes) and align buffer size
        shapes = [torch.Size(metadata.kv_shape)]
        dtypes = [metadata.kv_dtype]
        chunk_size_bytes = get_size_bytes(shapes, dtypes)
        origin_buffer_size = config.pd_buffer_size
        aligned_buffer_size = origin_buffer_size // chunk_size_bytes * chunk_size_bytes

        if aligned_buffer_size == 0 and origin_buffer_size > 0:
            raise ValueError(
                f"pd_buffer_size ({origin_buffer_size}) is smaller than a "
                f"single chunk ({chunk_size_bytes}), resulting in an aligned "
                f"buffer of size 0. Please increase pd_buffer_size to be at "
                f"least {chunk_size_bytes}."
            )

        if aligned_buffer_size != origin_buffer_size:
            logger.info(
                f"Auto align pd_buffer_size, origin: {origin_buffer_size}, "
                f"aligned: {aligned_buffer_size}, chunk size: {chunk_size_bytes}. "
                f"The remaining {origin_buffer_size - aligned_buffer_size} bytes "
                f"will not be allocated."
            )

        self._chunk_size_bytes = chunk_size_bytes
        self._aligned_buffer_size = aligned_buffer_size
        # Number of tokens per chunk (used for capacity checks).
        self._chunk_token_size = metadata.kv_shape[MemoryFormat.KV_2LTD.token_dim()]

        pd_max_prefill_len = config.pd_max_prefill_len
        if pd_max_prefill_len > 0:
            capacity_tokens = (
                aligned_buffer_size // chunk_size_bytes
            ) * self._chunk_token_size
            if capacity_tokens < pd_max_prefill_len:
                raise ValueError(
                    f"PD buffer too small for the configured pd_max_prefill_len "
                    f"(role={self.pd_config.role}): "
                    f"capacity_tokens={capacity_tokens} < "
                    f"pd_max_prefill_len={pd_max_prefill_len}. "
                    f"Inputs: aligned_buffer_size={aligned_buffer_size}, "
                    f"chunk_size={chunk_size_bytes}, "
                    f"chunk_token_size={self._chunk_token_size}. "
                    f"Increase pd_buffer_size so that the buffer holds at least "
                    f"pd_max_prefill_len={pd_max_prefill_len} tokens."
                )

        init_func(
            aligned_buffer_size,
            shapes,
            dtypes,
            MemoryFormat.KV_2LTD,  # TODO: remove this hardcode
        )

        return paged_mem_allocator

    def get_memory_allocator(self) -> PagedCpuGpuMemoryAllocator:
        """Return the underlying paged CPU/GPU memory allocator.

        :return: The memory allocator instance used by this backend.
        :rtype: PagedCpuGpuMemoryAllocator
        """
        return self.memory_allocator

    def get_allocator_backend(self) -> "PDBackendAsync":
        """Return the allocator backend instance (self).

        :return: This backend instance, which implements AllocatorBackendInterface.
        :rtype: PDBackendAsync
        """
        return self

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        """Allocate a single memory object from the PD buffer.

        For the sender role, this method enforces staging buffer flow control:
        it blocks until inflight chunks drop below the threshold, then attempts
        allocation with a configurable timeout. For the receiver role, allocation
        is delegated directly to the underlying memory allocator.

        Note: ``eviction`` and ``busy_loop`` parameters are accepted for interface
        compatibility but are not used in the PD backend.

        :param shapes: Shape(s) of the KV tensors to allocate.
        :param dtypes: Data type(s) of the KV tensors.
        :param fmt: Memory format, defaults to KV_2LTD.
        :param eviction: Unused; kept for interface compatibility.
        :param busy_loop: Unused; kept for interface compatibility.
        :return: The allocated MemoryObj, or None if allocation failed or the
            backend is shutting down.
        :rtype: Optional[MemoryObj]
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD
        # NOTE: no eviction and busy_loop in PD
        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"

        if self.pd_config.role == "sender":
            # Single unified loop: flow-control check and allocation attempt are
            # combined so a thread can never pass flow control only to have
            # another thread steal the last memory slot before it allocates.
            #
            # The loop body, executed under _sender_staging_condition, does:
            #   1. Check running flag — exit immediately on shutdown.
            #   2. Check flow-control threshold — only attempt allocation when
            #      inflight_chunks < max; otherwise fall through to wait.
            #   3. Attempt allocation — on success, increment the counter and
            #      return; the check + allocate + increment form an atomic unit
            #      protected by the condition lock.
            #   4. Wait — either the threshold is exceeded or allocation failed
            #      (fragmentation / pool exhausted).  In both cases we wait for
            #      _release_sender_staging_chunks to call notify_all().
            #
            # Flow-control waits are not counted against the allocation deadline:
            # they loop back to the top and re-check the threshold, so a long
            # backpressure pause does not consume the 5-second allocation budget.
            # Once the threshold is satisfied we start a fresh deadline for the
            # actual allocation attempts.
            with self._sender_staging_condition:
                # deadline is initialised here and reset whenever backpressure
                # is active so that flow-control waits do not consume the
                # allocation budget.
                deadline: float = time.monotonic() + self._allocation_timeout
                last_near_full_log = 0.0
                while True:
                    if not self.running:
                        return None

                    at_threshold = (
                        self._sender_inflight_chunks >= self._sender_max_inflight_chunks
                    )

                    if at_threshold:
                        # Log near-full warning at most once per second to avoid
                        # spamming.
                        now = time.monotonic()
                        if now - last_near_full_log >= 1.0:
                            logger.warning(
                                "Sender staging buffer near-full: "
                                "inflight_chunks=%d >= max=%d, waiting for "
                                "transfers to complete...",
                                self._sender_inflight_chunks,
                                self._sender_max_inflight_chunks,
                            )
                            last_near_full_log = now
                        # Reset the allocation deadline so that once
                        # backpressure clears the thread gets a fresh
                        # 5-second window for actual allocation attempts.
                        deadline = time.monotonic() + self._allocation_timeout
                        self._sender_staging_condition.wait(
                            timeout=self._condition_poll_interval
                        )
                        continue

                    # Under threshold: attempt allocation.  deadline is always
                    # a valid float at this point (set above or reset in the
                    # at_threshold branch on a previous iteration).
                    mem_obj = self.memory_allocator.allocate(
                        shapes, dtypes, fmt=fmt, allocator_type=alloc_type
                    )
                    if mem_obj is not None:
                        self._sender_inflight_chunks += 1
                        return mem_obj

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._sender_staging_condition.wait(
                        timeout=min(remaining, self._condition_poll_interval)
                    )

            logger.error("Sender staging allocation failed after timeout")
            return None
        else:
            return self.memory_allocator.allocate(
                shapes, dtypes, fmt=fmt, allocator_type=alloc_type
            )

    # allocation overhead.
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ):
        """Allocate a batch of memory objects from the PD buffer.

        Delegates directly to the underlying memory allocator without sender
        flow control. Currently a thin wrapper; see the TODO for planned
        improvements.

        :param shapes: Shape(s) of the KV tensors to allocate.
        :param dtypes: Data type(s) of the KV tensors.
        :param batch_size: Number of memory objects to allocate.
        :param fmt: Memory format, defaults to KV_2LTD.
        :param eviction: Unused; kept for interface compatibility.
        :param busy_loop: Unused; kept for interface compatibility.
        :return: A list of allocated MemoryObj instances, or None for slots
            that failed to allocate.
        :rtype: list[Optional[MemoryObj]]
        """
        if fmt is None:
            fmt = MemoryFormat.KV_2LTD

        if self.pd_config.role == "sender":
            return [
                self.allocate(shapes, dtypes, fmt, eviction, busy_loop)
                for _ in range(batch_size)
            ]

        alloc_type = "cpu" if self.corrected_device == "cpu" else "gpu"
        return self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt, allocator_type=alloc_type
        )

    # NOTE(Jiayi): If two requests have overlapped keys, will
    # the later one cause any problems here?
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check whether the given key exists in the local data store.

        :param key: The cache engine key to look up.
        :param pin: If True and the key exists, increment the memory object's
            reference count to prevent it from being freed.
        :return: True if the key is present, False otherwise.
        :rtype: bool
        :raises AssertionError: If ``key`` is not a CacheEngineKey instance.
        """
        assert isinstance(key, CacheEngineKey)
        with self.data_lock:
            if mem_obj := self.data.get(key, None):
                if pin:
                    mem_obj.ref_count_up()
                return True
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check whether the key is pending in any in-flight put task.

        PDBackendAsync does not maintain a put task queue, so this always
        returns False.

        :param key: The cache engine key to check.
        :return: Always False.
        :rtype: bool
        """
        return False

    ############################################################
    # Prefiller functions
    ############################################################
    def _init_sender(self):
        proxy_url = f"{self.pd_config.proxy_host}:{self.pd_config.proxy_port}"
        self.proxy_side_channel = get_zmq_socket(
            self.zmq_context,
            proxy_url,
            "tcp",
            zmq.PUSH,
            "connect",
        )
        self._proxy_send_lock = asyncio.Lock()

    def _ensure_peer_connection(
        self,
        receiver_id: str,
        receiver_host: str,
        receiver_init_port: int,
        receiver_alloc_port: int,
    ) -> None:
        # Fast path: no lock required if already connected.
        if receiver_id in self.initialized_peers:
            return
        with self._peer_connection_lock:
            # Double-check under the lock to prevent duplicate connections when
            # multiple vLLM worker threads call this concurrently.
            if receiver_id in self.initialized_peers:
                return

            receiver_init_url = f"{receiver_host}:{receiver_init_port}"
            receiver_mem_alloc_url = f"{receiver_host}:{receiver_alloc_port}"

            # Establish the connection with the receiver/decoder.
            # The transfer channel uses an async ZMQ context (async_mode=True), so
            # we must call the async version scheduled on the sender event loop.
            future = asyncio.run_coroutine_threadsafe(
                self.transfer_channel.async_lazy_init_peer_connection(
                    local_id=self.local_id,
                    peer_id=receiver_id,
                    peer_init_url=receiver_init_url,
                ),
                self._sender_loop,
            )
            future.result()  # Block until connection is established

            # Schedule socket creation on the sender event loop to avoid
            # cross-thread issues
            future = asyncio.run_coroutine_threadsafe(
                self._async_create_alloc_socket(receiver_id, receiver_mem_alloc_url),
                self._sender_loop,
            )
            future.result(timeout=10)  # Wait for socket to be created

            self.initialized_peers.add(receiver_id)

    async def _async_create_alloc_socket(
        self, receiver_id: str, receiver_mem_alloc_url: str
    ):
        async_alloc_socket = self._async_zmq_context.socket(zmq.DEALER)
        # Use a sender-unique identity so multiple Senders connecting to the
        # same Receiver ROUTER have distinct identities (avoids undefined ZMQ
        # behavior when two DEALER sockets share the same identity string).
        sender_identity = f"{self.local_id}-to-{receiver_id}".encode()
        async_alloc_socket.setsockopt(zmq.IDENTITY, sender_identity)
        async_alloc_socket.connect(f"tcp://{receiver_mem_alloc_url}")
        self._async_alloc_sockets[receiver_id] = async_alloc_socket

    async def _async_remote_allocate(
        self, receiver_id: str, alloc_request: AllocRequest
    ) -> AllocResponse:
        if receiver_id not in self._async_alloc_locks:
            self._async_alloc_locks[receiver_id] = asyncio.Lock()
        async with self._async_alloc_locks[receiver_id]:
            socket = self._async_alloc_sockets[receiver_id]
            await socket.send_multipart([b"", msgspec.msgpack.encode(alloc_request)])
            frames = await socket.recv_multipart()
            msg = frames[-1]
        alloc_response = msgspec.msgpack.decode(msg, type=PDMsg)
        return alloc_response

    def _get_remote_alloc_request(
        self,
        keys: Sequence[CacheEngineKey],
        mem_objs: List[MemoryObj],
        req_id: str = "",
        is_last_batch: bool = False,
    ) -> AllocRequest:
        """
        Get the allocation request given the keys and memory objects.

        Let's say there are N memory objects in total.
        We have the following assumptions:
        - The first N-1 memory objects are full chunks, each with
        `full_chunk_size_bytes` tokens.
        - The last memory object can be a partial chunk, which has
        `last_chunk_toks` tokens.
        """

        fmt = mem_objs[0].meta.fmt
        shape = mem_objs[0].meta.shape
        dtype = TORCH_DTYPE_TO_STR_DTYPE[mem_objs[0].meta.dtype]
        token_dim = fmt.token_dim()
        last_chunk_toks = mem_objs[-1].meta.shape[token_dim]

        str_keys = [key.to_string() for key in keys]

        return AllocRequest(
            keys=str_keys,
            fmt=fmt.value,
            shape=list(shape),
            dtype=dtype,
            last_chunk_toks=last_chunk_toks,
            req_id=req_id,
            is_last_batch=is_last_batch,
        )

    async def _async_transfer_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        receiver_id: str,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        transfer_spec: Any = None,
    ) -> None:
        """
        Async coroutine that performs the full KV transfer:
        remote alloc → async_batched_write → ref_count_down → callback.
        Runs in the dedicated sender event loop (_sender_loop).

        ``remote_indexes`` has one entry per key (1:1 with memory_objs).
        A value of ``-1`` means allocation failed on the receiver; in that
        case the entire request is aborted — all local objects are released
        and no RDMA write or ProxyNotif is sent.
        """
        completed_indexes: set[int] = set()
        num_chunks = len(memory_objs)

        # Extract req_id for per-request allocation accounting on the receiver.
        # Using getattr with a default of "" keeps this backwards-compatible with
        # any transfer_spec that pre-dates the req_id field.  An empty string
        # causes the receiver to skip per-request chunk counting (no fail-fast
        # detection), which is acceptable for legacy callers.
        req_id: str = (
            getattr(transfer_spec, "req_id", "") if transfer_spec is not None else ""
        )
        is_last_batch: bool = (
            getattr(transfer_spec, "is_last_prefill", False)
            if transfer_spec is not None
            else False
        )

        try:
            alloc_request = self._get_remote_alloc_request(
                keys, memory_objs, req_id=req_id, is_last_batch=is_last_batch
            )
            alloc_response = await self._async_remote_allocate(
                receiver_id, alloc_request
            )
            remote_indexes = alloc_response.remote_indexes

            # Abort the whole request if any slot failed to allocate.
            # The decoder requires all chunks before it can start consuming;
            # a partial transfer wastes receiver buffer and can never complete.
            for idx, (mem_obj, remote_addr) in enumerate(
                zip(memory_objs, remote_indexes, strict=True)
            ):
                if remote_addr == -1:
                    logger.warning(
                        "Receiver allocation failed for key %s (idx=%d), "
                        "aborting entire request.",
                        keys[idx],
                        idx,
                    )
                    for j, mo in enumerate(memory_objs):
                        if j not in completed_indexes:
                            mo.ref_count_down()
                            completed_indexes.add(j)
                    return

            if memory_objs:
                channel_transfer_spec = {
                    "receiver_id": receiver_id,
                    "remote_indexes": remote_indexes,
                }
                await self.transfer_channel.async_batched_write(
                    objects=memory_objs,
                    transfer_spec=channel_transfer_spec,
                )
                for idx, mem_obj in enumerate(memory_objs):
                    if idx not in completed_indexes:
                        mem_obj.ref_count_down()
                        completed_indexes.add(idx)

            # Send ProxyNotif if this is the last prefill chunk, BEFORE invoking
            # on_complete_callback.  The worker processes chunks sequentially,
            # so all prior chunks have already completed by the time we reach
            # this point.  Sending here (inside _async_transfer_task) guarantees
            # the notification is observable the moment the callback fires.
            is_last_prefill = transfer_spec is not None and getattr(
                transfer_spec, "is_last_prefill", False
            )
            if is_last_prefill:
                try:
                    notif_msg = ProxyNotif(req_id=transfer_spec.req_id)
                    notif_msg_bytes = msgspec.msgpack.encode(notif_msg)
                    async with self._proxy_send_lock:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None, self.proxy_side_channel.send, notif_msg_bytes
                        )
                except Exception as e:
                    logger.error(
                        "Failed to send ProxyNotif for req %s: %s",
                        transfer_spec.req_id,
                        e,
                    )

            if on_complete_callback is not None:
                for key in keys:
                    try:
                        on_complete_callback(key)
                    except Exception as e:
                        logger.warning(
                            f"on_complete_callback failed for key {key}: {e}"
                        )
        except BaseException as e:
            if not isinstance(e, asyncio.CancelledError):
                logger.error("Async transfer task failed: %s", str(e))
            # Release ref counts on error to avoid leaks (only those not yet released)
            for idx, mem_obj in enumerate(memory_objs):
                if idx not in completed_indexes:
                    try:
                        mem_obj.ref_count_down()
                    except Exception:
                        pass
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            # Release sender staging buffer slots so that allocate() waiters
            # can proceed.  num_chunks equals the number of memory objects that
            # were allocated from the staging buffer via allocate().
            self._release_sender_staging_chunks(num_chunks)

    def _release_sender_staging_chunks(self, count: int) -> None:
        """Decrement sender staging inflight counter and notify waiters.

        Called from ``_async_transfer_task`` (asyncio) after all staging
        buffers for a transfer have been freed via ``ref_count_down()``.
        ``threading.Condition.notify_all()`` is non-blocking and safe to call
        from an asyncio coroutine.

        :param count: Number of staging slots to release.
        """
        if self.pd_config.role == "sender" and count > 0:
            with self._sender_staging_condition:
                self._sender_inflight_chunks = max(
                    0, self._sender_inflight_chunks - count
                )
                self._sender_staging_condition.notify_all()

    async def _enqueue_transfer(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        receiver_id: str,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]],
        transfer_spec: Any,
    ) -> None:
        """Enqueue a transfer item onto the per-request asyncio.Queue.

        If no queue exists for the request yet, one is created and the req_id
        is placed onto the per-receiver FIFO queue so the receiver's worker
        will process it.  Within a single receiver, requests are serialized in
        FIFO order (one request fully occupies the decoder buffer at a time,
        preventing fragmentation/deadlock).  Requests to *different* receivers
        are handled by independent workers and proceed concurrently (1PxD).

        Workers are created on demand: the first time a new receiver_id is
        seen a worker task is started; if an existing worker has already
        finished (e.g. after draining all queued requests) a new one is
        started transparently.

        Must be called as a coroutine on ``_sender_loop`` (via
        ``asyncio.run_coroutine_threadsafe``).  All dict accesses are therefore
        single-threaded on the event loop.

        :param keys: Cache keys for this transfer batch.
        :param memory_objs: Memory objects to transfer.
        :param receiver_id: Identifier of the remote receiver.
        :param on_complete_callback: Optional per-key completion callback.
        :param transfer_spec: Transfer specification (carries ``req_id`` and
            ``is_last_prefill``).
        """
        req_id: str = (
            getattr(transfer_spec, "req_id", "unknown")
            if transfer_spec is not None
            else "unknown"
        )
        if req_id not in self._transfer_queues:
            q: asyncio.Queue[_TransferItem] = asyncio.Queue()
            self._transfer_queues[req_id] = q
            # Ensure a per-receiver queue exists and register this req_id.
            if receiver_id not in self._receiver_req_queues:
                self._receiver_req_queues[receiver_id] = asyncio.Queue()
            await self._receiver_req_queues[receiver_id].put(req_id)
            # Start (or restart) the per-receiver worker if needed.
            existing = self._receiver_worker_tasks.get(receiver_id)
            if existing is None or existing.done():
                self._receiver_worker_tasks[receiver_id] = asyncio.create_task(
                    self._receiver_req_worker(receiver_id)
                )
        item = _TransferItem(
            keys=keys,
            memory_objs=memory_objs,
            receiver_id=receiver_id,
            on_complete_callback=on_complete_callback,
            transfer_spec=transfer_spec,
        )
        self._transfer_queues[req_id].put_nowait(item)

    async def _drain_transfer_queue(
        self, req_id: str, q: "asyncio.Queue[_TransferItem]"
    ) -> None:
        """Drain remaining items from a per-request transfer queue after failure.

        Releases ref counts on all memory objects to prevent memory leaks and
        releases sender staging chunks to unblock allocate() waiters.

        :param req_id: The request identifier (used for logging).
        :param q: The asyncio.Queue to drain.
        """
        drained = 0
        while not q.empty():
            try:
                item = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            for mem_obj in item.memory_objs:
                try:
                    mem_obj.ref_count_down()
                except Exception as e:
                    logger.warning(
                        "ref_count_down() failed during drain for req %s: %s",
                        req_id,
                        e,
                    )
            self._release_sender_staging_chunks(len(item.memory_objs))
            q.task_done()
            drained += 1
        if drained > 0:
            logger.warning(
                "Drained %d remaining transfer items for req %s after failure.",
                drained,
                req_id,
            )

    async def _receiver_req_worker(self, receiver_id: str) -> None:
        """Per-receiver FIFO worker: process one request at a time for a receiver.

        Pulls req_ids from ``_receiver_req_queues[receiver_id]`` in arrival
        order and drains each request's per-request transfer queue serially
        until the item flagged with ``is_last_prefill=True`` is processed.
        Only then is the next req_id dequeued.

        This prevents receiver-buffer fragmentation: at most one request's
        chunks occupy a given decoder's buffer at any time, guaranteeing that
        the decoder can always find a complete request to process.  Having
        independent workers per receiver means req-A→Decoder-1 and
        req-B→Decoder-2 proceed concurrently (1PxD), while req-A and req-C
        to the same Decoder-1 remain serialized.

        If a transfer fails mid-request, remaining items are drained (ref
        counts released) and the worker moves on to the next request.

        On ANY exit (including CancelledError), the outer finally block drains
        all remaining transfer queues for this receiver to prevent memory leaks.

        :param receiver_id: Identifier of the remote receiver this worker
            services.
        """
        req_queue = self._receiver_req_queues[receiver_id]
        # Collect req_ids belonging to this receiver for cleanup in finally.
        receiver_req_ids: set[str] = set()
        try:
            while True:
                try:
                    req_id: str = await req_queue.get()
                except asyncio.CancelledError:
                    return

                receiver_req_ids.add(req_id)
                q = self._transfer_queues.get(req_id)
                if q is None:
                    # Queue was removed before we could process it — skip.
                    continue

                try:
                    while True:
                        try:
                            item: _TransferItem = await asyncio.wait_for(
                                q.get(),
                                timeout=self._allocation_timeout * 2,
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "Timed out waiting for next chunk of req %s "
                                "(no chunk arrived within %.1fs). "
                                "Sender may have crashed. Draining and moving on.",
                                req_id,
                                self._allocation_timeout * 2,
                            )
                            await self._drain_transfer_queue(req_id, q)
                            break
                        except asyncio.CancelledError:
                            return

                        transfer_failed = False
                        try:
                            await self._async_transfer_task(
                                keys=item.keys,
                                memory_objs=item.memory_objs,
                                receiver_id=item.receiver_id,
                                on_complete_callback=item.on_complete_callback,
                                transfer_spec=item.transfer_spec,
                            )
                        except Exception as e:
                            logger.error(
                                "Transfer worker error for req %s: %s. "
                                "Aborting remaining transfers for this request.",
                                req_id,
                                e,
                            )
                            transfer_failed = True
                        finally:
                            q.task_done()

                        is_last = item.transfer_spec is not None and getattr(
                            item.transfer_spec, "is_last_prefill", False
                        )
                        if transfer_failed:
                            # Drain remaining items and release their memory.
                            await self._drain_transfer_queue(req_id, q)
                            break
                        if is_last:
                            break
                finally:
                    # Drain any remaining items to prevent memory leaks on
                    # unexpected exit (e.g. CancelledError between the try and
                    # the is_last check).
                    await self._drain_transfer_queue(req_id, q)
                    self._transfer_queues.pop(req_id, None)
                    receiver_req_ids.discard(req_id)
        finally:
            # Drain any req_ids still in the per-receiver queue that we never
            # processed (e.g. enqueued during shutdown after our last get()).
            while True:
                try:
                    remaining_id = req_queue.get_nowait()
                    receiver_req_ids.add(remaining_id)
                except asyncio.QueueEmpty:
                    break
            # On ANY exit (including CancelledError), drain all remaining
            # transfer queues for this receiver to prevent memory leaks.
            for remaining_req_id in receiver_req_ids:
                remaining_q = self._transfer_queues.pop(remaining_req_id, None)
                if remaining_q is not None:
                    await self._drain_transfer_queue(remaining_req_id, remaining_q)

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Submit batched put tasks to transfer KV caches to peer.

        :param on_complete_callback: Optional callback invoked once per key
            after the transfer completes. Callback exceptions are caught and logged.
        """
        for mem_obj in memory_objs:
            mem_obj.ref_count_up()

        try:
            receiver_init_port = transfer_spec.receiver_init_port[self.tp_rank]
            receiver_alloc_port = transfer_spec.receiver_alloc_port[self.tp_rank]
            receiver_id = transfer_spec.receiver_host + str(receiver_init_port)
            receiver_host = transfer_spec.receiver_host

            self._ensure_peer_connection(
                receiver_id=receiver_id,
                receiver_host=receiver_host,
                receiver_init_port=receiver_init_port,
                receiver_alloc_port=receiver_alloc_port,
            )

            # Schedule via _enqueue_transfer so the item is placed on the
            # per-request asyncio.Queue.  The dedicated consumer coroutine
            # processes items sequentially, guaranteeing in-order RDMA writes and
            # sending ProxyNotif only after all chunks are complete.
            asyncio.run_coroutine_threadsafe(
                self._enqueue_transfer(
                    keys=list(keys),
                    memory_objs=list(memory_objs),
                    receiver_id=receiver_id,
                    on_complete_callback=on_complete_callback,
                    transfer_spec=transfer_spec,
                ),
                self._sender_loop,
            )
        except Exception as e:
            # Roll back ref counts to prevent memory leak
            for mem_obj in memory_objs:
                try:
                    mem_obj.ref_count_down()
                except Exception:
                    pass
            logger.error(
                "batched_submit_put_task failed, ref counts rolled back: %s", e
            )
            raise

    ############################################################
    # Prefiller functions end
    ############################################################

    ############################################################
    # Decoder functions
    ############################################################
    async def _create_inflight_condition(self) -> None:
        """Create asyncio.Conditions for inflight flow control and admission control.

        Must be called from within the receiver event loop so that the
        Conditions are bound to the correct loop.
        """
        self._inflight_condition = asyncio.Condition()
        self._admission_condition = asyncio.Condition()
        self._router_send_lock = asyncio.Lock()
        self._pending_alloc_tasks: set[asyncio.Task] = set()

    def _init_receiver(self):
        """
        Launch the async memory allocation server coroutine on the already-running
        receiver event loop (self._recv_loop, created before the transfer channel).
        """
        asyncio.run_coroutine_threadsafe(
            self._async_mem_alloc_server(), self._recv_loop
        )

    async def _async_mem_alloc_server(self):
        """
        Async ZMQ ROUTER server for memory allocation requests.
        Replaces the blocking _mem_alloc_loop / _mem_alloc_thread.
        Uses a ROUTER socket instead of REP so that multiple concurrent
        senders (xP1D topology) can each have their requests received and
        dispatched independently — admission control inside
        ``_handle_alloc_request`` only blocks the per-request coroutine, not
        the receive loop.
        """
        # Third Party
        import zmq.asyncio as azmq

        async_ctx = azmq.Context()
        socket = async_ctx.socket(zmq.ROUTER)
        alloc_port = self.pd_config.peer_alloc_port
        socket.bind(f"tcp://*:{alloc_port}")
        logger.info(f"Async mem alloc server listening on port {alloc_port}")
        try:
            while self.running:
                try:
                    frames = await socket.recv_multipart()
                    # ROUTER frames: [identity, empty_delimiter, payload]
                    identity = frames[0]
                    payload = frames[-1]
                    task = asyncio.create_task(
                        self._handle_alloc_request(socket, identity, payload)
                    )
                    self._pending_alloc_tasks.add(task)
                    task.add_done_callback(self._pending_alloc_tasks.discard)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Failed to process async mem alloc: %s", str(e))
                    if self.running:
                        await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        finally:
            socket.close()
            async_ctx.term()

    async def _handle_alloc_request(
        self,
        socket: zmq.asyncio.Socket,
        identity: bytes,
        payload: bytes,
    ) -> None:
        """Handle a single allocation request from a sender.

        Runs as an independent coroutine so that multiple requests from
        different senders can be processed concurrently. Admission control
        inside ``_async_allocate_and_put`` ensures only one req_id allocates
        memory at a time, but blocking on admission no longer prevents the
        ROUTER socket from receiving other messages.

        On any exception (including allocation failures), an error response
        with all ``-1`` remote_indexes is sent back so the sender is never
        left waiting indefinitely on ``recv_multipart``.

        :param socket: The ROUTER socket to send the response on.
        :param identity: The sender identity frame returned by ROUTER.
        :param payload: The raw msgpack-encoded AllocRequest bytes.
        """
        n_keys = 0
        try:
            alloc_req = msgspec.msgpack.decode(payload, type=PDMsg)
            assert isinstance(alloc_req, AllocRequest), (
                "The request from the remote peer is not an AllocRequest"
            )
            n_keys = len(alloc_req.keys)
            # NOTE: it's okay to put the memory objs into the storage backend
            # first because decode vllm will not be able to see the decode
            # request until proxy receives the ack.
            alloc_resp = await self._async_allocate_and_put(alloc_req)
            resp_bytes = msgspec.msgpack.encode(alloc_resp)
            async with self._router_send_lock:
                await socket.send_multipart([identity, b"", resp_bytes])
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Failed to process alloc request from %s: %s",
                identity,
                str(e),
            )
            # Send an error response so the sender is not left hanging
            # forever on recv_multipart().
            try:
                error_resp = AllocResponse(remote_indexes=[-1] * max(n_keys, 1))
                async with self._router_send_lock:
                    await socket.send_multipart(
                        [identity, b"", msgspec.msgpack.encode(error_resp)]
                    )
            except Exception:
                logger.warning("Failed to send error response to %s", identity)

    async def _async_allocate_and_put(
        self, alloc_request: AllocRequest
    ) -> AllocResponse:
        """
        Async version of _allocate_and_put.
        Uses event-driven waiting on ``_inflight_condition`` instead of
        ``asyncio.sleep`` polling so the event loop stays responsive while
        waiting for free memory.
        pin=False: PDBackendAsync has no eviction; pinning is unnecessary and
        causes ref_count leaks.

        ``remote_indexes`` has exactly one entry per key in the request.

        Error semantics:
        - Fail-fast (cumulative chunks > max_inflight_chunks): raises
          ``RuntimeError`` immediately; no rollback is performed.
        - Allocator timeout (allocate() returns None after
          ``self._allocation_timeout`` seconds): raises ``RuntimeError``;
          already-allocated chunks in the current batch are rolled back via
          ``remove()`` to prevent memory and inflight counter leaks.
        - Backpressure wait (inflight >= max_inflight_chunks): indefinite
          while ``self.running``.  The sender will not submit new chunks
          until the decoder frees some, so this wait is expected to resolve.
        """
        total_allocs = len(alloc_request.keys)
        req_id = alloc_request.req_id

        # Admission control: ensure only one req_id's batches are processed at
        # a time, preventing multi-sender interleaving deadlock.
        # Skip entirely for legacy senders that provide no req_id.
        #
        # Timeout guard: if the current admission owner's Sender crashes
        # before sending is_last_batch=True, _admission_owner would never
        # be cleared.  After ``_allocation_timeout`` seconds we forcibly
        # evict the stale owner so subsequent requests can proceed.
        if req_id:
            async with self._admission_condition:
                deadline = asyncio.get_running_loop().time() + self._allocation_timeout
                while self._admission_owner and self._admission_owner != req_id:
                    if not self.running:
                        raise RuntimeError(
                            f"Receiver shutting down while req {req_id} "
                            f"waits for admission (owner={self._admission_owner})"
                        )
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        stale_owner = self._admission_owner
                        logger.warning(
                            "Admission wait timed out for req %s after "
                            "%.1fs (stale owner=%s). Forcibly evicting "
                            "stale owner — its Sender likely crashed.",
                            req_id,
                            self._allocation_timeout,
                            stale_owner,
                        )
                        # Clean up stale owner's per-request tracking to
                        # prevent the fail-fast check from accumulating
                        # ghost chunk counts.
                        self._req_allocated_keys.pop(stale_owner, None)
                        # Fall through to claim admission below.
                        break
                    try:
                        await asyncio.wait_for(
                            self._admission_condition.wait(),
                            timeout=min(remaining, self._condition_poll_interval),
                        )
                    except asyncio.TimeoutError:
                        pass
                self._admission_owner = req_id

        async def _release_admission() -> None:
            """Clear the admission owner and notify all waiting coroutines.

            No-op when req_id is empty (legacy senders without a req_id).
            """
            if req_id:
                async with self._admission_condition:
                    self._admission_owner = ""
                    self._admission_condition.notify_all()

        # Fail-fast: detect if this request can never complete because it
        # requires more chunks than the decoder buffer can ever hold at once.
        # The decoder needs all chunks present before it can start consuming,
        # so C_req > max_inflight_chunks is an impossible configuration.
        if req_id:
            prev_count = len(self._req_allocated_keys.get(req_id, []))
            new_total = prev_count + total_allocs
            if new_total > self._max_inflight_chunks:
                await _release_admission()
                raise RuntimeError(
                    f"Request {req_id} requires {new_total} total chunks "
                    f"(already allocated {prev_count}, new batch {total_allocs}) "
                    f"but max_inflight_chunks={self._max_inflight_chunks}. "
                    f"This request can never complete because the decoder requires "
                    f"all chunks before it can start consuming. "
                    f"To fix: increase pd_buffer_size so that "
                    f"max_inflight_chunks >= total chunks needed for the largest "
                    f"request, or reduce prompt length / chunk size."
                )
        else:
            # No req_id provided (legacy sender or untracked call); per-request
            # fail-fast detection is unavailable.  Log at debug level so operators
            # can diagnose potential buffer exhaustion if needed.
            logger.debug(
                "AllocRequest has no req_id — per-request chunk accounting "
                "is disabled for this batch (C_req > T fail-fast will not fire)"
            )

        fmt = MemoryFormat(alloc_request.fmt)
        dtype = STR_DTYPE_TO_TORCH_DTYPE[alloc_request.dtype]
        shape = list(alloc_request.shape)  # copy — we mutate token_dim

        alloc_indexes: list[int] = []
        current_batch_keys: list[str] = []

        try:
            # Global inflight pre-check: wait until there are enough free slots
            # for the entire batch before entering the per-chunk loop.
            # This prevents the mid-allocation deadlock that occurs when residual
            # inflight chunks from a previous request fill the buffer: without this
            # check, a request can acquire admission, allocate some chunks, then
            # block mid-loop on _inflight_condition.  Since the request hasn't
            # finished sending, no ProxyNotif is issued, the decoder never starts
            # consuming, inflight never decreases, and the system deadlocks.
            async with self._inflight_condition:
                while (
                    self._max_inflight_chunks - self._inflight_chunks
                ) < total_allocs:
                    if not self.running:
                        await _release_admission()
                        remaining_allocs = total_allocs - len(alloc_indexes)
                        alloc_indexes.extend([-1] * remaining_allocs)
                        return AllocResponse(remote_indexes=alloc_indexes)
                    logger.info(
                        "Req %s admitted but waiting to allocate %d chunks "
                        "(inflight=%d, max=%d, free=%d)",
                        req_id,
                        total_allocs,
                        self._inflight_chunks,
                        self._max_inflight_chunks,
                        self._max_inflight_chunks - self._inflight_chunks,
                    )
                    await self._inflight_condition.wait()

            for idx, key_str in enumerate(alloc_request.keys):
                key = CacheEngineKey.from_string(key_str)

                if idx == total_allocs - 1:
                    token_dim = fmt.token_dim()
                    shape[token_dim] = alloc_request.last_chunk_toks

                # Wait until inflight count is below threshold before allocating.
                async with self._inflight_condition:
                    while self._inflight_chunks >= self._max_inflight_chunks:
                        if not self.running:
                            # Release admission and return failure on shutdown.
                            await _release_admission()
                            remaining_allocs = total_allocs - len(alloc_indexes)
                            alloc_indexes.extend([-1] * remaining_allocs)
                            return AllocResponse(remote_indexes=alloc_indexes)
                        logger.warning(
                            "Decoder buffer near-full: inflight_chunks=%d >= max=%d, "
                            "waiting for buffers to be freed...",
                            self._inflight_chunks,
                            self._max_inflight_chunks,
                        )
                        await self._inflight_condition.wait()
                    self._inflight_chunks += 1

                mem_obj = self.allocate(torch.Size(shape), dtype, fmt)
                # Event-driven retry: wait on _inflight_condition for notification
                # instead of asyncio.sleep polling so the coroutine wakes up
                # immediately when _notify_inflight_freed fires.
                deadline = asyncio.get_running_loop().time() + self._allocation_timeout
                while mem_obj is None:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        async with self._inflight_condition:
                            self._inflight_chunks -= 1
                            self._inflight_condition.notify_all()
                        raise RuntimeError(
                            f"Failed to allocate memory for key {key} after "
                            f"timeout (~{self._allocation_timeout:.0f}s). "
                            f"req_id={req_id}, key_index={idx}/{total_allocs}, "
                            f"inflight_chunks={self._inflight_chunks}, "
                            f"max_inflight_chunks={self._max_inflight_chunks}."
                        )
                    async with self._inflight_condition:
                        try:
                            await asyncio.wait_for(
                                self._inflight_condition.wait(),
                                timeout=min(remaining, self._condition_poll_interval),
                            )
                        except asyncio.TimeoutError:
                            pass
                    mem_obj = self.allocate(torch.Size(shape), dtype, fmt)

                alloc_indexes.append(mem_obj.meta.address)
                self.put(key, mem_obj)
                current_batch_keys.append(key_str)
        except BaseException:
            # Rollback: remove already-allocated chunks from this batch
            # to prevent memory and inflight counter leaks.
            for rollback_key_str in current_batch_keys:
                try:
                    rollback_key = CacheEngineKey.from_string(rollback_key_str)
                    self.remove(rollback_key)
                except Exception as re:
                    logger.warning(
                        "Rollback remove failed for key %s: %s",
                        rollback_key_str,
                        re,
                    )
            # Also clean up per-request tracking for this batch.
            if req_id:
                self._req_allocated_keys.pop(req_id, None)
            await _release_admission()
            raise

        # All allocations in this batch succeeded.
        if req_id:
            if req_id not in self._req_allocated_keys:
                self._req_allocated_keys[req_id] = []
            self._req_allocated_keys[req_id].extend(current_batch_keys)
            if alloc_request.is_last_batch:
                self._req_allocated_keys.pop(req_id, None)
                await _release_admission()

        return AllocResponse(remote_indexes=alloc_indexes)

    def put(
        self,
        key: CacheEngineKey,
        mem_obj: MemoryObj,
    ) -> None:
        """Store a memory object in the local data dictionary.

        If a memory object already exists for the given key, the old object is
        released (ref_count_down) and the inflight counter is decremented on
        the receiver side to prevent memory leaks.

        :param key: The cache engine key to associate with the memory object.
        :param mem_obj: The memory object to store.
        """
        with self.data_lock:
            old = self.data.pop(key, None)
            if old is not None:
                logger.warning(
                    "Overwriting existing MemoryObj for key %s in "
                    "PDBackendAsync.put(). "
                    "Releasing old object to prevent memory leak.",
                    key,
                )
                old.ref_count_down()
                if self.pd_config.role == "receiver":
                    asyncio.run_coroutine_threadsafe(
                        self._notify_inflight_freed(), self._recv_loop
                    )
            self.data[key] = mem_obj

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Retrieve the memory object for the given key (blocking).

        Since PDBackendAsync uses push-based transfer, the key is expected to
        already be present in local data.

        :param key: The cache engine key to retrieve.
        :return: The corresponding MemoryObj.
        :rtype: MemoryObj
        :raises AssertionError: If the key is not found in local data.
        """
        with self.data_lock:
            # NOTE(Jiayi): we assume that the key must be in local data
            # because we are using a push-based transfer
            mem_obj = self.data.get(key, None)
            assert mem_obj is not None, f"Key {key} not found in local data."
            return mem_obj

    def remove(
        self,
        key: CacheEngineKey,
        force: bool = True,
    ) -> bool:
        """
        Remove the key from the storage backend.

        :param key: The key to remove.
        """
        with self.data_lock:
            mem_obj = self.data.pop(key, None)
            if mem_obj is not None:
                mem_obj.ref_count_down()
                if self.pd_config.role == "receiver":
                    asyncio.run_coroutine_threadsafe(
                        self._notify_inflight_freed(), self._recv_loop
                    )
                return True
            return False

    async def _notify_inflight_freed(self) -> None:
        """Decrement the inflight chunk counter and notify waiting allocations.

        Scheduled on the receiver event loop from ``remove()`` (which runs in
        a vLLM worker thread) so that asyncio.Condition operations are always
        called from within the correct event loop.
        """
        async with self._inflight_condition:
            if self._inflight_chunks == 0:
                logger.warning(
                    "inflight_chunks is already 0 before decrement; "
                    "this indicates a counter synchronization bug."
                )
            else:
                self._inflight_chunks -= 1
            self._inflight_condition.notify_all()

    ############################################################
    # Decoder functions end
    ############################################################

    @staticmethod
    def _shutdown_loop(
        loop: asyncio.AbstractEventLoop,
        thread: threading.Thread,
        timeout: float = 5.0,
    ) -> None:
        """Cancel all pending tasks on *loop*, stop it, and join the thread.

        Uses a ``threading.Event`` to synchronize shutdown completion so that
        ``thread.join`` is only called after the loop has actually stopped,
        preventing thread or resource leaks when the loop takes time to drain.

        :param loop: The event loop to shut down.
        :param thread: The thread running the event loop.
        :param timeout: Maximum seconds to wait for shutdown and thread join.
        """
        shutdown_done = threading.Event()

        async def _cancel_and_stop() -> None:
            tasks = [
                t
                for t in asyncio.all_tasks(loop)
                if t is not asyncio.current_task() and not t.done()
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()
            shutdown_done.set()

        if loop.is_running():
            loop.call_soon_threadsafe(loop.create_task, _cancel_and_stop())
            shutdown_done.wait(timeout=timeout)
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning(
                "Event loop thread %s did not terminate within %.1fs timeout.",
                thread.name,
                timeout,
            )

    def close(self) -> None:
        """
        Close the storage backend.
        """
        self.running = False
        # Wake up any threads blocked on the sender staging condition so they
        # can observe running=False and exit cleanly.
        if hasattr(self, "_sender_staging_condition"):
            with self._sender_staging_condition:
                self._sender_staging_condition.notify_all()
        for thread in self.running_threads:
            thread.join()
        # Shut down sender async loop if present
        if hasattr(self, "_sender_loop"):
            # Cancel all per-receiver worker tasks before stopping the loop.
            # _shutdown_loop also cancels all tasks, but being explicit here
            # ensures each worker sees CancelledError promptly.
            if hasattr(self, "_receiver_worker_tasks"):
                for task in self._receiver_worker_tasks.values():
                    if task is not None:
                        task.cancel()
            self._shutdown_loop(
                self._sender_loop,
                self._sender_thread,
                timeout=self.pd_config.shutdown_timeout_sec,
            )
            # Close async alloc sockets
            for sock in self._async_alloc_sockets.values():
                try:
                    sock.close()
                except Exception:
                    pass
            try:
                self._async_zmq_context.term()
            except Exception:
                pass
        # Shut down receiver async loop if present
        if hasattr(self, "_recv_loop"):
            # Wait for any in-flight allocation tasks to finish gracefully
            # before tearing down the loop.  This lets _async_allocate_and_put
            # complete (and release admission) rather than being cancelled mid-
            # allocation, which would leak the admission lock on graceful restart.
            if hasattr(self, "_pending_alloc_tasks"):
                try:

                    async def _wait_pending() -> None:
                        """Await all pending alloc tasks with a timeout."""
                        pending = list(self._pending_alloc_tasks)
                        if pending:
                            await asyncio.wait(
                                pending,
                                timeout=self.pd_config.shutdown_timeout_sec,
                            )

                    future = asyncio.run_coroutine_threadsafe(
                        _wait_pending(), self._recv_loop
                    )
                    future.result(
                        # Add 1 second buffer beyond the inner asyncio.wait timeout
                        # so that future.result() does not expire before asyncio.wait
                        # has a chance to return naturally.
                        timeout=self.pd_config.shutdown_timeout_sec + 1
                    )
                except Exception:
                    logger.debug(
                        "Timed out waiting for pending alloc tasks during shutdown"
                    )
            # Wake up any coroutines blocked on _inflight_condition or
            # _admission_condition so they can observe running=False and exit
            # cleanly before the loop is stopped.
            if hasattr(self, "_inflight_condition"):
                try:

                    async def _wake_inflight() -> None:
                        async with self._inflight_condition:
                            self._inflight_condition.notify_all()

                    asyncio.run_coroutine_threadsafe(_wake_inflight(), self._recv_loop)
                except Exception as exc:
                    logger.debug(
                        "Could not schedule _inflight_condition wake-up "
                        "(loop may already be stopped): %s",
                        exc,
                    )
            if hasattr(self, "_admission_condition"):
                try:

                    async def _wake_admission() -> None:
                        async with self._admission_condition:
                            self._admission_condition.notify_all()

                    asyncio.run_coroutine_threadsafe(_wake_admission(), self._recv_loop)
                except Exception as exc:
                    logger.debug(
                        "Could not schedule _admission_condition wake-up "
                        "(loop may already be stopped): %s",
                        exc,
                    )
            self._shutdown_loop(
                self._recv_loop,
                self._recv_thread,
                timeout=self.pd_config.shutdown_timeout_sec,
            )
        self.transfer_channel.close()
        self.zmq_context.term()

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin the memory object for the given key to prevent eviction.

        PDBackendAsync has no eviction mechanism, so this is a no-op that
        always returns True.

        :param key: The cache engine key to pin.
        :return: Always True.
        :rtype: bool
        """
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin the memory object for the given key.

        PDBackendAsync has no eviction mechanism, so this is a no-op that
        always returns True.

        :param key: The cache engine key to unpin.
        :return: Always True.
        :rtype: bool
        """
        return True
