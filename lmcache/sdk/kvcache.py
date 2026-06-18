# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving and storing KV cache tensors via LMCache's MQ endpoints.
"""

# Standard
from __future__ import annotations
from multiprocessing import shared_memory
from collections.abc import Sequence
from multiprocessing.resource_tracker import unregister
import time
import uuid
import os
from typing import Any

# Third Party
import torch
import zmq

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor
from lmcache.logging import init_logger

logger = init_logger(__name__)

class KVCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache operation fails."""

 
class LMCacheSDKContext:
    """
    Retrieve and store KV cache tensors via LMCache's MQ endpoints.

    Communicates with LMCache through the ZMQ message queue.
    Data transfer is done through shared memory segments.

    The model layout must already be registered in the running LMCache server
    (e.g. by a vllm instance that called REGISTER_KV_CACHE).
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the SDK context and register the SDK transfer strategy.
        
        Args:
            url: ZMQ endpoint URL for the LMCache message queue.
            model_name: Model name used by the running LMCache server instance.
            timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
        
        Returns:
            LMCacheSDKContext instance.
        """
        self._zmq_context = zmq.Context.instance()
        self._mq_client = MessageQueueClient(url, self._zmq_context)
        self._mq_timeout = timeout
        self._model_name = model_name
        self.instance_id = os.getpid()

        self._world_size: int = self._mq_client.submit_request(
            RequestType.GET_WORLD_SIZE,
            [model_name],
            get_response_class(RequestType.GET_WORLD_SIZE),
        ).result(timeout=timeout)
        if self._world_size != 1:
            raise KVCacheSDKError(
                f"SDK currently supports world_size=1 only, got {self._world_size}"
            )

        self._chunk_size: int = self._mq_client.submit_request(
            RequestType.GET_CHUNK_SIZE,
            [],
            get_response_class(RequestType.GET_CHUNK_SIZE),
        ).result(timeout=timeout)

        shm_pool_info: dict[str, int | str] = self._mq_client.submit_request(
            RequestType.GET_SHM_POOL_INFO,
            [],
            get_response_class(RequestType.GET_SHM_POOL_INFO),
        ).result(timeout=timeout)
        self.shm_name = shm_pool_info.get("shm_name", "")
        self.shm_pool_size = shm_pool_info.get("pool_size", 0)

        self._mq_client.submit_request(
            RequestType.REGISTER_SDK_TRANSFER_STRATEGY,
            [self.instance_id, self._model_name, self._world_size],
            get_response_class(RequestType.REGISTER_SDK_TRANSFER_STRATEGY),
        ).result(timeout=timeout)
        
        self._pending_lookups: set[str] = set()
        self._finished_lookups: dict[str, int] = {}
    
    def close(self) -> None:
        """Close the MQ client and ZMQ context."""
        self._mq_client.close()
        self._zmq_context.term()
    
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
    ) -> None:
        """Submit a LOOKUP request for the given token IDs.
        
        Args:
            request_id: Unique ID for this lookup request.
            token_ids: List of token IDs to look up.
            cache_salt: Optional cache salt string for the lookup.
        """
        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()

        future = self._mq_client.submit_request(
            RequestType.LOOKUP,
            [key, self._world_size],
            get_response_class(RequestType.LOOKUP),
        )
        try:
            future.result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "LOOKUP request timed out after %ss.",
                self._mq_timeout,
            )
            return
        self._pending_lookups.add(request_id)

    def check_lookup_result(
        self, 
        request_id: str
    ) -> int | None:
        """Check the result of a LOOKUP request.

        Args:
            request_id: The request ID of the LOOKUP to check.
        
        Returns:
            The number of prefetched tokens if the LOOKUP is finished, 
            0 if not finished, or None if the request ID is not found.
        """
        if request_id not in self._pending_lookups:
            # No job — either unhealthy at submit time or already cleaned up.
            # If we have a cached result, return it to handle repeated calls.
            return self._finished_lookups.get(request_id, 0)

        if request_id in self._finished_lookups:
            # Return cached result if the job is already finished
            return self._finished_lookups[request_id]

        try:
            result = self._mq_client.submit_request(
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
                get_response_class(RequestType.QUERY_PREFETCH_STATUS),
            ).result(timeout=self._mq_timeout)
        except TimeoutError:
            logger.warning(
                "QUERY_PREFETCH_STATUS timed out after %ss.",
                self._mq_timeout,
            )
            return 0

        if result is None:
            return None

        token_count = result * self._chunk_size
        self._finished_lookups[request_id] = token_count
        return token_count
    
    def prepare_retrieve(
        self,
        key: IPCCacheEngineKey,
    ) -> list[dict[str, Any]] | None:
        """Called in phase 1: ask server to prepare KV in SHM or in pickle.
        KV data is already prefetched by the time this is called.
        Adapted from lmcache/v1/multiprocess/transfer_context/shm.py
        
        Args:
            key: The IPC cache engine key containing the lookup metadata.
        
        Returns:
            A list of SHM slot descriptors if using SHM transfer. 
            None if preparation fails or times out.
        """
        future = self._mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, self.instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self._mq_timeout)
        except TimeoutError:
            return None
        if not response.success:
            return None
        slots = response.context.get("slots", [])

        return slots if isinstance(slots, list) else None

    def commit_retrieve(
        self,
        key: IPCCacheEngineKey,
    ) -> bool:
        """Called in phase 3: tell server to release the SHM slots after retrieval is done.
        
        Args:
            key: The IPC cache engine key containing the lookup metadata.
        
        Returns:
            True if the commit is successful, False if it fails or times out.
        """
        future = self._mq_client.submit_request(
            RequestType.COMMIT_RETRIEVE,
            [key, self.instance_id],
            get_response_class(RequestType.COMMIT_RETRIEVE),
        )
        try:
            return future.result(timeout=self._mq_timeout)
        except TimeoutError:
            return False

    def prepare_store(
        self,
        key: IPCCacheEngineKey,
    ) -> list[dict[str, Any]] | None:
        """Called in phase 1: ask for slots or pickle path.
        Adapted from lmcache/v1/multiprocess/transfer_context/shm.py

        Args:
            key: The IPC cache engine key containing the store metadata.
        
        Returns:
            A list of SHM slot descriptors if using SHM transfer.
            None if preparation fails or times out.
        """

        future = self._mq_client.submit_request(
            RequestType.PREPARE_STORE,
            [key, self.instance_id],
            get_response_class(RequestType.PREPARE_STORE),
        )
        try:
            response = future.result(timeout=self._mq_timeout)
        except TimeoutError as err:
            raise TimeoutError(
                f"[PREPARE_STORE] timed out for instance_id={self.instance_id} "
                f"after {self._mq_timeout}s"
            ) from err

        context = response.context if isinstance(response.context, dict) else {}
        slots = context.get("slots")

        return slots if isinstance(slots, list) else None

    def commit_store(
        self,
        key: IPCCacheEngineKey,
        _chunks: list[torch.Tensor] | bytes,
    ) -> bool:
        """Called in phase 3: tell server to commit the data in SHM.
        Still retain _chunks as format for Pickle (todo).

        Args:
            key: The IPC cache engine key containing the store metadata.
            _chunks: The list of tensors to store, or bytes if using Pickle.
        
        Returns:
            True if the commit is successful, False if it fails or times out.
        """
        future = self._mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, self.instance_id, b""],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return future.result(timeout=self._mq_timeout)
        except TimeoutError as err:
            raise TimeoutError(
                f"[COMMIT_STORE] timed out for instance_id={self.instance_id} "
                f"after {self._mq_timeout}s"
            ) from err

    def end_session(self, request_id: str) -> None:
        """End a session and clean up associated resources on the server.
        
        Args:
            request_id: The request ID of the session to end.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookups.pop(request_id, None)
        self._mq_client.submit_request(
            RequestType.END_SESSION,
            [request_id],
            get_response_class(RequestType.END_SESSION),
        ).result(timeout=self._mq_timeout)

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        worker_id: int | None = None,
    ) -> IPCCacheEngineKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            worker_id: Optional worker ID for the key. 
                If None, the key will be created without a worker ID (for lookups).

        Returns:
            IPCCacheEngineKey: The constructed key.
        """
        return IPCCacheEngineKey(
            model_name=self._model_name,
            world_size=self._world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
    
def connect(
    url: str,
    model_name: str,
    timeout: float = 60.0,
) -> "LMCacheSDKContext":
    """Create and initialize the LMCache SDK context.
    
    Args:
        url: ZMQ endpoint URL for the LMCache message queue.
        model_name: Model name used by the running LMCache server instance.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
    
    Returns:
        An initialized LMCacheSDKContext instance.
        Ready to be passed to close(), retrieve(), and store() functions.
    """
    return LMCacheSDKContext(
        url=url, 
        model_name=model_name, 
        timeout=timeout
    )

def close(ctx: "LMCacheSDKContext") -> None:
    """Close the LMCache SDK context and release resources.
    
    Args:
        ctx: The LMCacheSDKContext instance to close.
    """
    ctx.close()

def retrieve(
    ctx: "LMCacheSDKContext",
    tokens: Sequence[int],
    cache_salt: str = "",
) -> torch.Tensor | None:
    """Retrieve KV cache tensors for the given token IDs.
    
    Args:
        ctx: The LMCacheSDKContext instance to use for retrieval.
        tokens: The list of token IDs to retrieve KV cache for.
        cache_salt: Optional cache salt string for the lookup.
    
    Returns:
        A contiguous CPU tensor containing the retrieved KV cache for 
        the requested tokens.
        None if retrieval fails or there are no tokens to retrieve.
    """
    if not tokens:
        return None

    # Drop tokens not fit into a whole chunk
    total_tokens = (len(tokens) // ctx._chunk_size) * ctx._chunk_size
    if total_tokens == 0:
        return None

    # Assign request ID to this request
    request_id = f"retrieve-{uuid.uuid4().hex}"
    key = ctx._create_key(
        token_ids=list(tokens[:total_tokens]),
        start=0,
        end=total_tokens,
        request_id=request_id,
        cache_salt=cache_salt,
        worker_id=0
    )

    # Phase 0: Trigger lookup
    ctx.maybe_submit_lookup_request(
        request_id,
        token_ids=list(tokens[:total_tokens]),
        cache_salt=cache_salt,
    )

    num_prefetched_tokens = ctx.check_lookup_result(request_id)
    while num_prefetched_tokens is None:
        logger.info(
            "Waiting for LOOKUP result for request_id=%s...",
            request_id,
        )
        time.sleep(0.01)
        num_prefetched_tokens = ctx.check_lookup_result(request_id)
    
    # Phase 1: Ask server to prepare KV in SHM or in pickle
    prep_result = ctx.prepare_retrieve(key)
    if not prep_result:
        raise KVCacheSDKError("PREPARE_RETRIEVE did not return SHM slot descriptors or pickle data")

    # Phase 2: SDK reads the SHM slots, copies into a contiguous tensor
    out_buffer = [ShmSlotDescriptor.from_dict(s) for s in prep_result]

    shm = shared_memory.SharedMemory(name=ctx.shm_name, create=False)
    unregister(f"/{shm.name}", "shared_memory")  # server owns the segment
    try:
        shards: list[torch.Tensor] = []
        for slot in out_buffer:
            dtype = getattr(torch, slot.dtype)
            view = torch.frombuffer(
                shm.buf,
                dtype=dtype,
                count=slot.length // torch.empty((), dtype=dtype).element_size(),
                offset=slot.offset,
            ).view(slot.shape)
            shards.append(view)
        assembled = _assemble_contiguous(shards, ctx._world_size, ctx._chunk_size)
    finally:
        shm.close()

    # Phase 3: tell server to release the SHM slots
    ctx.commit_retrieve(key)
    ctx.end_session(request_id)

    return assembled

def store(
    ctx: "LMCacheSDKContext",
    kv: torch.Tensor, 
    tokens: list[int], 
    cache_salt: str = ""
) -> bool:
    """Store KV cache tensors for the given token IDs.

    Args:
        ctx: The LMCacheSDKContext instance to use for storage.
        kv: The KV cache tensor to store, of shape [2, L, T, D].
        tokens: The list of token IDs corresponding to the KV cache tensor.
        cache_salt: Optional cache salt string for the store.
    
    Returns:
        True if the store operation is successful, False otherwise.
    """
    kv_cpu = kv.detach().cpu().contiguous()

    token_ids = list(tokens)
    _validate_store_tensor(kv_cpu, token_ids, ctx._model_name, ctx._chunk_size)

    # Phase 0: assign request ID to this request
    request_id = f"store-{uuid.uuid4().hex}"
    key = ctx._create_key(
        token_ids=token_ids,
        start=0,
        end=(len(tokens) // ctx._chunk_size) * ctx._chunk_size,
        request_id=request_id,
        cache_salt=cache_salt,
        worker_id=0
    )

    # Phase 1: server reserves SHM slots, returns descriptors
    prep_result = ctx.prepare_store(key)
    if not prep_result:
        raise KVCacheSDKError("PREPARE_STORE did not return SHM slot descriptors")

    # Phase 2: SDK copies data into SHM slots
    out_buffer = [ShmSlotDescriptor.from_dict(s) for s in prep_result]
    d_per_worker = kv_cpu.shape[3] // ctx._world_size

    shm = shared_memory.SharedMemory(name=ctx.shm_name, create=False)
    unregister(f"/{shm.name}", "shared_memory")
    try:
        for slot_idx, slot in enumerate(out_buffer):
            chunk_i = slot_idx // ctx._world_size
            worker_j = slot_idx % ctx._world_size
            t_start = chunk_i * ctx._chunk_size
            t_end = t_start + ctx._chunk_size
            d_start = worker_j * d_per_worker
            d_end = d_start + d_per_worker

            dtype = getattr(torch, slot.dtype)
            dst = torch.frombuffer(
                shm.buf,
                dtype=dtype,
                count=slot.length // torch.empty((), dtype=dtype).element_size(),
                offset=slot.offset,
            ).view(slot.shape)

            shard = kv_cpu[:, :, t_start:t_end, d_start:d_end].contiguous()
            dst.copy_(shard)
    finally:
        shm.close()

    # Phase 3: tell server to write the data from SHM to storage manager
    commit_result = ctx.commit_store(key, b"")
    ctx.end_session(request_id)
    
    return commit_result

# Helper functions

def _assemble_contiguous(
    shards: list[torch.Tensor],
    world_size: int,
    chunk_size: int,
) -> torch.Tensor:
    """Assemble per-shard tensors from the server into a single contiguous CPU tensor.

    Shards arrive ordered as:
    Each shard shape is [2, L, chunk_size, D // world_size].

    Args:
        shards: Flat shard list from the pickle payload/SHM read.
        world_size: TP shards per chunk.
        chunk_size: Tokens per LMCache chunk.

    Returns:
        Contiguous CPU tensor of shape [2, L, hit_tokens, D].

    Raises:
        KVCacheSDKError: If shards is empty or incomplete.
    """
    if not shards:
        raise KVCacheSDKError("no shards returned from COMMIT_RETRIEVE")

    hit_chunks = len(shards) // world_size
    if hit_chunks == 0:
        raise KVCacheSDKError(
            f"incomplete shard set: {len(shards)} shards for world_size={world_size}"
        )

    first = shards[0]
    num_layers = first.shape[1]
    d_per_worker = first.shape[3]
    hidden_dim = d_per_worker * world_size
    hit_tokens = hit_chunks * chunk_size

    result = torch.empty(
        (2, num_layers, hit_tokens, hidden_dim),
        dtype=first.dtype,
        device="cpu",
    )
    for shard_idx, shard in enumerate(shards[: hit_chunks * world_size]):
        chunk_idx = shard_idx // world_size
        worker_id = shard_idx % world_size
        t_start = chunk_idx * chunk_size
        d_start = worker_id * d_per_worker
        result[
            :, :, t_start : t_start + chunk_size, d_start : d_start + d_per_worker
        ].copy_(shard.cpu())

    return result.contiguous()

def _validate_store_tensor(
    kv: torch.Tensor,
    tokens: Sequence[int],
    model_name: str,
    chunk_size: int,
) -> None:
    """Validate tensor shape against token metadata and server chunk size.
    
    Args:
        kv: The KV cache tensor to validate.
        tokens: The list of token IDs corresponding to the KV cache tensor.
        model_name: The model name to check for emptiness.
        chunk_size: The chunk size configured in the LMCache server.
    
    Raises:
        KVCacheSDKError: If any validation check fails.
    """
    if not model_name:
        raise KVCacheSDKError("model_name must be provided")
    if not tokens:
        raise KVCacheSDKError("tokens must be provided")
    if kv.ndim != 4:
        raise KVCacheSDKError(
            f"kv tensor must be 4-D [2, L, T, D], got shape {tuple(kv.shape)}"
        )
    total_tokens = (len(tokens) // chunk_size) * chunk_size
    if total_tokens == 0:
        raise KVCacheSDKError("tokens must contain at least one complete chunk")
    if kv.shape[2] != total_tokens:
        raise KVCacheSDKError(
            f"kv tensor token dim {kv.shape[2]} does not match "
            f"complete token prefix {total_tokens}"
        )
