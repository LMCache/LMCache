# SPDX-License-Identifier: Apache-2.0
"""
SDK for retrieving and storing KV cache tensors.
"""

# Standard
from __future__ import annotations
from collections.abc import Sequence
import time
import uuid
import os
import collections
import threading

# Third Party
import requests
import torch
import zmq

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    send_lmcache_request,
)
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class
from lmcache.v1.multiprocess.transfer_context.contiguous import (
    ContiguousEngineDrivenTransfer,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    create_transfer_context,
)

logger = init_logger(__name__)

class KVCacheSDKError(RuntimeError):
    """Raised when an SDK KV-cache operation fails."""

class BlockPool:
    """Free-list of physical block indices into the SDK's paged KV buffer.
    Manages allocation and free for SDK's paged KV cache blocks.
    """

    def __init__(self, num_blocks: int) -> None:
        self._free: collections.deque[int] = collections.deque(range(num_blocks))
        self._lock = threading.Lock()
        self.num_blocks = num_blocks
        print(f"Initialized BlockPool with {num_blocks} blocks.", flush=True)

    def alloc(self, n: int) -> list[int]:
        print(f"Allocating {n} blocks from BlockPool with {len(self._free)} free blocks...")
        with self._lock:
            if n > len(self._free):
                raise KVCacheSDKError(
                    f"block pool exhausted: need {n}, have {len(self._free)} "
                    f"free of {self.num_blocks}"
                )
            return [self._free.popleft() for _ in range(n)]

    def free(self, ids: list[int]) -> None:
        print(f"Freeing #blocks: {len(ids)} back to BlockPool...", flush=True)
        with self._lock:
            self._free.extend(ids)
 
class LMCacheKVCacheContext:
    """
    Retrieve and store KV cache tensors via LMCache's MQ endpoints.

    The model layout must already be registered in the running LMCache server
    (e.g. by a vllm instance that called REGISTER_KV_CACHE).
    Getting the layout information from server requires inference engine running
    with GPU so that SDK can derive the shapes.

    SDK is running on CPU, so the allocated paged KV cache is on CPU, and the
    geometry is always in HND order regardless of the inference engine's.
    """

    def __init__(
        self,
        url: str,
        http_url: str,
        model_name: str,
        timeout: float = 60.0,
    ) -> None:
        """
        Initialize the SDK context and register the SDK transfer strategy.
        
        Args:
            url: ZMQ endpoint URL for the LMCache message queue.
            http_url: HTTP endpoint URL for fetching informations.
            model_name: Model name used by the running LMCache server instance.
            timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
        
        Returns:
            LMCacheKVCacheContext instance.
        """
        self._zmq_context = zmq.Context.instance()
        self._mq_client = MessageQueueClient(url, self._zmq_context)
        self._mq_timeout = timeout
        self._model_name = model_name
        self.instance_id = os.getpid()
        self._http_url = http_url

        mp_conf = {}
        try:
            response = requests.get(f"{self._http_url}/conf", timeout=timeout)
            response.raise_for_status()
            mp_conf = response.json()["mp"]
        except (requests.RequestException, KeyError, ValueError) as err:
            raise KVCacheSDKError(
                f"failed to fetch server config from {self._http_url}/conf"
            ) from err
        self._chunk_size: int = int(mp_conf["chunk_size"])
        self.shm_name: str = str(mp_conf.get("shm_name", "")).lstrip("/")
        if self.shm_name and not self.shm_name.startswith("lmcache_l1_pool_"):
            self.shm_name = f"lmcache_l1_pool_{self.shm_name}"
        print(
            f"Initialized LMCacheKVCacheContext with instance_id={self.instance_id}, "
            f"model_name={self._model_name}, chunk_size={self._chunk_size}, "
            f"shm_name={self.shm_name}",
            flush=True,
        )

        self._cache_context_meta_conf = {}
        try:
            response = requests.get(f"{self._http_url}/status", timeout=timeout)
            response.raise_for_status()
            self._cache_context_meta_conf = response.json().get("cache_context_meta", {})
        except (requests.RequestException, KeyError, ValueError) as err:
            raise KVCacheSDKError(
                f"failed to fetch server config from {self._http_url}/status"
            ) from err

        self._pending_lookups: set[str] = set()
        self._finished_lookups: dict[str, int] = {}

    def register_kv_caches(
        self,
        kv_buffer_bytes: int,
    ) -> None:
        """Register the KV cache layout for the model with the SDK context.
        
        Args:
            kv_buffer_bytes: The size of the KV cache buffer in bytes (user-supplied).
        """
        entry = next((e for e in self._cache_context_meta_conf.values()
                        if e.get("model_name") == self._model_name), None)
        if not entry:
            raise KVCacheSDKError(
                f"no registered GPU layout for model_name={self._model_name!r}; "
                "MP mode cannot derive geometry from model_name alone — "
                "register from a vLLM instance first, or pass the geometry explicitly."
            )
        
        self._world_size = int(entry.get("world_size", 1))
        kv_cache_layout = entry.get("kv_cache_layout", {})
        if not kv_cache_layout:
            raise KVCacheSDKError(
                f"no registered KV cache layout for {self._model_name!r}."
            )
        num_layers = kv_cache_layout.get("num_layers", 0)

        kernel_groups = kv_cache_layout.get("kernel_groups", [])
        if len(kernel_groups) != 1:
            raise KVCacheSDKError(
                f"Currently not supporting hybrid models with multiple kernel groups; "
                f"found {len(kernel_groups)} for model_name={self._model_name!r}."
            )
        kernel_group = kernel_groups[0]
        dtype = getattr(torch, kernel_group["dtype"].replace("torch.", ""))
        tokens_per_block = kernel_group.get("tokens_per_block", 0)

        inner = [
            int(x) for x in
            kernel_group["engine_kv_concrete_shape"].split("[")[1].rstrip("] ").split(",")
        ]

        # From get_num_heads (lmcache/v1/gpu_connector/utils.py)
        #     lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        #     # NHD: [..., BS, NH, HS] — num_heads at shape[3]
        #     lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        #     # HND: [..., NH, BS, HS] — num_heads at shape[2]
        head_dim = inner[-1]
        if "NH_BS" in kernel_group["engine_kv_format"]:
            num_kv_heads, block_size = inner[-3], inner[-2]
        else:
            block_size, num_kv_heads = inner[-3], inner[-2]
        if block_size != tokens_per_block:
            raise KVCacheSDKError(
                f"decoded block_size {block_size} != tokens_per_block "
                f"{tokens_per_block} for model_name={self._model_name!r}"
            )

        bytes_per_block_per_layer = (
            2 * num_kv_heads * block_size * head_dim * dtype.itemsize
        )
        num_blocks = kv_buffer_bytes // (num_layers * bytes_per_block_per_layer)
        if num_blocks < 1:
            raise KVCacheSDKError(
                f"budget {kv_buffer_bytes} too small for even one block "
                f"({num_layers * bytes_per_block_per_layer} bytes/block/layer)"
            )

        # SDK runs on CPU, LMCache's detects HND (gpu_connector/utils.py:663). 
        # Build in HND-physical order [NB, 2, NH, BS, HS] (flip from GPU order)
        # So, no matter the inference engine runs on CPU or GPU, SDK will always
        # use HND.
        self.block_pool = BlockPool(num_blocks=num_blocks)
        shape = (num_blocks, 2, num_kv_heads, block_size, head_dim)

        self._kv_caches = {
            f"layer.{i}": torch.zeros(shape, dtype=dtype, device="cpu")
            for i in range(num_layers)
        }

        transfer_ctx = create_transfer_context(self._kv_caches)
        self.blocks_in_chunk = self._chunk_size // block_size
        layout_hints = LayoutHints(
            kv_layout="HND",
            num_kv_heads=num_kv_heads,
            tokens_per_block=block_size,
            head_dim=head_dim,
        )

        transfer_ctx.register(
            self.instance_id, self._kv_caches, self._model_name, self._world_size,
            self.blocks_in_chunk, self._mq_client, self._mq_timeout,
            send_request=send_lmcache_request,
            layout_hints=layout_hints,
        )

        self._transfer_ctx = ContiguousEngineDrivenTransfer(
            transfer_ctx.engine_driven_context, self._chunk_size
        )
    
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
        Modification from lmcache/integration/vllm/vllm_multi_process_adapter.py.
        Need duplicate since SDK has TransferContext, not Adapter, but still need
        lookup and end_session functionality.
        
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
        Modification from lmcache/integration/vllm/vllm_multi_process_adapter.py.
        Need duplicate since SDK has TransferContext, not Adapter, but still need
        lookup and end_session functionality.

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
    
    def end_session(self, request_id: str, block_ids: list[int] | None = None) -> None:
        """End a session and clean up associated resources on the server.
        
        Args:
            request_id: The request ID of the session to end.
            block_ids: Optional list of block IDs to free.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookups.pop(request_id, None)
        self._mq_client.submit_request(
            RequestType.END_SESSION,
            [request_id],
            get_response_class(RequestType.END_SESSION),
        ).result(timeout=self._mq_timeout)
        
        if block_ids is not None:
            self.block_pool.free(block_ids)

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        worker_id: int | None = None,
    ) -> IPCCacheServerKey:
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
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey(
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
    http_url: str,
    model_name: str,
    kv_buffer_bytes: int,
    timeout: float = 60.0,
) -> "LMCacheKVCacheContext":
    """Create and initialize the LMCache SDK context.
    
    Args:
        url: ZMQ endpoint URL for the LMCache message queue.
        http_url: HTTP endpoint URL for fetching server configuration.
        model_name: Model name used by the running LMCache server instance.
        kv_buffer_bytes: The size of the KV cache buffer in bytes.
        timeout: Timeout in seconds for blocking MQ calls. Defaults to 60.
    
    Returns:
        An initialized LMCacheKVCacheContext instance.
        Ready to be passed to close(), retrieve(), and store() functions.
    """
    ctx = LMCacheKVCacheContext(
        url=url, 
        http_url=http_url,
        model_name=model_name, 
        timeout=timeout
    )
    ctx.register_kv_caches(
        kv_buffer_bytes=kv_buffer_bytes,
    )
    return ctx

def close(ctx: "LMCacheKVCacheContext") -> None:
    """Close the LMCache SDK context and release resources.
    
    Args:
        ctx: The LMCacheKVCacheContext instance to close.
    """
    ctx.close()

def retrieve(
    ctx: "LMCacheKVCacheContext",
    tokens: Sequence[int],
    cache_salt: str = "",
) -> torch.Tensor | None:
    """Retrieve KV cache tensors for the given token IDs.
    
    Args:
        ctx: The LMCacheKVCacheContext instance to use for retrieval.
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

    start_time = time.time()
    num_prefetched_tokens = ctx.check_lookup_result(request_id)
    while num_prefetched_tokens is None:
        if time.time() - start_time > ctx._mq_timeout:
            raise KVCacheSDKError(
                f"LOOKUP request timed out after {ctx._mq_timeout}s for request_id={request_id}"
            )
        logger.info(
            "Waiting for LOOKUP result for request_id=%s...",
            request_id,
        )
        time.sleep(0.01)
        num_prefetched_tokens = ctx.check_lookup_result(request_id)
    
    if num_prefetched_tokens <= 0:
        ctx.end_session(request_id)
        return None

    # Phase 1: retrieve the cached prefix as one contiguous tensor
    key = ctx._create_key(
        token_ids=list(tokens[:num_prefetched_tokens]),
        start=0,
        end=num_prefetched_tokens,
        request_id=request_id,
        cache_salt=cache_salt,
        worker_id=0,
    )
    try:
        return ctx._transfer_ctx.retrieve(key, ctx.instance_id)
    finally:
        ctx.end_session(request_id)

def store(
    ctx: "LMCacheKVCacheContext",
    kv: torch.Tensor, 
    tokens: Sequence[int], 
    cache_salt: str = ""
) -> bool:
    """Store KV cache tensors for the given token IDs.

    Args:
        ctx: The LMCacheKVCacheContext instance to use for storage.
        kv: The KV cache tensor to store, of shape [2, L, T, D].
        tokens: The list of token IDs corresponding to the KV cache tensor.
        cache_salt: Optional cache salt string for the store.
    
    Returns:
        True if the store operation is successful, False otherwise.
    """
    kv_cpu = kv.detach().cpu().contiguous()
    token_ids = list(tokens)
    total_tokens = (len(token_ids) // ctx._chunk_size) * ctx._chunk_size
    token_ids = token_ids[:total_tokens]

    # Phase 0: assign request ID to this request
    request_id = f"store-{uuid.uuid4().hex}"
    key = ctx._create_key(
        token_ids=token_ids,
        start=0,
        end=total_tokens,
        request_id=request_id,
        cache_salt=cache_salt,
        worker_id=0
    )

    # Phase 1: store the KV cache tensor
    try:
        return ctx._transfer_ctx.store(key, ctx.instance_id, kv_cpu)
    finally:
        ctx.end_session(request_id)
