# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import is_mla
from lmcache.v1.multiprocess.cpu_context import (
    CPUContext,
    CPUContextMetadata,
    compute_kv_layout,
    create_cpu_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType

logger = init_logger(__name__)


class IPCEvent(Protocol):
    """Protocol for IPC-capable CUDA events used by transport operations."""

    def ipc_handle(self) -> object:
        """Return an IPC handle consumable by the multiprocess server."""


SendRequest = Callable[
    [MessageQueueClient, RequestType, list[object]], MessagingFuture[object]
]


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server (for example,
    CUDA IPC futures or CPU-context gather/scatter flows).
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance id.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks in one LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
        """

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> None:
        """Submit a store request.

        Args:
            request_id: Request identifier.
            key: LMCache key object.
            instance_id: Worker process instance id.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block ids to store.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks in one LMCache chunk.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> None:
        """Submit a retrieve request.

        Args:
            request_id: Request identifier.
            key: LMCache key object.
            instance_id: Worker process instance id.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block ids to retrieve.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks in one LMCache chunk.
            skip_first_n_tokens: Number of tokens to skip for partial scatter.
        """

    @abstractmethod
    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        """Poll completed requests.

        Returns:
            Tuple of ``(finished_store_ids, finished_retrieve_ids, error_block_ids)``.
        """

    @abstractmethod
    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        """Drain all pending requests.

        Returns:
            Tuple of ``(finished_store_ids, finished_retrieve_ids, error_block_ids)``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""


class CudaTransferContext(TransferContext):
    """CUDA IPC + MQ future transport context."""

    def __init__(self) -> None:
        self._store_futures: dict[str, Any] = {}
        self._retrieve_futures: dict[str, tuple[Any, list[int]]] = {}
        self._mq_client: MessageQueueClient | None = None
        self._mq_timeout: float = 0.0
        self._send_request: SendRequest | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
    ) -> None:
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints
        from lmcache.integration.vllm.vllm_multi_process_adapter import wrap_kv_caches

        self._mq_client = mq_client
        self._mq_timeout = mq_timeout
        self._send_request = send_request
        layout_hints = vllm_layout_hints()
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> None:
        if (
            self._mq_client is None
            or self._send_request is None
            or self._mq_timeout < 0
        ):
            raise RuntimeError(
                "CUDA transfer context is not registered. "
                "Call register() before submit_store()."
            )
        future = self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self._store_futures[request_id] = future

    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> None:
        if (
            self._mq_client is None
            or self._send_request is None
            or self._mq_timeout < 0
        ):
            raise RuntimeError(
                "CUDA transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        future = self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event.ipc_handle(), skip_first_n_tokens],
        ).to_cuda_future()
        self._retrieve_futures[request_id] = (future, list(block_ids))

    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores: set[str] = set()
        finished_retrieves: set[str] = set()
        error_block_ids: set[int] = set()

        for request_id, s_future in list(self._store_futures.items()):
            if not s_future.query():
                continue
            s_result = s_future.result()
            finished_stores.add(request_id)
            if not s_result:
                logger.error(
                    "Something went wrong when processing the store request "
                    "for request_id=%s",
                    request_id,
                )
            self._store_futures.pop(request_id, None)

        for request_id, (r_future, r_block_ids) in list(self._retrieve_futures.items()):
            if not r_future.query():
                continue
            r_result = r_future.result()
            finished_retrieves.add(request_id)
            if not r_result:
                logger.error(
                    "Something went wrong when processing the retrieve request "
                    "for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )
                error_block_ids.update(r_block_ids)
            self._retrieve_futures.pop(request_id, None)

        return finished_stores, finished_retrieves, error_block_ids

    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores = set(self._store_futures.keys())
        finished_retrieves = set(self._retrieve_futures.keys())
        error_block_ids: set[int] = set()
        for _request_id, (_r_future, block_ids) in self._retrieve_futures.items():
            error_block_ids.update(block_ids)
        self._store_futures.clear()
        self._retrieve_futures.clear()
        return finished_stores, finished_retrieves, error_block_ids

    def close(self) -> None:
        self._store_futures.clear()
        self._retrieve_futures.clear()
        self._mq_client = None
        self._mq_timeout = 0.0
        self._send_request = None


class CPUTransferContext(TransferContext):
    """CPU context transport for non-CUDA workers."""

    def __init__(self) -> None:
        self._cpu_context: CPUContext | None = None
        self._layout_hints: Any = None
        self._gpu_kv_format: Any = None
        self._store_done: dict[str, bool] = {}
        self._retrieve_done: dict[str, tuple[bool, list[int]]] = {}
        self._mq_client: MessageQueueClient | None = None
        self._mq_timeout: float = 0.0
        self._send_request: SendRequest | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
    ) -> None:
        # First Party
        from lmcache.integration.vllm.utils import vllm_layout_hints

        self._mq_client = mq_client
        self._mq_timeout = mq_timeout
        self._send_request = send_request
        layout_hints = vllm_layout_hints()
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            gpu_kv_format,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._gpu_kv_format = gpu_kv_format

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_CPU_CONTEXT,
            [
                instance_id,
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                block_size,
                num_layers,
                hidden_dim_size,
                dtype_str,
                is_mla(gpu_kv_format),
            ],
        )

        use_mla_flag = is_mla(gpu_kv_format)
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        metadata = CPUContextMetadata(
            layout_desc=MemoryLayoutDesc(shapes=[shape], dtypes=[dtype]),
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._cpu_context = create_cpu_context(metadata, mq_client, mq_timeout)
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> None:
        if self._cpu_context is None:
            raise RuntimeError(
                "CPU transfer context is not registered. "
                "Call register() before submit_store()."
            )
        torch_dev.synchronize()
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            block_ids,
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            gpu_kv_format=self._gpu_kv_format,
        )
        handle = self._cpu_context.prepare_store(key, instance_id, cpu_chunks)
        ok = self._cpu_context.commit_store(handle)
        self._store_done[request_id] = ok

    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> None:
        if self._cpu_context is None:
            raise RuntimeError(
                "CPU transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        handle, chunks = self._cpu_context.prepare_retrieve(key, instance_id)
        ok = chunks is not None
        if chunks is not None:
            try:
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    block_ids,
                    chunks,
                    blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    gpu_kv_format=self._gpu_kv_format,
                )
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
        self._cpu_context.commit_retrieve(handle)
        self._retrieve_done[request_id] = (ok, list(block_ids))

    def poll_finished(self) -> tuple[set[str], set[str], set[int]]:
        finished_stores = set(self._store_done.keys())
        finished_retrieves = set(self._retrieve_done.keys())
        error_block_ids: set[int] = set()
        for ok, block_ids in self._retrieve_done.values():
            if not ok:
                error_block_ids.update(block_ids)
        self._store_done.clear()
        self._retrieve_done.clear()
        return finished_stores, finished_retrieves, error_block_ids

    def drain_all(self) -> tuple[set[str], set[str], set[int]]:
        return self.poll_finished()

    def close(self) -> None:
        if self._cpu_context is not None:
            self._cpu_context.close()
            self._cpu_context = None
        self._store_done.clear()
        self._retrieve_done.clear()
        self._mq_client = None
        self._mq_timeout = 0.0
        self._send_request = None


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty or has mixed device types.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    logger.info("Creating transfer context (device_type=%s)", device_type)
    if device_type == "cuda":
        return CudaTransferContext()
    return CPUTransferContext()
