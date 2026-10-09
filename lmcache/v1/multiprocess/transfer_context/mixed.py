# SPDX-License-Identifier: Apache-2.0
"""Bounded GPU staging for mixed CPU MLA and CUDA KV cache groups."""

# Standard
from collections.abc import Sequence
from dataclasses import replace
from typing import Literal
import math
import threading

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    IPCEvent,
    LMCacheDrivenTransferContext,
)
from lmcache.v1.multiprocess.transport.base import RequestClient


class MixedTransferContext(LMCacheDrivenTransferContext):
    """Stage CPU MLA pages through one reusable GPU chunk per layer.

    CUDA layers retain their original IPC views. Operations are serialized
    and complete synchronously, so staging cannot be reused while the server
    reads or writes it. CPU tensor storage is never rebound.
    """

    def __init__(self, instance_id: int, req_client: RequestClient) -> None:
        """Bind the context to its worker ID and transport client."""
        super().__init__(instance_id, req_client)
        self._transfer_lock = threading.RLock()
        self._staged_caches: dict[str, torch.Tensor] = {}
        self._host_layers: dict[str, int] = {}
        self._group_infos: tuple[EngineGroupInfo, ...] = ()
        self._chunk_size = 0
        self._copy_stream: torch.Stream | None = None

    def register(
        self,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_timeout: float,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register CUDA views and GPU staging for CPU MLA pages.

        Arguments follow ``TransferContext.register``. CPU groups must use
        MLA pages (rank 3, or rank 4 with a singleton head), with
        uncompressed, contiguous blocks; other layouts raise ValueError.
        GPU staging holds one chunk plus the null block.
        """
        with self._transfer_lock:
            if not engine_group_infos:
                raise ValueError("Mixed KV transfer requires engine group metadata")
            device = next(t.device for t in kv_caches.values() if t.is_cuda)
            self._chunk_size = self._req_client.get_chunk_size().result(mq_timeout)
            self._group_infos = tuple(engine_group_infos)
            names = list(kv_caches)
            self._staged_caches = dict(kv_caches)
            for group_idx, info in enumerate(engine_group_infos):
                for layer_idx in info.layer_indices:
                    name = names[layer_idx]
                    tensor = kv_caches[name]
                    if tensor.device.type != "cpu":
                        continue
                    if (
                        tensor.ndim not in (3, 4)
                        or (tensor.ndim == 4 and 1 not in tensor.shape[1:3])
                        or info.tokens_per_block != math.prod(tensor.shape[1:-1])
                        or not tensor[0].is_contiguous()
                        or self._chunk_size % info.tokens_per_block
                    ):
                        raise ValueError(
                            "Mixed KV transfer requires contiguous CPU MLA pages "
                            "with a block size dividing the LMCache chunk size; "
                            f"got {name}: shape={tuple(tensor.shape)}, "
                            f"tokens_per_block={info.tokens_per_block}"
                        )
                    self._host_layers[name] = group_idx
                    self._staged_caches[name] = torch.empty(
                        (
                            self._chunk_size // info.tokens_per_block + 1,
                            *tensor.shape[1:],
                        ),
                        dtype=tensor.dtype,
                        device=device,
                    )
            self._copy_stream = torch_dev.Stream(device=device)
            super().register(
                self._staged_caches,
                model_name,
                world_size,
                blocks_in_chunk,
                mq_timeout,
                layout_hints,
                engine_group_infos,
                engine_type,
            )

    def submit_store(
        self,
        request_id: str,
        key: IPCCacheServerKey,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
    ) -> MessagingFuture[bool]:
        """Stage each CPU chunk after the producer event and persist all groups.

        Arguments follow ``TransferContext.submit_store``. The returned future
        is complete only after the server has finished reading the chunk.
        """
        return self._transfer(
            "store",
            request_id,
            key,
            kv_caches,
            block_ids,
            event,
            blocks_in_chunk,
            0,
        )

    def submit_retrieve(
        self,
        request_id: str,
        key: IPCCacheServerKey,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture[bool]:
        """Restore all groups and copy staged MLA back into the original CPU pool.

        Arguments follow ``TransferContext.submit_retrieve``. Completion includes
        CPU writes; the already-computed block prefix is never overwritten.
        """
        return self._transfer(
            "retrieve",
            request_id,
            key,
            kv_caches,
            block_ids,
            event,
            blocks_in_chunk,
            skip_first_n_tokens,
        )

    def close(self) -> None:
        """Drain the staging stream before releasing its buffers."""
        with self._transfer_lock:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            super().close()
            self._staged_caches.clear()
            self._host_layers.clear()

    def _transfer(
        self,
        operation: Literal["store", "retrieve"],
        request_id: str,
        key: IPCCacheServerKey,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent | None,
        blocks_in_chunk: int,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[bool]:
        if event is None:
            raise ValueError("Mixed KV transfer requires a producer event")
        with self._transfer_lock:
            assert self._copy_stream is not None
            success = True
            with torch_dev.stream(self._copy_stream):
                event.wait(self._copy_stream)
                if operation == "store":
                    # vLLM's manually pinned views may look pageable to PyTorch;
                    # finish host writes before any CPU-side copy staging.
                    self._copy_stream.synchronize()
                for offset in range(0, key.end - key.start, self._chunk_size):
                    skip = max(0, skip_first_n_tokens - offset)
                    if skip >= self._chunk_size:
                        continue
                    chunk_key = replace(
                        key,
                        start=key.start + offset,
                        end=key.start + offset + self._chunk_size,
                    )
                    original_ids = [
                        ids[
                            offset // info.tokens_per_block : (
                                offset + self._chunk_size
                            )
                            // info.tokens_per_block
                        ]
                        for ids, info in zip(block_ids, self._group_infos, strict=True)
                    ]
                    staged_ids = [list(ids) for ids in original_ids]
                    for name, group_idx in self._host_layers.items():
                        ids = original_ids[group_idx]
                        staged_ids[group_idx] = [
                            i + 1 if block_id else 0 for i, block_id in enumerate(ids)
                        ]
                        if operation == "store":
                            for i, block_id in enumerate(ids):
                                if block_id:
                                    self._staged_caches[name][i + 1].copy_(
                                        kv_caches[name][block_id],
                                        non_blocking=True,
                                    )
                    ready = super().create_recorded_event()
                    if operation == "store":
                        pending = super().submit_store(
                            request_id,
                            chunk_key,
                            self._staged_caches,
                            staged_ids,
                            ready,
                            blocks_in_chunk,
                        )
                    else:
                        pending = super().submit_retrieve(
                            request_id,
                            chunk_key,
                            self._staged_caches,
                            staged_ids,
                            ready,
                            blocks_in_chunk,
                            skip,
                        )
                    chunk_success = pending.result(timeout=self._mq_timeout)
                    success = success and chunk_success
                    if operation == "retrieve" and chunk_success:
                        for name, group_idx in self._host_layers.items():
                            span = self._group_infos[group_idx].tokens_per_block
                            for i, block_id in enumerate(original_ids[group_idx]):
                                if block_id and i >= skip // span:
                                    kv_caches[name][block_id].copy_(
                                        self._staged_caches[name][i + 1],
                                        non_blocking=True,
                                    )
                    # Finish host writes, or outstanding reads after a failed save.
                    if operation == "retrieve" or not chunk_success:
                        self._copy_stream.synchronize()
            completed: MessagingFuture[bool] = MessagingFuture()
            completed.set_result(success)
            return completed
