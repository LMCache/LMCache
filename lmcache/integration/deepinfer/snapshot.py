# SPDX-License-Identifier: Apache-2.0
"""Synchronous, process-local prefix snapshots for DeepInfer's opaque cache."""

# Standard
import hashlib

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class DeepInferSnapshotCache:
    """Store compressed KV, index KV and the complete state row atomically.

    This proof of concept stores each prefix as one LMCache CPU object. It
    deliberately duplicates shared blocks between snapshots. Callers must stop
    writes to the source request before storing and own destination blocks
    exclusively while retrieving. Calls must be serialized on the engine thread.
    The backend remains owned by the caller.
    """

    def __init__(
        self,
        backend: LocalCPUBackend,
        gpu_blocks: torch.Tensor,
        gpu_states: torch.Tensor,
        block_size: int,
        namespace: str,
        worker_id: int = 0,
    ) -> None:
        """Bind a local backend to DeepInfer's byte buffers.

        Args:
            backend: Enabled LMCache CPU backend with a variable-size allocator.
            gpu_blocks: Contiguous uint8 buffer, [physical blocks, bytes/block].
            gpu_states: Contiguous uint8 buffer, [state slots, bytes/state].
            block_size: Number of original tokens represented by a KV block.
            namespace: Model revision and runtime-layout identity. Never share
                it between different weights, quantization or state layouts.
            worker_id: Identity of the local attention replica.

        Raises:
            ValueError: If buffers, block size, namespace or backend are invalid.
        """
        if not backend.use_hot or block_size <= 0 or not namespace:
            raise ValueError("Require local_cpu, a positive block_size and namespace")
        for buffer in (gpu_blocks, gpu_states):
            if (
                buffer.ndim != 2
                or buffer.dtype != torch.uint8
                or not buffer.is_contiguous()
            ):
                raise ValueError("Cache buffers must be contiguous 2-D uint8 tensors")
        if gpu_blocks.device != gpu_states.device or gpu_blocks.shape[1] == 0:
            raise ValueError("Cache buffers must share a device and have KV bytes")
        self.backend = backend
        self.gpu_blocks = gpu_blocks
        self.gpu_states = gpu_states
        self.block_size = block_size
        self.worker_id = worker_id
        self.namespace = (
            f"deepinfer-snapshot-v1/{namespace}/{block_size}/"
            f"{gpu_blocks.shape[1]}/{gpu_states.shape[1]}"
        )

    def store(self, token_ids: list[int], block_ids: list[int], state_id: int) -> bool:
        """Copy an exact, completed prefix and its full state to LMCache.

        Args:
            token_ids: Tokens whose KV has completed, at a nonempty block
                boundary. The state must correspond to exactly this boundary;
                rounding a later state down is incorrect.
            block_ids: Physical block table, in logical token order.
            state_id: Physical state slot; ignored when state rows are empty.

        Returns:
            True if the snapshot is present, False on CPU allocation pressure.

        Raises:
            ValueError: If the prefix, token IDs or physical mapping is invalid.
            RuntimeError: If memory allocation yields a non-tensor object.
        """
        if not token_ids or len(token_ids) % self.block_size:
            raise ValueError("Store requires a nonempty, block-aligned exact prefix")
        count = len(token_ids) // self.block_size
        self._validate_mapping(block_ids, state_id, count)
        key = self._keys(token_ids)[-1][1]
        if self.backend.contains(key):
            return True
        block_bytes = count * self.gpu_blocks.shape[1]
        total_bytes = block_bytes + self.gpu_states.shape[1]
        obj = self.backend.allocate(
            torch.Size((1, 1, 1, total_bytes)),
            torch.uint8,
            fmt=MemoryFormat.KV_2LTD,
            busy_loop=False,
        )
        if obj is None:
            return False
        try:
            if obj.tensor is None:
                raise RuntimeError("DeepInfer snapshots require a tensor allocator")
            payload = obj.tensor.view(-1)
            indices = torch.tensor(
                block_ids[:count], dtype=torch.long, device=self.gpu_blocks.device
            )
            payload[:block_bytes].copy_(
                self.gpu_blocks.index_select(0, indices).view(-1)
            )
            if self.gpu_states.shape[1]:
                payload[block_bytes:].copy_(self.gpu_states[state_id])
            # Blocking D2H copies complete before publishing the one object.
            self.backend.submit_put_task(key, obj)
        finally:
            obj.ref_count_down()
        return True

    def retrieve(
        self, token_ids: list[int], block_ids: list[int], state_id: int
    ) -> int:
        """Restore the longest available prefix, leaving a token to recompute.

        Args:
            token_ids: Complete prompt, including the uncomputed suffix.
            block_ids: Exclusively owned destination block table for the prompt.
            state_id: Exclusively owned destination state slot.

        Returns:
            Restored token count, or zero on miss. The caller may advance
            DeepInfer's num_cached_tokens only after this method succeeds.

        Raises:
            ValueError: If a matching snapshot has an invalid destination or
                payload size, or token IDs are invalid.
            RuntimeError: If a matching object has no tensor payload.

        Notes:
            A get holds a reference through copying, so eviction cannot reclaim
            the payload. A miss never changes the destination buffers.
        """
        for end, key in reversed(self._keys(token_ids[:-1])):
            obj = self.backend.get_blocking(key)
            if obj is None:
                continue
            try:
                count = end // self.block_size
                self._validate_mapping(block_ids, state_id, count)
                if obj.tensor is None:
                    raise RuntimeError("DeepInfer snapshot has no tensor payload")
                payload = obj.tensor.view(-1)
                block_bytes = count * self.gpu_blocks.shape[1]
                if payload.numel() != block_bytes + self.gpu_states.shape[1]:
                    raise ValueError("Snapshot payload does not match cache layout")
                indices = torch.tensor(
                    block_ids[:count], dtype=torch.long, device=self.gpu_blocks.device
                )
                blocks = payload[:block_bytes].view(count, -1)
                self.gpu_blocks.index_copy_(
                    0, indices, blocks.to(self.gpu_blocks.device)
                )
                if self.gpu_states.shape[1]:
                    self.gpu_states[state_id].copy_(payload[block_bytes:])
                if self.gpu_blocks.device.type != "cpu":
                    torch_dev.current_stream(self.gpu_blocks.device).synchronize()
                return end
            finally:
                obj.ref_count_down()
        return 0

    def _keys(self, token_ids: list[int]) -> list[tuple[int, CacheEngineKey]]:
        """Hash complete prefixes, independently of Python/vLLM hash settings."""
        if any(token < 0 or token > 2**31 - 1 for token in token_ids):
            raise ValueError("Expected nonnegative int32 token IDs")
        tokens = np.asarray(token_ids, dtype="<i4")
        digest = hashlib.sha256()
        keys = []
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            end = start + self.block_size
            digest.update(tokens[start:end].tobytes())
            keys.append(
                (
                    end,
                    CacheEngineKey(
                        model_name=self.namespace,
                        world_size=1,
                        worker_id=self.worker_id,
                        chunk_hash=int.from_bytes(digest.digest(), "big"),
                        dtype=torch.uint8,
                    ),
                )
            )
        return keys

    def _validate_mapping(
        self, block_ids: list[int], state_id: int, count: int
    ) -> None:
        """Reject aliasing and DeepInfer's reserved block/slot zero."""
        ids = block_ids[:count]
        if (
            len(ids) != count
            or len(set(ids)) != count
            or any(index <= 0 or index >= self.gpu_blocks.shape[0] for index in ids)
        ):
            raise ValueError("Invalid, duplicate or insufficient physical KV blocks")
        if self.gpu_states.shape[1] and not 0 < state_id < self.gpu_states.shape[0]:
            raise ValueError("Invalid physical state slot")
